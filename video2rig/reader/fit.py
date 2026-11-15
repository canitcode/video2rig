"""The rig onto the reader's joints, the whole clip in one fit: the poses of all frames at once, whose keypoints land
where the reader saw them in every panel (each reading weighted by how sure the reader was, a wrong reading barely
pulling), inside the rig's joint ranges, smooth from frame to frame, the feet not under the floor. The motion is what
the clip shows and the reader reads: nothing is added after the fit (no filter, no pinned feet, no travel by steps,
no springs).

    python -m video2rig.reader.fit <project> <clip> [--tag -reader] [--set KEY=VALUE]
        -> work/<clip>/fit<tag>.npz (review, report) and out/<clip><tag>.npz (blender/import_motion.py)

The fit starts from the clip itself: every keypoint placed in 3D where its readings in the panels meet, the body's
turn from the points riding on the root bone, every bone pointed at the next joint down its chain. The robust scale
of a miss starts wide (40 px: a joint far from its reading still pulls the pose out of a wrong start) and narrows to
6 px over the first 60% of the steps (then a wrong reading barely pulls). Heights come mostly from the side panel:
the front panel foreshortens a lifted leg and video models animate it less faithfully, so it mainly gives left and
right. The fit runs on the processor with one thread: a clip gives the same motion on every computer.
"""
import argparse
import json
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ..common import Project, write_json
from ..creature import Creature
from . import KeypointRig, end_pairs, is_offaxis, is_prop

SETTINGS = {
    'iters': 1500,             # Adam steps over the whole clip
    'scale_start_px': 40.0,    # robust (Cauchy) scale of a keypoint's miss at the first step, px of a 720-row frame,
    'scale_px': 6.0,           # narrowing to this one
    'anneal': .6,              # over this share of the steps
    'lr_move': .01,            # step sizes: the body's place (armature units), its turn (rotation vector), the joints
    'lr_turn': .03,            # (on the squashed scale inside their ranges)
    'lr_joint': .05,
    'lr_drop_at': .7,          # from this share of the steps on, the step sizes times lr_drop
    'lr_drop': .3,
    'front_height': .25,       # the front panel's weight on the up-down axis (the side panel's is 1)
    'w_acc': .005,             # every keypoint's acceleration from frame to frame, per px^2. At .02 a cow's canter kept
                               # its legs near vertical while the video swung them far (side panel's sure readings
                               # missed by 12.7 px; at .005 3.0), and the walk and the fall fitted closer too
                               # (3.3 / 1.7, 3.8 / 2.5 px against 3.5 / 2.1, 4.7 / 3.7), their acceleration up a quarter
    'w_vel': 2.0,              # every joint angle's (and the body turn's) change from frame to frame, per rad^2
    'w_rest': .05,             # pull of every joint angle toward the start pose, per rad^2
    'w_floor': .5,             # a foot's keypoint under the floor, per px^2
    'feet_share': .05,         # feet: keypoints within this share of the height above the floor in the start pose
    # for a game, each off unless set (creature.json "reader_fit", or --set): they change what the clip shows
    'w_contact': 0.0,          # a standing foot's slide along the floor, per px^2 (0: off; 0.3 suits). A reader's few
                               # px of shake make a planted foot slide; a game shows that
    'loop': False,             # true: the window whose end looks most like its start, written as out/<clip>-loop.npz
                               # with its end faded into its start (close_loop); or [start, end] seconds. The clip's own
                               # motion stays as fitted
    'travel': False,           # true: the motion file moves the body along the floor by what the standing feet slide
                               # back (a walk drawn on the spot walks forward; the review stays on the video)
    'contact_height': .03,     # a foot keypoint stands within this share of the height above its start height
    'contact_speed': .3,       # and moving along the floor slower than this many heights per second
    'shortest_loop': .5,       # seconds
    'stage2_iters': 400,       # Adam steps more, at the last step size, when w_contact is on
}


def settings_for(cfg, sets=None):
    """SETTINGS, the clip's "reader_fit" from creature.json over them (creature-wide, then the clip's own), and
    sets ({key: value}: --set) over those."""
    own = dict(cfg.get('reader_fit') or {})
    own.update(sets or {})
    unknown = sorted(set(own) - set(SETTINGS))
    if unknown:
        raise SystemExit(f'reader_fit: no such setting {", ".join(unknown)} ({", ".join(SETTINGS)})')
    return {**SETTINGS, **own}


def creature_for(pr, cfg, kps):
    """The clip's creature, with the bones the reader sees at both ends but the tool does not fit by default (a
    jaw: a bite opens it) added to the fitted ones. Returns the creature and {channel: (lo, hi)} ranges for them:
    the jaw opens one way, from shut to 40 degrees, and does not twist or swing sideways."""
    cr = Creature(pr, cfg)
    rig = cr.rig
    ends = {}
    for kp in kps:
        _, bone, f = kp
        if str(bone).startswith('@'):
            continue
        ends.setdefault(bone, set()).add(is_offaxis(kp) or f > 0)
    extra = [b for b, e in ends.items() if e == {False, True} and rig.names.index(b) not in rig.chan_of
             and not (rig.desc[rig.names.index(b), rig.root] and rig.names.index(b) != rig.root)]
    if not extra:
        return cr, {}
    fitted = list(dict.fromkeys(rig.names[j] for j, _ in rig.channels)) + extra
    cfg = {**cfg, 'fit_bones': fitted}
    cr = Creature(pr, cfg)
    rig = cr.rig
    p0 = cr.p0(None)
    narrow = {}
    for b in extra:
        j = rig.names.index(b)
        tip0 = rig.bone_points(p0)[1][j]
        best = None
        for a in (0, 1, 2):
            q = p0.copy()
            q[rig.index(b, a)] += .3
            down = float(-(rig.bone_points(q)[1][j] - tip0) @ cr.up)
            if best is None or abs(down) > abs(best[1]):
                best = (a, down)
        for a in (0, 1, 2):
            c = rig.index(b, a)
            if a == best[0]:
                narrow[c] = (p0[c] - .05, p0[c] + .7) if best[1] > 0 else (p0[c] - .7, p0[c] + .05)
            else:
                narrow[c] = (p0[c] - .08, p0[c] + .08)
    return cr, narrow


# ----- the start: the clip's own joints in 3D ---------------------------------------------------------------------
def front_panel(views):
    """Index of the front panel (its name says so) when there are two or more panels, else None."""
    if len(views) < 2:
        return None
    return next((i for i, v in enumerate(views) if 'front' in str(v['name']).lower()), None)


def triangulate(views, uv, conf, X0, front_height):
    """(T, K, 3) every keypoint where its readings in the panels meet (least squares in pixels, each panel weighted
    by the square of how sure its reading is, as in the fit, the front panel's up-down axis by front_height), held at
    its start-pose place X0 (K, 3) where no panel is sure of it. The reader says "not here" for a joint a panel cannot
    see (low confidence) and puts that reading anywhere: weighted by the confidence itself, a cow's spine, hidden from
    the front, was placed up to 300 px aside, the body turned on it and the spine fitted at the end of its range for
    the whole walk. None with one panel."""
    if len(views) < 2:
        return None
    T, V, K = conf.shape
    fi = front_panel(views)
    N = np.zeros((T, K, 3, 3))
    r = np.zeros((T, K, 3))
    for v, view in enumerate(views):
        cam = view['cam']
        w = np.clip(conf[:, v], 1e-3, 1.0) ** 2
        rows_w = [w, w * (front_height if v == fi else 1.0)]
        if cam.type == 'ortho':
            for i in (0, 1):
                a = cam.A[i]
                y = uv[:, v, :, i] - cam.b[i]
                N += rows_w[i][..., None, None] * np.outer(a, a)
                r += (rows_w[i] * y)[..., None] * a
        else:
            # pinhole: (u - cu) depth - f x = 0 and (v - cv) depth + f y = 0, linear in the point; divided by the
            # start pose's depth so a row's misfit is about a pixel
            f = cam.f0 * cam.s
            c = np.array([cam.w / 2, cam.h / 2]) + cam.shift + cam.offset
            z0 = max(float(((X0.mean(0) @ cam.Ma.T + cam.ta) - cam.loc) @ cam.fwd), 1e-6)
            for i, (axis, sign) in enumerate(((cam.right, -1.0), (cam.up, 1.0))):
                g = ((uv[:, v, :, i] - c[i])[..., None] * cam.fwd + sign * f * axis) / z0      # (T, K, 3) world
                a = g @ cam.Ma                                                                # armature
                y = -(g @ (cam.ta - cam.loc))
                N += rows_w[i][..., None, None] * np.einsum('tki,tkj->tkij', a, a)
                r += (rows_w[i] * y)[..., None] * a
    k = float(np.mean([v['cam'].px_per_unit for v in views]))
    hold = 1e-2 * k ** 2              # a reading at confidence 0.1 pulls as much as the start pose
    N += hold * np.eye(3)
    r += hold * X0[None]
    return np.linalg.solve(N, r[..., None])[..., 0]


def rot_between(a, b):
    """The smallest rotation taking unit vector a onto unit vector b."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        p = np.cross(a, [1.0, 0, 0] if abs(a[0]) < .9 else [0, 1.0, 0])
        return Rotation.from_rotvec(np.pi * p / np.linalg.norm(p)).as_matrix()
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def nearest_rotvec(r, ref):
    """The rotation vector of the same turn as r nearest ref (r + k 2 pi along its axis)."""
    th = np.linalg.norm(r)
    if th < 1e-9:
        return r
    u = r / th
    k = np.round((u @ ref - th) / (2 * np.pi))
    return r + 2 * np.pi * k * u


def near_angle(a, ref):
    return a + 2 * np.pi * np.round((ref - a) / (2 * np.pi))


def start_params(cr, kps, tri, conf, p0, lo, hi, log=print):
    """(T, rig.n) the start of the fit: the start pose everywhere with one panel; with two or more the root put on
    its read joint, the body turned by a weighted Kabsch fit of the points riding on the root bone (smoothed over 5
    frames), and every fitted bone, down the hierarchy, pointed at its next joint as read, inside its range."""
    rig = cr.rig
    T = len(conf)
    P = np.tile(p0, (T, 1))
    if tri is None:
        return P
    kr = KeypointRig(cr, kps)
    X0 = kr.points(p0)
    c = conf.min(1).clip(1e-3, 1.0)                                     # (T, K): sure in every panel
    G0, H0, _ = rig.fk(p0)
    head_kp = {}
    for k, (_, bone, f) in enumerate(kps):
        if not is_prop(kps[k]) and not is_offaxis(kps[k]) and float(f) == 0.0:
            head_kp.setdefault(rig.names.index(bone), k)
    root = rig.root
    if root not in head_kp:
        log('start: the root bone has no keypoint at its head: the start pose throughout')
        return P
    # the body's place
    q = p0.copy()
    q[:3] = 0
    P[:, :3] = tri[:, head_kp[root]] - rig.fk(q)[1][root]
    # the body's turn from the points riding on the root bone: its own points and its children's heads
    ride = [k for k, kp in enumerate(kps) if not is_prop(kp) and rig.names.index(kp[1]) == root]
    ride += [head_kp[j] for j in range(rig.nb) if rig.parents[j] == root and j in head_kp]
    ride = sorted(set(ride))
    root_ch = sorted(rig.chan_of.get(root, []))
    if len(ride) >= 3 and len(root_ch) == 3:
        h = head_kp[root]
        par = rig.parents[root]
        base = (G0[par] if par >= 0 else np.eye(3)) @ rig.Rrl[root]
        w0 = np.array([p0[k] for _, k in root_ch])
        E0 = Rotation.from_rotvec(w0).as_matrix()
        a = X0[ride] - X0[h]
        rv = np.zeros((T, 3))
        prev = w0
        for t in range(T):
            b = tri[t, ride] - tri[t, h]
            U, _, Vt = np.linalg.svd((a * c[t, ride][:, None]).T @ b)
            D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
            W = Vt.T @ D @ U.T                                          # b ~ W a
            rv[t] = nearest_rotvec(Rotation.from_matrix(base.T @ W @ base @ E0).as_rotvec(), prev)
            prev = rv[t]
        pad = np.pad(rv, ((2, 2), (0, 0)), mode='edge')
        rv = np.stack([np.convolve(pad[:, i], np.ones(5) / 5, mode='valid') for i in range(3)], 1)
        for i, (_, k) in enumerate(root_ch):
            P[:, k] = rv[:, i]
    # every other fitted bone pointed at its next joint as read
    targets = {}
    for j in rig.chan_of:
        if j == root or j not in head_kp:
            continue
        tail0 = H0[j] + G0[j][:, 1] * rig.length[j]
        kids = [k for k in range(rig.nb) if rig.parents[k] == j and k in head_kp
                and np.linalg.norm(H0[k] - tail0) <= .02 * cr.height]
        if kids:
            tgt = head_kp[min(kids, key=lambda k: np.linalg.norm(H0[k] - tail0))]
        else:
            tips = [k for k, kp in enumerate(kps) if not is_prop(kp) and not is_offaxis(kp)
                    and rig.names.index(kp[1]) == j and float(kp[2]) > 0]
            if not tips:
                continue
            tgt = tips[0]
        d0 = X0[tgt] - X0[head_kp[j]]
        if np.linalg.norm(d0) < 1e-6:
            continue
        targets[j] = (head_kp[j], tgt, G0[j].T @ (d0 / np.linalg.norm(d0)))

    def own_turn(q, j):
        """The bone's basis rotation in parameters q."""
        if j in rig.chan_of and j == root:
            return Rotation.from_rotvec([q[k] for _, k in sorted(rig.chan_of[j])]).as_matrix()
        if j in rig.chan_of:
            return Rotation.from_euler('XYZ', [q[k] for _, k in sorted(rig.chan_of[j])]).as_matrix()
        return rig.fixed[j]
    for t in range(T):
        G = np.zeros((rig.nb, 3, 3))
        for j in range(rig.nb):
            par = rig.parents[j]
            base = (G[par] if par >= 0 else np.eye(3)) @ rig.Rrl[j]
            if j in targets:
                chans = sorted(rig.chan_of[j])
                ang0 = np.array([P[t, k] for _, k in chans])
                R0 = Rotation.from_euler('XYZ', ang0).as_matrix()
                hk, tk, u = targets[j]
                d = tri[t, tk] - tri[t, hk]
                if np.linalg.norm(d) > 1e-6:
                    R = rot_between(R0 @ u, base.T @ (d / np.linalg.norm(d))) @ R0
                    ang = Rotation.from_matrix(R).as_euler('XYZ')
                    for i, (_, k) in enumerate(chans):
                        P[t, k] = np.clip(near_angle(ang[i], ang0[i]), lo[k], hi[k])
            G[j] = base @ own_turn(P[t], j)
    # props: moved with their read points, turned as at the start
    for i, pr in enumerate(rig.props):
        mine = [k for k, kp in enumerate(kps) if is_prop(kp) and kp[1][1:] == pr['name']]
        if mine:
            base_i = rig.prop0 + 6 * i
            P[:, base_i:base_i + 3] += (c[:, mine, None] * (tri[:, mine] - X0[mine])).sum(1) / \
                c[:, mine].sum(1, keepdims=True)
    return P


# ----- the fit ----------------------------------------------------------------------------------------------------
def content_grid(times):
    """The clip's content frames on one even grid: (the grid's times (N,), each video frame's place on it (T,)). A clip
    whose content dropped frames (30 fps with every 5th gone, 25 with every 25th: clip.cadence) has a step of two
    there; counted in frames, the fit's smoothness took that step for a jump and evened the motion out in frames, so
    played at the real times a hidden limb sped up and slowed down every 4th step. On the grid every step is the same
    time, and the motion gets the dropped frames too (from the frames around them). Times on no even grid: the video's
    own frames."""
    t = np.asarray(times, float)
    own = (t, np.arange(len(t)))
    if len(t) < 3:
        return own
    step = float(np.min(np.diff(t)))
    if step <= 0:
        return own
    at = np.round((t - t[0]) / step).astype(int)
    if np.abs(t[0] + at * step - t).max() > .05 * step or (np.diff(at) < 1).any():
        return own
    return np.interp(np.arange(at[-1] + 1), at, t), at


def _change_costs(X, joints, turn, steps):
    """(acceleration, frame-to-frame change) costs of the motion. steps (N - 1,): the grid's steps as shares of its
    usual one, None when they are all the same (the plain differences). Uneven steps: a change d over a step h costs
    d^2 / h (the squared rate over that time), an acceleration is the change of rate over the mean of its two steps,
    and costs by that mean; on an even grid both are the plain differences."""
    if steps is None:
        acc = ((X[2:] - 2 * X[1:-1] + X[:-2]) ** 2).sum()
        vel = ((joints[1:] - joints[:-1]) ** 2).sum() + ((turn[1:] - turn[:-1]) ** 2).sum()
        return acc, vel
    h0, h1 = steps[:-1, None, None], steps[1:, None, None]
    mean = (h0 + h1) / 2
    a = ((X[2:] - X[1:-1]) / h1 - (X[1:-1] - X[:-2]) / h0) / mean
    acc = (a ** 2 * mean).sum()
    h = steps[:, None]
    vel = ((joints[1:] - joints[:-1]) ** 2 / h).sum() + ((turn[1:] - turn[:-1]) ** 2 / h).sum()
    return acc, vel


def shake(X, k):
    """How much keypoints X (N, K, 3) shake on an even grid: the mean size of their third difference from frame to
    frame (it is about 0 for any move that speeds up or slows down smoothly), in px of a 720-row frame (k px per
    armature unit)."""
    X = np.asarray(X, float)
    if len(X) < 4:
        return 0.0
    return float(np.linalg.norm(X[3:] - 3 * X[2:-1] + 3 * X[1:-2] - X[:-3], axis=-1).mean() * k)


def along_floor(X, up):
    """X (..., 3) with the up-down part taken out."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    return X - (X @ up)[..., None] * up


def standing(F, times, up, start_h, height, s=SETTINGS):
    """(N, F) 1 where a foot keypoint stands on the floor, else 0: within contact_height of the height above its own
    start height (start_h (F,), along up) and moving along the floor slower than contact_speed heights per second; the
    median of 3 frames, so one shaky frame does not lift a foot. F (N, F, 3) the feet's keypoints at times (N,)."""
    F = np.asarray(F, float)
    up = np.asarray(up, float) / np.linalg.norm(up)
    if len(F) < 2:
        return np.zeros(F.shape[:2])
    rate = np.linalg.norm(np.diff(along_floor(F, up), axis=0), axis=-1) / np.diff(times)[:, None]       # (N - 1, F)
    speed = np.concatenate([rate[:1], (rate[1:] + rate[:-1]) / 2, rate[-1:]])
    on = ((F @ up - start_h <= s['contact_height'] * height) & (speed <= s['contact_speed'] * height)).astype(float)
    pad = np.pad(on, ((1, 1), (0, 0)), mode='edge')
    return np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]]), 0)


def travel(F, stand, up):
    """(N, 3) how far the body moves along the floor so that the standing feet stay where they stand: every step,
    the standing feet's mean slide backwards; through steps no foot stands on (a leap) the speed it had before."""
    step = np.diff(along_floor(np.asarray(F, float), up), axis=0)                                      # (N - 1, F, 3)
    both = stand[1:] * stand[:-1]
    out = np.zeros((len(F), 3))
    speed = np.zeros(3)
    for t in range(len(step)):
        if both[t].sum() > 0:
            speed = -(step[t] * both[t][:, None]).sum(0) / both[t].sum()
        out[t + 1] = out[t] + speed
    return out


def find_loop(X, times, up, shortest=.5):
    """(i, j) the grid frames of the window whose end looks most like its start, at least `shortest` seconds long:
    the keypoints round their middle along the floor (the travel taken out) and their steps, frame j against frame i.
    It starts a quarter of `shortest` into the clip or later: close_loop fades its end into the frames before it."""
    X = np.asarray(X, float)
    R = X - along_floor(X.mean(1, keepdims=True), up)
    step = np.gradient(R, axis=0)
    N = len(R)
    gap = int(np.ceil(shortest / float(np.median(np.diff(times)))))
    lead = min(int(np.ceil(gap / 4)), max(N - gap - 1, 0))
    best, out = np.inf, (lead, N - 1)
    for i in range(lead, N - gap):
        d = np.linalg.norm(R[i + gap:] - R[i], axis=-1).mean(1) + 4 * np.linalg.norm(step[i + gap:] - step[i],
                                                                                     axis=-1).mean(1)
        j = int(np.argmin(d))
        if d[j] < best:
            best, out = float(d[j]), (i, i + gap + j)
    return out


def close_loop(P, i, j, up, moves=(slice(0, 3),), share=.25):
    """The window [i, j] of the motion (P (N, rig.n) parameters) as a loop: its last frames cross-faded into the
    frames that lead into its start (a quarter of the window, at most as many as there are before i), so its last
    frame is its first again and it arrives there moving as the clip moved into it. Only fitted frames are mixed: where
    the window is a whole period the two are nearly the same and the fade changes almost nothing. The places in `moves`
    (the root's, the props': 3 channels each) keep the window's travel along the floor, `up` the up direction. Returns
    the loop (j - i + 1, rig.n) and the largest angle the fade moved (radians)."""
    P = np.asarray(P, float)
    L = P[i:j + 1].copy()
    n = j - i
    up = np.asarray(up, float) / np.linalg.norm(up)
    shift = np.zeros(P.shape[1])
    angles = np.ones(P.shape[1], bool)
    for m in moves:
        shift[m] = along_floor(P[j, m] - P[i, m], up)                      # the window's own travel
        angles[m] = False
    fade = min(int(round(share * n)), i)
    for k in range(n - fade, n + 1):
        w = 1.0 if fade == 0 else (k - (n - fade)) / fade
        w = w * w * (3 - 2 * w)                                            # smooth at both ends of the fade
        L[k] = (1 - w) * L[k] + w * (P[i - (n - k)] + shift)
    return L, float(np.abs(L - P[i:j + 1])[:, angles].max()) if angles.any() else 0.0


def _exp_so3(w):
    th2 = (w * w).sum(-1, keepdim=True)[..., None]
    th = torch.sqrt(th2 + 1e-24)
    small = th < 1e-4
    A = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    B = torch.where(small, .5 - th2 / 24, (1 - torch.cos(th)) / (th2 + 1e-24))
    z = torch.zeros_like(w[..., 0])
    K = torch.stack([z, -w[..., 2], w[..., 1], w[..., 2], z, -w[..., 0], -w[..., 1], w[..., 0], z], -1)
    K = K.view(*w.shape[:-1], 3, 3)
    return torch.eye(3, dtype=w.dtype) + A * K + B * K @ K


def _rot(axis, a):
    c, s = torch.cos(a), torch.sin(a)
    o, l = torch.zeros_like(a), torch.ones_like(a)
    if axis == 0:
        m = [l, o, o, o, c, -s, o, s, c]
    elif axis == 1:
        m = [c, o, s, o, l, o, -s, o, c]
    else:
        m = [c, -s, o, s, c, o, o, o, l]
    return torch.stack(m, -1).view(*a.shape, 3, 3)


class TorchRig:
    """The rig's forward kinematics (Rig.fk and the keypoints of KeypointRig) for all frames at once, in torch."""

    def __init__(self, cr, kps):
        rig = cr.rig
        self.rig = rig
        t = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float64)
        self.Rrl, self.trl, self.fixed = t(rig.Rrl), t(rig.trl), t(rig.fixed)
        kr = KeypointRig(cr, kps)
        self.kb = kr.idx[:, 0]
        self.body = self.kb < rig.nb
        self.local = []                                  # each keypoint in its bone's own frame
        for k in range(len(kps)):
            j = int(self.kb[k])
            if j < rig.nb:
                self.local.append(rig.rest_rot[j].T @ (kr.rest[k] - rig.rest_head[j]))
            else:
                self.local.append(kr.rest[k] - rig.prop_centre[j - rig.nb])
        self.local = t(np.array(self.local))
        self.centre = t(rig.prop_centre)
        self.body_idx = np.nonzero(self.body)[0]
        self.prop_idx = [np.nonzero(self.kb == rig.nb + i)[0] for i in range(len(rig.props))]

    def points(self, p):
        """(T, K, 3) keypoints of the parameters p (T, rig.n)."""
        rig = self.rig
        T = p.shape[0]
        G, Ph = [None] * rig.nb, [None] * rig.nb
        for j in range(rig.nb):
            par = rig.parents[j]
            if par < 0:
                Gj = self.Rrl[j].expand(T, 3, 3)
                Pj = self.trl[j].expand(T, 3)
            else:
                Gj = G[par] @ self.Rrl[j]
                Pj = Ph[par] + G[par] @ self.trl[j]
            if j == rig.root:
                Pj = Pj + p[:, :3]
            if j in rig.chan_of and j == rig.root:
                Gj = Gj @ _exp_so3(p[:, [k for _, k in sorted(rig.chan_of[j])]])
            elif j in rig.chan_of:
                for a, k in rig.chan_of[j]:
                    Gj = Gj @ _rot(a, p[:, k])
            else:
                Gj = Gj @ self.fixed[j]
            G[j], Ph[j] = Gj, Pj
        G, Ph = torch.stack(G, 1), torch.stack(Ph, 1)
        X = torch.zeros(T, len(self.kb), 3, dtype=p.dtype)
        b = self.body_idx
        X[:, b] = (G[:, self.kb[b]] @ self.local[b][..., None])[..., 0] + Ph[:, self.kb[b]]
        for i in range(len(rig.props)):
            mine = self.prop_idx[i]
            if not len(mine):
                continue
            c0 = rig.prop0 + 6 * i
            R = _rot(0, p[:, c0 + 3]) @ _rot(1, p[:, c0 + 4]) @ _rot(2, p[:, c0 + 5])
            X[:, mine] = self.local[mine] @ R.transpose(1, 2) + (self.centre[i] + p[:, c0:c0 + 3])[:, None]
        return X


def projector(cam):
    """Armature-space points (..., 3) -> the panel's pixels (..., 2), in torch."""
    t = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float64)
    if cam.type == 'ortho':
        A, b = t(cam.A), t(cam.b)
        return lambda X: X @ A.T + b
    Ma, ta, loc = t(cam.Ma), t(cam.ta), t(cam.loc)
    right, up, fwd = t(cam.right), t(cam.up), t(cam.fwd)
    f = float(cam.f0 * cam.s)
    c = np.array([cam.w / 2, cam.h / 2]) + cam.shift + cam.offset

    def proj(X):
        d = X @ Ma.T + ta - loc
        z = (d @ fwd).clamp(min=1e-6)
        return torch.stack([c[0] + f * (d @ right) / z, c[1] - f * (d @ up) / z], -1)
    return proj


def fit_clip(cr, clip, uv, conf, kps, s=SETTINGS, log=print, narrow=None):
    """(T, rig.n) the motion at the video's frames, a report, and the motion on the content's even grid
    (dict(times (N,), params (N, rig.n)): with the dropped content frames, for the motion file): the whole clip fitted
    at once (module docstring)."""
    rig = cr.rig
    views = cr.views(clip)
    T = min(len(uv), len(cr.times(clip)))
    uv, conf = np.asarray(uv, float)[:T], np.clip(np.asarray(conf, float)[:T], 0.0, 1.0)
    V = len(views)
    lo, hi, p0 = cr.bounds(clip)
    lo, hi = lo.copy(), hi.copy()
    for ch, (l, h) in (narrow or {}).items():
        lo[ch], hi[ch] = l, h
    kr = KeypointRig(cr, kps)
    X0 = kr.points(p0)
    t0 = time.time()
    tri = triangulate(views, uv, conf, X0, s['front_height'])
    start = start_params(cr, kps, tri, conf, p0, lo, hi, log=log)
    grid_t, at = content_grid(cr.times(clip)[:T])
    N = len(grid_t)
    if N > T:                                                             # a dropped frame starts between its neighbours
        start = np.stack([np.interp(np.arange(N), at, start[:, c]) for c in range(start.shape[1])], 1)
    dt = np.diff(grid_t)
    # 720-row pixels: the scales and weights hold for any frame size
    px = max(float(v['cam'].h) for v in views) / 720.0
    k = float(np.mean([v['cam'].px_per_unit for v in views])) / px
    root_ch = [kk for _, kk in sorted(rig.chan_of.get(rig.root, []))]
    joint_ch = [3 + c for c, (j, _) in enumerate(rig.channels) if j != rig.root]
    prop_move = [rig.prop0 + 6 * i + a for i in range(len(rig.props)) for a in (0, 1, 2)]
    prop_turn = [rig.prop0 + 6 * i + 3 + a for i in range(len(rig.props)) for a in (0, 1, 2)]
    order = np.array([0, 1, 2] + root_ch + joint_ch + prop_move + prop_turn)
    inv = np.argsort(order)
    f64 = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float64)
    jlo, jhi = f64(lo[joint_ch]), f64(hi[joint_ch])
    width = (jhi - jlo).clamp(min=1e-9)
    z0 = ((f64(start[:, joint_ch]) - jlo) / width).clamp(1e-3, 1 - 1e-3)
    move = f64(start[:, :3]).requires_grad_(True)
    turn = f64(start[:, root_ch] + 1e-4).requires_grad_(True)
    phi = torch.logit(z0).clone().requires_grad_(True)
    pmove = f64(start[:, prop_move].reshape(N, -1)).requires_grad_(True)
    pturn = f64(start[:, prop_turn].reshape(N, -1)).requires_grad_(True)
    seen = torch.as_tensor(at)                                            # the video's frames on the grid
    steps = None if N < 3 or np.ptp(dt) <= 1e-3 * dt.min() else f64(dt / np.median(dt))
    trig = TorchRig(cr, kps)
    projs = [projector(v['cam']) for v in views]
    obs = f64(uv / px)                                                    # (T, V, K, 2)
    wc = f64(conf) ** 2                                                   # (T, V, K)
    fi = front_panel(views)
    axis_w = torch.ones(V, 2, dtype=torch.float64)
    if fi is not None:
        axis_w[fi, 1] = s['front_height']
    up = f64(cr.up)
    feet = np.nonzero(trig.body & ((X0 @ cr.up - cr.floor) <= s['feet_share'] * cr.height))[0]
    pairs = end_pairs(kps)
    rest = f64(p0[joint_ch])
    log(f'fit {clip}: {T} frames, {V} panel(s), {len(kps)} keypoints, {len(joint_ch) // 3} bones, '
        f'{len(feet)} keypoints of the feet' + ('' if tri is not None else '; one panel: the start pose as the start') +
        (f'; {N - T} dropped content frames fitted too' if N > T else ''))

    def params():
        joints = jlo + width * torch.sigmoid(phi)
        return torch.cat([move, turn, joints, pmove, pturn], 1)[:, inv], joints

    floor_up = f64(cr.up / np.linalg.norm(cr.up))

    def loss_fn(scale, stand=None):
        p, joints = params()
        X = trig.points(p)                                                # (N, K, 3) on the grid
        pix = torch.stack([pr(X[seen]) for pr in projs], 1) / px          # (T, V, K, 2) the video's frames
        r2 = ((pix - obs) ** 2 * axis_w[None, :, None, :]).sum(-1)
        cost = wc * torch.log1p(r2 / scale ** 2)
        for a, b in pairs:                                                # a prop's two ends look alike
            r2x = ((pix[:, :, [b, a]] - obs[:, :, [a, b]]) ** 2 * axis_w[None, :, None, :]).sum(-1)
            crossed = (wc[:, :, [a, b]] * torch.log1p(r2x / scale ** 2)).sum(-1)
            pick = torch.minimum(cost[:, :, a] + cost[:, :, b], crossed)
            cost = torch.cat([cost[:, :, :a], pick[..., None], cost[:, :, a + 1:]], -1)
            cost = torch.cat([cost[:, :, :b], torch.zeros_like(pick)[..., None], cost[:, :, b + 1:]], -1)
        data = cost.sum()
        acc, vel = _change_costs(X, joints, turn, steps)
        acc, vel = acc * k ** 2 * s['w_acc'], vel * s['w_vel']
        prior = ((joints - rest) ** 2).sum() * s['w_rest']
        floor = (torch.relu(cr.floor - X[:, feet] @ up) ** 2).sum() * k ** 2 * s['w_floor'] if len(feet) else \
            torch.zeros((), dtype=torch.float64)
        parts = dict(data=data, acc=acc, vel=vel, prior=prior, floor=floor)
        if stand is not None:                                             # the standing feet stay where they stand
            Ff = X[:, feet] - (X[:, feet] @ floor_up)[..., None] * floor_up
            parts['contact'] = (((Ff[1:] - Ff[:-1]) ** 2).sum(-1) * stand).sum() * k ** 2 * s['w_contact']
        return sum(parts.values()), parts, pix

    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        opt = torch.optim.Adam([{'params': [move, pmove], 'lr': s['lr_move']},
                                {'params': [turn, pturn], 'lr': s['lr_turn']},
                                {'params': [phi], 'lr': s['lr_joint']}])
        iters = int(s['iters'])
        for it in range(iters):
            scale = s['scale_px'] + (s['scale_start_px'] - s['scale_px']) * max(0.0, 1 - it / (s['anneal'] * iters))
            if it == int(iters * s['lr_drop_at']):
                for g in opt.param_groups:
                    g['lr'] *= s['lr_drop']
            total, parts, _ = loss_fn(scale)
            opt.zero_grad()
            total.backward()
            opt.step()
            if it % 250 == 0:
                log(f'  step {it}: loss {total.item():.1f} ' +
                    ' '.join(f'{n} {v.item():.1f}' for n, v in parts.items()))
        stand = None
        if s['w_contact'] > 0 and len(feet):
            # for a game (SETTINGS): the feet's standing read off the fit so far, then more steps with it
            with torch.no_grad():
                Xs = trig.points(params()[0]).numpy()
            on = standing(Xs[:, feet], grid_t, cr.up, X0[feet] @ cr.up, cr.height, s)
            stand = f64(on[1:] * on[:-1])
            log(f'  contacts: {100 * on.mean():.0f}% of the feet\'s frames stand')
            for it in range(int(s['stage2_iters'])):
                total, parts, _ = loss_fn(s['scale_px'], stand)
                opt.zero_grad()
                total.backward()
                opt.step()
        with torch.no_grad():
            total, parts, pix = loss_fn(s['scale_px'], stand)
            p_grid = params()[0]
            X_grid = trig.points(p_grid).numpy()
            P_grid = p_grid.numpy()
            P = P_grid[at]
    finally:
        torch.set_num_threads(n_threads)
    miss = np.linalg.norm(pix.numpy() * px - uv, axis=-1)                 # (T, V, K) px of the clip
    sure = conf >= .3
    names = [v['name'] for v in views]
    report = dict(frames=T, secs=round(time.time() - t0, 1),
                  miss_px={n: round(float(miss[:, v].mean()), 2) for v, n in enumerate(names)},
                  sure_miss_px={n: (round(float(np.median(miss[:, v][sure[:, v]])), 2) if sure[:, v].any() else None)
                                for v, n in enumerate(names)},
                  loss={n: round(float(v.item()), 1) for n, v in parts.items()},
                  shake_px=round(shake(X_grid[:, trig.body_idx], k), 3))
    if N > T:
        report['frames_added'] = N - T
    motion = dict(times=grid_t, params=P_grid)
    if s['travel'] and len(feet):
        moved = travel(X_grid[:, feet], standing(X_grid[:, feet], grid_t, cr.up, X0[feet] @ cr.up, cr.height, s),
                       cr.up)
        P_out = P_grid.copy()
        P_out[:, :3] += moved
        for i in range(len(rig.props)):
            P_out[:, rig.prop0 + 6 * i:rig.prop0 + 6 * i + 3] += moved
        motion['params'] = P_out
        report['travel_heights'] = round(float(np.linalg.norm(moved[-1]) / cr.height), 2)
    if s['loop']:
        if isinstance(s['loop'], (list, tuple)):
            i, j = (int(np.argmin(np.abs(grid_t - float(v)))) for v in s['loop'])
        else:
            i, j = find_loop(X_grid[:, trig.body_idx], grid_t, cr.up, s['shortest_loop'])
        moves = [slice(0, 3)] + [slice(rig.prop0 + 6 * q, rig.prop0 + 6 * q + 3) for q in range(len(rig.props))]
        L, bent = close_loop(motion['params'], i, j, cr.up, moves)
        motion['loop'] = dict(times=grid_t[i:j + 1] - grid_t[i], params=L)
        report['loop'] = dict(start_s=round(float(grid_t[i]), 3), end_s=round(float(grid_t[j]), 3),
                              faded_rad=round(bent, 4))
        log(f'  loop: {grid_t[i]:.2f}-{grid_t[j]:.2f} s, its end faded into its start ({bent:.3f} rad at most)')
    log(f'fit {clip}: {T} frames in {report["secs"]} s; keypoint miss (mean px) ' +
        ', '.join(f'{n} {m}' for n, m in report['miss_px'].items()) + '; sure readings (median px) ' +
        ', '.join(f'{n} {m}' for n, m in report['sure_miss_px'].items()))
    return P, report, motion


def main():
    ap = argparse.ArgumentParser(prog='video2rig.reader.fit')
    ap.add_argument('project')
    ap.add_argument('clip')
    ap.add_argument('--reading', default='', help='work/<clip>/reader.npz (default)')
    ap.add_argument('--tag', default='-reader')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='one SETTINGS value for this run')
    a = ap.parse_args()
    from .road import export, fit_reading
    t0 = time.time()
    pr = Project(a.project)
    cfg = pr.clip_cfg(a.clip)
    cfg.setdefault('solve', {})
    settings = settings_for(cfg, {kv.split('=', 1)[0]: json.loads(kv.split('=', 1)[1]) for kv in a.set})
    d = np.load(a.reading or pr.work(a.clip) / 'reader.npz')
    kps = [tuple(k) for k in json.loads(str(d['keypoints']))]
    res = fit_reading(pr, a.clip, cfg, d['uv'], d['conf'], kps, [str(v) for v in d['views']], s=settings)
    cr, P, report = res['creature'], res['params'], res['report']
    path = cr.save(a.clip, P, a.tag, {'reader': True})
    out = export(pr, cr, a.clip, res['motion'], a.tag)
    write_json(pr.work(a.clip) / f'fit{a.tag}-report.json', report)
    print(f'FIT {a.clip}: {len(P)} frames in {time.time() - t0:.1f} s -> {path}, {out}', flush=True)


if __name__ == '__main__':
    main()
