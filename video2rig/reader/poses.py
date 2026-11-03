"""Training poses for the joint reader: the creature's own rig in random poses within the ranges the fit allows
(Creature.bounds), the body turned and shifted a little as Seedance's creatures move, or placed anywhere the start
frame leaves it room, standing on the floor or in the air, or lying (LIE: on the belly with the legs folded, or on
its side); the jaw opened in half of them (the fit does not move
it, but a bite shows it open); in about half, the bones also move over the frame (quats_a -> quats_b), which the
renders turn into motion blur, in some of those the whole body lunging or spinning.

    python -m video2rig.reader.poses <project> [--n 3000] [--seed 0]     -> <project>/reader/poses.npz

No motion of any kind goes in (no game animation, no earlier fit): the reader learns the creature, not a move.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .. import clip as clipmod
from ..common import Project
from ..creature import Creature
from ..rig import rot
from . import side_of, twin_label

JAW = re.compile(r'jaw', re.I)
TOGETHER, ONTO_BODY = .18, .17     # share of the poses with two free limbs' ends brought together / one end laid on
                                   # the creature's own body
STRIKE = .25                       # share of the moving poses in which the free limbs move 2.5 times as fast
ROAM = .5                          # share of the poses placed anywhere in the start frame's room, not at the start
LUNGE = .3                         # share of the moving poses in which the whole body lunges, charges or spins
TURN = 1.2                         # the fastest whole-body turn in one frame, radians (69 degrees: 4.6 turns a second)
TRAVEL = .15                       # the fastest whole-body travel in one frame, a share of the body's longest side
LIE = .12                          # share of the poses lying down: on the belly with the standing legs folded under
                                   # it (a cow asleep), or on its side (dead, or rolling); standing poses alone never
                                   # showed the reader a body on the floor


def axis_rot(axis, angle):
    return Rotation.from_rotvec(np.asarray(axis, float) * angle).as_matrix()


def free_limbs(cr):
    """The limbs the creature does not stand on (arms, wings), as chains of bones, hub side first."""
    rig = cr.rig
    return [ch for ch in cr.limb_chains(as_list=True)
            if side_of(rig.names[ch[0]]) is not None and not set(ch) & cr.legs]


def reacher(cr, lo, hi, coords, idx, wts):
    """reach(p, rng): p with, in some of the poses, the ends of two twin free limbs brought to one place (two
    hands on one hilt, hands clasped, wing tips folded together) or one free limb's end laid on the creature's own
    body (a hand on the chest, on the other arm, on the head). Joint angles drawn one by one almost never do
    either, and those are the poses where one camera cannot see the hand: a centaur with both hands on its sword
    over its head had its left wrist "read" on its own forearm guard, sure of it (conf .6-.8), in half the frames
    (attack01, 33-54). The limb is bent there by its own joints within their ranges, staying near the pose drawn."""
    rig = cr.rig
    limbs = free_limbs(cr)
    if not limbs:
        return lambda p, rng: p
    cols = [[c for c in sorted({c for j in ch for _, c in rig.chan_of.get(j, [])}) if hi[c] - lo[c] > 1e-6]
            for ch in limbs]
    H0, T0 = rig.bone_points(cr.p0(None))
    length = [float(sum(np.linalg.norm(T0[j] - H0[j]) for j in ch)) for ch in limbs]
    first = {rig.names[ch[0]]: i for i, ch in enumerate(limbs)}
    twins = [(i, first[twin_label(rig.names[ch[0]])]) for i, ch in enumerate(limbs)
             if side_of(rig.names[ch[0]]) == 'L' and twin_label(rig.names[ch[0]]) in first]
    top = idx[np.arange(len(idx)), np.argmax(wts, axis=1)]            # the bone each sampled vertex follows most
    fwd, up = cr.facing, cr.up

    def bring(p, i, target):
        j, cs = limbs[i][-1], cols[i]
        if not cs:
            return
        q0 = np.clip(p[cs], lo[cs], hi[cs])

        def res(z):
            q = p.copy()
            q[cs] = z
            H, T = rig.bone_points(q)
            return np.concatenate([40.0 * (.5 * (H[j] + T[j]) - target) / cr.height, .15 * (z - q0)])
        p[cs] = least_squares(res, q0, bounds=(lo[cs], hi[cs]), max_nfev=30).x

    def reach(p, rng):
        u = rng.random()
        if twins and u < TOGETHER:
            a, b = twins[int(rng.integers(len(twins)))]
            H, T = rig.bone_points(p)
            d = rng.normal(0, 1, 3) + fwd + .3 * up                  # mostly before the body, also above and beside
            target = .5 * (H[limbs[a][0]] + H[limbs[b][0]]) \
                + d / np.linalg.norm(d) * rng.uniform(.35, .9) * min(length[a], length[b])
            off = rng.normal(0, 1, 3)                                # the two ends side by side, up to an end bone apart
            off *= rng.uniform(0, .5) * np.linalg.norm(T[limbs[a][-1]] - H[limbs[a][-1]]) / np.linalg.norm(off)
            bring(p, a, target + off)
            bring(p, b, target - off)
        elif u < TOGETHER + ONTO_BODY:
            i = int(rng.integers(len(limbs)))
            X = rig.skin(p, coords, idx, wts)
            H, _ = rig.bone_points(p)
            near = ~np.isin(top, limbs[i]) & (np.linalg.norm(X - H[limbs[i][0]], axis=1) < .95 * length[i])
            if near.any():
                bring(p, i, X[int(rng.choice(np.nonzero(near)[0]))])
        return p
    return reach


def roamer(cr, coords, idx, wts):
    """place(p, rng): p moved along the floor. Some poses stay near the start place, as most clips do; the rest go
    anywhere the body still fits inside every panel of the start frame. A start frame rendered with room for a
    charge (render_start.py --margin-wide, --lead) has the creature all over that room in the video, and the
    reader had only ever seen it within 5% of its height of where it starts. A tight start frame has no room:
    nothing fits but the start place, and the poses come out as before."""
    rig = cr.rig
    up, fwd = cr.up, cr.facing
    lat = np.cross(up, fwd)
    lat /= np.linalg.norm(lat)
    path = cr.project.start_dir / 'cameras.json'
    cams = []
    if path.exists():
        cj = json.loads(path.read_text(encoding='utf-8'))
        pivot = rig.skin(cr.p0(None), coords, idx, wts).mean(0)
        cams = [clipmod.Camera(rec, cj['frame_px'], cj['frame_px'], cr.data['armature_world'], pivot)
                for rec in cj['views']]
    # how far a panel reaches, in the creature's own units (at the body's distance for a camera with perspective)
    reach = max([max(c.w, c.h) / c.px_per_unit for c in cams], default=0.0)

    def inside(X):
        for c in cams:
            pix = c.project(X)
            size = np.array([c.w, c.h])
            if (pix.min(0) < .03 * size).any() or (pix.max(0) > .97 * size).any():
                return False
        return True

    def place(p, rng):
        near = fwd * rng.normal(0, .05 * cr.height) + lat * rng.normal(0, .04 * cr.height)
        if cams and rng.random() < ROAM:
            X = rig.skin(p, coords, idx, wts)
            for _ in range(10):
                off = fwd * rng.uniform(-.5, .5) * reach + lat * rng.uniform(-.5, .5) * reach
                if inside(X + off):
                    p[:3] += off
                    return p
        p[:3] += near
        return p
    return place


def sample(cr, n, seed=0, log=print):
    rig = cr.rig
    lo, hi, p0 = cr.bounds(None)
    rng = np.random.default_rng(seed)
    up, fwd = cr.up, cr.facing
    lat = np.cross(up, fwd)
    lat /= np.linalg.norm(lat)
    # the body's centre and lowest point from a fixed sample of the skin (a tenth of the vertices: millimetres off,
    # ten times faster)
    pick = np.random.default_rng(seed + 7).choice(len(cr.idx), min(len(cr.idx), 1200), replace=False)
    coords, idx, wts = cr.mesh['coords'][pick], cr.idx[pick], cr.wts[pick]
    root_chans = [k for _, k in sorted(rig.chan_of[rig.root])]
    chans = []
    for j, a in rig.channels:
        if j == rig.root:
            continue
        c = rig.index(rig.names[j], a)
        if rig.names[j] in cr.hinge and a == cr.bend_axis(j):
            chans.append((c, 'hinge', 0.0))
            continue
        role = cr.role.get((j, a), 'bend')
        reach = max(hi[c] - p0[c], p0[c] - lo[c])
        sd = {'bend': .5 * min(reach, 1.6), 'twist': .3 * min(reach, 1.2), 'side': .4 * min(reach, 1.2)}[role]
        chans.append((c, role, sd))
    params = np.zeros((n, rig.n))
    reach = reacher(cr, lo, hi, coords, idx, wts)
    place = roamer(cr, coords, idx, wts)
    leg_hinges = {rig.index(rig.names[j], cr.bend_axis(j)) for j in cr.legs if rig.names[j] in cr.hinge}
    for i in range(n):
        calm = rng.random() < .3                      # a third near the stance: walks and idles are common
        lie = None if calm or rng.random() >= LIE / .7 else ('belly' if rng.random() < .5 else 'side')
        k = .4 if calm else 1.0
        p = p0.copy()
        for c, role, sd in chans:
            v = rng.uniform(lo[c], hi[c]) if role == 'hinge' else p0[c] + rng.normal(0.0, sd)
            if lie == 'belly' and c in leg_hinges:
                # folded: from half way to fully bent, toward the end of its range far from straight
                bent = hi[c] if hi[c] - p0[c] > p0[c] - lo[c] else lo[c]
                v = p0[c] + rng.uniform(.5, 1.0) * (bent - p0[c])
            p[c] = np.clip(p0[c] + k * (v - p0[c]), lo[c], hi[c])
        p = reach(p, rng)
        # the whole body: turned about its centre (yaw round, pitch as in a jump or a dive, a little roll)
        G, P, _ = rig.fk(p)
        X = rig.skin(p, coords, idx, wts)
        cen = X.mean(0)
        yaw = np.clip(rng.normal(0, np.radians(12)), -np.radians(30), np.radians(30)) * k
        pitch = np.clip(rng.normal(0, np.radians(8)), -np.radians(25), np.radians(25)) * k
        if not calm and rng.random() < .2:
            # a deep lean: a troll running like a gorilla (Seedance bent it 60+ degrees), a dive, a rearing horse;
            # with 25 degrees at most the reader never saw one and lost the troll's run
            pitch = np.clip(rng.normal(0, np.radians(30)), -np.radians(75), np.radians(75))
        if not calm and rng.random() < .25:
            # turned any way round: a spin, a turn away (an alien's combo spun once; with 30 degrees at most the
            # reader never saw its side or its back and the fit stayed facing the camera through the spin)
            yaw = rng.uniform(-np.pi, np.pi)
        roll = np.clip(rng.normal(0, np.radians(4)), -np.radians(12), np.radians(12)) * k
        if lie == 'belly':
            pitch = np.clip(rng.normal(0, np.radians(8)), -np.radians(20), np.radians(20))
            roll = rng.normal(0, np.radians(10))
        elif lie == 'side':
            pitch = rng.normal(0, np.radians(10))
            roll = rng.choice([-1.0, 1.0]) * rng.uniform(np.radians(65), np.radians(105))
        Q = axis_rot(up, yaw) @ axis_rot(lat, pitch) @ axis_rot(fwd, roll)
        j = rig.root
        par = rig.parents[j]
        base = (G[par] if par >= 0 else np.eye(3)) @ rig.Rrl[j]
        w = Rotation.from_matrix(base.T @ Q @ G[j]).as_rotvec()
        p[root_chans] = w
        p[:3] += cen + Q @ (P[j] - cen) - P[j]
        # on the floor (or up to a third of its height in the air), shifted a little round its place
        X = rig.skin(p, coords, idx, wts)
        jump = 0.0 if lie or rng.random() < .7 else rng.uniform(0, .3) * cr.height
        p[:3] += up * (cr.floor - (X @ up).min() + jump)
        p = place(p, rng)
        place_props(cr, p, rng)
        params[i] = p
        if (i + 1) % 500 == 0:
            log(f'poses {i + 1}/{n}')
    return params


HAND = re.compile(r'hand(\.[lr]|_[lr])?$', re.I)


def place_props(cr, p, rng):
    """Every prop somewhere the clip may show it: a quarter of the poses where it lay at the start (turned a little
    on the floor), the rest between the pose's hands (or in front of the chest) turned any way, never under the floor.
    The reader learns to find it anywhere, held or not."""
    rig = cr.rig
    if not rig.props:
        return
    H, _ = rig.bone_points(p)
    hands = [j for j, nm in enumerate(rig.names) if HAND.search(nm)]
    up = cr.up
    for k in range(len(rig.props)):
        base = rig.prop0 + 6 * k
        c0 = rig.prop_centre[k]
        X0 = cr.mesh['coords'][cr.idx[:, 0] == rig.nb + k]
        if rng.random() < .25:
            R = axis_rot(up, rng.normal(0, np.radians(15)))
            target = c0 + rng.normal(0, .03 * cr.height, 3) * (1 - np.abs(up))
        else:
            R = Rotation.random(random_state=int(rng.integers(1 << 31))).as_matrix()
            if rng.random() < .5:                                     # level-ish, as logs and stones are carried
                R = axis_rot(up, rng.uniform(-np.pi, np.pi)) @ axis_rot(np.cross(up, cr.facing),
                                                                        rng.normal(0, np.radians(25)))
            centre = H[hands].mean(0) if hands else H.mean(0)
            target = centre + rng.normal(0, .06 * cr.height, 3)
        t = target - c0
        Xp = (X0 - c0) @ R.T + c0 + t
        t += up * max(0.0, cr.floor - float((Xp @ up).min()))
        p[base:base + 3] = t
        p[base + 3:base + 6] = Rotation.from_matrix(R).as_euler('XYZ')      # rig.prop_rotation: Rx Ry Rz


def prop_matrices(rig, params):
    """(n, props, 4, 4) every prop's matrix in armature space (rig.pose_basis's), per pose."""
    out = np.zeros((len(params), len(rig.props), 4, 4))
    for i, p in enumerate(params):
        for k in range(len(rig.props)):
            R = rig.prop_rotation(p, k)
            c0 = rig.prop_centre[k]
            out[i, k] = np.eye(4)
            out[i, k, :3, :3] = R
            out[i, k, :3, 3] = c0 + p[rig.prop0 + 6 * k:rig.prop0 + 6 * k + 3] - R @ c0
    return out


def velocities(cr, params, rng, share=.55):
    """Per pose, a change of every channel over one video frame (zero in the poses rendered sharp): the shutter of
    blender/render_poses.py smears the creature along it, the way Seedance blurs a galloping leg. Limbs move
    fastest, the legs it stands on and the free ones alike (arms were as slow as the trunk here, and a sword strike
    is the most blurred thing Seedance draws), the tail next, the trunk and head least; the whole body turns and
    drifts a little. In STRIKE of the moving poses the free limbs go 2.5 times as fast: a strike, a throw. In LUNGE
    of them the whole body goes fast (below)."""
    rig = cr.rig
    lo, hi, p0 = cr.bounds(None)
    sd = np.zeros(rig.n)
    sd[:3] = .01 * cr.height
    free = {j for ch in free_limbs(cr) for j in ch}
    free_cols = []
    for j, a in rig.channels:
        c = rig.index(rig.names[j], a)
        if j == rig.root:
            sd[c] = .03
        elif j in cr.legs or j in free:
            sd[c] = .2 if a != 1 else .05
            if j in free:
                free_cols.append(c)
        elif re.search(r'tail', rig.names[j], re.I):
            sd[c] = .15
        else:
            sd[c] = .06
    moving = rng.random(len(params)) < share
    v = rng.normal(0, 1, params.shape) * sd * moving[:, None]
    if free_cols:
        strike = np.nonzero(rng.random(len(params)) < STRIKE)[0]
        v[np.ix_(strike, free_cols)] *= 2.5
    # The whole body fast: a charge, a lunge, a spin. A reader is trained once for its creature and must then read
    # ANY move of it, so the pictures cover every speed a body can have, not the moves seen so far: in LUNGE of
    # the moving poses the body turns anything up to TURN and travels anything up to TRAVEL in the frame, spread
    # evenly from slow to the fastest, its limbs half as fast again (they gallop under a charge). Measured on the
    # first reader of a lizard: the body turned 16 degrees a frame at most and travelled 2% of its length in its
    # training pictures; its first charge with a spin turned up to 55 degrees a frame and travelled 11%.
    size = float(np.ptp(cr.mesh['coords'], axis=0).max())
    lunge = np.nonzero(moving & (rng.random(len(params)) < LUNGE))[0]
    limb_cols = [rig.index(rig.names[j], a) for j, a in rig.channels if j in cr.legs or j in free]
    v[np.ix_(lunge, limb_cols)] *= 1.5
    pa, pb = np.clip(params - v, lo, hi), np.clip(params + v, lo, hi)
    up, fwd = cr.up, cr.facing
    lat = np.cross(up, fwd)
    lat /= np.linalg.norm(lat)
    root_chans = [k for _, k in sorted(rig.chan_of[rig.root])]
    j = rig.root
    par = rig.parents[j]
    for i in lunge:
        yaw = rng.uniform(-.5, .5) * TURN                    # each way from the middle pose: half of the frame's
        lean = rng.normal(0, .05, 2)
        d = fwd * rng.normal(0, 1) + lat * rng.normal(0, .5) + up * rng.normal(0, .2)
        d *= rng.uniform(0, .5) * TRAVEL * size / max(np.linalg.norm(d), 1e-9)
        for q, sgn in ((pa, -1.0), (pb, 1.0)):
            p = q[i]
            G, P, _ = rig.fk(p)
            H, _ = rig.bone_points(p)
            cen = H.mean(0)
            Q = axis_rot(up, sgn * yaw) @ axis_rot(lat, sgn * lean[0]) @ axis_rot(fwd, sgn * lean[1])
            base = (G[par] if par >= 0 else np.eye(3)) @ rig.Rrl[j]
            p[root_chans] = Rotation.from_matrix(base.T @ Q @ G[j]).as_rotvec()
            p[:3] += cen + Q @ (P[j] - cen) - P[j] + sgn * d
    return pa, pb


def jaw_turns(cr, n, rng):
    """Per pose, per jaw bone: the extra turn (radians) about its own opening axis, open in half of the poses."""
    rig = cr.rig
    out = {}
    p0 = cr.p0(None)
    G, P, _ = rig.fk(p0)
    lat = np.cross(cr.up, cr.facing)
    for j, name in enumerate(rig.names):
        if not JAW.search(name) or j in rig.chan_of:
            continue
        par = rig.parents[j]
        Gj = (G[par] if par >= 0 else np.eye(3)) @ rig.Rrl[j] @ rig.fixed[j]
        a = max((0, 2), key=lambda a: abs(Gj[:, a] @ lat))         # the axis nearest the sideways one
        tip = lambda th: (Gj @ rot(a, th))[:, 1] @ cr.up
        sign = 1.0 if tip(.3) < tip(-.3) else -1.0                 # opening = the tip goes down
        ang = np.where(rng.random(n) < .5, rng.uniform(0, .55, n), 0.0) * sign
        out[name] = (a, ang)
    return out


def rot_batch(axis, a):
    """rig.rot for an array of angles: (n, 3, 3)."""
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    if axis == 0:
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == 1:
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.moveaxis(np.array(m), [0, 1], [-2, -1])


def pose_basis_batch(rig, params, jaws):
    """Rig.pose_basis for every pose at once (quaternions w x y z, the root's location), the jaw turns applied."""
    n = len(params)
    R = np.broadcast_to(rig.fixed, (n, rig.nb, 3, 3)).copy()
    for j, chans in rig.chan_of.items():
        if j == rig.root:
            R[:, j] = Rotation.from_rotvec(params[:, [k for _, k in sorted(chans)]]).as_matrix()
            continue
        M = np.broadcast_to(np.eye(3), (n, 3, 3))
        for a, k in chans:
            M = M @ rot_batch(a, params[:, k])
        R[:, j] = M
    for name, (ax, ang) in jaws.items():
        j = rig.names.index(name)
        R[:, j] = rig.fixed[j] @ rot_batch(ax, ang)
    quats = Rotation.from_matrix(R.reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(n, rig.nb, 4)
    G, _, _ = rig.fk(params[0])
    par = rig.parents[rig.root]
    assert par < 0 or par not in rig.chan_of, 'a fitted bone above the root'
    frame = (G[par] if par >= 0 else np.eye(3)) @ rig.Rrl[rig.root]
    return quats, params[:, :3] @ frame


def main():
    ap = argparse.ArgumentParser(prog='video2rig.reader.poses')
    ap.add_argument('project')
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--blur', type=float, default=.55, help='share of the poses rendered with motion blur')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    t0 = time.time()
    pr = Project(a.project)
    cfg = pr.clip_cfg('')
    cfg.setdefault('solve', {})
    cr = Creature(pr, cfg)
    rig = cr.rig
    params = sample(cr, a.n, a.seed)
    rng = np.random.default_rng(a.seed + 1)
    jaws = jaw_turns(cr, a.n, rng)
    quats, locs = pose_basis_batch(rig, params, jaws)
    pa, pb = velocities(cr, params, np.random.default_rng(a.seed + 2), share=a.blur)
    quats_a, locs_a = pose_basis_batch(rig, pa, jaws)
    quats_b, locs_b = pose_basis_batch(rig, pb, jaws)
    out = Path(a.out) if a.out else pr.root / 'reader' / 'poses.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, bones=np.array(rig.names), root=rig.names[rig.root], quats=quats, root_location=locs,
                        quats_a=quats_a, root_location_a=locs_a, quats_b=quats_b, root_location_b=locs_b,
                        params=params, seed=a.seed, prop_names=np.array([pr['name'] for pr in rig.props]),
                        prop_mats=prop_matrices(rig, params))
    print(f'POSES {a.n} -> {out} ({time.time() - t0:.1f} s)', flush=True)


if __name__ == '__main__':
    main()
