"""The two panels put in step. A video model sometimes plays one panel ahead of the other, and not always by the same
amount (a trot's front view a few frames early, later in step again); then a joint's two readings come from two
moments and no 3D point fits both. Two orthographic panels both see the axis across their lines of sight (the up-down
axis of a front and a side panel), so every joint's place along it is read twice. The panels are matched frame to
frame on those places (dynamic time warping, left and right not trusted: a pair of twin joints counts as its mean place
and its spread), and when the match cuts the panels' disagreement along that axis by GAIN, the front panel's readings
are resampled onto the other panel's frames.

    python -m video2rig.reader.sync <project> <clip>      how far the panels are out of step; nothing is written
"""
import numpy as np

from . import is_prop, twin_label

BAND = 48            # frames a match may wander from the same frame
HOLD_PX = 1.0        # cost of holding one panel on a frame while the other moves on (px of a 720-row frame)
GAIN = .15           # the match must cut the panels' disagreement by this share


def shared_axis(views):
    """For two orthographic panels: the direction both see (across both view axes; armature space) and, per panel, the
    pixels a step of 1 along it moves a point. None for other cameras or panels looking the same way."""
    if len(views) != 2 or any(not hasattr(v['cam'], 'A') or v['cam'].type != 'ortho' for v in views):
        return None
    n = [np.cross(v['cam'].A[0], v['cam'].A[1]) for v in views]       # each panel's view axis
    e = np.cross(n[0], n[1])
    if np.linalg.norm(e) < .2 * np.linalg.norm(n[0]) * np.linalg.norm(n[1]):
        return None
    e /= np.linalg.norm(e)
    return e, [v['cam'].A @ e for v in views]


def places(views, uv):
    """(T, 2, K) every keypoint's place along the axis both panels see, as each panel reads it, in px of a 720-row
    frame; None for other cameras."""
    sa = shared_axis(views)
    if sa is None:
        return None
    _, g = sa
    px = max(float(v['cam'].h) for v in views) / 720.0
    scale = float(np.mean([np.linalg.norm(gi) for gi in g])) / px
    return np.stack([(uv[:, v] - views[v]['cam'].b) @ g[v] / (g[v] @ g[v]) * scale for v in (0, 1)], 1)


def features(s, c, kps):
    """Per frame, places that do not care which twin is which: a middle joint's place as read, a pair of twin joints'
    mean place and their spread. (T, F) and their weights (T, F)."""
    labels = [k[0] for k in kps]
    f, w = [], []
    for k, kp in enumerate(kps):
        if is_prop(kp):
            continue
        tw = twin_label(kp[0])
        if tw is None or tw not in labels:
            f.append(s[:, k]); w.append(c[:, k])
        elif labels.index(tw) > k:
            j = labels.index(tw)
            cc = np.minimum(c[:, k], c[:, j])
            f += [(s[:, k] + s[:, j]) / 2, np.abs(s[:, k] - s[:, j])]
            w += [cc, cc]
    return np.stack(f, 1), np.stack(w, 1).clip(0, 1)


def gap(s, conf, kps):
    """The panels' mean disagreement along the shared axis (px), weighted by how sure both readings are."""
    body = np.array([not is_prop(k) for k in kps])
    c = np.minimum(conf[:, 0], conf[:, 1])[:, body].clip(0, 1)
    d = np.abs(s[:, 0, body] - s[:, 1, body])
    return float(((c * d).sum(1) / c.sum(1).clip(1e-6)).mean())


def match_frames(F, Fw, S, Sw, band=BAND, hold=HOLD_PX):
    """For every frame of panel S, the frame of panel F that matches it (dynamic time warping, smoothed over 7 frames
    so the resampled readings do not move in steps)."""
    T = len(F)
    C = np.full((T, T), np.inf)
    for i in range(T):
        lo_, hi_ = max(0, i - band), min(T, i + band + 1)
        w = np.minimum(Fw[i][None], Sw[lo_:hi_])
        C[i, lo_:hi_] = (w * np.abs(F[i][None] - S[lo_:hi_])).sum(1) / w.sum(1).clip(1e-6)
    D = np.full((T, T), np.inf)
    D[0, 0] = C[0, 0]
    for i in range(T):
        for j in range(max(0, i - band), min(T, i + band + 1)):
            if i == j == 0:
                continue
            best = min(D[i - 1, j - 1] if i and j else np.inf, D[i - 1, j] + hold if i else np.inf,
                       D[i, j - 1] + hold if j else np.inf)
            D[i, j] = C[i, j] + best
    i, j, path = T - 1, T - 1, [(T - 1, T - 1)]
    while i or j:
        steps = [(D[i - 1, j - 1], i - 1, j - 1) if i and j else (np.inf, 0, 0),
                 (D[i - 1, j], i - 1, j) if i else (np.inf, 0, 0), (D[i, j - 1], i, j - 1) if j else (np.inf, 0, 0)]
        _, i, j = min(steps)
        path.append((i, j))
    path = np.array(path[::-1])
    m = np.array([path[path[:, 1] == j, 0].mean() for j in range(T)])
    return np.convolve(np.pad(m, 3, mode='edge'), np.ones(7) / 7, mode='valid')


def resample(a, m):
    """a (T, ...) read at the (fractional) frames m (T,), linear between frames."""
    T = len(a)
    flat = a.reshape(T, -1)
    t = np.arange(T)
    return np.stack([np.interp(m, t, flat[:, i]) for i in range(flat.shape[1])], 1).reshape(a.shape)


def find_match(views, uv, conf, kps):
    """dict(panel, gap_px, gap_matched_px, shift_mean, shift_max, match) for two orthographic panels: the panel to move
    (the front one), how far the panels disagree as read and once matched, the match's shift in frames; None for
    other cameras. 'uv', 'conf' hold the moved panel's resampled readings."""
    s = places(views, uv)
    if s is None:
        return None
    move = next((i for i, v in enumerate(views) if 'front' in str(v['name']).lower()), 0)
    other = 1 - move
    F, Fw = features(s[:, move], conf[:, move], kps)
    S, Sw = features(s[:, other], conf[:, other], kps)
    m = match_frames(F, Fw, S, Sw)
    uv2, conf2 = np.array(uv, float), np.array(conf, float)
    uv2[:, move], conf2[:, move] = resample(uv2[:, move], m), resample(conf2[:, move], m)
    shift = m - np.arange(len(m))
    return dict(panel=move, gap_px=gap(s, conf, kps), gap_matched_px=gap(places(views, uv2), conf2, kps),
                shift_mean=float(np.abs(shift).mean()), shift_max=float(np.abs(shift).max()), match=m,
                uv=uv2, conf=conf2)


def align(views, uv, conf, kps, log=print):
    """uv, conf with the front panel matched onto the other when the match pays (GAIN), and the record (+ 'applied';
    None for other cameras)."""
    rec = find_match(views, uv, conf, kps)
    if rec is None:
        return uv, conf, None
    uv2, conf2, m = rec.pop('uv'), rec.pop('conf'), rec.pop('match')
    gain = 1 - rec['gap_matched_px'] / max(rec['gap_px'], 1e-6)
    rec['applied'] = bool(gain > GAIN)
    rec = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items()}
    if not rec['applied']:
        log(f'panels in step: matching them would cut their disagreement by {100 * max(gain, 0.0):.0f}% only; '
            'left as read')
        return uv, conf, rec
    rec['shift_frames'] = [round(float(x), 2) for x in m - np.arange(len(m))]
    # the panel's pictures and outlines move with it (to the nearest frame), so the review shows the moment its
    # readings came from
    p = rec['panel']
    view = views[p]
    if 'retimed' not in view:
        T = len(m)
        near = np.clip(np.round(m).astype(int), 0, T - 1)
        frame0, mask0 = view['frame'], view['mask']
        view['frame'] = lambda i, f=frame0: f(int(near[min(i, T - 1)]))
        view['mask'] = lambda i, f=mask0: f(int(near[min(i, T - 1)]))
        view['retimed'] = True
    log(f'panels in step: the {view["name"]} panel matched onto the other frame by frame (shift '
        f'{rec["shift_mean"]:.1f} frames on average, {rec["shift_max"]:.1f} at most), their disagreement '
        f'{rec["gap_px"]:.1f} -> {rec["gap_matched_px"]:.1f} px')
    return uv2, conf2, rec


def main():
    import argparse
    import json
    from ..common import Project
    from ..creature import Creature
    ap = argparse.ArgumentParser(prog='video2rig.reader.sync')
    ap.add_argument('project')
    ap.add_argument('clip')
    a = ap.parse_args()
    pr = Project(a.project)
    d = np.load(pr.work(a.clip) / 'reader.npz')
    kps = [tuple(k) for k in json.loads(str(d['keypoints']))]
    names = [str(v) for v in d['views']]
    cfg = pr.clip_cfg(a.clip)
    cfg.setdefault('solve', {})
    cr = Creature(pr, cfg)
    views = cr.views(a.clip)
    order = [names.index(v['name']) for v in views]
    rec = find_match(views, d['uv'][:, order], d['conf'][:, order], kps)
    if rec is not None:
        for k in ('uv', 'conf', 'match'):
            rec.pop(k)
    print('SYNC', a.clip, json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
