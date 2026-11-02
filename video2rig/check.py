"""Is a Seedance clip fit to be fitted? Seconds after reading it, before the fit: how far its two panels disagree on
the body and on the limbs, as a share of the creature's height, and whether its first frame is the start image.

    python -m video2rig check <project> <clip>          (exit code 1 when the clip should be generated again)

Seedance does not always draw the same motion in both panels. Two orthographic panels see every point of the body
twice; where they cannot agree on one 3D point, there is no one true motion to fit, and no rule in the fit can make
one. Measured (dev/DECISION-2026-10-11.md): bench clips agree on the body to 0.3-0.9% of the height, clean real clips
to about 1.2%; the clips that drove the fit's one-clip rules were at 1.6-4.7%. Limbs disagree more on fast runs
(2.1-3.6%): Seedance again, or the reader mixing up near and far legs: a warning.
"""
import numpy as np

BODY_FAIL = .015            # share of the height: over this the body is drawn differently in the two panels
LIMBS_WARN = .02            # limbs: look at the review film
FRAME0_WARN = .85           # the start pose covers the first frame's outline less than this: not our start image


def panel_misses(views, uv, conf, kps, sure=.3):
    """Per frame, the median px by which two orthographic panels miss one 3D point, over the sure readings (conf
    `sure`+ in both panels) of the centre line (trunk, neck, head, tail) and of the limbs: two arrays (NaN where
    none is sure), or (None, None) for other cameras."""
    from .reader import side_of
    if len(views) != 2 or any(not hasattr(v['cam'], 'A') or v['cam'].type != 'ortho' for v in views):
        return None, None
    A = np.concatenate([v['cam'].A for v in views], 0)
    b = np.concatenate([v['cam'].b for v in views], 0)
    H = A @ np.linalg.pinv(A) - np.eye(len(A))
    centre = np.array([side_of(kp[0]) is None and not str(kp[1]).startswith('@') for kp in kps])
    limb = np.array([side_of(kp[0]) is not None and not str(kp[1]).startswith('@') for kp in kps])
    T = len(uv)
    out_c, out_l = np.full(T, np.nan), np.full(T, np.nan)
    for t in range(T):
        ok = (conf[t] >= sure).all(0)
        r = np.linalg.norm((np.concatenate([uv[t, 0], uv[t, 1]], 1) - b) @ H.T, axis=1)
        if (ok & centre).any():
            out_c[t] = np.median(r[ok & centre])
        if (ok & limb).any():
            out_l[t] = np.median(r[ok & limb])
    return out_c, out_l


def check(project, clip, log=print):
    """The clip's numbers and a verdict: OK, WARN (look at it) or FAIL (generate it again)."""
    from .creature import Creature
    from .reader.road import read_clip
    uv, conf, kps, names = read_clip(project, clip, log=log)
    cfg = project.clip_cfg(clip)
    cfg.setdefault('solve', {})
    cr = Creature(project, cfg)
    views = cr.views(clip)
    order = [names.index(v['name']) for v in views]
    uv, conf = np.asarray(uv)[:, order], np.asarray(conf)[:, order]
    height = cr.height * float(np.mean([np.linalg.norm(v['cam'].A[1]) for v in views if hasattr(v['cam'], 'A')]))
    # judged as the fit will take it: a panel drawn a few frames behind the other put in step first (reader/sync.py)
    from .reader.sync import align
    uv, conf, sync = align(views, uv, conf, kps, log=lambda m: None)
    body, limbs = panel_misses(views, uv, conf, kps)
    frame0 = [a['frame0_iou'] for a in cr._clip[clip]['aligned']]
    out = dict(clip=clip, frames=len(uv), height_px=round(height, 1), frame0_outline=frame0, panels_in_step=sync)
    msgs = []
    if sync and sync['applied']:
        msgs.append(f'NOTE: the {views[sync["panel"]]["name"]} panel was out of step (up to {sync["shift_max"]:.1f} '
                    f'frames): matched frame by frame, the panels disagree by {sync["gap_px"]:.1f} -> '
                    f'{sync["gap_matched_px"]:.1f} px')
    if body is not None:
        q = lambda x, p: round(float(np.nanpercentile(x, p)) / height, 4) if not np.isnan(x).all() else None
        out.update(body=q(body, 50), body_p90=q(body, 90), limbs=q(limbs, 50), limbs_p90=q(limbs, 90))
        if out['body'] is not None and out['body'] > BODY_FAIL:
            msgs.append(f'FAIL: the panels disagree on the body by {100 * out["body"]:.1f}% of its height (median; '
                        f'clean clips up to about 1.2%): Seedance drew the move differently in each panel, generate it '
                        f'again')
        if out['limbs'] is not None and out['limbs'] > LIMBS_WARN:
            msgs.append(f'WARN: the panels disagree on the limbs by {100 * out["limbs"]:.1f}% of the height: fast legs '
                        f'drawn differently, or near and far legs mixed up; watch the limbs in the review film')
    else:
        msgs.append('WARN: not two orthographic panels: no panel check')
    if min(frame0) < FRAME0_WARN:
        msgs.append(f'WARN: the start pose covers the first frame only {min(frame0):.2f}: is this clip from this '
                    f'project\'s start image?')
    out['verdict'] = 'FAIL' if any(m.startswith('FAIL') for m in msgs) else \
        'WARN' if any(m.startswith('WARN') for m in msgs) else 'OK'
    out['messages'] = msgs
    return out
