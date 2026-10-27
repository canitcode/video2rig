"""Where the elbows, knees and paws of a four-legged animal are in every frame: an animal pose model (ViTPose++,
Apache 2.0, its animal expert trained on AP-10K) finds 17 points inside the box of the creature's silhouette. Its
left and right are often swapped in a side view; each leg's SAM 2 outline (parts.py) decides instead, and a point
that falls on no outline of its pair is dropped. The thin lower legs are where the point tracker and the outline say
least, and these points say the most.

Optional: needs the transformers package and downloads the model on first use; the tool works without it.
"""
import numpy as np

MODEL = 'usyd-community/vitpose-plus-base'
AP10K = 3                                      # ViTPose++ experts: 0 COCO, 1 AiC, 2 MPII, 3 AP-10K, 4 APT-36K
# AP-10K: 5 L shoulder, 6 L elbow, 7 L front paw, 8 R shoulder, 9 R elbow, 10 R front paw, 11 L hip, 12 L knee,
# 13 L back paw, 14 R hip, 15 R knee, 16 R back paw
LEGS = {('front', 'L'): (6, 7), ('front', 'R'): (9, 10), ('back', 'L'): (12, 13), ('back', 'R'): (15, 16)}


def detect(cr, clip, log=print):
    """Per view: points (T, 17, 2) and scores (T, 17), cached in work/<clip>/keypoints.npz; None without the model."""
    path = cr.project.work(clip) / 'keypoints.npz'
    views = cr.views(clip)
    if path.exists():
        d = np.load(path)
        return [(d[f'xy{k}'], d[f'score{k}']) for k in range(len(views))]
    try:
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        proc = AutoProcessor.from_pretrained(MODEL)
        from .reader.device import pick_device
        dev = pick_device()
        model = VitPoseForPoseEstimation.from_pretrained(MODEL).to(dev).eval()
    except Exception as e:                              # no transformers, no network for the first download
        log(f'keypoints: animal pose model unavailable ({type(e).__name__}), going on without it')
        return None
    import cv2
    T = len(cr.times(clip))
    out = []
    for v in views:
        P, S = np.zeros((T, 17, 2)), np.zeros((T, 17))
        for t in range(T):
            img = cv2.cvtColor(v['frame'](t), cv2.COLOR_BGR2RGB)
            ys, xs = np.nonzero(v['mask'](t))
            if not len(ys):
                continue
            x0, y0 = max(xs.min() - 12, 0), max(ys.min() - 12, 0)
            box = [float(x0), float(y0), float(min(xs.max() + 12, img.shape[1]) - x0),
                   float(min(ys.max() + 12, img.shape[0]) - y0)]
            inp = proc(img, boxes=[[box]], return_tensors='pt').to(dev)
            with torch.inference_mode():
                o = model(**inp, dataset_index=torch.tensor([AP10K], device=dev))
            r = proc.post_process_pose_estimation(o, boxes=[[box]])[0][0]
            P[t], S[t] = r['keypoints'].cpu().numpy(), r['scores'].cpu().numpy()
        out.append((P, S))
    np.savez_compressed(path, **{f'xy{k}': p for k, (p, _) in enumerate(out)},
                        **{f'score{k}': s for k, (_, s) in enumerate(out)})
    log(f'keypoints: elbows, knees and paws found by the animal pose model in {T} frames x {len(views)} views')
    return out


def leg_map(cr):
    """The four standing legs as {('front'|'back', 'L'|'R'): chain} from the skeleton's shape (front: its first bone
    ahead of the root along the facing; left: on the creature's left side), or None for other than four legs."""
    rig = cr.rig
    chains = [c for c in cr.limb_chains(as_list=True) if set(c) & cr.legs]
    if len(chains) != 4:
        return None
    Hd, _ = rig.bone_points(cr.p0(None))
    root = Hd[rig.root]
    left = np.cross(cr.up, cr.facing)                  # right-handed: up x forward points to the left side
    ahead = sorted(chains, key=lambda c: -(Hd[c[0]] - root) @ cr.facing)
    out = {}
    for end, pair in (('front', ahead[:2]), ('back', ahead[2:])):
        a, b = sorted(pair, key=lambda c: -(Hd[c[0]] - root) @ left)
        out[(end, 'L')], out[(end, 'R')] = a, b
    return out


def assigned(points, labels, legs, of, min_score=.4, near_px=4):
    """Per frame: [(AP-10K index, x, y, score)] of the elbow/knee and paw points, each pair's left and right taken
    from the legs' own outlines (the way more points fall inside), points on no outline of their pair dropped."""
    import cv2
    P, S = points
    T, H, W = labels.shape
    k = np.ones((2 * near_px + 1,) * 2, np.uint8)
    out, swaps = [], 0
    for t in range(T):
        frame = []
        for end in ('front', 'back'):
            oL, oR = of[legs[(end, 'L')][-1]], of[legs[(end, 'R')][-1]]
            mL = cv2.dilate((labels[t] == oL).astype(np.uint8), k).astype(bool)
            mR = cv2.dilate((labels[t] == oR).astype(np.uint8), k).astype(bool)
            iL, iR = LEGS[(end, 'L')], LEGS[(end, 'R')]

            def inside(i, m):
                x, y = int(P[t, i, 0]), int(P[t, i, 1])
                return S[t, i] >= min_score and 0 <= x < W and 0 <= y < H and bool(m[y, x])
            keep = sum(inside(i, mL) for i in iL) + sum(inside(i, mR) for i in iR)
            swap = sum(inside(i, mR) for i in iL) + sum(inside(i, mL) for i in iR)
            if swap > keep:
                swaps += 1
                iL, iR = iR, iL
            for want, src, m in ((LEGS[(end, 'L')], iL, mL), (LEGS[(end, 'R')], iR, mR)):
                for w_i, s_i in zip(want, src):
                    if inside(s_i, m):
                        frame.append((w_i, float(P[t, s_i, 0]), float(P[t, s_i, 1]), float(S[t, s_i])))
        out.append(frame)
    return out, swaps
