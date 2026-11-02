"""A second look at the limb ends (dev/PLAN-BETTER-INPUT.md, stage 3): the reader sees a whole panel at 448 px, where
a hand or a paw is a few dozen pixels and its joints shake most from frame to frame. A crop reader reads each limb end
again in a square cut round it from the same picture and enlarged to the reader's size, and its reading of that end's
keypoints replaces the first one where it is surer.

Both the training crops and the reading's crops are cut from the reader's own input (a panel letterboxed into
SIZE x SIZE, at the scale the training renders are drawn at), so the crop reader sees at reading time exactly the
kind of picture it learned on. A limb end: the last bone of a limb chain and the bones below it (a hand and its
tip, toes and their tip, a hoof); the crop's side: CROP_K x the length of the limb's last two bones in that panel's
pixels, between SIDE_MIN and SIDE_MAX input pixels.

    python -m video2rig.reader.train <project> --crops --init reader/reader.pt --out reader/reader-crops.pt
    reading: road.read_clip(..., crops=True) / read.read_frames(..., crop_model=...)
"""
import re

import numpy as np
import torch
import torch.nn.functional as F

from .net import SIZE

LOOSE = re.compile(r'(tail|(^|[^a-z])ear([._0-9]|$))', re.I)     # tails and ears: no limb ends

CROP_K = 1.6
SIDE_MIN, SIDE_MAX = 72.0, 320.0
JITTER = 4.0                  # input px: the crop's centre off the end's true place in training (the first reading's
                              # typical miss: 2 input px median, 6-8 at p90; the re-framing shifts it further)


def end_groups(cr, kps):
    """[(name, [keypoint indices], reach)] of every limb end: the keypoints on a limb chain's last bone and below it,
    and the length of the chain's last two bones (armature units). Tails and ears are not limbs here."""
    from . import side_of
    rig = cr.rig
    out = []
    for ch in cr.limb_chains(as_list=True):
        if side_of(rig.names[ch[0]]) is None or any(LOOSE.search(rig.names[b]) for b in ch):
            continue                                          # a neck and head, a tail: on the centre line
        last = ch[-1]
        ks = [k for k, kp in enumerate(kps) if not str(kp[1]).startswith('@') and
              (rig.names.index(kp[1]) == last or rig.desc[last, rig.names.index(kp[1])])]
        if not ks:
            continue
        reach = float(sum(rig.length[b] for b in ch[-2:]))
        out.append((rig.names[last], ks, reach))
    return out


def crop_sides(cams, arm_world, reach, sizes):
    """(V, G) crop sides in input pixels: CROP_K x each end's reach in that panel. cams: cameras.json views; sizes:
    each panel's letterboxed size in input pixels (w, h) (its image drawn at that size inside the reader's square)."""
    M = np.asarray(arm_world, float)
    unit = float(np.linalg.norm(M[:3, 0]))                    # armature units -> world units
    out = []
    for rec, (w, h) in zip(cams, sizes):
        px = h / float(rec.get('ortho_height') or rec['fov_height']) * unit   # input px per armature unit at the body
        out.append([float(np.clip(CROP_K * r * px, SIDE_MIN, SIDE_MAX)) for r in reach])
    return out


def cut(x, corners, sides):
    """Squares cut from pictures x (B, H, W, 3) uint8 on the card, corner (B, 2) and side (B,) in the pictures'
    pixels, each resized to SIZE x SIZE (bilinear; outside the picture its edge repeated): (B, SIZE, SIZE, 3) uint8."""
    B, H, W, _ = x.shape
    xf = x.permute(0, 3, 1, 2).float()
    ax = (torch.arange(SIZE, device=x.device, dtype=torch.float32) + .5) / SIZE      # 0..1 across the crop
    cx = corners[:, 0:1] + ax[None] * sides[:, None]                                   # (B, SIZE) picture px
    cy = corners[:, 1:2] + ax[None] * sides[:, None]
    gx = cx / W * 2 - 1
    gy = cy / H * 2 - 1
    grid = torch.stack([gx[:, None, :].expand(B, SIZE, SIZE), gy[:, :, None].expand(B, SIZE, SIZE)], -1)
    out = F.grid_sample(xf, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return out.round().clamp(0, 255).byte().permute(0, 2, 3, 1)


def training_crops(imgs, uv, inside, seen, views, groups, sides, rng, dev, every=False, chunk=64):
    """Crops of render pictures (imgs (N, SIZE, SIZE, 3) uint8, uv (N, K, 2) input px): per picture one limb end at
    random among those with a keypoint in the picture (every=True: all of them, for the held-out check), its square
    centred on the end's true keypoints plus JITTER. Returns crops (M, SIZE, SIZE, 3), their keypoints in crop
    pixels (M, K, 2), inside (M, K), seen (M, K), the picture (M,), the group (M,), zoom (M,) crop px per input px."""
    jobs = []
    for n in range(len(imgs)):
        ok = [g for g, (_, ks, _) in enumerate(groups) if (inside[n, ks] & seen[n, ks]).any()]
        if not ok:
            continue
        for g in (ok if every else [ok[int(rng.integers(len(ok)))]]):
            ks = groups[g][1]
            ins = inside[n, ks]
            centre = uv[n, ks][ins].mean(0) if ins.any() else uv[n, ks].mean(0)
            centre = centre + np.clip(rng.normal(0, JITTER, 2), -2.5 * JITTER, 2.5 * JITTER)
            side = sides[int(views[n])][g]
            jobs.append((n, g, centre - side / 2, side))
    M = len(jobs)
    crops = np.zeros((M, SIZE, SIZE, 3), np.uint8)
    cuv = np.zeros((M, uv.shape[1], 2), np.float32)
    for i in range(0, M, chunk):
        part = jobs[i:i + chunk]
        x = torch.from_numpy(imgs[[j[0] for j in part]]).to(dev)
        corners = torch.tensor(np.array([j[2] for j in part]), dtype=torch.float32, device=dev)
        sd = torch.tensor([j[3] for j in part], dtype=torch.float32, device=dev)
        crops[i:i + len(part)] = cut(x, corners, sd).cpu().numpy()
    for i, (n, g, corner, side) in enumerate(jobs):
        cuv[i] = (uv[n] - corner) * (SIZE / side)
    pic = np.array([j[0] for j in jobs])
    grp = np.array([j[1] for j in jobs])
    zoom = np.array([SIZE / j[3] for j in jobs], np.float32)
    cin = inside[pic] & (cuv[..., 0] >= 0) & (cuv[..., 0] < SIZE) & (cuv[..., 1] >= 0) & (cuv[..., 1] < SIZE)
    return crops, cuv, cin, seen[pic], pic, grp, zoom


@torch.no_grad()
def read_crops(read_fn, model, ck, canvases, p, c, view, dev, tta=0, tta_mode='heat', bs=16):
    """The limb ends read again: canvases (T, SIZE, SIZE, 3) uint8 (each panel's picture letterboxed as read), p
    (T, K, 2) input px and c (T, K) the first reading; per frame and end a square round the end's first reading
    (confidence-weighted), read by the crop reader (read_fn = read._read, with tta), its keypoints of that end
    taken where it is surer. Returns p, c and the share of the ends' readings it replaced."""
    groups, sides = ck['crops']['groups'], ck['crops']['sides'][view]
    T = len(canvases)
    jobs = []
    for t in range(T):
        for g, ks in enumerate(groups):
            w = np.clip(c[t, ks], .05, None)
            centre = (p[t, ks] * w[:, None]).sum(0) / w.sum()
            jobs.append((t, g, centre - sides[g] / 2, sides[g]))
    crops = []
    x_all = torch.from_numpy(canvases).to(dev)
    for i in range(0, len(jobs), 64):
        part = jobs[i:i + 64]
        corners = torch.tensor(np.array([j[2] for j in part]), dtype=torch.float32, device=dev)
        sd = torch.tensor([j[3] for j in part], dtype=torch.float32, device=dev)
        crops.append(cut(x_all[[j[0] for j in part]], corners, sd).cpu().numpy())
    crops = np.concatenate(crops) if crops else np.zeros((0, SIZE, SIZE, 3), np.uint8)
    q, cq, _ = read_fn(model, list(crops), dev, bs, tta=tta, tta_mode=tta_mode)
    p, c = p.copy(), c.copy()
    took = total = 0
    for i, (t, g, corner, side) in enumerate(jobs):
        ks = groups[g]
        new = corner + q[i, ks] * (side / SIZE)
        better = cq[i, ks] > c[t, ks]
        p[t, ks] = np.where(better[:, None], new, p[t, ks])
        c[t, ks] = np.where(better, cq[i, ks], c[t, ks])
        took += int(better.sum())
        total += len(ks)
    return p, c, took / max(total, 1)
