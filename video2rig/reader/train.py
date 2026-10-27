"""Train the joint reader of one creature on its own renders (blender/render_poses.py), within a time budget.

    python -m video2rig.reader.train <project> [--minutes 8] [--renders reader/renders] [--batch 24]
        -> <project>/reader/reader.pt (+ reader-train.json: the losses and the held-out error)

Every picture is "damaged" on the card the ways a Seedance frame differs from our render: re-framed (scale, turn,
shift), other colours and light, blur along a stroke, a fast limb smeared into a see-through streak (Seedance draws a
strike as a 30-100 px smear while the body stays sharp; a reader that never saw one put an alien's slashing arms out
wide and a centaur's sword the wrong way), lost resolution, noise, a shadow and a tinted floor behind the creature. Never mirrored: left and right are what the reader must tell apart. The last 5% of the poses are held out;
their error in pixels of the clip (seen joints) is the number that says whether the reader can be trusted. Joints
hidden in a picture (video2rig/reader/visible.py) get an empty target: the reader learns to say "not here".
"""
import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..common import Project, write_json
from ..creature import Creature
from . import end_pairs, is_offaxis, keypoints, points, project_ortho, render_keypoints
from .device import autocast, describe, interpolate_area, pick_device
from .net import SIZE, Reader, count_params, decode, gaussians, letterbox, normalise, sample_maps

HOLD_OUT = .05


def sight_depth(rec, arm_world, X, kps, height):
    """Each keypoint's depth along camera `rec`'s line of sight (away from it), from the middle of the body's
    keypoints (props left out of the middle), in the creature's heights: (..., K). Armature space, as the tool's
    cameras (video2rig.clip.Camera) take points; the camera's forward axis brought into it."""
    from . import is_prop
    Ma = np.asarray(arm_world, float)[:3, :3]
    n = Ma.T @ np.asarray(rec['forward'], float)
    n = n / np.linalg.norm(n)
    body = [k for k, kp in enumerate(kps) if not is_prop(kp)]
    mid = np.asarray(X)[..., body, :].mean(-2, keepdims=True)
    return (np.asarray(X) - mid) @ n / height


def render_file(renders, name):
    """A render set's own poses.npz / visible.npz (inside its folder: a second set next to the first), else the
    project's in reader/."""
    own = Path(renders) / name
    return own if own.exists() else Path(renders).parent / name


def load_items(pr, kps, renders, height=None, cr=None):
    """Every (strip, slot, view, keypoints in the slot's pixels, inside-the-panel mask, pose index); with the
    creature's height also each keypoint's depth along that panel's line of sight (sight_depth). cr: the creature,
    needed when a keypoint lies off its bone's axis (side_points: placed from the render's full pose)."""
    cams = json.loads((pr.start_dir / 'cameras.json').read_text(encoding='utf-8'))
    lab = np.load(renders / 'labels.npz')
    names = [str(n) for n in lab['names']]
    if any(is_offaxis(kp) for kp in kps):
        if cr is None:
            cfg = pr.clip_cfg('')
            cfg.setdefault('solve', {})
            cr = Creature(pr, cfg)
        X = render_keypoints(cr, kps, lab, render_file(renders, 'poses.npz'), cache=renders / 'params.npy')
    else:
        X = points(kps, names, lab['heads'], lab['tails'], lab['prop_mats'] if 'prop_mats' in lab.files else None,
                   lab['prop_names'] if 'prop_names' in lab.files else ())
    vpath = render_file(renders, 'visible.npz')                         # video2rig.reader.visible: hidden joints
    seen = np.load(vpath)['vis'] if vpath.exists() else None
    items = []
    for v, rec in enumerate(cams['views']):
        size = tuple(int(s) for s in lab['sizes'][v])
        per_clip_px = float(size[1] / lab['clip_sizes'][v][1])           # slot pixels per pixel of the clip
        uv = project_ortho(rec, lab['armature_world'], X, size)
        dd = sight_depth(rec, lab['armature_world'], X, kps, height) if height else None
        vis_v = seen[:, v].copy() if seen is not None else None
        for k1, k2 in end_pairs(kps):                       # a prop's ends named by place: the left one first
            sw = uv[:, k1, 0] > uv[:, k2, 0]
            uv[sw, k1], uv[sw, k2] = uv[sw, k2].copy(), uv[sw, k1].copy()
            if dd is not None:
                dd[sw, k1], dd[sw, k2] = dd[sw, k2].copy(), dd[sw, k1].copy()
            if vis_v is not None:
                rows = np.asarray(lab['index'])[sw]
                vis_v[rows, k1], vis_v[rows, k2] = vis_v[rows, k2].copy(), vis_v[rows, k1].copy()
        inside = (uv[..., 0] >= 0) & (uv[..., 0] < size[0]) & (uv[..., 1] >= 0) & (uv[..., 1] < size[1])
        for k, i in enumerate(lab['index']):
            b, c = (int(t) for t in lab['slot_of'][k])
            items.append(dict(strip=renders / rec['name'] / f'{b:04d}.jpg', x0=int(lab['x0'][v][c]), view=v,
                              size=size, per_clip_px=per_clip_px, uv=uv[k].astype(np.float32), inside=inside[k],
                              pose=int(i), seen=vis_v[int(i)] if seen is not None else np.ones(len(kps), bool),
                              d=dd[k].astype(np.float32) if dd is not None else None))
    return items


def load_pictures(items, workers=12):
    """Every slot letterboxed into SIZE x SIZE (uint8 BGR), its keypoints in input pixels, the backdrop's colour."""
    imgs = np.zeros((len(items), SIZE, SIZE, 3), np.uint8)
    uv = np.zeros((len(items), len(items[0]['uv']), 2), np.float32)
    bg = np.zeros((len(items), 3), np.float32)
    by_strip = {}
    for n, it in enumerate(items):
        by_strip.setdefault(it['strip'], []).append(n)

    def one(strip):
        big = cv2.imread(str(strip), cv2.IMREAD_COLOR)
        for n in by_strip[strip]:
            it = items[n]
            w0, h0 = it['size']
            img = big[:h0, it['x0']:it['x0'] + w0]
            s, (ox, oy) = letterbox(it['size'])
            w, h = int(round(w0 * s)), int(round(h0 * s))
            small = img if (w, h) == (w0, h0) else cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            back = np.median(np.concatenate([img[:8].reshape(-1, 3), img[:, :8].reshape(-1, 3)]), 0)
            imgs[n] = back.astype(np.uint8)
            x0, y0 = int(round(ox)), int(round(oy))
            imgs[n, y0:y0 + h, x0:x0 + w] = small
            # the keypoints follow the same map, the rounding of the placement included
            uv[n] = it['uv'] * np.array([w / w0, h / h0]) + np.array([x0, y0])
            bg[n] = back
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, list(by_strip)))
    return imgs, uv, bg


def blur_kernel(sigma, dev):
    r = max(1, int(math.ceil(2.5 * sigma)))
    ax = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    k = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def blur(x, sigma):
    k = blur_kernel(sigma, x.device)
    C = x.shape[1]
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), padding=(0, len(k) // 2), groups=C)
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), padding=(len(k) // 2, 0), groups=C)


def augment(img, uv, bg, rng, extra=(), maps=None):
    """img uint8 (B, H, W, 3) BGR, uv (B, K, 2) input px, bg (B, 3): a damaged copy, its keypoints (and every point set
    in `extra`, (B, K, 2) each, moved by the same re-framing: the neighbours' keypoints of a context reader). maps
    (B, C, h, w), any resolution over the same square (a skin map, dev/skin_map.py): moved by the same re-framing,
    nearest cell, and returned last."""
    B = img.shape[0]
    dev = img.device
    x = img.permute(0, 3, 1, 2).float() / 255.0
    bgc = (bg / 255.0).view(B, 3, 1, 1)
    body = ((x - bgc).abs().amax(1, keepdim=True) > .035).float()       # the creature (the backdrop is flat)
    # 1. re-framing: scale, turn, shift about the centre
    s = torch.exp(torch.empty(B, device=dev).uniform_(math.log(.85), math.log(1.15)))
    th = torch.empty(B, device=dev).uniform_(-math.radians(12), math.radians(12))
    t = torch.empty(B, 2, device=dev).uniform_(-.08, .08) * SIZE
    c, sn = torch.cos(th), torch.sin(th)
    R = torch.stack([torch.stack([c, -sn], -1), torch.stack([sn, c], -1)], -2)    # (B, 2, 2)
    Rt = R.transpose(1, 2) / s.view(B, 1, 1)
    tn = 2 * t / SIZE
    theta = torch.cat([Rt, -(Rt @ tn.unsqueeze(-1))], -1)
    grid = F.affine_grid(theta, (B, 3, SIZE, SIZE), align_corners=False)
    x = F.grid_sample(x - bgc, grid, mode='bilinear', padding_mode='zeros', align_corners=False) + bgc
    body = F.grid_sample(body, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    if maps is not None:
        gm = F.affine_grid(theta, (B, maps.shape[1]) + tuple(maps.shape[-2:]), align_corners=False)
        maps = F.grid_sample(maps, gm, mode='nearest', padding_mode='zeros', align_corners=False)
    centre = torch.tensor([SIZE / 2, SIZE / 2], device=dev)
    uv = ((uv - centre) @ R.transpose(1, 2)) * s.view(B, 1, 1) + centre + t.unsqueeze(1)
    extra = [((e - centre) @ R.transpose(1, 2)) * s.view(B, 1, 1) + centre + t.unsqueeze(1) for e in extra]
    # 2. the backdrop: a tint and a gradient, a soft shadow under the feet (behind the creature)
    yy = torch.linspace(-.5, .5, SIZE, device=dev).view(1, 1, SIZE, 1)
    grad = 1 + torch.empty(B, 1, 1, 1, device=dev).uniform_(-.18, .18) * yy
    tint = 1 + torch.empty(B, 3, 1, 1, device=dev).uniform_(-.03, .03)
    back = x * grad * tint
    shade = torch.ones(B, 1, SIZE, SIZE, device=dev)
    xs = torch.arange(SIZE, device=dev, dtype=torch.float32)
    for b in range(B):
        if rng.random() < .45:
            u = uv[b, :, 0].clamp(0, SIZE - 1)
            v = uv[b, :, 1].clamp(0, SIZE - 1)
            cx = float(u.mean()) + rng.uniform(-30, 30)
            cy = float(v.max()) + rng.uniform(-6, 10)
            rx = max(float(u.max() - u.min()) * rng.uniform(.35, .7), 20.0)
            ry = rng.uniform(.02, .06) * SIZE
            e = ((xs.view(1, -1) - cx) / rx) ** 2 + ((xs.view(-1, 1) - cy) / ry) ** 2
            shade[b, 0] = 1 - rng.uniform(.15, .45) * torch.exp(-e * 1.5)
    x = x * body + back * shade * (1 - body)
    # 3. colour and light
    gain = torch.empty(B, 3, 1, 1, device=dev).uniform_(.94, 1.06)
    bright = torch.empty(B, 1, 1, 1, device=dev).uniform_(.75, 1.25)
    x = x * gain * bright
    mean = x.mean((1, 2, 3), keepdim=True)
    x = (x - mean) * torch.empty(B, 1, 1, 1, device=dev).uniform_(.75, 1.3) + mean
    grey = x.mean(1, keepdim=True)
    x = (x - grey) * torch.empty(B, 1, 1, 1, device=dev).uniform_(.7, 1.25) + grey
    x = x.clamp(0, 1) ** torch.exp(torch.empty(B, 1, 1, 1, device=dev).uniform_(math.log(.8), math.log(1.25)))
    # 4. a fast limb (or a prop) smeared: the piece between two near keypoints, swept across itself, see-through
    x = smear_limbs(x, uv, rng)
    # 5. softness: blur, a stroke along a fast limb (the whole frame), lost resolution
    out = x.clone()
    for b in range(B):
        r = rng.random()
        xb = x[b:b + 1]
        if r < .35:
            xb = blur(xb, rng.uniform(.5, 1.6))
        elif r < .55:
            L = rng.integers(3, 8)
            ang = rng.uniform(0, math.pi)
            k = torch.zeros(L, L, device=dev)
            for i in range(L):
                px = int(round((L - 1) / 2 + (i - (L - 1) / 2) * math.cos(ang)))
                py = int(round((L - 1) / 2 + (i - (L - 1) / 2) * math.sin(ang)))
                k[py, px] = 1
            k = (k / k.sum()).view(1, 1, L, L).repeat(3, 1, 1, 1)
            xb = F.conv2d(F.pad(xb, (L // 2, (L - 1) // 2, L // 2, (L - 1) // 2), mode='replicate'), k, groups=3)
        if rng.random() < .3:
            f = rng.uniform(.5, .8)
            xb = F.interpolate(interpolate_area(xb, scale_factor=f), size=(SIZE, SIZE), mode='bilinear',
                               align_corners=False)
        out[b:b + 1] = xb
    x = out + torch.randn_like(out) * torch.empty(B, 1, 1, 1, device=dev).uniform_(0, .025)
    x = (x.clamp(0, 1) * 255).permute(0, 2, 3, 1)
    out = (x, uv, extra) if len(extra) else (x, uv)
    return out + (maps,) if maps is not None else out


def neighbour_uv(pr, cr, kps, items):
    """Per training picture, the keypoints one video frame before and after its pose (the poses render_poses.py blurred
    between), in input pixels as load_pictures maps the picture's own: (N, K, 2) twice. Static poses: the same."""
    from . import KeypointRig
    from .poses import velocities
    d = np.load(pr.root / 'reader' / 'poses.npz')
    params = d['params']
    if 'params_a' in d.files:
        pa, pb = d['params_a'], d['params_b']
    else:                                                               # as poses.py made them (its seed + 2)
        pa, pb = velocities(cr, params, np.random.default_rng(int(d['seed']) + 2), share=.55)
    cams = json.loads((pr.start_dir / 'cameras.json').read_text(encoding='utf-8'))
    lab = np.load(pr.root / 'reader' / 'renders' / 'labels.npz')
    kr = KeypointRig(cr, kps)
    Xa = {i: kr.points(pa[i]) for i in sorted({it['pose'] for it in items})}
    Xb = {i: kr.points(pb[i]) for i in sorted({it['pose'] for it in items})}
    pairs = end_pairs(kps)
    out_a = np.zeros((len(items), len(kps), 2), np.float32)
    out_b = np.zeros_like(out_a)
    for n, it in enumerate(items):
        rec = cams['views'][it['view']]
        w0, h0 = it['size']
        s, (ox, oy) = letterbox(it['size'])
        w, h = int(round(w0 * s)), int(round(h0 * s))
        for X, out in ((Xa[it['pose']], out_a), (Xb[it['pose']], out_b)):
            uv = project_ortho(rec, lab['armature_world'], X, it['size'])
            for k1, k2 in pairs:                                        # a prop's ends named by place, as the pose's
                if uv[k1, 0] > uv[k2, 0]:
                    uv[[k1, k2]] = uv[[k2, k1]]
            out[n] = uv * np.array([w / w0, h / h0]) + np.array([int(round(ox)), int(round(oy))])
    return out_a, out_b


def corrupt_context(ua, ub, twins, rng, dev):
    """The neighbours' keypoints made to look like a first reading: each a gaussian whose height is a confidence
    (.3-1, a sixth dropped), jittered, now and then far off, now and then a whole limb read as its twin, and in a fifth
    of the pictures no context at all (the first reading of a clip has none). (B, 2K, HEAT, HEAT)."""
    from .net import context_maps
    B, K, _ = ua.shape
    out = []
    for u in (ua, ub):
        u = u + torch.randn_like(u) * 1.5
        far = torch.rand(B, K, 1, device=dev) < .05
        u = torch.where(far, u + (torch.rand_like(u) - .5) * .4 * SIZE, u)
        c = torch.empty(B, K, device=dev).uniform_(.3, 1.0) * (torch.rand(B, K, device=dev) > .16)
        for b in range(B):
            if twins and rng.random() < .25:                          # a whole limb read as its twin
                limb = twins[int(rng.integers(len(twins)))]
                ia, ib = [l for l, _ in limb], [r for _, r in limb]
                u[b, ia + ib] = u[b, ib + ia].clone()
        out.append((u, c))
    none = torch.rand(B, device=dev) < .2
    ctx = context_maps(out[0][0], out[0][1], out[1][0], out[1][1])
    return ctx * (~none).float().view(B, 1, 1, 1)


def twin_limbs(cr, kps):
    """Lists of (left keypoint, right keypoint) per pair of twin limbs."""
    from . import side_of, twin_label
    rig = cr.rig
    labels = [k[0] for k in kps]
    chains = cr.limb_chains(as_list=True)
    chain_of = {j: ci for ci, ch in enumerate(chains) for j in ch}
    limbs = {}
    for k, lb in enumerate(labels):
        if side_of(lb) != 'L' or twin_label(lb) not in labels or str(kps[k][1]).startswith('@'):
            continue
        j = rig.names.index(kps[k][1])
        while j >= 0 and j not in chain_of:
            j = rig.parents[j]
        if j >= 0:
            limbs.setdefault(chain_of[j], []).append((k, labels.index(twin_label(lb))))
    return list(limbs.values())


def smear_limbs(x, uv, rng, p=.35):
    """x (B, 3, S, S) 0..1, uv (B, K, 2): in about p of the pictures, one or two limb pieces smeared across their
    length by 15-70% of the creature's height, as a fast strike shows in a 24 fps video: a capsule round the piece between a random keypoint and
    its nearest neighbour, shifted along the sweep and averaged (premultiplied), over the picture. The keypoints stay
    put: the streak is centred on the limb's real place."""
    B, _, S, _ = x.shape
    dev = x.device
    yy, xx = torch.meshgrid(torch.arange(S, device=dev, dtype=torch.float32),
                            torch.arange(S, device=dev, dtype=torch.float32), indexing='ij')
    out = x.clone()
    for b in range(B):
        if rng.random() >= p:
            continue
        for _ in range(int(rng.integers(1, 3))):
            k = int(rng.integers(0, uv.shape[1]))
            a = uv[b, k]
            d = torch.linalg.norm(uv[b] - a, dim=1)
            d[k] = 1e9
            j = int(torch.argmin(d))
            if float(d[j]) > .3 * S:
                continue
            c = uv[b, j]
            seg = a - c
            L2 = float(seg @ seg) + 1e-6
            tpar = (((xx - c[0]) * seg[0] + (yy - c[1]) * seg[1]) / L2).clamp(0, 1)
            dist = torch.sqrt((xx - (c[0] + tpar * seg[0])) ** 2 + (yy - (c[1] + tpar * seg[1])) ** 2)
            r = rng.uniform(.025, .045) * S
            mask = (1 - ((dist - r) / (.5 * r)).clamp(0, 1)).view(1, 1, S, S)
            # sweep across the piece, give or take 35 degrees
            ang = math.atan2(float(seg[1]), float(seg[0])) + math.pi / 2 + rng.uniform(-.6, .6)
            tall = float(uv[b, :, 1].max() - uv[b, :, 1].min())     # the creature's height in the picture
            length = rng.uniform(.15, .7) * max(tall, .2 * S)
            n = 13
            limb = out[b:b + 1] * mask
            acc, acc_a = torch.zeros_like(limb), torch.zeros_like(mask)
            for i in range(n):
                f = (i / (n - 1) - .5) * length
                dx, dy = int(round(f * math.cos(ang))), int(round(f * math.sin(ang)))
                acc += torch.roll(limb, shifts=(dy, dx), dims=(2, 3))
                acc_a += torch.roll(mask, shifts=(dy, dx), dims=(2, 3))
            acc, acc_a = acc / n, acc_a / n
            out[b:b + 1] = out[b:b + 1] * (1 - acc_a) + acc
    return out


def batches(n, bs, rng):
    while True:
        order = rng.permutation(n)
        for i in range(0, n - bs + 1, bs):
            yield order[i:i + bs]


@torch.no_grad()
def evaluate(model, imgs, uv, valid, scale, views, dev, bs=32, ctx=None):
    """Median and 90th percentile error per view, in pixels of the clip's panel (input px / letterbox scale). ctx: the
    neighbours' keypoints (ua, ub), given to a context reader as a clean first reading."""
    from .net import context_maps
    model.eval()
    errs = {v: [] for v in set(views.tolist())}
    for i in range(0, len(imgs), bs):
        x = normalise(torch.from_numpy(imgs[i:i + bs]).to(dev))
        c = None
        if ctx is not None:
            ua, ub = (torch.from_numpy(z[i:i + bs]).to(dev) for z in ctx)
            ones = torch.ones(ua.shape[:2], device=dev)
            c = context_maps(ua, ones, ub, ones)
        with autocast(dev):
            out = model(x, c) if model.context else model(x)
        heat = out[0] if isinstance(out, tuple) else out                  # a depth reader gives its depth maps too
        p, _ = decode(heat.float())
        e = (p.cpu().numpy() - uv[i:i + bs]) / scale[i:i + bs, None, None]
        d = np.linalg.norm(e, axis=-1)
        for k in range(len(d)):
            errs[int(views[i + k])].append(d[k][valid[i + k]])
    model.train()
    out = {}
    for v, e in errs.items():
        e = np.concatenate(e) if e else np.zeros(0)
        out[v] = dict(median_px=round(float(np.median(e)), 2), p90_px=round(float(np.percentile(e, 90)), 2), n=len(e))
    return out


def twin_pairs(kps):
    """(left, right) keypoint indices of the same joint on both sides."""
    from . import side_of, twin_label
    labels = [k[0] for k in kps]
    return [(k, labels.index(twin_label(lb))) for k, lb in enumerate(labels)
            if side_of(lb) == 'L' and twin_label(lb) in labels]


@torch.no_grad()
def evaluate_skin(model, imgs, maps, dev, bs=32):
    """The skin map on held-out pictures: how well the creature's cells are found (overlap) and how far the place
    named misses the true one (in the creature's heights, on its true cells)."""
    model.eval()
    ious, miss = [], []
    for i in range(0, len(imgs), bs):
        x = normalise(torch.from_numpy(imgs[i:i + bs]).to(dev))
        with autocast(dev):
            smap = model(x)[2].float()
        mt = torch.from_numpy(maps[i:i + bs].astype(np.float32)).to(dev)
        pm, tm = torch.sigmoid(smap[:, 3]) > .5, mt[..., 3] > .5
        ious += ((pm & tm).sum((1, 2)).float() / (pm | tm).sum((1, 2)).clamp(min=1).float()).tolist()
        miss += (smap[:, :3].permute(0, 2, 3, 1) - mt[..., :3]).norm(dim=-1)[tm].tolist()
    model.train()
    return dict(mask_overlap=round(float(np.median(ious)), 3), miss_pct=round(float(np.median(miss)) * 100, 2),
                miss_p90_pct=round(float(np.percentile(miss, 90)) * 100, 2))


@torch.no_grad()
def evaluate_depth(model, imgs, uv, dd, inside, seen, views, kps, height, dev, bs=32, gap=.02):
    """The depth read where each keypoint truly is, on the held-out pictures, per view: the median miss in % of the
    creature's height (seen and hidden keypoints apart), and for twin joints (a left and a right one) further apart in
    depth than `gap` heights, how often the reading puts them in the right order (which one is nearer)."""
    model.eval()
    pred = np.zeros_like(dd)
    for i in range(0, len(imgs), bs):
        x = normalise(torch.from_numpy(imgs[i:i + bs]).to(dev))
        with autocast(dev):
            dmap = model(x)[1]
        pred[i:i + bs] = sample_maps(dmap.float(), torch.from_numpy(uv[i:i + bs]).to(dev)).cpu().numpy()
    model.train()
    pairs = twin_pairs(kps)
    out = {}
    for v in sorted(set(views.tolist())):
        sel = views == v
        e = np.abs(pred[sel] - dd[sel]) * 100
        ins, sn = inside[sel], seen[sel]
        right = n = 0
        for l, r in pairs:
            g_true = dd[sel][:, l] - dd[sel][:, r]
            g_pred = pred[sel][:, l] - pred[sel][:, r]
            use = (np.abs(g_true) > gap) & ins[:, l] & ins[:, r]
            n += int(use.sum())
            right += int((np.sign(g_pred[use]) == np.sign(g_true[use])).sum())
        out[int(v)] = dict(seen_pct=round(float(np.median(e[ins & sn])), 2) if (ins & sn).any() else None,
                           hidden_pct=round(float(np.median(e[ins & ~sn])), 2) if (ins & ~sn).any() else None,
                           twin_order_right=round(right / max(n, 1), 3), twin_pairs=n,
                           height_m=round(float(height), 3))
    return out


def main():
    ap = argparse.ArgumentParser(prog='video2rig.reader.train')
    ap.add_argument('project')
    ap.add_argument('--renders', default='reader/renders')
    ap.add_argument('--minutes', type=float, default=8.0)
    ap.add_argument('--steps', type=int, default=0,
                    help='train this many steps whatever the time (a card stuck hot at 0.2 GHz would else end the '
                         'clock with a tenth of them)')
    ap.add_argument('--init', default='', help="another creature's reader.pt to start from (all but the keypoint "
                                                 "head): it already knows what a joint looks like in our renders")
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--out', default='reader/reader.pt')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--context', action='store_true', help='a reader that also takes its own first reading of the '
                    'frame before and after (net.Reader context); best started from that creature\'s reader (--init)')
    ap.add_argument('--depth', action='store_true', help='a reader that also reads each keypoint\'s depth along the '
                    'panel\'s line of sight (stage R: one camera); best started from that creature\'s reader (--init)')
    ap.add_argument('--depth-weight', type=float, default=20.0, help='the depth error (in heights) against the heat '
                    'maps\' loss')
    ap.add_argument('--skin', action='store_true', help='a depth reader that also reads the skin map (which place of '
                    'the rest-pose model every cell shows; labels: dev/skin_map.py labels); best started from that '
                    'creature\'s depth reader (--init)')
    ap.add_argument('--skin-weight', type=float, default=1.0, help='the skin map\'s loss against the heat maps\'')
    ap.add_argument('--skin-xyz', type=float, default=5.0, help='the place\'s miss (in heights) against the mask\'s loss')
    ap.add_argument('--real', default='', help='real Seedance frames with labels taken from the fit (an .npz of '
                    'pictures letterboxed like a reading, their keypoints and which keypoints count; '
                    'dev/bench/pseudo_labels.py), mixed into every batch')
    ap.add_argument('--real-share', type=float, default=.5, help='the share of every batch taken from --real')
    ap.add_argument('--eval', action='store_true', help='no training: the held-out error of --init, as a training run '
                    'reports it')
    ap.add_argument('--crops', action='store_true', help='a crop reader: limb ends in squares cut round them and '
                    'enlarged (crops.py); start it from the same creature\'s reader (--init)')
    a = ap.parse_args()
    t0 = time.time()
    log = lambda s: print(f'[{time.time() - t0:6.1f}s] {s}', flush=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = pick_device()
    log(f'device: {describe(dev)}')
    pr = Project(a.project)
    cfg = pr.clip_cfg('')
    cfg.setdefault('solve', {})
    cr = Creature(pr, cfg)
    kps = keypoints(cr)
    items = load_items(pr, kps, pr.root / a.renders, height=cr.height if a.depth else None, cr=cr)
    imgs, uv, bg = load_pictures(items)
    dd_all = np.stack([it['d'] for it in items]).astype(np.float32) if a.depth else None
    views = np.array([it['view'] for it in items])
    inside = np.stack([it['inside'] for it in items])
    seen = np.stack([it['seen'] for it in items])
    scale = np.array([letterbox(it['size'])[0] * it['per_clip_px'] for it in items], np.float32)   # input px per clip px
    poses = np.array([it['pose'] for it in items])
    cut = np.quantile(poses, 1 - HOLD_OUT)
    val = poses > cut
    tr = ~val
    log(f'{len(items)} pictures ({int(tr.sum())} to learn, {int(val.sum())} held out), {len(kps)} keypoints, '
        f'{imgs.nbytes / 1e9:.1f} GB; seen {seen.mean():.2f} of the keypoints')
    skin_maps, skin_ck = None, {}
    if a.skin:
        # the skin map's labels (dev/skin_map.py labels): one 4-channel map per training picture, in its order
        if not a.depth or a.context or a.crops or a.real:
            raise SystemExit('--skin goes with --depth (no --context, --crops or --real)')
        sk = np.load(pr.root / 'reader' / 'skin_labels.npz')
        if len(sk['maps']) != len(items):
            raise SystemExit(f'skin_labels.npz: {len(sk["maps"])} maps for {len(items)} pictures (make them again)')
        skin_maps = sk['maps']
        skin_ck = {'skin': True, 'skin_mid': sk['mid'], 'skin_height': float(sk['height'])}
    model = Reader(len(kps), context=2 * len(kps) if a.context else 0, depth=a.depth, skin=a.skin).to(dev)
    log(f'reader: {count_params(model):.1f} M parameters')
    if a.init:
        ck = torch.load(a.init, map_location='cpu', weights_only=False)
        own = model.state_dict()
        same = [tuple(k) for k in ck.get('keypoints', [])] == [tuple(k) for k in kps]      # this creature's own reader
        take = {k: v for k, v in ck['state'].items()
                if k in own and own[k].shape == v.shape and (same or not k.startswith('head.'))}
        w_src = ck['state'].get('mix.0.0.weight')
        if a.context and w_src is not None and w_src.shape[1] < own['mix.0.0.weight'].shape[1]:
            # the reader without context: its mixing layer copied, the context channels starting at zero, so the new
            # reader starts as the old one and learns what the context is worth
            w = torch.zeros_like(own['mix.0.0.weight'])
            w[:, :w_src.shape[1]] = w_src
            take['mix.0.0.weight'] = w
        model.load_state_dict(take, strict=False)
        log(f'warm start from {a.init}: {len(take)} of {len(own)} tensors'
            + ('' if same else ' (the keypoint head starts new)'))
    model = model.to(memory_format=torch.channels_last)
    names = json.loads((pr.start_dir / 'cameras.json').read_text(encoding='utf-8'))['views']
    ev_valid = inside & seen
    extra_ck, extra_report = {}, {}
    if a.crops:
        # stage 3: the reader learns limb ends in squares cut round them from the same pictures (crops.py)
        if a.context or a.depth or a.real:
            raise SystemExit('--crops is for a plain reader (no --context, --depth or --real)')
        from .crops import CROP_K, JITTER, crop_sides, end_groups, training_crops
        groups = end_groups(cr, kps)
        lab = np.load(pr.root / a.renders / 'labels.npz')
        sizes_in = []
        for v in range(len(names)):
            w0, h0 = next(it['size'] for it in items if it['view'] == v)
            s_ = letterbox((w0, h0))[0]
            sizes_in.append((w0 * s_, h0 * s_))
        sides = crop_sides(names, lab['armature_world'], [g[2] for g in groups], sizes_in)
        ends = np.zeros(len(kps), bool)
        for _, ks, _ in groups:
            ends[ks] = True
        ev0 = evaluate(model, imgs[val], uv[val], inside[val] & seen[val] & ends[None], scale[val], views[val], dev)
        extra_report['full_reader_ends'] = {names[v]['name']: e for v, e in ev0.items()}
        log(f'limb ends {[g[0] for g in groups]}, crop sides (input px) '
            f'{[[round(x) for x in sv] for sv in sides]}; the reader it starts from, on them: '
            f'{extra_report["full_reader_ends"]}')
        crng = np.random.default_rng(a.seed + 7)
        tr_i, va_i = np.nonzero(tr)[0], np.nonzero(val)[0]
        A = training_crops(imgs[tr_i], uv[tr_i], inside[tr_i], seen[tr_i], views[tr_i], groups, sides, crng, dev)
        B = training_crops(imgs[va_i], uv[va_i], inside[va_i], seen[va_i], views[va_i], groups, sides, crng, dev,
                           every=True)
        pic = np.concatenate([tr_i[A[4]], va_i[B[4]]])
        own = np.zeros((len(B[0]), len(kps)), bool)                  # a held-out crop is scored on its own end
        for i, g in enumerate(B[5]):
            own[i, groups[g][1]] = True
        imgs = np.concatenate([A[0], B[0]])
        uv = np.concatenate([A[1], B[1]])
        inside = np.concatenate([A[2], B[2]])
        seen = np.concatenate([A[3], B[3]])
        bg, views = bg[pic], views[pic]
        scale = scale[pic] * np.concatenate([A[6], B[6]])
        tr = np.r_[np.ones(len(A[0]), bool), np.zeros(len(B[0]), bool)]
        val = ~tr
        ev_valid = inside & seen
        ev_valid[val] &= own
        extra_ck['crops'] = dict(groups=[ks for _, ks, _ in groups], names=[n for n, _, _ in groups], sides=sides,
                                 k=CROP_K, jitter=JITTER)
        log(f'crops: {len(A[0])} to learn, {len(B[0])} held out ({imgs.nbytes / 1e9:.1f} GB)')
    if a.eval:
        ev = evaluate(model, imgs[val], uv[val], ev_valid[val], scale[val], views[val], dev)
        report = {names[v]['name']: e for v, e in ev.items()}
        log(f'held out, {a.init}: {report}')
        write_json(Path(a.init).with_name(Path(a.init).stem + '-heldout.json'), {'held_out': report,
                                                                                'renders': a.renders})
        return
    real, n_real = None, 0
    if a.real:
        # real frames labelled by the fit (dev/bench/pseudo_labels.py): a keypoint counts only where `keep` says so
        # (seen, and the fit and the reading agree); the rest neither peaks nor is taught "not here"
        if a.context or a.depth:
            raise SystemExit('--real is for a plain reader (no --context, no --depth)')
        rd = np.load(a.real)
        real = dict(imgs=rd['imgs'], uv=rd['uv'].astype(np.float32), keep=rd['keep'].astype(bool),
                    bg=rd['bg'].astype(np.float32))
        if real['uv'].shape[1] != len(kps):
            raise SystemExit(f'{a.real}: {real["uv"].shape[1]} keypoints, the reader has {len(kps)}')
        n_real = max(1, min(a.batch - 1, int(round(a.batch * a.real_share))))
        it_real = batches(len(real['imgs']), n_real, rng)
        log(f'real frames: {len(real["imgs"])} pictures from {a.real}, {real["keep"].mean():.2f} of their keypoints '
            f'labelled; {n_real} of every {a.batch} in a batch')
    vit = [p for p in model.vit.parameters() if p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith('vit.')]
    opt = torch.optim.AdamW([{'params': vit, 'lr': a.lr * .1}, {'params': rest, 'lr': a.lr}], weight_decay=1e-2)
    base = [g['lr'] for g in opt.param_groups]
    tri = np.nonzero(tr)[0]
    it_tr = batches(len(tri), a.batch - n_real, rng)
    if a.context:
        ua_all, ub_all = neighbour_uv(pr, cr, kps, items)
        twins = twin_limbs(cr, kps)
        log(f'context: the keypoints a frame before and after every picture; {len(twins)} pairs of twin limbs')
    budget = a.minutes * 60 - (time.time() - t0) - 20     # 20 s kept for the held-out check and saving
    tstart = time.time()
    warm, step = 100, 0
    hist = []
    model.train()
    # the schedule runs on the clock, not on a step count: a warm laptop card slows down (it fell from 3 GHz to
    # 0.5 GHz under load) and the training still ends on time, with fewer steps
    while True:
        frac = step / a.steps if a.steps else (time.time() - tstart) / budget
        if frac >= 1.0:
            break
        sel = tri[next(it_tr)]
        if real is None:
            x_u8 = torch.from_numpy(imgs[sel]).to(dev, non_blocking=True)
            u = torch.from_numpy(uv[sel]).to(dev)
            b = torch.from_numpy(bg[sel]).to(dev)
            ins = torch.from_numpy(inside[sel]).to(dev)
            pos = torch.from_numpy(seen[sel]).to(dev)
            keep = torch.ones(pos.shape, device=dev)
        else:
            rs = next(it_real)
            x_u8 = torch.from_numpy(np.concatenate([imgs[sel], real['imgs'][rs]])).to(dev, non_blocking=True)
            u = torch.from_numpy(np.concatenate([uv[sel], real['uv'][rs]])).to(dev)
            b = torch.from_numpy(np.concatenate([bg[sel], real['bg'][rs]])).to(dev)
            ins = torch.from_numpy(np.concatenate([inside[sel], real['keep'][rs]])).to(dev)
            pos = torch.from_numpy(np.concatenate([seen[sel], real['keep'][rs]])).to(dev)
            keep = torch.from_numpy(np.concatenate([np.ones(inside[sel].shape, bool), real['keep'][rs]])).to(dev).float()
        ctx = None
        smt = None
        if a.context:
            ua, ub = torch.from_numpy(ua_all[sel]).to(dev), torch.from_numpy(ub_all[sel]).to(dev)
            x, u, (ua, ub) = augment(x_u8, u, b, rng, extra=(ua, ub))
            ctx = corrupt_context(ua, ub, twins, rng, dev)
        elif skin_maps is not None:
            smt = torch.from_numpy(skin_maps[sel].astype(np.float32)).permute(0, 3, 1, 2).to(dev)
            x, u, smt = augment(x_u8, u, b, rng, maps=smt)
        else:
            x, u = augment(x_u8, u, b, rng)
        ok = ins & (u[..., 0] >= 0) & (u[..., 0] < SIZE) & (u[..., 1] >= 0) & (u[..., 1] < SIZE)
        # a hidden joint's map is taught to stay empty (a low peak = "not seen here"), a seen one to peak on it
        target = gaussians(u, (ok & pos).float(), sigma=2.0)
        lr_k = min((step + 1) / warm, 1.0) * .5 * (1 + math.cos(math.pi * frac))
        for g, l0 in zip(opt.param_groups, base):
            g['lr'] = l0 * lr_k
        with autocast(dev):
            out = model(normalise(x).contiguous(memory_format=torch.channels_last), ctx)
        heat, dmap = (out[0], out[1]) if a.depth else (out, None)
        # the few cells round each joint weigh ten times the empty rest (else the maps learn "nowhere" first)
        weight = 1 + 9 * (target > .05).float()
        loss = ((weight * (heat.float() - target) ** 2).mean((2, 3)) * keep).sum(1).mean() * 100
        loss_d = None
        if dmap is not None:
            # every keypoint in the picture, hidden ones too (a far leg behind the near one has a depth all the same),
            # read where it truly is; the re-framing turns and scales the picture, which leaves depth in heights alone
            m = ok.float()
            err = (sample_maps(dmap.float(), u) - torch.from_numpy(dd_all[sel]).to(dev)).abs()
            loss_d = (err * m).sum() / m.sum().clamp(min=1)
            loss = loss + a.depth_weight * loss_d
        loss_s = None
        if smt is not None:
            # the skin map: whether a cell is the creature, and where on the rest-pose model it is
            smap = out[2].float()
            mk = smt[:, 3]
            loss_s = (F.binary_cross_entropy_with_logits(smap[:, 3], mk)
                      + a.skin_xyz * ((smap[:, :3] - smt[:, :3]).abs().sum(1) * mk).sum() / mk.sum().clamp(min=1))
            loss = loss + a.skin_weight * loss_s
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            el = time.time() - tstart
            hist.append(dict(step=step, loss=round(float(loss), 4), s=round(el, 1),
                             **({'depth': round(float(loss_d), 4)} if loss_d is not None else {}),
                             **({'skin': round(float(loss_s), 4)} if loss_s is not None else {})))
            log(f'step {step} loss {float(loss):.4f}' + (f' (depth {float(loss_d):.4f} heights)' if loss_d is not None
                                                         else '')
                + (f' (skin {float(loss_s):.4f})' if loss_s is not None else '')
                + f'  ({el / step:.3f} s per step, {100 * frac:.0f}% of the time)')
    out = pr.root / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    # the weights first: a check that fails afterwards must not throw the training away
    extra_ck.update(skin_ck)
    torch.save({'state': model.state_dict(), 'keypoints': kps, 'size': SIZE, 'context': model.context,
                'depth': model.depth, 'renders': a.renders, 'steps': step, **extra_ck}, out)
    if skin_maps is not None:
        extra_ck['held_out_skin'] = evaluate_skin(model, imgs[val], skin_maps[val], dev)
        log(f'held out, skin: {extra_ck["held_out_skin"]}')
    ev = evaluate(model, imgs[val], uv[val], ev_valid[val], scale[val], views[val], dev,
                  ctx=(ua_all[val], ub_all[val]) if a.context else None)
    report = {names[v]['name']: e for v, e in ev.items()}
    log(f'held out: {report}')
    off = np.array([is_offaxis(kp) for kp in kps])
    if off.any() and not a.crops:
        ev_off = evaluate(model, imgs[val], uv[val], ev_valid[val] & off[None], scale[val], views[val], dev,
                          ctx=(ua_all[val], ub_all[val]) if a.context else None)
        extra_report['held_out_side_points'] = {names[v]['name']: e for v, e in ev_off.items()}
        log(f'held out, the {int(off.sum())} points off the bones (claws, antlers): '
            f'{extra_report["held_out_side_points"]}')
    depth_report = None
    if a.depth:
        depth_report = {names[v]['name']: e for v, e in evaluate_depth(
            model, imgs[val], uv[val], dd_all[val], inside[val], seen[val], views[val], kps, cr.height, dev).items()}
        log(f'held out, depth: {depth_report}')
    torch.save({'state': model.state_dict(), 'keypoints': kps, 'size': SIZE, 'held_out': report,
                'context': model.context, 'depth': model.depth, 'held_out_depth': depth_report,
                'renders': a.renders, 'steps': step, **extra_ck}, out)
    write_json(out.with_name(out.stem + '-train.json'), {'held_out': report, 'held_out_depth': depth_report,
                                                         **extra_report, 'real': a.real or None,
                                                         'steps': step, 'batch': a.batch,
                                                         'minutes': round((time.time() - t0) / 60, 2),
                                                         'pictures': len(items), 'history': hist})
    log(f'saved {out}')


if __name__ == '__main__':
    main()
