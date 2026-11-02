"""The dense reader: for every cell of the picture, which bone's skin shows there and where on that bone (stage B of
dev/PLAN-DENSE.md). One camera cannot measure depth, and 31 points on the bones' lines say nothing about a bone's
turn about itself; every pixel of the creature is some place of our own skin, and that place, read for thousands of
pixels a frame, pins the pose down (the oracle on the Leshy's exam: 12.7 -> 3.5 degrees a bone).

The map is read the way DensePose reads people: a class per cell (the bone whose skin shows, or the background) and,
for the bone named, the place on it in the bone's own rest frame (three numbers in the creature's heights). A cell is
one leg or the other, never a blur between the two. The reader sees the creature through a square window round the
body, cut from pictures rendered at twice the usual size (render_poses.py --fit 896): about 1.8 times as large in
its input as a whole letterboxed panel shows it. The keypoints (with the side points) and the depth are trained
along, so one reader reads everything from the window.

    python -m video2rig.reader.dense labels <project> --renders reader/renders-dense
        -> <renders>/dense_labels.npz (the windows and the maps), dense_labels_check.jpg
    python -m video2rig.reader.dense train <project> --renders reader/renders-dense --init reader/reader-pts.pt
        [--steps 8000] [--out reader/reader-dense.pt]
    python -m video2rig.reader.dense read <project> <clip> [--reader reader/reader-dense.pt]
        [--locator reader/reader-pts.pt] [--tag x]
        -> work/<clip>/dense<tag>/reading.npz (uv, conf, dep), dense.npz (the cells), sheet.jpg
    python -m video2rig.reader.dense fit <project> <clip> [--read dense<tag>] [--init fx_direct] [--tag x]
        -> work/<clip>/fx_dense<tag>/params.npy, motion.npz (the fit as fitted); a bench clip (gt_*) is scored
           against its truth
"""
import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ..common import Project, write_json
from ..creature import Creature
from . import KeypointRig, is_offaxis, is_prop, keypoints, project_ortho
from .net import HEAT, SIZE, STRIDE, Reader, decode, letterbox, normalise

WIN_K = 1.12         # the window's side: this x the 95th percentile of the training poses' extent in the picture
JITTER = .07         # the window's centre off the body's in training, share of the side (normal)
BG = 255             # the background in the stored part maps


# ----- the windows ----------------------------------------------------------------------------------------------
def window_side(items):
    """The side of the square window round the body, in the render slot's pixels: WIN_K times the 95th percentile of
    the training pictures' keypoint extent (the larger of width and height), kept inside the slot."""
    ext = []
    for it in items:
        q = it['uv'][it['inside']]
        if len(q) >= 3:
            ext.append(max(float(np.ptp(q[:, 0])), float(np.ptp(q[:, 1]))))
    side = float(np.percentile(ext, 95)) * WIN_K
    w0, h0 = items[0]['size']
    return int(round(min(side, w0, h0)))


def window_rects(items, side, rng):
    """Per picture the window's corner (slot pixels, whole): centred on the body's keypoints, moved a little at
    random, kept inside the slot."""
    out = np.zeros((len(items), 2), np.int32)
    for n, it in enumerate(items):
        w0, h0 = it['size']
        q = it['uv'][it['inside']]
        c = q.mean(0) if len(q) else np.array([w0 / 2, h0 / 2])
        c = c + rng.normal(0, JITTER * side, 2)
        x = int(round(np.clip(c[0] - side / 2, 0, w0 - side)))
        y = int(round(np.clip(c[1] - side / 2, 0, h0 - side)))
        out[n] = (x, y)
    return out


def load_windows(items, rects, side, workers=12):
    """Every picture's window, resized to SIZE x SIZE (uint8 BGR), its keypoints in input pixels, the backdrop's
    colour."""
    imgs = np.zeros((len(items), SIZE, SIZE, 3), np.uint8)
    uv = np.zeros((len(items), len(items[0]['uv']), 2), np.float32)
    bg = np.zeros((len(items), 3), np.float32)
    z = SIZE / side
    by_strip = {}
    for n, it in enumerate(items):
        by_strip.setdefault(it['strip'], []).append(n)

    def one(strip):
        big = cv2.imread(str(strip), cv2.IMREAD_COLOR)
        for n in by_strip[strip]:
            it = items[n]
            w0, h0 = it['size']
            slot = big[:h0, it['x0']:it['x0'] + w0]
            x, y = rects[n]
            win = slot[y:y + side, x:x + side]
            imgs[n] = cv2.resize(win, (SIZE, SIZE), interpolation=cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR)
            uv[n] = (it['uv'] - np.array([x, y])) * z
            bg[n] = np.median(np.concatenate([slot[:8].reshape(-1, 3), slot[:, :8].reshape(-1, 3)]), 0)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, list(by_strip)))
    return imgs, uv, bg


def part_colours(nb):
    """A fixed colour per bone (BGR), spread round the hue wheel, twins apart."""
    out = np.zeros((nb + 1, 3), np.uint8)
    for j in range(nb):
        h = int(180 * ((j * 7) % nb) / nb)
        out[j] = cv2.cvtColor(np.uint8([[[h, 200, 230]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return out


# ----- labels ---------------------------------------------------------------------------------------------------
def labels(a):
    from ..raster import rasterize
    from .train import load_items
    from .visible import triangle_bones
    t0 = time.time()
    pr = Project(a.project)
    cfg = pr.clip_cfg('')
    cfg.setdefault('solve', {})
    cr = Creature(pr, cfg)
    rig = cr.rig
    kps = keypoints(cr)
    renders = pr.root / a.renders
    items = load_items(pr, kps, renders, height=cr.height, cr=cr)
    params = np.load(renders / 'params.npy')                       # render_keypoints cached the renders' full poses
    lab = np.load(renders / 'labels.npz')
    assert len(params) == len(lab['index']), (len(params), len(lab['index']))
    row = {int(i): k for k, i in enumerate(lab['index'])}             # a pose's row (every panel's picture shares it)
    cams = json.loads((pr.start_dir / 'cameras.json').read_text(encoding='utf-8'))
    recs = cams['views']                                              # one camera, or front and side
    M = np.asarray(lab['armature_world'], float)
    side = window_side(items)
    rects = window_rects(items, side, np.random.default_rng(a.seed))
    z = SIZE / side
    coords = np.asarray(cr.mesh['coords'], float)
    tris = np.asarray(cr.mesh['tris'], np.int64)
    tri_bone = triangle_bones(cr)
    nb = rig.nb
    H = cr.height
    Rt = rig.rest_rot.transpose(0, 2, 1)                              # bone frame <- armature (rest)
    N = len(items)
    parts = np.full((N, HEAT, HEAT), BG, np.uint8)
    local = np.zeros((N, HEAT, HEAT, 3), np.float16)
    size = items[0]['size']
    print(f'DENSE labels: {N} pictures, window {side} px of a {size[0]}x{size[1]} slot (x{z:.2f} into {SIZE})',
          flush=True)
    for n, it in enumerate(items):
        X = np.asarray(rig.skin(params[row[it['pose']]], coords, cr.idx, cr.wts), float)
        rec = recs[it['view']]
        uv = project_ortho(rec, M, X, it['size'])
        d = (X @ M[:3, :3].T + M[:3, 3] - np.asarray(rec['location'], float)) @ np.asarray(rec['forward'], float)
        cells = (uv - rects[n]) * z / STRIDE
        tri, bary, _ = rasterize(cells, d, tris, (HEAT, HEAT))
        on = tri >= 0
        part = tri_bone[tri[on]]
        ok = part < nb                                                  # a prop's skin is background here
        r = (bary[on][:, :, None] * coords[tris[tri[on]]]).sum(1)
        lc = np.einsum('nij,nj->ni', Rt[np.minimum(part, nb - 1)], r - rig.rest_head[np.minimum(part, nb - 1)]) / H
        pm = np.where(ok, part, BG).astype(np.uint8)
        parts[n][on] = pm
        lm = np.where(ok[:, None], lc, 0.0)
        local[n][on] = lm.astype(np.float16)
        if (n + 1) % 1000 == 0:
            print(f'DENSE labels {n + 1}/{N} ({time.time() - t0:.0f} s)', flush=True)
    out = renders / 'dense_labels.npz'
    np.savez_compressed(out, parts=parts, local=local, rects=rects, side=side, index=np.array([it['pose'] for it in items]),
                        nb=nb, height=H)
    cover = (parts != BG).mean()
    print(f'DENSE labels -> {out}: {N} maps {HEAT}x{HEAT}, the creature covers {100 * cover:.0f}% of the cells, '
          f'{nb} parts ({time.time() - t0:.0f} s)', flush=True)
    # a check sheet: windows with the parts coloured over them
    pick = list(range(0, N, max(1, N // 8)))[:8]
    imgs, uv, _ = load_windows([items[i] for i in pick], rects[pick], side)
    cols = part_colours(nb)
    rows = []
    for i, n in enumerate(pick):
        col = np.zeros((HEAT, HEAT, 3), np.uint8)
        on = parts[n] != BG
        col[on] = cols[parts[n][on]]
        big = cv2.resize(col, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        mk = cv2.resize(on.astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0
        over = imgs[i].copy()
        over[mk] = (.45 * over[mk] + .55 * big[mk]).astype(np.uint8)
        for (u, v) in uv[i]:
            cv2.circle(over, (int(u), int(v)), 2, (255, 255, 255), -1)
        rows.append(np.hstack([imgs[i], over]))
    cv2.imwrite(str(renders / 'dense_labels_check.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ----- training -------------------------------------------------------------------------------------------------
def train(a):
    import torch
    import torch.nn.functional as F
    from .net import count_params, gaussians, sample_maps
    from .train import augment, batches, evaluate, evaluate_depth, load_items
    t0 = time.time()
    log = lambda s: print(f'[{time.time() - t0:6.1f}s] {s}', flush=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    pr = Project(a.project)
    cfg = pr.clip_cfg('')
    cfg.setdefault('solve', {})
    cr = Creature(pr, cfg)
    rig = cr.rig
    nb = rig.nb
    kps = keypoints(cr)
    renders = pr.root / a.renders
    items = load_items(pr, kps, renders, height=cr.height, cr=cr)
    dl = np.load(renders / 'dense_labels.npz')
    assert len(dl['parts']) == len(items) and int(dl['nb']) == nb, 'dense_labels.npz: not these pictures (labels again)'
    side, rects = int(dl['side']), dl['rects']
    imgs, uv, bg = load_windows(items, rects, side)
    parts_all, local_all = dl['parts'], dl['local']
    z = SIZE / side
    dd_all = np.stack([it['d'] for it in items]).astype(np.float32)
    seen = np.stack([it['seen'] for it in items])
    inside = (uv[..., 0] >= 0) & (uv[..., 0] < SIZE) & (uv[..., 1] >= 0) & (uv[..., 1] < SIZE)
    views = np.array([it['view'] for it in items])
    scale = np.array([z * it['per_clip_px'] for it in items], np.float32)       # input px per clip px
    poses = np.array([it['pose'] for it in items])
    cut = np.quantile(poses, .95)
    val = poses > cut
    tr = ~val
    log(f'{len(items)} windows of {side} px (x{z:.2f}), {int(tr.sum())} to learn, {int(val.sum())} held out; '
        f'{len(kps)} keypoints, {nb} parts; {imgs.nbytes / 1e9:.1f} GB; seen {seen.mean():.2f} of the keypoints')
    model = Reader(len(kps), depth=True, dense=nb).to(dev)
    log(f'reader: {count_params(model):.1f} M parameters')
    if a.init:
        ck = torch.load(a.init, map_location='cpu', weights_only=False)
        own = model.state_dict()
        same = [tuple(k) for k in ck.get('keypoints', [])] == [tuple(k) for k in kps]
        take = {k: v for k, v in ck['state'].items()
                if k in own and own[k].shape == v.shape and (same or not k.startswith(('head.', 'dhead.')))}
        model.load_state_dict(take, strict=False)
        log(f'warm start from {a.init}: {len(take)} of {len(own)} tensors' + ('' if same else ' (new keypoint heads)'))
    model = model.to(memory_format=torch.channels_last)
    names = json.loads((pr.start_dir / 'cameras.json').read_text(encoding='utf-8'))['views']
    vit = [p for p in model.vit.parameters() if p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith('vit.')]
    opt = torch.optim.AdamW([{'params': vit, 'lr': a.lr * .1}, {'params': rest, 'lr': a.lr}], weight_decay=1e-2)
    base = [g['lr'] for g in opt.param_groups]
    tri = np.nonzero(tr)[0]
    it_tr = batches(len(tri), a.batch, rng)
    cls_w = torch.full((nb + 1,), a.part_weight_fg, device=dev)
    cls_w[0] = 1.0
    warm, hist = 100, []
    model.train()
    for step in range(a.steps):
        sel = tri[next(it_tr)]
        x_u8 = torch.from_numpy(imgs[sel]).to(dev, non_blocking=True)
        u = torch.from_numpy(uv[sel]).to(dev)
        b = torch.from_numpy(bg[sel]).to(dev)
        ins = torch.from_numpy(inside[sel]).to(dev)
        pos = torch.from_numpy(seen[sel]).to(dev)
        # the maps go through the same re-framing: class 0 = background (what the re-framing uncovers too)
        pm = parts_all[sel].astype(np.int64)
        cls = np.where(pm == BG, 0, pm + 1).astype(np.float32)
        maps = torch.from_numpy(np.concatenate([cls[..., None], local_all[sel].astype(np.float32)], -1)).permute(0, 3, 1, 2).to(dev)
        x, u, maps = augment(x_u8, u, b, rng, maps=maps)
        cls_t = maps[:, 0].round().long()
        loc_t = maps[:, 1:4]
        ok = ins & (u[..., 0] >= 0) & (u[..., 0] < SIZE) & (u[..., 1] >= 0) & (u[..., 1] < SIZE)
        target = gaussians(u, (ok & pos).float(), sigma=2.0)
        lr_k = min((step + 1) / warm, 1.0) * .5 * (1 + math.cos(math.pi * step / a.steps))
        for g, l0 in zip(opt.param_groups, base):
            g['lr'] = l0 * lr_k
        with torch.autocast('cuda', dtype=torch.bfloat16):
            heat, dmap, plog, loc = model(normalise(x).contiguous(memory_format=torch.channels_last))
        weight = 1 + 9 * (target > .05).float()
        loss_h = ((weight * (heat.float() - target) ** 2).mean((2, 3))).sum(1).mean() * 100
        m = ok.float()
        err = (sample_maps(dmap.float(), u) - torch.from_numpy(dd_all[sel]).to(dev)).abs()
        loss_d = (err * m).sum() / m.sum().clamp(min=1)
        loss_p = F.cross_entropy(plog.float(), cls_t, weight=cls_w)
        fg = cls_t > 0
        j = (cls_t - 1).clamp(min=0)                                     # the bone named, per cell
        loc_f = loc.float()
        B_, _, h_, w_ = loc_f.shape
        idx = (3 * j).unsqueeze(1) + torch.arange(3, device=dev).view(1, 3, 1, 1)
        pred = torch.gather(loc_f, 1, idx)                                # (B, 3, h, w): the named bone's place
        loss_l = ((pred - loc_t).abs().sum(1) * fg.float()).sum() / fg.float().sum().clamp(min=1)
        loss = loss_h + a.depth_weight * loss_d + a.part_weight * loss_p + a.local_weight * loss_l
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        if (step + 1) % 100 == 0:
            el = time.time() - t0
            hist.append(dict(step=step + 1, loss=round(float(loss), 4), heat=round(float(loss_h), 4),
                             depth=round(float(loss_d), 4), parts=round(float(loss_p), 4), place=round(float(loss_l), 4)))
            log(f'step {step + 1} loss {float(loss):.3f} (heat {float(loss_h):.3f}, depth {float(loss_d):.4f} heights, '
                f'parts {float(loss_p):.3f}, place {float(loss_l):.4f} heights)  ({el / (step + 1):.3f} s per step)')
    out = pr.root / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    w0, h0 = items[0]['size']
    window = dict(side_px=side, strip_w=int(w0), strip_h=int(h0), per_clip_px=float(items[0]['per_clip_px']))
    ck = {'state': model.state_dict(), 'keypoints': kps, 'size': SIZE, 'context': 0, 'depth': True, 'dense': nb,
          'renders': a.renders, 'steps': a.steps, 'window': window, 'bones': rig.names[:nb]}
    torch.save(ck, out)                                   # the weights first: a check that fails must not lose them
    ev_valid = inside & seen
    report = {names[v]['name']: e for v, e in evaluate(model, imgs[val], uv[val], ev_valid[val], scale[val],
                                                        views[val], dev).items()}
    log(f'held out, keypoints: {report}')
    off = np.array([is_offaxis(kp) for kp in kps])
    if off.any():
        rep_off = {names[v]['name']: e for v, e in evaluate(model, imgs[val], uv[val], ev_valid[val] & off[None],
                                                            scale[val], views[val], dev).items()}
        log(f'held out, the side points: {rep_off}')
    else:
        rep_off = None
    depth_report = {names[v]['name']: e for v, e in evaluate_depth(
        model, imgs[val], uv[val], dd_all[val], inside[val], seen[val], views[val], kps, cr.height, dev).items()}
    log(f'held out, depth: {depth_report}')
    dense_report = evaluate_dense(model, imgs[val], parts_all[val], local_all[val], dev)
    log(f'held out, dense: {dense_report}')
    ck.update({'held_out': report, 'held_out_side_points': rep_off, 'held_out_depth': depth_report,
               'held_out_dense': dense_report})
    torch.save(ck, out)
    write_json(out.with_name(out.stem + '-train.json'), {'held_out': report, 'held_out_side_points': rep_off,
                                                         'held_out_depth': depth_report, 'held_out_dense': dense_report,
                                                         'window': window, 'steps': a.steps, 'batch': a.batch,
                                                         'minutes': round((time.time() - t0) / 60, 2),
                                                         'pictures': len(items), 'history': hist})
    log(f'saved {out}')


def evaluate_dense(model, imgs, parts, local, dev, bs=32):
    """The dense map on held-out windows: the creature's cells found (overlap of creature against background), the
    bone named right on the true creature cells (accuracy), and, where it is right, how far the place on the bone
    misses (median and p90, % of the height)."""
    import torch
    model.eval()
    hit = tot = inter = union = 0
    miss = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            x = normalise(torch.from_numpy(imgs[i:i + bs]).to(dev))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, _, plog, loc = model(x)
            cls = plog.float().argmax(1).cpu().numpy()                        # 0 background, j + 1 bone j
            pm = parts[i:i + bs].astype(np.int64)
            true = np.where(pm == BG, 0, pm + 1)
            fg = true > 0
            right = (cls == true) & fg
            hit += int(right.sum())
            tot += int(fg.sum())
            inter += int(((cls > 0) & fg).sum())
            union += int(((cls > 0) | fg).sum())
            locn = loc.float().cpu().numpy()
            b_, y_, x_ = np.nonzero(right)
            j = true[b_, y_, x_] - 1
            pred = np.stack([locn[b_, 3 * j + c, y_, x_] for c in range(3)], 1)
            miss += list(np.linalg.norm(pred - local[i:i + bs].astype(np.float32)[b_, y_, x_], axis=1))
    model.train()
    miss = np.array(miss) if miss else np.zeros(1)
    return dict(part_accuracy=round(hit / max(tot, 1), 3), mask_overlap=round(inter / max(union, 1), 3),
                place_miss_pct=round(float(np.median(miss)) * 100, 2),
                place_miss_p90_pct=round(float(np.percentile(miss, 90)) * 100, 2))


# ----- reading --------------------------------------------------------------------------------------------------
def read(a):
    import torch
    from .. import clip as clipmod
    from .net import sample_maps
    from .read import _read, _warp_back, _warp_copy, load_reader, panels, tta_copies
    t0 = time.time()
    pr = Project(a.project)
    dev = torch.device('cuda')
    rpath = Path(a.reader) if a.reader else pr.root / 'reader' / 'reader-dense.pt'
    model, ck = load_reader(rpath, dev)
    if not ck.get('dense'):
        raise SystemExit(f'{rpath}: not a dense reader')
    kps = [tuple(k) for k in ck['keypoints']]
    K, nb = len(kps), int(ck['dense'])
    cams = json.loads((pr.start_dir / 'cameras.json').read_text(encoding='utf-8'))
    frames = clipmod.read_frames(pr.clip_file(a.clip))
    pvs = panels(frames, cams)
    V, T = len(pvs), len(frames)
    win = ck['window']
    # pass 1: where the body is in each panel. A whole-panel reader (the locator) reads every frame, or the clip's
    # own silhouettes (--locator masks: the tool's silhouettes against the studio backdrop) give the body's centre
    cr = locator = None
    if a.locator == 'masks':
        cfg = pr.clip_cfg(a.clip)
        cfg.setdefault('solve', {})
        cr = Creature(pr, cfg)
    elif a.locator != 'none':
        lpath = Path(a.locator) if a.locator else pr.root / 'reader' / 'reader-pts.pt'
        locator, _ = load_reader(lpath, dev)
    copies = tta_copies(a.dense_tta) if a.dense_tta > 1 else [(1.0, 0.0, 0.0)]
    grid = (np.stack(np.meshgrid(np.arange(HEAT), np.arange(HEAT)), -1).reshape(-1, 2) + .5) * STRIDE   # input px
    uv = np.zeros((T, V, K, 2), np.float32)
    conf = np.zeros((T, V, K), np.float32)
    dep = np.zeros((T, V, K), np.float32)
    cells, sides, corners, canvases = [], [], [], []
    for v, pv in enumerate(pvs):
        pn = pv['panel']
        crops = [np.ascontiguousarray(pn.cut(f)) for f in frames]
        hh, ww = crops[0].shape[:2]
        centre = np.full((T, 2), np.nan)
        if cr is not None:
            view = cr.views(a.clip)[v]
            for t in range(T):
                ys, xs = np.nonzero(view['mask'](t))
                if len(xs) >= 50:
                    centre[t] = (float(xs.mean()), float(ys.mean()))
        elif locator is not None:
            with torch.no_grad():
                p1, c1, _ = _read(locator, crops, dev, tta=a.tta)
            for t in range(T):
                ok = c1[t] >= .3
                if ok.sum() >= 3:
                    centre[t] = np.median(p1[t, ok], 0)
        have = ~np.isnan(centre[:, 0])
        if not have.any():
            centre[:] = (ww / 2, hh / 2)
        else:
            for d in range(2):
                centre[:, d] = np.interp(np.arange(T), np.nonzero(have)[0], centre[have, d])
        centre = np.stack([np.convolve(np.pad(centre[:, d], 3, mode='edge'), np.ones(7) / 7, mode='valid')
                           for d in range(2)], 1)
        side = int(round(win['side_px'] / win['per_clip_px'] * (hh / (win['strip_h'] / win['per_clip_px']))))
        side = min(side, ww, hh)
        cx = np.clip(np.round(centre[:, 0] - side / 2), 0, ww - side).astype(int)
        cy = np.clip(np.round(centre[:, 1] - side / 2), 0, hh - side).astype(int)
        z = SIZE / side
        canv = np.empty((T, SIZE, SIZE, 3), np.uint8)
        for t in range(T):
            w_ = crops[t][cy[t]:cy[t] + side, cx[t]:cx[t] + side]
            canv[t] = cv2.resize(w_, (SIZE, SIZE), interpolation=cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR)
        c_uv, c_part, c_loc, c_prob = [None] * T, [None] * T, [None] * T, [None] * T
        with torch.no_grad():
            for i in range(0, T, 16):
                x0 = normalise(torch.from_numpy(canv[i:i + 16]).to(dev))
                # every window read once per copy (shifted and scaled a little, as the keypoints' TTA), the maps
                # brought back onto the window and averaged: an error that does not repeat between the copies shrinks
                heat_s = dmap_s = prob_s = loc_s = None
                for s, dx, dy in copies:
                    x = x0 if (s, dx, dy) == (1.0, 0.0, 0.0) else _warp_copy(x0, s, dx, dy)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        heat, dmap, plog, loc = model(x)
                    heat, dmap, prob, loc = heat.float(), dmap.float(), torch.softmax(plog.float(), 1), loc.float()
                    if (s, dx, dy) != (1.0, 0.0, 0.0):
                        heat, dmap, prob, loc = (_warp_back(m, s, dx, dy) for m in (heat, dmap, prob, loc))
                    heat_s = heat if heat_s is None else heat_s + heat
                    dmap_s = dmap if dmap_s is None else dmap_s + dmap
                    prob_s = prob if prob_s is None else prob_s + prob
                    loc_s = loc if loc_s is None else loc_s + loc
                n_c = len(copies)
                heat, dmap, prob, loc = heat_s / n_c, dmap_s / n_c, prob_s / n_c, loc_s / n_c
                p, c = decode(heat)
                d = sample_maps(dmap, p)
                pc, cls = prob.max(1)
                p, c, d = p.cpu().numpy(), c.cpu().numpy(), d.cpu().numpy()
                pc, cls, locn = pc.cpu().numpy(), cls.cpu().numpy(), loc.cpu().numpy()
                for b in range(len(p)):
                    t = i + b
                    uv[t, v] = pn.to_camera((p[b] / z) + np.array([cx[t], cy[t]]))
                    conf[t, v], dep[t, v] = c[b], d[b]
                    on = (cls[b] > 0) & (pc[b] > a.min_prob)
                    yy, xx = np.nonzero(on)
                    j = cls[b][yy, xx] - 1
                    lc = np.stack([locn[b, 3 * j + k, yy, xx] for k in range(3)], 1)
                    c_uv[t] = pn.to_camera((grid[yy * HEAT + xx] / z) + np.array([cx[t], cy[t]])).astype(np.float32)
                    c_part[t] = j.astype(np.uint8)
                    c_loc[t] = lc.astype(np.float16)
                    c_prob[t] = pc[b][yy, xx].astype(np.float16)
        cells.append(dict(uv=c_uv, part=c_part, loc=c_loc, prob=c_prob))
        sides.append(side)
        corners.append(np.stack([cx, cy], 1))
        canvases.append(canv)
    # the cells of frame t: the first panel's, then the next's; offsets (T + 1) over frames, `view` per cell
    all_uv, all_part, all_loc, all_prob, all_view, offsets = [], [], [], [], [], [0]
    for t in range(T):
        for v in range(V):
            all_uv.append(cells[v]['uv'][t])
            all_part.append(cells[v]['part'][t])
            all_loc.append(cells[v]['loc'][t])
            all_prob.append(cells[v]['prob'][t])
            all_view.append(np.full(len(cells[v]['uv'][t]), v, np.uint8))
        offsets.append(offsets[-1] + sum(len(cells[v]['uv'][t]) for v in range(V)))
    out = pr.work(a.clip) / f'dense{a.tag}'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'reading.npz', uv=uv, conf=conf, dep=dep, views=np.array([pv['name'] for pv in pvs]),
                        keypoints=json.dumps(kps), reader=str(rpath))
    np.savez_compressed(out / 'dense.npz', offsets=np.array(offsets), uv=np.concatenate(all_uv),
                        part=np.concatenate(all_part), local=np.concatenate(all_loc), prob=np.concatenate(all_prob),
                        view=np.concatenate(all_view), side=np.array(sides), corners=np.stack(corners),
                        height=float(ck.get('height', 0.0)))
    n_cells = np.diff(offsets)
    print(f'DENSE read {a.clip}: {T} frames x {V} panels through windows of {sides} px; median confidence '
          f'{np.median(conf):.2f}; creature cells per frame median {int(np.median(n_cells))} ({time.time() - t0:.0f} s) '
          f'-> {out}', flush=True)
    # a sheet: each panel's window with the parts coloured, the keypoints as dots, the panels side by side
    cols = part_colours(nb)
    picks = [int(s) for s in a.sheet.split(',') if s and int(s) < T]
    rows = []
    for t in picks:
        tiles = []
        for v in range(V):
            z = SIZE / sides[v]
            cx, cy = corners[v][t]
            col = np.zeros((HEAT, HEAT, 3), np.uint8)
            pts = ((pvs[v]['panel'].from_camera(cells[v]['uv'][t].astype(float)) - np.array([cx, cy])) * z / STRIDE - .5)
            pts = np.clip(pts.round().astype(int), 0, HEAT - 1)
            col[pts[:, 1], pts[:, 0]] = cols[cells[v]['part'][t]]
            on = col.any(2)
            big = cv2.resize(col, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            mk = cv2.resize(on.astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0
            over = canvases[v][t].copy()
            over[mk] = (.4 * over[mk] + .6 * big[mk]).astype(np.uint8)
            for k in range(K):
                q = ((pvs[v]['panel'].from_camera(uv[t, v, k].astype(float)) - np.array([cx, cy])) * z).astype(int)
                cv2.circle(over, (int(q[0]), int(q[1])), 3, (255, 255, 255) if conf[t, v, k] >= .3 else (0, 0, 0), -1)
            tiles += [canvases[v][t], over]
        row = np.hstack(tiles)
        cv2.putText(row, f'{t}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(row)
    if rows:
        cv2.imwrite(str(out / 'sheet.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ----- the fit --------------------------------------------------------------------------------------------------
def fit(a):
    from scipy.optimize import least_squares
    from .fit import creature_for
    t0 = time.time()
    pr = Project(a.project)
    clip = a.clip
    cfg = pr.clip_cfg(clip)
    cfg.setdefault('solve', {})
    rd = np.load(pr.work(clip) / a.read / 'reading.npz')
    de = np.load(pr.work(clip) / a.read / 'dense.npz')
    kps = [tuple(k) for k in json.loads(str(rd['keypoints']))]
    cr, _ = creature_for(pr, cfg, kps)
    rig = cr.rig
    kr = KeypointRig(cr, kps)
    views = cr.views(clip)
    V = len(views)
    # the reading's panels in the creature's order
    names_rd = [str(v) for v in rd['views']]
    order = [names_rd.index(v['name']) for v in views]
    inv = np.zeros(len(names_rd), int)
    inv[order] = np.arange(V)
    H = cr.height
    body = np.array([not is_prop(kp) for kp in kps])
    sights = []
    for v in views:
        n = v['cam'].Ma.T @ v['cam'].fwd
        sights.append((n / np.linalg.norm(n), v['cam'].px_per_unit))
    lo, hi, p0 = cr.bounds(clip)
    if a.no_bounds:
        lo, hi = np.full(rig.n, -np.inf), np.full(rig.n, np.inf)
    T = min(len(rd['uv']), len(de['offsets']) - 1)
    if a.frames:
        T = min(T, a.frames)
    uv, conf, dep = rd['uv'][:T][:, order], rd['conf'][:T][:, order], rd['dep'][:T][:, order] * H
    offsets = de['offsets']
    cview = inv[de['view']] if 'view' in de.files else np.zeros(len(de['uv']), int)
    smooth = np.full(rig.n, a.smooth)
    smooth[:3] = a.smooth * 20
    tw = np.array([rig.index(rig.names[j], ax) for j, ax in rig.channels if ax == 1 and j != rig.root], int)
    init = np.load(pr.work(clip) / a.init / 'params.npy')[:T] if a.init else None
    P = np.zeros((T, rig.n))
    prev = p0.copy()
    miss, used, n_init = [], [], 0
    per_view = max(1, a.cells // V)
    for t in range(T):
        s_, e_ = offsets[t], offsets[t + 1]
        cuv_a, part_a, loc_a, prob_a, cv_a = (de['uv'][s_:e_], de['part'][s_:e_].astype(int), de['local'][s_:e_].astype(float),
                                              de['prob'][s_:e_].astype(float), cview[s_:e_])
        keep = []
        for v in range(V):
            iv = np.nonzero(cv_a == v)[0]
            if len(iv) > per_view:
                # every k-th cell in the map's own order, the same rule in every frame (a random set each frame made
                # the body jump from frame to frame)
                iv = iv[np.linspace(0, len(iv) - 1, per_view).round().astype(int)]
            keep.append(iv)
        keep = np.concatenate(keep) if keep else np.zeros(0, int)
        cuv, part, local, prob, cv = cuv_a[keep], part_a[keep], loc_a[keep], prob_a[keep], cv_a[keep]
        used.append(len(cuv))
        rest = rig.rest_head[part] + np.einsum('nij,nj->ni', rig.rest_rot[part], local * H)
        idx = np.zeros((len(rest), 4), np.int64)
        idx[:, 0] = part
        wts = np.zeros((len(rest), 4))
        wts[:, 0] = 1.0
        wc = a.dense_weight * prob
        by_view = [np.nonzero(cv == v)[0] for v in range(V)]

        def resid(p, jac=False):
            X, J = kr.points(p, jac=True) if jac else (kr.points(p), None)
            if len(rest):
                Xc, Jc = rig.skin(p, rest, idx, wts, jac=True) if jac else (rig.skin(p, rest, idx, wts), None)
            mid = X[body].mean(0)
            r, rows = [], []
            for v, view in enumerate(views):
                n, kpx = sights[v]
                pix, Jp = view['project'](X, True) if jac else (view['project'](X), None)
                r.append(((pix - uv[t, v]) * conf[t, v][:, None]).ravel())
                r.append(a.depth * kpx * conf[t, v] * ((X - mid) @ n - dep[t, v]))
                iv = by_view[v]
                if len(iv):
                    pc, Jpc = view['project'](Xc[iv], True) if jac else (view['project'](Xc[iv]), None)
                    r.append(((pc - cuv[iv]) * wc[iv][:, None]).ravel())
                if jac:
                    rows.append((np.einsum('kab,kbn->kan', Jp, J) * conf[t, v][:, None, None]).reshape(-1, rig.n))
                    Jd = J - J[body].mean(0)[None]
                    rows.append((a.depth * kpx * conf[t, v])[:, None] * np.einsum('d,kdn->kn', n, Jd))
                    if len(iv):
                        rows.append((np.einsum('kab,kbn->kan', Jpc, Jc[iv]) * wc[iv][:, None, None]).reshape(-1, rig.n))
            r.append(smooth * (p - prev))
            r.append(a.twist * (p[tw] - p0[tw]))
            if not jac:
                return np.concatenate(r)
            rows.append(np.diag(smooth))
            Jt = np.zeros((len(tw), rig.n))
            Jt[np.arange(len(tw)), tw] = a.twist
            rows.append(Jt)
            return np.concatenate(r), np.concatenate(rows)
        n_rob = V * 3 * len(kps) + 2 * len(cuv)

        def loss(z):
            """Cauchy on the readings and the cells, plain squares on the smoothness and the twist."""
            out = np.empty((3, len(z)))
            d = slice(0, n_rob)
            out[0, d] = np.log1p(z[d])
            out[1, d] = 1.0 / (1.0 + z[d])
            out[2, d] = -1.0 / (1.0 + z[d]) ** 2
            out[0, n_rob:] = z[n_rob:]
            out[1, n_rob:] = 1.0
            out[2, n_rob:] = 0.0
            return out
        starts = [np.clip(prev if t else p0, lo + 1e-9, hi - 1e-9)]
        if init is not None:
            starts.append(np.clip(init[t], lo + 1e-9, hi - 1e-9))
        sols = [least_squares(lambda p: resid(p), x0, jac=lambda p: resid(p, True)[1], bounds=(lo, hi), loss=loss,
                              f_scale=a.cauchy, max_nfev=a.iters, x_scale='jac') for x0 in starts]
        best = int(np.argmin([s.cost for s in sols]))
        n_init += int(best == 1)
        P[t] = sols[best].x
        prev = P[t]
        X = kr.points(P[t])
        m = []
        for v, view in enumerate(views):
            sure = conf[t, v] > .5
            if sure.any():
                m += list(np.linalg.norm(view['project'](X)[sure] - uv[t, v][sure], axis=1))
        miss.append(float(np.median(m)) if m else np.nan)
    fx = pr.work(clip) / f'fx_dense{a.tag}'
    fx.mkdir(parents=True, exist_ok=True)
    np.save(fx / 'params.npy', P)
    cr.export_motion(clip, P, str(fx / 'motion.npz'))
    print(f'DENSE fit {clip}: {T} frames x {V} panels in {time.time() - t0:.0f} s, {int(np.median(used))} cells a frame '
          f'(median), the sure points missed by {np.nanmedian(miss):.1f} px (median), worst frame {np.nanmax(miss):.1f} px'
          + (f'; the 1:1 start won in {n_init} frames' if init is not None else '') + f' -> {fx}', flush=True)
    gt_path = pr.root / 'gt' / clip[3:] / 'gt.npz'
    if clip.startswith('gt_') and gt_path.exists():
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'dev' / 'bench'))
        from score import score
        gt = np.load(gt_path)
        for name, Q in (('dense', P),):
            sc = score(cr, Q, gt)
            print(f'DENSE score {clip} {name}: {sc["fitted_median_deg"]:.2f} deg a bone (p90 {sc["fitted_p90_deg"]:.2f}), '
                  f'limbs {sc["limbs_body_median_deg"]:.2f}', flush=True)
            if name == 'dense':
                for row in sc['chains']:
                    print(f'  chain {row["chain"]:12s} median {row["median_deg"]:5.1f}  p90 {row["p90_deg"]:5.1f}  '
                          f'vs body {row["body_median_deg"]:5.1f}  worst {row["worst_bone"]}', flush=True)


def main():
    ap = argparse.ArgumentParser(prog='video2rig.reader.dense')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('labels')
    p.add_argument('project')
    p.add_argument('--renders', default='reader/renders-dense')
    p.add_argument('--seed', type=int, default=0)
    p = sub.add_parser('train')
    p.add_argument('project')
    p.add_argument('--renders', default='reader/renders-dense')
    p.add_argument('--init', default='', help='the reader to start from (stage A: reader/reader-pts.pt)')
    p.add_argument('--steps', type=int, default=8000)
    p.add_argument('--batch', type=int, default=24)
    p.add_argument('--lr', type=float, default=4e-4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--depth-weight', type=float, default=20.0)
    p.add_argument('--part-weight', type=float, default=5.0, help='the parts\' cross-entropy against the heat maps\' loss')
    p.add_argument('--part-weight-fg', type=float, default=3.0, help='a creature cell against a background one')
    p.add_argument('--local-weight', type=float, default=20.0, help='the place\'s miss (heights) against the heat maps\'')
    p.add_argument('--out', default='reader/reader-dense.pt')
    p = sub.add_parser('read')
    p.add_argument('project')
    p.add_argument('clip')
    p.add_argument('--reader', default='')
    p.add_argument('--locator', default='', help='a whole-panel reader that finds the body first (default '
                   'reader/reader-pts.pt), or "masks": the clip\'s silhouettes give the body\'s centre')
    p.add_argument('--tta', type=int, default=0, help='the locator\'s copies per frame')
    p.add_argument('--dense-tta', type=int, default=1, help='the dense reader\'s copies per window (shifted, scaled), '
                   'its maps averaged')
    p.add_argument('--min-prob', type=float, default=.5, help='a cell counts when its bone\'s probability is over this')
    p.add_argument('--sheet', default='0,30,60,90,120')
    p.add_argument('--tag', default='')
    p = sub.add_parser('fit')
    p.add_argument('project')
    p.add_argument('clip')
    p.add_argument('--read', default='dense', help='work/<clip>/<read>/: the dense reading')
    p.add_argument('--init', default='', help='a fit to start every frame from as well (fx_direct: the 1:1 pose)')
    p.add_argument('--cells', type=int, default=1500, help='creature cells used per frame (at random)')
    p.add_argument('--dense-weight', type=float, default=1.0, help='a cell\'s pull against a keypoint\'s (times its '
                   'probability): the place read on the bone, where the pose must put it (0 = off)')
    p.add_argument('--icp', type=float, default=0.0, help='a cell\'s pull toward the nearest seen vertex of the bone '
                   'it names, as the pose projects it (the part\'s shape; 0 = off)')
    p.add_argument('--icp-verts', type=int, default=300, help='vertices per bone for those matches')
    p.add_argument('--consistent', action='store_true', help='solve each frame twice: the cells whose bone the fitted '
                   'pose does not show at their pixel are left out the second time')
    p.add_argument('--depth', type=float, default=1.0, help='the depth read against the picture (px per px)')
    p.add_argument('--smooth', type=float, default=30.0, help='px-equivalent a radian toward the frame before (the '
                   'shipped fit\'s 30; at 5 the roar\'s bones jolted 1.8 deg a frame squared, five times the shipped fit)')
    p.add_argument('--twist', type=float, default=5.0, help='px-equivalent a radian of a bone\'s twist from the start')
    p.add_argument('--cauchy', type=float, default=6.0, help='robust scale of a miss, px')
    p.add_argument('--iters', type=int, default=40)
    p.add_argument('--no-bounds', action='store_true', help='no joint ranges')
    p.add_argument('--frames', type=int, default=0, help='only the first N frames (a quick check)')
    p.add_argument('--tag', default='')
    a = ap.parse_args()
    {'labels': labels, 'train': train, 'read': read, 'fit': fit}[a.cmd](a)


if __name__ == '__main__':
    main()
