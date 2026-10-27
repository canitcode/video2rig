"""Read every joint of a clip with the creature's reader: per frame and panel, each keypoint's pixel and how sure the
reader is (the peak of its heat map, about 1 where it sees the joint plainly, low where it guesses).

    python -m video2rig.reader.read <project> <clip> [--reader reader/reader.pt] [--sheet 0,20,36,52,75,100]
        -> work/<clip>/reader.npz   uv (T, views, K, 2) in the panel's pixels (the tool's camera convention: column j
                                    covers [j, j + 1) from the panel's left edge), conf (T, views, K), labels, views
           work/<clip>/reader-sheet.jpg   the joints drawn on some frames, both panels (left orange, right blue)
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .. import clip as clipmod
from ..common import Project
from .device import autocast, pick_device
from .net import SIZE, Reader, decode, letterbox, normalise


def load_reader(path, dev):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = Reader(len(ck['keypoints']), pretrained=False, context=ck.get('context', 0), depth=ck.get('depth', False),
                   skin=ck.get('skin', False), dense=ck.get('dense', 0))
    model.load_state_dict(ck['state'])
    return model.to(dev).eval(), ck


def panels(frames, cams):
    """Per view: where its camera's image lies in the frame (clip.Panel: a column of a side-by-side frame, or a
    strip of a stacked frame turned upright) and the size of that image as cut out of a frame. The cut starts at a
    whole pixel, the camera's image maybe not (the front panel of a 1280 px clip starts at 853.33 px; the cut at
    853): Panel.to_camera puts the remainder back on the readings."""
    from ..clip import Panel
    H, W = frames[0].shape[:2]
    out = []
    for rec in cams['views']:
        pn = Panel(rec, (W, H), cams['frame_px'])
        w, h = (pn.h, pn.w) if pn.turn % 2 else (pn.w, pn.h)
        out.append(dict(name=rec['name'], panel=pn, x0=pn.x0, x0i=pn.x0i, w=w, h=h))
    return out


TTA_SHIFT = 6.0        # a copy's shift, up to this many of the reader's input pixels each way
TTA_SCALE = .05        # and its scale, 1 -+ this, about the square's middle
TTA_AGREE = 6.0        # a copy agrees when its own peak lies within this many input pixels of the joint read
TTA_CONF = 'mean'      # a read-N-times keypoint's confidence: 'mean' (the copies' mean peak), 'agree' (that x the
                       # share of copies that agree: it starved half-sure hidden arms of weight, the fit let them go,
                       # an alien's landing and a troll's far arm came out wrong), 'heat' (the peak of the averaged heat
                       # map); a reading file keeps all three (conf_agree, conf_mean, conf_heat), `conf` as chosen here


def tta_copies(n, seed=0):
    """(scale, dx, dy) of the n copies of a picture the reader reads: the first as it is, the rest shifted by a few
    pixels and scaled a little about the reader's square's middle, the same copies for every picture and clip (a
    clip reads the same every time). Never mirrored: that swaps left and right."""
    rng = np.random.default_rng(seed)
    out = [(1.0, 0.0, 0.0)]
    while len(out) < n:
        s = float(rng.uniform(1 - TTA_SCALE, 1 + TTA_SCALE))
        dx, dy = (float(v) for v in rng.uniform(-TTA_SHIFT, TTA_SHIFT, 2))
        out.append((s, dx, dy))
    return out


def _warp_back(heat, s, dx, dy):
    """Heat maps (B, K, h, w) read from a copy of the square (scaled by s about its middle, shifted by dx, dy input
    pixels) -> the same maps in the square's own frame (bilinear; what the copy did not show: 0)."""
    import torch.nn.functional as F
    from .net import STRIDE
    B, K, h, w = heat.shape
    c0 = SIZE / 2
    gx = (s * ((torch.arange(w, device=heat.device, dtype=heat.dtype) + .5) * STRIDE - c0) + c0 + dx) / STRIDE / w
    gy = (s * ((torch.arange(h, device=heat.device, dtype=heat.dtype) + .5) * STRIDE - c0) + c0 + dy) / STRIDE / h
    grid = torch.stack(torch.meshgrid(gy * 2 - 1, gx * 2 - 1, indexing='ij')[::-1], -1)        # (h, w, 2): x, y
    return F.grid_sample(heat, grid[None].expand(B, h, w, 2), mode='bilinear', padding_mode='zeros',
                         align_corners=False)


def _warp_copy(x, s, dx, dy):
    """The reader's input (B, 3, SIZE, SIZE) as a copy scaled by s about the square's middle and shifted by dx, dy
    pixels (bilinear, the edges repeated), on the card."""
    import torch.nn.functional as F
    c0 = SIZE / 2
    ax = torch.arange(SIZE, device=x.device, dtype=x.dtype) + .5
    gx = ((ax - c0 - dx) / s + c0) / SIZE * 2 - 1                 # where each copy pixel comes from in the square
    gy = ((ax - c0 - dy) / s + c0) / SIZE * 2 - 1
    grid = torch.stack(torch.meshgrid(gy, gx, indexing='ij')[::-1], -1)
    return F.grid_sample(x, grid[None].expand(len(x), SIZE, SIZE, 2), mode='bilinear', padding_mode='border',
                         align_corners=False)


def _read(model, crops, dev, bs=16, corners=None, tta=0, tta_mode='heat', second=None, extra=None):
    """Every keypoint in same-size images, each letterboxed into the reader's square: (p (T, K, 2) in the images'
    pixels, conf (T, K), depth (T, K)). A reader with context reads twice: every frame alone, then every frame again
    with what the first reading found in the frame before and the frame after (at the clip's ends the frame's own
    first reading stands in for the missing one). corners (T, 2): where each image was cut from a larger one (moving
    reading windows): the neighbours' readings are moved into the frame's own window first.

    tta > 1: every picture is read that many times (tta_copies: shifted by a few pixels, scaled 0.95-1.05), each
    copy's heat maps brought back to the picture and averaged ('heat'), or the copies' joints' median taken
    ('median'). The reader reads every frame on its own and errs most on small fast parts; an error that does not
    repeat between the copies shrinks in their average. Confidence: the copies' mean peak times the share of copies
    whose own peak lies within TTA_AGREE input pixels of the joint read (copies that disagree: an unsure reading).

    second (crop reader, its checkpoint, the panel's index): every limb end read again in a square cut round its first
    reading from the same picture (crops.read_crops), its keypoints taken where the crop reader is surer.
    extra: a dict that gets the three confidences of a read-N-times keypoint (conf_agree, conf_mean, conf_heat; T, K)."""
    from .net import context_maps, sample_maps
    T, K = len(crops), model.k
    copies = tta_copies(tta) if tta and tta > 1 else [(1.0, 0.0, 0.0)]
    c0 = SIZE / 2
    hh, ww = crops[0].shape[:2]
    s, (ox, oy) = letterbox((ww, hh))
    w, h = int(round(ww * s)), int(round(hh * s))
    x0, y0 = int(round(ox)), int(round(oy))
    canvases = np.empty((T, SIZE, SIZE, 3), np.uint8)
    for i, crop in enumerate(crops):
        back = np.median(np.concatenate([crop[:8].reshape(-1, 3), crop[:, :8].reshape(-1, 3)]), 0)
        canvases[i] = back.astype(np.uint8)
        canvases[i, y0:y0 + h, x0:x0 + w] = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
    p_in = torch.zeros((T, K, 2), device=dev)                                 # input pixels
    c_in = torch.zeros((T, K), device=dev)
    c_all = {k: torch.zeros((T, K), device=dev) for k in ('conf_agree', 'conf_mean', 'conf_heat')}
    dep = np.zeros((T, K), np.float32)
    cor = None if corners is None else torch.as_tensor(np.asarray(corners, np.float32) * np.array([w / ww, h / hh],
                                                                                                 np.float32), device=dev)
    for rnd in range(2 if model.context else 1):
        prev_p, prev_c = p_in.clone(), c_in.clone()
        for i in range(0, T, bs):
            j = min(T, i + bs)
            heat_sum, pts, cfs, deps = None, [], [], []
            x0_ = normalise(torch.from_numpy(canvases[i:j]).to(dev))
            for s, dx, dy in copies:
                shift = torch.tensor([c0 - s * c0 + dx, c0 - s * c0 + dy], device=dev)     # square -> copy
                x = x0_ if (s, dx, dy) == (1.0, 0.0, 0.0) else _warp_copy(x0_, s, dx, dy)
                ctx = None
                if rnd:
                    tb = torch.arange(i, j, device=dev)
                    ta, tn = (tb - 1).clamp(min=0), (tb + 1).clamp(max=T - 1)
                    ta = torch.where(tb == 0, tb, ta)
                    tn = torch.where(tb == T - 1, tb, tn)
                    pa, pn_ = prev_p[ta], prev_p[tn]
                    if cor is not None:
                        pa = pa + (cor[ta] - cor[tb])[:, None]
                        pn_ = pn_ + (cor[tn] - cor[tb])[:, None]
                    ctx = context_maps(pa * s + shift, prev_c[ta], pn_ * s + shift, prev_c[tn])
                with autocast(dev):
                    out = model(x, ctx) if model.context else model(x)
                heat, dmap = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
                heat = heat.float()
                p, c = decode(heat)
                if dmap is not None:
                    deps.append(sample_maps(dmap.float(), p))
                pts.append((p - shift) / s)                                            # copy -> square
                cfs.append(c)
                if len(copies) > 1 and tta_mode == 'heat':
                    back = heat if (s, dx, dy) == (1.0, 0.0, 0.0) else _warp_back(heat, s, dx, dy)
                    heat_sum = back if heat_sum is None else heat_sum + back
            if len(copies) == 1:
                p, c = pts[0], cfs[0]
                kinds = dict(conf_agree=c, conf_mean=c, conf_heat=c)
            else:
                P = torch.stack(pts)                                                   # (copies, B, K, 2)
                if tta_mode == 'heat':
                    p, c_heat = decode(heat_sum / len(copies))
                else:
                    p, c_heat = P.median(0).values, None
                agree = ((P - p[None]).norm(dim=-1) <= TTA_AGREE).float().mean(0)
                c_mean = torch.stack(cfs).mean(0)
                kinds = dict(conf_agree=c_mean * agree, conf_mean=c_mean,
                             conf_heat=c_heat if c_heat is not None else c_mean)
                c = kinds[f'conf_{TTA_CONF}']
            p_in[i:j], c_in[i:j] = p, c
            for k_, v_ in kinds.items():
                c_all[k_][i:j] = v_
            if deps:
                dep[i:j] = torch.stack(deps).mean(0).cpu().numpy()
    p, c = p_in.cpu().numpy(), c_in.cpu().numpy()
    if extra is not None:
        extra.update({k_: v_.cpu().numpy() for k_, v_ in c_all.items()})
    if second is not None:
        from .crops import read_crops
        p, c, _ = read_crops(_read, second[0], second[1], canvases, p, c, second[2], dev, tta=tta, tta_mode=tta_mode,
                             bs=bs)
    # input pixels -> the images' pixels
    p = (p - np.array([x0, y0])) / np.array([w / ww, h / hh])
    return p, c, dep


def _windows(crops, p, c, win):
    """A reading window per frame: win (w, h) px around the body as the whole panel showed it (the median of the
    sure readings, smoothed over 7 frames), kept inside the panel. Returns (windows, their corners (T, 2))."""
    T = len(crops)
    hh, ww = crops[0].shape[:2]
    wx, wy = min(ww, int(round(win[0]))), min(hh, int(round(win[1])))
    centre = np.full((T, 2), np.nan)
    for t in range(T):
        ok = c[t] >= .3
        if ok.sum() >= 3:
            centre[t] = np.median(p[t, ok], 0)
    have = ~np.isnan(centre[:, 0])
    if not have.any():
        centre[:] = (ww / 2, hh / 2)
    else:
        for d in range(2):
            centre[:, d] = np.interp(np.arange(T), np.where(have)[0], centre[have, d])
    centre = np.stack([np.convolve(np.pad(centre[:, d], 3, mode='edge'), np.ones(7) / 7, mode='valid')
                       for d in range(2)], 1)
    cx = np.clip(np.round(centre[:, 0] - wx / 2), 0, ww - wx).astype(int)
    cy = np.clip(np.round(centre[:, 1] - wy / 2), 0, hh - wy).astype(int)
    return [np.ascontiguousarray(crops[t][cy[t]:cy[t] + wy, cx[t]:cx[t] + wx]) for t in range(T)], np.stack([cx, cy], 1)


@torch.no_grad()
def read_frames(model, frames, views, dev, bs=16, with_depth=False, tta=0, tta_mode='heat', crop_model=None,
                extra=None):
    """Every keypoint of every frame and panel: (uv (T, V, K, 2) in the panel's pixels, conf (T, V, K)).
    with_depth (a reader with a depth map): also each keypoint's depth along the panel's line of sight, from the
    body's middle, in the creature's heights, read where the keypoint was found: (uv, conf, depth (T, V, K)).
    A panel with a reading window (cameras.json read_window: a long strip of a stacked frame, where the whole panel
    would show the body far smaller than the reader learned it) is read whole first, then through a window that
    follows the body at the size the reader learned. tta: every picture read that many times (_read). crop_model
    (the crop reader, its checkpoint): the limb ends read again in zoomed squares (crops.py; not through a window).
    extra: a dict that gets the read-N-times confidences (conf_agree, conf_mean, conf_heat; T, V, K) of the first
    reading (not the crops')."""
    T, V = len(frames), len(views)
    K = model.k
    uv = np.zeros((T, V, K, 2), np.float32)
    conf = np.zeros((T, V, K), np.float32)
    dep = np.zeros((T, V, K), np.float32)
    kinds = {k: np.zeros((T, V, K), np.float32) for k in ('conf_agree', 'conf_mean', 'conf_heat')}
    for v, pv in enumerate(views):
        pn = pv.get('panel')
        if pn is None:                               # a view given by hand as a column of the frame
            crops = [f[:, pv['x0i']:pv['x0i'] + pv['w']] for f in frames]
        else:
            crops = [pn.cut(f) if pn.turn == 0 else np.ascontiguousarray(pn.cut(f)) for f in frames]
        windowed = pn is not None and pn.window is not None
        second = None if crop_model is None or windowed else (crop_model[0], crop_model[1], v)
        got = {}
        p, c, d = _read(model, crops, dev, bs, tta=tta, tta_mode=tta_mode, second=second, extra=got)
        if pn is not None and pn.window is not None:
            wins, corners = _windows(crops, p, c, pn.window)
            got = {}
            p, c, d = _read(model, wins, dev, bs, corners=corners, tta=tta, tta_mode=tta_mode, extra=got)
            p = p + corners[:, None, :]
        # the cut's pixels -> the camera's (its corner at x0, y0, not at the cut's whole pixel)
        if pn is None:
            p[..., 0] += pv['x0i'] - pv['x0']
        else:
            p = pn.to_camera(p)
        uv[:, v] = p
        conf[:, v] = c
        for k_, v_ in got.items():
            kinds[k_][:, v] = v_
        dep[:, v] = d
    if extra is not None:
        extra.update(kinds)
    return (uv, conf, dep) if with_depth else (uv, conf)


def colour(label):
    from video2rig.reader import side_of
    return {'L': (60, 200, 255), 'R': (255, 160, 60)}.get(side_of(label), (80, 255, 80))


def sheet(frames, views, uv, conf, labels, picks, path, extra=None):
    """The readings drawn on frames `picks`, both panels side by side, one row per frame (1.5x for the eye). A
    point the reader is unsure of (peak < 0.3) is drawn hollow. extra: (T, V, K, 2) another reading (the truth),
    drawn as small white crosses."""
    rows = []
    for i in picks:
        tiles = []
        for v, pv in enumerate(views):
            pn = pv.get('panel')
            img = (frames[i][:, pv['x0i']:pv['x0i'] + pv['w']] if pn is None else pn.cut(frames[i])).copy()
            img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

            def at(pts, pn=pn, off=pv['x0i'] - pv['x0']):     # the camera's pixels -> the cut's
                pts = np.asarray(pts, float)
                return pts - np.array([off, 0.0]) if pn is None else pn.from_camera(pts)
            if extra is not None:
                for (u, w) in at(extra[i, v]):
                    c = (int(round(u * 1.5 * 4)), int(round(w * 1.5 * 4)))
                    cv2.drawMarker(img, (c[0] >> 2, c[1] >> 2), (255, 255, 255), cv2.MARKER_CROSS, 7, 1)
            for k, (lb, (u, w)) in enumerate(zip(labels, at(uv[i, v]))):
                c = (int(round(u * 1.5 * 4)), int(round(w * 1.5 * 4)))
                cv2.circle(img, c, 16, colour(lb), -1 if conf[i, v, k] >= .3 else 2, cv2.LINE_AA, shift=2)
            cv2.putText(img, f'{i}', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(img)
        hmax = max(t.shape[0] for t in tiles)                     # a stacked frame's panels differ in height
        tiles = [t if t.shape[0] == hmax else cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT)
                 for t in tiles]
        rows.append(np.concatenate(tiles, 1))
    out = np.concatenate(rows, 0)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def main():
    ap = argparse.ArgumentParser(prog='video2rig.reader.read')
    ap.add_argument('project')
    ap.add_argument('clip')
    ap.add_argument('--reader', default='', help='the reader (default <project>/reader/reader.pt)')
    ap.add_argument('--sheet', default='0,20,36,52,75,100')
    ap.add_argument('--tag', default='')
    ap.add_argument('--tta', type=int, default=None, help='read every picture this many times, shifted and scaled a '
                    'little, and average (0: once; default road.TTA)')
    ap.add_argument('--tta-mode', default='heat', choices=['heat', 'median'])
    a = ap.parse_args()
    t0 = time.time()
    dev = pick_device()
    pr = Project(a.project)
    model, ck = load_reader(Path(a.reader) if a.reader else pr.root / 'reader' / 'reader.pt', dev)
    labels = [kp[0] for kp in ck['keypoints']]
    frames = clipmod.read_frames(pr.clip_file(a.clip))
    cams = json.loads((pr.start_dir / 'cameras.json').read_text(encoding='utf-8'))
    views = panels(frames, cams)
    if a.tta is None:
        from .road import TTA
        a.tta = int(pr.cfg.get('reader_tta', TTA))
    uv, conf = read_frames(model, frames, views, dev, tta=a.tta, tta_mode=a.tta_mode)
    work = pr.work(a.clip)
    np.savez_compressed(work / f'reader{a.tag}.npz', uv=uv, conf=conf, labels=np.array(labels),
                        views=np.array([v['name'] for v in views]), keypoints=json.dumps(ck['keypoints']),
                        tta=a.tta, tta_mode=a.tta_mode)
    picks = [int(i) for i in a.sheet.split(',') if i and int(i) < len(frames)]
    path = sheet(frames, views, uv, conf, labels, picks, work / f'reader-sheet{a.tag}.jpg')
    print(f'READ {a.clip}: {len(frames)} frames x {len(views)} panels in {time.time() - t0:.1f} s; median confidence '
          f'{float(np.median(conf)):.2f} -> {work / f"reader{a.tag}.npz"}, {path}', flush=True)


if __name__ == '__main__':
    main()
