"""The reader's road through one clip, as `python -m video2rig run` takes it when the creature has a reader
(<project>/reader/reader.pt, made once by `python -m video2rig.reader.prepare`): every joint read in every frame
and panel, the left/right check, the rig fitted, the motion file for blender/import_motion.py. About half a minute.

The road is two steps, and the ground-truth bench (dev/bench/suite.py) calls the same two, so it measures exactly
what ships:
    read_clip()     the reading, on the graphics card; cached in work/<clip>/reader<tag>.npz with the fingerprints
                    of the reader and of the clip (a new reader or a new clip is read again)
    fit_reading()   creature_for -> fix_twins -> panels_agree -> fit_hidden, on the processor
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..common import write_json


def reader_path(project):
    p = project.root / 'reader' / 'reader.pt'
    return p if p.exists() else None


def fingerprint(path):
    """The first 12 hex digits of a file's sha1."""
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()[:12]


TTA = 8          # read every picture this many times (read._read; 0: once); creature.json "reader_tta" overrides.
                 # 8 copies: the arms' frame-to-frame reading error -34..-47% on 30 clips, the reading step about 2.8x
                 # as long (dev/INPUT-RUN.md, stage 1); "reader_tta": 0 reads as before
CROPS = False    # read the limb ends again in zoomed squares with reader/reader-crops.pt when the creature has one
                 # (crops.py); creature.json "reader_crops" overrides


def crop_reader_path(project):
    p = project.root / 'reader' / 'reader-crops.pt'
    return p if p.exists() else None


def read_clip(project, clip, reader=None, cpu=False, tag='', log=print, cache=True, tta=None, crops=None):
    """Every keypoint of the clip in every frame and panel, read by `reader` (default the project's own):
    (uv (T, V, K, 2) in each panel's pixels, conf (T, V, K), keypoints, the panels' names). tta: every picture read
    that many times and averaged (default the creature's "reader_tta", else TTA). crops: the limb ends read again by
    the crop reader (default the creature's "reader_crops", else CROPS; a path: that crop reader)."""
    path = Path(reader) if reader else reader_path(project)
    if path is None or not path.exists():
        raise SystemExit(f'no reader for {project.root.name}: run python -m video2rig.reader.prepare first')
    tta = int(project.cfg.get('reader_tta', TTA) if tta is None else tta)
    crops = project.cfg.get('reader_crops', CROPS) if crops is None else crops
    cpath = Path(crops) if isinstance(crops, (str, Path)) else crop_reader_path(project) if crops else None
    rid, cid = fingerprint(path), fingerprint(project.clip_file(clip))
    xid = fingerprint(cpath) if cpath is not None else ''
    out = project.work(clip) / f'reader{tag}.npz'
    if cache and out.exists():
        d = np.load(out)
        read_tta = int(d['tta']) if 'tta' in d.files else 0
        read_xid = str(d['crops_id']) if 'crops_id' in d.files else ''
        same = 'reader_id' in d.files and str(d['reader_id']) == rid and str(d['clip_id']) == cid
        if same and read_tta == tta and read_xid == xid:
            return d['uv'], d['conf'], [tuple(k) for k in json.loads(str(d['keypoints']))], [str(v) for v in d['views']]
    import torch
    from .. import clip as clipmod
    from .read import load_reader, panels, read_frames
    t0 = time.time()
    from .device import pick_device
    dev = pick_device(cpu=cpu)
    model, ck = load_reader(path, dev)
    crop_model = load_reader(cpath, dev) if cpath is not None else None
    kps = [tuple(k) for k in ck['keypoints']]
    frames = clipmod.read_frames(project.clip_file(clip))
    cams = json.loads((project.start_dir / 'cameras.json').read_text(encoding='utf-8'))
    pv = panels(frames, cams)
    uv, conf = read_frames(model, frames, pv, dev, tta=tta, crop_model=crop_model)
    del model, crop_model
    names = [v['name'] for v in pv]
    np.savez_compressed(out, uv=uv, conf=conf, labels=np.array([k[0] for k in kps]), views=np.array(names),
                        keypoints=json.dumps(kps), reader_id=rid, clip_id=cid, reader=str(path), tta=tta, crops_id=xid)
    log(f'read: {len(frames)} frames x {len(pv)} panels' + (f', every picture {tta} times' if tta > 1 else '') +
        (', the limb ends again in zoomed squares' if cpath is not None else '') +
        f', median confidence {float(np.median(conf)):.2f} ({time.time() - t0:.0f} s)')
    return uv, conf, kps, names


def fit_reading(project, clip, cfg, uv, conf, kps, view_names, s=None, log=print, dep=None):
    """The rig fitted to a reading, as `run` does it: the clip's creature (a jaw the reader sees is fitted too),
    the panels put in the creature's order, the left/right check, the panels made to agree, the fit.
    Returns dict(creature, params, report, uv, conf, swaps); uv and conf as the fit used them.
    dep (T, V, K): a depth reader's depths (stage R), fitted as one more measurement; a point the left/right check
    hands to its twin keeps no depth (it was read for the other limb).

    The fit runs with one processor thread for numpy's linear algebra: with the machine's default (16 here) the
    sums come out in another order, a fit on a knife's edge takes another branch (the alien's pirouette, combo01
    frames 97-148: up to 175 deg apart), so a clip gave different motions on different computers; and the small
    matrices of the fit ran 2.4 times slower on 16 threads (111 s against 46 s for that clip)."""
    from threadpoolctl import threadpool_limits
    from .fit import SETTINGS, creature_for, fit_hidden, fix_twins, panels_agree
    s = SETTINGS if s is None else s
    with threadpool_limits(limits=1):
        cr, narrow = creature_for(project, cfg, kps)
        names = [v['name'] for v in cr.views(clip)]
        order = [list(view_names).index(n) for n in names]
        uv, conf = np.asarray(uv)[:, order].copy(), np.asarray(conf)[:, order].copy()
        before = uv.copy()
        uv, swaps = fix_twins(cr, clip, uv, conf, kps, log=log, s=s)
        if dep is not None:
            dep = np.asarray(dep, float)[:, order].copy()
            T = min(len(dep), len(uv))
            moved = np.linalg.norm(uv[:T] - before[:T], axis=-1) > 1e-6
            dep[:T][moved] = np.nan
        uv, conf = panels_agree(cr, clip, uv, conf, kps, s=s, log=log)
        P, rep = fit_hidden(cr, clip, uv, conf, kps, s=s, log=log, narrow=narrow, dep=dep)
    return dict(creature=cr, params=P, report=rep, uv=uv, conf=conf, swaps=swaps)


def run(project, clip, cfg, tag='', log=print, cpu=False):
    uv, conf, kps, names = read_clip(project, clip, cpu=cpu, tag=tag, log=log)
    res = fit_reading(project, clip, cfg, uv, conf, kps, names, log=log)
    cr, P = res['creature'], res['params']
    cr.save(clip, P, tag, {'reader': True})
    write_json(project.work(clip) / f'fit{tag}-report.json', res['report'])
    out = cr.export_motion(clip, P, project.out() / f'{clip}{tag}.npz')
    return cr, P, out
