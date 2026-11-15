"""Prepare a creature's joint reader in one go, timed: random poses, their renders (Blender in the background, no
window), the training. Run once per creature; every clip of it is read afterwards in seconds.

    python -m video2rig.reader.prepare <project> --blend model.blend [--poses 3000] [--minutes 8] [--copies 16]
        [--blender "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"]

The .blend is the creature's own file (the one export_rig.py and render_start.py ran on), in its start pose.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..common import TOOL, Project, write_json
from .net import SIZE

DEFAULT_BLENDER = r'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe'


def find_blender(given):
    for c in (given, os.environ.get('V2R_BLENDER'), DEFAULT_BLENDER, shutil.which('blender')):
        if c and Path(c).exists():
            return str(c)
    raise SystemExit('Blender not found: pass --blender or set V2R_BLENDER')


def run(cmd, log_path):
    """A child process with no window of its own (Windows), its output to a log."""
    flags = 0x08000000 if os.name == 'nt' else 0                    # CREATE_NO_WINDOW
    with open(log_path, 'w', encoding='utf-8') as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, creationflags=flags, cwd=str(TOOL))
    if p.returncode != 0:
        raise SystemExit(f'failed ({p.returncode}): {" ".join(map(str, cmd))}\n  see {log_path}')


def main():
    ap = argparse.ArgumentParser(prog='video2rig.reader.prepare')
    ap.add_argument('project')
    ap.add_argument('--blend', required=True)
    ap.add_argument('--poses', type=int, default=3000)
    ap.add_argument('--minutes', type=float, default=8.0, help='training time')
    ap.add_argument('--steps', type=int, default=0, help='train a step count instead of a time (a hot card)')
    ap.add_argument('--init', default='', help="another creature's reader.pt to start the training from")
    ap.add_argument('--backbone', default='', help='the picture model the reader is built on (train.py --backbone): '
                    'facebook/dinov2-base reads closer than the default facebook/dinov2-small, its training takes '
                    'about 1.4 times as long')
    ap.add_argument('--copies', type=int, default=16)
    ap.add_argument('--samples', type=int, default=8)
    ap.add_argument('--blender', default='')
    ap.add_argument('--keep', action='store_true', help='keep earlier renders (poses and pictures) if present')
    ap.add_argument('--workers', type=int, default=6, help='processes for the hidden-joint check')
    ap.add_argument('--skip-train', action='store_true',
                    help='poses and pictures only (render the next creature while another one trains; then --keep)')
    ap.add_argument('--out', default='reader/reader.pt', help='where the reader goes, in the project (an experiment '
                    'next to the reader in use: reader/reader-try.pt)')
    a = ap.parse_args()
    t0 = time.time()
    clock = {}
    pr = Project(a.project)
    rd = pr.root / 'reader'
    rd.mkdir(exist_ok=True)
    py = sys.executable
    renders = rd / 'renders'
    if not (a.keep and (renders / 'labels.npz').exists()):
        t = time.time()
        run([py, '-m', 'video2rig.reader.poses', str(pr.root), '--n', str(a.poses)], rd / 'poses.log')
        clock['poses_s'] = round(time.time() - t, 1)
        t = time.time()
        if renders.exists():
            shutil.rmtree(renders)
        renders.mkdir()
        # the pictures on the card (Blender) and the hidden joints on the processor, side by side
        flags = 0x08000000 if os.name == 'nt' else 0
        log = open(renders / 'render.log', 'w', encoding='utf-8')
        blender = subprocess.Popen([find_blender(a.blender), str(Path(a.blend).resolve()), '--background',
                                    '--factory-startup', '--python', str(TOOL / 'blender' / 'render_poses.py'), '--',
                                    '--poses', str(rd / 'poses.npz'), '--cameras', str(pr.start_dir / 'cameras.json'),
                                    '--out', str(renders), '--copies', str(a.copies), '--samples', str(a.samples),
                                    '--fit', str(SIZE), '--project', str(pr.root)],
                                   stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        from .visible import compute
        compute(pr.root, workers=a.workers, log=lambda m: print(m, flush=True))
        clock['visible_s'] = round(time.time() - t, 1)
        if blender.wait() != 0:
            raise SystemExit(f'the render failed: see {renders / "render.log"}')
        log.close()
        clock['render_s'] = round(time.time() - t, 1)
    if a.skip_train:
        print(f'PREPARE {pr.name}: pictures ready, training skipped: {clock}', flush=True)
        return
    t = time.time()
    run([py, '-m', 'video2rig.reader.train', str(pr.root), '--minutes', str(a.minutes), '--steps', str(a.steps),
         '--out', a.out] + (['--init', str(Path(a.init).resolve())] if a.init else []) +
        (['--backbone', a.backbone] if a.backbone else []),
        rd / f'train{Path(a.out).stem.removeprefix("reader")}.log')
    clock['train_s'] = round(time.time() - t, 1)
    clock['total_min'] = round((time.time() - t0) / 60, 2)
    write_json(rd / 'prepare-clock.json', clock)
    print(f'PREPARE {pr.name}: {clock}', flush=True)


if __name__ == '__main__':
    main()
