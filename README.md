# video2rig

Turn an AI video of **your own 3D model** into an animation of **your own rig** in Blender.

![A boss's start picture animated by an AI video generator, and the same move on its own rig](docs/images/fit-over-video.jpg)

**New here? Start with [TUTORIAL.md](TUTORIAL.md):** every step in order, from your model to an animation in your
game, with pictures of what the start picture and a good video look like and how to write the video's prompt.

The tool renders a start picture of your rigged model, an image-to-video generator animates it from that first frame
("he lifts the log onto his shoulder", "the giant staggers back"), and video2rig rebuilds that motion on your
skeleton, frame by frame, with no hand-placed points. The result is an ordinary Blender Action you can edit, retime
and export (glTF, FBX).

The video generator is your choice: anything that makes a video **from a given first frame** and keeps it. We made
and tested every clip with Seedance (ByteDance; on Dreamina, Topview or the BytePlus API), so the examples here are
Seedance's; Kling, Veo, Runway, Hailuo and others with a first-frame mode work the same way in principle.

How: once per creature, the tool trains a small joint reader on renders of your model in random poses through the
start picture's own cameras. After that, every clip is read joint by joint, in every frame and every view, and the
rig is fitted to those joints (docs/HOW-IT-WORKS.md).

- Your own armature: knees and elbows are found by their names (Mixamo, Rigify, Unreal and common names) or set in
  `creature.json`. Tested on a Mixamo person and on custom rigs (a troll, wolves, a centaur, a wyvern, the boss above).
- No skeleton yet? The auto-rig makes one (UniRig + checks on the mesh) and skins the model.
- Props come along: a log, a stone, a weapon is a rigid object tracked together with the character.
- Two cameras in one frame (front | side) when depth matters; one camera works too.
- The tool runs locally on your own computer: no cloud, no account, no API key (only the video generator is online;
  the reader's backbone and the tracker's weights are downloaded once).

Made for the Can It Code? channel (YouTube). Tested on a giant troll (two-camera clips: stagger, roar, throw, combo,
walk, run), wolves (walk, run, bite, howl), the boss of the release video (idle, roar, combo, stagger, gore, death)
and a man lifting a log onto his shoulder and a stone to his chest (one camera, with the prop: through the camera both
match; from the side the log rests on his shoulder, but the stone ends up beside his hip. Film props held in front of
the body with two views).

## What you need

| | |
| --- | --- |
| Blender | tested on 5.1 (recent 4.x should work): small scripts run inside it, with or without its window; it also renders the reader's training pictures (in the background) |
| Python | 3.10-3.12 with `numpy`, `scipy`, `opencv-python`, `threadpoolctl`, `transformers` 5 (the joint reader), `networkx` and `scikit-image` (the auto-rig) |
| PyTorch | with CUDA (NVIDIA), or the default macOS build on an Apple-silicon Mac (Metal), for the joint reader (Meta's DINOv2-small, Apache 2.0, through Hugging Face `transformers`) and for the older road's point tracker (Google DeepMind's BootsTAPIR, Apache 2.0, bundled in `third_party/`) |
| ffmpeg | on PATH: the review film, and the films and sheets the Blender scripts make (`render_actions.py`, the auto-rig's pose sheet). The review also takes `V2R_FFMPEG`, the full path to the ffmpeg program |
| Graphics card | **NVIDIA, RTX 30 series or newer**: the reader trains and reads in bfloat16, which these cards do natively (older cards are untested). Tested on an RTX 5070 Ti Laptop and an RTX 4070, 12 GB each. Training a reader needs the card. Reading a clip takes seconds on the card; `run --cpu` (or a computer without CUDA) reads on the processor instead, about 0.8 s per panel picture on a 24-thread laptop processor (a 5 s two-panel clip: about 3 minutes). The older road's tracker peaks at about 6.5 GB for a 5 s clip at full size (a smaller card tracks at 512 px automatically) and also runs on the processor, much slower. **Apple-silicon Mac**: the reader trains and reads on the chip's own GPU through Metal (MPS), in bfloat16 as well; measured on an M4 Pro, about 0.65 s per training step at batch 8 (a 9000-step reader: a few hours, against 40 minutes on the RTX 4070) and a 1 s two-panel clip read in 12 s. The one op Metal lacks runs on the processor by itself. The device is picked in the order CUDA, Metal, processor (`video2rig/reader/device.py`) |
| Disk | about 85 MB for DINOv2-small (downloaded once from the Hugging Face hub when the first reader trains; reading fetches its small config file, so the first read needs the internet too; both stay in `~/.cache/huggingface`), a reader and its training pictures per creature (`<project>/reader/`), 220 MB for the tracker's weights (older road, on first use) |
| A video generator | image-to-video from a first frame, 720p (two views) or 480p (one view), 4-8 s. Tested: Seedance on Dreamina (CapCut), Topview or the BytePlus API |

Install (Windows PowerShell; the same with `pip` elsewhere):

```powershell
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128   # PyTorch with CUDA
# macOS (Apple silicon): python -m pip install torch   (the default build runs the reader on Metal)
winget install Gyan.FFmpeg                                                      # or any ffmpeg on PATH
```

The joint reader runs in the same Python as the tool, so PyTorch must be installed there. Only the older road's
tracker can use PyTorch from another Python (a separate venv):
`$env:V2R_TORCH_PYTHON = "C:\path\to\venv\Scripts\python.exe"`.

## The whole thing in six steps

```powershell
# 0. A folder for the creature with its settings file ('{}' is enough; docs/CONFIG.md)
mkdir my_troll; Set-Content -Encoding ascii my_troll/creature.json '{}'
# 1. Blender: export the rig (armature + skinned meshes + props) in the pose you will start from
blender model.blend --background --python blender/export_rig.py -- --out my_troll/rig
# 2. Blender: render the start picture for the video generator (front | side) and its cameras
blender model.blend --background --python blender/render_start.py -- --out my_troll/start
# 3. Once per creature: train its joint reader on renders of it through those cameras (Blender renders in the
#    background, then the graphics card trains; about 40 minutes on an RTX 4070 with --steps 9000)
python -m video2rig.reader.prepare my_troll --blend model.blend --steps 9000
# 4. The video generator: my_troll/start/start.png as the FIRST FRAME, write the move, save the clip as
#    my_troll/clips/stagger.mp4      (how to write prompts that work: docs/VIDEO.md)
# 5. The tool: one command
python -m video2rig run my_troll stagger
# 6. Blender: the motion onto the rig as an Action
blender model.blend --background --python blender/import_motion.py -- --motion my_troll/out/stagger.npz --save my_troll/stagger.blend
```

**Do not skip step 3.** `run` uses the reader when `my_troll/reader/reader.pt` exists. Without it, it takes the
older road (a point tracker and a whole-clip solve), which is far less accurate: on the bench's wolf gallop 21 degrees
per bone against about 3 with the reader. A creature needs step 3 again only when its model, rig or start picture
changes; one reader reads every move of it.

Step 5 prints what it does and how long it took (about a minute for a 5 s clip), and writes
`my_troll/out/stagger.npz` plus a quality report. Before it, `python -m video2rig check my_troll stagger` says in
seconds whether the generator drew the same motion in both views (FAIL: generate the clip again). After it,
`python -m video2rig review my_troll stagger` makes a film of the video next to the fit (through the same camera)
and from a three-quarter view, to check it before you use it.

Read next:
- [TUTORIAL.md](TUTORIAL.md): the whole road, in order, with pictures.
- [docs/WORKFLOW.md](docs/WORKFLOW.md): every step in detail, the project folder, what to check.
- [docs/VIDEO.md](docs/VIDEO.md): start pictures and prompts that give clips the tool can read.
- [docs/CONFIG.md](docs/CONFIG.md): `creature.json` (joints, limits, smoothing) and the reader fit's settings, when
  the defaults are not enough.
- [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md): the method (the joint reader), what it can and cannot see.
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md): graphics card memory, timing, wrong legs, sliding feet.

## Licence

Copyright (C) 2026 Can It Code. This tool is free software under the **GNU Affero General Public
License v3.0** ([LICENSE](LICENSE)). In plain words:

- **Use it for anything**, commercial games included. The animations you make with it are yours.
- **Change it and share it**, keeping this licence and the copyright notice.
- **Nobody can turn it into a closed paid tool.** Whoever shares a changed version, or offers it to other people as a
  program, an app or a website (paid or free), must publish their complete source code under this same licence, free
  for everyone to use.

The point tracker in `third_party/tapnet` and `third_party/einshape` is Google DeepMind's code under Apache 2.0
(licence files next to it; local changes are listed in `third_party/README.md`); its weights are Apache 2.0 too.
The joint reader is built on Meta's DINOv2-small (Apache 2.0), downloaded from the Hugging Face hub on first use.
UniRig (the auto-rig's skeleton model) is VAST-AI-Research's, MIT, installed separately.
