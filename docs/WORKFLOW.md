# Workflow, step by step

## 0. The project folder

One folder per creature (or per creature + prop set). The tool keeps everything there:

```
my_troll/
  creature.json        settings (`{}` is enough; examples/creature.json; every field is optional)
  rig/                 rig.json, mesh.npz, texture_*.png       <- step 1
  start/               start.png, panel-*.png, cameras.json    <- step 2
  reader/              reader.pt (the joint reader), its training poses, pictures and logs  <- step 2b
  clips/<clip>.mp4     your generated clips                     <- step 3
  work/<clip>/         cache: silhouettes, timing, aligned cameras, the reading, fits, review film, report
  out/<clip>.npz       the motion for Blender                  <- step 4
```

`work/` can be deleted at any time; it is rebuilt. `reader/` cannot: it takes most of an hour to make again.

`creature.json` must exist before the tool's first command. From PowerShell, without a byte order mark (the tool
cannot read a file that starts with one): `Set-Content -Encoding ascii my_troll/creature.json '{}'`.

## 0b. A model without bones: the auto-rig

A model straight from a generator (Tripo, Meshy, a sculpt) has no skeleton. The auto-rig makes one, skins the mesh
and saves a rigged `.blend` for step 1. Three commands, about five minutes:

```powershell
blender --background --factory-startup --python blender/autorig_prepare.py -- --model troll.glb --out my_troll/autorig
python -m video2rig.autorig.unirig my_troll/autorig
blender --background --factory-startup --python blender/autorig_build.py -- --work my_troll/autorig
```

- `autorig_prepare.py` saves the model as `model.blend` and a watertight copy (`shell.npz`) to measure.
- `video2rig.autorig.unirig` draws the skeleton with UniRig (below) and checks it on the mesh: every place the model
  stands on the floor gets a leg, mirrored limbs come out exactly mirrored (or at least equally long when the model
  is posed unevenly), fingers and toes fold into the hand and foot, wings are told from arms by their flat
  membrane, and the bones get the names the tool reads (`Thigh/Shin`, `UpperArm/Forearm`, `.L/.R`). Without UniRig
  it falls back to the skeleton from the mesh's centre lines (`python -m video2rig.autorig.skeleton`).
- A creature standing on six or more legs (a spider, a crab, an insect) skips UniRig: each leg is traced from the
  body to its tip and gets four bones (`Coxa1.L .. Tarsus1.L`, pairs numbered front to back).
- A mouth modelled open gets a `Jaw` bone, from the mouth's corner to the chin: the reader then reads the jaw's tip
  and the fit opens the mouth when the video does (a bite, a roar). A mouth modelled shut gets none: generate the
  model with its jaws a little apart if it should bite.
- A robed or hooded figure made of thin one-sided cloth with nothing under it (a ghost) needs a lining first: a
  dark body under the robe and a head in the hood, as closed meshes in the same file. Without them UniRig's head
  and legs lie in empty air and are dropped as "outside the skin". With them the legs under the robe swing its
  cloth, and cloth that reaches the floor is not taken for feet.
- `autorig_build.py` skins the mesh (Blender's bone heat, loose parts like claws and swords follow their limb) and
  renders `poses-sheet.png`: the rig in four test poses. Look at it before going on: tearing or a limb dragging the
  body means bad weights.
- Then step 1 with `my_troll/autorig/rigged.blend` as the model.
- The model should stand along X or Y (facing one of them). A model standing at an angle confuses the left/right
  test: rotate it first.

**Installing UniRig** (once; NVIDIA card, about 6 GB of downloads): clone
[UniRig](https://github.com/VAST-AI-Research/UniRig) (MIT) next to this folder (`..\unirig`) or anywhere and set
`V2R_UNIRIG` to it, then in its folder make a Python 3.11 venv `.venv` and install its requirements, `spconv` for your
CUDA and the PyTorch Geometric wheels for your torch (UniRig's README lists them). On Windows two edits make the
skeleton model run without flash-attention (only UniRig's skin model needs it, which this tool does not use):
`src/model/parse.py` imports `unirig_skin` only inside `get_model`, and
`configs/model/unirig_ar_350m_1024_81920_float32.yaml` sets `_attn_implementation: sdpa`.

## 1. Export the rig (Blender)

Pose the model the way it will stand in the start frame (a relaxed stance with the arms off the body is best; a
T-pose works but looks odd to the generator). Props (a log, a stone, a sword) are separate mesh objects placed where they
are at the start.

```powershell
blender model.blend --background --python blender/export_rig.py -- --out my_troll/rig --props Log
```

- `--armature NAME` when the file has more than one armature; `--props A,B` or put the props in a collection called
  `props`.
- Exported in the armature's own space, so a Mixamo FBX at 0.01 scale or a rotated armature is fine.
- Apply modifiers other than the Armature first (mirror, subdivision, displace): the exporter reads the base mesh.
- Vertices without bone weights follow the first bone (a warning is printed).

## 2. Render the start frame (Blender)

```powershell
blender model.blend --background --python blender/render_start.py -- --out my_troll/start
```

- Default: one 1920x1080 frame with two orthographic views, front | side, same scale, floor visible. Two views make
  depth visible (is the hand in front of the body or behind it?). Single view: `--views side` or
  `--views three-quarter`.
- The creature's facing is read from its feet; `--facing -Y` if it guesses wrong. `--side right` to see the right
  flank in the side view.
- `--margin 2.2` for big moves (jumps, raised arms): more room around the body. Nothing may leave the frame in the
  video.
- A long, low creature (a belly slider, a snake, a lizard, a seal-like boss): `--views side,top --widths 1280,640
  --margin-wide 1.45 --lead 0.08 --facing +X`. The side view sees its whole length and how high it rears, the view
  from above sees its arms reach and its body bend; `--margin-wide` frames it by its length, `--lead` leaves room in
  front for a charge. Pass `--facing` when its "feet" are claws that splay sideways (the guess from the feet goes
  diagonal).
- Rendered your start frame some other way? Write its camera with
  `blender scene.blend --background --python blender/export_camera.py -- --out my_troll/start --cameras Camera`.
  A split frame of several cameras: `--cameras CamSide,CamTop` (left to right), and `--widths 1280,640` when the
  panels differ in width.

Do not change the pose, the model or the cameras after this: the start frame is the one moment the tool knows
exactly.

## 2b. Train the joint reader (once per creature)

The reader is a small network that finds every joint of this one creature in a frame of a clip. It learns from
renders of your own model, so every label is exact and nothing is drawn by hand:

```powershell
python -m video2rig.reader.prepare my_troll --blend model.blend --steps 9000
```

What it does, in one go:

1. **Poses** (about 10 s): 3000 random poses of the rig within its joint ranges, the body turned, shifted or placed
   anywhere the start frame leaves room; in about half of them the bones move during the frame (motion blur), some
   with the whole body lunging or spinning. No animation goes in: the reader learns the creature, not a move, so one
   reader reads every move of it.
2. **Pictures** (about 3 minutes; 2.7 for 3000 poses x 2 views on an RTX 5070 Ti Laptop): Blender renders every pose
   through the start frame's own cameras, in the background, without a window. Beside it, on the processor, the
   tool works out which joints each picture shows: a joint hidden behind the body is taught as "not here".
3. **Training** (on the graphics card): DINOv2-small (Meta) with a small decoder that draws one heat map per joint.
   Every picture is damaged the ways a generated frame differs from a render (re-framed, other colours and light,
   blur, a fast limb smeared into a streak, noise), but never mirrored: left and right are what it must tell apart.
   The last 5% of the poses are held out; their error in pixels is printed and saved in `reader/reader-train.json`
   (`held_out`, per view): this is the number that says whether the reader can be trusted (the readers in use:
   2-4 px).

At the end it prints `PREPARE my_troll: {...}` with the time of each part (also in `reader/prepare-clock.json`).
The logs are in `reader/` (`poses.log`, `renders/render.log`, `train.log`).

How long to train: longer is better. On the wolf's bench clips, 1538 steps gave 3.21 degrees per bone, 4614 steps
2.83, 9228 steps 2.51 (a perfect reader: 1.31); after about 9000 steps more steps added nothing. The readers in use
were trained to 9000 steps: 35-46 minutes on an RTX 4070. The default, 8 minutes, gives 600-1500 steps on a laptop.

| Option | Default | What it does |
| --- | --- | --- |
| `--blend` | (required) | the creature's own .blend in its start pose: the file `export_rig.py` and `render_start.py` ran on (`my_troll/autorig/rigged.blend` after the auto-rig) |
| `--steps N` | 0 (off) | train N steps whatever the time. Use it: a laptop card that runs hot slows down to a fraction of its clock, and a time budget then gets a fifth of the steps |
| `--minutes` | 8 | training time, when `--steps` is not given |
| `--poses` | 3000 | how many random poses |
| `--init` | | another creature's `reader.pt` to start from (it already knows what a joint looks like in these renders) |
| `--keep` | | keep the poses and pictures already in `reader/` and only train (to train longer, or after `--skip-train`) |
| `--skip-train` | | poses and pictures only: render the next creature while another one trains, then run again with `--keep` |
| `--out` | `reader/reader.pt` | where the reader goes, in the project. `run` reads only `reader/reader.pt`; an experiment can go beside it (`reader/reader-try.pt`) |
| `--copies` | 16 | creatures per Blender render call (one Blender start pays for many pictures) |
| `--samples` | 8 | render samples per picture |
| `--workers` | 6 | processes for the hidden-joint check |
| `--blender` | | the Blender program; else `V2R_BLENDER`, else `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`, else `blender` on PATH |

Train again when the model, the rig or the start frame's cameras change: the reader learned exactly those.

## 3. Make the clip (the video generator)

In your image-to-video generator, give `start/start.png` as the first frame. Prompts that work, resolution, length:
[VIDEO.md](VIDEO.md).
Save the clip as `clips/<name>.mp4` (any name; that is the clip's name for the tool). Check it before using it: both
panels move together, the body never leaves the frame, nothing appears that is not there (sticks, shockwaves).

Then let the tool check it (seconds, with the reader):

```powershell
python -m video2rig check my_troll stagger
```

It reads the clip and measures how far the two views disagree about where the body and the limbs are, as a share
of the creature's height. **FAIL** (the body over 1.5%; clean clips stay under about 1.2%): the generator drew the move
differently in each view, and no fit can make one true motion out of two; generate the clip again. **WARN** (the
limbs over 2%, or a first frame that is not your start image): watch the limbs in the review film. The reading is
cached, so `run` does not read the clip again.

## 4. Run the tool

```powershell
python -m video2rig run my_troll stagger
```

With `reader/reader.pt` in the project, `run` takes the reader's road. What happens, with the time for a 4-5 s clip:

| Stage | What | Time |
| --- | --- | --- |
| prepare | frames, silhouettes, the clip's real frame rate, camera alignment on frame 0 | 0.5-1 min |
| read | every joint in every frame and view, with how sure the reader is (graphics card; cached in `work/<clip>/reader.npz`, read again only when the reader or the clip changes) | about 7 s |
| in step | a front panel the generator played ahead of or behind the side panel is matched onto it frame by frame: its readings, pictures and outlines (`reader/sync.py`; the log says by how many frames) | seconds |
| fit | the rig fitted to the joints of the whole clip at once; nothing added after it | about a minute (one processor thread) |
| export | `out/<clip>.npz` + `work/<clip>/report.json` | seconds |

Options: `--cpu` reads on the processor (no graphics card: a few minutes instead of seconds), `--tag NAME` adds a
suffix to the fit and motion files (write `--tag=-try` when it starts with a dash), `--old` takes the older road
below even when the creature has a reader.

The log says how sure the reading was and how far the fitted joints miss the readings (mean px of all readings and
median px of the sure ones, per view; also in `work/<clip>/fit-report.json`). The report
gives the outline match with the video per view (0.9+ is good), the worst frames, the largest twist about a bone and
the largest joint jump in one frame. Numbers only point where to look. Look at the result:

```powershell
python -m video2rig review my_troll stagger      # work/stagger/review.mp4
```

The review film shows the video, the fit through the same camera, both outlines on top of each other, and the fit
from a three-quarter view the video never had: that is where a wrong depth or a hidden limb shows. A fit made with
`--tag` is reviewed with the same tag: `python -m video2rig review my_troll stagger --tag=-try`.

To fit the same reading again with another setting (about a minute; the reading is reused):
`python -m video2rig.reader.fit my_troll stagger --set KEY=VALUE` ([CONFIG.md](CONFIG.md), "The reader fit's
settings").

### The older road (`--old`)

Before the reader, the tool followed points instead of joints: `python -m video2rig run my_troll stagger --old`
(also what `run` does when the creature has no `reader/reader.pt`; the log then says so). It needs no training,
but it is far less accurate (on the bench's wolf gallop 21 degrees per bone against about 3 with the reader) and
slower:

| Stage | What | Time (12 GB laptop card) |
| --- | --- | --- |
| track | points on the start frame followed through the clip (graphics card; a game or a game engine on the same card makes it 10x slower) | 1-3 min |
| solve | the rig fitted to the whole clip at once | 5-15 min (processor) |

Its options: `--refine` (a second tracking pass seeded on later frames from the first fit: a limb that hides for a
while gets fresh points; +5-10 min), `--cpu` (track on the processor, much slower), `--retrack`, `--set KEY=VALUE`
(one `solve.*` setting for this run, [CONFIG.md](CONFIG.md)). `python -m video2rig track` and `solve` run its two
stages alone.

## 5. Into Blender

```powershell
blender model.blend --background --python blender/import_motion.py -- --motion my_troll/out/stagger.npz --save my_troll/stagger.blend
```

Or open `model.blend`, load `blender/import_motion.py` in the Scripting tab, set `MOTION = r"...\stagger.npz"` and
run it. You get an Action named after the clip: quaternion keys at the clip's real times (the scene's frame rate),
the root's location, and the props moving with their own keys. From here it is a normal Blender animation: trim it,
loop it, retime phases, export it.

## Tips from our tests

- Feet and contacts: a generated character's standing foot can slide a few centimetres; pin feet in your engine or
  with a few IK keys if it matters.
- Long holds and slow starts inside a clip are best fixed by retiming the Action, not by generating again.
- Humans with one camera work when the arms stay mostly visible; an arm hidden behind the body for long is guessed.
  Use two views (front | side) for anything that crosses the body.
- A prop seen by one camera has no measured depth. The tool keeps the body out of it and rests a lifted prop on the
  body: right for a log on the shoulder, wrong for a stone held in front of the chest (it ended up beside the hip).
  Look at the review film's three-quarter view, and film such moves with two views.
- Thin fast parts (legs of a small animal) need 480p or better and no motion blur in the prompt.
