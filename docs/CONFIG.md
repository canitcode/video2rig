# creature.json

Every field is optional; an empty `{}` works for most humanoids and animals. Settings for one clip go under
`clips.<name>` and override the creature-wide ones. Most fields serve both roads of `run`; `dense_bones`,
`seed_dense_px`, `seed_body_px`, `track_side` and `solve.*` only the older road (the point tracker, `--old`). The
reader fit has settings of its own (below).

```json
{
 "name": "troll",
 "rig": "rig",
 "start": "start",
 "root_bone": "Hips",
 "facing": "-Y",
 "fit_bones": "auto",
 "hinges": "auto",
 "joints": {"LeftShoulder": {"range": 0.6, "prior": 40}},
 "dense_bones": "auto",
 "seed_dense_px": 3,
 "seed_body_px": 10,
 "track_side": 768,
 "fps": null,
 "solve": {"smooth_body": 300, "smooth_legs": 60, "rounds": 3, "lm_iters": 6},
 "clips": {
  "stagger": {"file": "clips/stagger.mp4"},
  "roar": {"solve": {"smooth_legs": 40}}
 }
}
```

| Field | Default | What it does |
| --- | --- | --- |
| `rig`, `start` | `rig`, `start` | folders of the exported rig and the start frame, relative to the project |
| `root_bone` | Hips / pelvis / first bone | the bone that carries the body's position |
| `facing` | from the feet | which way the creature faces (`+X -X +Y -Y`, world); needed only if it has no feet bones |
| `fit_bones` | `auto` | bones the fit moves: every bone with real skin weight and its parents; fingers, eyes, jaw and end bones keep their start pose. Or a list of names |
| `hinges` | `auto` | knees and elbows by name (Mixamo `LeftLeg`/`LeftForeArm`, Rigify `shin`/`forearm`, Unreal `calf`/`lowerarm`...) bend one way only; or `{"Shin.L": "back", "Elbow.R": "front"}` |
| `joints.<bone>.range` | 2.8 rad, twist 1.2 | how far a joint may turn from the start pose (radians), or `[lo, hi]` |
| `joints.<bone>.prior` | 2-60 | how strongly the joint stays near the start pose when the video does not say (twist, collarbones, spine, toes get more) |
| `dense_bones` | limbs by name | (older road) parts that get dense tracking points and lighter smoothing (arms, legs, hands, feet, wings); props always. Bones named otherwise (femur, tibia, L1...): chains that leave a bone with three or more children (a pelvis, an insect's body) |
| `seed_dense_px`, `seed_body_px` | 2-3, 8-10 | (older road) spacing of the tracking points on limbs / the rest, in pixels of a 768 px view (a bigger video gets proportionally wider spacing, so 1080p costs no more than 720p) |
| `track_side` | 768 | (older road) the tracker's working resolution (square). 512 saves graphics memory; 1024 for tiny parts |
| `fps` | measured | force the content timing: `"24"`, `"30-5"` (30 fps, every 5th dropped), `"25-25"` |
| `reader_tta` | 8 | how many times the reader reads every picture (shifted and scaled a little; HOW-IT-WORKS step 3). 0 reads each picture once, as before 2026-10-20: about 2.8 times faster, a shakier reading |
| `reader_crops` | false | (experimental) read every hand, paw and hoof again in a zoomed square with `reader/reader-crops.pt` (`python -m video2rig.reader.train <project> --crops --init reader/reader.pt --out reader/reader-crops.pt`) |
| `clips.<name>.time_scale` | 1 | a clip Seedance drew in slow motion (`python -m video2rig.generate ... --slow-motion 0.5`): its times x this are the real ones, so the motion plays at normal speed. Twice the Seedance cost for a move; measured on one strike: more frames per strike, but a softer move |
| `views.<name>.turn` | true | allow a small camera turn when aligning on frame 0 (Seedance sometimes re-frames by 10+ degrees) |
| `solve.smooth_body` / `smooth_legs` | 300 / 60 | (older road, as every `solve.*`) smoothness of the body / limbs (higher = smoother, may lose fast strikes) |
| `solve.rounds`, `solve.lm_iters` | 4, 12 | solver effort; 3 and 6 are usually enough |
| `solve.rescue` | true | where the first pass loses the outline, try the body turned over (rolls, flips, deaths on the back) |
| `solve.rescue_turn_cost` | 0.25 | how much outline a turn must win per radian before the body turns (lower = turns more easily) |
| `solve.w_cover` | 150 | pull of the model into parts of the video silhouette it does not cover; 0 when the silhouette holds things that are not the model (a prop you did not export, a big shadow) |

The defaults in this table were tuned on a troll (two views), wolves (two views) and a man with a log or a stone
(one view).

## The reader fit's settings

The reader's fit has a few settings of its own: `SETTINGS` in `video2rig/reader/fit.py` (the number of steps, the
robust scale of a miss, the weights of smoothness, of the pull toward the start pose and of the floor, the front
view's weight on heights). They are not read from `creature.json`, and `run --set` does not reach them (it changes
the older road's `solve.*`). To try one, fit the clip's reading again with it:

```powershell
python -m video2rig.reader.fit my_troll stagger --set w_acc=0.05
```

`--set KEY=VALUE` can be given more than once. It reuses the reading `work/stagger/reader.npz` that `run` or `check`
made, so it takes about a minute, and writes `work/stagger/fit-reader.npz` and `out/stagger-reader.npz` next to the
run's own files (another name: `--tag=-soft`). Review it with `python -m video2rig review my_troll stagger
--tag=-reader`. Change one to fix a clip you have watched, and compare the review films.
