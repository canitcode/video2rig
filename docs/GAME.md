# Into a game engine (Unity)

The motion is an ordinary Blender Action, so any engine that reads FBX or glTF takes it. These steps keep it exactly
as the fit made it: every content frame as a key, no keyframe reduction on the way, Unity's import set so that it
does not reduce it either.

## 1. Blender: the motions as one FBX

```powershell
# a scene at the clip's content rate: 30 for a "30-5" clip, 25 for "25-25", else 24 (work/<clip>/timing.json says
# which); import_motion.py keys at the scene's rate, so every content frame lands on a whole frame
blender model.blend --background --python-expr "import bpy; bpy.context.scene.render.fps = 30" --python blender/import_motion.py -- --motion my_troll/out/walk.npz --save my_troll/game.blend
blender my_troll/game.blend --background --python blender/import_motion.py -- --motion my_troll/out/stagger.npz --save my_troll/game.blend
# every motion import_motion.py made, one clip each, the model with them
blender my_troll/game.blend --background --python blender/export_fbx.py -- --out my_troll/unity/Video2Rig/troll.fbx
```

`export_fbx.py` writes one take per Action, named as the Action, baked on every frame with no keyframe reduction, in
metres (1 Blender metre = 1 Unity unit), Y up, -Z forward, without leaf bones. Beside the FBX, `troll.v2r.json` names
the bone that carries the body's place. `--actions a,b` exports only those; `--fps` bakes at another rate (it prints
a note when keys would fall between frames).

## 2. Unity: the import settings

Copy `unity/Video2RigImport.cs` into an `Editor` folder of your project, and the FBX with its `.v2r.json` into any
folder named `Video2Rig` (for example `Assets/Characters/Troll/Video2Rig/`). Every FBX there is imported:

- as a Generic rig, with the bone from `.v2r.json` as the root motion node;
- with keyframe reduction at 0.1 degrees / 0.1% instead of Unity's default (Optimal at 0.5);
- with every clip whose name ends in `-loop` set to loop (`reader_fit` `"loop"` writes those), and its loop pose off:
  the fit already made its end its start;
- with the original orientation and height kept, so a clip starts where the character stands.

A clip made with `reader_fit` `"travel"` moves the character when the Animator's Apply Root Motion is on; a clip drawn
on the spot stays on the spot either way. For Unity's humanoid retargeting, switch the model's rig to Humanoid by hand
after the first import.

## What the import changes (measured)

A 4.9 s motion and its 2.5 s loop of the bench's test creature (29 bones, 30 fps content), imported into Unity 6000.6
four ways, every bone's place sampled at 60 Hz and compared with the import with compression off:

| Import | Joint error mean / max | Shake (vs off) | Clip memory |
| --- | --- | --- | --- |
| Compression off | 0 / 0 mm | 1.00 | 1920 KB |
| Unity's default: Optimal, 0.5 | 1.05 / 11.7 mm | 1.16 | 558 KB |
| Keyframe reduction at 1.0 | 3.4 / 22.6 mm | 1.38 | 583 KB |
| `Video2RigImport.cs`: keyframe reduction at 0.1 | 0.01 / 1.05 mm | 1.00 | 1172 KB |

Unity's default reduction leaves out keys where the curve looks straight enough, and on a fast move the missing keys
show as steps: 16-18% more shake on the bench's clips (the loop alike). At 0.1 the motion stays as exported at
61% of the uncompressed memory.

With compression off, Unity's bones sit where Blender's are at every key (0.01 mm at most, after one rigid fit of the
axes: Unity is left-handed). Between keys Unity's curves are smooth and Blender's import keys are linear, so they part
by up to 13 mm there on a fast move; neither is the video's.
