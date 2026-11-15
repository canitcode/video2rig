"""Export the model and its video2rig motions as one FBX for a game engine (made for Unity; Unreal and Godot read the
same file): the armature, its skinned meshes, and every chosen Action as its own clip, baked on every frame with no
keyframe reduction (the fit's motion as it is: a reduced curve can step or overshoot where the motion is fast).

    blender model.blend --background --python blender/export_fbx.py -- --out my_troll/troll.fbx
        [--actions stagger-reader,walk-reader-loop] [--armature Armature] [--fps 30]

Actions: those blender/import_motion.py made (every Action with a fake user that animates the armature's bones) or the
ones named. Each becomes an NLA strip of its own name for the export, so the FBX holds one take per Action, named as
the Action; a prop's Action that import_motion made with it (<action>_<prop>) goes into the same take.

--fps: the scene's frame rate for the bake. Keys land on the motion's content frames (30 per second for a "30-5"
clip, 24 or 25 otherwise); a bake at another rate samples between them. Give the clip's own rate, or import the motion
into a scene of that rate (blender/import_motion.py keys at the scene's rate).

Scale and axes for Unity: 1 Blender metre is 1 Unity unit (apply unit scale, FBX units), Y up, -Z forward, no leaf
bones. Beside the FBX, <name>.v2r.json names the bone that carries the body's place (Unity's root motion node). In
Unity, put both files under a folder named Video2Rig and copy unity/Video2RigImport.cs into an Editor folder: it sets
the import (docs/GAME.md).
"""
import argparse
import json
import os
import sys

import bpy


def args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser(prog='export_fbx.py')
    ap.add_argument('--out', required=True)
    ap.add_argument('--actions', default='', help='comma-separated Action names (default: every motion import_motion made)')
    ap.add_argument('--armature', default='')
    ap.add_argument('--fps', type=float, default=0.0, help='the bake\'s frame rate (default: the scene\'s)')
    return ap.parse_args(argv)


def action_fcurves(act):
    """Every F-curve of an Action (Blender 4.x: action.fcurves; 5.x: in its layers' strips' channel bags)."""
    fcurves = getattr(act, 'fcurves', None)
    if fcurves is not None:
        return list(fcurves)
    return [fc for layer in getattr(act, 'layers', ()) for strip in layer.strips
            for bag in getattr(strip, 'channelbags', ()) for fc in bag.fcurves]


def bone_actions(arm):
    """The Actions that animate the armature's bones and have a fake user (import_motion.py keeps its motions so)."""
    bones = {b.name for b in arm.data.bones}
    out = []
    for act in bpy.data.actions:
        paths = [fc.data_path for fc in action_fcurves(act)]
        if act.use_fake_user and any(p.startswith('pose.bones["') and p.split('"')[1] in bones for p in paths):
            out.append(act)
    return out


def as_strips(obj, actions):
    """The object's animation as one NLA track per Action, each strip named as its Action, the active Action cleared:
    the FBX exporter then writes one take per strip."""
    ad = obj.animation_data or obj.animation_data_create()
    ad.action = None
    for track in list(ad.nla_tracks):
        ad.nla_tracks.remove(track)
    for name, act in actions:
        track = ad.nla_tracks.new()
        track.name = name
        strip = track.strips.new(name, int(act.frame_range[0]), act)
        strip.name = name


def main():
    a = args()
    scene = bpy.context.scene
    arms = [o for o in scene.objects if o.type == 'ARMATURE']
    if not arms:
        sys.exit('export_fbx: no armature in the scene')
    arm = scene.objects[a.armature] if a.armature else arms[0]
    actions = [bpy.data.actions[n] for n in a.actions.split(',') if n] if a.actions else bone_actions(arm)
    if not actions:
        sys.exit('export_fbx: no motion to export (import one with blender/import_motion.py, or give --actions)')
    if a.fps:
        scene.render.fps, scene.render.fps_base = int(round(a.fps)), round(a.fps) / a.fps
    as_strips(arm, [(act.name, act) for act in actions])
    meshes = [o for o in scene.objects if o.type == 'MESH' and (o.parent == arm or any(
        m.type == 'ARMATURE' and m.object == arm for m in o.modifiers))]
    props = []
    for o in scene.objects:
        mine = [(act.name, bpy.data.actions[f'{act.name}_{o.name}']) for act in actions
                if f'{act.name}_{o.name}' in bpy.data.actions]
        if o.type == 'MESH' and o not in meshes and mine:
            as_strips(o, mine)
            props.append(o)
    bpy.ops.object.select_all(action='DESELECT')
    for o in [arm] + meshes + props:
        o.select_set(True)
    bpy.context.view_layer.objects.active = arm
    start = min(int(act.frame_range[0]) for act in actions)
    scene.frame_start, scene.frame_end = start, max(int(round(act.frame_range[1])) for act in actions)
    off_frame = [act.name for act in actions for fc in action_fcurves(act)[:1]
                 if any(abs(k.co[0] - round(k.co[0])) > 1e-3 for k in fc.keyframe_points)]
    if off_frame:
        print(f'EXPORT_FBX note: keys between frames in {", ".join(off_frame)}: the bake samples at '
              f'{scene.render.fps / scene.render.fps_base:g} fps (--fps the clip\'s own rate keeps every key)', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=os.path.abspath(a.out), use_selection=True, object_types={'ARMATURE', 'MESH'},
        apply_unit_scale=True, apply_scale_options='FBX_SCALE_UNITS', axis_forward='-Z', axis_up='Y',
        use_armature_deform_only=False, add_leaf_bones=False, primary_bone_axis='Y', secondary_bone_axis='X',
        armature_nodetype='NULL', bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=True,
        bake_anim_use_all_actions=False, bake_anim_force_startend_keying=True, bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0, mesh_smooth_type='FACE')
    # beside it, for unity/Video2RigImport.cs: the bone that carries the body's place (Unity's root motion node)
    root = next((fc.data_path.split('"')[1] for act in actions for fc in action_fcurves(act)
                 if fc.data_path.startswith('pose.bones["') and fc.data_path.endswith('.location')), None)
    side = os.path.splitext(os.path.abspath(a.out))[0] + '.v2r.json'
    with open(side, 'w', encoding='utf-8') as f:
        json.dump(dict(root=f'{arm.name}/{root}' if root else '', clips=[act.name for act in actions],
                       fps=scene.render.fps / scene.render.fps_base), f, indent=1)
    print(f'EXPORT_FBX {os.path.abspath(a.out)}: {len(actions)} clips ({", ".join(act.name for act in actions)}), '
          f'{len(meshes)} meshes, {len(props)} props, {scene.render.fps / scene.render.fps_base:g} fps', flush=True)


main()
