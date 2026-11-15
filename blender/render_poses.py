"""Render the creature in many poses through the clip's cameras: the training pictures of the joint reader.

    blender model.blend --background --python blender/render_poses.py -- --poses P/reader/poses.npz
        --cameras P/start/cameras.json --out P/reader/renders [--copies 12] [--samples 8] [--height 720] [--fit 448]

poses.npz (python -m video2rig.reader.poses): per pose every bone's basis rotation (quaternion w x y z) and the
root's location. The studio is render_start.py's (grey backdrop, two suns, the floor under the start pose); the
cameras are the start frame's. Each render call holds --copies creatures side by side, each in its own pose and
its own slot along the camera's right axis (the cameras are orthographic, so a slot is exactly the picture that
camera would take of that copy alone; slots are one and a half panels wide, so a neighbour's tail never reaches in, and
every other copy stands further back, out of its neighbours' shadows). Blender pays its overhead once per call,
not once per pose: one process does what four could not (they wait for each other on the card).
Panels are rendered at the reader's own input size (--fit: each panel letterboxed into a square of that side);
--height is the clip's frame height, kept with the labels to give errors in the clip's pixels.

Writes <out>/<view>/<batch>.jpg (the strip of slots) and <out>/labels.npz: every bone's head and tail in armature
space as Blender posed it (the truth the reader learns), every prop's matrix (a log, a stone: its own object, moved
rigidly as poses.npz says), and where each pose's slot lies in which strip. A .blend saved by render_start.py may
carry its studio (v2r_* objects): it is removed and built again here.
"""
import argparse
import json
import math
import os
import sys
import time
import traceback

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

GREY = (0.42, 0.41, 0.40, 1.0)


def args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser(prog='render_poses.py')
    ap.add_argument('--poses', required=True)
    ap.add_argument('--cameras', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--copies', type=int, default=12)
    ap.add_argument('--samples', type=int, default=8)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fit', type=int, default=448)
    ap.add_argument('--engine', default='BLENDER_EEVEE')
    ap.add_argument('--limit', type=int, default=0, help='render only the first N poses (timing)')
    ap.add_argument('--armature', default='')
    ap.add_argument('--sharp', action='store_true', help='no motion blur even if the poses carry motion')
    ap.add_argument('--project', default='', help="the project: its rig/rig.json is checked against the armature, and "
                    "its rig/texture_<k>.png fill in base colour images the .blend cannot find")
    return ap.parse_args(argv)


def bounds(objs):
    deps = bpy.context.evaluated_depsgraph_get()
    lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
    for o in objs:
        ev = o.evaluated_get(deps)
        for v in ev.data.vertices:
            w = ev.matrix_world @ v.co
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
    return lo, hi


def set_engine(scene, engine):
    """The render engine by name. EEVEE is BLENDER_EEVEE before Blender 4.2 and again from 5.0, BLENDER_EEVEE_NEXT in
    4.2-4.5: either name works on every version (--engine BLENDER_EEVEE_NEXT stopped render_start.py on 5.2)."""
    twin = {'BLENDER_EEVEE': 'BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE_NEXT': 'BLENDER_EEVEE'}
    try:
        scene.render.engine = engine
    except TypeError:
        if engine not in twin:
            raise
        scene.render.engine = twin[engine]


def studio(lo, hi, samples, engine):
    """The studio of blender/render_start.py (same backdrop, lights and floor)."""
    scene = bpy.context.scene
    world = bpy.data.worlds.new('video2rig_studio')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = GREY
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.9
    scene.world = world
    for name, rot, energy in (('v2r_key', (math.radians(50), 0, math.radians(-35)), 3.2),
                              ('v2r_fill', (math.radians(60), 0, math.radians(140)), 1.2)):
        light = bpy.data.lights.new(name, 'SUN')
        light.energy = energy
        light.angle = math.radians(12)
        ob = bpy.data.objects.new(name, light)
        ob.rotation_euler = rot
        scene.collection.objects.link(ob)
    r = (hi - lo).length * 40                         # the floor under every slot
    floor = bpy.data.meshes.new('v2r_floor')
    floor.from_pydata([(-r, -r, lo.z), (r, -r, lo.z), (r, r, lo.z), (-r, r, lo.z)], [], [(0, 1, 2, 3)])
    mat = bpy.data.materials.new('v2r_floor')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = GREY
    bsdf.inputs['Roughness'].default_value = 0.9
    floor.materials.append(mat)
    scene.collection.objects.link(bpy.data.objects.new('v2r_floor', floor))
    set_engine(scene, engine)
    if scene.render.engine == 'CYCLES':
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
    else:
        scene.eevee.taa_render_samples = samples
    scene.view_settings.view_transform = 'AgX'
    scene.render.use_motion_blur = False
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.quality = 92
    scene.render.resolution_percentage = 100


def camera(rec):
    data = bpy.data.cameras.new(rec['name'])
    data.sensor_fit = 'VERTICAL'
    if rec['type'] == 'ortho':
        data.type = 'ORTHO'
        data.ortho_scale = rec['ortho_height']
    else:                                            # render_start.py --lens: focal_px at the camera's own height
        data.type = 'PERSP'
        data.sensor_height = 24.0
        data.lens = 24.0 * float(rec['focal_px']) / float(rec['height_px'])
    data.clip_end = 1000.0
    cam = bpy.data.objects.new(rec['name'], data)
    bpy.context.scene.collection.objects.link(cam)
    R = Matrix((rec['right'], rec['up'], [-c for c in rec['forward']])).transposed()
    cam.matrix_world = Matrix.Translation(rec['location']) @ R.to_4x4()
    return cam


def make_copies(arm, meshes, n, props=()):
    """n creatures: the original and n - 1 copies sharing its armature and mesh data, each with its own pose (and
    its own copy of every prop)."""
    scene = bpy.context.scene
    out = [(arm, meshes)]
    copies = [list(props)]
    for _ in range(n - 1):
        pc = []
        for pr in props:
            c = pr.copy()
            scene.collection.objects.link(c)
            pc.append(c)
        copies.append(pc)
        a = arm.copy()
        scene.collection.objects.link(a)
        ms = []
        for m in meshes:
            mc = m.copy()
            scene.collection.objects.link(mc)
            if m.parent == arm:
                mc.parent = a
            for mod in mc.modifiers:
                if mod.type == 'ARMATURE' and mod.object == arm:
                    mod.object = a
            ms.append(mc)
        out.append((a, ms))
    return out, copies


def main():
    a = args()
    t0 = time.time()
    out = os.path.abspath(a.out)
    d = np.load(a.poses)
    names = [str(n) for n in d['bones']]
    root = str(d['root'])
    quats, locs = d['quats'], d['root_location']
    blur = 'quats_a' in d.files and not a.sharp
    if blur:                                          # each pose keyed at frames 0 and 2, rendered at 1: the shutter
        qa, la, qb, lb = d['quats_a'], d['root_location_a'], d['quats_b'], d['root_location_b']
    scene = bpy.context.scene
    for o in [o for o in scene.objects if o.name.startswith('v2r_')]:     # render_start.py's studio and cameras
        bpy.data.objects.remove(o, do_unlink=True)
    prop_names = [str(n) for n in d['prop_names']] if 'prop_names' in d.files else []
    prop_mats = d['prop_mats'] if prop_names else None
    arms = [o for o in scene.objects if o.type == 'ARMATURE']
    arm = scene.objects[a.armature] if a.armature else arms[0]
    if a.project:                                     # a texture left on the machine that made the .blend: the copy
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # export_rig.py saved in the project
        from textures import check_rig, relink_missing
        info = json.load(open(os.path.join(a.project, 'rig', 'rig.json')))
        why = check_rig(arm, info)
        if why:
            sys.exit(f'render_poses: the armature {arm.name} is not the project\'s rig ({why})')
        relink_missing(arm, os.path.join(a.project, 'rig'), info, log=lambda m: print('RENDER_POSES ' + m, flush=True))
    if arm.animation_data is not None:
        arm.animation_data.action = None              # the poses below, not a stored animation
    for pb in arm.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    objs = [o for o in scene.objects if o.type == 'MESH' and o.visible_get()]
    lo, hi = bounds(objs)                             # the start pose: the floor under it, as render_start.py
    meshes = [o for o in objs if o.parent == arm or any(m.type == 'ARMATURE' and m.object == arm for m in o.modifiers)]
    props = [scene.objects[n] for n in prop_names]
    meshes = [m for m in meshes if m not in props]
    prop0 = [pr.matrix_world.copy() for pr in props]
    studio(lo, hi, a.samples, a.engine)
    if blur:
        scene.render.use_motion_blur = True
        scene.render.motion_blur_shutter = .5         # as dev/bench/render_gt.py, Seedance's blur on fast limbs
        scene.frame_set(1)
    arm_world = np.array(arm.matrix_world)
    cams = json.load(open(a.cameras))
    clip_scale = a.height / cams['frame_px'][1]
    K = a.copies
    if any(rec['type'] != 'ortho' for rec in cams['views']) and K != 1:
        sys.exit('render_poses: a camera with perspective sees copies side by side at other angles: --copies 1')
    crowd, prop_copies = make_copies(arm, meshes, K, props)
    A0 = arm.matrix_world.copy()
    A0i = A0.inverted()
    posed = [[pm.copy() for pm in prop0] for _ in range(K)]
    base = [(ar.matrix_world.copy(), [m.matrix_world.copy() for m in ms]) for ar, ms in crowd]
    views = []
    for rec in cams['views']:
        cw, ch = rec['width_px'] * clip_scale, rec['height_px'] * clip_scale       # the panel in the clip
        s = min(a.fit / cw, a.fit / ch)
        h = int(round(ch * s))
        w = int(round(cw * h / ch))
        slot = (3 * w) // 2 | 1 if K > 1 else w        # half a panel of backdrop between neighbours
        k_px = h / rec.get('ortho_height', rec.get('fov_height', 1.0))
        wide = K * slot
        x0 = [c * slot + (slot - w) // 2 for c in range(K)]
        shift = [(x + w / 2 - wide / 2) / k_px for x in x0]                       # metres along the camera's right
        os.makedirs(os.path.join(out, rec['name']), exist_ok=True)
        views.append(dict(cam=camera(rec), rec=rec, size=(w, h), clip_size=(cw, ch), wide=wide, x0=x0, shift=shift,
                          right=Vector(rec['right']), fwd=Vector(rec['forward'])))
    order = list(range(len(quats)))
    if a.limit:
        order = order[:a.limit]
    bone_names = [pb.name for pb in arm.pose.bones]
    heads = np.zeros((len(order), len(bone_names), 3), np.float32)
    tails = np.zeros_like(heads)
    slot_of = np.zeros((len(order), 2), np.int32)     # (batch, slot)
    t1 = time.time()
    nb = (len(order) + K - 1) // K
    for b in range(nb):
        batch = order[b * K:(b + 1) * K]
        for c, (ar, ms) in enumerate(crowd):
            used = c < len(batch)
            ar.hide_render = not used
            for m in ms:
                m.hide_render = not used
            for pc in prop_copies[c]:
                pc.hide_render = not used
            if not used:
                continue
            i = batch[c]
            for kp, pm0 in enumerate(prop0):                 # armature-space matrix -> the object's world matrix
                posed[c][kp] = A0 @ Matrix(prop_mats[i, kp].tolist()) @ A0i @ pm0
            if blur:                                  # the keys make each copy's own action (Blender does)
                for frame, Q, L in ((0, qa[i], la[i]), (2, qb[i], lb[i])):
                    for n, q in zip(names, Q):
                        pb = ar.pose.bones.get(n)
                        if pb is not None:
                            pb.rotation_mode = 'QUATERNION'
                            pb.rotation_quaternion = Quaternion(q)
                            pb.keyframe_insert('rotation_quaternion', frame=frame)
                    ar.pose.bones[root].location = Vector(L)
                    ar.pose.bones[root].keyframe_insert('location', frame=frame)
                continue
            for n, q in zip(names, quats[i]):
                pb = ar.pose.bones.get(n)
                if pb is not None:
                    pb.rotation_mode = 'QUATERNION'
                    pb.rotation_quaternion = Quaternion(q)
            ar.pose.bones[root].location = Vector(locs[i])
        if blur:
            scene.frame_set(1)                        # the truth: the pose at the middle of the shutter
        bpy.context.view_layer.update()
        for c, i in enumerate(batch):
            ar = crowd[c][0]
            k = b * K + c
            for j, pb in enumerate(ar.pose.bones):
                heads[k, j], tails[k, j] = pb.head, pb.tail
            slot_of[k] = (b, c)
        for v in views:
            for c, (ar, ms) in enumerate(crowd):
                move = Matrix.Translation(v['right'] * v['shift'][c] + v['fwd'] * (2.5 * (c % 2)))
                ar.matrix_world = move @ base[c][0]
                for m, m0 in zip(ms, base[c][1]):
                    if m.parent != ar:
                        m.matrix_world = move @ m0
                for pc, pm in zip(prop_copies[c], posed[c]):
                    pc.matrix_world = move @ pm
            scene.camera = v['cam']
            scene.render.resolution_x, scene.render.resolution_y = v['wide'], v['size'][1]
            scene.render.filepath = os.path.join(out, v['rec']['name'], f'{b:04d}.jpg')
            bpy.ops.render.render(write_still=True)
        if (b + 1) % 10 == 0 or b + 1 == nb:
            el = time.time() - t1
            done = min((b + 1) * K, len(order))
            print(f'RENDER_POSES {done}/{len(order)}  {el / done:.3f} s per pose', flush=True)
    np.savez_compressed(os.path.join(out, 'labels.npz'), index=np.array(order), names=np.array(bone_names),
                        heads=heads, tails=tails, armature_world=arm_world, slot_of=slot_of,
                        views=np.array([v['rec']['name'] for v in views]),
                        sizes=np.array([v['size'] for v in views]), clip_sizes=np.array([v['clip_size'] for v in views]),
                        x0=np.array([v['x0'] for v in views]), copies=K, clip_height=a.height,
                        prop_names=np.array(prop_names),
                        prop_mats=prop_mats[order] if prop_names else np.zeros((len(order), 0, 4, 4)))
    print(f'RENDER_POSES done: {len(order)} poses in {time.time() - t0:.1f} s (setup {t1 - t0:.1f} s)', flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
    sys.exit(1)
