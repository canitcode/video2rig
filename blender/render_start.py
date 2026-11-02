"""Render the start frame for Seedance from your rigged model, in the pose it stands in: a plain grey studio, two
orthographic cameras (front | side, one 1920x1080 frame split in two halves) or one, and start/cameras.json.

    blender model.blend --background --python blender/render_start.py -- --out my_creature/start
    blender model.blend --background --python blender/render_start.py -- --out my_creature/start --views side

Options: --views front,side (default) | side | front | three-quarter | top (any two: side,top suits a long, low
creature: a belly slider, a snake, a lizard); --widths 1280,640 (each view's width in px, adding up to 1920; default
equal); --facing auto|+X|-X|+Y|-Y (which way the creature faces; auto = from its feet bones); --side left|right
(which flank the side and three-quarter cameras see); --samples 64; --margin 1.8 (frame height / body height: room for raised arms and
jumps); --margin-wide 1.45 (room around the body's length or width across a panel: a long creature is framed by
its length, not only its height; off by default, so earlier start frames come out as before); --lead 0.18 (the side and top views look this share of their frame ahead of the
body: room in front for a charge or a crawl forward); --hover 0.07 (a creature that floats: the floor lies this
share of the body's height under its lowest point, so its shadow stands free of it and Seedance keeps it in the
air; put the same drop into creature.json as "floor", see the line the script prints); --engine CYCLES|BLENDER_EEVEE_NEXT;
--divider 4 (a dark line between the two views, px; 0 = none: it keeps Seedance from blending the panels).
Upload start/start.png to Seedance as the first frame. Keep the model, its pose and the cameras unchanged
afterwards: the start frame is the one moment where the tool knows the pose and the camera exactly.
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

GREY = (0.42, 0.41, 0.40, 1.0)


def args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser(prog='render_start.py')
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', default='front,side')
    ap.add_argument('--facing', default='auto')
    ap.add_argument('--side', default='left')
    ap.add_argument('--samples', type=int, default=64)
    ap.add_argument('--margin', type=float, default=1.8)
    ap.add_argument('--margin-wide', type=float, default=0.0)
    ap.add_argument('--widths', default='')
    ap.add_argument('--lead', type=float, default=0.0)
    ap.add_argument('--hover', type=float, default=0.0)
    ap.add_argument('--engine', default='CYCLES')
    ap.add_argument('--divider', type=int, default=4)
    ap.add_argument('--lens', type=float, default=0.0, help='a lens in mm (35, 50) on a 24 mm tall sensor: a camera '
                    'with perspective, as the video models draw (a creature coming at the camera grows); 0 = '
                    'orthographic (every earlier start frame)')
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


def facing(arm, how):
    if how != 'auto':
        return Vector({'+X': (1, 0, 0), '-X': (-1, 0, 0), '+Y': (0, 1, 0), '-Y': (0, -1, 0)}[how])
    v = Vector((0, 0, 0))
    for pb in arm.pose.bones:
        if 'foot' in pb.name.lower() or 'toe' in pb.name.lower():
            d = (arm.matrix_world @ pb.tail) - (arm.matrix_world @ pb.head)
            v += Vector((d.x, d.y, 0))
    if v.length < 1e-6:
        sys.exit('render_start: cannot tell which way the creature faces, pass --facing')
    v.normalize()
    return v


def studio(lo, hi, samples, engine):
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
    r = (hi - lo).length * 10
    floor = bpy.data.meshes.new('v2r_floor')
    floor.from_pydata([(-r, -r, lo.z), (r, -r, lo.z), (r, r, lo.z), (-r, r, lo.z)], [], [(0, 1, 2, 3)])
    mat = bpy.data.materials.new('v2r_floor')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = GREY
    bsdf.inputs['Roughness'].default_value = 0.9
    floor.materials.append(mat)
    scene.collection.objects.link(bpy.data.objects.new('v2r_floor', floor))
    scene.render.engine = engine
    if engine == 'CYCLES':
        scene.cycles.device = 'CPU'
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
    scene.view_settings.view_transform = 'AgX'


def camera(name, look, target, ortho_h, w, h, up=None, lens=0.0):
    """An orthographic camera ortho_h tall, or with --lens (mm, a 24 mm tall sensor) one with perspective placed
    so that its view is ortho_h tall at the target."""
    data = bpy.data.cameras.new(name)
    data.sensor_fit = 'VERTICAL'
    if lens:
        data.type = 'PERSP'
        data.lens = lens
        data.sensor_height = 24.0
        dist = ortho_h * lens / 24.0
    else:
        data.type = 'ORTHO'
        data.ortho_scale = ortho_h
        dist = 30.0
    data.clip_end = 1000.0
    cam = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = target - look * dist
    if up is None:
        cam.rotation_euler = look.to_track_quat('-Z', 'Y').to_euler()
    else:                                          # (a camera looking straight down: its up given, the head)
        z = -look.normalized()
        y = (up - z * up.dot(z)).normalized()
        x = y.cross(z)
        from mathutils import Matrix
        cam.rotation_euler = Matrix((x, y, z)).transposed().to_euler()
    bpy.context.view_layer.update()
    R = cam.matrix_world.to_3x3().normalized()
    rec = {'name': name, 'width_px': w, 'height_px': h, 'type': 'persp' if lens else 'ortho',
           'location': list(cam.location), 'right': list(R.col[0]), 'up': list(R.col[1]), 'forward': list(-R.col[2])}
    if lens:
        rec.update({'focal_px': h * lens / 24.0, 'lens_mm': lens, 'fov_height': ortho_h, 'distance': dist})
    else:
        rec.update({'ortho_width': ortho_h * w / h, 'ortho_height': ortho_h})
    return cam, rec


def main():
    a = args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    scene = bpy.context.scene
    arm = next(o for o in scene.objects if o.type == 'ARMATURE')
    objs = [o for o in scene.objects if o.type == 'MESH' and o.visible_get()]
    lo, hi = bounds(objs)
    floor = lo.copy()
    floor.z = lo.z - a.hover * (hi.z - lo.z)
    studio(floor, hi, a.samples, a.engine)
    fwd = facing(arm, a.facing)
    left = Vector((0, 0, 1)).cross(fwd)            # the creature's left
    views = [v for v in a.views.split(',') if v]
    split = len(views) > 1
    H = 1080
    widths = [int(x) for x in a.widths.split(',')] if a.widths else [1920 // len(views)] * len(views)
    assert len(widths) == len(views) and sum(widths) == 1920, 'render_start: --widths must add up to 1920'
    body_h = hi.z - lo.z
    # the body's extent along its length and across it (the mesh's corners on the creature's own axes)
    corners = [Vector((x, y, 0)) for x in (lo.x, hi.x) for y in (lo.y, hi.y)]
    length = max(c.dot(fwd) for c in corners) - min(c.dot(fwd) for c in corners)
    width = max(c.dot(left) for c in corners) - min(c.dot(left) for c in corners)
    # one scale for every panel: the height with room to rear up, and each panel's horizontal extent (and the top
    # view's vertical one: its length) with room to move
    across = {'front': width, 'side': length, 'three-quarter': .77 * length + .64 * width, 'top': width}
    ortho_h = body_h * a.margin
    for v, w in zip(views, widths):
        if a.margin_wide > 0:
            ortho_h = max(ortho_h, across[v] * a.margin_wide * H / w)
        if v == 'top':
            ortho_h = max(ortho_h, length * max(a.margin_wide, 1.2))
    centre = (lo + hi) * .5
    target_z = floor.z + ortho_h * (.5 - .12)      # floor line 12 % above the bottom edge
    looks = {'front': -fwd, 'side': (-left if a.side == 'left' else left),
             'three-quarter': (-fwd * math.cos(math.radians(40))
                               - (left if a.side == 'left' else -left) * math.sin(math.radians(40))),
             'top': Vector((0, 0, -1))}
    recs, paths = [], []
    x0 = 0
    for k, v in enumerate(views):
        W = widths[k]
        look = looks[v].normalized()
        target = Vector((centre.x, centre.y, target_z if v != 'top' else centre.z))
        if v in ('side', 'top') and a.lead:
            span = ortho_h * W / H if v == 'side' else ortho_h
            target = target + fwd * a.lead * span
        cam, rec = camera(f'v2r_{v}', look, target, ortho_h, W, H, up=fwd if v == 'top' else None, lens=a.lens)
        rec['x0_px'] = x0
        x0 += W
        scene.camera = cam
        scene.render.resolution_x, scene.render.resolution_y = W, H
        scene.render.resolution_percentage = 100
        path = os.path.join(out, f'panel-{v}.png')
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        recs.append(rec)
        paths.append(path)
        print('RENDER_START', v, '->', path, flush=True)
    panels = []
    for p in paths:
        img = bpy.data.images.load(p)
        px = np.array(img.pixels[:], np.float32).reshape(img.size[1], img.size[0], 4)
        panels.append(px)
        bpy.data.images.remove(img)
    full = np.concatenate(panels, axis=1)
    if split and a.divider > 0:
        for k in range(1, len(views)):
            x = sum(widths[:k])
            full[:, x - a.divider // 2:x + (a.divider + 1) // 2, :3] = .03
    img = bpy.data.images.new('v2r_start', full.shape[1], full.shape[0], alpha=True)
    img.pixels.foreach_set(full.ravel())
    img.filepath_raw = os.path.join(out, 'start.png')
    img.file_format = 'PNG'
    img.save()
    with open(os.path.join(out, 'cameras.json'), 'w', newline='\n') as f:
        json.dump({'frame_px': [sum(widths), H], 'views': recs, 'hover': a.hover * body_h}, f, indent=1)
        f.write('\n')
    if a.hover:
        up_scale = (arm.matrix_world.to_3x3().inverted() @ Vector((0, 0, 1))).length
        print(f'RENDER_START hover: the floor is {a.hover * body_h:.4f} under the body; in creature.json set '
              f'"floor" to the body\'s lowest point less {a.hover * body_h * up_scale:.4f} (armature units)', flush=True)
    print('RENDER_START done ->', os.path.join(out, 'start.png'), flush=True)


main()
