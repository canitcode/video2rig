"""Auto-rig, step 3 (Blender): the armature from joints.json (video2rig.autorig.skeleton) on the model of
autorig_prepare.py, skin weights, a saved rigged .blend and a sheet of test poses to look at.

    blender --background --factory-startup --python blender/autorig_build.py -- --work P/autorig [--no-sheet]
        -> <work>/rigged.blend, <work>/weights.json, <work>/poses-sheet.png

Weights: Blender's bone heat on each mesh's main part (heat fails on a whole Tripo mesh with fur cards and plates:
every bone came out empty on the wolf); a main part so full of holes that heat finds no solution (the Leshy) is
solved on a watertight voxel copy and takes its nearest point's weights. A loose part (a claw, a plate, a sword) belongs to the limb it touches most:
on a limb it takes that limb's bones by distance (Tripo made the centaur's forearm-and-hand one loose piece: copied
from the elbow skin it left both hand bones empty), what lies far from every bone of it (a blade beyond the fist)
moves rigidly with its nearest close point; on the trunk it copies the nearest skin vertex. (The centaur's sword
first followed the front leg its tip touched.) Then: at most
four influences, and a vertex of one limb never listens to another limb's bones (the pheasant's toes followed the
other foot).
"""
import argparse
import json
import math
import os
import sys
import traceback

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector
from mathutils.kdtree import KDTree

LIMB_ROLES = ('hind', 'front', 'biped', 'arm', 'wing', 'chain')


def args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser(prog='autorig_build.py')
    ap.add_argument('--work', required=True)
    ap.add_argument('--no-sheet', action='store_true')
    ap.add_argument('--solid', action='store_true', help='bone heat on the watertight voxel copy even when the mesh '
                    'itself gives a solution (a Tripo goat: heat reached every vertex, yet its hoof bones held the whole '
                    'lower leg and the spine leaked into the shins)')
    return ap.parse_args(argv)


def loose_parts(me):
    """Connected pieces of the mesh; vertices at the same place count as joined (a GLB splits every UV seam: a man
    came in as 4126 pieces, bone heat ran on the largest, a scrap, and left every limb bone empty)."""
    n = len(me.vertices)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for e in me.edges:
        a, b = find(e.vertices[0]), find(e.vertices[1])
        if a != b:
            parent[a] = b
    co = np.array([v.co for v in me.vertices])
    _, same = np.unique(np.round(co / max(float(np.ptp(co[:, 2])), 1e-9) * 1e5).astype(np.int64), axis=0,
                        return_inverse=True)
    first = {}
    for i, k in enumerate(np.asarray(same).ravel()):
        if k in first:
            a, b = find(i), find(first[k])
            if a != b:
                parent[a] = b
        else:
            first[k] = i
    roots = np.array([find(i) for i in range(n)])
    _, part = np.unique(roots, return_inverse=True)
    return part


def solid_copy(me, voxel):
    """A closed surface with nothing inside it round the volume a mesh encloses, for bone heat: heat counts a bone
    only where the skin sees it, and a Tripo body with inner walls (the Leshy, 2026-10-20) or a voxel remesh of an
    open one (a shell with a second wall inside) hides every bone. The surface is marked in a voxel grid, its holes
    sealed by one voxel, the outside flooded in from the grid's border; every voxel the flood does not reach is inside,
    and the faces between inside and outside are the new mesh (in the same local space as `me`)."""
    co = np.array([v.co for v in me.vertices], float)
    me.calc_loop_triangles()
    tri = np.array([t.vertices[:] for t in me.loop_triangles], int)
    lo = co.min(0) - 3 * voxel
    dims = np.ceil((co.max(0) + 3 * voxel - lo) / voxel).astype(int) + 1
    surf = np.zeros(dims, bool)
    A, B, C = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
    steps = np.ceil(np.max([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1),
                            np.linalg.norm(A - C, axis=1)], axis=0) / (.5 * voxel)).astype(int).clip(1, 200)
    for n in np.unique(steps):
        sel = steps == n
        u, v = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
        keep = (u + v) <= n
        u, v = u[keep] / n, v[keep] / n
        P = (A[sel][:, None] * (1 - u - v)[None, :, None] + B[sel][:, None] * u[None, :, None]
             + C[sel][:, None] * v[None, :, None]).reshape(-1, 3)
        idx = np.floor((P - lo) / voxel).astype(int)
        surf[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    def grow(m):
        g = m.copy()
        g[1:] |= m[:-1]
        g[:-1] |= m[1:]
        g[:, 1:] |= m[:, :-1]
        g[:, :-1] |= m[:, 1:]
        g[:, :, 1:] |= m[:, :, :-1]
        g[:, :, :-1] |= m[:, :, 1:]
        return g
    wall = grow(surf)                                  # holes up to about a voxel wide sealed
    out = np.zeros(dims, bool)
    out[0], out[-1], out[:, 0], out[:, -1], out[:, :, 0], out[:, :, -1] = True, True, True, True, True, True
    out &= ~wall
    while True:
        nxt = grow(out) & ~wall
        if nxt.sum() == out.sum():
            break
        out = nxt
    solid = ~out
    verts, faces, at = [], [], {}

    def vid(p):
        k = (int(p[0]), int(p[1]), int(p[2]))
        if k not in at:
            at[k] = len(verts)
            verts.append(lo + np.array(k, float) * voxel)
        return at[k]
    # one square face wherever an inside voxel meets an outside one, wound to face outward
    for axis in range(3):
        for sign in (1, -1):
            nb = np.roll(solid, -sign, axis=axis)
            edge = solid & ~nb
            if sign == 1:
                edge[tuple(slice(-1, None) if a == axis else slice(None) for a in range(3))] = solid[
                    tuple(slice(-1, None) if a == axis else slice(None) for a in range(3))]
            else:
                edge[tuple(slice(0, 1) if a == axis else slice(None) for a in range(3))] = solid[
                    tuple(slice(0, 1) if a == axis else slice(None) for a in range(3))]
            o1, o2 = [a for a in range(3) if a != axis]
            for c in np.argwhere(edge):
                base = c.copy()
                if sign == 1:
                    base[axis] += 1
                q = []
                for d1, d2 in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    p = base.copy()
                    p[o1] += d1
                    p[o2] += d2
                    q.append(vid(p))
                # the right-hand turn of (o1, o2) is +axis for axes in cyclic order: flip where it points inward
                cyclic = (o1 - axis) % 3 == 1
                faces.append(q if (sign == 1) == cyclic else q[::-1])
    new = bpy.data.meshes.new(me.name + '_solid')
    new.from_pydata([tuple(v) for v in verts], [], faces)
    new.update()
    print(f'AUTORIG_HEAT solid copy: {int(solid.sum())} voxels inside, {len(faces)} faces', flush=True)
    return new


def heat_on_main(obj, arm, chains, solid=False):
    """Bone heat on the object's big connected pieces; small loose parts from the limb they touch most."""
    me = obj.data
    part = loose_parts(me)
    # bone heat on every big piece (a clothed man: jacket, trousers, boots, head; 2% of the vertices or more), the
    # small ones (claws, plates, fur cards: heat failed on all of a wolf's) by the loose-part rules below
    sizes = np.bincount(part)
    main = int(np.argmax(sizes))
    mains = set(np.nonzero(sizes >= .02 * len(part))[0].tolist()) | {main}
    keep = np.nonzero(np.isin(part, sorted(mains)))[0]
    # a copy holding only the main part, parented with automatic weights
    tmp = obj.copy()
    tmp.data = me.copy()
    bpy.context.scene.collection.objects.link(tmp)
    bpy.ops.object.select_all(action='DESELECT')
    tmp.select_set(True)
    bpy.context.view_layer.objects.active = tmp
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    drop = np.ones(len(me.vertices), bool)
    drop[keep] = False
    for i in np.nonzero(drop)[0]:
        tmp.data.vertices[int(i)].select = True
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.delete(type='VERT')
    # seams split for the UVs welded shut: bone heat needs one surface (it failed on a GLB man, every bone empty)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=1e-5 * max(obj.dimensions))
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    tmp.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    names = [b.name for b in arm.data.bones if b.use_deform]
    W_main = np.zeros((len(keep), len(names)))
    gid = {g.index: g.name for g in tmp.vertex_groups}
    col = {n: k for k, n in enumerate(names)}
    W_tmp = np.zeros((len(tmp.data.vertices), len(names)))
    for i, v in enumerate(tmp.data.vertices):
        for g in v.groups:
            n = gid.get(g.group)
            if n in col:
                W_tmp[i, col[n]] = g.weight
    # a surface full of holes and of edges shared by three or more faces gives bone heat no solution at all (Tripo's
    # Leshy, 2026-10-20: 1739 open and 1938 non-manifold edges on its body, every bone empty): the heat then runs on a
    # watertight copy of the piece built from voxels (1% of its size), and each vertex takes the weights of the copy's
    # nearest point below. Only when heat reached under half the piece: every mesh it solved keeps its own weights
    reached = float((W_tmp.sum(1) > 1e-6).mean()) if len(W_tmp) else 0.0
    if reached < .5 or solid:
        voxel = .01 * max(tmp.dimensions)
        print(f'AUTORIG_HEAT bone heat reached {reached * 100:.0f}% of {obj.name}\'s main piece (inner walls or holes '
              f'hide the bones from the skin): heat on a solid copy of {voxel * 100:.1f} cm voxels', flush=True)
        bpy.ops.object.select_all(action='DESELECT')
        tmp.parent = None
        tmp.vertex_groups.clear()
        for mod in list(tmp.modifiers):
            tmp.modifiers.remove(mod)
        old = tmp.data
        tmp.data = solid_copy(old, voxel)
        bpy.data.meshes.remove(old)
        rem = tmp.modifiers.new('Smooth', 'REMESH')
        rem.mode = 'VOXEL'
        rem.voxel_size = voxel
        rem.adaptivity = 0.0
        bpy.context.view_layer.objects.active = tmp
        tmp.select_set(True)
        bpy.ops.object.modifier_apply(modifier=rem.name)
        tmp.select_set(True)
        arm.select_set(True)
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        gid = {g.index: g.name for g in tmp.vertex_groups}
        W_tmp = np.zeros((len(tmp.data.vertices), len(names)))
        for i, v in enumerate(tmp.data.vertices):
            for g in v.groups:
                n = gid.get(g.group)
                if n in col:
                    W_tmp[i, col[n]] = g.weight
        print(f'AUTORIG_HEAT the watertight copy: {len(tmp.data.vertices)} vertices, heat reached '
              f'{(W_tmp.sum(1) > 1e-6).mean() * 100:.0f}%', flush=True)
    # back onto the main part's own vertices, by place (the weld renumbered them)
    kd_t = KDTree(len(tmp.data.vertices))
    for i, v in enumerate(tmp.data.vertices):
        kd_t.insert(v.co, i)
    kd_t.balance()
    for r, i in enumerate(keep):
        W_main[r] = W_tmp[kd_t.find(me.vertices[int(i)].co)[1]]
    empty = W_main.sum(1) <= 1e-6
    bpy.data.objects.remove(tmp)
    # every vertex: its own weights if it is on the main part (and heat reached it), else the nearest main vertex's
    co = np.array([v.co for v in me.vertices])
    ok = keep[~empty]
    kd = KDTree(len(ok))
    for k, i in enumerate(ok):
        kd.insert(Vector(co[i]), k)
    kd.balance()
    row_of = {int(i): k for k, i in enumerate(keep)}
    W = np.zeros((len(co), len(names)))
    near = np.zeros(len(co), int)                  # nearest weighted main vertex (a row of W_main)
    dist = np.zeros(len(co))
    for i in range(len(co)):
        k = row_of.get(i)
        if k is not None and not empty[k]:
            near[i], dist[i] = k, 0.0
        else:
            _, j, d = kd.find(Vector(co[i]))
            near[i], dist[i] = row_of[int(ok[j])], d
        W[i] = W_main[near[i]]
    chain_of = np.array([chains.get(n, -1) for n in names])
    owner = chain_of[np.argmax(W_main, 1)]         # the limb (or -1, the trunk) of each main vertex
    height = float(np.ptp(co[:, 2]))
    touch = .015 * height
    to_mesh = obj.matrix_world.inverted() @ arm.matrix_world
    seg = {c: (np.array(to_mesh @ arm.data.bones[n].head_local), np.array(to_mesh @ arm.data.bones[n].tail_local))
           for c, n in enumerate(names)}

    def bone_dist(X, cols):
        D = np.zeros((len(X), len(cols)))
        for k, c in enumerate(cols):
            h, t = seg[c]
            d = t - h
            u = np.clip((X - h) @ d / max(d @ d, 1e-12), 0, 1)
            D[:, k] = np.linalg.norm(X - (h + u[:, None] * d), axis=1)
        return D
    for q in np.unique(part):
        if q in mains:
            continue
        idx = np.nonzero(part == q)[0]
        lim = max(touch, float(np.quantile(dist[idx], .05)))
        on = idx[dist[idx] <= lim]
        votes = np.bincount(owner[near[on]] + 1)
        mine = int(np.argmax(votes)) - 1
        anchor = on[owner[near[on]] == mine]
        if mine >= 0:
            # on a limb: the limb's bones by distance (two nearest, inverse fourth power); far points rigid with the
            # nearest close point of the part
            cols = [c for c in range(len(names)) if chain_of[c] == mine]
            D = bone_dist(co[idx], cols)
            close = D.min(1) <= .06 * height
            ends = [k for k, c in enumerate(cols) if not any(chain_of[cc] == mine and arm.data.bones[names[cc]].parent
                    and arm.data.bones[names[cc]].parent.name == names[c] for cc in range(len(names)))]
            far = ~close
            if close.any() and D.min(1).max() > .1 * height and \
                    np.isin(np.argmin(D[far], 1), ends).mean() > .5:
                # reaching on past the limb's end bone: a held prop (a sword), rigid with the bone nearest its handle
                c = cols[int(np.bincount(np.argmin(D[close], 1), minlength=len(cols)).argmax())]
                W[idx] = 0.0
                W[idx, c] = 1.0
                continue
            if close.any():
                for r, i in enumerate(idx[close]):
                    d = D[np.nonzero(close)[0][r]]
                    two = np.argsort(d)[:2]
                    w = 1.0 / np.maximum(d[two], 1e-4) ** 4
                    W[i] = 0.0
                    W[i, [cols[k] for k in two]] = w / w.sum()
                kd_c = KDTree(int(close.sum()))
                for k, i in enumerate(idx[close]):
                    kd_c.insert(Vector(co[i]), k)
                kd_c.balance()
                near_ids = idx[close]
                for i in idx[~close]:
                    _, k, _ = kd_c.find(Vector(co[i]))
                    W[i] = W[near_ids[k]]
                continue
        kd_a = KDTree(len(anchor))
        for k, i in enumerate(anchor):
            kd_a.insert(Vector(co[i]), k)
        kd_a.balance()
        trunk_cols = [c for c in range(len(names)) if chain_of[c] == -1]
        Dt = bone_dist(co[idx], trunk_cols) if mine < 0 and trunk_cols else None
        for r, i in enumerate(idx):
            if dist[i] <= lim and owner[near[i]] == mine:
                continue
            _, k, _ = kd_a.find(Vector(co[i]))
            W[i] = W_main[near[anchor[k]]]
            if Dt is not None:
                # a part of the trunk lying away from the skin it touches (Tripo made a centaur's head a separate
                # piece: copied from the neck skin, the Head bone moved nothing) blends to its nearest trunk bones
                far = float(np.clip((dist[i] - .02 * height) / (.04 * height), 0, 1))
                if far > 0:
                    d = Dt[r]
                    two = np.argsort(d)[:2]
                    w = 1.0 / np.maximum(d[two], 1e-4) ** 4
                    Wb = np.zeros(len(names))
                    Wb[[trunk_cols[k2] for k2 in two]] = w / w.sum()
                    W[i] = (1 - far) * W[i] + far * Wb
    return W, names, part, main


def clean(W, names, roles, chains):
    """At most four influences; a vertex keeps no weight on another limb than its own strongest one."""
    chain_of = np.array([chains.get(n, -1) for n in names])
    top = np.argmax(W, 1)
    own = chain_of[top]
    for i in range(len(W)):
        if own[i] >= 0:
            W[i, (chain_of >= 0) & (chain_of != own[i])] = 0.0
        order = np.argsort(-W[i])
        W[i, order[4:]] = 0.0
        s = W[i].sum()
        if s > 0:
            W[i] /= s
    return W


def main():
    a = args()
    work = os.path.abspath(a.work)
    spec = json.load(open(os.path.join(work, 'joints.json')))
    bpy.ops.wm.open_mainfile(filepath=os.path.join(work, 'model.blend'))
    scene = bpy.context.scene
    meshes = [o for o in scene.objects if o.type == 'MESH']
    arm_data = bpy.data.armatures.new('AutoRig')
    arm = bpy.data.objects.new('AutoRig', arm_data)
    scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    up, lateral = Vector((0, 0, 1)), Vector(spec.get('left', (1, 0, 0)))
    for b in spec['bones']:
        eb = arm_data.edit_bones.new(b['name'])
        eb.head, eb.tail = Vector(b['head']), Vector(b['tail'])
        if (eb.tail - eb.head).length < 1e-4:
            eb.tail = eb.head + Vector((0, 0, .01))
        d = (eb.tail - eb.head).normalized()
        eb.align_roll(lateral if abs(d @ lateral) < .7 else up)   # local Z ~ sideways: a bend is a Z turn
        if b['parent']:
            eb.parent = arm_data.edit_bones[b['parent']]
        eb.use_connect = False
        eb.use_deform = b['role'] != 'root'
    bpy.ops.object.mode_set(mode='OBJECT')
    roles = {b['name']: b['role'] for b in spec['bones']}
    # limb chains: the bones under each limb's first bone
    chains, k = {}, 0
    kids = {}
    for b in spec['bones']:
        kids.setdefault(b['parent'], []).append(b['name'])
    for b in spec['bones']:
        if b['role'] in LIMB_ROLES and roles.get(b['parent']) not in LIMB_ROLES:
            stack = [b['name']]
            while stack:
                n = stack.pop()
                chains[n] = k
                stack += kids.get(n, [])
            k += 1
    report = {}
    for o in meshes:
        W, names, part, main = heat_on_main(o, arm, chains, solid=a.solid)
        W = clean(W, names, roles, chains)
        o.vertex_groups.clear()
        groups = {n: o.vertex_groups.new(name=n) for n in names}
        for c, n in enumerate(names):
            idx = np.nonzero(W[:, c] > 1e-4)[0]
            for i in idx:
                groups[n].add([int(i)], float(W[i, c]), 'REPLACE')
        for mod in list(o.modifiers):
            if mod.type == 'ARMATURE':
                o.modifiers.remove(mod)
        mod = o.modifiers.new('Armature', 'ARMATURE')
        mod.object = arm
        mw = o.matrix_world.copy()
        o.parent = arm
        o.matrix_parent_inverse = arm.matrix_world.inverted()
        o.matrix_world = mw
        mass = W.sum(0)
        report[o.name] = {'vertices': len(W), 'loose_parts': int(part.max() + 1),
                          'empty_bones': [n for c, n in enumerate(names) if mass[c] < 1e-3]}
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(work, 'rigged.blend'))
    json.dump(report, open(os.path.join(work, 'weights.json'), 'w'), indent=1)
    print('AUTORIG_BUILD', json.dumps(report), flush=True)
    if not a.no_sheet:
        pose_sheet(scene, arm, meshes, spec, work)


def pose_sheet(scene, arm, meshes, spec, work):
    """The rig in four test poses (rest, legs and arms bent, the spine and neck turned, everything at once), seen
    from the side, the front and three-quarter: bad weights show as tearing, candy-wrapper twists or a limb dragging
    the body."""
    bpy.context.view_layer.update()
    lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
    for o in meshes:
        for v in o.data.vertices:
            w = o.matrix_world @ v.co
            lo, hi = Vector(map(min, lo, w)), Vector(map(max, hi, w))
    world = bpy.data.worlds.new('sheet')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (.42, .41, .40, 1)
    scene.world = world
    for name, rot, energy in (('key', (math.radians(50), 0, math.radians(-35)), 3.2),
                              ('fill', (math.radians(60), 0, math.radians(140)), 1.2)):
        light = bpy.data.lights.new(name, 'SUN')
        light.energy = energy
        ob = bpy.data.objects.new(name, light)
        ob.rotation_euler = rot
        scene.collection.objects.link(ob)
    try:
        scene.render.engine = 'BLENDER_EEVEE'
    except TypeError:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.eevee.taa_render_samples = 8
    scene.render.resolution_x = scene.render.resolution_y = 360
    scene.view_settings.view_transform = 'AgX'
    centre = (lo + hi) / 2
    size = (hi - lo).length
    cam_data = bpy.data.cameras.new('sheet')
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = size * 1.05
    cam = bpy.data.objects.new('sheet', cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    views = {'side': Vector((1, 0, 0)), 'front': Vector((0, 1, 0)), '3/4': Vector((.7, .7, -.25)).normalized()}
    hinge = {b['name'] for b in spec['bones'] if any(s in b['name'] for s in ('Shin', 'Forearm', 'LowerArm'))}
    def pose(kind):
        for pb in arm.pose.bones:
            pb.rotation_mode = 'XYZ'
            pb.rotation_euler = (0, 0, 0)
            role = next(b['role'] for b in spec['bones'] if b['name'] == pb.name)
            if kind in ('bend', 'all') and pb.name in hinge:
                pb.rotation_euler = (0, 0, math.radians(45 if pb.name.endswith('.L') else -45))
            if kind in ('bend', 'all') and role in ('hind', 'front', 'biped', 'arm', 'wing') and pb.name not in hinge \
                    and pb.parent is not None and pb.parent.name not in hinge and pb.name.split('.')[0] in (
                    'Thigh', 'UpperArm', 'WingUpper'):
                pb.rotation_euler = (0, 0, math.radians(-30))
            if kind in ('bend', 'all') and role == 'chain' and pb.parent is not None and \
                    next(b['role'] for b in spec['bones'] if b['name'] == pb.parent.name) == 'chain':
                # many legs, tentacles, extra pairs: every bone below the first bends
                pb.rotation_euler = (0, 0, math.radians(30 if pb.name.endswith('.L') else -30))
            if kind in ('turn', 'all') and role in ('neck', 'trunk', 'tail') and pb.name != 'Hips':
                pb.rotation_euler = (math.radians(12), 0, math.radians(10))
    import tempfile
    tiles = []
    tmpdir = tempfile.mkdtemp()
    for kind in ('rest', 'bend', 'turn', 'all'):
        pose(kind)
        bpy.context.view_layer.update()
        row = []
        for vname, look in views.items():
            cam.location = centre - look * size * 3
            cam.rotation_euler = look.to_track_quat('-Z', 'Y').to_euler()
            path = os.path.join(tmpdir, f'{kind}-{vname.replace("/", "")}.png')
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            row.append(path)
        tiles.append(row)
    import subprocess
    out = os.path.join(work, 'poses-sheet.png')
    inputs = []
    for row in tiles:
        inputs += row
    args = ['ffmpeg', '-v', 'error', '-y']
    for p in inputs:
        args += ['-i', p]
    n = len(inputs)
    args += ['-filter_complex', ''.join(f'[{i}]' for i in range(n)) + f'xstack=inputs={n}:layout=' +
             '|'.join(f'{(i % 3) * 360}_{(i // 3) * 360}' for i in range(n)), out]
    subprocess.run(args, check=True)
    print('AUTORIG_SHEET', out, flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
    sys.exit(1)
