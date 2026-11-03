"""A creature built from its project folder: what tracking, solving and the review need, with no code to write.

Defaults (all can be overridden in creature.json, see docs/CONFIG.md):
  fitted bones   every bone that carries real skin weight, and its parents; fingers, eyes and end bones keep their
                 start-pose basis (a 480p video cannot resolve them)
  hinges         knees and elbows found by name (Mixamo, Rigify, Unreal and common names): they bend one way only,
                 the way found by turning them on the rig
  limbs          arms, legs, hands, feet (and props): dense seeds, lighter smoothing
  joint ranges   ankles, wrists, toes, every spine and neck segment, the head, collarbones and hips keep to ranges a
                 body can reach, by name (around the start pose, which is a relaxed stance); "anatomy": "none" turns
                 them off. Bones above the root (a Mixamo-style "Root" over "Hips") are not fitted: the root turns the
                 whole body already, and two bones doing it split a bend between them
  priors         twist about the bone, collarbones, spine and toes stay near the start pose unless the video says
                 otherwise; the depth along a single camera stays near the start
"""
import json
import re

import cv2
import numpy as np

from . import clip as clipmod
from .camera import depth as view_depth
from .common import write_json
from .raster import load_textures, rasterize, render
from .rig import Rig

FINGER = re.compile(r'(finger|thumb|index|middle|ring|pinky|pinkie|little|eye|jaw|tongue|teeth|_end$|end$)', re.I)
FOOT_LOW = .10      # a limb's branching bone this close to the floor (share of the height, start pose) is its foot
KNEE = re.compile(r'((^|[^p])leg(?!.*up)|shin|calf|knee|lowerleg|lower_leg)', re.I)
ELBOW = re.compile(r'(forearm|fore_arm|lowerarm|lower_arm|elbow)', re.I)
LIMB = re.compile(r'(arm|leg|hand|foot|toe|shin|thigh|calf|forearm|knee|elbow|wrist|ankle|paw|hoof|wing)', re.I)
SPINE = re.compile(r'(spine|chest|neck|torso|back|abdomen)', re.I)
CLAVICLE = re.compile(r'(shoulder|clavicle|collar)', re.I)
TOE = re.compile(r'toe', re.I)
# Joint ranges in degrees from the start pose: (bend, twist about the bone, sideways), each channel's role found from
# the geometry (Creature.axis_roles; for a Mixamo skeleton bend is X, twist Y, sideways Z). Without them a limb that the
# camera cannot see well (hidden by a prop or the body) bends anywhere the outline allows: a foot turned 100 degrees
# up, a wrist turned over, the lower back folded 90 degrees while the pelvis stood still (a man lifting a stone).
RANGES = [
    (re.compile(r'(foot|ankle)', re.I), (45, 20, 25)),
    # animals walk on their toes: a wolf's bend 40 degrees in a walk; toes along the whole sole (the auto-rig) curl
    # under a lifted paw by 90 (at 60 the curled paw's toes pointed into the floor)
    (re.compile(r'toe', re.I), (90, 10, 10)),
    (re.compile(r'(hand|wrist)', re.I), (70, 40, 35)),
    # a paw or hoof (a wolf's wrist or hock): the bench's run, walk and bite turn the hock up to 90 degrees from the
    # start, the front wrist 25; unbounded, a poorly seen front paw folded 160 degrees back
    (re.compile(r'(paw|hoof)', re.I), (100, 20, 25)),
    (re.compile(r'(spine|chest|torso|abdomen)', re.I), (35, 30, 30)),
    (re.compile(r'neck', re.I), (45, 50, 35)),
    (re.compile(r'(^|[^a-z])head$', re.I), (60, 50, 35)),  # a howling wolf raised its head past 45 for most of the clip
    (re.compile(r'(shoulder|clavicle|collar)', re.I), (40, 20, 40)),
    (re.compile(r'(upleg|up_leg|thigh)', re.I), (130, 45, 60)),
]
# A leg the creature stands on, by its shape (whatever its bones are called: Paw, Hoof, Scapula, UpperArm, femur):
# twist about the bone and the swing out of the leg's plane, degrees from the start pose. Without them a wolf's
# front legs spread like a frog's under the body and its paws turned 70 degrees about themselves.
LEG_TWIST, LEG_SIDE = 30, 40
# An upper arm (not a leg) swings forward up over the head but back only so far: a hidden far arm read on a spine of
# the back (an alien's run) swung back 100-110 degrees while the video showed about 60. Degrees back from the start pose.
UPPER_ARM = re.compile(r'(^|[^a-z]|left|right|upper)arm([._]|$)', re.I)
ARM_BACK = 75
PAW_BACK = 20       # a paw's (wrist's, hock's) bend against its own way, degrees from the start pose


class Creature:
    def __init__(self, project, clip_cfg):
        self.project, self.cfg = project, clip_cfg
        rig_dir = project.rig_dir
        self.data = json.loads((rig_dir / 'rig.json').read_text(encoding='utf-8'))
        m = np.load(rig_dir / 'mesh.npz')
        self.mesh = {'coords': m['verts'], 'triangles': m['tris'], 'tris': m['tris'], 'tri_uv': m['tri_uv'],
                     'tri_material': m['tri_material']}
        self.idx, self.wts = m['idx'], m['wts']
        self.name = project.name
        names = [b['name'] for b in self.data['bones']]
        nb = len(names)
        fitted = self.cfg.get('fit_bones', 'auto')
        if fitted == 'auto':
            mass = np.zeros(nb)
            body = self.idx[:, 0] < nb
            np.add.at(mass, self.idx[body].ravel(), self.wts[body].ravel())
            keep = {j for j in range(nb) if mass[j] > .004 * mass.sum() and not FINGER.search(names[j])}
            parents = {b['name']: b['parent'] for b in self.data['bones']}
            for j in list(keep):
                p = parents[names[j]]
                while p is not None:
                    keep.add(names.index(p))
                    p = parents[p]
            fitted = [n for n in names if names.index(n) in keep]
        self.rig = Rig(self.data, fitted, self.cfg.get('root_bone'))
        above = [n for n in fitted if names.index(n) != self.rig.root and self.rig.desc[names.index(n), self.rig.root]]
        if above and self.cfg.get('fit_bones', 'auto') == 'auto':
            fitted = [n for n in fitted if n not in above]
            self.rig = Rig(self.data, fitted, self.cfg.get('root_bone'))
        self.n = self.rig.n
        M = np.array(self.data['armature_world'], float)
        self.up = np.linalg.solve(M[:3, :3], [0.0, 0.0, 1.0])        # world up in armature space
        self.up /= np.linalg.norm(self.up)
        X0 = self.rig.skin(self.p0(None), self.mesh['coords'], self.idx, self.wts)
        h = X0 @ self.up
        self.floor = float(self.cfg.get('floor', h.min()))
        self.height = float(h.max() - h.min())
        self.facing = self.find_facing(X0)
        self.role = self.axis_roles()
        self.hinge = self.find_hinges()
        self.legs = self.standing_legs()
        self.textures = load_textures(rig_dir, self.data.get('materials', []))
        self._clip = {}

    # ----- body ------------------------------------------------------------------------------------------------
    def find_facing(self, X0):
        """The way the character faces, in armature space: from the feet (heel to toe; a hoofed animal's hooves, whose
        Foot bones are the near-vertical cannons: the goat's four cannons alone read it facing backward, 2026-10-30),
        else creature.json."""
        if 'facing' in self.cfg:
            f = np.array({'+X': (1, 0, 0), '-X': (-1, 0, 0), '+Y': (0, 1, 0), '-Y': (0, -1, 0)}[self.cfg['facing']], float)
            M = np.array(self.data['armature_world'], float)
            f = np.linalg.solve(M[:3, :3], f)
            return f / np.linalg.norm(f)
        rig = self.rig
        H, T = rig.bone_points(self.p0(None))
        v = np.zeros(3)
        for j, n in enumerate(rig.names):
            if re.search(r'foot|toe|hoof', n, re.I):
                d = T[j] - H[j]
                v += d - (d @ self.up) * self.up
        if np.linalg.norm(v) < 1e-9:
            raise SystemExit('cannot tell which way the creature faces: set "facing" in creature.json (+X, -X, +Y, -Y)')
        return v / np.linalg.norm(v)

    def bend_axis(self, j):
        """The channel (0 X, 2 Z) a knee or elbow bends about: the axis nearest the normal of the plane its bone and
        the one above it make in the rig's own rest pose (where the joint's axes are set up; a quadruped's joints
        stand a little bent there); when the two lie in line (a T-pose arm), the bend role from axis_roles."""
        rig = self.rig
        par = rig.parents[j]
        if par >= 0 and j in rig.chan_of:
            rest = np.zeros(rig.n)
            Hd, Tl = rig.bone_points(rest)
            u, v = Tl[par] - Hd[par], Tl[j] - Hd[j]
            n = np.cross(u, v)
            if np.linalg.norm(n) > np.sin(np.radians(8)) * np.linalg.norm(u) * np.linalg.norm(v):
                _, _, axes = rig.fk(rest)
                chans = dict(rig.chan_of[j])
                axis_of = {k: ax for _, ax, k in axes}
                return max((a for a in (0, 2) if a in chans), key=lambda a: abs(axis_of[chans[a]] @ n))
        return next((a for a in (0, 2) if self.role.get((j, a)) == 'bend'), 0)

    def find_hinges(self):
        """Knees bend back, elbows forward: the sign of the turn about the bone's bend axis that does it, measured on
        the rig (the bend axis from the geometry: X on a Mixamo skeleton, Z on the wolves')."""
        rig = self.rig
        cfg = self.cfg.get('hinges', 'auto')
        out = {}
        self.reversed = set()                 # knees that fold forward (a bird's leg): the name said otherwise
        for j, a in rig.channels:
            if a != self.bend_axis(j):
                continue
            name = rig.names[j]
            way = None
            if isinstance(cfg, dict):
                way = cfg.get(name)
            elif cfg == 'auto':
                way = 'back' if KNEE.search(name) and not re.search(r'upleg|up_leg|thigh', name, re.I) else \
                    'front' if ELBOW.search(name) else None
            if way is None:
                continue
            p = self.p0(None)
            tip0 = rig.bone_points(p)[1][j]
            p2 = p.copy()
            p2[rig.index(name, a)] += .3
            d = (rig.bone_points(p2)[1][j] - tip0) @ self.facing
            out[name] = (1.0 if d < 0 else -1.0) if way == 'back' else (1.0 if d > 0 else -1.0)
            # the name says what a joint usually is; a joint the start pose already shows bent says which way it
            # folds: a bird's or a reverse-jointed alien's "knee" folds forward (named Shin by the auto-rig, it was
            # held to a human knee's way and stood on its limit in 56-78 of 121 frames)
            if not isinstance(cfg, dict) or name not in cfg:
                bent = self.rest_bend(j, rig.index(name, a))
                if bent is not None:
                    if bent != out[name] and way == 'back':
                        self.reversed.add(j)
                    out[name] = bent
        return out

    def rest_bend(self, j, c, min_deg=12.0):
        """+1 or -1: the turn of channel c that folds bone j further onto its parent, when the start pose shows the
        joint bent by more than min_deg (else None: a straight joint does not tell)."""
        rig = self.rig
        par = rig.parents[j]
        if par < 0:
            return None

        def angle(q):
            H, T = rig.bone_points(q)
            u, v = T[par] - H[par], T[j] - H[j]
            return float(np.degrees(np.arccos(np.clip(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12),
                                                      -1, 1))))
        p = self.p0(None)
        a0 = angle(p)
        if a0 < min_deg:
            return None
        q = p.copy()
        q[c] += .15
        return 1.0 if angle(q) > a0 else -1.0

    def p0(self, clip):
        if not hasattr(self, '_p0'):
            self._p0 = self.rig.start_params()
        return self._p0.copy()

    def bounds(self, clip):
        rig = self.rig
        lo, hi = np.full(rig.n, -np.inf), np.full(rig.n, np.inf)
        p0 = self.p0(clip)
        reach = 5.0 * self.height
        lo[:3], hi[:3] = p0[:3] - reach, p0[:3] + reach
        own = self.cfg.get('joints', {})
        for j, a in rig.channels:
            name = rig.names[j]
            c = rig.index(name, a)
            if j == rig.root:
                # the whole body's turn (a rotation vector): any way round, a roll onto the back, a spin, a
                # somersault (the old twist limit of 70 degrees about the root bone held every roll and turn back)
                lo[c], hi[c] = p0[c] - 4 * np.pi, p0[c] + 4 * np.pi       # two spins either way
                continue
            r = 2.8 if a != 1 else 1.2                           # twist about the bone: within 70 degrees
            bend = self.bend_axis(j)
            if name in self.hinge:
                # the bend is free one way (below); the forearm turns the palm over (about 80 degrees each way),
                # a bent knee turns a little, neither bends sideways
                r = 2.8 if a == bend else (1.4 if ELBOW.search(name) else .45) if a == 1 else .2
            lo[c], hi[c] = p0[c] - r, p0[c] + r
            if name in self.hinge and a == bend:
                s = self.hinge[name]
                # one way only: from straight to fully bent. Straight is where the bone lines up with its parent,
                # measured on the rig: a creature standing on bent legs (a wolf's knee) straightens past its start
                # pose in a gallop's push-off, which the start pose taken as straight held folded
                st = self.straight(j, c)
                lo[c], hi[c] = (min(p0[c], st) - .05, 2.7) if s > 0 else (-2.7, max(p0[c], st) + .05)
            if self.cfg.get('anatomy', 'auto') != 'none' and name not in self.hinge and not FINGER.search(name):
                lim = next((d for rx, d in RANGES if rx.search(name)), None)
                if lim is not None and rig.parents[j] in getattr(self, 'reversed', ()) and RANGES[0][0].search(name):
                    # below a knee that folds forward, the "foot" is a bird's or a dog's long hock: the paw's range
                    lim = next(d for rx, d in RANGES if rx.search('paw'))
                if lim is not None:
                    rr_ = np.radians(lim[('bend', 'twist', 'side').index(self.role.get((j, a), 'bend'))])
                    lo[c], hi[c] = max(lo[c], p0[c] - rr_), min(hi[c], p0[c] + rr_)
                    fold = self.paw_fold(j, a) if RANGES[3][0].search(name) and self.role.get((j, a)) == 'bend' \
                        else None
                    if fold is not None:
                        # a front paw (a wrist, a horse's knee) folds back, a hind one (a hock) forward, the other way
                        # only PAW_BACK degrees: free both ways, a centaur's galloping front hoof kinked forward
                        other = np.radians(PAW_BACK)
                        if fold > 0:
                            lo[c] = max(lo[c], p0[c] - other)
                        else:
                            hi[c] = min(hi[c], p0[c] + other)
            if self.cfg.get('anatomy', 'auto') != 'none' and UPPER_ARM.search(name) and j not in self.legs:
                swing = self.swing_back(j)
                if swing is not None and swing[0] == a:
                    back = np.radians(ARM_BACK)
                    if swing[1] > 0:                      # + turns it back
                        hi[c] = min(hi[c], p0[c] + back)
                    else:
                        lo[c] = max(lo[c], p0[c] - back)
            # creature.json "leg_limits": true puts the rule back (off since 2026-10-21: the bench, 5 creatures, 11 clips, scored the
            # same with it off, 2.78 deg both ways, the alien combo's worst frames 1 deg worse; a beast that stands on its front limbs and also
            # rears up and spreads them: the roar's arms stayed 40 degrees from the crouch's, the reading 100 px off)
            role = self.role.get((j, a)) if j in self.legs and self.cfg.get('leg_limits', False) else None
            if self.cfg.get('anatomy', 'auto') != 'none' and role in ('twist', 'side') and name not in self.hinge:
                rr_ = np.radians(LEG_TWIST if role == 'twist' else LEG_SIDE)
                lo[c], hi[c] = max(lo[c], p0[c] - rr_), min(hi[c], p0[c] + rr_)
            if name in own and 'range' in own[name]:
                rr = own[name]['range']
                lo[c], hi[c] = (p0[c] - rr, p0[c] + rr) if np.isscalar(rr) else (rr[0], rr[1])
        lo[rig.prop0:], hi[rig.prop0:] = -np.inf, np.inf
        for k in range(len(rig.props)):
            base = rig.prop0 + 6 * k
            lo[base:base + 3], hi[base:base + 3] = -reach, reach
            lo[base + 3:base + 6], hi[base + 3:base + 6] = -2 * np.pi, 2 * np.pi
        return lo, hi, p0

    def paw_fold(self, j, a):
        """+1 or -1: the sign of channel a's turn that folds paw bone j its own way (below an elbow-named bone, a
        front leg: its end moves back; below a knee-named one, a hind leg: forward), or None when its parent says
        neither."""
        rig = self.rig
        par = rig.parents[j]
        if par < 0:
            return None
        pname = rig.names[par]
        way = -1.0 if ELBOW.search(pname) else 1.0 if KNEE.search(pname) and not re.search(r'upleg|up_leg|thigh', pname,
                                                                                              re.I) else None
        if way is None:
            return None
        p = self.p0(None)
        tip = rig.bone_points(p)[1][j]
        q = p.copy()
        q[rig.index(rig.names[j], a)] += .3
        d = float((rig.bone_points(q)[1][j] - tip) @ self.facing)
        if abs(d) < 1e-6:
            return None
        return 1.0 if d * way > 0 else -1.0

    def swing_back(self, j):
        """(axis, sign) of bone j's channel that swings its end most along the facing, the sign of a turn that moves
        it back; measured on the rig in its start pose."""
        rig = self.rig
        p = self.p0(None)
        tip = rig.bone_points(p)[1][j]
        best = None
        for a in (0, 1, 2):
            if (j, a) not in [(jj, aa) for jj, aa in rig.channels]:
                continue
            q = p.copy()
            q[rig.index(rig.names[j], a)] += .3
            d = float((rig.bone_points(q)[1][j] - tip) @ self.facing)
            if best is None or abs(d) > abs(best[1]):
                best = (a, d)
        return None if best is None else (best[0], -1.0 if best[1] > 0 else 1.0)

    def straight(self, j, c):
        """The value of channel c (bone j's bend) at which the bone points the same way as its parent: the joint
        fully straight. The start pose's value when the bone has no parent."""
        rig = self.rig
        p = self.p0(None)
        par = rig.parents[j]
        if par < 0:
            return float(p[c])

        def off(v):
            q = p.copy()
            q[c] = v
            H, T = rig.bone_points(q)
            a, b = T[j] - H[j], T[par] - H[par]
            return float(np.arccos(np.clip(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12), -1, 1)))
        vs = p[c] + np.linspace(-2.7, 2.7, 109)
        v = float(vs[int(np.argmin([off(x) for x in vs]))])
        for step in (.025, .006):
            v = min((v - step, v, v + step), key=off)
        return v

    def active(self, clip):
        return np.arange(self.rig.n)

    def prior(self, clip):
        """Pull toward the start pose, px-equivalent per rad."""
        rig = self.rig
        w = np.zeros(rig.n)
        own = self.cfg.get('joints', {})
        one_view = clip is not None and len(self.views(clip)) == 1
        for j, a in rig.channels:
            name = rig.names[j]
            c = rig.index(name, a)
            if j == rig.root:
                # the body's turn is the video's to say: next to nothing with two views; one camera leaves its
                # depth roll unseen, so a light pull to the start there
                w[c] = 2.0 if one_view else 0.2
                continue
            w[c] = 2.0
            if a == 1:
                w[c] = 25.0
            if SPINE.search(name):
                w[c] = max(w[c], 15.0)
            if CLAVICLE.search(name) and not LIMB.search(name.replace('Shoulder', '')):
                w[c] = 40.0
            if TOE.search(name):
                w[c] = 30.0
            if name in self.hinge and a != self.bend_axis(j):
                w[c] = 60.0
            if name in own and 'prior' in own[name]:
                w[c] = float(own[name]['prior'])
        return w

    def bends(self):
        return []

    def parts(self):
        """Dense seeding groups as bone ids: every limb chain (from its first limb bone down), and every prop."""
        rig = self.rig
        dense = {j for j, n in enumerate(rig.names) if LIMB.search(n) and not FINGER.search(n)}
        if not dense:
            dense = self.limb_chains()                    # names say nothing (femur, tibia, L1...): the shape does
        if isinstance(self.cfg.get('dense_bones'), list):
            dense = {rig.names.index(n) for n in self.cfg['dense_bones']}
        out = {}
        for j in sorted(dense):
            top = j
            while rig.parents[top] >= 0 and rig.parents[top] in dense:
                top = rig.parents[top]
            out.setdefault(rig.names[top], []).append(j)
        for k, pr in enumerate(rig.props):
            out['prop:' + pr['name']] = [rig.nb + k]
        return out

    def limb_chains(self, as_list=False):
        """Limbs by the skeleton's shape, for rigs whose bone names say nothing: a chain of bones that leaves a hub
        (a bone with three or more children: a pelvis, a chest, an insect's or spider's body) and runs on without
        branching (end bones and fingers aside), with skin on at least two of its bones. A foot that branches into its
        toes (a bird's front and hind toes, a paw's claws: every child an unbranched run of bones, the foot within
        FOOT_LOW of the height above the floor in the start pose) ends its leg; before, the branch threw the whole leg
        out and the turkey stood on no legs, so nothing pinned its feet. A set of bone ids, or the chains (hub side
        first) with as_list."""
        rig = self.rig
        mass = np.zeros(rig.nb)
        body = self.idx[:, 0] < rig.nb
        np.add.at(mass, self.idx[body].ravel(), self.wts[body].ravel())
        skinned = mass > .002 * mass.sum()
        kids = {j: [k for k in range(rig.nb) if rig.parents[k] == j and not FINGER.search(rig.names[k])]
                for j in range(rig.nb)}

        def unbranched(j):
            while len(kids[j]) == 1:
                j = kids[j][0]
            return not kids[j]

        low = set()
        if hasattr(self, 'floor') and hasattr(self, 'height'):
            _, tails = rig.bone_points(self.p0(None))
            low = {j for j in range(rig.nb) if tails[j] @ self.up - self.floor <= FOOT_LOW * self.height}
        chains = []
        for hub in (j for j in range(rig.nb) if len(kids[j]) >= 3):
            for c in kids[hub]:
                chain, j = [], c
                while True:
                    chain.append(j)
                    if len(kids[j]) != 1:
                        break
                    j = kids[j][0]
                foot = len(kids[j]) >= 2 and j in low and len(chain) >= 2 and all(unbranched(k) for k in kids[j])
                if (len(kids[j]) <= 1 or foot) and sum(bool(skinned[b]) for b in chain) >= 2:
                    chains.append(chain)
        return chains if as_list else {b for ch in chains for b in ch}

    def axis_roles(self):
        """(bone, channel) -> 'bend' (the axis closest to the creature's side-to-side axis in the start pose: a knee,
        a spine, a neck bending forward and back), 'twist' (about the bone) or 'side' (the other)."""
        rig = self.rig
        lateral = np.cross(self.up, self.facing)
        lateral /= max(np.linalg.norm(lateral), 1e-9)
        _, _, axes = rig.fk(self.p0(None))
        axis_of = {k: ax for _, ax, k in axes}
        Hd, Tl = rig.bone_points(self.p0(None))
        roles = {}
        for j, chans in rig.chan_of.items():
            if j == rig.root:
                continue
            chans = dict(chans)
            along = (Tl[j] - Hd[j]) / max(np.linalg.norm(Tl[j] - Hd[j]), 1e-9)
            ref = self.up if abs(along @ lateral) > .7 else lateral     # an arm held out sideways bends about up
            bend = max((a for a in (0, 2) if a in chans), key=lambda a: abs(axis_of[chans[a]] @ ref), default=None)
            for a in chans:
                roles[(j, a)] = 'twist' if a == 1 else 'bend' if a == bend else 'side'
        return roles

    def standing_legs(self):
        """Bones of the limbs the creature stands on: limb chains whose last two bones' skin reaches the floor in the
        start pose (a leg; an arm, a tail, a wing does not)."""
        X0 = self.rig.skin(self.p0(None), self.mesh['coords'], self.idx, self.wts)
        h = X0 @ self.up - self.floor
        out = set()
        for chain in self.limb_chains(as_list=True):
            mine = np.isin(self.idx[:, 0], chain[-2:]) & (self.wts[:, 0] > .5)
            if mine.any() and h[mine].min() <= .04 * self.height:
                out |= set(chain)
        return out

    def limbs(self):
        return [self.rig.names[j] for ids in self.parts().values() for j in ids if j < self.rig.nb]

    def depth_axis(self, clip):
        views = self.views(clip)
        if len(views) > 1:
            return None
        cam = views[0]['cam']
        d = cam.fwd @ cam.Ma
        return d / np.linalg.norm(d)

    # ----- clip ------------------------------------------------------------------------------------------------
    def _load(self, clip):
        if clip in self._clip:
            return self._clip[clip]
        work = self.project.work(clip)
        frames = clipmod.read_frames(self.project.clip_file(clip))
        H, W = frames[0].shape[:2]
        cams_json = json.loads((self.project.start_dir / 'cameras.json').read_text(encoding='utf-8'))
        pivot = self.rig.skin(self.p0(clip), self.mesh['coords'], self.idx, self.wts).mean(0)
        cams = [clipmod.Camera(rec, (W, H), cams_json['frame_px'], self.data['armature_world'], pivot)
                for rec in cams_json['views']]
        mpath = work / 'masks.npz'
        cpath = work / 'cameras-aligned.json'
        X0 = self.rig.skin(self.p0(clip), self.mesh['coords'], self.idx, self.wts)
        views_cfg = self.cfg.get('views', {})

        def align(k, cam, mask0, start=None):
            vc = views_cfg.get(str(k), views_cfg.get(cams_json['views'][k]['name'], {}))
            return clipmod.align(cam, X0, self.mesh['tris'], mask0, rasterize, turn=vc.get('turn', True), start=start)
        al = json.loads(cpath.read_text()) if cpath.exists() and mpath.exists() else None
        if mpath.exists():
            masks = np.load(mpath)['masks']
        else:
            # frame 0 outline, the cameras aligned on it, the clip's shadows learned under the known start pose,
            # then every frame without them and the cameras aligned again on the cleaner outline
            masks = np.zeros((len(frames), H, W), bool)
            al = []
            for k, cam in enumerate(cams):
                pn = cam.panel
                panel = [np.ascontiguousarray(pn.cut(f)) for f in frames]
                bg = clipmod.background(panel)
                first = clipmod.silhouette(panel[0], bg)
                rec_ = align(k, cam, first)
                shade = clipmod.learn_shadow(panel[0], bg, first, clipmod.model_mask(cam, X0, self.mesh['tris'],
                                                                                    rasterize))
                for i, f in enumerate(panel):
                    pn.put(masks[i], clipmod.silhouette(f, bg, shade))
                if shade is not None:
                    before = rec_['frame0_iou']
                    rec_ = align(k, cam, pn.cut(masks[0]), start=rec_)
                    rec_['shadow'] = dict(learned=True, frame0_px_before=int(first.sum()),
                                          frame0_px_after=int(pn.cut(masks[0]).sum()), frame0_iou_before=before)
                else:
                    rec_['shadow'] = dict(learned=False)
                al.append(rec_)
            np.savez_compressed(mpath, masks=masks)
            write_json(cpath, al)
        tpath = work / 'timing.json'
        if 'fps' in self.cfg:
            fps = self.cfg['fps']
            i = np.arange(len(frames))
            times = {'24': i / 24, '30-5': (i + i // 4) / 30, '25-25': (i + i // 24) / 25}[str(fps)]
            rec = {'content': f'set in creature.json: {fps}'}
        elif tpath.exists():
            t = json.loads(tpath.read_text())
            times, rec = np.array(t['times']), t
        else:
            times, rec = clipmod.cadence(frames, masks)
            write_json(tpath, {**rec, 'times': [float(v) for v in times]})
        if al is None:                              # masks kept from an earlier run, cameras not
            al = [align(k, cam, cam.panel.cut(masks[0])) for k, cam in enumerate(cams)]
            write_json(cpath, al)
        # a clip Seedance drew in slow motion (generate --slow-motion): its times x time_scale are the real ones, so
        # the fit's speeds and filters are in real time and the motion plays at normal speed
        times = np.asarray(times, float) * float(self.cfg.get('time_scale', 1.0))
        for cam, a in zip(cams, al):
            cam.set(a['scale'], a['offset'], np.radians(a['turn_deg']))
        views = []
        for k, cam in enumerate(cams):
            pn = cam.panel

            def frame(i, pn=pn):
                return pn.cut(frames[i]).copy()

            def mask(i, pn=pn):
                m = pn.cut(masks[i])
                return m if pn.turn == 0 else np.ascontiguousarray(m)
            views.append(dict(name=cams_json['views'][k].get('name', f'view{k}'), size=cam.size, frame=frame,
                              mask=mask, project=cam.project, depth=cam.depth, cam=cam, panel=pn))
        self._clip[clip] = dict(views=views, times=np.asarray(times, float), timing=rec, aligned=al)
        return self._clip[clip]

    def views(self, clip):
        return self._load(clip)['views']

    def times(self, clip):
        return self._load(clip)['times']

    def rasterize(self, pix, depth, tris, size):
        return rasterize(pix, depth, tris, size)

    # ----- output ----------------------------------------------------------------------------------------------
    def render(self, clip, X, view=None):
        cam = (view or self.views(clip)[0])['cam'] if isinstance(view, dict) or view is None else view
        return render(X, cam, self.mesh, self.textures, tri_colour=self.painted(clip))

    def painted(self, clip):
        """(triangles, 3) BGR of the limbs painted in the clip's start frame (markers.json), NaN elsewhere; None for
        a plain clip. Without it the seeds on painted limbs were thrown away: the video's flat colour did not look
        like the fur of our own render there (a painted wolf lost 43% of its seeds, nearly all on the legs)."""
        if clip is None:
            return None
        key = ('painted', clip)
        if key not in self._clip:
            path = self.project.markers(clip)
            out = None
            if path is not None and path.exists():
                cols = json.loads(path.read_text(encoding='utf-8'))['colours']
                names = [b['name'] for b in self.data['bones']]
                bone = self.idx[np.arange(len(self.idx)), np.argmax(self.wts, axis=1)]
                tris = self.mesh['tris']
                vb = bone[tris]
                top = np.where(vb[:, 1] == vb[:, 2], vb[:, 1], vb[:, 0])      # the corners' most common bone
                out = np.full((len(tris), 3), np.nan, np.float32)
                for n, rgb in cols.items():
                    if n in names:
                        r, g, b = [min(255.0, 255.0 * max(c, 0.0) ** (1 / 2.2)) for c in rgb]
                        out[top == names.index(n)] = (b, g, r)
            self._clip[key] = out
        return self._clip[key]

    def save(self, clip, params, tag, meta):
        path = self.project.work(clip) / f'fit{tag}.npz'
        np.savez_compressed(path, params=params, times=self.times(clip), meta=str(meta))
        return path

    def load_params(self, path):
        return np.load(path)['params']

    def export_motion(self, clip, params, path):
        """The motion for blender/import_motion.py: per frame every bone's basis rotation (quaternion), the root's
        location, every prop's matrix (armature space), at the clip's content times."""
        quats, locs, props = [], [], []
        for p in params:
            q, loc, pr = self.rig.pose_basis(p)
            quats.append(q); locs.append(loc); props.append(pr)
        np.savez_compressed(path, bones=np.array(self.rig.names), root=self.rig.names[self.rig.root],
                            quaternions=np.array(quats), root_location=np.array(locs),
                            props=np.array([pr['name'] for pr in self.rig.props]),
                            prop_matrices=np.array(props).reshape(len(params), len(self.rig.props), 4, 4),
                            times=self.times(clip), armature_world=np.array(self.data['armature_world']))
        return path
