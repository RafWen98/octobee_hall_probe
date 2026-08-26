#!/usr/bin/env python3
"""
octobee/machine.py -- the coil set the probe is measuring inside.

A field map is a table of vectors at rig millimetres. On its own it says
nothing about the machine: 40 mT at (120, 45, 80) is only meaningful once you
know where that point sits relative to the coils, and which coils were carrying
current at the time. This module supplies both halves, and nothing else -- it
draws no field and predicts no field. What it knows is geometry:

  * the coil centrelines, read out of a simsopt configuration file;
  * a circular cross-section swept along each one, which is the volume the
    probe cannot enter;
  * where the probe body sits in that machine, given a mount pose and the
    live stage reading;
  * how close the two are, in millimetres of clearance.

Reading simsopt files without simsopt
-------------------------------------
The configuration is a SIMSON JSON graph -- simsopt's own serialisation, a flat
`simsopt_objs` table with `{"$type": "ref"}` links between entries. Installing
simsopt to read it would drag in a compiled stack (and its own numpy pin) onto
a machine whose whole job is talking to two carriers and three stages, so this
reads the graph directly. Only the classes a coil set is made of are
understood: Curve{XYZ,RZ}Fourier, RotatedCurve, Coil, Current, ScaledCurrent,
CurrentSum and BiotSavart. Anything else in the file -- surfaces, Boozer
solves, the optimiser's bookkeeping -- is ignored rather than refused, because
a file that also contains a plasma boundary is still a perfectly good source of
coils.

The stored `quadpoints` are the optimiser's sampling, not a resolution limit:
the Fourier coefficients are exact, so the curves here are re-evaluated at
whatever resolution the caller asks for.

One coil, three configurations
------------------------------
A simsopt file usually carries the same physical coils several times over, once
per BiotSavart object, because that is how a multi-configuration optimisation
records its current sets. designA_after_scaled.json is exactly that: eighteen
Coil objects, six distinct curves, three current configurations. Treating those
as eighteen coils would draw each one three times and offer eighteen switches
for six physical windings, so coils are identified by their GEOMETRY -- the base
curve plus the chain of rotations applied to it -- and the currents hang off
that identity, one per configuration.

Frames and units
----------------
The file is in metres; everything here is in millimetres, because the probe,
the stages and every existing config file in this project are, and one
conversion at the boundary is easier to keep right than a mixture.

    machine frame   the coil file's own coordinates, in mm. Z is the axis of
                    the torus; the coils sit around it.
    mount frame     the probe's world frame from probe_geometry (tube axis
                    along +Y, tip forward, +Z up) -- the rig's own axes, which
                    is also what the stages move along.

A Placement is the rigid transform between them: where the probe's mounting
flange sits in the machine, and how it is turned. The stage reading is added in
the MOUNT frame before that rotation, because the stages move the probe along
the rig's axes, not the machine's.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

from octobee import paths
from octobee.calib import geometry as pgeom

CONFIG_NAME = "machine.json"

# The winding pack has to be given a thickness before anything can be said
# about clearance, and no simsopt file records one: the optimiser works with
# infinitely thin filaments. 60 mm is the winding pack this rig is being built
# around -- still a number to check against the real conductor rather than one
# read out of any file, but close enough that the clearance figure starts out
# in the right place instead of a hundred millimetres optimistic.
DEFAULT_COIL_RADIUS_MM = 60.0

# Points per coil when a centreline is evaluated. 256 puts the chord error of a
# 3 m coil at well under a tenth of a millimetre, which is far below anything
# the clearance number is trusted to.
CURVE_POINTS = 256

# LTS300C travel. Only a fallback: once the stages are open, the GUI replaces
# this with what they actually report.
DEFAULT_TRAVEL_MM = 300.0

M_TO_MM = 1000.0

_CLASSES_UNDERSTOOD = (
    "CurveXYZFourier", "CurveRZFourier", "RotatedCurve",
    "Coil", "Current", "ScaledCurrent", "CurrentSum", "BiotSavart")


# ==========================================================================
# reading the simsopt graph
# ==========================================================================

def _is_ref(obj):
    return isinstance(obj, dict) and obj.get("$type") == "ref"


def _ref(obj):
    """The name a `{"$type": "ref"}` link points at."""
    return obj["value"] if _is_ref(obj) else obj


def _array(obj):
    """A numpy array out of simsopt's serialised-array dict (or a plain list)."""
    if isinstance(obj, dict) and obj.get("@class") == "array":
        return np.asarray(obj["data"], dtype=float)
    return np.asarray(obj, dtype=float)


def _rotation(phi, flip):
    """simsopt's RotatedCurve transform, as a matrix that multiplies on the right.

    Points are rows here, as they are in simsopt, so a curve is transformed by
    `gamma @ rotation(...)`. The flip is the stellarator-symmetry reflection
    (y, z -> -y, -z) that turns a curve into its mirror in the neighbouring
    half period; applying it after the rotation is simsopt's order.
    """
    c, s = math.cos(phi), math.sin(phi)
    rot = np.array([[c, -s, 0.0],
                    [s, c, 0.0],
                    [0.0, 0.0, 1.0]]).T
    if flip:
        rot = rot @ np.array([[1.0, 0.0, 0.0],
                              [0.0, -1.0, 0.0],
                              [0.0, 0.0, -1.0]])
    return rot


class _Graph:
    """The `simsopt_objs` table, indexed the two ways references arrive."""

    def __init__(self, doc):
        objs = doc.get("simsopt_objs")
        if not isinstance(objs, dict):
            raise ValueError("not a simsopt SIMSON file: no 'simsopt_objs' "
                             "object at the top level")
        self.objs = {}
        for key, value in objs.items():
            if not isinstance(value, dict):
                continue
            self.objs[key] = value
            name = value.get("@name")
            if name:
                self.objs[name] = value

    def get(self, name):
        obj = self.objs.get(_ref(name))
        if obj is None:
            raise ValueError(f"the file refers to {_ref(name)!r}, which it "
                             f"does not contain")
        return obj

    def of_class(self, cls):
        """Every object of a class, de-duplicated and in file order."""
        out, seen = [], set()
        for obj in self.objs.values():
            if obj.get("@class") == cls and id(obj) not in seen:
                seen.add(id(obj))
                out.append(obj)
        return out

    # ---- dofs ----
    def dofs(self, obj):
        """{name: value} for an object's degrees of freedom.

        Reading them by NAME rather than by position is the point: simsopt's
        ordering of a curve's coefficients is an implementation detail, and a
        file written by another version that ordered them differently would
        otherwise load without complaint and draw a coil that is wrong in a way
        nobody would spot.
        """
        dof = self.get(obj["dofs"])
        names = dof["names"]
        if isinstance(names, dict):
            names = names.get("data", [])
        values = _array(dof["x"])
        if len(names) != len(values):
            raise ValueError(f"{obj.get('@name')}: {len(names)} dof names "
                             f"against {len(values)} values")
        return dict(zip(names, values))

    # ---- curves ----
    def curve_points(self, name, n=CURVE_POINTS):
        """The closed centreline of a curve, metres, (n+1, 3) with the ends met."""
        obj = self.get(name)
        cls = obj.get("@class")
        if cls == "RotatedCurve":
            pts = self.curve_points(obj["curve"], n)
            return pts @ _rotation(float(obj["phi"]), bool(obj["flip"]))
        if cls == "CurveXYZFourier":
            return self._xyz_fourier(obj, n)
        if cls == "CurveRZFourier":
            return self._rz_fourier(obj, n)
        raise ValueError(f"{obj.get('@name', name)}: curves of type {cls!r} "
                         f"are not understood (known: CurveXYZFourier, "
                         f"CurveRZFourier, RotatedCurve)")

    def _closed(self, pts):
        return np.vstack([pts, pts[:1]])

    def _xyz_fourier(self, obj, n):
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        pts = np.zeros((n, 3))
        for name, value in self.dofs(obj).items():
            comp = "xyz".find(name[0])
            trig = name[1]
            order = int(name[name.index("(") + 1:name.index(")")])
            if comp < 0 or trig not in "cs":
                raise ValueError(f"{obj.get('@name')}: unexpected dof {name!r}")
            angle = 2.0 * math.pi * order * t
            pts[:, comp] += value * (np.cos(angle) if trig == "c"
                                     else np.sin(angle))
        return self._closed(pts)

    def _rz_fourier(self, obj, n):
        nfp = int(obj.get("nfp", 1))
        phi = 2.0 * math.pi * np.linspace(0.0, 1.0, n, endpoint=False)
        r = np.zeros(n)
        z = np.zeros(n)
        for name, value in self.dofs(obj).items():
            kind, trig = name[0], name[1]
            order = int(name[name.index("(") + 1:name.index(")")])
            angle = order * nfp * phi
            wave = np.cos(angle) if trig == "c" else np.sin(angle)
            if kind == "r":
                r += value * wave
            elif kind == "z":
                z += value * wave
            else:
                raise ValueError(f"{obj.get('@name')}: unexpected dof {name!r}")
        return self._closed(np.column_stack([r * np.cos(phi),
                                             r * np.sin(phi), z]))

    def curve_key(self, name):
        """An identity for a curve: the base it came from and how it was moved.

        Two RotatedCurve objects wrapping the same base with the same angle and
        flip are the same physical winding, however many times the file names
        them. Angles are rounded to a micro-radian so that a value written out
        and read back cannot split one coil into two.
        """
        obj = self.get(name)
        chain = []
        while obj.get("@class") == "RotatedCurve":
            chain.append((round(float(obj["phi"]), 6), bool(obj["flip"])))
            obj = self.get(obj["curve"])
        return (obj.get("@name", "?"), tuple(reversed(chain)))

    # ---- currents ----
    def current(self, name):
        """A current object's value, in amperes (amp-turns, as simsopt uses it)."""
        obj = self.get(name)
        cls = obj.get("@class")
        if cls == "ScaledCurrent":
            return float(obj["scale"]) * self.current(obj["current_to_scale"])
        if cls == "CurrentSum":
            return (self.current(obj["current_a"])
                    + self.current(obj["current_b"]))
        if cls == "Current":
            # The dof is the live value; the "current" field is a copy of it
            # written at save time. They agree in every file seen so far, but
            # if they ever do not, the dof is the one the optimiser was using.
            dofs = self.dofs(obj)
            if dofs:
                return float(next(iter(dofs.values())))
            return float(obj["current"])
        raise ValueError(f"{obj.get('@name', name)}: currents of type {cls!r} "
                         f"are not understood")


# ==========================================================================
# coils
# ==========================================================================

class Coil:
    """One physical winding: a centreline, and what it carries per configuration."""

    def __init__(self, label, key, points_mm, description=""):
        self.label = label
        self.key = key
        self.points_mm = np.asarray(points_mm, dtype=float)
        self.description = description
        self.currents = {}              # configuration name -> amp-turns

    # ---- shape ----
    @property
    def centroid_mm(self):
        # The closing point is a repeat of the first, so it must not be
        # counted twice in an average.
        return self.points_mm[:-1].mean(axis=0)

    @property
    def length_mm(self):
        return float(np.linalg.norm(np.diff(self.points_mm, axis=0),
                                    axis=1).sum())

    @property
    def bounds_mm(self):
        return self.points_mm.min(axis=0), self.points_mm.max(axis=0)

    @property
    def label_anchor_mm(self):
        """Somewhere ON the coil to hang its name.

        Not the centroid: a coil is a loop, so its centroid sits in the hole in
        the middle of it, and at machine scale a label floating there reads as
        belonging to whatever happens to be behind it. The outermost point of
        the winding is unambiguous and is on the side facing the room.
        """
        pts = self.points_mm[:-1]
        out = pts[int(np.argmax(np.hypot(pts[:, 0], pts[:, 1])))]
        return out * np.array([1.06, 1.06, 1.0])

    def where(self):
        """Human-readable position: the toroidal angle and radius of its centre."""
        c = self.centroid_mm
        phi = math.degrees(math.atan2(c[1], c[0])) % 360.0
        return f"phi {phi:5.1f} deg, R {math.hypot(c[0], c[1]) / 1000:.3f} m"

    def current(self, configuration, scale=1.0):
        return self.currents.get(configuration, 0.0) * scale


class CoilSet:
    """The coils of one machine, plus the current configurations on offer."""

    def __init__(self, coils, configurations, path="", note=""):
        self.coils = list(coils)
        self.configurations = list(configurations)
        self.path = path
        self.note = note

    # ---- loading ----
    @classmethod
    def load(cls, path, n_points=CURVE_POINTS):
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict):
            raise ValueError(f"{path}: expected a JSON object, got "
                             f"{type(doc).__name__}")
        graph = _Graph(doc)

        # Group the Coil objects by the field they belong to. A file with no
        # BiotSavart at all still has coils worth drawing, so they become a
        # single unnamed configuration rather than nothing.
        groups = []
        for field in graph.of_class("BiotSavart"):
            names = [_ref(c) for c in field.get("coils", [])]
            if names:
                groups.append((field.get("@name", "field"), names))
        if not groups:
            names = [c.get("@name") for c in graph.of_class("Coil")]
            if not names:
                raise ValueError(f"{path}: contains no coils")
            groups = [("as loaded", names)]

        coils, by_key = [], {}
        configurations = []
        for config_name, coil_names in groups:
            configurations.append(config_name)
            for coil_name in coil_names:
                coil_obj = graph.get(coil_name)
                curve = _ref(coil_obj["curve"])
                key = graph.curve_key(curve)
                coil = by_key.get(key)
                if coil is None:
                    label = f"C{len(coils) + 1}"
                    base, chain = key
                    moves = ", ".join(
                        f"rotated {math.degrees(phi):g} deg"
                        + (" and flipped" if flip else "")
                        for phi, flip in chain) or "as drawn"
                    coil = Coil(label, key,
                                graph.curve_points(curve, n_points) * M_TO_MM,
                                description=f"{base}, {moves}")
                    by_key[key] = coil
                    coils.append(coil)
                coil.currents[config_name] = graph.current(coil_obj["current"])

        version = doc.get("@version", "?")
        note = (f"{len(coils)} coils in {len(configurations)} current "
                f"configuration(s), from a simsopt {version} file")
        return cls(coils, configurations, path=path, note=note)

    @classmethod
    def load_or_none(cls, path, on_error=None, n_points=CURVE_POINTS):
        """Load, or report why not and return None.

        A missing or unreadable coil file must not stop the window opening:
        the probe and the stages work perfectly well without one, and losing
        the whole application because a path went stale would be absurd.
        """
        if not path:
            return None
        if not os.path.exists(path):
            if on_error is not None:
                on_error(f"{path}: no such coil file -- the machine view has "
                         f"nothing to draw until one is loaded")
            return None
        try:
            return cls.load(path, n_points=n_points)
        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            if on_error is not None:
                on_error(f"{path} could not be read as a simsopt coil set "
                         f"({type(exc).__name__}: {exc})")
            return None

    # ---- lookup ----
    def __len__(self):
        return len(self.coils)

    def __iter__(self):
        return iter(self.coils)

    def __getitem__(self, label):
        for coil in self.coils:
            if coil.label == label:
                return coil
        raise KeyError(label)

    @property
    def labels(self):
        return [c.label for c in self.coils]

    def bounds_mm(self):
        lo = np.min([c.points_mm.min(axis=0) for c in self.coils], axis=0)
        hi = np.max([c.points_mm.max(axis=0) for c in self.coils], axis=0)
        return lo, hi

    def extent_mm(self):
        lo, hi = self.bounds_mm()
        return float(np.max(hi - lo))

    def summary(self, configuration=None, scale=1.0, energised=None):
        lo, hi = self.bounds_mm()
        lines = [f"{os.path.basename(self.path) or 'coil set'}: {self.note}",
                 f"  extent  x {lo[0] / 1000:+.3f}..{hi[0] / 1000:+.3f} m,"
                 f"  y {lo[1] / 1000:+.3f}..{hi[1] / 1000:+.3f} m,"
                 f"  z {lo[2] / 1000:+.3f}..{hi[2] / 1000:+.3f} m"]
        if configuration is None:
            configuration = self.configurations[0] if self.configurations else None
        for coil in self.coils:
            on = energised is None or coil.label in energised
            amps = coil.current(configuration, scale)
            lines.append(
                f"  {coil.label:<4} {coil.where():<28} "
                f"{coil.length_mm / 1000:5.3f} m  "
                f"{amps / 1000:+9.3f} kA-turns  "
                f"{'ON ' if on else 'off'}  {coil.description}")
        return "\n".join(lines)


# ==========================================================================
# meshes: the volume a coil occupies
# ==========================================================================

def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0.0, 1.0, n)


def tube_mesh(points, radius, sides=12):
    """A circular cross-section swept along a closed centreline.

    Returns (vertices, faces). The frame is carried along the curve by parallel
    transport -- rotating the previous ring onto the next tangent rather than
    rebuilding it from a fixed up-vector, which is what stops the tube flipping
    inside out where the curve turns over the top of the torus. The residual
    twist a closed loop accumulates is then spread evenly over the whole tube,
    so the seam closes without a kink.
    """
    P = np.asarray(points, dtype=float)
    if len(P) > 1 and np.allclose(P[0], P[-1]):
        P = P[:-1]                      # the loop closes through the faces
    n = len(P)
    if n < 3:
        raise ValueError("a coil needs at least three points to be a tube")

    tangents = _unit(np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0))
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(seed @ tangents[0])) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    normals = np.zeros_like(tangents)
    normals[0] = _unit(seed - tangents[0] * (seed @ tangents[0]))
    for i in range(1, n):
        prev = normals[i - 1]
        t = tangents[i]
        normals[i] = _unit(prev - t * (prev @ t))

    # How far the transported frame has slipped by the time it comes back
    # round, undone linearly along the loop.
    binormal0 = np.cross(tangents[0], normals[0])
    closing = _unit(normals[-1] - tangents[0] * (normals[-1] @ tangents[0]))
    drift = math.atan2(float(closing @ binormal0), float(closing @ normals[0]))
    unwind = drift * np.arange(n) / n

    binormals = np.cross(tangents, normals)
    theta = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    ang = theta[None, :] - unwind[:, None]              # (n, sides)
    verts = (P[:, None, :]
             + radius * (np.cos(ang)[:, :, None] * normals[:, None, :]
                         + np.sin(ang)[:, :, None] * binormals[:, None, :]))

    idx = np.arange(n)
    nxt = (idx + 1) % n
    faces = []
    for j in range(sides):
        k = (j + 1) % sides
        a = idx * sides + j
        b = idx * sides + k
        c = nxt * sides + k
        d = nxt * sides + j
        faces.append(np.column_stack([a, b, c]))
        faces.append(np.column_stack([a, c, d]))
    return (verts.reshape(-1, 3).astype(np.float32),
            np.vstack(faces).astype(np.int32))


# ==========================================================================
# clearance: how close the probe is to a winding
# ==========================================================================

class Clearance:
    """The closest approach between the probe body and any coil."""

    def __init__(self, gap_mm=float("inf"), coil=None,
                 probe_point=None, coil_point=None):
        self.gap_mm = float(gap_mm)
        self.coil = coil
        self.probe_point = probe_point
        self.coil_point = coil_point

    @property
    def collides(self):
        return self.gap_mm < 0.0

    def __repr__(self):
        if self.coil is None:
            return "Clearance(no coils)"
        return f"Clearance({self.gap_mm:.1f} mm to {self.coil})"

    def text(self):
        if self.coil is None:
            return "no coils loaded"
        if self.collides:
            return (f"INSIDE {self.coil} by {-self.gap_mm:.0f} mm -- the probe "
                    f"cannot be where it is drawn")
        return f"{self.gap_mm:.0f} mm clear of {self.coil}"


def _segment_distances(points, verts):
    """Squared distance and segment parameter, every point against every segment.

    Written entirely in (P, S) matrix operations rather than the obvious (P, S,
    3) ones. The obvious form builds four arrays of three-vectors -- for a
    two-thousand-point probe against a six-coil machine that is a couple of
    hundred megabytes of traffic per check, and it measured 86 ms, which is
    half of the interval the stage display runs at. Expanding
    |p - a|^2 and (p - a).(b - a) instead turns the whole thing into two matrix
    products, which BLAS does in a few milliseconds. The identities:

        |p - a|^2       = |p|^2 - 2 p.a + |a|^2
        (p - a).(b - a) = p.(b - a) - a.(b - a)

    so no array of differences ever has to exist.
    """
    a = verts[:-1]
    ab = verts[1:] - a
    abab = np.einsum("ij,ij->i", ab, ab)
    abab = np.where(abab == 0.0, 1.0, abab)
    aab = np.einsum("ij,ij->i", a, ab)

    pa = points @ a.T                                   # (P, S)
    pab = points @ ab.T                                 # (P, S)
    d2 = ((points * points).sum(axis=1)[:, None]
          - 2.0 * pa + (a * a).sum(axis=1)[None, :])
    proj = pab - aab[None, :]
    t = np.clip(proj / abab[None, :], 0.0, 1.0)
    d2 += t * (t * abab[None, :] - 2.0 * proj)
    return d2, t


def _nearest_on_polyline(points, verts):
    """Closest approach from a set of points to a polyline -> (distance, p, s, t)."""
    d2, t = _segment_distances(points, verts)
    flat = int(np.argmin(d2))
    p, s = divmod(flat, d2.shape[1])
    # Squared distance is exact enough to compare but can go a hair negative
    # through cancellation when a point sits on the wire.
    return (math.sqrt(max(float(d2[p, s]), 0.0)), p, s, float(t[p, s]))


# How many segments a coil is cut down to for the first pass. 32 chords round a
# 3 m coil deviate from it by a couple of centimetres, which is a tight enough
# bracket to throw away nineteen probe points in twenty.
COARSE_SEGMENTS = 32


def _coarse_outline(coil):
    """A decimated centreline, and how far it can be from the real one.

    The bound is a Hausdorff distance measured both ways -- every fine vertex
    to the coarse chords, and every coarse chord to the fine curve. That makes
    it a bound in the sense actually needed here: for ANY point in space, the
    distance to the coarse outline and the distance to the real coil differ by
    no more than this. Anything looser and the narrowing below could discard
    the point that was about to touch a winding.

    Cached on the coil: it depends only on geometry, and the geometry does not
    change once a file is loaded.
    """
    cached = getattr(coil, "_coarse", None)
    if cached is not None:
        return cached
    fine = coil.points_mm
    step = max(1, (len(fine) - 1) // COARSE_SEGMENTS)
    coarse = np.vstack([fine[::step], fine[-1:]])
    if len(coarse) < 3:
        coarse = fine
    d2, _t = _segment_distances(fine, coarse)
    away = math.sqrt(max(float(d2.min(axis=1).max()), 0.0))
    # Sample along the coarse chords too: their midpoints are the parts that
    # stand furthest off the real curve.
    mid = 0.5 * (coarse[:-1] + coarse[1:])
    d2, _t = _segment_distances(mid, fine)
    back = math.sqrt(max(float(d2.min(axis=1).max()), 0.0))
    cached = (coarse, max(away, back))
    coil._coarse = cached
    return cached


_COARSE_BY_CURVE = {}


def _coarse_outline_of(points_mm):
    """_coarse_outline for a bare centreline, cached by array identity."""
    key = id(points_mm)
    cached = _COARSE_BY_CURVE.get(key)
    if cached is not None:
        return cached
    shim = type("_Curve", (), {"points_mm": points_mm, "_coarse": None})()
    cached = _coarse_outline(shim)
    # Bounded: a session loads a handful of coil sets, not thousands.
    if len(_COARSE_BY_CURVE) > 256:
        _COARSE_BY_CURVE.clear()
    _COARSE_BY_CURVE[key] = cached
    return cached


def clearance(points_mm, coilset, radius_mm, labels=None):
    """Closest approach from a cloud of probe points to the coil surfaces.

    `labels` restricts the check to some coils; the default is all of them,
    because a coil that is switched off is still solid copper and still in the
    way.

    Coils are ordered by a cheap bound first -- the distance from the probe's
    bounding sphere to each centreline -- and any coil whose bound is already
    worse than the best exact answer so far is skipped. With the probe near one
    coil out of six that usually leaves one full check to do.
    """
    best = Clearance()
    if coilset is None or not len(coilset):
        return best
    pts = np.asarray(points_mm, dtype=float).reshape(-1, 3)
    if not len(pts):
        return best

    centre = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    reach = float(np.linalg.norm(pts - centre, axis=1).max())

    candidates = []
    for coil in coilset:
        if labels is not None and coil.label not in labels:
            continue
        far, _p, _s, _t = _nearest_on_polyline(centre[None, :], coil.points_mm)
        candidates.append((far - reach - radius_mm, coil))
    candidates.sort(key=lambda item: item[0])

    for bound, coil in candidates:
        if bound > best.gap_mm:
            continue
        # Two passes. The first is against a decimated coil, which costs an
        # eighth as much and says which handful of probe points could possibly
        # be the closest one; the second is exact, on those alone.
        coarse, slack = _coarse_outline(coil)
        d2, _t = _segment_distances(pts, coarse)
        near = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
        keep = np.flatnonzero(near <= near.min() + 2.0 * slack)
        subset = pts[keep]
        dist, p, s, t = _nearest_on_polyline(subset, coil.points_mm)
        gap_mm = dist - radius_mm
        if gap_mm < best.gap_mm:
            a = coil.points_mm[s]
            on_coil = a + t * (coil.points_mm[s + 1] - a)
            best = Clearance(gap_mm, coil.label, subset[p], on_coil)
    return best


def _mesh_edge_points(verts, faces, spacing_mm):
    """Every mesh edge, sampled along its length.

    Corners alone are not enough to test a body against a wire: the probe's
    tube is 159 mm long and a coil can pass close to the middle of an edge
    while staying far from both of its ends. Sampling the edges costs a few
    hundred points and removes that blind spot; the faces themselves are thin
    enough (a 1.2 mm board, a 25 mm tube) that their interiors add nothing.
    """
    verts = np.asarray(verts, dtype=float)
    edges = set()
    for tri in np.asarray(faces, dtype=int):
        for i in range(3):
            u, v = int(tri[i]), int(tri[(i + 1) % 3])
            edges.add((u, v) if u < v else (v, u))
    out = [verts]
    for u, v in edges:
        a, b = verts[u], verts[v]
        n = int(np.linalg.norm(b - a) // max(spacing_mm, 0.5))
        if n > 1:
            t = np.linspace(0.0, 1.0, n + 1)[1:-1, None]
            out.append(a + t * (b - a))
    return np.vstack(out)


def probe_cloud(geom, spacing_mm=8.0):
    """The probe body as points, in the mount frame, mm.

    Tube, boards and chip packages -- the whole thing that can hit something,
    not just the sensitive volumes. Built once per geometry and then only
    transformed, because the shape does not change when the stage moves.
    """
    clouds = []
    verts, faces = pgeom.tube_mesh(geom)
    clouds.append(_mesh_edge_points(verts, faces, spacing_mm))
    for sid in range(1, pgeom.N_SENSORS + 1):
        for verts, faces in (pgeom.arm_mesh(geom, sid),
                             pgeom.chip_mesh(geom, sid)):
            clouds.append(_mesh_edge_points(verts, faces, spacing_mm))
    return pgeom.to_world(np.vstack(clouds))


# How many grid nodes are put against the coils in one go. Each costs an
# (N, 32) matrix per coil, so the whole grid at once is hundreds of megabytes
# for no gain; sixty-four thousand keeps the working set inside cache-sized
# BLAS calls and still amortises the per-call overhead away.
REACH_CHUNK = 65536

# Refuse a dilation whose structuring element is bigger than this. The probe
# voxelises to a few thousand cells at any sensible scan step; a hundred
# thousand means the step was set to something like a tenth of a millimetre,
# where this whole approach is the wrong tool and a straight answer beats an
# hour of grinding.
MAX_BODY_VOXELS = 100000

# And refuse the dilation itself past this much work -- one OR of the whole
# grid per probe cell, so cells x nodes. A 300 mm cube is 12 M at a 10 mm step
# and 320 M at 5 mm, which are a tenth of a second and five seconds; 1 mm would
# be seven thousand times the 5 mm figure, and the useful thing to do about
# that is say so immediately rather than appear to be working.
MAX_DILATION_OPS = 2_000_000_000


class VolumeTooFine(ValueError):
    """The requested step makes the reachable-volume calculation absurd."""


def reachable_grid(origin_rig_mm, step_mm, shape, placement, cloud_mount_mm,
                   coilset, radius_mm, margin_mm=0.0, labels=None,
                   progress=None):
    """Which nodes of a regular rig-frame grid the probe body can occupy.

    This is the question a scan plan actually asks, and it is not the question
    `clearance` answers. `clearance` says how close the probe is to a winding
    at ONE place, exactly, in a few milliseconds. A 300 mm cube on a 10 mm grid
    has thirty thousand places in it, and asking exactly costs a minute -- long
    enough that nobody moves a slider and watches the volume change, which is
    the whole point of drawing it.

    So it is done the other way round. The probe body is rigid and, on this
    rig, cannot tilt: the stage only translates it. The set of flange positions
    that put some part of the probe inside a winding is therefore exactly the
    forbidden region DILATED by the probe's own shape -- a Minkowski sum, which
    on a regular grid is a handful of shifted ORs over a boolean array. One
    pass marks the nodes that are too close to a coil; the dilation turns that
    into "somewhere the flange cannot be" in a fraction of a second.

    Everything is in the RIG frame, which is the frame the stages move in and
    the frame a scan is specified in: node (i, j, k) is
    `origin_rig_mm + step_mm * (i, j, k)`, a stage reading, carried into the
    machine by `placement.flange_path_mm` -- stage zero included, so a node
    here and a clearance printed for the same stage reading agree.
    `cloud_mount_mm` is probe_cloud() output, already in that frame.

    CONSERVATIVE, AND BY A STATED AMOUNT. Two approximations are made and both
    are paid for up front rather than hidden:

      * the probe is snapped to the grid, which moves any part of it by at
        most half a diagonal, `step * sqrt(3) / 2`;
      * coils are measured against their decimated outlines FIRST, and every
        node the decimation cannot settle is then measured against the real
        curve -- so the decimation costs time, not standoff.

    Only the snap is therefore paid for, and it is added to the exclusion
    radius: a node is called reachable only if it is reachable for every body
    position consistent with it. The answer can refuse a position that is in
    fact fine -- never the reverse. At a 10 mm step that is 8.7 mm of extra
    standoff, which `grid_standoff_mm` reports so it can be read as part of the
    margin rather than mistaken for precision.

    Returns a boolean array of `shape`: True where the probe fits.
    """
    shape = tuple(int(v) for v in shape)
    step = float(step_mm)
    origin = np.asarray(origin_rig_mm, dtype=float).reshape(3)
    if step <= 0.0:
        raise VolumeTooFine("the grid step must be positive")
    if coilset is None or not len(coilset):
        return np.ones(shape, dtype=bool)
    curves = [c.points_mm for c in coilset
              if labels is None or c.label in labels]
    if not curves:
        return np.ones(shape, dtype=bool)

    # ---- the probe, as whole grid cells ----
    body = np.asarray(cloud_mount_mm, dtype=float).reshape(-1, 3)
    cells = np.unique(np.rint(body / step).astype(np.int64), axis=0)
    if len(cells) > MAX_BODY_VOXELS:
        raise VolumeTooFine(
            f"a {step:g} mm step chops the probe into {len(cells):,} cells; "
            f"the reachable volume is computed by dilating the coils with the "
            f"probe, and that is more shifts than it is worth. Use a coarser "
            f"step for the volume, or turn coil avoidance off.")
    snap_mm = step * math.sqrt(3.0) / 2.0
    lo_c, hi_c = cells.min(axis=0), cells.max(axis=0)
    work = len(cells) * int(np.prod(shape, dtype=np.int64))
    if work > MAX_DILATION_OPS:
        raise VolumeTooFine(
            f"a {step:g} mm step over this box needs {work / 1e9:.0f} billion "
            f"operations to work out where the probe fits. Use a coarser step "
            f"for the volume -- it only sets how far apart the swept LINES "
            f"are, not how finely each line is sampled -- or turn coil "
            f"avoidance off.")

    # ---- the region the probe can reach into, one node per grid cell ----
    big = tuple(int(shape[a] + hi_c[a] - lo_c[a]) for a in range(3))
    axes = [origin[a] + step * (np.arange(big[a]) + lo_c[a]) for a in range(3)]

    # Two passes per coil, as `clearance` does it: the decimated outline says
    # which nodes are certainly in and certainly out, and only the band it
    # cannot settle -- a shell a couple of centimetres thick around each
    # winding -- is put against the real 256-point curve. That is an eighth of
    # the cost of measuring everything exactly, and gives the same answer.
    # Nodes are built a slab at a time rather than all at once: the expanded
    # grid is bigger than the volume by the probe's own size on every side, and
    # materialising it as one (N, 3) float array is where a fine step runs the
    # machine out of memory long before it runs out of patience.
    threshold = radius_mm + margin_mm + snap_mm
    total = int(np.prod(big, dtype=np.int64))
    forbidden = np.zeros(total, dtype=bool)
    per_slab = max(1, REACH_CHUNK // max(1, big[1] * big[2]))
    for i0 in range(0, big[0], per_slab):
        i1 = min(i0 + per_slab, big[0])
        nodes = np.stack(np.meshgrid(axes[0][i0:i1], axes[1], axes[2],
                                     indexing="ij"), axis=-1).reshape(-1, 3)
        chunk = placement.flange_path_mm(nodes)
        hit = np.zeros(len(chunk), dtype=bool)
        for curve in curves:
            coarse, slack = _coarse_outline_of(curve)
            idx = np.flatnonzero(~hit)
            if not len(idx):
                break
            d2, _t = _segment_distances(chunk[idx], coarse)
            near = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
            hit[idx[near <= threshold - slack]] = True
            band = idx[(near > threshold - slack) & (near <= threshold + slack)]
            if len(band):
                d2, _t = _segment_distances(chunk[band], curve)
                exact = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
                hit[band[exact <= threshold]] = True
        forbidden[i0 * big[1] * big[2]:i1 * big[1] * big[2]] = hit
        if progress is not None:
            progress(i1, big[0])
    forbidden = forbidden.reshape(big)

    # ---- dilate: the flange cannot be anywhere that puts a cell in there ----
    blocked = np.zeros(shape, dtype=bool)
    for cell in cells:
        i, j, k = (int(cell[a] - lo_c[a]) for a in range(3))
        blocked |= forbidden[i:i + shape[0], j:j + shape[1], k:k + shape[2]]
    return ~blocked


def grid_standoff_mm(step_mm, coilset=None, labels=None):
    """How much standoff `reachable_grid` adds on top of the asked-for margin.

    Only the grid snap: the coil decimation is refined away rather than paid
    for. `coilset` is accepted and ignored so callers need not know that.
    """
    return float(step_mm) * math.sqrt(3.0) / 2.0


def clear_of_coils(origins_mm, placement, cloud_mount_mm, coilset, radius_mm,
                   margin_mm=0.0, labels=None, progress=None,
                   should_abort=None):
    """Exactly which of these flange positions the probe body can occupy.

    The authority, and the slow one: one full `clearance` per position, the
    same answer the tab prints for the pose on screen. `reachable_grid` plans
    with it; this checks what the plan actually ended up asking for, which is a
    far smaller set -- the ends of each scan line rather than every node of the
    volume they were carved out of.

    `origins_mm` are flange positions in MACHINE millimetres, one per row.
    Returns a boolean (N,) array.
    """
    origins = np.asarray(origins_mm, dtype=float).reshape(-1, 3)
    if coilset is None or not len(coilset) or not len(origins):
        return np.ones(len(origins), dtype=bool)
    body = np.asarray(cloud_mount_mm, dtype=float) @ placement.rotation().T
    ok = np.zeros(len(origins), dtype=bool)
    for i, here in enumerate(origins):
        if should_abort is not None and should_abort():
            break
        gap = clearance(here + body, coilset, radius_mm, labels=labels)
        ok[i] = gap.gap_mm > margin_mm
        if progress is not None and not i % 32:
            progress(i, len(origins))
    if progress is not None:
        progress(len(origins), len(origins))
    return ok


# ==========================================================================
# where the probe is: the mount pose, and the stage on top of it
# ==========================================================================

def rotation_matrix(rot_z_deg):
    """Rz -- the only rotation this rig has.

    The probe is bolted to a three-axis cartesian gantry that cannot tilt it:
    the tube lies horizontal along the rig's Y and the only freedom is which
    way round the machine the whole assembly is turned, about the machine's Z,
    which is the axis of the torus. Pitch and roll used to be settable and were
    always zero, and a pose box that can be set to something the rig cannot do
    is a way to produce a confident clearance number about an impossible
    position.
    """
    a = math.radians(rot_z_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Placement:
    """Where the probe's mounting flange sits in the machine, and how it moves.

    Pose is the flange with every stage at its zero; the stage reading is added
    in the mount frame, so driving x moves the probe along the rig's x wherever
    the assembly happens to be pointing.
    """

    AXES = ("x", "y", "z")

    def __init__(self, x_mm=0.0, y_mm=0.0, z_mm=0.0, rot_z_deg=0.0,
                 stage_zero_mm=None, travel_mm=None):
        self.x_mm = float(x_mm)
        self.y_mm = float(y_mm)
        self.z_mm = float(z_mm)
        self.rot_z_deg = float(rot_z_deg)
        self.stage_zero_mm = dict.fromkeys(self.AXES, 0.0)
        self.stage_zero_mm.update({k: float(v) for k, v
                                   in (stage_zero_mm or {}).items()
                                   if k in self.AXES})
        self.travel_mm = dict.fromkeys(self.AXES,
                                       (0.0, DEFAULT_TRAVEL_MM))
        for k, v in (travel_mm or {}).items():
            if k in self.AXES and v is not None and len(v) == 2:
                self.travel_mm[k] = (float(v[0]), float(v[1]))

    # ---- the transform ----
    def rotation(self):
        return rotation_matrix(self.rot_z_deg)

    def offset_mm(self, stage_mm=None):
        """Where the stages have carried the probe, in the mount frame."""
        stage_mm = stage_mm or {}
        return np.array([float(stage_mm.get(ax, self.stage_zero_mm[ax]))
                         - self.stage_zero_mm[ax] for ax in self.AXES])

    def origin_mm(self, stage_mm=None):
        """The flange, in machine mm."""
        base = np.array([self.x_mm, self.y_mm, self.z_mm])
        return base + self.rotation() @ self.offset_mm(stage_mm)

    def flange_path_mm(self, rig_mm):
        """(N, 3) rig positions -> where the flange sits, machine mm.

        The vector form of origin_mm(), for the things that ask about a whole
        set of stage positions at once rather than the one the probe is at: the
        corners of a volume, the ends of every line of a sweep, the nodes of a
        reachability grid. Same arithmetic, same stage zero, so a box drawn by
        this and a clearance computed by origin_mm() cannot disagree about
        where the rig's 100 mm is.
        """
        rig = np.asarray(rig_mm, dtype=float).reshape(-1, 3)
        zero = np.array([self.stage_zero_mm[a] for a in self.AXES])
        base = np.array([self.x_mm, self.y_mm, self.z_mm])
        return base + (rig - zero) @ self.rotation().T

    def to_machine(self, points_mount_mm, stage_mm=None):
        """Mount-frame points (probe_geometry's world frame) -> machine mm."""
        pts = np.asarray(points_mount_mm, dtype=float)
        return pts @ self.rotation().T + self.origin_mm(stage_mm)

    def reach_corners_mm(self):
        """The eight corners of the volume the flange can be driven through."""
        rot = self.rotation()
        base = np.array([self.x_mm, self.y_mm, self.z_mm])
        corners = []
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    local = np.array([
                        self.travel_mm["x"][i] - self.stage_zero_mm["x"],
                        self.travel_mm["y"][j] - self.stage_zero_mm["y"],
                        self.travel_mm["z"][k] - self.stage_zero_mm["z"]])
                    corners.append(base + rot @ local)
        return np.array(corners)

    # ---- persistence ----
    def to_dict(self):
        return {"x_mm": self.x_mm, "y_mm": self.y_mm, "z_mm": self.z_mm,
                "rot_z_deg": self.rot_z_deg,
                "stage_zero_mm": dict(self.stage_zero_mm),
                "travel_mm": {k: list(v) for k, v in self.travel_mm.items()}}

    _FIELDS = frozenset(("x_mm", "y_mm", "z_mm", "rot_z_deg",
                         "stage_zero_mm", "travel_mm"))

    @classmethod
    def from_dict(cls, doc, on_note=None):
        """Read a pose, including one written before the tilts were removed.

        `yaw_deg` was rotation about the machine's Z under a different name, so
        it is carried straight across. A file with a non-zero pitch or roll
        describes a pose this rig has never been able to reach; it is dropped
        rather than silently approximated, and said out loud, because the
        difference shows up as a clearance number and not as an error.
        """
        doc = dict(doc or {})
        if "rot_z_deg" not in doc and "yaw_deg" in doc:
            doc["rot_z_deg"] = doc["yaw_deg"]
        stale = [k for k in ("pitch_deg", "roll_deg")
                 if abs(float(doc.get(k) or 0.0)) > 1e-9]
        if stale and on_note is not None:
            on_note("the saved pose has " + " and ".join(
                f"{k.split('_')[0]} {float(doc[k]):g} deg" for k in stale)
                + ", which this rig cannot do -- dropped, leaving the "
                  "rotation about Z. Re-measure the pose if it mattered.")
        return cls(**{k: v for k, v in doc.items() if k in cls._FIELDS})

    def describe(self):
        return (f"flange at ({self.x_mm:+.0f}, {self.y_mm:+.0f}, "
                f"{self.z_mm:+.0f}) mm, turned {self.rot_z_deg:g} deg about Z")


class MachineConfig:
    """Everything the machine view needs to be told, and remembers.

    Kept in machine.json beside the other config files, for the same reason
    they are: the pose of a probe inside a coil set is measured once with a
    tape and a lot of care, and re-typing it every session is how it drifts.
    """

    def __init__(self, coil_file="", coil_radius_mm=DEFAULT_COIL_RADIUS_MM,
                 configuration="", current_scale=1.0, energised=None,
                 track_stage=True, pose=None, notes=""):
        self.coil_file = str(coil_file or "")
        self.coil_radius_mm = float(coil_radius_mm)
        self.configuration = str(configuration or "")
        self.current_scale = float(current_scale)
        # None means "not decided yet" and is resolved against the coil set on
        # load; an empty list means every coil is off, which is a real state
        # worth being able to save.
        self.energised = None if energised is None else list(energised)
        self.track_stage = bool(track_stage)
        # Anything the saved pose could not be read literally -- a tilt from
        # before the rig's freedoms were stated honestly. Held rather than
        # printed, because this is constructed before there is anywhere to
        # print to; the GUI empties it into the log when the tab announces.
        self.pose_notes = []
        self.pose = (pose if isinstance(pose, Placement)
                     else Placement.from_dict(pose, self.pose_notes.append))
        self.notes = str(notes or "")

    # ---- coils ----
    def adopt(self, coilset):
        """Reconcile the saved selection with the coil set actually loaded.

        A configuration name or coil label that is not in the file is dropped
        rather than carried: it would be a switch for a coil that does not
        exist, and the operator would have no way of telling it apart from one
        that does.
        """
        if coilset is None or not len(coilset):
            return []
        lost = []
        if self.configuration not in coilset.configurations:
            if self.configuration:
                lost.append(f"configuration {self.configuration!r} is not in "
                            f"this file")
            self.configuration = coilset.configurations[0]
        if self.energised is None:
            self.energised = list(coilset.labels)
        else:
            unknown = [c for c in self.energised if c not in coilset.labels]
            if unknown:
                lost.append(f"no coil(s) {', '.join(unknown)} in this file")
            self.energised = [c for c in self.energised if c in coilset.labels]
        return lost

    def is_on(self, label):
        return self.energised is None or label in self.energised

    def current(self, coil):
        return coil.current(self.configuration, self.current_scale)

    def energised_summary(self, coilset):
        """One line for the log and for the metadata of a scan."""
        if coilset is None or not len(coilset):
            return "no coil set loaded"
        on = [c for c in coilset if self.is_on(c.label)]
        if not on:
            return "every coil off"
        amps = ", ".join(f"{c.label} {self.current(c) / 1000:+.2f} kA-turns"
                         for c in on)
        return (f"{len(on)}/{len(coilset)} coils energised, "
                f"{self.configuration} x{self.current_scale:g}: {amps}")

    def to_scan_meta(self, coilset, stage_mm=None):
        """What a saved field map should carry about the machine around it.

        Written into the map's sidecar so that a file found next year still
        says which coils were on, at what current, and where the probe was
        bolted -- none of which can be recovered from the numbers themselves.
        """
        meta = {"pose": self.pose.to_dict(),
                "pose_note": self.pose.describe(),
                "coil_radius_mm": self.coil_radius_mm,
                "tracked_stage": self.track_stage,
                "notes": self.notes}
        if coilset is None or not len(coilset):
            meta["coils"] = None
            return meta
        meta["coil_file"] = os.path.abspath(coilset.path)
        meta["configuration"] = self.configuration
        meta["current_scale"] = self.current_scale
        meta["summary"] = self.energised_summary(coilset)
        meta["coils"] = {c.label: {"on": self.is_on(c.label),
                                   "amp_turns": self.current(c),
                                   "curve": c.description}
                         for c in coilset}
        if stage_mm:
            meta["probe_origin_mm"] = self.pose.origin_mm(stage_mm).tolist()
        return meta

    # ---- persistence ----
    def to_dict(self):
        return {"coil_file": self.coil_file,
                "coil_radius_mm": self.coil_radius_mm,
                "configuration": self.configuration,
                "current_scale": self.current_scale,
                "energised": self.energised,
                "track_stage": self.track_stage,
                "pose": self.pose.to_dict(),
                "notes": self.notes}

    _FIELDS = frozenset(("coil_file", "coil_radius_mm", "configuration",
                         "current_scale", "energised", "track_stage", "pose",
                         "notes"))

    def save(self, path=None):
        path = path or paths.config(CONFIG_NAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path=None):
        path = path or paths.config(CONFIG_NAME)
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict):
            raise ValueError(f"{path}: expected a JSON object, got "
                             f"{type(doc).__name__}")
        return cls(**{k: v for k, v in doc.items() if k in cls._FIELDS})

    @classmethod
    def load_or_default(cls, path=None, on_error=None):
        path = path or paths.config(CONFIG_NAME)
        if path and os.path.exists(path):
            try:
                return cls.load(path)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                if on_error is not None:
                    on_error(f"{path} exists but could not be read "
                             f"({type(exc).__name__}: {exc}) -- the machine "
                             f"view starts with nothing placed, and the pose "
                             f"in that file will be overwritten if it is "
                             f"saved again.")
        return cls()


# ==========================================================================
# command line
# ==========================================================================

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("coil_file", nargs="?",
                   help="a simsopt configuration file (default: whatever "
                        f"{CONFIG_NAME} points at)")
    p.add_argument("--config", default=paths.config(CONFIG_NAME),
                   help=f"placement file to read (default: {CONFIG_NAME})")
    p.add_argument("--at", nargs=3, type=float, metavar=("X", "Y", "Z"),
                   help="stage reading, mm, to place the probe at")
    p.add_argument("--radius", type=float, default=None,
                   help="coil cross-section radius in mm, overriding the "
                        "config")
    args = p.parse_args(argv)

    cfg = MachineConfig.load_or_default(args.config, on_error=print)
    path = args.coil_file or cfg.coil_file
    if not path:
        print(f"no coil file given and none named in {args.config}")
        return 2
    if args.radius is not None:
        cfg.coil_radius_mm = args.radius

    coils = CoilSet.load(path)
    for problem in cfg.adopt(coils):
        print(f"note: {problem}")
    print(coils.summary(cfg.configuration, cfg.current_scale, cfg.energised))
    print()
    print(cfg.energised_summary(coils))
    print(cfg.pose.describe())

    stage = None
    if args.at:
        stage = dict(zip(Placement.AXES, args.at))
        print("stage at " + ", ".join(f"{k}={v:g} mm" for k, v in stage.items()))
    geom = pgeom.Geometry.load_or_default(on_error=print)
    cloud = cfg.pose.to_machine(probe_cloud(geom), stage)
    gap = clearance(cloud, coils, cfg.coil_radius_mm)
    print(f"probe flange at {cfg.pose.origin_mm(stage)} mm")
    print("clearance: " + gap.text())
    return 1 if gap.collides else 0


if __name__ == "__main__":
    sys.exit(main())
