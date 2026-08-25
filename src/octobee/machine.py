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

from octobee.calib import geometry as pgeom

CONFIG_NAME = "machine.json"

# The winding pack has to be given a thickness before anything can be said
# about clearance, and no simsopt file records one: the optimiser works with
# infinitely thin filaments. 20 mm is a placeholder that is honest about being
# one -- it is the number to change first, in the tool, once the real conductor
# is measured.
DEFAULT_COIL_RADIUS_MM = 20.0

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


# ==========================================================================
# where the probe is: the mount pose, and the stage on top of it
# ==========================================================================

def rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    """Rz(yaw) Ry(pitch) Rx(roll) -- yaw applied last, as read out loud.

    Yaw is the one that matters on this rig (it swings the probe round the
    torus), so it is the outermost: changing it turns the whole assembly about
    the machine's Z axis whatever pitch and roll are already set, which is what
    somebody typing into the box expects.
    """
    y, p, r = (math.radians(v) for v in (yaw_deg, pitch_deg, roll_deg))
    cz, sz = math.cos(y), math.sin(y)
    cy, sy = math.cos(p), math.sin(p)
    cx, sx = math.cos(r), math.sin(r)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


class Placement:
    """Where the probe's mounting flange sits in the machine, and how it moves.

    Pose is the flange with every stage at its zero; the stage reading is added
    in the mount frame, so driving x moves the probe along the rig's x wherever
    the assembly happens to be pointing.
    """

    AXES = ("x", "y", "z")

    def __init__(self, x_mm=0.0, y_mm=0.0, z_mm=0.0,
                 yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
                 stage_zero_mm=None, travel_mm=None):
        self.x_mm = float(x_mm)
        self.y_mm = float(y_mm)
        self.z_mm = float(z_mm)
        self.yaw_deg = float(yaw_deg)
        self.pitch_deg = float(pitch_deg)
        self.roll_deg = float(roll_deg)
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
        return rotation_matrix(self.yaw_deg, self.pitch_deg, self.roll_deg)

    def offset_mm(self, stage_mm=None):
        """Where the stages have carried the probe, in the mount frame."""
        stage_mm = stage_mm or {}
        return np.array([float(stage_mm.get(ax, self.stage_zero_mm[ax]))
                         - self.stage_zero_mm[ax] for ax in self.AXES])

    def origin_mm(self, stage_mm=None):
        """The flange, in machine mm."""
        base = np.array([self.x_mm, self.y_mm, self.z_mm])
        return base + self.rotation() @ self.offset_mm(stage_mm)

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
                "yaw_deg": self.yaw_deg, "pitch_deg": self.pitch_deg,
                "roll_deg": self.roll_deg,
                "stage_zero_mm": dict(self.stage_zero_mm),
                "travel_mm": {k: list(v) for k, v in self.travel_mm.items()}}

    _FIELDS = frozenset(("x_mm", "y_mm", "z_mm", "yaw_deg", "pitch_deg",
                         "roll_deg", "stage_zero_mm", "travel_mm"))

    @classmethod
    def from_dict(cls, doc):
        return cls(**{k: v for k, v in (doc or {}).items() if k in cls._FIELDS})

    def describe(self):
        return (f"flange at ({self.x_mm:+.0f}, {self.y_mm:+.0f}, "
                f"{self.z_mm:+.0f}) mm, yaw {self.yaw_deg:g} deg, "
                f"pitch {self.pitch_deg:g} deg, roll {self.roll_deg:g} deg")


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
        self.pose = pose if isinstance(pose, Placement) else Placement.from_dict(pose)
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

    def save(self, path=CONFIG_NAME):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path=CONFIG_NAME):
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict):
            raise ValueError(f"{path}: expected a JSON object, got "
                             f"{type(doc).__name__}")
        return cls(**{k: v for k, v in doc.items() if k in cls._FIELDS})

    @classmethod
    def load_or_default(cls, path=CONFIG_NAME, on_error=None):
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
    p.add_argument("--config", default=CONFIG_NAME,
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
    geom = pgeom.Geometry.load_or_default(pgeom.CONFIG_NAME, on_error=print)
    cloud = cfg.pose.to_machine(probe_cloud(geom), stage)
    gap = clearance(cloud, coils, cfg.coil_radius_mm)
    print(f"probe flange at {cfg.pose.origin_mm(stage)} mm")
    print("clearance: " + gap.text())
    return 1 if gap.collides else 0


if __name__ == "__main__":
    sys.exit(main())
