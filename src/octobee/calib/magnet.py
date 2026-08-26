#!/usr/bin/env python3
"""
octobee/calib/magnet.py -- the guided single-magnet calibration: one magnet, a
motorised sweep of the plane the sensors lie in, repeated once per quarter turn
of the head.

What it is for
--------------
Two separate things, from one set of passes:

  1. A GAIN TRIM that owes nothing to a geometry model.
  2. The SENSOR-TO-FACE MAPPING and the physical order along the tube, which
     `probe_geometry.json` still carries as an assumption.

Why the geometry cancels
------------------------
The hand magnet pass on the Calibration tab has one unavoidable weakness: a
magnet held near the probe is at a different distance from every chip, and
field falls off as roughly 1/r^3, so the raw peaks span orders of magnitude
before any gain difference is involved. Dividing that out needs
`probe_geometry.expected_response()` -- i.e. it needs the geometry file to be
right, which is exactly what is not yet established. The correction and the
thing to be corrected are the same unknown.

Driving the probe along its own axis past a FIXED magnet removes it. The tube
axis lies along the rig's y (see probe_geometry.MOUNT_ROT), so a sweep in y
carries S1, S2, S3, S4 of one face past the magnet at the same NOMINAL closest
approach, one after another. Turn the head 90 degrees and the next face gets
its turn on the same terms. After four poses all 16 peaks are comparable, and

    trim_i = median(peak) / peak_i

is a gain ratio with no 1/r^n anywhere in it.

Where the axial sweep alone is not enough
-----------------------------------------
"Same nominal approach" is doing real work in that paragraph, and it is only
true if the sixteen arms are identical. They are not. Each chip rides at the
tip of a 92 mm board, so a degree of board rotation about its mounting bolt, a
bolt-hole clearance, or a seated-proud foot puts the chip a millimetre from
where the file says. At 1/r^3 and a 20 mm standoff,

    dB/B ~ 3 dr/r  =  15% per millimetre

which is larger than the gain spread the trim is trying to measure. Left alone,
the trim mostly records where the arms are.

Resolve the misplacement along the three directions of the magnet-to-chip line
and they behave completely differently:

  * ALONG THE TUBE -- the axial sweep already passes through it, so the peak is
    taken at the top of the curve and the error is second order. Handled.

  * TRANSVERSE (across the face, the direction the arms reach) -- the axial
    sweep takes a SLICE through the peak at whatever offset the operator
    parked at, and the error is first order. Sweeping the transverse axis too
    and peaking over the plane finds each sensor's own maximum instead, which
    drops 1 mm at 20 mm from 15% to 3 d^2/2 r^2 = 0.4%. This is the plane
    sweep, and it is why this module works in two axes rather than one.

  * ALONG THE LINE (the standoff) -- no amount of sweeping in the plane touches
    this one, because |B| has no maximum in that direction; it just keeps
    rising as the chip gets closer. What that direction carries is not a peak
    but a SLOPE, and a few points of dither along it measure the standoff
    outright:

        |B| = C |d - z|^-n   =>   dln|B|/dz = n/d,  d2ln|B|/dz2 = n/d^2

    so d = slope / curvature and n = slope^2 / curvature, both of them read off
    the dither with no constant assumed and nothing taken from the geometry
    file. Correcting each sensor to a common standoff removes the last
    first-order term. See fit_falloff() and Dither.

Three passes per pose, then, not one:

    A  locate   coarse axial sweep -- finds the four rings of this face
    B  cut      transverse sweep at each ring    -- the plane peak
    C  dither   a few standoffs at each ring     -- the distance

which is about 5x the points of a bare axial sweep, not the 900x a full
volume raster would cost to learn the same thing. suggested_sweep(),
suggested_plane() and suggested_dither() size the three.

B and C are a package -- do not run one without the other
---------------------------------------------------------
This is the one part of the routine that is not obvious, and running B alone
makes the trim WORSE, so it is worth being plain about.

Pass C fits |B| = C |d - z|^-n, which is a model of moving straight TOWARD the
magnet. It is only that if the chip is directly under the magnet when the
dither starts. If the chip is off to one side by a, and the magnet is h away
along the dither axis, then what comes back is not the distance:

    d_fit = h r^2 / (h^2 - a^2),    r^2 = a^2 + h^2

-- on this rig's own geometry, 51 mm for a chip that is really 26 mm away.
Pass B is what makes a ~ 0, so it is a precondition for C meaning anything
rather than an accuracy improvement on it.

The dependency runs the other way too. Peaking over the plane moves every chip
from wherever the operator parked it to its true closest approach, which is
NEARER -- on the same geometry, 51 mm of effective standoff becomes 26 mm --
and every first-order error still left is amplified by that ratio. So pass B on
its own takes a 9.9 % trim error to 12.8 %, and only B and C together take it
to 1.8 %.

Run both, or run neither.

A run with no transverse axis and no dither still works and still reports; it
is the old one-axis measurement, with the first-order terms left in. Every
correction here degrades to the identity when its pass is missing.

Two conditions to keep the geometry cancelled, which is why the routine is
guided rather than a button:

  * The magnet must not move between poses, and nothing ferrous may. It is the
    common reference for all four.
  * The head must be re-indexed about its own axis and nothing else. A pose
    that also shifts the head sideways breaks the equal-approach argument
    quietly -- the numbers still look plausible.

What the order tells you
------------------------
Each sensor peaks at the y where it passed the magnet, so sorting the 16 peak
positions puts the sensors in their true physical order along the tube, and
grouping by which pose was loudest sorts them onto faces. Both are measured,
not assumed, and `check_geometry()` reports where they disagree with
probe_geometry.json rather than silently rewriting it.

A note on what this does NOT do
-------------------------------
It does not calibrate absolute scale, and it does not fix chip orientation.
The Earth-field roll sweep (octobee/calib/roll.py) is still what matches the
sensors to each other in three dimensions and pins the axes; this fixes the
scalar per-sensor response and identifies which sensor is where. They are
complementary: run this first, because knowing which sensor is on which face
makes the roll solve's per-face report readable.
"""

import argparse
import contextlib
import json
import os

import numpy as np

from octobee import paths
from octobee.calib import convert as ocal
from octobee.calib import geometry as pgeom

N_SENSORS = pgeom.N_SENSORS
N_POSES = pgeom.N_FACES          # a square tube indexes in quarter turns

# Which face is toward the magnet in each pose, if pose 0 has face 0 there.
# Only used to phrase the report; nothing here depends on it being right.
POSE_NAMES = tuple(f"pose {i + 1}" for i in range(N_POSES))

# A sensor whose best peak is this far below the median is not "quiet", it is
# not responding: 1/r^3 over the length of one face is a factor of a few, not
# a factor of fifty. Reported rather than trimmed.
DEAD_FRACTION = 0.02

# How far two opposite faces may differ before it is worth saying so. They see
# the same magnet from the same nominal distance, so anything much past this is
# the head not being concentric in its cradle rather than the chips differing
# -- see MagnetRun.face_balance(). 5 % of field is under 2 % of distance, which
# is a few tenths of a millimetre: tight enough to be worth reporting, loose
# enough not to cry wolf about a well-set-up run.
FACE_IMBALANCE_WARN = 0.05

# When to disbelieve pass C, for MagnetRun.dither_quality(). All three of these
# were set from the run of 2026-08-25, which was sized for a 20 mm standoff and
# driven against a real ~50 mm one: the dither was then a tenth of the distance
# instead of a quarter, and the fit went degenerate without failing.
#
# It did not look broken from the outside. Every sensor returned a finite
# distance, the residuals were small, and the correction sailed into the trim
# and multiplied its spread from 1.30x to 5.9x. What DID show it was these:
#
#   - the exponent. It is a property of the magnet, so all sixteen sensors must
#     agree on it, and a dipole gives 3. That run returned 1.85 to 4.98.
#   - the correlation between fitted distance and fitted exponent, ACROSS the
#     sixteen. Real measurements of sixteen chip positions have no reason to
#     correlate with anything; a fit sliding along a degenerate valley
#     correlates near 1. That run: 0.994.
#
# The correlation is the sharper of the two and the one to trust: it needs no
# idea of what the magnet is, only that the sensors are independent.
DITHER_N_TOLERANCE = 0.8         # |median n - 3| worth refusing over
DITHER_DEGENERACY_R = 0.9        # |corr(d, n)| above this is one fit, not 16
DITHER_STANDOFF_TOLERANCE = 0.3  # fitted vs entered standoff, as a fraction

# The standoff fit searches for the magnet between these distances from the
# dither centre, in millimetres. The lower bound is not a guess about the rig:
# a magnet closer than this is inside the dither's own travel, where the model
# has a pole and the fit would chase it. The upper bound is where a 5 mm dither
# stops being able to see any curvature at all, so a "solution" out there is
# the fit reporting that it found nothing.
FIT_RANGE_MM = (5.0, 400.0)
FIT_STEPS = 2000

# A dither whose ln|B| swings less than this across its span has not moved the
# sensor anywhere the field cares about, so slope and curvature are both noise.
# 2 % is a few times the per-point SEM at a second of averaging.
MIN_DITHER_CONTRAST = 0.02

# A fitted falloff exponent outside this band is not a magnet seen from a
# sensible distance -- it is a bad fit, a saturated channel, or a sensor that
# was nowhere near the magnet. Reported, not used.
FALLOFF_BAND = (1.0, 6.0)

# ---- when to say the sixteen chips are not turned the same way ------------
# MagnetRun.orientation_check() compares the sixteen peak VECTORS after the
# pose is undone. Under the geometry file being right they are sixteen looks
# at one physical vector and should agree to within the noise on a peak, which
# on this rig is a fraction of a degree.
#
# The two thresholds are deliberately different quantities and not two
# sensitivities of the same one:
#
#   POSE   a whole face off together is the head not having been indexed a
#          true quarter turn. Four chips do not turn together by accident. It
#          does not touch the trim -- the trim is a magnitude -- so this is
#          reported rather than refused, and 3 deg is comfortably past the
#          couple of degrees a V-block or a scribed line actually repeats to.
#   SENSOR one chip off on its own, after its own face's offset is taken out,
#          is that chip: mounted turned, or its axis_signs wrong. 5 deg is
#          about where a mounting error stops being plausible as fit noise on
#          a peak and starts being a board seated wrong.
ORIENT_POSE_WARN_DEG = 3.0
ORIENT_SENSOR_WARN_DEG = 5.0

# How tightly the sixteen unit vectors must cluster before the routine will
# claim to have worked out which way the head was turned. The mean of sixteen
# agreeing unit vectors has length ~1; sixteen scattered ones average toward
# 0. Below this, neither turn direction explains the data and saying which one
# "won" would be reading a coin flip as a measurement.
ORIENT_CLUSTER_MIN = 0.90

# ---- when to say a channel ran out of range -------------------------------
# MagnetRun.range_check(). The sensor's analogue output clips at its own full
# scale, and the failure is silent and asymmetric: a clipped peak reads LOW,
# so the trim compensates by turning that sensor UP. 80 % leaves room for the
# pose-to-pose variation a real run has without crying wolf, and is well
# inside the SENM3Dx's own linearity spec.
#
# RANGE_TARGET_OF_WARN is what "far enough away" means when the report
# suggests a distance: aim for three quarters of the warning line rather than
# the line itself, so the suggested move does not land the next run one per
# cent inside the same warning.
RANGE_WARN_FRACTION = 0.80
RANGE_TARGET_OF_WARN = 0.75

# Dither half-span as a fraction of the standoff being measured, and how many
# points to put across it. The temptation is to dither by a hair, on the
# reasoning that a small perturbation is a clean one. It is exactly wrong here:
# the standoff comes out of the CURVATURE, which is second order, so a +-1 mm
# dither at 20 mm buries the thing being measured 0.4 % below the slope and the
# fit returns noise.
#
# Both numbers are off a sweep of the synthetic bench rather than off a
# derivation, because what is being traded is fit precision against the field
# rise at the near end of the dither, and the second one is a hardware limit.
# Residual trim error, 1 mm of misplacement on all three axes, 3 uT/point:
#
#       fraction   points   near-end field    trim error
#         0.15        5          1.6x            7.5 %
#         0.25        5          2.4x            4.2 %
#         0.25        7          2.4x            2.1 %     <- this
#         0.35        7          3.6x            1.9 %
#         0.45        7          6.0x            2.1 %
#
# Seven points beat five everywhere and cost two moves per ring. Past a quarter
# of the standoff the accuracy stops improving but the near end of the dither
# keeps climbing toward the top of the range: at 0.35 a 5 mT peak becomes
# 18 mT, which is where the 20 mT range clips and the fit is then reading a
# flat top rather than a field. A quarter keeps that at 2.4x with no measurable
# loss, so it is the one to have.
DITHER_FRACTION = 0.25
DITHER_POINTS = 7


def _robust_direction(u):
    """(n, 3) unit vectors -> the direction they agree on, ignoring one rebel.

    A plain mean is the wrong centre for this, and not by a little. The whole
    point of comparing sixteen chips is to find the one that is turned wrong,
    and that chip is inside the mean it is being compared against: turning one
    of four sensors by 30 degrees drags their mean by about 8, so its three
    innocent siblings then read 8 degrees out and the culprit reads 22. Every
    sensor on the face looks guilty and the actual answer is buried.

    So the worst-agreeing member is dropped and the centre recomputed without
    it. One outlier is what this can survive, which for four sensors on a face
    is the honest limit -- two chips turned the same wrong way on one face are
    indistinguishable from the other two being turned, and no amount of
    statistics on four vectors fixes that.
    """
    u = np.asarray(u, float)
    if len(u) == 0:
        return None
    m = np.mean(u, axis=0)
    n = np.linalg.norm(m)
    if n <= 0:
        return None
    m = m / n
    if len(u) < 4:
        return m
    keep = np.ones(len(u), bool)
    keep[int(np.argmin(u @ m))] = False
    m2 = np.mean(u[keep], axis=0)
    n2 = np.linalg.norm(m2)
    return m if n2 <= 0 else m2 / n2


def peak_response(b_mt):
    """(npoints, 16, 3) uncorrected mT -> per-sensor peak |B| and its index.

    Peak rather than mean because the magnet is only near a given sensor for a
    few points of the sweep; the mean would mostly measure how long the sweep
    was.
    """
    b = np.asarray(b_mt, float)
    mag = np.linalg.norm(b, axis=2)                  # (npoints, 16)
    if mag.size == 0:
        raise ValueError("empty sweep: no points were captured")
    idx = np.argmax(mag, axis=0)
    return mag[idx, np.arange(mag.shape[1])], idx


def fit_falloff(z_mm, b_mag):
    """|B| at a few standoffs -> (distance, exponent, residual, uncertainty).

    Fits |B| = C |d - z|^-n, where z is the offset from wherever the dither was
    centred and d is the signed offset from that centre to the magnet. Both d
    and n come out of the data; nothing is assumed about the magnet, and
    nothing is read from probe_geometry.json.

    Why fit rather than difference: the closed form d = slope/curvature is
    exact for this model but only for infinitesimal z, and a dither big enough
    to SEE the curvature is not infinitesimal -- a quarter of the standoff
    leaves several per cent of truncation error in d, which is most of what the
    correction was worth. Fitting the model itself has no such term.

    Why a grid search rather than a solver: for a fixed d the model is linear
    in ln|B| against ln|d - z|, so the only genuinely nonlinear parameter is d,
    and it is one-dimensional and bounded. Scanning it is a dozen lines with no
    dependency, no starting guess, and no local minimum to fall into. scipy is
    not a dependency of this project and this is not the reason to make it one.

    The sign of d says which side the magnet is on, which is worth keeping: on
    a rig where +z is up and the magnet is clamped above, a negative d means
    the dither axis is inverted relative to what the operator thinks.

    The fourth return is what stops this doing harm. d comes out of a
    CURVATURE, which is a second-order feature of a handful of noisy points,
    and it is entirely possible for the fit to return a confident-looking
    distance that is mostly noise -- at which point "correcting" for it injects
    that noise straight into the trim. So the uncertainty is measured too, from
    the width of the region where the residual is statistically indistinguish-
    able from the best one (the usual one-parameter profile interval), and
    MagnetRun.standoff_correction() uses it to decide how much of each
    sensor's apparent displacement to believe.

    Returns all-nan when the dither says nothing -- too few points, no
    contrast, or a best fit sitting on the edge of the search range, which
    means the data did not constrain it.
    """
    z = np.asarray(z_mm, float).ravel()
    b = np.asarray(b_mag, float).ravel()
    bad = (np.nan, np.nan, np.nan, np.nan)
    ok = np.isfinite(z) & np.isfinite(b) & (b > 0)
    z, b = z[ok], b[ok]
    # Three points are the minimum that can separate a slope from a curvature,
    # and the model has three parameters.
    if len(z) < 3 or np.ptp(z) <= 0:
        return bad
    logb = np.log(b)
    if np.ptp(logb) < MIN_DITHER_CONTRAST:
        return bad

    # d comes out relative to the centre of the dither, so centre the offsets
    # here rather than trusting every caller to have done it. It also keeps the
    # trial grid, which is symmetric about zero, from ever landing exactly on a
    # sample -- which is a log(0) and an inf residual that argmin would then
    # happily pick as the best fit.
    z = z - np.mean(z)
    reach = np.max(np.abs(z))
    lo = max(FIT_RANGE_MM[0], 1.2 * reach)
    if lo >= FIT_RANGE_MM[1]:
        return bad
    mags = np.geomspace(lo, FIT_RANGE_MM[1], FIT_STEPS)
    trial = np.concatenate([-mags[::-1], mags])       # the magnet may be either side

    # For every trial d at once: regress ln|B| on ln|d - z| and keep the
    # residual. (FIT_STEPS*2, nz) is a few hundred kilobytes.
    u = np.log(np.abs(trial[:, None] - z[None, :]))
    du = u - u.mean(axis=1, keepdims=True)
    dy = logb - logb.mean()
    var = np.sum(du * du, axis=1)
    var[var <= 0] = np.nan
    slope = np.sum(du * dy, axis=1) / var             # = -n
    resid = dy - slope[:, None] * du
    rms = np.sqrt(np.mean(resid * resid, axis=1))

    i = int(np.nanargmin(rms))
    # A minimum against a stop is the fit saying the data does not pin d down:
    # accepting it would report a confident distance derived from nothing.
    if i in (0, len(trial) - 1) or not np.isfinite(rms[i]):
        return bad
    d = float(trial[i])
    n = float(-slope[i])
    if not (FALLOFF_BAND[0] <= n <= FALLOFF_BAND[1]):
        return bad

    # How far d can move before the fit is measurably worse. Three parameters
    # are used (C, n, d), so a fit with only three points has nothing left to
    # estimate the noise with and cannot claim any precision at all.
    dof = len(z) - 3
    if dof < 1:
        return abs(d), n, float(rms[i]), np.inf
    chi2 = rms ** 2
    inside = np.isfinite(chi2) & (chi2 <= chi2[i] * (1.0 + 1.0 / dof))
    same = trial[inside]
    # Only the lobe around the minimum: the mirror-image solution on the other
    # side of the dither can fit just as well, and its distance is not evidence
    # about how well THIS one is pinned down.
    same = same[np.sign(same) == np.sign(d)]
    sigma = float(np.max(np.abs(same - d))) if len(same) else np.inf
    return abs(d), n, float(rms[i]), sigma


class Dither:
    """A few points along the standoff direction, taken at one located peak.

    This is the only pass that measures a DISTANCE rather than comparing two
    readings, and it is what lets the trim be gain rather than gain plus
    wherever that particular arm happens to have put its chip.

    `at_mm` is where along the tube the dither was taken -- the ring it
    belongs to -- so a sensor can find the dither that was done at its own
    peak rather than at some other ring's.
    """

    def __init__(self, z_mm, b_mt, at_mm=None, note=""):
        self.z_mm = np.asarray(z_mm, float).ravel()
        self.b_mt = np.asarray(b_mt, float)
        self.at_mm = None if at_mm is None else float(at_mm)
        self.note = note
        if self.b_mt.ndim != 3 or self.b_mt.shape[1] != N_SENSORS:
            raise ValueError(
                f"dither: expected (npoints, {N_SENSORS}, 3) field data, got "
                f"{self.b_mt.shape}")
        if len(self.z_mm) != len(self.b_mt):
            raise ValueError(
                f"dither: {len(self.z_mm)} offsets but {len(self.b_mt)} rows")
        self._fit = None

    def __len__(self):
        return len(self.z_mm)

    def fit(self):
        """(16,) each of standoff mm, exponent, residual, sigma -- nan if unfit."""
        if self._fit is None:
            mag = np.linalg.norm(self.b_mt, axis=2)          # (nz, 16)
            out = np.array([fit_falloff(self.z_mm, mag[:, i])
                            for i in range(N_SENSORS)], float)
            self._fit = tuple(out[:, k] for k in range(out.shape[1]))
        return self._fit

    @property
    def standoff_mm(self):
        return self.fit()[0]

    @property
    def falloff(self):
        return self.fit()[1]


class PoseSweep:
    """One pose: the head at one quarter turn, and everything measured there.

    Positions are held as (npoints, naxes) with `axes` naming the columns, so a
    bare axial sweep and a plane sweep are the same object with a different
    number of columns and the rest of this module cannot tell them apart. The
    three named axes are:

        axis    along the tube -- the one the sensors are strung out on
        across  transverse, the direction the arms reach; the plane's other
                axis, and the one whose peak the plane sweep is looking for
        normal  the standoff, toward the magnet; not swept, only dithered

    `dithers` are the pass-C measurements, one per ring. They are kept beside
    the sweep rather than mixed into it because they deliberately visit points
    CLOSER to the magnet than the peak: concatenated in, they would win every
    argmax and quietly redefine "peak" as "wherever the dither went nearest".
    """

    def __init__(self, pose, pos_mm, b_mt, axis="y", across=None, normal=None,
                 axes=None, dithers=(), note=""):
        self.pose = int(pose)
        pos = np.asarray(pos_mm, float)
        if pos.ndim == 1:
            pos = pos[:, None]
        self.pos_mm = pos
        self.b_mt = np.asarray(b_mt, float)
        self.axis = axis
        self.across = across
        self.normal = normal
        self.dithers = list(dithers)
        self.note = note
        if axes is None:
            if pos.shape[1] != 1:
                raise ValueError(
                    f"pose {pose}: {pos.shape[1]} position columns but no "
                    f"names for them -- pass axes=(...)")
            axes = (axis,)
        self.axes = tuple(axes)
        if len(self.axes) != pos.shape[1]:
            raise ValueError(
                f"pose {pose}: {len(self.axes)} axis names for "
                f"{pos.shape[1]} position columns")
        if self.b_mt.ndim != 3 or self.b_mt.shape[1] != N_SENSORS:
            raise ValueError(
                f"pose {pose}: expected (npoints, {N_SENSORS}, 3) field data, "
                f"got {self.b_mt.shape}")
        if len(self.pos_mm) != len(self.b_mt):
            raise ValueError(
                f"pose {pose}: {len(self.pos_mm)} positions but "
                f"{len(self.b_mt)} field rows")

    @classmethod
    def from_fieldmap(cls, pose, fm, axis="y", across=None, normal=None,
                      note=""):
        """Build from an octobee_scan.FieldMap, keeping every axis it moved."""
        pos = np.asarray(fm.pos_mm, float)
        if pos.ndim == 1:
            pos = pos[:, None]
        axes = tuple(fm.axes) if getattr(fm, "axes", None) else (axis,)
        if axis not in axes:
            # A map that did not move the tube axis cannot be a pose of this
            # routine, but the old one-axis callers relied on column 0 being
            # it, so keep that reading rather than refusing.
            axes = (axis, *axes[1:])
        return cls(pose, pos, fm.b_mt, axis=axis, across=across,
                   normal=normal, axes=axes, note=note)

    def merge(self, other):
        """Append another pass's rows to this one. Same pose, same axes.

        Passes A and B are two scans of the same plane at different densities,
        so the peak is the peak over both of them together -- the coarse pass
        is not thrown away once the fine one exists, it is part of the same
        point cloud.
        """
        if tuple(other.axes) != self.axes:
            raise ValueError(
                f"cannot merge a pass over {other.axes} into one over "
                f"{self.axes}: the columns are different quantities")
        self.pos_mm = np.vstack([self.pos_mm, other.pos_mm])
        self.b_mt = np.concatenate([self.b_mt, other.b_mt], axis=0)
        self.dithers.extend(other.dithers)
        return self

    # ---- one axis at a time ---------------------------------------------
    def column(self, name):
        """(npoints,) positions along one named axis, or None if not swept."""
        if name is None or name not in self.axes:
            return None
        return self.pos_mm[:, self.axes.index(name)]

    @property
    def along(self):
        """(npoints,) position along the tube. Column 0 if the name is absent."""
        col = self.column(self.axis)
        return self.pos_mm[:, 0] if col is None else col

    @property
    def is_plane(self):
        return self.across is not None and self.across in self.axes

    @property
    def peaks(self):
        """(16,) peak |B| in mT for each sensor over every point of this pose.

        Over the PLANE when there is one, which is the whole point: a sensor
        whose arm sits a millimetre off the swept line peaks at a different
        transverse position from its neighbours, and taking the maximum over
        both axes finds each one's own maximum instead of a slice through it.
        """
        return peak_response(self.b_mt)[0]

    @property
    def peak_at_mm(self):
        """(16,) position along the tube at which each sensor peaked."""
        return self.along[peak_response(self.b_mt)[1]]

    @property
    def peak_vector(self):
        """(16, 3) the field VECTOR at each sensor's own peak, chip frame, mT.

        Everything else in this module works on |B|, because a magnitude is
        rotation-invariant and that is the whole reason sixteen differently
        turned chips can be compared at all. This is the one place the three
        components are kept, and it is kept for the opposite reason: what a
        magnitude throws away is precisely the orientation information, and at
        the top of the plane peak the magnet is in a known place relative to
        the chip, so the direction of this vector in the chip's own frame is a
        measurement of how that chip is turned.

        Taken at the same index peaks and peak_at_mm use, so the three always
        describe the same point of the sweep.
        """
        idx = peak_response(self.b_mt)[1]
        return self.b_mt[idx, np.arange(N_SENSORS), :]

    @property
    def peak_across_mm(self):
        """(16,) transverse position of each sensor's peak, or None.

        This is a measurement of where the arms actually are, to whatever the
        stage repeats to. The spread across four sensors of one face is the
        arm-placement scatter that the plane sweep exists to remove.
        """
        col = self.column(self.across)
        if col is None:
            return None
        return col[peak_response(self.b_mt)[1]]

    # ---- what the dither says -------------------------------------------
    def _dither_for(self, at_mm):
        """The dither taken nearest this position along the tube."""
        placed = [d for d in self.dithers if d.at_mm is not None]
        if not placed:
            return self.dithers[0] if len(self.dithers) == 1 else None
        return min(placed, key=lambda d: abs(d.at_mm - at_mm))

    def _dither_fit(self, which):
        out = np.full(N_SENSORS, np.nan)
        if not self.dithers:
            return out
        at = self.peak_at_mm
        for i in range(N_SENSORS):
            d = self._dither_for(at[i])
            if d is not None:
                out[i] = d.fit()[which][i]
        return out

    @property
    def standoff_mm(self):
        """(16,) distance from each sensor to the magnet, nan where unmeasured.

        Taken from the dither done at that sensor's own ring. Doing every
        sensor's dither at the mean transverse peak rather than its own costs
        nothing: at the top of the plane peak the field is flat to second
        order, so a millimetre of transverse error is the same 0.4 % it was
        for the peak itself.
        """
        return self._dither_fit(0)

    @property
    def falloff(self):
        """(16,) the exponent the dither measured, nan where unmeasured."""
        return self._dither_fit(1)

    @property
    def standoff_sigma_mm(self):
        """(16,) how well the dither pinned each standoff down."""
        return self._dither_fit(3)


def _replace_via_temp(path, mode, write):
    """Write `path` through a temporary file and rename it into place.

    The rename is the whole point: a half-written file never appears under the
    real name, so an interrupted write leaves the previous contents intact
    instead of a truncated replacement. Used by MagnetRun.save, which is
    called after every pose and therefore has something to lose every time
    after the first.

    A file object is handed to `write` rather than a path because
    np.savez_compressed appends '.npz' to any path that does not already end
    in it -- given a temporary name it would helpfully write somewhere else.
    """
    tmp = path + ".part"
    kw = {} if "b" in mode else {"encoding": "utf-8"}
    try:
        with open(tmp, mode, **kw) as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


class MagnetRun:
    """A complete guided run: N_POSES sweeps of the same magnet.

    Poses may be added as they are recorded; every method works on however
    many are present, so a run abandoned after two poses still reports what
    those two established.
    """

    def __init__(self, sweeps=(), magnet_note="", axis="y", across=None,
                 normal=None, ranges_mt=None):
        self.sweeps = list(sweeps)
        self.magnet_note = magnet_note
        self.axis = axis
        self.across = across
        self.normal = normal
        # The full scale each sensor was on when this was measured. Recorded
        # with the run rather than looked up at analysis time for the same
        # reason a capture records its ADC range: the calibration on disk will
        # have moved on by the time anyone reads this back, and "did this run
        # clip" is a question about the day it was taken. None means an older
        # run that predates the check; range_check() is simply not run on it.
        self.ranges_mt = (None if ranges_mt is None else
                          np.asarray(ranges_mt, float).reshape(N_SENSORS))

    def add(self, sweep):
        self.sweeps.append(sweep)
        return self

    def __len__(self):
        return len(self.sweeps)

    @property
    def complete(self):
        return len(self.sweeps) >= N_POSES

    # ---- the measurement -------------------------------------------------
    def peak_table(self):
        """(nposes, 16) peak |B| per sensor per pose."""
        if not self.sweeps:
            raise ValueError("no poses recorded yet")
        return np.array([s.peaks for s in self.sweeps])

    def position_table(self):
        """(nposes, 16) where along the axis each sensor peaked."""
        return np.array([s.peak_at_mm for s in self.sweeps])

    def best_pose(self):
        """(16,) which pose each sensor answered loudest in.

        Read off the RAW peaks, before any standoff correction. The correction
        is a few per cent; the difference between the pose that has a sensor's
        face toward the magnet and the pose that has it facing away is a factor
        of many, so nothing the correction does can change this -- and taking
        it from the raw table keeps the choice of pose independent of a fit
        that might have failed.
        """
        return np.argmax(self.peak_table(), axis=0)

    def standoffs(self):
        """(16,) each sensor's measured distance to the magnet, in its best pose.

        These are comparable ACROSS poses, which is the property the correction
        needs and the reason it is computed here rather than inside a pose. The
        magnet does not move and the stage is homed, so a distance the dither
        measured in pose 1 and one it measured in pose 3 are the same quantity
        in the same frame. That is only true because the whole routine already
        insists the magnet stays clamped.
        """
        best = self.best_pose()
        out = np.full(N_SENSORS, np.nan)
        for i in range(N_SENSORS):
            d = self.sweeps[best[i]].standoff_mm
            out[i] = d[i]
        return out

    def falloffs(self):
        """(16,) the falloff exponent each sensor's dither measured."""
        best = self.best_pose()
        out = np.full(N_SENSORS, np.nan)
        for i in range(N_SENSORS):
            out[i] = self.sweeps[best[i]].falloff[i]
        return out

    def standoff_sigmas(self):
        """(16,) how well each sensor's standoff was pinned down, in mm."""
        best = self.best_pose()
        out = np.full(N_SENSORS, np.nan)
        for i in range(N_SENSORS):
            out[i] = self.sweeps[best[i]].standoff_sigma_mm[i]
        return out

    def standoff_correction(self):
        """(16,) the factor that puts every sensor at a common standoff.

        A sensor at d_i reading B_i would read B_i (d_i/d_ref)^n at the
        reference distance, so that is the factor. What is left after it is
        gain.

        The exponent is the MEDIAN of the sixteen measured ones, not each
        sensor's own. n is a property of the magnet, shared by every sensor
        looking at it; the per-sensor scatter in it is fit noise, and feeding
        that noise back in as an exponent would amplify it. The reference is
        the median standoff for the same reason it is the median everywhere
        else here -- it survives a sensor whose fit failed.

        Sensors with no usable dither get 1.0, which leaves them exactly where
        the plane sweep alone put them: implicitly assumed to be at d_ref,
        which is what the routine assumed about all sixteen before pass C
        existed. Nothing is made worse by a fit that did not converge.

        The displacements are SHRUNK toward zero by how much of their scatter
        survives the fit's own noise, and that is not statistical decoration --
        without it this method makes a good probe worse. d is read off a
        curvature, so a run on arms that are genuinely identical still returns
        sixteen slightly different distances, and correcting for differences
        that are not there injects the fit's noise into the trim: on the
        synthetic bench, a perfect probe went from 0.06 % trim error to 2.5 %.
        Weighting each displacement by var_real / (var_real + sigma_i^2) makes
        the correction fade out exactly when there is nothing to correct, and
        go through at full strength when the arms really are a millimetre
        apart. var_real is what is left of the observed spread after the
        measurement noise is taken out of it -- if that is nothing, so is the
        correction.
        """
        d = self.standoffs()
        n = self.falloffs()
        sig = self.standoff_sigmas()
        ok = np.isfinite(d) & (d > 0) & np.isfinite(n)
        corr = np.ones(N_SENSORS)
        if ok.sum() < 2:
            return corr
        d_ref = float(np.median(d[ok]))
        n_ref = float(np.median(n[ok]))

        delta = d[ok] - d_ref
        noise = np.where(np.isfinite(sig[ok]), sig[ok], np.inf)
        var_obs = float(np.mean(delta ** 2))
        var_noise = float(np.mean(np.minimum(noise, 1e6) ** 2))
        var_real = max(var_obs - var_noise, 0.0)
        weight = var_real / (var_real + noise ** 2) if var_real > 0 else \
            np.zeros_like(delta)

        corr[ok] = (1.0 + weight * delta / d_ref) ** n_ref
        return corr

    def response(self, use_dither=True):
        """(16,) each sensor's corrected peak in its best pose, and that pose.

        The best pose is the one where the sensor's face was turned toward the
        magnet. Taking each sensor there, at the top of its own plane peak, and
        scaling it to a common standoff is what makes the 16 numbers
        comparable -- same distance and same angle by measurement rather than
        by assuming the arms are identical.

        use_dither=False drops pass C and returns the bare peaks. That is the
        right call when the dither did not converge, because the standoff
        correction MULTIPLIES into the trim: a fit that failed does not degrade
        the answer a little, it replaces it. On the run of 2026-08-25 it turned
        a 1.30x spread into a 5.9x one. dither_quality() is the check for
        whether it converged; do not guess.
        """
        table = self.peak_table()
        best = self.best_pose()
        raw = table[best, np.arange(N_SENSORS)]
        if not use_dither:
            return raw, best
        return raw * self.standoff_correction(), best

    def dither_quality(self, expect_mm=None):
        """Did pass C actually measure anything? -> dict, with `usable`.

        Three things have to hold, and on a run where the standoff was entered
        wrongly none of them do:

        * enough sensors returned a finite distance at all;
        * the exponent came back near 3. It is a property of the MAGNET, not of
          the sensor, so all sixteen should agree; scatter in it is the fit
          failing, not sixteen different measurements;
        * the fitted distance and the fitted exponent are not locked together.
          Over too short a dither the two trade off almost exactly -- 41 mm
          with n=1.9 fits the same five points as 91 mm with n=5.0 -- and the
          give-away is their correlation across the sixteen sensors. Genuine
          measurements have no reason to correlate; a degenerate fit slides
          along that valley and correlates near 1.

        `expect_mm` is the standoff you believe you clamped, if you want the
        fitted distances checked against it as well.
        """
        d, n = self.standoffs(), self.falloffs()
        ok = np.isfinite(d) & np.isfinite(n)
        out = {"n_fitted": int(ok.sum()), "median_mm": float("nan"),
               "median_n": float("nan"), "corr_d_n": float("nan"),
               "usable": False, "notes": []}
        if ok.sum() < 4:
            out["notes"].append(
                f"only {int(ok.sum())} of {N_SENSORS} sensors returned a "
                f"standoff at all -- there is nothing to apply")
            return out
        out["median_mm"] = float(np.median(d[ok]))
        out["median_n"] = float(np.median(n[ok]))
        r = float(np.corrcoef(d[ok], n[ok])[0, 1]) if ok.sum() > 2 else np.nan
        out["corr_d_n"] = r
        if abs(out["median_n"] - 3.0) > DITHER_N_TOLERANCE:
            out["notes"].append(
                f"the falloff exponent came back at {out['median_n']:.2f}, "
                f"not the 3 a dipole gives. The fit is not describing the "
                f"magnet")
        if np.isfinite(r) and abs(r) > DITHER_DEGENERACY_R:
            out["notes"].append(
                f"fitted distance and fitted exponent correlate at "
                f"{r:.3f} across the sensors. They have no physical reason to "
                f"-- this is one under-determined fit sliding along a valley, "
                f"not {int(ok.sum())} measurements. The dither is too short "
                f"for the real standoff: it wants about a quarter of it")
        if expect_mm and out["median_mm"] > 0:
            off = abs(out["median_mm"] - expect_mm) / expect_mm
            if off > DITHER_STANDOFF_TOLERANCE:
                out["notes"].append(
                    f"the fits average {out['median_mm']:.0f} mm against the "
                    f"{expect_mm:g} mm entered on the panel. Whichever is "
                    f"right, the passes were sized from the wrong one")
        out["usable"] = not out["notes"]
        return out

    def peak_positions(self):
        """(16,) the axis position of each sensor's peak, in its best pose."""
        pos = self.position_table()
        best = self.best_pose()
        return pos[best, np.arange(N_SENSORS)]

    def across_positions(self):
        """(16,) the transverse position of each sensor's peak, or None.

        Where the arms actually put the chips. Subtract the per-face mean and
        what is left is the placement scatter -- the thing that would have gone
        straight into the trim as fake gain if the sweep had stayed on one
        axis.
        """
        best = self.best_pose()
        out = np.full(N_SENSORS, np.nan)
        any_found = False
        for i in range(N_SENSORS):
            col = self.sweeps[best[i]].peak_across_mm
            if col is not None:
                out[i] = col[i]
                any_found = True
        return out if any_found else None

    def peak_vectors(self):
        """(16, 3) each sensor's field vector at its own peak, chip frame, mT.

        The three components the rest of the routine deliberately throws away.
        Taken in each sensor's BEST pose -- the one that turned its face to the
        magnet -- so all sixteen are vectors of the same magnet seen from the
        same nominal place, and the only thing that can make two of them point
        differently is how the two chips are turned.

        Stored rather than consumed. This module does not solve for
        orientation; orientation_check() says how far the sixteen disagree,
        and the Earth-field roll sweep is what turns that into a matrix.
        """
        best = self.best_pose()
        out = np.full((N_SENSORS, 3), np.nan)
        for i in range(N_SENSORS):
            out[i] = self.sweeps[best[i]].peak_vector[i]
        return out

    def placement(self):
        """(16, 3) where each chip actually was, as (along, across, standoff) mm.

        The three passes each measure one component of the same displacement,
        and until now each was read off a different method and none was kept.
        Gathering them here is what makes them storable: `along` and `across`
        are stage coordinates of the peak (so their SPREAD across a face is the
        placement error, not their absolute value), and `standoff` is an
        absolute distance from the dither fit, nan where pass C did not run or
        did not converge.
        """
        out = np.full((N_SENSORS, 3), np.nan)
        out[:, 0] = self.peak_positions()
        across = self.across_positions()
        if across is not None:
            out[:, 1] = across
        out[:, 2] = self.standoffs()
        return out

    def presented_vectors(self, geom=None):
        """Every sensor's peak vector rotated into ONE common frame.

        Returns (unit vectors (16,3), roll sign, note).

        The argument is the same one the trim rests on. At its own peak each
        sensor sits at the same place relative to the magnet as every other
        sensor does at its peak -- that is what the four poses buy. So the
        field VECTOR at those sixteen points is sixteen looks at one physical
        vector, and once the pose is undone they must all agree. Whatever is
        left over is the chips being turned differently from what
        probe_geometry.json says.

        Undoing the pose is a rotation about the tube axis by 90 degrees per
        pose index, and the SIGN of that rotation is which way the operator
        turned the head, which nothing in the file records. Rather than assume
        it, both signs are tried and the one that clusters the sixteen vectors
        is taken -- with sixteen vectors voting, that choice is not a close
        call, and it is reported rather than hidden. If neither sign clusters
        them, that is itself the answer and the note says so.

        Note what this CANNOT see: a chip rotated about its own board normal
        by psi leaves a vector that points along that normal unchanged, and at
        the top of the peak the field is mostly along the normal. So the tilt
        is measured well and `chip_rot_deg` is measured weakly or not at all,
        depending on how much transverse field there is at the peak. Read
        orientation_check()'s per-sensor angles as a lower bound on the
        disagreement, never as a full orientation solve.
        """
        v = self.peak_vectors()
        best = self.best_pose()
        R = (geom or pgeom.Geometry.load_or_default()).rotations()

        def unroll(sign):
            out = np.full((N_SENSORS, 3), np.nan)
            for i in range(N_SENSORS):
                if not np.isfinite(v[i]).all():
                    continue
                th = sign * np.deg2rad(90.0 * int(best[i]))
                c, s = np.cos(th), np.sin(th)
                # About the tube axis, which is tube +Z.
                rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                w = rz @ (R[i] @ v[i])
                n = np.linalg.norm(w)
                if n > 0:
                    out[i] = w / n
            return out

        tried = {}
        for sign in (+1, -1):
            u = unroll(sign)
            ok = np.isfinite(u).all(axis=1)
            if ok.sum() < 2:
                continue
            # How tightly the unit vectors cluster: the length of their mean is
            # 1 for perfect agreement and falls toward 0 as they scatter. No
            # reference direction is needed, which matters because the magnet's
            # own direction is unknown and never enters any of this.
            tried[sign] = (u, float(np.linalg.norm(np.mean(u[ok], axis=0))))
        if not tried:
            return np.full((N_SENSORS, 3), np.nan), +1, "no usable peak vectors"
        best_sign = max(tried, key=lambda s: tried[s][1])
        best_u, best_score = tried[best_sign]
        other = tried.get(-best_sign, (None, float("nan")))[1]
        note = (f"the head was turned {'+' if best_sign > 0 else '-'}90 deg "
                f"per pose about the tube axis (chosen from the data, "
                f"clustering {best_score:.3f} against {other:.3f} the other "
                f"way)")
        if best_score < ORIENT_CLUSTER_MIN:
            note = (f"neither turn direction lines the sixteen vectors up "
                    f"(best clustering {best_score:.3f}). Either the head did "
                    f"not turn about its own axis, or the chip rotations in "
                    f"probe_geometry.json are wrong by more than a nudge")
        return best_u, best_sign, note

    def orientation_check(self, geom=None):
        """How far the sixteen chips disagree about which way the field points.

        -> dict with per-sensor angles in degrees from the common direction,
        the same broken down per pose, and notes.

        This is a RELATIVE measurement and says so: it compares the sixteen
        chips against each other, not against the lab, because the magnet's
        own direction never enters. That is exactly the question worth asking
        of a probe whose sixteen sensors are supposed to be interchangeable.

        A whole pose sitting off on its own is an indexing error -- the head
        was not turned a true quarter -- and is reported separately from a
        single chip sitting off on its own, which is that chip. The two look
        identical in a bare per-sensor list, which is why they are split here.
        """
        u, sign, note = self.presented_vectors(geom)
        best = self.best_pose()
        ok = np.isfinite(u).all(axis=1)
        out = {"sign": sign, "n": int(ok.sum()), "per_sensor_deg":
               np.full(N_SENSORS, np.nan),
               "within_pose_deg": np.full(N_SENSORS, np.nan),
               "per_pose_deg": {},
               "median_deg": float("nan"), "max_deg": float("nan"),
               "notes": [], "turn_note": note}
        if ok.sum() < 3:
            out["notes"].append(
                f"only {int(ok.sum())} sensors returned a usable peak vector, "
                f"which is not enough to compare orientations")
            return out

        ref = _robust_direction(u[ok])
        if ref is None:
            out["notes"].append("the peak vectors cancel rather than agreeing, "
                                "which is not a probe pointing anywhere")
            return out
        ang = np.degrees(np.arccos(np.clip(u[ok] @ ref, -1.0, 1.0)))
        out["per_sensor_deg"][ok] = ang
        out["median_deg"] = float(np.median(ang))
        out["max_deg"] = float(np.max(ang))

        # Per pose: the mean direction of one face's four sensors against the
        # same reference. Four chips cannot all be turned the same way by
        # accident, so a whole face off together is the index, not the chips.
        #
        # And per sensor WITHIN its own pose, which is the degeneracy-free
        # number: those four sensors were measured with the head in one
        # position, so no un-rolling was involved in comparing them and the
        # index error cancels exactly. It is the right test for "is this one
        # chip mounted wrong", where the global angle is not -- the global
        # angle carries its whole face's indexing error along with it, and
        # subtracting one from the other is not how angles in three dimensions
        # behave.
        for p in sorted({int(b) for b in best}):
            sel = ok & (best == p)
            if sel.sum() < 2:
                continue
            m = _robust_direction(u[sel])
            if m is None:
                continue
            out["per_pose_deg"][p] = float(np.degrees(
                np.arccos(np.clip(float(m @ ref), -1.0, 1.0))))
            out["within_pose_deg"][sel] = np.degrees(np.arccos(
                np.clip(u[sel] @ m, -1.0, 1.0)))

        worst_pose = max(out["per_pose_deg"].items(),
                         key=lambda kv: kv[1], default=None)
        if worst_pose and worst_pose[1] > ORIENT_POSE_WARN_DEG:
            out["notes"].append(
                f"every sensor of {POSE_NAMES[worst_pose[0]]} points "
                f"{worst_pose[1]:.1f} deg away from the other twelve. Four "
                f"chips do not turn together by accident -- that is the head "
                f"not having been indexed a true quarter turn for that pose, "
                f"and it does not affect the trim")
        odd = [f"S{i + 1} ({out['within_pose_deg'][i]:.1f} deg)"
               for i in range(N_SENSORS)
               if np.isfinite(out["within_pose_deg"][i])
               and out["within_pose_deg"][i] > ORIENT_SENSOR_WARN_DEG]
        if odd:
            out["notes"].append(
                f"these chips point somewhere the other three on their own "
                f"face do not: {', '.join(odd)}. Those four were measured in "
                f"one head position, so no indexing error is involved -- "
                f"either they are mounted turned from what "
                f"probe_geometry.json says, or their axis_signs are wrong. "
                f"The Earth-field roll sweep is what settles which")
        return out

    def range_check(self, ranges_mt, headroom=RANGE_WARN_FRACTION):
        """Did any channel run out of range? -> dict, with `usable`.

        The sensor's analogue output clips at its configured full scale, and a
        clipped peak does not look like an error -- it looks like a slightly
        smaller peak, taken at a flat top, on a sensor that will then be
        trimmed UP to compensate. That is the one failure in this routine that
        makes the calibration worse in a way no later check can see.

        Checked per AXIS, not on |B|. The chip has three independent outputs
        and each clips on its own, so a sensor can be well inside range on
        magnitude while one of its three components is on the rail -- which is
        the case that produces a plausible |B| out of nonsense.

        The dither is included, and it is usually what trips this: pass C
        deliberately visits points CLOSER to the magnet than the peak, and at
        a dither of a quarter of the standoff the near end sees about 2.4x the
        peak field. A run sized against the peak alone can still clip there.

        The fix is distance, and the report says how much: at 1/r^n, backing
        off by (worst/target)^(1/n) brings the whole run inside. Moving the
        magnet is better than raising the range, because raising the range
        costs resolution everywhere and this only needs a few millimetres.
        """
        fs = np.asarray(ranges_mt, float).reshape(N_SENSORS)
        worst = np.zeros(N_SENSORS)
        for s in self.sweeps:
            blocks = [s.b_mt] + [d.b_mt for d in s.dithers]
            for b in blocks:
                if len(b):
                    worst = np.maximum(worst, np.nanmax(np.abs(b), axis=(0, 2)))
        frac = np.divide(worst, fs, out=np.zeros(N_SENSORS), where=fs > 0)
        out = {"worst_mt": worst, "full_scale_mt": fs, "fraction": frac,
               "over": [], "near": [], "usable": True, "notes": [],
               "back_off_factor": 1.0}
        for i in range(N_SENSORS):
            if frac[i] >= 1.0:
                out["over"].append(i + 1)
            elif frac[i] >= headroom:
                out["near"].append(i + 1)
        if not out["over"] and not out["near"]:
            return out

        n = self.falloffs()
        n_ref = float(np.nanmedian(n)) if np.isfinite(n).any() else 3.0
        n_ref = min(max(n_ref, FALLOFF_BAND[0]), FALLOFF_BAND[1])
        target = headroom * RANGE_TARGET_OF_WARN
        out["back_off_factor"] = float((frac.max() / target) ** (1.0 / n_ref))
        d = self.standoffs()
        d_ref = float(np.nanmedian(d)) if np.isfinite(d).any() else float("nan")
        move = ("" if not np.isfinite(d_ref) else
                f" -- from the {d_ref:.0f} mm this run measured, that is "
                f"about {d_ref * out['back_off_factor']:.0f} mm")
        if out["over"]:
            out["usable"] = False
            out["notes"].append(
                f"{len(out['over'])} sensors reached full scale: "
                f"{', '.join('S' + str(s) for s in out['over'])}, worst "
                f"{100 * frac.max():.0f} % of +/-{fs.max():g} mT. A clipped "
                f"peak reads LOW and flat, so those sensors would be trimmed "
                f"up to compensate and the error would be baked in. Move the "
                f"magnet {out['back_off_factor']:.2f}x further away{move}, or "
                f"use a weaker one, and run again")
        elif out["near"]:
            out["notes"].append(
                f"{', '.join('S' + str(s) for s in out['near'])} came within "
                f"{100 * (1 - headroom):.0f} % of full scale (worst "
                f"{100 * frac.max():.0f} %). Nothing clipped, but there is no "
                f"headroom left for a stronger pose or a longer dither -- "
                f"{out['back_off_factor']:.2f}x further away would restore it"
                f"{move}")
        return out

    def trim(self, use_dither=True):
        """(16,) the multiplicative gain trim this run implies, 1.0 = neutral.

        Returned rather than applied: what to do with it is the caller's
        decision, and Calibration.cross_calibrate() is the thing that owns how
        a trim is folded in.
        """
        resp, _ = self.response(use_dither=use_dither)
        ok = resp > DEAD_FRACTION * np.median(resp)
        ref = np.median(resp[ok]) if ok.any() else 1.0
        return np.where(ok & (resp > 0), ref / np.where(resp > 0, resp, 1.0), 1.0)

    def quiet_sensors(self, use_dither=True):
        """Sensors that never really answered, as S-labels."""
        resp, _ = self.response(use_dither=use_dither)
        thresh = DEAD_FRACTION * np.median(resp)
        return [f"S{i + 1}" for i in range(N_SENSORS) if resp[i] <= thresh]

    # ---- what it says about the geometry --------------------------------
    def slot_groups(self):
        """[(axis position, [sensor ids])] -- the rings, in the order swept.

        All four sensors of one ring pass the magnet at the SAME point of the
        sweep, so what a pass measures is not a sequence of 16 but a sequence
        of four rings of four. Splitting at the largest gaps rather than at a
        tolerance means this needs no idea of the plate pitch: there are
        SENSORS_PER_FACE rings, so there are that many minus one real gaps.
        """
        pos = self.peak_positions()
        order = np.argsort(pos)
        gaps = np.diff(pos[order])
        n_cuts = min(pgeom.SENSORS_PER_FACE - 1, len(gaps))
        cuts = sorted(int(c) + 1 for c in np.argsort(gaps)[len(gaps) - n_cuts:])
        return [(float(np.mean(pos[grp])), sorted(int(i) + 1 for i in grp))
                for grp in np.split(order, cuts)]

    def order_along_axis(self):
        """Sensor ids (1-based) sorted by where they passed the magnet."""
        return [int(i) + 1 for i in np.argsort(self.peak_positions())]

    def face_groups(self):
        """pose index -> the sensor ids that were loudest in that pose."""
        _, best = self.response()
        out = {}
        for i in range(N_SENSORS):
            out.setdefault(int(best[i]), []).append(i + 1)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def measured_slots(self):
        """sensor id -> slot along the tube, as this run actually found them.

        Which end of the tube the sweep meets first is NOT a convention to be
        chosen here; it follows from how the probe is mounted. A sensor at a
        larger tube coordinate sits further along the rig axis when the tube's
        +Z maps to +axis (probe_geometry.MOUNT_ROT), and the head is driven in
        the +ve direction, so that sensor reaches a FIXED magnet at a SMALLER
        stage position -- it passes earlier. Reverse the mounting and the rule
        reverses with it, which is why it is read off MOUNT_ROT rather than
        written down.

        On this rig that makes the first ring past the magnet the tip end, and
        the tip is where S16 is -- the one thing about this probe's layout that
        was established independently, by looking at it.
        """
        try:
            axis_i = {"x": 0, "y": 1, "z": 2}[self.axis]
        except KeyError:
            raise ValueError(f"unknown sweep axis {self.axis!r}") from None
        along = float(pgeom.to_world(pgeom.TUBE_AXIS)[axis_i])
        if abs(along) < 0.5:
            raise ValueError(
                f"the tube axis is not along {self.axis}: sweeping across the "
                f"probe cannot say which sensor is at which end of it, and the "
                f"equal-approach argument does not hold either")
        groups = self.slot_groups()
        n = len(groups)
        if n != pgeom.SENSORS_PER_FACE:
            raise ValueError(
                f"the sweep found {n} rings, not {pgeom.SENSORS_PER_FACE} -- "
                f"it did not carry every ring past the magnet, so the slots "
                f"cannot be assigned")
        # groups come in ascending sweep position; +along means the far end
        # arrives first, so the slots count down.
        order = list(range(n))[::-1] if along > 0 else list(range(n))
        out = {}
        for slot, (_at, ids) in zip(order, groups):
            for sid in ids:
                out[int(sid)] = int(slot)
        return out

    def face_balance(self, geom=None, use_dither=True):
        """Per-face mean response, and what opposite faces say about the setup.

        The whole routine rests on the head turning about its OWN axis, so that
        every face is presented to the magnet at the same distance. If the tube
        sits a little off-centre in its cradle, one face comes closer and the
        opposite one goes further by the same amount -- and since B falls as
        1/r^3, a sub-millimetre offset is a ten per cent error that lands in
        the trim looking exactly like gain.

        It is separable because it is ANTI-symmetric: the geometric factor
        multiplies one face by k and its opposite by 1/k, so the product of two
        opposite faces' mean responses is unchanged while each on its own is
        not. A pair whose product is 1.00 but whose members differ is telling
        you about the cradle, not about the chips.

        Returns {"means": {face: mean response},
                 "pairs": [(face, opposite, ratio, offset_fraction)],
                 "notes": [...]}
        """
        g = geom or pgeom.Geometry.load_or_default()
        # Same use_dither as the trim being judged. Run on dither-corrected
        # response when the dither failed, this check reports the fit's own
        # scatter as a head that is not concentric -- which is exactly what it
        # did on 2026-08-25: 10.2 % off-centre from a head that is within
        # 1.4 %, sending an operator to re-centre a cradle that was fine.
        resp, _ = self.response(use_dither=use_dither)
        means = {}
        for f in range(pgeom.N_FACES):
            ids = [i for i in range(N_SENSORS) if g.face(i + 1) == f]
            if ids:
                means[f] = float(np.mean(resp[ids]))

        pairs, notes = [], []
        for f in range(pgeom.N_FACES):
            opp = next((o for o in range(pgeom.N_FACES)
                        if np.allclose(pgeom.FACE_NORMALS[o],
                                       -pgeom.FACE_NORMALS[f])), None)
            if opp is None or opp < f or f not in means or opp not in means:
                continue
            ratio = means[f] / means[opp] if means[opp] else float("inf")
            # r ratio from the field ratio, then the offset as a fraction of
            # the nominal approach: (1+e)/(1-e) = (ratio)**(1/3).
            k = abs(ratio) ** (1.0 / 3.0)
            offset = (k - 1.0) / (k + 1.0)
            pairs.append((f, opp, ratio, offset))
            if abs(ratio - 1.0) > FACE_IMBALANCE_WARN:
                notes.append(
                    f"faces {pgeom.FACE_NAMES[f]} and {pgeom.FACE_NAMES[opp]} "
                    f"are opposite each other but answered "
                    f"{ratio:.2f}x differently. They see the same magnet from "
                    f"the same nominal distance, so this is the head not "
                    f"being concentric with whatever it was turned about -- "
                    f"or its faces not carrying their chips at equal radius "
                    f"-- by roughly {100 * abs(offset):.1f}% of the magnet "
                    f"distance. 1/r^3 turns that into a gain error of this "
                    f"size, so this much of the trim is geometry wearing "
                    f"gain's clothes. Re-centre and run again if the trim has "
                    f"to be gain alone")
        return {"means": means, "pairs": pairs, "notes": notes}

    def measured_faces(self):
        """sensor id -> face index, as the poses found them.

        The pose numbers are not face numbers -- pose 1 is wherever the tube
        happened to start -- so this keeps whatever face label the sensors
        already share and only reports the GROUPING. Renaming faces would
        rotate the whole tube frame for no measurement reason.
        """
        out = {}
        for pose, ids in self.face_groups().items():
            for sid in ids:
                out[int(sid)] = int(pose)
        return out

    def apply_to_geometry(self, geom):
        """Write the measured slots into `geom`. Returns what changed.

        Slots only. The face GROUPING is checked and reported, but which face
        index a group is given is a naming choice that the magnet cannot make
        -- turning the tube renames them all.
        """
        slots = self.measured_slots()
        changed = []
        for spec in geom.sensors:
            sid = int(spec["id"])
            if sid in slots and int(spec["slot"]) != slots[sid]:
                changed.append((sid, int(spec["slot"]), slots[sid]))
                spec["slot"] = slots[sid]
        if changed:
            geom.mapping = "measured"
        return changed

    def check_geometry(self, geom=None):
        """Compare the measured arrangement with probe_geometry.json.

        Returns a list of human-readable disagreements -- empty means the file
        matches what the magnet just did. Nothing is rewritten here: a mapping
        is worth changing on evidence, and the evidence should be read first.
        """
        g = geom or pgeom.Geometry.load_or_default()
        notes = []
        if not self.complete:
            notes.append(f"only {len(self)} of {N_POSES} poses recorded, so "
                         f"faces cannot be assigned -- the order along the "
                         f"tube below is still valid")

        # Sensors sharing a face should group together, whatever the face is
        # called. Compare the PARTITION, not the labels: which face index the
        # file gives them is a naming choice, four-on-a-face is a fact.
        measured = {frozenset(v) for v in self.face_groups().values()}
        expected = {}
        for sid in range(1, N_SENSORS + 1):
            expected.setdefault(g.face(sid), []).append(sid)
        expected = {frozenset(v) for v in expected.values()}
        if self.complete and measured != expected:
            notes.append(
                "the four sensors that answer together are not the four the "
                "geometry puts on a face: measured "
                + " | ".join("+".join(f"S{s}" for s in sorted(grp))
                             for grp in sorted(measured, key=sorted))
                + "  vs file "
                + " | ".join("+".join(f"S{s}" for s in sorted(grp))
                             for grp in sorted(expected, key=sorted)))

        # Which ring is which is independent of which face is which. The sweep
        # may run either way past the magnet, so the slots are expected to be
        # monotonic, not ascending: a descending run is a direction, not a
        # fault, and flagging it would train the operator to ignore this.
        groups = self.slot_groups()
        sizes = [len(ids) for _, ids in groups]
        if sizes != [pgeom.SENSORS_PER_FACE] * len(groups):
            notes.append(
                f"the sensors do not pass the magnet in rings of "
                f"{pgeom.SENSORS_PER_FACE}: got groups of {sizes}. Either a "
                f"sensor is not where the file says, or the sweep did not "
                f"carry every ring past the magnet")
        slots = []
        for at_mm, ids in groups:
            found = {g.slot(s) for s in ids}
            if len(found) > 1:
                notes.append(
                    f"sensors {', '.join('S' + str(s) for s in ids)} passed "
                    f"the magnet together at {at_mm:.1f} mm, so they share a "
                    f"ring, but the file puts them on slots "
                    f"{sorted(found)}")
            slots.append(min(found))
        if len(slots) > 1 and slots != sorted(slots) and slots != sorted(
                slots, reverse=True):
            notes.append(f"the rings passed the magnet in slot order {slots}, "
                         f"which is not a straight run along the tube")
        return notes

    # ---- reporting -------------------------------------------------------
    def _placement_lines(self, geom=None):
        """What the plane and the dither said about where the arms really are.

        Worth printing even when the numbers are boring, because "the arms are
        within a tenth of a millimetre of each other" is a result: it says the
        trim above is gain, and it is the only evidence for that claim.
        """
        lines = []
        across = self.across_positions()
        stand = self.standoffs()
        fall = self.falloffs()

        if across is not None and np.isfinite(across).sum() > 1:
            spread = float(np.nanmax(across) - np.nanmin(across))
            lines.append(
                f"transverse peaks span {spread:.2f} mm across the 16 -- this "
                f"is arm placement, and the plane sweep is what stopped it "
                f"becoming gain")
            if np.isfinite(stand).any():
                d_ref = float(np.nanmedian(stand))
                if d_ref > 0:
                    lines.append(
                        f"    a one-axis sweep down the middle would have "
                        f"read the outliers low by up to "
                        f"{100 * 1.5 * (spread / 2.0) ** 2 / d_ref ** 2:.1f}%")

        if np.isfinite(stand).any():
            n_ok = int(np.isfinite(stand).sum())
            lines.append(
                f"standoffs measured for {n_ok} of {N_SENSORS} sensors: "
                f"{np.nanmin(stand):.1f} to {np.nanmax(stand):.1f} mm, median "
                f"{np.nanmedian(stand):.1f} mm")
            if np.isfinite(fall).any():
                n_med = float(np.nanmedian(fall))
                lines.append(
                    f"the dither measured the field falling as 1/r^{n_med:.2f}"
                    + ("  (a dipole is 3; near a long magnet 2 is normal, so "
                       "this is a sanity check on the fit, not a fault)"
                       if 1.5 <= n_med <= 3.6 else
                       "  -- that is not a magnet seen from a sensible "
                       "distance. Treat the standoff column, and the "
                       "correction that came from it, as unreliable"))
            corr = self.standoff_correction()
            worst = float(np.max(np.abs(corr - 1.0)))
            lines.append(
                f"largest standoff correction applied: {100 * worst:.1f}% "
                f"(this much of the old trim was distance, not gain)")
            # The report is written after every pose and read long afterwards,
            # so it must not present a degenerate fit as a measurement. The
            # trim columns above are the A+B+C view whatever the fit did; this
            # is the line that says whether to believe them.
            q = self.dither_quality()
            if not q["usable"]:
                lines.append(
                    f"  BUT pass C did not converge -- corr(d,n) "
                    f"{q['corr_d_n']:+.3f}, median n {q['median_n']:.2f}, "
                    f"median distance {q['median_mm']:.0f} mm. The standoff "
                    f"column and that correction are a fit sliding along a "
                    f"valley, not sixteen measurements, and the trim to "
                    f"believe is the A+B one:")
                for note in q["notes"]:
                    lines.append(f"    - {note}")
        elif self.sweeps and not any(s.dithers for s in self.sweeps):
            lines.append(
                "no standoff dither in this run, so each chip's distance from "
                "the magnet is still assumed rather than measured -- a "
                "millimetre of that is ~15% of trim at a 20 mm standoff")
        lines.append("")
        return lines

    def _orientation_lines(self, geom=None):
        """What the peak VECTORS say about how the chips are turned.

        Printed even when it is boring, and especially then: "the sixteen
        agree to a fraction of a degree" is the evidence that the rotation
        matrices in probe_geometry.json are right, and until this existed
        there was none either way.
        """
        o = self.orientation_check(geom)
        if o["n"] < 3:
            return []
        lines = ["",
                 f"orientation, from the peak vectors ({o['n']} of "
                 f"{N_SENSORS} sensors, relative to each other -- the "
                 f"magnet's own direction never enters):",
                 f"  the sixteen agree to {o['median_deg']:.2f} deg median, "
                 f"{o['max_deg']:.2f} deg worst",
                 f"  within a face, where the index cannot reach: "
                 f"{np.nanmedian(o['within_pose_deg']):.2f} deg median, "
                 f"{np.nanmax(o['within_pose_deg']):.2f} deg worst",
                 f"  {o['turn_note']}"]
        if o["per_pose_deg"]:
            lines.append("  per pose (a whole face off together is the index, "
                         "not the chips):")
            for p, deg in sorted(o["per_pose_deg"].items()):
                lines.append(
                    f"    {POSE_NAMES[p] if p < len(POSE_NAMES) else p}: "
                    f"{deg:5.2f} deg")
        lines.append("  this measures TILT well and rotation about each "
                     "board's own normal weakly -- the roll sweep is what "
                     "settles chip_rot_deg")
        return lines

    def report(self, geom=None):
        resp, best = self.response()
        pos = self.peak_positions()
        trim = self.trim()
        across = self.across_positions()
        stand = self.standoffs()
        corr = self.standoff_correction()
        raw = self.peak_table()[best, np.arange(N_SENSORS)]

        head = (f"{'sensor':>7} {'pose':>5} {self.axis + ' peak [mm]':>13}")
        if across is not None:
            head += f" {(self.across or 'across') + ' peak [mm]':>16}"
        head += f" {'|B| peak [mT]':>14}"
        if np.isfinite(stand).any():
            head += f" {'standoff [mm]':>14} {'d corr':>7}"
        head += f" {'trim':>7}"

        passes = ["axial"]
        if any(s.is_plane for s in self.sweeps):
            passes.append("plane")
        if any(s.dithers for s in self.sweeps):
            passes.append("standoff dither")
        lines = [f"guided magnet calibration -- {len(self)} of {N_POSES} poses",
                 f"magnet: {self.magnet_note or 'position not noted'}",
                 f"passes: {', '.join(passes)}",
                 "",
                 head]
        for i in range(N_SENSORS):
            row = (f"{'S' + str(i + 1):>7} {best[i] + 1:>5} {pos[i]:13.2f}")
            if across is not None:
                row += f" {across[i]:16.2f}"
            row += f" {raw[i]:14.4f}"
            if np.isfinite(stand).any():
                row += (f" {stand[i]:14.2f}" if np.isfinite(stand[i])
                        else f" {'--':>14}")
                row += f" {corr[i]:7.3f}"
            row += f" {trim[i]:7.3f}"
            lines.append(row)
        spread = (resp.max() / resp.min()) if resp.min() > 0 else float("inf")
        balanced = not self.face_balance(geom)["notes"]
        lines += ["",
                  f"peak spread across the 16: {spread:.2f}x  "
                  + ("(this is gain only -- every sensor was measured at the "
                     "same approach)" if balanced else
                     "(NOT gain alone -- see the opposite-face check below)")]
        lines += self._placement_lines(geom)
        lines.append("rings, in the order they passed the magnet:")
        for at_mm, ids in self.slot_groups():
            lines.append(f"  {self.axis} = {at_mm:7.1f} mm   "
                         + " ".join(f"S{s}" for s in ids))
        for pose, ids in self.face_groups().items():
            lines.append(f"  {POSE_NAMES[pose] if pose < len(POSE_NAMES) else pose}"
                         f" answered: {', '.join('S' + str(s) for s in ids)}")
        bal = self.face_balance(geom)
        if bal["pairs"]:
            lines.append("")
            lines.append("opposite faces (equal unless the head is off-centre):")
            for f, opp, ratio, offset in bal["pairs"]:
                lines.append(
                    f"  {pgeom.FACE_NAMES[f]} vs {pgeom.FACE_NAMES[opp]}: "
                    f"{bal['means'][f]:7.3f} vs {bal['means'][opp]:7.3f} mT  "
                    f"({ratio:.3f}x, ~{100 * abs(offset):.1f}% off-centre)")
        quiet = self.quiet_sensors()
        if quiet:
            lines.append(f"no usable response, left untrimmed: {', '.join(quiet)}")
        lines += self._orientation_lines(geom)
        rng = self.range_check(self.ranges_mt) if self.ranges_mt is not None \
            else None
        if rng is not None and (rng["over"] or rng["near"]):
            lines.append("")
            lines.append(f"range: worst channel reached "
                         f"{100 * rng['fraction'].max():.0f} % of full scale")
        notes = self.check_geometry(geom) + self.face_balance(geom)["notes"]
        if rng is not None:
            notes = rng["notes"] + notes
        notes += self.orientation_check(geom)["notes"]
        lines.append("")
        lines += (["nothing to report: the file agrees with what the magnet "
                   "did, and the head was concentric"] if not notes
                  else ["worth reading before trusting the trim:"]
                       + [f"  - {n}" for n in notes])
        return "\n".join(lines)

    # ---- persistence -----------------------------------------------------
    def save(self, path):
        base = os.path.splitext(path)[0]
        d = os.path.dirname(os.path.abspath(base))
        if d:
            os.makedirs(d, exist_ok=True)
        # Per-pose keys rather than one stacked array: the passes make the
        # point count depend on how many rings pass A found, so two poses of
        # the same run can legitimately differ in length. Stacking them would
        # raise on a run that is worth keeping.
        arrays, poses = {}, []
        for k, s in enumerate(self.sweeps):
            arrays[f"pos_{k}"] = s.pos_mm
            arrays[f"b_{k}"] = np.asarray(s.b_mt, dtype=np.float32)
            for j, d in enumerate(s.dithers):
                arrays[f"dz_{k}_{j}"] = d.z_mm
                arrays[f"db_{k}_{j}"] = np.asarray(d.b_mt, dtype=np.float32)
            poses.append({"pose": s.pose, "axes": list(s.axes),
                          "note": s.note,
                          "n_dithers": len(s.dithers),
                          "dither_at_mm": [d.at_mm for d in s.dithers]})
        # Written to one side and renamed into place, because this is now
        # called after every pose rather than once at the end: a crash during
        # the write would otherwise take out the three good poses already on
        # disk along with the fourth, which is the exact loss saving early was
        # meant to prevent. os.replace is atomic on both platforms this runs
        # on, so a reader sees the old pair or the new one.
        #
        # Arrays first, then the sidecar. load() takes its pose list from the
        # JSON and the data from the NPZ, so a crash between the two renames
        # leaves an NPZ holding one pose more than the JSON admits to -- which
        # reads back cleanly as the earlier run. The other order would leave a
        # JSON promising a pose whose arrays are not there, and that raises.
        _replace_via_temp(base + ".npz", "wb",
                          lambda fh: np.savez_compressed(fh, **arrays))
        _replace_via_temp(
            base + ".json", "w",
            lambda fh: (json.dump(
                {"magnet_note": self.magnet_note, "axis": self.axis,
                 "across": self.across, "normal": self.normal,
                 "ranges_mt": (None if self.ranges_mt is None
                               else self.ranges_mt.tolist()),
                 "n_poses": len(self.sweeps), "poses": poses,
                 "sensors": self._sensor_records(),
                 "report": self.report()}, fh, indent=2), fh.write("\n")))
        return base + ".npz"

    def _sensor_records(self):
        """Per-sensor measurements, for the sidecar -- what the run FOUND.

        The arrays in the .npz are the evidence and this is the finding, and
        the finding is worth writing down separately: reading a peak vector
        back out of the raw sweep means knowing which pose was best and which
        row was the peak, which is the whole of best_pose() and
        peak_response(). Anything downstream that wants "where is S7 and which
        way is it turned" should be able to open one JSON file and read it.

        Kept to what was MEASURED. No trim, no correction -- those depend on
        choices (use_dither, which reference) that belong to whoever applies
        the run, not to the run.
        """
        if not self.sweeps:
            return []
        best = self.best_pose()
        place = self.placement()
        vec = self.peak_vectors()
        peaks = self.peak_table()[best, np.arange(N_SENSORS)]
        fall = self.falloffs()
        sig = self.standoff_sigmas()

        def num(x):
            return None if not np.isfinite(x) else round(float(x), 6)

        out = []
        for i in range(N_SENSORS):
            out.append({
                "id": i + 1,
                "pose": int(best[i]),
                "peak_mt": num(peaks[i]),
                f"{self.axis}_mm": num(place[i, 0]),
                f"{self.across or 'across'}_mm": num(place[i, 1]),
                "standoff_mm": num(place[i, 2]),
                "standoff_sigma_mm": num(sig[i]),
                "falloff_n": num(fall[i]),
                "peak_vector_mt": [num(c) for c in vec[i]],
            })
        return out

    @classmethod
    def load(cls, path):
        base = os.path.splitext(path)[0]
        side = {}
        if os.path.exists(base + ".json"):
            with open(base + ".json", encoding="utf-8") as fh:
                side = json.load(fh)
        axis = side.get("axis", "y")
        across = side.get("across")
        normal = side.get("normal")
        with np.load(base + ".npz") as z:
            if "pos_mm" in z:
                # Runs saved before the plane sweep existed: one axis, one
                # stacked array, no dithers. They still analyse -- every
                # correction added since degrades to the identity.
                sweeps = [PoseSweep(int(p), pos, b, axis=axis)
                          for p, pos, b in zip(z["pose"], z["pos_mm"],
                                               z["b_mt"])]
            else:
                sweeps = []
                for k, spec in enumerate(side.get("poses", [])):
                    at = spec.get("dither_at_mm") or []
                    dithers = [
                        Dither(z[f"dz_{k}_{j}"], z[f"db_{k}_{j}"],
                               at_mm=at[j] if j < len(at) else None)
                        for j in range(int(spec.get("n_dithers", 0)))]
                    sweeps.append(PoseSweep(
                        int(spec.get("pose", k)), z[f"pos_{k}"], z[f"b_{k}"],
                        axis=axis, across=across, normal=normal,
                        axes=tuple(spec.get("axes") or (axis,)),
                        dithers=dithers, note=spec.get("note", "")))
        return cls(sweeps, magnet_note=side.get("magnet_note", ""), axis=axis,
                   across=across, normal=normal,
                   ranges_mt=side.get("ranges_mt"))


# The standardized run: what the wizard opens with, so that two runs a month
# apart are the same measurement and their trims can be compared rather than
# just believed one at a time. The suggested_* helpers below still derive a
# sizing from the geometry and the standoff, and they are still what answers
# when an operator moves the standoff off the standard -- but a value derived
# from a geometry file is not a constant, and the point of a standard is that
# it does not move when probe_geometry.json does.
#
# Where this differs from what the helpers derive, and what it costs:
#
#   sweep 140 mm, not the derived 179.  At the 33 mm plate pitch the four
#     rings span 99 mm, so this is 20.5 mm of lead-in at each end instead of
#     40. The outermost peak is about one standoff wide, so at a 20 mm
#     standoff it still closes inside the sweep -- but only just. Raise this
#     if the standoff goes up, or the end rings peak at the edge of the data.
#   step 3.5 mm, not 8.25.  Finer than pass A needs (it only has to say WHERE
#     each ring is), and the reason the locate pass costs 41 points.
#   cut half-span 10 mm at a 20 mm standoff, not 20.  Half the span the
#     physics argument in suggested_plane() asks for, at the same 21 points --
#     the same cost spent closer in around the peak. Its own docstring puts
#     +-10 mm inside what works at a 10-20 mm standoff; it is NOT enough at 40,
#     so this pairs with the 20 mm standoff and does not survive without it.
#   dither 5 points, not the DITHER_POINTS=7 the bench measured as best. By
#     the table at DITHER_FRACTION this is 4.2 % residual trim error against
#     2.1 % -- the standard buys 8 moves per pose with half the standoff
#     accuracy. DITHER_POINTS stays 7 because that is still what the
#     measurement says; this is the operational choice, and the two are
#     deliberately separate numbers.
#   2.0 s per point, not 1.0.  The dither reads a curvature out of 5 points
#     and is where averaging actually pays; with 5 rather than 7 points it
#     pays more.
#
# 145 points a pose, about 10.9 min each, 44 min for all four.
STANDARD_RUN = {
    "sweep_mm": 140.0,
    "step_mm": 3.5,
    "seconds_per_point": 2.0,
    "standoff_mm": 20.0,
    "cut_half_span_mm": 10.0,
    "cut_step_mm": 1.0,
    "dither_half_span_mm": 5.0,
    "dither_points": 5,
}


def suggested_sweep(geom=None, clearance_mm=40.0):
    """(span_mm, step_mm) for a sweep that carries every ring past the magnet.

    A span rather than a start and a stop: where the sweep begins depends on
    where the magnet was clamped, which only the operator knows. Park the head
    with the magnet just clear of the first ring and run this far.

    The span is the distance from the first sensor to the last, plus enough
    lead-in at each end for the outermost sensor's peak to be a peak rather
    than the edge of the data. Step is a quarter of the plate pitch, which puts
    about four samples across each ring's peak -- enough to place it without
    paying for a fine scan, since the peak VALUE is what the trim uses and it
    is flat to second order at the top.
    """
    g = geom or pgeom.Geometry.load_or_default()
    span = (pgeom.SENSORS_PER_FACE - 1) * g.plate_pitch_mm + 2 * clearance_mm
    return span, g.plate_pitch_mm / 4.0


def suggested_plane(standoff_mm=20.0):
    """(half_span_mm, step_mm) for the transverse cut at each ring.

    HALF-SPAN scales with the standoff, and that is not a detail. The peak this
    pass is looking for has a width set by the distance to the magnet -- at
    d mm away, |B| is still 72 % of its maximum d mm off to the side. Sweep
    much less than the standoff and the cut is nearly flat, so the peak
    position is poorly determined and the whole point is lost; sweep much more
    and the extra points are spent out where there is no signal. A half-span of
    about one standoff puts the ends of the cut at roughly 70 % of the peak,
    which places the maximum well and wastes nothing.

    This is the answer to "is +-10 mm enough?": it is, at a 10-20 mm standoff,
    and it is not at 40 mm. Size it from the distance, not from a round number.

    STEP is a tenth of the half-span, ~21 points. The peak is quadratic at the
    top, so its position is found to far better than one step; this is about
    having enough points either side of it to fit rather than about resolution.
    """
    half = max(4.0, float(standoff_mm))
    return half, half / 10.0


def suggested_dither(standoff_mm=20.0):
    """(offsets_mm,) along the standoff axis, centred on zero.

    See DITHER_FRACTION for why this is a quarter of the standoff rather than
    the fraction of a millimetre that instinct suggests: what is being measured
    is the curvature, and curvature is second order.
    """
    half = max(1.0, DITHER_FRACTION * float(standoff_mm))
    return np.linspace(-half, half, DITHER_POINTS)


def ring_positions(sweep, n_rings=pgeom.SENSORS_PER_FACE):
    """Where pass A found this pose's rings, for pass B to cut across.

    Only the sensors on the face turned toward the magnet answer in a given
    pose, so this takes the loudest n_rings and reports where each of them
    peaked, rather than trying to group all sixteen. The twelve on the other
    faces are metres of field away in 1/r^3 terms and their "peaks" are noise.
    """
    peaks = sweep.peaks
    loud = np.argsort(peaks)[-int(n_rings):]
    return np.sort(sweep.peak_at_mm[loud])


# --------------------------------------------------------------------------
# re-deriving a trim from a run already on disk
# --------------------------------------------------------------------------

def apply_run(run, cal, use_dither=True, replacing=None, geom=None):
    """Fold a saved run's trim into `cal`. Returns a list of report lines.

    Every capture is converted through the calibration that was live when it
    was taken, so a run's peaks already carry whatever trim was in force. That
    is why Calibration.cross_calibrate MULTIPLIES rather than assigns, and it
    is also why re-deriving from a run you have already applied needs
    `replacing`: hand it the run whose trim is currently folded in and that
    factor is divided back out first, putting the calibration into the state
    it was in when the measurement was made. Without it the same run gets
    counted twice and every sensor is trimmed by the square of its factor.

    `replacing` is usually the same run -- that is the shape of "apply this
    one again, differently", which is what a failed pass C leaves you needing.
    """
    lines = []
    if replacing is not None:
        undo = replacing.trim()
        cal.gain_corr = cal.gain_corr / undo[:, None]
        lines.append(f"divided out the trim already applied from that run "
                     f"({undo.min():.3f}..{undo.max():.3f}, "
                     f"{undo.max() / undo.min():.2f}x spread)")
    resp, _best = run.response(use_dither=use_dither)
    _corr, skipped = cal.cross_calibrate(resp)
    t = run.trim(use_dither=use_dither)
    lines.append(f"applied the {'A+B+C' if use_dither else 'A+B'} trim "
                 f"({t.min():.3f}..{t.max():.3f}, "
                 f"{t.max() / t.min():.2f}x spread)")
    if skipped:
        lines.append(f"kept the previous trim for {', '.join(skipped)}")
    bal = run.face_balance(geom, use_dither=use_dither)
    for note in bal["notes"]:
        lines.append(note)
    return lines


def _cmd_apply(a):
    run = MagnetRun.load(a.run)
    q = run.dither_quality(expect_mm=a.standoff)
    print(f"{os.path.basename(a.run)}: {len(run)} poses, "
          f"axis={run.axis} across={run.across} normal={run.normal}")
    print(f"  pass C: {q['n_fitted']}/{N_SENSORS} fitted, median "
          f"{q['median_mm']:.1f} mm, median n {q['median_n']:.2f}, "
          f"corr(d,n) {q['corr_d_n']:+.3f} -- "
          f"{'usable' if q['usable'] else 'NOT usable'}")
    for note in q["notes"]:
        print(f"    - {note}")

    use_dither = not a.no_dither
    if use_dither and not q["usable"]:
        print("  refusing to apply pass C. Re-run with --no-dither to take "
              "the plane answer, or fix the standoff and measure again.")
        return 1

    cal = ocal.Calibration.load(a.calibration)
    before = cal.gain_corr[:, 0].copy()
    prev = MagnetRun.load(a.replacing) if a.replacing else None
    for line in apply_run(run, cal, use_dither=use_dither, replacing=prev):
        print(f"  {line}")
    after = cal.gain_corr[:, 0]
    print(f"  gain {before.min():.3f}..{before.max():.3f} "
          f"({before.max() / before.min():.2f}x) -> "
          f"{after.min():.3f}..{after.max():.3f} "
          f"({after.max() / after.min():.2f}x)")
    if a.dry_run:
        print("  --dry-run: nothing written")
        return 0
    cal.notes = (
        f"gain trim re-derived from {os.path.basename(a.run)} using passes "
        f"{'A+B+C' if use_dither else 'A+B (pass C dropped)'}"
        + ("" if use_dither else
           f"; the dither fit was degenerate -- corr(d,n) {q['corr_d_n']:+.3f}, "
           f"median n {q['median_n']:.2f}, median distance "
           f"{q['median_mm']:.0f} mm"))
    path = cal.save(a.calibration)
    print(f"  wrote {path}")
    if cal.archived_to:
        print(f"  archived as {os.path.basename(cal.archived_to)}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Re-derive a gain trim from a guided magnet run on disk.")
    sub = p.add_subparsers(dest="cmd", required=True)
    ap = sub.add_parser("apply", help="fold a saved run's trim into a calibration")
    ap.add_argument("run", help="captures/magcal_*.npz")
    ap.add_argument("--calibration", default=None,
                    help="calibration.json to update (default: the config one)")
    ap.add_argument("--no-dither", action="store_true",
                    help="ignore pass C and trim on the plane peaks alone")
    ap.add_argument("--replacing", default=None, metavar="RUN.npz",
                    help="a run whose trim is already folded into that "
                         "calibration; its factor is divided out first. Pass "
                         "the same run to re-apply it differently.")
    ap.add_argument("--standoff", type=float, default=None,
                    help="the standoff you believe you clamped, in mm, to "
                         "check the dither fits against")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and write nothing")
    ap.set_defaults(func=_cmd_apply)
    a = p.parse_args(argv)
    if a.calibration is None:
        a.calibration = paths.config(ocal.CONFIG_NAME)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
