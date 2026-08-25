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

import json
import os

import numpy as np

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


class MagnetRun:
    """A complete guided run: N_POSES sweeps of the same magnet.

    Poses may be added as they are recorded; every method works on however
    many are present, so a run abandoned after two poses still reports what
    those two established.
    """

    def __init__(self, sweeps=(), magnet_note="", axis="y", across=None,
                 normal=None):
        self.sweeps = list(sweeps)
        self.magnet_note = magnet_note
        self.axis = axis
        self.across = across
        self.normal = normal

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

    def response(self):
        """(16,) each sensor's corrected peak in its best pose, and that pose.

        The best pose is the one where the sensor's face was turned toward the
        magnet. Taking each sensor there, at the top of its own plane peak, and
        scaling it to a common standoff is what makes the 16 numbers
        comparable -- same distance and same angle by measurement rather than
        by assuming the arms are identical.
        """
        table = self.peak_table()
        best = self.best_pose()
        raw = table[best, np.arange(N_SENSORS)]
        return raw * self.standoff_correction(), best

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

    def trim(self):
        """(16,) the multiplicative gain trim this run implies, 1.0 = neutral.

        Returned rather than applied: what to do with it is the caller's
        decision, and Calibration.cross_calibrate() is the thing that owns how
        a trim is folded in.
        """
        resp, _ = self.response()
        ok = resp > DEAD_FRACTION * np.median(resp)
        ref = np.median(resp[ok]) if ok.any() else 1.0
        return np.where(ok & (resp > 0), ref / np.where(resp > 0, resp, 1.0), 1.0)

    def quiet_sensors(self):
        """Sensors that never really answered, as S-labels."""
        resp, _ = self.response()
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

    def face_balance(self, geom=None):
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
        resp, _ = self.response()
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
        elif self.sweeps and not any(s.dithers for s in self.sweeps):
            lines.append(
                "no standoff dither in this run, so each chip's distance from "
                "the magnet is still assumed rather than measured -- a "
                "millimetre of that is ~15% of trim at a 20 mm standoff")
        lines.append("")
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
        notes = self.check_geometry(geom) + self.face_balance(geom)["notes"]
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
        np.savez_compressed(base + ".npz", **arrays)
        with open(base + ".json", "w", encoding="utf-8") as fh:
            json.dump({"magnet_note": self.magnet_note, "axis": self.axis,
                       "across": self.across, "normal": self.normal,
                       "n_poses": len(self.sweeps), "poses": poses,
                       "report": self.report()}, fh, indent=2)
            fh.write("\n")
        return base + ".npz"

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
                   across=across, normal=normal)


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
