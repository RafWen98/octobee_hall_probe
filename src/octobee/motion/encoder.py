#!/usr/bin/env python3
"""
octobee/motion/encoder.py -- position that arrives on the same clock as the field.

What the hardware does
----------------------
acq1001_695 aggregates three quadrature encoder modules into its sample stream
alongside the ADC. Its rc.user says so:

    # Enable X, Y, Z Quadrature Encoder counts
    for sites in 2 5 6; do set.site $sites phaseA_en 1; ... done
    # Site 1 - ACQ423, Site 2,5,6 - Quadrature Encoder, 7 LWord SPAD
    run0 1,2,5,6 1,7,0

So every frame the 695 emits carries the encoder counts for that sample, in the
longwords between the analogue channels and the scratchpad -- which is what
octobee/acq/carrier.py already hands back as `enc`. acq1001_694 has no encoder
sites and contributes nothing here.

Why that matters more than it sounds
------------------------------------
It is NOT a better position than the Thorlabs controllers give. These are
rotary encoders on the motors, so they see the same leadscrew the step counter
does and inherit the same ~47 um of absolute pitch error; they cannot tell you
anything the controller could not.

What they can tell you is WHEN. A swept map's hard problem was never where the
carriage was, it was matching a position read over USB with tens of
milliseconds of latency against a field sample that arrived through a stream
buffer with its own. Both were on the wall clock and neither was on the ADC's,
so every sample's position carried an unknown common offset -- a millimetre or
so at 10 mm/s, and the honest thing was to say so in the sidecar.

Counted in the frame, that offset is gone by construction. The count in sample
n was latched when sample n was converted. There is nothing to interpolate and
no clock to reconcile.

Absolute where, relative when
-----------------------------
A quadrature counter is incremental: it counts from wherever it happened to be
at power-up, and knows nothing about home. So it is used as a DISPLACEMENT
against a datum taken from the controller:

    mm(n) = mm_at_datum + (counts(n) - counts_at_datum) / counts_per_mm

which puts the absolute position where it belongs -- on the homing -- and the
timing where it belongs, on the ADC clock. Both halves are then as good as the
thing actually responsible for them.

The scale is measured, not typed
--------------------------------
`fit_axis` steps an axis along a span under the controller, reading the
counts and the controller's own position at each standstill, and regresses one
against the other. That identifies which longword belongs to which axis as
well, empirically, rather than trusting the site order in a comment -- and it
is a check that can be re-run, which is the half worth having. A typed-in
counts-per-mm that is wrong by a factor rescales every position in every map
and looks entirely plausible doing it.

What it fits against is the controller's MEASURED displacement, not the
distance the move was asked for. Those are not the same number, and the
difference goes straight into the scale: the first calibration run on this rig
fitted x at 14,341 counts/mm against a commanded 20 mm while y and z came back
at 14,400 within seven parts per million of each other. Three identical
stages do not have different gearing -- x had undershot its commanded move by
81 um, and dividing by 20 instead of by 19.919 put 0.41% into the scale, which
is 1.2 mm across the 300 mm axis in a column labelled X_mm.

Fitting against the controller is also the only thing that is being asked
here. The encoder is on the motor and the controller counts the same
leadscrew, so what is wanted is counts per CONTROLLER millimetre -- a gearing
ratio. Whatever the leadscrew's own pitch error is, both sides inherit it and
it cancels out of the ratio; it is not something this could measure and not
something it needs to.

Why more than two points
------------------------
Two points always fit a line perfectly, so a two-point fit has no residual and
therefore no way to say whether it worked. Ten points across a long span give
one: a fit whose residual is a micrometre is measuring the gearing, and a fit
whose residual is fifty is measuring something else -- a stage that stalled
partway, a column that belongs to another axis, a counter that dropped edges.
That number is the difference between a calibration and a number that came
back from a calibration.

Running it in both directions gives a second check for free, and it is worth
being clear about what that check is and is not. It is NOT backlash: the
encoder and the controller's own counter are both on the motor, so slack in
the leadscrew nut moves the carriage relative to both of them equally and
neither can see it. Nothing on this bench can measure that, and this does not
pretend to.

What the two directions do check is that the two counters agree with each
other about the same place regardless of how it was approached. They are
geared together, so they should, exactly -- and `direction_offset_um` being
anything other than about zero means a reading was taken while an axis was
still moving, or the quadrature counter is dropping edges at speed. It is a
symptom, not a property of the rig.
"""

import json
import os

import numpy as np

from octobee import paths
from octobee.motion.config import AXIS_CONFIG

AXES = ("x", "y", "z")

# A quadrature counter is 32 bits and wraps. At a few thousand counts per mm a
# 300 mm axis is a millionth of the range, so a wrap is rare -- which is
# exactly why it has to be handled here rather than discovered in a map.
WRAP = 1 << 32

# How far a calibration run spans by default, and how many standstills it
# takes along it. The span is the lever on every fixed error in the fit -- a
# stage that ends 80 um short of each target puts that 80 um into the scale
# divided by the span, so 100 mm is five times better than 20 for free. It is
# only a default: the run clamps it to whatever room the axis actually has,
# and the operator can ask for more or less.
CALIBRATION_SPAN_MM = 100.0
CALIBRATION_POINTS = 11

# Under this there is no span worth fitting over, whatever was asked for.
MIN_CALIBRATION_SPAN_MM = 5.0

# Below this the axis did not really move and the fit means nothing.
MIN_CALIBRATION_COUNTS = 100


def unwrap32(raw, previous=None, offset=0):
    """Continuous int64 counts from wrapping uint32 ones.

    `raw` is one block of counts for one axis, oldest first. `previous` is the
    last raw value of the block before it, `offset` the accumulated wrap
    correction so far. Returns (counts, last_raw, offset) so a stream can
    carry the state forward.

    Differences are masked back into 32 bits and re-signed, which is the same
    trick check_continuity uses on the sample counter, and for the same
    reason: subtracting in int64 turns one wrap into a four-billion-count jump
    that looks exactly like the stage teleporting.
    """
    raw = np.asarray(raw, dtype=np.uint32).astype(np.int64)
    if not len(raw):
        return np.zeros(0, dtype=np.int64), previous, offset
    first = raw[0] if previous is None else np.int64(previous)
    steps = np.diff(np.concatenate([[first], raw]))
    steps = (steps + (WRAP // 2)) % WRAP - (WRAP // 2)      # signed, wrapped
    return (offset + np.cumsum(steps) + first, int(raw[-1]),
            int(offset + steps.sum()) - int(raw[-1]) + int(first))


class EncoderStream:
    """Per-axis unwrapping across a stream of blocks. One per source."""

    def __init__(self, n_columns):
        self.n_columns = int(n_columns)
        self._prev = [None] * self.n_columns
        self._offset = [0] * self.n_columns

    def reset(self):
        self._prev = [None] * self.n_columns
        self._offset = [0] * self.n_columns

    def push(self, block):
        """(n, columns) raw uint32 -> (n, columns) continuous float counts.

        Float rather than int because the next thing to happen is block
        averaging alongside the field, and an integer mean would quantise the
        position to the decimation factor for no reason.
        """
        block = np.asarray(block)
        if block.ndim != 2 or block.shape[1] != self.n_columns:
            raise ValueError(f"expected (n, {self.n_columns}) counts, got "
                             f"{block.shape}")
        out = np.empty(block.shape, dtype=np.float64)
        for c in range(self.n_columns):
            counts, self._prev[c], self._offset[c] = unwrap32(
                block[:, c], self._prev[c], self._offset[c])
            out[:, c] = counts
        return out


class EncoderMap:
    """Which stream column is which axis, and how many counts make a millimetre.

    `axes` maps an axis name to {"column": int, "counts_per_mm": float}. The
    count is signed: a negative counts_per_mm means the encoder runs the
    opposite way to the rig axis, which is a fact about how the thing was
    wired and not something to correct anywhere else.

    An axis that is absent, or whose scale is zero, is simply not derived from
    the encoder -- the sweep falls back to the controller's own position for
    it and says so. Partial is a real state: one axis can be wired and the
    others not.
    """

    def __init__(self, axes=None):
        self.axes = {}
        for name, spec in (axes or {}).items():
            if name not in AXES or not isinstance(spec, dict):
                continue
            scale = float(spec.get("counts_per_mm") or 0.0)
            if spec.get("column") is None or scale == 0.0:
                continue
            self.axes[str(name)] = {"column": int(spec["column"]),
                                    "counts_per_mm": scale}

    def __bool__(self):
        return bool(self.axes)

    def __contains__(self, axis):
        return axis in self.axes

    @property
    def calibrated(self):
        return sorted(self.axes)

    def columns_needed(self):
        return max((s["column"] for s in self.axes.values()), default=-1) + 1

    def displacement_mm(self, counts, axis, datum_counts):
        """Continuous counts for one axis -> millimetres from the datum."""
        spec = self.axes[axis]
        return (np.asarray(counts, dtype=float) - float(datum_counts)) \
            / spec["counts_per_mm"]

    def to_mm(self, counts, datum_counts_mm):
        """(n, columns) counts -> (n, 3) rig mm, NaN where an axis is not wired.

        `datum_counts_mm` is {axis: (counts_at_datum, mm_at_datum)} -- what the
        controller said the axis was at, and what the encoder read at the same
        moment. Everything after that is the encoder's own displacement added
        to it.
        """
        counts = np.asarray(counts, dtype=float).reshape(
            -1, max(1, self.columns_needed()))
        out = np.full((len(counts), 3), np.nan)
        for i, axis in enumerate(AXES):
            datum = datum_counts_mm.get(axis)
            if axis not in self.axes or datum is None:
                continue
            out[:, i] = float(datum[1]) + self.displacement_mm(
                counts[:, self.axes[axis]["column"]], axis, datum[0])
        return out

    def describe(self):
        if not self.axes:
            return "no encoder axes calibrated"
        return ", ".join(
            f"{a} = column {s['column']} at {s['counts_per_mm']:,.1f} counts/mm"
            for a, s in sorted(self.axes.items()))

    def to_dict(self):
        return {a: dict(s) for a, s in sorted(self.axes.items())}

    # ---- persistence, in stages.json beside the rest of the axis facts ----
    @classmethod
    def load(cls, path=None):
        path = path or paths.config(AXIS_CONFIG)
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return cls((doc.get("encoders") or {}).get("axes"))

    def save(self, path=None, note=""):
        path = path or paths.config(AXIS_CONFIG)
        doc = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        doc["encoders"] = {"axes": self.to_dict(), "note": note}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
        return path


class AxisFit:
    """What one calibration run measured about one axis.

    Falsy when the run did not identify a column, in which case `why` says
    what stopped it. A truthy fit carries the numbers that say how good it is
    as well as the number that gets used, because a counts/mm on its own
    cannot be told apart from a wrong counts/mm.
    """

    def __init__(self, column=None, counts_per_mm=0.0, intercept=0.0,
                 span_mm=0.0, centre_mm=0.0, n_points=0, residual_um=None,
                 worst_um=None, runner_up=None, why=""):
        self.column = column
        self.counts_per_mm = float(counts_per_mm)
        self.intercept = float(intercept)
        self.span_mm = float(span_mm)
        self.centre_mm = float(centre_mm)
        self.n_points = int(n_points)
        # None where it cannot be measured -- two points always fit a line
        # perfectly, and reporting that as a residual of zero would read as a
        # perfect fit rather than as no information.
        self.residual_um = residual_um
        self.worst_um = worst_um
        self.runner_up = runner_up
        self.why = why

    def __bool__(self):
        return self.column is not None

    def counts_at(self, mm):
        """What this fit says the counter reads at a given position."""
        return self.counts_per_mm * float(mm) + self.intercept

    def to_spec(self):
        """The two numbers EncoderMap stores, or None."""
        if not self:
            return None
        return {"column": int(self.column),
                "counts_per_mm": float(self.counts_per_mm)}

    def describe(self):
        if not self:
            return self.why
        out = (f"column {self.column}, {self.counts_per_mm:+,.1f} counts/mm "
               f"over {self.span_mm:g} mm from {self.n_points} points")
        if self.residual_um is None:
            out += " -- two points, so the fit has no residual to check it by"
        else:
            out += (f", residual {self.residual_um:.1f} um rms "
                    f"({self.worst_um:.1f} um worst)")
        return out


def datum_from(counts, encoders, stages, on_error=None):
    """Pair the counts arriving now with the controllers, axis by axis.

    Returns {axis: (counts_at_datum, mm_at_datum)} for every calibrated axis
    that can be anchored, which is what turns a counter that measures
    displacement into a column that says where the head is.

    ONE implementation, because there are two callers and they must not drift
    apart: the live recorder anchors once when Record is pressed, and a volume
    sweep anchors once per line. A difference between them would show up as
    two files of the same rig disagreeing about where it was, with nothing in
    either to say which was right.

    An axis whose counter is not trusted is SKIPPED rather than used. The
    controller answers position_mm regardless -- steps lost to a stall or an
    immediate stop leave the homed bit set and the number wrong -- and
    anchoring to that produces a column that is confidently somewhere the head
    never was. No anchor at all is a column that is missing, which is the
    failure that can be seen.
    """
    if counts is None or not encoders or stages is None:
        return {}
    counts = np.asarray(counts, dtype=float).ravel()
    out = {}
    for axis, spec in encoders.axes.items():
        st = getattr(stages, "axes", {}).get(axis)
        if st is None or spec["column"] >= len(counts):
            continue
        try:
            if not st.position_trusted:
                continue
            out[axis] = (float(counts[spec["column"]]), float(st.position_mm))
        except Exception as exc:              # a controller that has gone away
            if on_error is not None:
                on_error(f"no encoder datum for {axis}: {exc}")
    return out


def fit_axis(positions_mm, counts, min_counts=MIN_CALIBRATION_COUNTS):
    """Which column follows this axis, and by how many counts per millimetre.

    `positions_mm` is what the controller said its position was at each
    standstill; `counts` is the (n, columns) block of continuous counts read
    at those same standstills. Returns an AxisFit.

    The fit is a least squares line through every point, so a stage that
    stopped short of one target moves the residual rather than moving the
    slope -- which is the whole reason for taking more than two.

    The column is chosen by which one tracks the axis furthest, and then
    checked: a second column moving a comparable amount means either two axes
    were driven at once or the columns are not what they are assumed to be,
    and quietly picking the larger of two similar numbers is how an axis ends
    up calibrated against its neighbour.
    """
    pos = np.asarray(positions_mm, dtype=float).ravel()
    c = np.asarray(counts, dtype=float)
    if c.ndim == 1:
        c = c.reshape(-1, 1)
    if c.ndim != 2 or not c.size:
        return AxisFit(why="no encoder columns in the stream")
    if len(pos) != len(c):
        return AxisFit(why=f"{len(pos)} positions against {len(c)} count rows")
    if len(pos) < 2:
        return AxisFit(why="a line needs at least two standstills")
    if not np.isfinite(pos).all() or not np.isfinite(c).all():
        return AxisFit(why="a position or a count reading was not a number")

    span = float(pos.max() - pos.min())
    if span < 1e-9:
        return AxisFit(why="the axis never left where it started")

    # One design matrix, every column solved against it at once.
    design = np.column_stack([pos, np.ones_like(pos)])
    sol, *_ = np.linalg.lstsq(design, c, rcond=None)      # (2, columns)
    slope, intercept = sol[0], sol[1]

    moved = np.abs(slope) * span
    order = np.argsort(moved)[::-1]
    best = int(order[0])
    if moved[best] < min_counts:
        return AxisFit(why=(
            f"no column moved more than {moved[best]:.0f} counts over "
            f"{span:g} mm -- nothing here is wired to this axis"))
    if len(order) > 1 and moved[order[1]] > 0.2 * moved[best]:
        return AxisFit(why=(
            f"columns {best} and {int(order[1])} both moved "
            f"({moved[best]:.0f} and {moved[order[1]]:.0f} counts), so which "
            f"belongs to this axis cannot be told apart -- move one axis at a "
            f"time"))

    resid_counts = c[:, best] - design @ sol[:, best]
    if len(pos) > 2:
        # In micrometres of position rather than counts, because that is the
        # unit the number has to be judged in: 400 counts means nothing until
        # you know it is 28 um.
        per_um = abs(slope[best]) / 1000.0
        residual_um = float(np.sqrt(np.mean(resid_counts ** 2)) / per_um)
        worst_um = float(np.max(np.abs(resid_counts)) / per_um)
    else:
        residual_um = worst_um = None

    return AxisFit(column=best, counts_per_mm=float(slope[best]),
                   intercept=float(intercept[best]), span_mm=span,
                   centre_mm=float((pos.max() + pos.min()) / 2.0),
                   n_points=len(pos), residual_um=residual_um,
                   worst_um=worst_um,
                   runner_up=(int(order[1]) if len(order) > 1 else None))


def direction_offset_um(forward, back):
    """How far the outbound and return fits disagree, in micrometres.

    Compares what the two fits say the counter reads at the same place -- the
    middle of the span, where both are interpolating rather than
    extrapolating -- and turns the disagreement back into position.

    This is NOT backlash, and it must not be read as it. The quadrature
    encoder and the controller's own counter are both on the motor, so slack
    in the leadscrew nut moves the carriage relative to both equally and
    neither can see it. What this number checks is that two counters which are
    geared rigidly together agree about the same position whichever way it was
    approached -- which they should, exactly. A few micrometres is the noise
    on the standstill readings. Much more than that means a reading was taken
    while the axis was still moving, or the counter is dropping edges at
    speed: a symptom, not a property.

    Returns None unless both fits are good and found the same column;
    comparing two different columns is not a measurement of anything.
    """
    if not (forward and back) or forward.column != back.column:
        return None
    scale = (abs(forward.counts_per_mm) + abs(back.counts_per_mm)) / 2.0
    if scale <= 0:
        return None
    where = (forward.centre_mm + back.centre_mm) / 2.0
    return float(abs(forward.counts_at(where) - back.counts_at(where))
                 / scale * 1000.0)


def odd_axis_out(scales, tol=1e-3):
    """The axis whose scale disagrees with the others, or None.

    `scales` is {axis: counts_per_mm}; the sign is ignored, since an axis
    wired backwards is a fact about the wiring and not a different gearing.
    Returns (axis, fraction_from_the_median) for the worst offender past
    `tol`, or None when they all agree.

    Identical stages on identical leadscrews share a gearing ratio, which
    makes the axes each other's check -- and on this bench the only one
    available, because nothing else here measures millimetres. It is what
    caught the 0.41% on x: y and z agreed to seven parts per million and x did
    not, which is not something three of the same stage do.

    Needs three axes to have a median worth comparing against. With two, a
    disagreement says one of them is wrong and not which.
    """
    good = {a: abs(float(v)) for a, v in (scales or {}).items() if v}
    if len(good) < 3:
        return None
    median = sorted(good.values())[len(good) // 2]
    if median <= 0:
        return None
    off = {a: (v - median) / median for a, v in good.items()
           if abs(v - median) / median > tol}
    if not off:
        return None
    worst = max(off, key=lambda a: abs(off[a]))
    return worst, off[worst]


def fit_scale(counts_before, counts_after, moved_mm,
              min_counts=MIN_CALIBRATION_COUNTS):
    """Two-point form of fit_axis, in the shape its callers expect.

    Kept because two readings either side of one move is still the right
    procedure when all that is wanted is which column belongs to which axis --
    the idmap question rather than the calibration one. For a scale that gets
    used, prefer fit_axis: two points cannot tell you whether they were any
    good.

    Returns (column, counts_per_mm, report) or (None, 0.0, why-not).
    """
    before = np.asarray(counts_before, dtype=float).ravel()
    after = np.asarray(counts_after, dtype=float).ravel()
    if before.shape != after.shape or not len(before):
        return None, 0.0, "no encoder columns in the stream"
    if abs(moved_mm) < 1e-9:
        return None, 0.0, "the axis was asked to move nowhere"

    fit = fit_axis([0.0, float(moved_mm)], np.vstack([before, after]),
                   min_counts=min_counts)
    if not fit:
        return None, 0.0, fit.why
    return fit.column, fit.counts_per_mm, (
        f"column {fit.column}, {fit.counts_per_mm:+,.1f} counts/mm from "
        f"{moved_mm:g} mm")
