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
`fit_scale` drives an axis a known distance under the controller and fits the
counts against it. That identifies which longword belongs to which axis as
well, empirically, rather than trusting the site order in a comment -- and it
is a check that can be re-run, which is the half worth having. A typed-in
counts-per-mm that is wrong by a factor rescales every position in every map
and looks entirely plausible doing it.
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

# How far `fit_scale` drives an axis. Long enough that the fit is not dominated
# by the ends of the move, short enough to be safe to run from wherever the rig
# happens to be parked.
CALIBRATION_MM = 20.0

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


def fit_scale(counts_before, counts_after, moved_mm,
              min_counts=MIN_CALIBRATION_COUNTS):
    """Which column moved with the axis, and by how many counts per mm.

    `counts_before`/`counts_after` are continuous counts for every column,
    read either side of a move of `moved_mm`. Returns
    (column, counts_per_mm, report) or (None, 0.0, why-not).

    The column is chosen by which one moved most, and then checked: a second
    column moving a comparable amount means either two axes were driven at
    once or the columns are not what they are assumed to be, and quietly
    picking the larger of two similar numbers is how an axis ends up
    calibrated against its neighbour.
    """
    before = np.asarray(counts_before, dtype=float).ravel()
    after = np.asarray(counts_after, dtype=float).ravel()
    if before.shape != after.shape or not len(before):
        return None, 0.0, "no encoder columns in the stream"
    if abs(moved_mm) < 1e-9:
        return None, 0.0, "the axis was asked to move nowhere"

    delta = np.abs(after - before)
    order = np.argsort(delta)[::-1]
    best = int(order[0])
    if delta[best] < min_counts:
        return None, 0.0, (
            f"no column moved more than {delta[best]:.0f} counts over "
            f"{moved_mm:g} mm -- nothing here is wired to this axis")
    if len(order) > 1 and delta[order[1]] > 0.2 * delta[best]:
        return None, 0.0, (
            f"columns {best} and {int(order[1])} both moved "
            f"({delta[best]:.0f} and {delta[order[1]]:.0f} counts), so which "
            f"belongs to this axis cannot be told apart -- move one axis at a "
            f"time")
    scale = (after[best] - before[best]) / float(moved_mm)
    return best, float(scale), (
        f"column {best}, {scale:+,.1f} counts/mm from {moved_mm:g} mm")
