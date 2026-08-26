#!/usr/bin/env python3
"""
octobee/motion/sweep.py -- map a volume by sweeping through it, logging as it goes.

Why this exists alongside octobee/motion/scan.py
------------------------------------------------
`scan.py` moves, stops, settles and averages. That is the right way to measure
ONE point: the noise is white, so twenty seconds of stationary averaging at the
carriers' own 200 kSPS reaches 0.020 uT where the moving stream manages 0.235.
Nothing here disputes that.

It stops being the right way when the question is a whole volume. The stage
envelope is 300 mm on each axis; on a 10 mm grid that is 29,791 points, and at
the seven-and-a-half seconds a settled average really costs it is sixty-two
hours -- not a long experiment, an impossible one. Halving the grid step
multiplies it by eight.

So a volume is swept instead. One axis runs at constant velocity while the
stream is logged continuously; the other two step between lines. The same
29,791 samples come off a 300 mm line in thirty seconds rather than sixty-two
hours, and what is lost is per-sample noise -- which is recoverable, because
every sample is stored and averaging afterwards is free. What is NOT recoverable
is a point never visited, which is what the settled raster actually costs you.

Both are kept. Use `scan.py` for a small box measured carefully, this for a
survey.

Every line still runs the same way
----------------------------------
scan.py's rule about backlash is not weakened by sweeping: the LTS300C drives a
leadscrew with no scale on the carriage, so reversing direction costs backlash,
and a serpentine would stamp a fixed offset into every other line. Every line
here is therefore swept in the +ve direction and the axis returns to the start
of the next one. It costs travel time and buys a map with no comb in it.

Lines also over-run at both ends where the travel allows. The controller drives
a trapezoidal profile, so the first and last few millimetres of a move are ramp,
not sweep; starting before the box and finishing after it puts the ramps outside
the region being measured and leaves the interesting part at constant velocity.

Where position comes from, and what it is worth
-----------------------------------------------
Two sources, and they answer different halves of the question.

**Where.** Neither is better than the other at this. The LTS300C has no scale on
the carriage and the quadrature encoders are rotary, on the leadscrews, so both
count turns of the same 1 mm screw and inherit the same ~47 um of absolute pitch
error without Thorlabs' per-serial calibration files. Absolute position comes
from homing either way.

**When.** Here they are not close. The controller is read over USB with tens of
milliseconds of latency, so a poll describes where the carriage was some
unknown moment ago; at 10 mm/s that is millimetres. The encoders are aggregated
into acq1001_695's sample stream, so the count in sample n was latched when
sample n was converted -- the same clock as the field, by construction, with
nothing to interpolate and no offset to estimate. See octobee/motion/encoder.py.

So, per row:

    counts                 raw encoder counts, per sample, on the ADC clock
    pos_mm                 counts against a datum, where the axis is calibrated
    poll_t_s / poll_mm     every controller read, wall clock, stored as taken

The datum is the controller's own position at the moment the line started,
which puts absolute position on the homing and timing on the ADC clock -- each
on whichever is actually responsible for it. An axis with no calibrated encoder
falls back to interpolating its polls onto the sample clock, and `pos_source`
in the sidecar says, per axis, which it was. Nothing is thrown away: the raw
counts and the raw polls are both stored, so either can be redone.
"""

import json
import math
import os
import time

import numpy as np

from octobee.calib import geometry as pgeom
from octobee.motion import encoder as oenc
from octobee.motion import timing as otim

AXES = ("x", "y", "z")

# Sweep velocity if nothing says otherwise, and the ceiling the GUI offers.
# Not a hardware limit -- the axes will do 10 -- a measurement one: the head is
# on the end of a cantilever, and a sweep is the one thing here that measures
# while the thing it is measuring with is moving.
DEFAULT_SPEED_MM_S = 5.0
MAX_SPEED_MM_S = 5.0

# Rows per second written to the log. The physics has nothing above a few Hz
# once the probe is moving at centimetres per second, and 100 Hz at 10 mm/s is a
# sample every 0.1 mm -- already finer than the leadscrew is accurate.
DEFAULT_LOG_HZ = 100.0

# How long to sit still at the start of a line before the sweep begins, so the
# first samples are not the tail of the move that got there.
DEFAULT_SETTLE_S = 0.5


class Volume:
    """The box to be mapped, in RIG millimetres, and how finely.

    Rig millimetres because that is what the stages move in and what a scan can
    actually be commanded in. Where the box sits in the machine depends on the
    probe's placement, which is a separate declaration and can be re-stated
    without re-planning the sweep.

    `sweep` names the axis that runs continuously; the other two step by
    `step_mm`. Sweeping the axis with the longest run in the box is what makes
    the whole thing quick, and is the default when nothing is chosen.
    """

    def __init__(self, lo_mm=(0.0, 0.0, 0.0), hi_mm=(300.0, 300.0, 300.0),
                 step_mm=10.0, sweep="x"):
        self.lo_mm = np.asarray(lo_mm, dtype=float).reshape(3)
        self.hi_mm = np.asarray(hi_mm, dtype=float).reshape(3)
        if np.any(self.hi_mm < self.lo_mm):
            raise ValueError("the volume's upper corner is below its lower one")
        self.step_mm = float(step_mm)
        if self.step_mm <= 0.0:
            raise ValueError("the grid step must be positive")
        if sweep not in AXES:
            raise ValueError(f"sweep axis must be one of {AXES}, got {sweep!r}")
        self.sweep = sweep

    @classmethod
    def whole_travel(cls, travel_mm, step_mm=10.0, sweep="x"):
        """The entire envelope the stages can reach, as Placement records it."""
        lo = [float(travel_mm[a][0]) for a in AXES]
        hi = [float(travel_mm[a][1]) for a in AXES]
        return cls(lo, hi, step_mm=step_mm, sweep=sweep)

    @property
    def size_mm(self):
        return self.hi_mm - self.lo_mm

    @property
    def sweep_index(self):
        return AXES.index(self.sweep)

    @property
    def step_axes(self):
        """The two axes that step, in order."""
        return tuple(a for a in AXES if a != self.sweep)

    @property
    def shape(self):
        """Nodes along each of x, y, z at the grid step, stop inclusive."""
        n = np.floor(self.size_mm / self.step_mm + 1e-9).astype(int) + 1
        return tuple(int(max(1, v)) for v in n)

    def nodes(self, axis):
        """The grid coordinates along one axis, mm."""
        i = AXES.index(axis)
        return self.lo_mm[i] + self.step_mm * np.arange(self.shape[i])

    def corners_mm(self):
        """The eight corners, for drawing. Rig millimetres."""
        return np.array([[self.lo_mm[0] if not (i & 4) else self.hi_mm[0],
                          self.lo_mm[1] if not (i & 2) else self.hi_mm[1],
                          self.lo_mm[2] if not (i & 1) else self.hi_mm[2]]
                         for i in range(8)])

    def describe(self):
        return (f"{self.size_mm[0]:g}x{self.size_mm[1]:g}x{self.size_mm[2]:g} mm "
                f"from ({self.lo_mm[0]:g}, {self.lo_mm[1]:g}, "
                f"{self.lo_mm[2]:g}), {self.step_mm:g} mm grid, sweeping "
                f"{self.sweep}")

    def to_dict(self):
        return {"lo_mm": self.lo_mm.tolist(), "hi_mm": self.hi_mm.tolist(),
                "step_mm": self.step_mm, "sweep": self.sweep}

    @classmethod
    def from_dict(cls, doc):
        doc = dict(doc or {})
        return cls(doc.get("lo_mm", (0.0, 0.0, 0.0)),
                   doc.get("hi_mm", (300.0, 300.0, 300.0)),
                   doc.get("step_mm", 10.0), doc.get("sweep", "x"))


class SweepLine:
    """One constant-velocity pass: where it starts, where it ends, what is fixed.

    `start_mm`/`stop_mm` are the ends of the MEASURED span. `run_in_mm` and
    `run_out_mm` are the extra travel outside it, where the stage is ramping;
    samples taken there are logged and flagged rather than thrown away, because
    a sample whose position is known is never worthless and the flag is cheaper
    than deciding now.
    """

    def __init__(self, sweep, start_mm, stop_mm, fixed, run_in_mm=0.0,
                 run_out_mm=0.0):
        self.sweep = sweep
        self.start_mm = float(start_mm)
        self.stop_mm = float(stop_mm)
        self.fixed = dict(fixed)
        self.run_in_mm = float(run_in_mm)
        self.run_out_mm = float(run_out_mm)

    @property
    def span_mm(self):
        return self.stop_mm - self.start_mm

    @property
    def from_mm(self):
        """Where the stage is actually commanded to start, ramp included."""
        return self.start_mm - self.run_in_mm

    @property
    def to_mm(self):
        """Where the stage is actually commanded to stop, ramp included."""
        return self.stop_mm + self.run_out_mm

    def approach(self):
        """The full coordinate dict for the start of this line."""
        return {**self.fixed, self.sweep: self.from_mm}

    def to_dict(self):
        return {"sweep": self.sweep, "start_mm": self.start_mm,
                "stop_mm": self.stop_mm, "fixed": dict(self.fixed),
                "run_in_mm": self.run_in_mm, "run_out_mm": self.run_out_mm}


class SweepPlan:
    """Every line to be swept, in the order they will be swept."""

    def __init__(self, volume, lines, speed_mm_s=DEFAULT_SPEED_MM_S,
                 log_hz=DEFAULT_LOG_HZ, settle_s=DEFAULT_SETTLE_S,
                 skipped=0, note=""):
        self.volume = volume
        self.lines = list(lines)
        self.speed_mm_s = float(speed_mm_s)
        self.log_hz = float(log_hz)
        self.settle_s = float(settle_s)
        self.skipped = int(skipped)
        self.note = str(note)

    def __len__(self):
        return len(self.lines)

    @property
    def swept_mm(self):
        return float(sum(ln.span_mm for ln in self.lines))

    def duration_s(self, accel_mm_s2=20.0):
        """Roughly how long this takes, in seconds.

        The sweeps themselves are exact -- distance over velocity, plus the
        ramps. So is the return: these axes are capped at MAX_VEL_MM_S, so
        going back to the start of the next line costs the same as sweeping it
        did, and a volume takes twice as long as the sweeping alone suggests.
        That is the price of never reversing into a line, and it is included
        here rather than discovered overnight.
        """
        v, a = otim.clamp_velocity(self.speed_mm_s), float(accel_mm_s2)
        back = otim.clamp_velocity(otim.MAX_VEL_MM_S)
        total = 0.0
        for ln in self.lines:
            reach = abs(ln.to_mm - ln.from_mm)
            total += otim.move_time_s(reach, v, a) + self.settle_s
            total += otim.move_time_s(reach, back, a)
        return total

    def rows(self):
        """How many log rows this will produce, roughly."""
        return int(round(self.duration_s() * self.log_hz))

    def describe(self):
        if not self.lines:
            return "no lines -- nothing to sweep"
        mins = self.duration_s() / 60.0
        bits = (f"{len(self.lines)} lines, {self.swept_mm / 1000.0:.1f} m swept "
                f"at {self.speed_mm_s:g} mm/s, roughly "
                f"{mins:.0f} min ({mins / 60.0:.1f} h), about "
                f"{self.rows():,} rows at {self.log_hz:g} Hz")
        if self.skipped:
            bits += f" -- {self.skipped} line(s) dropped as unreachable"
        return f"{self.note}: {bits}" if self.note else bits

    def to_dict(self):
        return {"volume": self.volume.to_dict(),
                "speed_mm_s": self.speed_mm_s, "log_hz": self.log_hz,
                "settle_s": self.settle_s, "skipped": self.skipped,
                "note": self.note,
                "lines": [ln.to_dict() for ln in self.lines]}


def _runs(mask):
    """Maximal runs of True in a 1-D boolean array, as (first, last) indices."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks, [len(idx) - 1]])
    return [(int(idx[a]), int(idx[b])) for a, b in zip(starts, stops)]


def _face_rasters(volume, shell):
    """(sweep axis, first stepping axis, second) for each raster to run.

    Filling a volume is one raster: sweep one axis, step the other two over
    every node. Mapping only its OUTSIDE is six, one per face, and each has to
    be swept in the plane of its own face -- the two faces perpendicular to
    the usual sweep axis cannot be swept along it at all. Each entry says which
    axis runs and which two step; the caller restricts the stepping axes to
    the face.
    """
    sweep = volume.sweep
    a, b = volume.step_axes
    if not shell:
        return [(sweep, a, b, None)]
    # Four faces parallel to the sweep axis: pin one stepping axis to each end
    # and step the other across. Then two faces perpendicular to it, swept
    # along `a` with `b` stepping, pinned at each end of the sweep axis.
    out = []
    for pinned, across in ((a, b), (b, a)):
        for end in (0, -1):
            out.append((sweep, pinned, across, (pinned, end)))
    for end in (0, -1):
        out.append((a, sweep, b, (sweep, end)))
    return out


def plan(volume, reachable=None, speed_mm_s=DEFAULT_SPEED_MM_S,
         log_hz=DEFAULT_LOG_HZ, settle_s=DEFAULT_SETTLE_S, accel_mm_s2=20.0,
         travel_mm=None, min_span_mm=None, shell=False):
    """Turn a Volume into the lines that will actually be swept.

    `reachable` is a boolean array of `volume.shape` -- machine.reachable_grid's
    answer -- saying which grid nodes the probe body can occupy. Where it is
    given, each line is cut into the maximal clear runs along the swept axis, so
    a line that passes behind a winding becomes two shorter lines with the coil
    between them rather than one line through it. That is what makes the mapped
    region the shape of the space that is actually there, instead of the largest
    box that happens to fit in it.

    `shell` maps the OUTSIDE of the box instead of filling it: the six faces,
    each swept in its own plane. For a field in a current-free region that is
    not a poor relation of the filled version -- the boundary values determine
    the interior -- and it is five or six times fewer lines, which is the
    difference between an evening and a week.

    `travel_mm` is {axis: (lo, hi)}; the run-up and run-down are fitted into
    whatever it leaves outside the box, and silently shortened where it does
    not. A line with no room to ramp is still swept -- it just measures its own
    first millimetres at less than full speed, which the position log records
    faithfully.
    """
    ramp = otim.ramp_distance_mm(speed_mm_s, accel_mm_s2)
    span_floor = (volume.step_mm if min_span_mm is None else float(min_span_mm))
    reachable = None if reachable is None else np.asarray(reachable)

    lines, skipped, seen = [], 0, set()
    for axis, first_ax, second_ax, pin in _face_rasters(volume, shell):
        along = volume.nodes(axis)
        firsts = range(volume.shape[AXES.index(first_ax)])
        seconds = range(volume.shape[AXES.index(second_ax)])
        if pin is not None:
            pin_ax, pin_at = pin
            n = volume.shape[AXES.index(pin_ax)]
            at = pin_at % n
            if pin_ax == first_ax:
                firsts = [at]
            elif pin_ax == second_ax:
                seconds = [at]
        for i1 in firsts:
            for i2 in seconds:
                idx = {first_ax: i1, second_ax: i2}
                fixed = {k: float(volume.nodes(k)[v]) for k, v in idx.items()}
                key = (axis, tuple(sorted(idx.items())))
                if key in seen:
                    continue          # the faces share their edges
                seen.add(key)
                if reachable is None:
                    spans = [(0, len(along) - 1)]
                else:
                    sel = [slice(None)] * 3
                    for k, v in idx.items():
                        sel[AXES.index(k)] = v
                    spans = _runs(reachable[tuple(sel)])
                if not spans:
                    skipped += 1
                    continue
                for start_i, stop_i in spans:
                    start, stop = float(along[start_i]), float(along[stop_i])
                    if stop - start < span_floor - 1e-9:
                        skipped += 1
                        continue
                    lo_t, hi_t = ((-math.inf, math.inf) if not travel_mm
                                  else tuple(float(v) for v in travel_mm[axis]))
                    run_in = max(0.0, min(ramp, start - lo_t))
                    run_out = max(0.0, min(ramp, hi_t - stop))
                    lines.append(SweepLine(axis, start, stop, fixed,
                                           run_in, run_out))
    note = ("the outside of the box" if shell else "the whole box, filled")
    note += (", cut to the space the probe body actually fits in"
             if reachable is not None else ", coils ignored")
    return SweepPlan(volume, lines, speed_mm_s, log_hz, settle_s, skipped, note)


class Aborted(RuntimeError):
    """The caller asked the sweep to stop."""


# How often the sweeping axis is asked where it is, seconds. The DLL answers
# out of its own polling cache, so this is not a USB round trip per call; 20 ms
# at 10 mm/s is a poll every 0.2 mm, which is finer than the leadscrew is
# accurate and leaves the interpolation nothing to invent.
POLL_S = 0.02


class SweepRunner:
    """Drives the stages through a SweepPlan. Blocking; run it in a thread.

    It does NOT touch the carriers. The field arrives through whatever is
    already draining the stream -- in the GUI, the acquisition tick -- and is
    handed to the same SweepLog from there. That split is deliberate: the
    stream has one consumer and it stays that way, so a sweep cannot race the
    live plot for blocks and cannot silently put holes in a recording.

    What this object publishes for that consumer to read is two attributes:

        logging      True only while an axis is actually sweeping a line
        line_index   which line that is

    Both are plain assignments, read from the other thread without a lock. A
    block landing on the wrong side of a boundary would be misfiled, so the
    flag goes down BEFORE the return move starts and up only AFTER the next
    line has settled -- the samples either side are ramp and return, which are
    not wanted anyway.

    And one it reads back: `counts_now`, the most recent encoder counts, which
    the same consumer keeps up to date. It is used once per line, standing
    still, to write down where that line's counting started from.
    """

    def __init__(self, stages, plan_, log, accel_mm_s2=None, log_fn=print):
        self.stages = stages
        self.plan = plan_
        self.log = log
        self.accel_mm_s2 = accel_mm_s2
        self.log_fn = log_fn
        self.logging = False
        self.line_index = -1
        self.counts_now = None
        self.lines_done = 0
        self.aborted = False

    def _fresh_counts(self, timeout_s=1.0):
        """Wait for a counts reading taken since this moment, or give up.

        The datum pairs an encoder count with a controller position, and the
        whole value of it is that the two describe the same instant. The
        counts arrive on someone else's tick, so the one sitting in
        `counts_now` when a line starts may be a tick old -- which at sweep
        speed is most of a millimetre, silently added to every position on
        that line. Waiting for a new object (the tick hands over a fresh array
        each time) costs one tick period per line and removes it.

        The axis is stationary here, so the wait is not lost motion. Giving up
        is safe: a stale datum is still better than none, and a missing one
        drops the line back to the polls, which is what happens anyway on a
        rig with no encoders.
        """
        was = self.counts_now
        deadline = time.monotonic() + timeout_s
        while self.counts_now is was and time.monotonic() < deadline:
            time.sleep(POLL_S)
        return self.counts_now

    def _poll(self, index):
        try:
            where = self.stages.position()
        except Exception:
            return None                      # not a reason to end a sweep
        self.log.add_poll(time.time(), where, index)
        return where

    def run(self, should_abort=None, on_progress=None):
        """Sweep every line. Returns the number completed.

        The swept axis is taken from each LINE, not from the volume: mapping
        the outside of a box means sweeping in the plane of each face, so the
        two faces perpendicular to the usual sweep axis are swept along a
        different one. Anything that touched the axis -- the velocity profile,
        the move, putting the profile back -- therefore does so per line.
        """
        self.stages.interlock.require_clear("a volume sweep")

        unhomed = [(n, self.stages[n].distrust_reason) for n in AXES
                   if n in self.stages.names
                   and not self.stages[n].position_trusted]
        if unhomed:
            raise ValueError(
                "these axes' position counters cannot be believed, so the "
                "sweep would have no origin -- home them first: "
                + "; ".join(f"{n} ({why})" for n, why in unhomed))

        # What each axis is normally set to, so it can be put back however the
        # sweep ends. Read once: these are the profiles the rig arrived with.
        normal = {a: self.stages[a].vel_params for a in AXES
                  if a in self.stages.names}
        speed = otim.clamp_velocity(self.plan.speed_mm_s)
        self.log.start()
        self.log_fn(f"sweep: {self.plan.describe()}")
        try:
            for i, line in enumerate(self.plan.lines):
                if should_abort and should_abort():
                    self.aborted = True
                    break
                axis = line.sweep
                st = self.stages[axis]
                accel = self.accel_mm_s2 or normal[axis][1]
                # 1. get there at the ordinary speed, and let it stop ringing.
                st.set_vel_params(*normal[axis])
                self.stages.move_to(settle_s=self.plan.settle_s,
                                    **line.approach())
                if should_abort and should_abort():
                    self.aborted = True
                    break

                # 2. sweep, logging position the whole way.
                self.log.lines.append(line.to_dict())
                st.set_vel_params(speed, accel)
                self.line_index = i
                # Where this line's counting starts from, taken standing still
                # -- the only moment the controller's USB latency costs
                # nothing, and the anchor everything the encoder says after it
                # is measured against. Counts first, then the position, both
                # with nothing moving.
                counts = (self._fresh_counts() if self.log.encoders else None)
                here = self._poll(i)
                if here is not None and counts is not None:
                    self.log.add_datum(i, counts, here)
                self.logging = True
                try:
                    st.move_to(line.to_mm, wait=False)
                    deadline = time.monotonic() + otim.move_timeout_s(
                        line.to_mm - line.from_mm, speed, accel)
                    while True:
                        self._poll(i)
                        if not st.moving:
                            break
                        if time.monotonic() > deadline:
                            self.log_fn(f"sweep: line {i + 1} did not finish "
                                        f"in time -- stopping it")
                            st.stop()
                            break
                        if should_abort and should_abort():
                            self.aborted = True
                            st.stop()
                            break
                        time.sleep(POLL_S)
                finally:
                    self.logging = False
                    self._poll(i)
                self.lines_done = i + 1
                if on_progress:
                    on_progress(i + 1, len(self.plan.lines), line)
                if self.aborted:
                    break
        finally:
            self.logging = False
            for a, profile in normal.items():
                try:
                    self.stages[a].set_vel_params(*profile)
                except Exception as exc:
                    self.log_fn(f"sweep: could not put {a}'s speed back "
                                f"({exc}) -- check it before the next move")
        self.log_fn(f"sweep: {self.lines_done} of {len(self.plan.lines)} lines"
                    + (" (aborted)" if self.aborted else ""))
        return self.lines_done


class SweepLog:
    """Everything a sweep produced, growing as it runs.

    Rows are appended as the stream arrives and positions as the stages are
    polled; the two are married at the end, once, rather than sample by sample
    -- interpolating against a poll list that is still being appended to is how
    the last point of every line ends up extrapolated.
    """

    def __init__(self, geom=None, meta=None, encoders=None):
        self.geom = geom or pgeom.Geometry()
        self.meta = dict(meta or {})
        self.encoders = encoders or oenc.EncoderMap()
        self._b = []                  # list of (n, 16, 3) mT blocks
        self._c = []                  # list of (n, columns) encoder counts
        self._line_of = []            # line index per block
        self._t0 = []                 # sample time of each block's first row
        self.poll_t_s = []            # wall clock of every stage read
        self.poll_mm = []             # (x, y, z) rig mm at that read
        self.poll_line = []
        self.lines = []               # SweepLine.to_dict(), in order run
        # {line index: {axis: (counts_at_datum, mm_at_datum)}}. Taken once per
        # line, standing still, before the sweep starts: an encoder counts
        # displacement and has to be told where it started from.
        self.datum = {}
        self.t_wall0 = None
        self.n_rows = 0

    # ---- collection --------------------------------------------------
    def start(self):
        self.t_wall0 = time.time()

    def add_block(self, b_mt, t_s, line_index, counts=None):
        """One decimated block of field samples, and the counts that go with it.

        `counts` must be the SAME rows -- the encoder columns for exactly these
        samples. A block whose counts do not match in length is stored without
        them rather than with them mis-aligned, because a position column that
        is a few samples out is undetectable downstream and a missing one is
        not.
        """
        b = np.asarray(b_mt, dtype=np.float32)
        if b.ndim != 3 or b.shape[1:] != (pgeom.N_SENSORS, 3):
            raise ValueError(f"expected (n, {pgeom.N_SENSORS}, 3) mT, got "
                             f"{b.shape}")
        if not len(b):
            return
        self._b.append(b)
        if counts is not None and len(counts) == len(b):
            self._c.append(np.asarray(counts, dtype=np.float64))
        else:
            self._c.append(None)
        self._t0.append(float(t_s))
        self._line_of.append(int(line_index))
        self.n_rows += len(b)

    def add_datum(self, line_index, counts, stage_mm):
        """Where this line started, in counts and in millimetres, standing still.

        Both read at the same moment with nothing moving, which is the only
        moment the USB latency on the controller read costs nothing.
        """
        if counts is None:
            return
        counts = np.asarray(counts, dtype=float).ravel()
        self.datum[int(line_index)] = {
            a: (float(counts[spec["column"]]), float(stage_mm[a]))
            for a, spec in self.encoders.axes.items()
            if spec["column"] < len(counts) and a in stage_mm}

    def add_poll(self, t_s, pos_mm, line_index):
        self.poll_t_s.append(float(t_s))
        self.poll_mm.append([float(pos_mm.get(a, np.nan)) for a in AXES])
        self.poll_line.append(int(line_index))

    # ---- what came out ------------------------------------------------
    def assemble(self, log_hz, placement=None):
        """(t_s, pos_rig_mm, pos_machine_mm, b_mt, line_index, in_span, counts).

        Position comes from the encoders on every axis that has a calibrated
        one, because those counts were latched with the samples they are being
        put against. Whatever is left is interpolated from the controller polls
        PER LINE -- interpolating across a line boundary would draw a straight
        line through the return move, inventing a sweep that never happened --
        and samples outside a line's polls are left NaN rather than
        extrapolated.
        """
        if not self._b:
            empty3 = np.zeros((0, 3))
            return (np.zeros(0), empty3, empty3,
                    np.zeros((0, pgeom.N_SENSORS, 3), np.float32),
                    np.zeros(0, int), np.zeros(0, bool), np.zeros((0, 0)))
        dt = 1.0 / float(log_hz)
        t_parts, line_parts = [], []
        for block, t0, li in zip(self._b, self._t0, self._line_of):
            t_parts.append(t0 + dt * np.arange(len(block)))
            line_parts.append(np.full(len(block), li, dtype=int))
        t_s = np.concatenate(t_parts)
        line_index = np.concatenate(line_parts)
        b_mt = np.concatenate(self._b, axis=0)

        width = max((0 if c is None else c.shape[1]) for c in self._c)
        counts = np.full((len(t_s), width), np.nan)
        if width:
            at = 0
            for block, c in zip(self._b, self._c):
                if c is not None:
                    counts[at:at + len(block), :c.shape[1]] = c
                at += len(block)

        pos = np.full((len(t_s), 3), np.nan)
        # ---- the encoders first, where they are calibrated ----
        for li in np.unique(line_index):
            datum = self.datum.get(int(li))
            if not datum or not width:
                continue
            rows = np.flatnonzero(line_index == li)
            here = counts[rows]
            for i, axis in enumerate(AXES):
                if axis not in self.encoders or axis not in datum:
                    continue
                col = self.encoders.axes[axis]["column"]
                if col >= width:
                    continue
                c0, mm0 = datum[axis]
                pos[rows, i] = mm0 + self.encoders.displacement_mm(
                    here[:, col], axis, c0)

        # ---- and the polls for whatever is left ----
        polls_t = np.asarray(self.poll_t_s, dtype=float)
        polls_p = np.asarray(self.poll_mm, dtype=float).reshape(-1, 3)
        polls_l = np.asarray(self.poll_line, dtype=int)
        for li in np.unique(line_index):
            rows = line_index == li
            keep = (polls_l == li) & np.isfinite(polls_p).all(axis=1)
            if keep.sum() < 2:
                continue
            pt, pp = polls_t[keep], polls_p[keep]
            order = np.argsort(pt)
            pt, pp = pt[order], pp[order]
            inside = rows & (t_s >= pt[0]) & (t_s <= pt[-1])
            for a in range(3):
                gap = inside & ~np.isfinite(pos[:, a])
                if gap.any():
                    pos[gap, a] = np.interp(t_s[gap], pt, pp[:, a])

        in_span = np.zeros(len(t_s), dtype=bool)
        for li, doc in enumerate(self.lines):
            rows = line_index == li
            if not rows.any():
                continue
            k = AXES.index(doc["sweep"])
            here = pos[:, k]
            in_span |= rows & (here >= doc["start_mm"]) & (here <= doc["stop_mm"])

        if placement is None:
            machine = np.full_like(pos, np.nan)
        else:
            machine = placement.to_machine(np.nan_to_num(pos), None)
            machine[~np.isfinite(pos).all(axis=1)] = np.nan
        return t_s, pos, machine, b_mt, line_index, in_span, counts

    def pos_source(self):
        """Which axes' positions came from the encoders, and which from polls."""
        return {a: ("encoder, on the ADC clock" if a in self.encoders
                    else "controller poll, interpolated onto the sample clock")
                for a in AXES}

    def save(self, path, log_hz, placement=None, sync_note=""):
        """Write <path>.npz (bulk) and <path>.json (provenance)."""
        base = os.path.splitext(path)[0]
        d = os.path.dirname(os.path.abspath(base))
        if d:
            os.makedirs(d, exist_ok=True)
        t_s, pos, machine, b_mt, line_index, in_span, counts = self.assemble(
            log_hz, placement)
        np.savez_compressed(
            base + ".npz",
            t_s=t_s.astype(np.float64),
            pos_mm=pos.astype(np.float64),
            pos_machine_mm=machine.astype(np.float64),
            b_mt=b_mt.astype(np.float32),
            counts=counts.astype(np.float64),
            line_index=line_index.astype(np.int32),
            in_span=in_span,
            poll_t_s=np.asarray(self.poll_t_s, dtype=np.float64),
            poll_mm=np.asarray(self.poll_mm, dtype=np.float64).reshape(-1, 3),
            poll_line=np.asarray(self.poll_line, dtype=np.int32))
        side = dict(self.meta)
        side.update({
            "n_rows": int(len(t_s)),
            "n_lines": len(self.lines),
            "n_polls": len(self.poll_t_s),
            "log_hz": float(log_hz),
            "axes": list(AXES),
            "started": (None if self.t_wall0 is None else time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self.t_wall0))),
            "lines": self.lines,
            "columns": {
                "t_s": "seconds from the start of the sweep",
                "pos_mm": "rig mm (x, y, z) -- see position_source, per axis",
                "pos_machine_mm": "the same point in the coil file's frame, mm",
                "b_mt": "(rows, 16, 3) millitesla, Bx By Bz per sensor",
                "counts": "raw quadrature encoder counts, unwrapped, one row "
                          "per field sample, latched by the same clock",
                "line_index": "which swept line each row belongs to",
                "in_span": "row lies inside the line's measured span, not a ramp",
                "poll_t_s/poll_mm": "raw controller reads on the wall clock, "
                                    "so the fallback can be redone",
            },
            "position_source": self.pos_source(),
            "encoders": self.encoders.to_dict(),
            "encoder_datum": {str(k): v for k, v in sorted(self.datum.items())},
            "position_accuracy": (
                "Both sources count turns of the same 1 mm leadscrew: the "
                "encoders are rotary, on the motors, and the LTS300C has no "
                "scale on the carriage. Repeatable to micrometres, absolute to "
                "~47 um without Thorlabs' per-serial calibration files, and "
                "the absolute datum comes from homing either way. What the "
                "encoders fix is timing, not accuracy."),
            "sync_note": sync_note,
            "uncorrected": ("VCM subtracted and scaled by nominal V/T only -- "
                            "no tare, no gain trim, no matrix"),
        })
        with open(base + ".json", "w", encoding="utf-8") as fh:
            json.dump(side, fh, indent=2, default=str)
            fh.write("\n")
        return base + ".npz"

    @classmethod
    def load(cls, path):
        """(arrays dict, sidecar dict) for a saved sweep."""
        base = os.path.splitext(path)[0]
        with np.load(base + ".npz") as z:
            arrays = {k: z[k] for k in z.files}
        side = {}
        if os.path.exists(base + ".json"):
            with open(base + ".json", encoding="utf-8") as fh:
                side = json.load(fh)
        return arrays, side
