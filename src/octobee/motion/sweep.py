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

That is the default, and `plan(both_ways=True)` is the way out of it. The
return is about a third of a map's running time, and sweeping on the way back
as well gets it back. What it costs is exactly what the paragraph above says:
alternate lines are approached from opposite sides, so the leadscrew's
backlash sits between them. The encoders are on the screws and cannot see it.
Every line records `reverse` in the sidecar, so the offset can be measured --
map one box each way -- and taken out afterwards rather than guessed at.

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

import contextlib
import itertools
import json
import math
import os
import threading
import time
from collections import deque

import numpy as np

from octobee.calib import geometry as pgeom
from octobee.motion import encoder as oenc
from octobee.motion import timing as otim
from octobee.motion.kinesis import MoveAborted, StageError

AXES = ("x", "y", "z")

# Sweep velocity if nothing says otherwise, and the ceiling the GUI offers.
# Not a hardware limit -- the axes will do 10 -- a measurement one: the head is
# on the end of a cantilever, and a sweep is the one thing here that measures
# while the thing it is measuring with is moving.
DEFAULT_SPEED_MM_S = 5.0
MAX_SPEED_MM_S = 6.0

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

    def outside(self, travel_mm):
        """Where this box leaves `travel_mm` ({axis: (lo, hi)}), as sentences.

        Empty when it fits. Checked when the box is planned, so a box typed
        past the end of an axis is refused with the numbers in front of you,
        rather than by the stage itself once the probe is already moving.
        """
        out = []
        for i, a in enumerate(AXES):
            if a not in (travel_mm or {}):
                continue
            lo, hi = (float(v) for v in travel_mm[a])
            if self.lo_mm[i] < lo - 1e-9 or self.hi_mm[i] > hi + 1e-9:
                out.append(f"{a}: the box runs {self.lo_mm[i]:g}..."
                           f"{self.hi_mm[i]:g} mm, but the axis only goes "
                           f"{lo:g}...{hi:g} mm")
        return out

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
                 run_out_mm=0.0, part="box", via=(), reverse=False):
        self.sweep = sweep
        # Where to go, in order, on the way to this line's start: full
        # {axis: mm} positions. Empty when the straight move there is clear --
        # see route().
        self.via = [dict(p) for p in via]
        # Run from stop down to start rather than up. The edge trace does
        # this: it is one continuous path round the box, and a path that
        # goes round a corner has to come back down the far side. A map's
        # lines do it only when the plan was asked to sweep both ways -- see
        # "Every line still runs the same way" above.
        # start_mm < stop_mm either way; this only changes which end is
        # driven to first.
        self.reverse = bool(reverse)
        # "box" for the filled box or its six faces, "slice" for a plane
        # inside it. Stored per line so a tool reading the map back never has
        # to guess from the positions which lines are boundary and which are
        # interior -- fill_volume.py must not take a slice for a face.
        self.part = str(part)
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
        if self.reverse:
            return self.stop_mm + self.run_out_mm
        return self.start_mm - self.run_in_mm

    @property
    def to_mm(self):
        """Where the stage is actually commanded to stop, ramp included."""
        if self.reverse:
            return self.start_mm - self.run_in_mm
        return self.stop_mm + self.run_out_mm

    def approach(self):
        """The full coordinate dict for the start of this line."""
        return {**self.fixed, self.sweep: self.from_mm}

    def leave(self):
        """The full coordinate dict for where this line ends."""
        return {**self.fixed, self.sweep: self.to_mm}

    end = leave

    def to_dict(self):
        return {"sweep": self.sweep, "start_mm": self.start_mm,
                "stop_mm": self.stop_mm, "fixed": dict(self.fixed),
                "run_in_mm": self.run_in_mm, "run_out_mm": self.run_out_mm,
                "part": self.part, "reverse": self.reverse,
                "via": [dict(p) for p in self.via]}


class SweepPlan:
    """Every line to be swept, in the order they will be swept."""

    def __init__(self, volume, lines, speed_mm_s=DEFAULT_SPEED_MM_S,
                 log_hz=DEFAULT_LOG_HZ, settle_s=DEFAULT_SETTLE_S,
                 skipped=0, note="", slices=None,
                 accel_mm_s2=otim.DEFAULT_ACCEL_MM_S2, reachable=None,
                 both_ways=False):
        self.volume = volume
        # Alternate lines swept in the -ve direction: see _both_ways().
        self.both_ways = bool(both_ways)
        self.accel_mm_s2 = float(accel_mm_s2)
        # The grid the lines were cut on, kept so the one move that cannot be
        # planned in advance -- from wherever the head is parked to the first
        # line -- can be checked against it when the sweep starts. Not saved:
        # it is megabytes, and the lines and their `via` points are its result.
        self.reachable = reachable
        self.slices = dict(slices) if slices else None
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

    def entry(self, position_mm):
        """How to get from where the head is to the first line.

        Returns (waypoints, warning). The waypoints are full {axis: mm}
        positions to visit in order before the first approach; the warning is
        text to show when the move could not be checked, and "" when it was.

        Every other transit is routed when the plan is made. This one depends
        on where the rig was left, so it is worked out when the sweep starts.
        A head parked outside the planned box is outside the grid the coils
        were tested on: nothing here can say whether the way in is clear, and
        saying so is better than pretending. Raises ValueError when the head
        is inside the box and there is no clear way to the first line.
        """
        if not self.lines or self.reachable is None:
            return [], ""
        vol = self.volume
        try:
            here = np.array([float(position_mm[a]) for a in AXES])
        except (KeyError, TypeError, ValueError):
            here = np.full(3, np.nan)
        slack = vol.step_mm / 2.0
        if not (np.isfinite(here).all()
                and np.all(here >= vol.lo_mm - slack)
                and np.all(here <= vol.hi_mm + slack)):
            return [], ("the head starts outside the planned volume, so the "
                        "move to the first line is NOT checked against the "
                        "coils -- park it inside the box, or watch it go")
        via = route(vol, self.reachable, dict(zip(AXES, here)),
                    self.lines[0].approach())
        if via is None:
            raise ValueError(
                "there is no clear way from where the head is now to the "
                "first line of this plan -- the coil is between them. Move "
                "the head to open space by hand and start again.")
        return via, ""

    def duration_s(self, accel_mm_s2=None):
        """Roughly how long this takes, in seconds.

        The sweeps themselves are exact -- distance over velocity, plus the
        ramps. So is the return: these axes are capped at MAX_VEL_MM_S, so
        going back to the start of the next line costs most of what sweeping
        it did. That is the price of never reversing into a line, and it is
        included here rather than discovered overnight. A plan swept both
        ways has no return to charge, only the step to the next line.
        """
        return float(sum(self.line_durations_s(accel_mm_s2)))

    def line_durations_s(self, accel_mm_s2=None):
        """What each line costs, in seconds, in the order they are run.

        Separate from duration_s() so a run in progress can say how much is
        left: the lines are not the same length, so "lines done over lines
        planned" is a poor clock.
        """
        a = float(self.accel_mm_s2 if accel_mm_s2 is None else accel_mm_s2)
        v = otim.clamp_velocity(self.speed_mm_s)
        back = otim.clamp_velocity(otim.MAX_VEL_MM_S)
        out, prev = [], None
        for ln in self.lines:
            reach = abs(ln.to_mm - ln.from_mm)
            # Getting to the start of the line, from wherever the last one
            # ended: the axes move together, so the longest of them sets it.
            # Before the first line nobody knows where the stage is, so it is
            # charged a line's length, as a raster's return would be.
            if prev is None:
                gap = reach
            else:
                to = ln.approach()
                gap = max(abs(to[k] - prev.get(k, to[k])) for k in to)
            out.append(otim.move_time_s(reach, v, a) + self.settle_s
                       + otim.move_time_s(gap, back, a))
            prev = ln.leave()
        return out

    def reach_mm(self):
        """{axis: (lowest, highest)} the plan will command, ramps included."""
        out = {}
        for ln in self.lines:
            for where in (ln.approach(), ln.end()):
                for a, v in where.items():
                    lo, hi = out.get(a, (v, v))
                    out[a] = (min(lo, v), max(hi, v))
        return out

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
        detours = sum(1 for ln in self.lines if ln.via)
        if detours:
            bits += (f" -- {detours} move(s) between lines routed round the "
                     f"coil rather than straight")
        return f"{self.note}: {bits}" if self.note else bits

    def to_dict(self):
        return {"volume": self.volume.to_dict(),
                "speed_mm_s": self.speed_mm_s, "log_hz": self.log_hz,
                "accel_mm_s2": self.accel_mm_s2,
                "transits_checked": self.reachable is not None,
                "both_ways": self.both_ways,
                "settle_s": self.settle_s, "skipped": self.skipped,
                "note": self.note, "slices": self.slices,
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


def _node(volume, point_mm):
    """The grid node nearest a rig position, as (ix, iy, iz), kept on the grid."""
    return tuple(
        int(min(max(math.floor((float(point_mm[a]) - volume.lo_mm[i])
                               / volume.step_mm + 0.5), 0),
                volume.shape[i] - 1))
        for i, a in enumerate(AXES))


def _grid_path(reachable, start, goal):
    """Shortest path of face-adjacent reachable nodes, or None if there is none.

    Breadth first over the same grid the lines were cut on. The grid is at
    most a few hundred thousand nodes and this runs only for the handful of
    transits whose straight move is blocked, so nothing cleverer is needed.
    """
    shape = reachable.shape
    came = {start: None}
    todo = deque([start])
    while todo:
        at = todo.popleft()
        if at == goal:
            path = []
            while at is not None:
                path.append(at)
                at = came[at]
            return path[::-1]
        for k in range(3):
            for d in (-1, 1):
                nxt = (*at[:k], at[k] + d, *at[k + 1:])
                if (0 <= nxt[k] < shape[k] and nxt not in came
                        and reachable[nxt]):
                    came[nxt] = at
                    todo.append(nxt)
    return None


def route(volume, reachable, start_mm, end_mm):
    """How to move between two rig positions without crossing a coil.

    Returns the positions to visit on the way, in order, as full {axis: mm}
    dicts: [] when the direct move is clear, None when there is no way at all.

    Why this exists. The lines of a plan are cut to the space the probe body
    fits in, but a sweep is mostly NOT sweeping: after every line the head
    goes to the start of the next, and that move was never tested against
    anything. Where a line had been cut in two because it passes behind a
    winding, the move from the end of the first piece to the start of the
    second ran along the very stretch that had been cut out -- through the
    winding. The plan looked right, the preview flew only the swept lines,
    and the stages would have driven the head into the coil.

    What "clear" means here. A direct move starts every axis at once and each
    runs its own ramp, so the head does not travel a straight line; it stays
    inside the box spanned by the two ends. The direct move is therefore
    allowed only when EVERY node of that box is reachable. Otherwise the way
    round is found on the grid and returned as its corners, so each leg moves
    one axis only, along nodes that were each tested.

    The two ends may sit off the grid by a line's run-in or run-out. They are
    judged by their nearest node, and when a detour is needed it begins and
    ends on those nodes, so the off-grid part is a move of at most one ramp
    length along the line's own axis.
    """
    if reachable is None:
        return []
    a, b = _node(volume, start_mm), _node(volume, end_mm)
    if not (reachable[a] and reachable[b]):
        return None
    box = tuple(slice(min(p, q), max(p, q) + 1) for p, q in zip(a, b))
    if reachable[box].all():
        return []
    path = _grid_path(reachable, a, b)
    if path is None:
        return None
    corners = [path[0]]
    for before, at, after in zip(path, path[1:], path[2:]):
        if tuple(np.subtract(at, before)) != tuple(np.subtract(after, at)):
            corners.append(at)
    corners.append(path[-1])
    return [{ax: float(volume.nodes(ax)[i]) for ax, i in zip(AXES, node)}
            for node in corners]


def _route_lines(volume, reachable, lines):
    """Give every line the way to its start from the end of the one before.

    Returns (lines kept, how many were dropped). A line with no clear way to
    it is dropped rather than reached through a coil; the next line is then
    routed from the last one that was kept.
    """
    kept, dropped, last = [], 0, None
    for line in lines:
        if last is not None:
            via = route(volume, reachable, last.leave(), line.approach())
            if via is None:
                dropped += 1
                continue
            line.via = via
        kept.append(line)
        last = line
    return kept, dropped


def _both_ways(lines):
    """Turn rows round so each starts at the end nearest where the last ended.

    A row is every line with the same swept axis and the same two fixed
    coordinates: one line, or the several pieces a winding cut it into. A row
    taken backwards has its pieces in the opposite order too, each swept from
    its high end down. The first row runs upward. Inside a raster this comes
    out as a serpentine; between rasters it picks whichever end is closer,
    which a strict alternation would get wrong half the time.

    Only the swept axis decides: the other two travel the same distance to
    the row whichever end it is entered from.
    """
    out, at = [], None
    for _, group in itertools.groupby(
            lines, key=lambda ln: (ln.sweep, tuple(sorted(ln.fixed.items())),
                                   ln.part)):
        row = list(group)
        low = row[0].start_mm - row[0].run_in_mm
        high = row[-1].stop_mm + row[-1].run_out_mm
        here = None if at is None else at.get(row[0].sweep)
        if here is not None and abs(here - high) < abs(here - low):
            row.reverse()
            for ln in row:
                ln.reverse = True
        out.extend(row)
        at = row[-1].leave()
    return out


# One continuous path over all twelve edges of a box, as corners (x, y, z),
# each 0 for the low end of that axis and 1 for the high one.
#
# Every corner of a box has three edges, an odd number, so no path covers
# each edge exactly once -- that needs at most two odd corners and a box has
# eight. The fewest edges that must be run twice is three, and this route
# repeats exactly three (marked *): the bottom face, up one edge, the top face,
# then down, across and up the three remaining verticals.
EDGE_ROUTE = (
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0),     # bottom face
    (0, 0, 1),                                                  # up
    (1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 0, 1),                 # top face
    (1, 0, 1),                                                  # * again
    (1, 0, 0),                                                  # down
    (1, 1, 0),                                                  # * again
    (1, 1, 1),                                                  # up
    (0, 1, 1),                                                  # * again
    (0, 1, 0),                                                  # down
)


def _edge_lines(volume, reachable, span_floor):
    """The box's outline as one continuous path, corner to corner.

    Run before a long map, it drives the probe round exactly what the map is
    about to cover -- the cheapest way to find out the travel or the box was
    typed wrong. Each line starts where the last one ended, so the stage never
    goes back for a return move: the path is the outline and nothing else.
    That means half the edges are run backwards, which a map would never do
    (backlash), and is fine for a check.

    No run-up or run-down past the corners: overshooting a corner is exactly
    what a check of the travel must not do.

    Where `reachable` cuts an edge, only its clear parts are kept, and the
    path is no longer continuous there -- the stage then drives between the
    pieces as it would between any two lines. Returns (lines, skipped, cut).
    """
    lines, skipped, cut = [], 0, 0
    top = {a: volume.shape[AXES.index(a)] - 1 for a in AXES}
    for c0, c1 in itertools.pairwise(EDGE_ROUTE):
        k = next(i for i in range(3) if c0[i] != c1[i])
        axis = AXES[k]
        if top[axis] == 0:
            continue              # a flat box: this edge has no length
        idx = {AXES[i]: c0[i] * top[AXES[i]] for i in range(3) if i != k}
        fixed = {a: float(volume.nodes(a)[v]) for a, v in idx.items()}
        reverse = c1[k] < c0[k]
        along = volume.nodes(axis)
        if reachable is None:
            spans = [(0, len(along) - 1)]
        else:
            sel = [slice(None)] * 3
            for a, v in idx.items():
                sel[AXES.index(a)] = v
            spans = _runs(reachable[tuple(sel)])
            if spans != [(0, len(along) - 1)]:
                cut += 1
        if reverse:
            spans = spans[::-1]
        for s0, s1 in spans:
            start, stop = float(along[s0]), float(along[s1])
            if stop - start < span_floor - 1e-9:
                skipped += 1
                continue
            lines.append(SweepLine(axis, start, stop, fixed, 0.0, 0.0,
                                   "edge", reverse=reverse))
    return lines, skipped, cut


def _face_rasters(volume, shell):
    """(sweep axis, first stepping axis, second, restrict, part) per raster.

    Filling a volume is one raster: sweep one axis, step the other two over
    every node. Mapping only its OUTSIDE is six, one per face, and each has to
    be swept in the plane of its own face -- the two faces perpendicular to
    the usual sweep axis cannot be swept along it at all. Each entry says which
    axis runs and which two step; `restrict` maps a stepping axis to the node
    indices it is limited to, or the swept axis to the (first, last) node of
    its span, and an axis it does not name uses every node.
    """
    sweep = volume.sweep
    a, b = volume.step_axes
    if not shell:
        return [(sweep, a, b, {}, "box")]
    # Four faces parallel to the sweep axis: pin one stepping axis to each end
    # and step the other across. Then two faces perpendicular to it, swept
    # along `a` with `b` stepping, pinned at each end of the sweep axis.
    out = []
    for pinned, across in ((a, b), (b, a)):
        n = volume.shape[AXES.index(pinned)]
        for end in (0, n - 1):
            out.append((sweep, pinned, across, {pinned: [end]}, "box"))
    n = volume.shape[AXES.index(sweep)]
    for end in (0, n - 1):
        out.append((a, sweep, b, {sweep: [end]}, "box"))
    return out


def slice_planes(volume, count, inset_mm, normal="y"):
    """Where `count` planes square to `normal` sit inside the box, and how big.

    The planes are spread evenly through the box -- for three, a quarter, a
    half and three quarters of the way along `normal` -- and each is a
    rectangle smaller than the face it is parallel to by `inset_mm` on every
    side. Everything is snapped to the volume's own grid: the reachable mask
    is worked out on that grid and a line off it could not be cut by it. A
    plane that snaps onto a face would only measure the face twice, so it is
    dropped, as is a second plane snapping onto the same node.

    Returns (node indices along `normal`, {in-plane axis: (first, last) node}),
    or ([], {}) when the inset leaves no rectangle or the box is too thin to
    have an inside.
    """
    count = int(count)
    if count <= 0:
        return [], {}
    if normal not in AXES:
        raise ValueError(f"slice normal must be one of {AXES}, got {normal!r}")
    k = max(0, int(math.floor(float(inset_mm) / volume.step_mm + 0.5)))
    rect = {}
    for ax in AXES:
        if ax == normal:
            continue
        n = volume.shape[AXES.index(ax)]
        if n - 1 - 2 * k < 1:
            return [], {}
        rect[ax] = (k, n - 1 - k)
    n = volume.shape[AXES.index(normal)]
    # Halves round toward the middle, so the set stays symmetric about it:
    # three planes in 0..300 on a 10 mm grid land at 80, 150, 220 rather
    # than 80, 150, 230.
    mid = (n - 1) / 2.0
    at = sorted({int(math.floor(t + 0.5)) if t <= mid
                 else int(math.ceil(t - 0.5))
                 for t in (j * (n - 1) / (count + 1)
                           for j in range(1, count + 1))})
    at = [i for i in at if 0 < i < n - 1]
    if not at:
        return [], {}
    return at, rect


def _slice_rasters(volume, count, inset_mm, normal):
    """The rasters for the planes slice_planes() places, one per plane.

    Each plane is swept along the volume's own sweep axis where that lies in
    the plane, and along the first in-plane axis otherwise; the other in-plane
    axis steps.
    """
    at, rect = slice_planes(volume, count, inset_mm, normal)
    if not at:
        return []
    in_plane = [a for a in AXES if a != normal]
    sweep = volume.sweep if volume.sweep in in_plane else in_plane[0]
    other = in_plane[1] if sweep == in_plane[0] else in_plane[0]
    o0, o1 = rect[other]
    return [(sweep, normal, other,
             {normal: [i], other: list(range(o0, o1 + 1)), sweep: rect[sweep]},
             "slice")
            for i in at]


def plan(volume, reachable=None, speed_mm_s=DEFAULT_SPEED_MM_S,
         log_hz=DEFAULT_LOG_HZ, settle_s=DEFAULT_SETTLE_S,
         accel_mm_s2=otim.DEFAULT_ACCEL_MM_S2,
         travel_mm=None, min_span_mm=None, shell=False, slices=0,
         slice_inset_mm=0.0, slice_normal="y", edges=False,
         both_ways=False):
    """Turn a Volume into the lines that will actually be swept.

    `reachable` is a boolean array of `volume.shape` -- machine.reachable_grid's
    answer -- saying which grid nodes the probe body can occupy. Where it is
    given, each line is cut into the maximal clear runs along the swept axis, so
    a line that passes behind a winding becomes two shorter lines with the coil
    between them rather than one line through it. That is what makes the mapped
    region the shape of the space that is actually there, instead of the largest
    box that happens to fit in it.

    The moves BETWEEN lines are held to the same grid: each line carries the
    way to its start from the end of the one before (see route()), a line
    with no clear way to it is dropped, and a run-in or run-out is not taken
    where the node it would run toward is one the probe cannot occupy.

    `accel_mm_s2` must be the acceleration the axes really run at. It sets
    the run-in and run-out lengths, and too large a value puts the tail of
    the ramp inside the span that is flagged as measured at constant speed.

    `shell` maps the OUTSIDE of the box instead of filling it: the six faces,
    each swept in its own plane. For a field in a current-free region that is
    not a poor relation of the filled version -- the boundary values determine
    the interior -- and it is five or six times fewer lines, which is the
    difference between an evening and a week.

    `slices` adds that many planes INSIDE the box, square to `slice_normal`
    (y by default, so they lie in x-z), each inset from the box by
    `slice_inset_mm` on every side -- see slice_planes(). With `shell` that is
    the box's surface plus a few cuts through it: the faces fix the interior
    of a current-free field, and the cuts are measured values of that
    interior to check the reconstruction against. Their lines are tagged
    part="slice" so nothing reading the map back takes them for faces.

    `edges` traces only the box's outline, ignoring `shell` and `slices`: one
    continuous path over all twelve edges (see EDGE_ROUTE), to check the
    travel and the box are what was meant before committing hours to them.
    No settle at the corners and no overshoot past them. Its lines are
    tagged part="edge".

    `both_ways` sweeps a line on the way back as well, instead of returning
    empty to start every line from its low end (see _both_ways()). It takes
    the return moves out of the map, about a third of its running time, and
    puts the leadscrew's backlash between lines swept in opposite directions.
    Off unless asked for; each line's `reverse` says which way it went. The
    edge trace has its own route and ignores it.

    `travel_mm` is {axis: (lo, hi)}; the run-up and run-down are fitted into
    whatever it leaves outside the box, and silently shortened where it does
    not. A line with no room to ramp is still swept -- it just measures its own
    first millimetres at less than full speed, which the position log records
    faithfully.
    """
    ramp = otim.ramp_distance_mm(speed_mm_s, accel_mm_s2)
    span_floor = (volume.step_mm if min_span_mm is None else float(min_span_mm))
    reachable = None if reachable is None else np.asarray(reachable)

    edge_cut = 0
    if edges:
        rasters = []
        slices = 0
        settle_s = 0.0
    else:
        rasters = (_face_rasters(volume, shell)
                   + _slice_rasters(volume, slices, slice_inset_mm,
                                    slice_normal))
    lines, skipped, seen = [], 0, set()
    if edges:
        lines, skipped, edge_cut = _edge_lines(volume, reachable, span_floor)
    for axis, first_ax, second_ax, restrict, part in rasters:
        along = volume.nodes(axis)
        a0, a1 = restrict.get(axis, (0, len(along) - 1))
        firsts = restrict.get(first_ax,
                              range(volume.shape[AXES.index(first_ax)]))
        seconds = restrict.get(second_ax,
                               range(volume.shape[AXES.index(second_ax)]))
        for i1 in firsts:
            for i2 in seconds:
                idx = {first_ax: i1, second_ax: i2}
                fixed = {k: float(volume.nodes(k)[v]) for k, v in idx.items()}
                key = (axis, tuple(sorted(idx.items())))
                if key in seen:
                    continue          # the faces share their edges
                seen.add(key)
                if reachable is None:
                    spans, clear = [(a0, a1)], None
                else:
                    sel = [slice(None)] * 3
                    for k, v in idx.items():
                        sel[AXES.index(k)] = v
                    clear = reachable[tuple(sel)]      # along the whole axis
                    spans = [(a0 + s0, a0 + s1)
                             for s0, s1 in _runs(clear[a0:a1 + 1])]
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
                    # No ramp toward a node the probe cannot occupy. A line
                    # cut short by a winding ends at the last clear node, and
                    # running on past it is running at the winding.
                    if clear is not None:
                        if start_i > 0 and not clear[start_i - 1]:
                            run_in = 0.0
                        if stop_i < len(clear) - 1 and not clear[stop_i + 1]:
                            run_out = 0.0
                    lines.append(SweepLine(axis, start, stop, fixed,
                                           run_in, run_out, part))
    both_ways = bool(both_ways) and not edges
    if both_ways:
        lines = _both_ways(lines)       # before routing: it moves the ends
    if reachable is not None:
        lines, unroutable = _route_lines(volume, reachable, lines)
        skipped += unroutable
    note = ("the outline of the box, one continuous path" if edges
            else "the outside of the box" if shell
            else "the whole box, filled")
    if edge_cut:
        note += (f" -- broken where the coils cut {edge_cut} edge(s), so the "
                 f"stage drives between the pieces")
    slice_doc = None
    if int(slices) > 0:
        at, rect = slice_planes(volume, slices, slice_inset_mm, slice_normal)
        in_plane = "-".join(a for a in AXES if a != slice_normal)
        if at:
            slice_doc = {
                "normal": slice_normal, "plane": in_plane,
                "at_mm": [float(volume.nodes(slice_normal)[i]) for i in at],
                "rect_mm": {ax: [float(volume.nodes(ax)[i0]),
                                 float(volume.nodes(ax)[i1])]
                            for ax, (i0, i1) in rect.items()},
                "inset_mm": float(next(iter(rect.values()))[0]
                                  * volume.step_mm)}
            dims = " x ".join(f"{v[1] - v[0]:g}"
                              for v in slice_doc["rect_mm"].values())
            note += (f" plus {len(at)} {in_plane} slice(s) of {dims} mm at "
                     f"{slice_normal} = "
                     + ", ".join(f"{v:g}" for v in slice_doc["at_mm"]) + " mm")
        else:
            note += (f" (no {in_plane} slices: the inset or the box leaves no "
                     f"room for them on a {volume.step_mm:g} mm grid)")
    note += (", cut to the space the probe body actually fits in"
             if reachable is not None else ", coils ignored")
    if both_ways:
        note += ", swept both ways"
    return SweepPlan(volume, lines, speed_mm_s, log_hz, settle_s, skipped, note,
                     slices=slice_doc, accel_mm_s2=accel_mm_s2,
                     reachable=reachable, both_ways=both_ways)


class Aborted(RuntimeError):
    """The caller asked the sweep to stop."""


# How often the sweeping axis is asked where it is, seconds. The DLL answers
# out of its own polling cache, so this is not a USB round trip per call; 20 ms
# at 10 mm/s is a poll every 0.2 mm, which is finer than the leadscrew is
# accurate and leaves the interpolation nothing to invent.
POLL_S = 0.02

# How long to let the controller admit it has started moving before giving up
# on ever seeing it, seconds.
#
# Stage.moving reads ISC_GetStatusBits, which does NOT go to the wire -- it
# returns the Kinesis DLL's cached copy, refreshed by its own polling thread
# every 100 ms (ISC_StartPolling(serial, 100) in Stage.open). So for up to
# 100 ms after a move is commanded the status still reads "not moving", left
# over from before the command.
#
# Waiting for the move to STOP without first watching it START therefore ends
# the line on the first poll, POLL_S = 20 ms in, while the carriage is still
# accelerating -- and since the logging window is exactly that loop, the whole
# 250 mm sweep then happens with nothing recording. It does not look like a
# failure: every line "completes", no error is raised, and the map comes back
# with a handful of rows out of two million. Measured 2026-09-24: 262 lines
# swept in 14 minutes instead of 5.6 hours, 5 rows logged.
#
# 5x the DLL's own 100 ms cache period, so a slow status refresh cannot end a
# line early. A move that genuinely never starts costs this much and then
# falls through to the loop below, which is the behaviour without it.
MOVE_START_S = 0.5

# How close to its commanded end a line has to finish to count as swept, mm,
# and how long to let the controller's cached position catch up before
# deciding it did not. The cache is the same 100 ms one MOVE_START_S is about.
ARRIVE_TOL_MM = 0.05
ARRIVE_WAIT_S = 0.5


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

    def _go(self, where, should_abort, settle_s=0.0):
        """A positioning move that an abort can end. True if it arrived."""
        # Passed only when there is one, so a rig object that predates the
        # argument still works.
        extra = {"should_abort": should_abort} if should_abort else {}
        try:
            self.stages.move_to(settle_s=settle_s, **extra, **where)
        except MoveAborted:
            self.aborted = True
            return False
        return True

    def _arrived(self, st, target_mm):
        """Did the line end where it was sent? (ok, where it is)."""
        deadline = time.monotonic() + ARRIVE_WAIT_S
        while True:
            at = float(st.position_mm)
            if abs(at - target_mm) <= ARRIVE_TOL_MM:
                return True, at
            if time.monotonic() > deadline:
                return False, at
            time.sleep(POLL_S)

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

    def _already_at(self, coords, tol_mm=0.002):
        """True if every named axis is within `tol_mm` of `coords` now.

        2 um: well under the step anything is planned on and well over the
        ~1 um the controller settles to. A failed read answers False, which
        only means the ordinary move is made.
        """
        try:
            where = self.stages.position()
            return all(abs(float(where[k]) - float(v)) <= tol_mm
                       for k, v in coords.items())
        except Exception:
            return False

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

        # Every position the plan will command, against what each axis allows,
        # before anything moves. Each move checks itself too, but that check
        # fires with the probe already part way round: the plan has driven to
        # its first corner and then finds the second one is off the end.
        outside = []
        for a, (lo, hi) in self.plan.reach_mm().items():
            if a not in self.stages.names:
                continue
            lim_lo, lim_hi = (float(v) for v in self.stages[a].limit_mm)
            if lo < lim_lo - 1e-9 or hi > lim_hi + 1e-9:
                outside.append(f"{a} needs {lo:g}...{hi:g} mm, the axis "
                               f"allows {lim_lo:g}...{lim_hi:g} mm")
        if outside:
            raise ValueError(
                "the sweep leaves what the axes are allowed to use, so "
                "nothing was moved: " + "; ".join(outside)
                + ". Tick 'the whole travel', or bring 'from' + 'size' "
                  "inside it.")

        # What each axis is normally set to, so it can be put back however the
        # sweep ends. Read once: these are the profiles the rig arrived with.
        normal = {a: self.stages[a].vel_params for a in AXES
                  if a in self.stages.names}
        speed = otim.clamp_velocity(self.plan.speed_mm_s)
        # The way in. Worked out before anything moves, so a head with no
        # clear way to the first line is refused rather than discovered.
        entry, unchecked = self.plan.entry(self.stages.position())
        if unchecked:
            self.log_fn(f"sweep: WARNING -- {unchecked}")
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
                # 1. get there at the ordinary speed, by the way that was
                # checked, and let it stop ringing. Every one of these moves
                # can be ended by an abort: together they are half the sweep's
                # running time, and a stop that waits for one to finish is
                # most of a minute late.
                # Unless it is there already: on a continuous path (the edge
                # trace) every line starts where the last one ended, and a
                # zero-length move still costs a wait for "stopped" at every
                # corner. Judged from where the stage IS, not from where the
                # last line was meant to end -- a line stopped short by its
                # timeout ends somewhere else, and then the move is needed.
                st.set_vel_params(*normal[axis])
                way = entry if i == 0 else line.via
                for point in way:
                    if not self._go(point, should_abort):
                        break
                if self.aborted:
                    break
                if not self._already_at(line.approach()):
                    if not self._go(line.approach(), should_abort,
                                    self.plan.settle_s):
                        break
                elif way and self.plan.settle_s > 0:
                    # A detour can end exactly on the line's start -- a line
                    # cut short by a winding has no run-in -- and then the
                    # move above is skipped, and its settle with it. The head
                    # has just arrived at full speed all the same.
                    time.sleep(self.plan.settle_s)
                if should_abort and should_abort():
                    self.aborted = True
                    break

                # 2. sweep, logging position the whole way.
                doc = line.to_dict()
                doc["completed"] = False
                self.log.lines.append(doc)
                st.set_vel_params(speed, accel)
                self.line_index = i
                # Where this line's counting starts from, taken standing
                # still -- the only moment the controller's USB latency costs
                # nothing, and the anchor everything the encoder says after it
                # is measured against. Counts first, then the poll that the
                # fallback interpolates from, both with nothing moving.
                if self.log.encoders:
                    datum = oenc.datum_from(
                        self._fresh_counts(), self.log.encoders, self.stages,
                        on_error=self.log_fn)
                    self.log.add_datum(i, datum)
                    lost = sorted(a for a in self.log.encoders.axes
                                  if a not in datum)
                    if lost:
                        # Said here, and recorded per line in the sidecar: the
                        # map otherwise reads "encoder" for the whole axis
                        # while this line's positions are interpolated polls.
                        self.log_fn(
                            f"sweep: line {i + 1} has no encoder anchor for "
                            f"{', '.join(lost)} -- its position falls back to "
                            f"the controller polls")
                self._poll(i)
                self.logging = True
                timed_out = False
                try:
                    st.move_to(line.to_mm, wait=False)
                    # Watch it start before waiting for it to stop. The status
                    # bit is up to 100 ms stale (see MOVE_START_S), so a loop
                    # that only waits for "stopped" reads the pre-move value
                    # and closes this line's logging window 20 ms in, with the
                    # sweep still to come. Poll throughout, so the position
                    # record covers the run-up rather than starting late.
                    began = time.monotonic() + MOVE_START_S
                    while not st.moving and time.monotonic() < began:
                        self._poll(i)
                        if should_abort and should_abort():
                            self.aborted = True
                            st.stop()
                            break
                        time.sleep(POLL_S)
                    deadline = time.monotonic() + otim.move_timeout_s(
                        abs(line.to_mm - line.from_mm), speed, accel)
                    while not self.aborted:
                        self._poll(i)
                        if not st.moving:
                            break
                        if self.stages.interlock.tripped:
                            # Latched while this line is still running: the
                            # stop that tripped it did not reach this axis, or
                            # lost a race with the move. Stop it now rather
                            # than at the end of the line.
                            with contextlib.suppress(Exception):
                                st.stop(immediate=True)
                            break
                        if time.monotonic() > deadline:
                            timed_out = True
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
                # A line ending is not a line having been swept. The loop
                # above ends just the same on an emergency stop, a stall, a
                # timeout or a move that never started, and each of those
                # used to be counted as a completed line and followed by the
                # next approach move.
                self.stages.interlock.require_clear(f"sweeping line {i + 1}")
                if self.aborted:
                    break
                if timed_out:
                    with contextlib.suppress(Exception):
                        st.distrust(f"{axis} did not finish sweeping line "
                                    f"{i + 1} in time, so it may have stalled")
                    raise StageError(
                        f"{axis}: line {i + 1} did not finish in time and was "
                        f"stopped -- the sweep ends here")
                ok, at = self._arrived(st, line.to_mm)
                if not ok:
                    raise StageError(
                        f"{axis}: line {i + 1} ended at {at:.3f} mm, not the "
                        f"{line.to_mm:.3f} mm it was sent to -- the move did "
                        f"not run, or did not finish. The sweep ends here.")
                doc["completed"] = True
                self.lines_done = i + 1
                # Between lines, standing still: the only place a part can be
                # cut without splitting a line between files.
                self.log.checkpoint()
                if on_progress:
                    on_progress(i + 1, len(self.plan.lines), line)
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


# A crash-safe .csv part is written once this many rows have piled up since
# the last one -- about 3.3 minutes of sweeping at 500 Hz, 17 at 100 Hz. Cut at
# the end of a line, never inside one.
CHECKPOINT_ROWS = 100_000


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
        self._parts = None            # set by keep_parts()
        self.parts_written = []

    # ---- crash-safe parts --------------------------------------------
    def keep_parts(self, directory, log_hz, placement=None,
                   every_rows=CHECKPOINT_ROWS, log_fn=print):
        """Write the sweep out as .csv parts while it runs.

        Everything else lives in memory until the end, so a hard crash, a
        power cut or a killed window would lose the lot. With this on, at most
        the last `every_rows` rows (plus the line in progress) are lost.

        Each part is complete on its own -- positions included -- because the
        polls and encoder datum of every line it touches go into it. Parts are
        written from a background thread so the stages do not wait on the
        disk, and under a temporary name first, so a part that exists is a
        part that was finished.
        """
        os.makedirs(directory, exist_ok=True)
        self._parts = {"dir": directory, "hz": float(log_hz),
                       "placement": placement, "every": int(every_rows),
                       "log": log_fn, "from_block": 0, "n": 0,
                       "thread": None}
        info = dict(self.meta)
        info.update({"log_hz": float(log_hz), "every_rows": int(every_rows),
                     "encoders": self.encoders.to_dict(),
                     "note": "crash-safe parts of a volume sweep; the .npz "
                             "written at the end holds the same rows"})
        with open(os.path.join(directory, "info.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(info, fh, indent=2, default=str)
            fh.write("\n")

    def checkpoint(self, force=False):
        """Write the rows logged since the last part, if there are enough.

        Called by the runner between lines. Returns the path of the part being
        written, or None. `force` writes whatever is pending however little.
        """
        st = self._parts
        if st is None:
            return None
        # Blocks keep arriving: fix the end now. add_block appends to four
        # lists one after another from another thread, so the end is the
        # shortest of them -- a block caught half appended waits for the next
        # part rather than going into this one with its time missing.
        n = min(len(self._b), len(self._c), len(self._t0), len(self._line_of))
        k0 = st["from_block"]
        pending = sum(len(b) for b in self._b[k0:n])
        if not pending or (pending < st["every"] and not force):
            return None

        part = SweepLog(self.geom, {}, self.encoders)
        part._b = self._b[k0:n]
        part._c = self._c[k0:n]
        part._t0 = self._t0[k0:n]
        part._line_of = self._line_of[k0:n]
        touched = set(part._line_of)
        npoll = min(len(self.poll_t_s), len(self.poll_mm), len(self.poll_line))
        keep = [j for j in range(npoll) if self.poll_line[j] in touched]
        part.poll_t_s = [self.poll_t_s[j] for j in keep]
        part.poll_mm = [self.poll_mm[j] for j in keep]
        part.poll_line = [self.poll_line[j] for j in keep]
        part.lines = list(self.lines)
        part.datum = {k: v for k, v in self.datum.items() if k in touched}
        part.n_rows = pending

        st["from_block"] = n
        st["n"] += 1
        path = os.path.join(st["dir"], f"part_{st['n']:04d}.csv")
        self.finish_parts()             # one writer at a time, in order
        th = threading.Thread(target=self._write_part, args=(part, path),
                              name="sweep-part", daemon=True)
        st["thread"] = th
        th.start()
        return path

    def _write_part(self, part, path):
        st = self._parts
        try:
            t_s, pos, machine, b_mt, line_index, in_span, counts = \
                part.assemble(st["hz"], st["placement"])
            names = (["t_s", "line", "in_span", "x_mm", "y_mm", "z_mm",
                      "machine_x_mm", "machine_y_mm", "machine_z_mm"]
                     + [f"s{s + 1:02d}_b{c}_mT"
                        for s in range(pgeom.N_SENSORS) for c in "xyz"]
                     + [f"counts_{j}" for j in range(counts.shape[1])])
            table = np.column_stack([
                t_s, line_index, in_span.astype(int), pos, machine,
                b_mt.reshape(len(b_mt), -1).astype(np.float64), counts])
            fmt = (["%.6f", "%d", "%d"] + ["%.5f"] * 6
                   + ["%.7g"] * (3 * pgeom.N_SENSORS)
                   + ["%.0f"] * counts.shape[1])
            tmp = path + ".tmp"
            np.savetxt(tmp, table, fmt=fmt, delimiter=",",
                       header=",".join(names), comments="")
            os.replace(tmp, path)
            self.parts_written.append(path)
            st["log"](f"volume map: saved {len(t_s):,} rows so far -> {path}")
        except Exception as exc:        # a failed part must not end the sweep
            st["log"](f"WARNING: volume map: could not write {path} ({exc}) "
                      f"-- the sweep carries on, the final save is unaffected")

    def finish_parts(self, timeout_s=120.0):
        """Wait for a part still being written."""
        th = None if self._parts is None else self._parts["thread"]
        if th is not None:
            th.join(timeout_s)

    def drop_parts(self):
        """Delete the parts, once the full map is safely on disk."""
        if self._parts is None:
            return
        self.finish_parts()
        d = self._parts["dir"]
        if os.path.isdir(d):
            for name in os.listdir(d):
                if name.startswith("part_") or name == "info.json":
                    # A part open in a spreadsheet cannot be removed on
                    # Windows. The map is already saved; a leftover file is
                    # not a reason to fail the handler that says so.
                    with contextlib.suppress(OSError):
                        os.remove(os.path.join(d, name))
            # Fails if something else is in there, which is then left.
            with contextlib.suppress(OSError):
                os.rmdir(d)
        self.parts_written = []

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

    def add_datum(self, line_index, datum):
        """Where this line's counting started from.

        `datum` is {axis: (counts, mm)} as encoder.datum_from builds it -- the
        same rule the live recorder anchors by, including its refusal to
        anchor an axis whose counter is not trusted. Built by the caller
        rather than here because it needs the stages, and this module does not
        otherwise.

        Per line rather than once per sweep. The counter would carry across
        perfectly well; re-anchoring costs one standstill reading and means a
        line is still placed correctly if a count is dropped somewhere in the
        hours before it.
        """
        if datum:
            self.datum[int(line_index)] = dict(datum)

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
            # flange_path_mm, which subtracts the stage zero -- the same
            # arithmetic that drew the volume and tested it against the coils.
            # to_machine(pos, None) left the stage zero out, so once "Stage
            # zero is here" had been pressed this column sat displaced from
            # everything else in the machine frame.
            machine = placement.flange_path_mm(np.nan_to_num(pos))
            machine[~np.isfinite(pos).all(axis=1)] = np.nan
        return t_s, pos, machine, b_mt, line_index, in_span, counts

    def pos_source(self):
        """Which axes' positions came from the encoders, and which from polls."""
        return {a: ("encoder, on the ADC clock" if a in self.encoders
                    else "controller poll, interpolated onto the sample clock")
                for a in AXES}

    def fallback_lines(self):
        """{axis: [line indices]} where an encoder axis had to use the polls.

        pos_source() speaks for the axis as a whole. A line with no anchor --
        the counts had not arrived, or the axis was not trusted at that moment
        -- is positioned from the polls all the same, and this is the list of
        which, so the map does not claim encoder timing for rows without it.
        """
        ran = sorted(set(self._line_of))
        out = {}
        for a in AXES:
            if a not in self.encoders:
                continue
            missing = [li for li in ran if a not in self.datum.get(li, {})]
            if missing:
                out[a] = missing
        return out

    def save(self, path, log_hz, placement=None, sync_note=""):
        """Write <path>.npz (bulk) and <path>.json (provenance)."""
        base = os.path.splitext(path)[0]
        d = os.path.dirname(os.path.abspath(base))
        if d:
            os.makedirs(d, exist_ok=True)
        t_s, pos, machine, b_mt, line_index, in_span, counts = self.assemble(
            log_hz, placement)
        # Both files go to a temporary name and are renamed into place. This
        # is hours of an unattended run held only in memory; a crash or a full
        # disk half way through writing must not leave a truncated file under
        # the name the finished one would have had.
        np.savez_compressed(
            base + ".part.npz",
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
            "position_fallback_lines": self.fallback_lines(),
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
        with open(base + ".part.json", "w", encoding="utf-8") as fh:
            json.dump(side, fh, indent=2, default=str)
            fh.write("\n")
        os.replace(base + ".part.npz", base + ".npz")
        os.replace(base + ".part.json", base + ".json")
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
