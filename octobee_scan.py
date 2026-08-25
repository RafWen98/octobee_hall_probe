#!/usr/bin/env python3
"""
octobee_scan.py -- map a field over a volume: move the stages, stop, average,
repeat.

The whole method in one paragraph
---------------------------------
octobee_posecap.py already established that a STATIONARY average beats a
moving one by an order of magnitude: the sensor noise is white and falls as
1/sqrt(N), so a pose you can sit on for 20 s at the carriers' own 200 kSPS
reaches 0.020 uT where a continuous sweep off the reduced-rate live stream
manages 0.235 uT. A motorised scan is that same argument applied to position
instead of roll -- the stage stops, the head settles, the ADC averages, and
nothing about the measurement depends on knowing where the carriage was at
some particular microsecond. That is why this scans point-to-point rather
than on the fly, and it is also why the stage's USB position readout is good
enough: a reading with tens of milliseconds of latency is exact when the
thing it describes is not moving.

Every point is approached from the same direction
-------------------------------------------------
The obvious raster is a serpentine -- go right along one row, left along the
next, never waste a travel. It is the wrong choice here. The LTS300C drives a
leadscrew and has no linear scale on the carriage, so reversing direction
costs backlash: the motor turns and the platform does not, until the nut takes
up. Approaching alternate rows from opposite directions therefore stamps a
fixed offset into every other row of your map, which looks exactly like a real
field structure with the periodicity of the raster.

So the scan is a plain raster: each row is traversed in the same direction,
and the fast axis returns to the start of the next row rather than reversing
into it. It costs travel time and buys a map without a comb in it. The
controller's own backlash compensation is left in place on top of that (see
Stage.backlash_mm) -- belt and braces, since it also covers the return move.

What "position" means in the output
-----------------------------------
Two arrays are stored, and they are not the same thing:

    pos_cmd     where the scan asked the stage to go
    pos_mm      where the stage says it ended up, read after the move settled

They normally agree to well under a micrometre. When they do not, something
stalled or hit a limit, and having both is the difference between noticing
that and quietly folding a bad point into a field map. Neither is a *measured*
carriage position -- see octobee_stage.py on why absolute accuracy is ~47 um
without Thorlabs' per-serial calibration files, and ~5 um with them.

Usage
-----
    python octobee_scan.py --x 0:100:10 --y 0:100:10 --seconds 5
    python octobee_scan.py --x 0:50:5 --z 10:60:5 --seconds 20 --out captures/map1

Each axis is START:STOP:STEP in millimetres, stop inclusive. Axes you do not
name are left where they are and recorded as constant.
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

import octobee as ob
import octobee_calibration as ocal
import octobee_posecap as opcap
import octobee_stage as ostage

N_SENSORS = ocal.N_SENSORS

# Consecutive failed points before a scan gives up. One point can fail for a
# reason that clears -- a dropped socket, a carrier still releasing the stream.
# Three in a row is something that is not going to clear on its own, and
# grinding through 400 more points to find that out wastes the afternoon.
MAX_CONSECUTIVE_FAILURES = 3

# How long to wait after the last axis reports "not moving" before trusting
# the head to be still. The stages report motion complete when the CONTROLLER
# has finished its profile, which is not the same as the mechanics having
# stopped ringing -- and the probe is on the end of a cantilever, so it rings
# for longer than the stage does. 0.5 s is a starting point, not a measurement:
# see --settle-scan, which measures the real number on your rig.
DEFAULT_SETTLE_S = 0.5


def parse_axis_spec(spec):
    """'0:100:10' -> array of positions, stop inclusive.

    Inclusive because a scan from 0 to 100 that silently stops at 90 is a bug
    report waiting to happen, and because the end of a travel range is exactly
    where you want a sample.
    """
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"axis spec must be START:STOP:STEP, got {spec!r}")
    start, stop, step = (float(p) for p in parts)
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    if stop < start:
        raise ValueError(f"stop ({stop}) is before start ({start}); the scan "
                         f"always runs in the +ve direction so that every "
                         f"point is approached the same way")
    n = int(round((stop - start) / step))
    pts = start + step * np.arange(n + 1)
    # Floating point can put the last point a hair past stop; clamp rather
    # than drop it, since dropping it silently shortens the scan.
    if pts[-1] > stop + 1e-9:
        pts = pts[:-1]
    return pts


class ScanGrid:
    """The set of points to visit, in the order they will be visited.

    axes    dict of axis name -> 1-D array of positions, in mm.

    Ordering: the LAST axis named varies fastest. Name the cheapest axis to
    move last if you care about wall-clock; name the one whose backlash you
    trust least last, if you care about the map.
    """

    def __init__(self, axes):
        self.axes = {k: np.asarray(v, float) for k, v in axes.items()}
        if not self.axes:
            raise ValueError("a scan needs at least one axis")

    @property
    def names(self):
        return list(self.axes)

    @property
    def shape(self):
        return tuple(len(v) for v in self.axes.values())

    def __len__(self):
        n = 1
        for v in self.axes.values():
            n *= len(v)
        return n

    def points(self):
        """Yield dicts of axis -> mm, raster order, never reversing."""
        for combo in itertools.product(*self.axes.values()):
            yield dict(zip(self.axes.keys(), combo))

    def describe(self):
        bits = []
        for name, v in self.axes.items():
            if len(v) == 1:
                bits.append(f"{name}={v[0]:g}")
            else:
                bits.append(f"{name} {v[0]:g}..{v[-1]:g} mm "
                            f"step {v[1] - v[0]:g} ({len(v)} pts)")
        return "; ".join(bits)


class FieldMap:
    """A completed (or partial) scan.

    b_mt is UNCORRECTED millitesla, exactly as capture_pose returns it: VCM
    subtracted, divided by the nominal V/T for the configured range, no tare,
    no trim, no matrix. Same convention as RollSweep, and for the same reason
    -- the corrections are a separate, revisable step, and baking them into
    the stored data makes a scan unrepeatable the moment the calibration
    changes.
    """

    def __init__(self, pos_mm, pos_cmd, b_mt, axes, meta=None, stats=None):
        self.pos_mm = np.asarray(pos_mm, float)
        self.pos_cmd = np.asarray(pos_cmd, float)
        self.b_mt = np.asarray(b_mt, float)
        self.axes = list(axes)
        self.meta = dict(meta or {})
        self.stats = list(stats or [])

    def __len__(self):
        return self.b_mt.shape[0]

    def save(self, path):
        """Write <path>.npz (bulk) and <path>.json (provenance)."""
        base = os.path.splitext(path)[0]
        d = os.path.dirname(os.path.abspath(base))
        if d:
            os.makedirs(d, exist_ok=True)
        np.savez_compressed(
            base + ".npz",
            pos_mm=self.pos_mm.astype(np.float64),
            pos_cmd=self.pos_cmd.astype(np.float64),
            b_mt=self.b_mt.astype(np.float32),
            sem_ut=np.array([s.get("sem_ut", np.nan) for s in self.stats]),
            lost=np.array([s.get("lost", 0) for s in self.stats]),
        )
        side = dict(self.meta)
        side["axes"] = self.axes
        side["n_points"] = len(self)
        with open(base + ".json", "w", encoding="utf-8") as fh:
            json.dump(side, fh, indent=2, default=str)
            fh.write("\n")
        return base + ".npz"

    @classmethod
    def load(cls, path):
        base = os.path.splitext(path)[0]
        with np.load(base + ".npz") as z:
            pos_mm, pos_cmd, b_mt = z["pos_mm"], z["pos_cmd"], z["b_mt"]
            stats = [{"sem_ut": float(s), "lost": int(n)}
                     for s, n in zip(z.get("sem_ut", []), z.get("lost", []))]
        side = {}
        if os.path.exists(base + ".json"):
            with open(base + ".json", encoding="utf-8") as fh:
                side = json.load(fh)
        return cls(pos_mm, pos_cmd, b_mt,
                   side.get("axes", []), meta=side, stats=stats)


class Aborted(RuntimeError):
    """The caller asked the scan to stop."""


def measure_settle(stages, axis, distance_mm, hosts, cal, probe_s=0.25,
                   n_probes=12, log=print):
    """How long the head really takes to stop ringing after a move.

    Moves the axis, then takes a run of short captures and watches the field
    stop changing. Returns the elapsed time at which successive captures agree
    to within their own noise -- i.e. the settle time this rig actually needs,
    as opposed to the 0.5 s guess in DEFAULT_SETTLE_S.

    Worth running once per mechanical configuration. A cantilevered probe on a
    long Z arm can ring for seconds, and a settle time that is too short does
    not look like an error -- it looks like a field gradient.

    RESOLUTION LIMIT: each probe is a full capture_pose(), which opens fresh
    sockets to both carriers and decodes, so probes land seconds apart however
    small `probe_s` is. That is fine for a rig that rings for seconds and
    useless below about two: a fast rig simply reports "settled at the first
    probe", which is a floor, not a measurement. Read the result as an upper
    bound, and do not conclude from it that DEFAULT_SETTLE_S is too generous.
    """
    st = stages[axis]
    st.move_by(distance_mm)
    t0 = time.monotonic()
    trace = []
    for _ in range(n_probes):
        row, _stats = opcap.capture_pose(hosts, probe_s, cal, chunk_s=probe_s,
                                         drain_s=0.0)
        trace.append((time.monotonic() - t0, np.linalg.norm(row, axis=1).mean()))
        log(f"  t={trace[-1][0]:5.2f} s  |B| mean = {trace[-1][1] * 1e3:9.3f} uT")
    final = trace[-1][1]
    spread = np.std([v for _, v in trace[-3:]])
    for t, v in trace:
        if abs(v - final) <= 2 * spread:
            return t, trace
    return trace[-1][0], trace


def run_scan(hosts, stages, grid, seconds, cal, settle_s=DEFAULT_SETTLE_S,
             chunk_s=None, drain_s=0.0, progress=None, should_abort=None,
             log=print):
    """Visit every point in `grid`, averaging `seconds` at each.

    hosts        the carriers, as octobee uses them
    stages       an OPEN octobee_stage.StageSet
    grid         ScanGrid
    cal          octobee_cal.Calibration, for the V/T scaling
    progress     callback(i, n, point, row, stats) after each point
    should_abort callback() -> bool, polled before each move and each capture

    The live stream must already be released and the carriers back on their own
    clock before calling this -- capture_pose takes over the stream itself, and
    a scan that quietly ran at the reduced live rate would produce a map that
    looks fine and is four times noisier than it should be.

    Returns a FieldMap. On abort OR on failure, returns the points completed so
    far rather than nothing: a partial map of a long scan is still worth
    having, and at the settings this module recommends a scan is measured in
    hours. A point that raises is logged, recorded in meta["failures"], and
    skipped; MAX_CONSECUTIVE_FAILURES in a row ends the scan, still with the
    map. The only things that raise out of here are the up-front guards below,
    which fire before any measuring has been done and so cost nothing.
    """
    unknown = set(grid.names) - set(stages.names)
    if unknown:
        raise ostage.StageError(
            f"scan names axes the stage set does not have: "
            f"{', '.join(sorted(unknown))} (have: {', '.join(stages.names)})")
    stages.interlock.require_clear("a field map")
    unhomed = [(n, stages[n].distrust_reason) for n in grid.names
               if not stages[n].position_trusted]
    if unhomed:
        raise ostage.StageError(
            "these axes' position counters cannot be believed, so the map "
            "would have no origin -- home them first: "
            + "; ".join(f"{n} ({why})" for n, why in unhomed))
    outside = []
    for name, pts in grid.axes.items():
        lo, hi = stages[name].limit_mm
        if not (lo <= float(pts.min()) and float(pts.max()) <= hi):
            outside.append(f"{name} asks for {pts.min():g}..{pts.max():g} mm, "
                           f"allowed {lo:g}..{hi:g} mm")
    if outside:
        # Up front rather than at the point that leaves the envelope. Each
        # move checks itself, but discovering it two hours into a six-hour
        # raster means the map is already half-taken and the operator has
        # already gone away.
        raise ostage.StageError(
            "the scan range leaves what these axes are allowed to use: "
            + "; ".join(outside))

    n = len(grid)
    pos_mm, pos_cmd, rows, stats = [], [], [], []
    failures = []
    consecutive = 0
    t_start = time.time()
    log(f"scan: {n} points, {seconds:g} s each, settle {settle_s:g} s "
        f"-- {grid.describe()}")

    for i, point in enumerate(grid.points()):
        if should_abort and should_abort():
            log(f"scan: aborted after {len(rows)} of {n} points")
            break
        try:
            stages.move_to(settle_s=settle_s, **point)
            if should_abort and should_abort():
                log(f"scan: aborted after {len(rows)} of {n} points")
                break

            row, st = opcap.capture_pose(
                hosts, seconds, cal,
                chunk_s=chunk_s or opcap.CHUNK_S, drain_s=drain_s)
            here = stages.position()
        except KeyboardInterrupt:
            # Ctrl-C is not an Exception, so it would otherwise unwind past the
            # handler below and take the completed points with it.
            log(f"scan: interrupted, keeping {len(rows)} of {n} points")
            break
        except ostage.MotionInterlocked as exc:
            # Its own clause, ABOVE the catch-all, and fatal. The generic
            # handler below skips a failed point and carries on, which is
            # right for a point that failed and completely wrong for a machine
            # that has been stopped: it would spend the next several points
            # re-commanding moves into a latched interlock, logging a failure
            # each time, until the consecutive-failure count happened to run
            # out. A stop should end the scan on the spot.
            log(f"scan: stopped -- {exc}")
            log(f"scan: keeping {len(rows)} of {n} points")
            break
        except Exception as exc:
            # Everything measured so far stays. Letting this propagate is how a
            # six-hour map used to end with nothing on disk, because both
            # callers only ever saw the exception and never the rows.
            consecutive += 1
            where = " ".join(f"{k}={v:g}" for k, v in point.items())
            failures.append({"index": i, "point": dict(point),
                             "error": f"{type(exc).__name__}: {exc}"})
            log(f"  point {i + 1}/{n} at {where} FAILED "
                f"({type(exc).__name__}: {exc}) -- skipping it and carrying on")
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                log(f"scan: {consecutive} points failed in a row, stopping "
                    f"with {len(rows)} of {n} points kept")
                break
            continue
        consecutive = 0

        pos_cmd.append([point.get(a, here.get(a, np.nan)) for a in grid.names])
        pos_mm.append([here.get(a, np.nan) for a in grid.names])
        rows.append(row)
        stats.append(st)

        # Flag a point the stage did not actually reach. 5 um is well outside
        # the ~1 um the controller settles to and well inside the 47 um the
        # leadscrew is uncertain by, so it catches a stall without crying
        # about ordinary positioning error.
        drift = np.max(np.abs(np.array(pos_cmd[-1]) - np.array(pos_mm[-1])))
        if drift > 0.005:
            log(f"  point {i + 1}: stage is {drift * 1000:.0f} um from the "
                f"commanded position -- possible stall or limit")

        if progress:
            progress(i + 1, n, point, row, st)

    meta = {
        "hosts": list(hosts),
        "seconds_per_point": seconds,
        "settle_s": settle_s,
        "n_requested": n,
        "n_captured": len(rows),
        "n_failed": len(failures),
        "failures": failures,
        "grid": {k: v.tolist() for k, v in grid.axes.items()},
        "stage_serials": {name: stages[name].serial for name in grid.names},
        # Without this a map cannot be re-read years later: the positions are
        # in RIG millimetres, and which way each rig axis runs relative to its
        # stage is a fact about the bracket that nothing else in the file
        # records.
        "axis_frames": {name: stages[name].frame_note() for name in grid.names},
        "ranges_mt": np.asarray(cal.ranges_mt).tolist(),
        "elapsed_s": time.time() - t_start,
        "started": time.strftime("%Y-%m-%d %H:%M:%S",
                                 time.localtime(t_start)),
        "uncorrected": ("VCM subtracted and scaled by nominal V/T only -- no "
                        "tare, no gain trim, no matrix"),
    }
    if not rows:
        return FieldMap(np.zeros((0, len(grid.names))),
                        np.zeros((0, len(grid.names))),
                        np.zeros((0, N_SENSORS, 3)), grid.names, meta, stats)
    return FieldMap(np.array(pos_mm), np.array(pos_cmd), np.stack(rows),
                    grid.names, meta, stats)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uut", action="append", default=None)
    for ax in "xyz":
        p.add_argument(f"--{ax}", metavar="START:STOP:STEP",
                       help=f"{ax} axis range in mm, stop inclusive")
    p.add_argument("--seconds", type=float, default=5.0,
                   help="average this long at each point (default 5)")
    p.add_argument("--settle", type=float, default=DEFAULT_SETTLE_S,
                   help=f"wait after the stage stops (default "
                        f"{DEFAULT_SETTLE_S})")
    p.add_argument("--out", default=None,
                   help="output basename (default captures/fieldmap_<time>)")
    p.add_argument("--cal", default=ocal.CONFIG_NAME)
    p.add_argument("--config", default=ostage.AXIS_CONFIG,
                   help="axis -> stage serial map")
    p.add_argument("--settle-scan", metavar="AXIS",
                   help="instead of scanning, measure how long AXIS takes to "
                        "stop ringing after a move")
    p.add_argument("--settle-distance", type=float, default=10.0,
                   help="move size for --settle-scan, mm")
    p.add_argument("--yes", action="store_true", help="skip the confirmation")
    a = p.parse_args(argv)

    hosts = tuple(a.uut) if a.uut else ob.DEFAULT_UUTS
    cal = ocal.Calibration.load_or_default(
        a.cal, on_error=lambda m: print(f"warning: {m}", file=sys.stderr))

    fm = None
    try:
        with ostage.StageSet.from_config(a.config) as stages:
            if a.settle_scan:
                t, _trace = measure_settle(stages, a.settle_scan,
                                           a.settle_distance, hosts, cal)
                print(f"\nsettled after about {t:.2f} s -- use --settle "
                      f"{max(t, 0.1):.1f} or a little more")
                return 0

            axes = {}
            for ax in "xyz":
                spec = getattr(a, ax)
                if spec:
                    axes[ax] = parse_axis_spec(spec)
            if not axes:
                p.error("give at least one of --x/--y/--z as START:STOP:STEP")
            grid = ScanGrid(axes)

            est = len(grid) * (a.seconds + a.settle + 2.0)
            print(f"{len(grid)} points, {grid.describe()}")
            print(f"roughly {est / 60:.0f} min at {a.seconds:g} s per point")
            if not a.yes and input("start? [y/N] ").strip().lower() not in (
                    "y", "yes"):
                print("aborted")
                return 1

            # capture_pose takes the stream over at the carriers' own rate, so
            # anything already holding it has to let go first.
            for h in hosts:
                if ob.stop_live_stream(h):
                    print(f"{h}: released a running capture")

            fm = run_scan(hosts, stages, grid, a.seconds, cal,
                          settle_s=a.settle)

    # A partial map is written on every exit path that reached the scan, so an
    # interrupt or a stage fault an hour in still leaves the hour on disk.
    except ostage.StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        rc = 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        rc = 130
    else:
        rc = 0

    if fm is None or not len(fm):
        return rc

    out = a.out or os.path.join(
        "captures", time.strftime("fieldmap_%Y%m%d_%H%M%S"))
    path = fm.save(out)
    n_req = fm.meta.get("n_requested", len(fm))
    partial = "" if len(fm) == n_req else f" of {n_req} requested -- PARTIAL"
    print(f"\nwrote {path} ({len(fm)} points{partial})")
    if fm.meta.get("n_failed"):
        print(f"{fm.meta['n_failed']} point(s) failed and were skipped; see "
              f"'failures' in the .json sidecar")
    sem = np.array([s.get("sem_ut", np.nan) for s in fm.stats])
    if np.isfinite(sem).any():
        print(f"per-point noise: median {np.nanmedian(sem):.3f} uT, "
              f"worst {np.nanmax(sem):.3f} uT")
    return rc


if __name__ == "__main__":
    sys.exit(main())
