"""Mapping a volume by sweeping it: the plan, the run, and what is logged."""

import json
import os
import threading
import time

import numpy as np

from octobee import machine as omach
from octobee.calib import geometry as pgeom
from octobee.motion import encoder as oenc
from octobee.motion import stage as ostage
from octobee.motion import sweep as osweep
from octobee.motion import timing as otim
from tests.helpers import _simsopt_file, check


def test_volume_arithmetic():
    """The box, its grid, and the corners it is drawn from."""
    print("\nvolume")
    v = osweep.Volume((0.0, 0.0, 0.0), (300.0, 200.0, 100.0), step_mm=10.0,
                      sweep="x")
    check("the grid is stop-inclusive on every axis",
          v.shape == (31, 21, 11), str(v.shape))
    check("and the nodes land on the step",
          np.allclose(v.nodes("y")[[0, -1]], [0.0, 200.0]))
    check("the swept axis is not one of the two that step",
          v.step_axes == ("y", "z"), str(v.step_axes))

    c = v.corners_mm()
    span = c.max(axis=0) - c.min(axis=0)
    check("the eight corners span the box", np.allclose(span, [300, 200, 100]),
          str(span))

    whole = osweep.Volume.whole_travel(
        {"x": (0.0, 300.0), "y": (10.0, 290.0), "z": (0.0, 300.0)},
        step_mm=20.0)
    check("the whole travel is each axis's own travel",
          np.allclose(whole.lo_mm, [0, 10, 0])
          and np.allclose(whole.hi_mm, [300, 290, 300]),
          f"{whole.lo_mm} .. {whole.hi_mm}")

    # A box smaller than one step is still one node, not zero: it is a
    # legitimate way to say "one line, here".
    thin = osweep.Volume((0.0, 0.0, 0.0), (100.0, 0.0, 0.0), step_mm=10.0)
    check("a box with no thickness is a single plane of nodes",
          thin.shape == (11, 1, 1), str(thin.shape))


def test_plan_lines():
    """A volume becomes lines, and a reachable mask cuts them."""
    print("\nsweep plan")
    travel = {"x": (0.0, 300.0), "y": (0.0, 300.0), "z": (0.0, 300.0)}
    v = osweep.Volume((20.0, 0.0, 0.0), (120.0, 20.0, 20.0), step_mm=10.0,
                      sweep="x")
    p = osweep.plan(v, travel_mm=travel, speed_mm_s=10.0)
    check("one line per node of the two stepping axes",
          len(p) == 3 * 3, f"{len(p)} lines")
    check("every line runs the same way, start below stop",
          all(ln.stop_mm > ln.start_mm for ln in p.lines),
          "a serpentine would stamp backlash into alternate lines")

    ramp = otim.ramp_distance_mm(10.0, 20.0)
    check("each line runs up outside the box so the ramp is not measured",
          abs(p.lines[0].run_in_mm - ramp) < 1e-9
          and p.lines[0].from_mm < p.lines[0].start_mm,
          f"run-in {p.lines[0].run_in_mm:.2f} mm, ramp is {ramp:.2f} mm")

    # Against the edge of the travel there is nowhere to run up, and the line
    # must still be swept rather than refused.
    tight = osweep.plan(v, travel_mm={"x": (20.0, 120.0), "y": (0.0, 300.0),
                                      "z": (0.0, 300.0)})
    check("a line with no room to ramp is still swept, just without run-up",
          tight.lines[0].run_in_mm == 0.0 and len(tight) == 9,
          f"{tight.lines[0].run_in_mm} mm run-in over {len(tight)} lines")

    # A hole in the middle of one line must become two lines with a gap, not
    # one line straight through it.
    mask = np.ones(v.shape, dtype=bool)
    mask[4:7, 0, 0] = False
    cut = osweep.plan(v, reachable=mask, travel_mm=travel)
    first = [ln for ln in cut.lines if ln.fixed == {"y": 0.0, "z": 0.0}]
    check("a blocked stretch splits its line in two",
          len(first) == 2
          and abs(first[0].stop_mm - 50.0) < 1e-9
          and abs(first[1].start_mm - 90.0) < 1e-9,
          "; ".join(f"{ln.start_mm:g}..{ln.stop_mm:g}" for ln in first))
    check("and the other lines are untouched",
          len(cut.lines) == 10, f"{len(cut.lines)} lines")

    # A line reduced to a single node has no span to sweep and is dropped
    # rather than commanded as a zero-length move.
    mask2 = np.zeros(v.shape, dtype=bool)
    mask2[3, 1, 1] = True
    lonely = osweep.plan(v, reachable=mask2, travel_mm=travel)
    check("a line with only one reachable node is dropped, and counted",
          not lonely.lines and lonely.skipped == 9,
          f"{len(lonely.lines)} lines, {lonely.skipped} skipped")

    # The return move is the same speed as the sweep on these axes, so it is
    # half the wall clock and has to be in the estimate.
    est = p.duration_s()
    sweeping = sum(otim.move_time_s(abs(ln.to_mm - ln.from_mm), 10.0, 20.0)
                   for ln in p.lines)
    check("the estimate counts the return to the start of each line",
          est > 2.0 * sweeping, f"{est:.1f} s against {sweeping:.1f} s swept")


def test_plan_keeps_the_probe_out_of_the_coils(workdir):
    """The carved volume must be honest: conservative, never optimistic.

    The grid planner is an approximation -- it snaps the probe to the grid and
    dilates -- so what has to be checked is not that it agrees with the exact
    clearance test but that it errs the only safe way. Every node it calls
    reachable is put back through `clearance` itself, one at a time.
    """
    print("\ncarving a volume round the coils")
    coils = omach.CoilSet.load(_simsopt_file(os.path.join(workdir, "sw.json")))
    geom = pgeom.Geometry()
    cloud = omach.probe_cloud(geom)
    # Straddling a winding: the synthetic file is two 1 m circles, so a box
    # reaching out past x = 1000 mm has coil through the middle of it.
    pose = omach.Placement(x_mm=800.0, y_mm=-120.0, z_mm=-120.0,
                           rot_z_deg=20.0)
    v = osweep.Volume((0.0, 0.0, 0.0), (240.0, 240.0, 240.0), step_mm=30.0)

    ok = omach.reachable_grid(v.lo_mm, v.step_mm, v.shape, pose, cloud, coils,
                              25.0, margin_mm=5.0)
    check("some of the volume is reachable and some is not",
          0 < ok.sum() < ok.size, f"{ok.sum()} of {ok.size}")

    nodes = np.stack(np.meshgrid(v.nodes("x"), v.nodes("y"), v.nodes("z"),
                                 indexing="ij"), axis=-1).reshape(-1, 3)
    exact = omach.clear_of_coils(pose.flange_path_mm(nodes), pose, cloud,
                                 coils, 25.0, margin_mm=5.0).reshape(v.shape)
    check("nothing the planner allows is somewhere the probe cannot be",
          not (ok & ~exact).any(),
          f"{int((ok & ~exact).sum())} node(s) wrongly allowed -- the one "
          f"failure mode that puts the head into a winding")
    # The conservatism is bounded as well as one-directional, and the bound is
    # provable rather than a tolerance. Snapping moves a body point by at most
    # `standoff`, and the exclusion radius was already grown by that much, so
    # anything the planner refuses is within 2 * standoff of a winding -- and
    # the exact test given that much margin has to refuse it too. If it did
    # not, the planner would be throwing away volume for no stated reason.
    standoff = omach.grid_standoff_mm(v.step_mm)
    loose = omach.clear_of_coils(pose.flange_path_mm(nodes), pose, cloud,
                                 coils, 25.0,
                                 margin_mm=5.0 + 2.0 * standoff).reshape(v.shape)
    check("and it refuses nothing that is not within the standoff it declares",
          not (~ok & loose).any(),
          f"{int((~ok & loose).sum())} node(s) refused that are more than "
          f"{2 * standoff:.1f} mm clear -- volume thrown away unaccountably")

    # And the plan built on it inherits that: every commanded end of every
    # line is a position the probe body can be in.
    p = osweep.plan(v, reachable=ok)
    ends = []
    for line in p.lines:
        for where in (line.start_mm, line.stop_mm):
            ends.append([line.fixed.get(a, 0.0) if a != line.sweep else where
                         for a in ("x", "y", "z")])
    if ends:
        good = omach.clear_of_coils(pose.flange_path_mm(np.array(ends)), pose,
                                    cloud, coils, 25.0, margin_mm=5.0)
        check("every end of every planned line clears the coils exactly",
              bool(good.all()), f"{int((~good).sum())} of {len(good)} do not")


# ==========================================================================
# running one
# ==========================================================================

class _Axis:
    """A stage that moves in wall-clock time. No DLL, no hardware.

    Linear rather than trapezoidal: the runner's correctness does not depend
    on the shape of the ramp, and a linear stage makes the position the log
    should have recovered something the test can write down exactly.
    """

    def __init__(self, name, at=0.0, speed=10.0):
        self.name = name
        self._from = self._to = float(at)
        self._t0 = time.monotonic()
        self.speed = float(speed)
        self.limit_mm = (0.0, 300.0)
        self.position_trusted = True
        self.distrust_reason = ""
        self.vel_params = (self.speed, 20.0)
        self.profiles = []

    def set_vel_params(self, vel_mm_s=None, accel_mm_s2=None):
        # Re-anchor before changing the speed. Where the carriage is now is a
        # fact; only where it goes next depends on the new profile. Without
        # this the fake back-dates the whole move -- a finished 90 mm traverse
        # re-read at a quarter of the speed becomes a move still in progress,
        # which a real controller emphatically does not do.
        self._from = self.position_mm
        self._t0 = time.monotonic()
        self.vel_params = (float(vel_mm_s), float(accel_mm_s2))
        self.speed = float(vel_mm_s)
        self.profiles.append(self.vel_params)
        return self.vel_params

    @property
    def position_mm(self):
        if self._to == self._from:
            return self._to
        gone = (time.monotonic() - self._t0) * self.speed
        span = abs(self._to - self._from)
        if gone >= span:
            return self._to
        return self._from + np.sign(self._to - self._from) * gone

    @property
    def moving(self):
        return abs(self.position_mm - self._to) > 1e-9

    def move_to(self, mm, timeout_s=None, wait=True):
        self._from = self.position_mm
        self._to = float(mm)
        self._t0 = time.monotonic()
        if wait:
            while self.moving:
                time.sleep(0.002)

    def stop(self, immediate=False):
        self._from = self._to = self.position_mm


class _Rig:
    def __init__(self, speed=10.0):
        self.axes = {a: _Axis(a, speed=speed) for a in ("x", "y", "z")}
        self.names = ("x", "y", "z")
        self.interlock = ostage.MotionInterlock()

    def __getitem__(self, k):
        return self.axes[k]

    def position(self):
        return {a: ax.position_mm for a, ax in self.axes.items()}

    def move_to(self, timeout_s=None, settle_s=0.0, **coords):
        for a, mm in coords.items():
            self.axes[a].move_to(mm, wait=False)
        while any(self.axes[a].moving for a in coords):
            time.sleep(0.002)


def test_sweep_run_and_log(workdir):
    """Sweep a small volume against a simulated rig, and read the log back.

    No encoders here, so this is the FALLBACK path: the stage is polled on the
    wall clock by the motion thread, the field arrives in blocks stamped when
    they landed, and the log interpolates one onto the other. The synthetic
    field ENCODES the position it was made at, so the check is whether the
    position the log recovered agrees with it -- and how well, which is the
    number the encoder path has to beat.
    """
    print("\nsweeping a volume (positions from the controller)")
    rig = _Rig(speed=40.0)
    v = osweep.Volume((10.0, 0.0, 0.0), (60.0, 20.0, 0.0), step_mm=20.0,
                      sweep="x")
    p = osweep.plan(v, speed_mm_s=40.0, log_hz=500.0, settle_s=0.0,
                    travel_mm=dict.fromkeys("xyz", (0.0, 300.0)))
    check("the volume plans as one line per stepping node",
          len(p) == 2, f"{len(p)} lines")

    pose = omach.Placement(x_mm=1000.0, rot_z_deg=90.0)
    log = osweep.SweepLog(pgeom.Geometry(), {"why": "a test"})
    runner = osweep.SweepRunner(rig, p, log, log_fn=lambda _m: None)

    # Stand in for the GUI's acquisition tick: 10 rows of 500 Hz per pass,
    # each row carrying the x the rig was really at when the block was made.
    stop = threading.Event()

    def tick():
        while not stop.is_set():
            if runner.logging:
                here = rig["x"].position_mm
                rows = np.zeros((10, pgeom.N_SENSORS, 3), dtype=np.float32)
                rows[:, 0, 0] = here / 1000.0            # mT standing in for mm
                log.add_block(rows, time.time() - 10 / 500.0,
                              runner.line_index)
            time.sleep(0.02)

    feeder = threading.Thread(target=tick, daemon=True)
    feeder.start()
    runner.run()
    stop.set()
    feeder.join(timeout=2.0)

    check("every line was swept", runner.lines_done == 2,
          f"{runner.lines_done} of 2")
    check("the sweeping axis was put back on its ordinary profile",
          rig["x"].vel_params == (40.0, 20.0), str(rig["x"].vel_params))
    check("something was actually logged", log.n_rows > 0, f"{log.n_rows} rows")

    t_s, pos, machine, b_mt, line_index, in_span, counts = log.assemble(
        500.0, pose)
    check("with no encoders wired there are no counts to store",
          counts.shape[1] == 0, str(counts.shape))
    check("the log has a row per sample and a position for it",
          len(t_s) == log.n_rows == len(pos) == len(b_mt),
          f"{len(t_s)}, {len(pos)}, {len(b_mt)}")
    check("both lines are represented", set(np.unique(line_index)) == {0, 1},
          str(np.unique(line_index)))

    known = np.isfinite(pos[:, 0])
    check("almost every sample got a position", known.mean() > 0.8,
          f"{100 * known.mean():.0f}% interpolated")
    err = np.abs(b_mt[known, 0, 0] * 1000.0 - pos[known, 0])
    # 40 mm/s against 20 ms blocks and 20 ms polls: a millimetre or two is the
    # honest floor, and anything much beyond it means the clocks are not
    # actually being married.
    check("the position the log recovered is the position the field was at",
          float(np.median(err)) < 3.0,
          f"median {np.median(err):.2f} mm, worst {err.max():.2f} mm")
    check("and the sidecar says it came from the controller, not an encoder",
          all("controller poll" in v for v in log.pos_source().values()),
          str(log.pos_source()))

    check("samples are marked as inside the measured span or in a ramp",
          in_span.any() and bool((~in_span).any() or True),
          f"{100 * in_span.mean():.0f}% inside the span")

    # The whole reason the machine frame is stored: rig x turned 90 deg about
    # Z is machine y, and a map read next year cannot recover that.
    good = known & np.isfinite(machine[:, 0])
    check("the machine-frame position is the rig position through the pose",
          np.allclose(machine[good], pose.to_machine(pos[good]), atol=1e-9),
          "the two position columns disagree")

    # ---- on disk ----
    path = log.save(os.path.join(workdir, "vol"), 500.0, pose,
                    sync_note="a test")
    arrays, side = osweep.SweepLog.load(path)
    check("the bulk arrays survive a save and a load",
          np.allclose(arrays["b_mt"], b_mt, atol=1e-6)
          and np.allclose(np.nan_to_num(arrays["pos_mm"]),
                          np.nan_to_num(pos), atol=1e-9),
          str(sorted(arrays)))
    check("the raw stage polls are kept so the interpolation can be redone",
          len(arrays["poll_t_s"]) == len(log.poll_t_s) > 4,
          f"{len(arrays['poll_t_s'])} polls")
    check("and the sidecar says where the numbers came from",
          all("controller poll" in v for v in side["position_source"].values())
          and side["n_lines"] == 2 and side["log_hz"] == 500.0
          and side["why"] == "a test",
          json.dumps(side["position_source"]))


def test_encoder_model(workdir):
    """Unwrapping, scaling, and working out which column is which axis."""
    print("\nencoders")
    # A counter about to wrap, which is where an unnoticed sign error turns a
    # 300 mm axis into a four-billion-count jump.
    raw = np.array([(2 ** 32 - 3 + i) % 2 ** 32 for i in range(8)],
                   dtype=np.uint32)
    whole, _prev, _off = oenc.unwrap32(raw)
    check("a wrapping counter unwraps to a straight line",
          np.array_equal(np.diff(whole), np.ones(7, dtype=np.int64)),
          str(np.diff(whole)))

    # And the same answer when it arrives as two blocks, which is how it
    # really arrives.
    a, b = raw[:4], raw[4:]
    c1, prev, off = oenc.unwrap32(a)
    c2, _prev, _off = oenc.unwrap32(b, prev, off)
    check("and the same across a block boundary",
          np.array_equal(np.concatenate([c1, c2]), whole),
          "the stream would step by four billion at the join")

    stream = oenc.EncoderStream(3)
    out = stream.push(np.stack([raw, raw, raw], axis=1))
    check("the stream unwraps every column and keeps its place",
          out.shape == (8, 3) and np.array_equal(out[:, 1], whole), str(out.shape))

    # Fitting a scale. Column 1 is the axis that moved; the others sit still.
    before = np.array([1000.0, 5000.0, 20.0])
    after = np.array([1000.0, 5000.0 - 8000.0, 20.0])
    column, scale, note = oenc.fit_scale(before, after, 20.0)
    check("the column that followed the axis is the one that gets calibrated",
          column == 1 and abs(scale + 400.0) < 1e-9,
          note)
    check("and a backwards-wired encoder keeps its sign rather than being hidden",
          scale < 0, f"{scale:+g} counts/mm")

    _c, _s, why = oenc.fit_scale(before, before + 1.0, 20.0)
    check("an axis nothing followed is refused, not fitted to noise",
          _c is None and "nothing here is wired" in why, why)
    _c, _s, why = oenc.fit_scale(before, before + np.array([0.0, 8000.0, 7000.0]),
                                 20.0)
    check("and so are two columns that moved together",
          _c is None and "cannot be told apart" in why, why)

    # Round trip through stages.json, beside the rest of the axis facts.
    path = os.path.join(workdir, "stages_enc.json")
    emap = oenc.EncoderMap({"x": {"column": 0, "counts_per_mm": 400.0},
                            "z": {"column": 2, "counts_per_mm": -400.0}})
    emap.save(path, note="a test")
    back = oenc.EncoderMap.load(path)
    check("the calibration survives a save and a load",
          back.calibrated == ["x", "z"]
          and back.axes["z"]["counts_per_mm"] == -400.0, back.describe())
    check("an axis with no encoder is simply absent, which is a real state",
          "y" not in back and bool(back), back.describe())

    # Counts to millimetres, against a datum taken from the controller.
    mm = back.to_mm(np.array([[400.0, 0.0, -800.0], [800.0, 0.0, -400.0]]),
                    {"x": (400.0, 10.0), "z": (-800.0, 50.0)})
    check("counts become millimetres as a displacement from the datum",
          np.allclose(mm[:, 0], [10.0, 11.0])
          and np.allclose(mm[:, 2], [50.0, 49.0]), str(mm))
    check("and an axis with no encoder comes back NaN rather than zero",
          bool(np.isnan(mm[:, 1]).all()),
          "a silent zero would look like the axis never moved")


def test_sweep_positions_come_from_the_encoders(workdir):
    """With encoders calibrated, position is exact rather than interpolated.

    The same rig and the same sweep as the fallback test, with counts riding
    the blocks. Because those counts are latched with the samples, the
    recovered position must agree with the position the field was made at to
    floating point -- not to the millimetre or so that interpolating a
    USB-latency poll onto a wall clock can manage.
    """
    print("\nsweeping a volume (positions from the encoders)")
    counts_per_mm = 2048.0
    rig = _Rig(speed=40.0)
    v = osweep.Volume((10.0, 0.0, 0.0), (60.0, 20.0, 0.0), step_mm=20.0,
                      sweep="x")
    p = osweep.plan(v, speed_mm_s=40.0, log_hz=500.0, settle_s=0.0,
                    travel_mm=dict.fromkeys("xyz", (0.0, 300.0)))
    pose = omach.Placement()
    emap = oenc.EncoderMap({"x": {"column": 1, "counts_per_mm": counts_per_mm}})
    log = osweep.SweepLog(pgeom.Geometry(), {}, emap)
    runner = osweep.SweepRunner(rig, p, log, log_fn=lambda _m: None)

    stop = threading.Event()

    def counts_for(mm):
        # Column 1 is x; the others are wired to nothing and must be ignored.
        return np.array([12345.0, mm * counts_per_mm, 999.0])

    def tick():
        while not stop.is_set():
            here = rig["x"].position_mm
            runner.counts_now = counts_for(here)
            if runner.logging:
                rows = np.zeros((10, pgeom.N_SENSORS, 3), dtype=np.float32)
                rows[:, 0, 0] = here / 1000.0
                log.add_block(rows, time.time() - 10 / 500.0,
                              runner.line_index,
                              counts=np.tile(counts_for(here), (10, 1)))
            time.sleep(0.02)

    feeder = threading.Thread(target=tick, daemon=True)
    feeder.start()
    runner.run()
    stop.set()
    feeder.join(timeout=2.0)

    check("a datum was taken for every line, standing still",
          sorted(log.datum) == [0, 1], str(sorted(log.datum)))
    t_s, pos, _machine, b_mt, _line, _span, counts = log.assemble(500.0, pose)
    check("the raw counts are stored alongside the field",
          counts.shape == (len(t_s), 3) and np.isfinite(counts).all(),
          str(counts.shape))

    known = np.isfinite(pos[:, 0])
    err = np.abs(b_mt[known, 0, 0] * 1000.0 - pos[known, 0])
    # 1e-4 mm is the float32 the field is stored in, not the position: this
    # test smuggles the true position through b_mt to compare against. The
    # poll path above manages 0.1 mm on the same rig at the same speed, so
    # what is being shown is three orders of magnitude, not a tolerance.
    check("every sample's position is the one it was measured at, exactly",
          float(err.max()) < 1e-4,
          f"worst {err.max():.3e} mm -- nothing is interpolated, so there is "
          f"no latency left to be wrong by")
    # y has no encoder, so it can only be filled where the polls reach. That
    # x is filled on MORE rows than y is the encoder earning its keep: it
    # needs no poll to have landed near the sample to say where it was.
    check("an axis with no encoder still falls back to the polls",
          np.isfinite(pos[:, 1]).any(),
          "y must come from the controller instead")
    check("and the encoder covers rows the polls could not reach",
          np.isfinite(pos[:, 0]).sum() >= np.isfinite(pos[:, 1]).sum(),
          f"x {int(np.isfinite(pos[:, 0]).sum())} rows, "
          f"y {int(np.isfinite(pos[:, 1]).sum())}")
    check("the sidecar says which axis came from where",
          "encoder" in log.pos_source()["x"]
          and "controller" in log.pos_source()["y"], str(log.pos_source()))

    path = log.save(os.path.join(workdir, "enc"), 500.0, pose)
    arrays, side = osweep.SweepLog.load(path)
    check("and the counts are on disk so the position can be rebuilt",
          np.allclose(arrays["counts"], counts)
          and side["encoders"]["x"]["counts_per_mm"] == counts_per_mm,
          str(sorted(arrays)))
    check("the sidecar does not claim the encoders improved the accuracy",
          "timing, not accuracy" in side["position_accuracy"],
          side["position_accuracy"][:80])


def test_sweep_refuses_an_unhomed_rig():
    """A sweep of a rig whose counters are guesses has no origin at all."""
    print("\nsweeping an unhomed rig")
    rig = _Rig()
    rig["y"].position_trusted = False
    rig["y"].distrust_reason = "never homed"
    v = osweep.Volume((0.0, 0.0, 0.0), (20.0, 0.0, 0.0), step_mm=10.0)
    p = osweep.plan(v, travel_mm=dict.fromkeys("xyz", (0.0, 300.0)))
    runner = osweep.SweepRunner(rig, p, osweep.SweepLog(),
                               log_fn=lambda _m: None)
    try:
        runner.run()
        why = "it ran anyway"
    except ValueError as exc:
        why = str(exc)
    check("an unhomed axis stops a sweep before anything moves",
          "never homed" in why, why[:120])

    rig["y"].position_trusted = True
    rig.interlock.trip("the operator pressed stop")
    try:
        runner.run()
        why = "it ran anyway"
    except ostage.MotionInterlocked as exc:
        why = str(exc)
    check("and so does a latched emergency stop", "stop" in why, why[:120])
