"""Motorised field maps: the grid, the guards, the file."""

import itertools
import os

import numpy as np

from octobee.calib import convert as ocal
from octobee.motion import scan as oscan
from octobee.motion import stage as ostage
from tests.helpers import (
    check,
)



def test_scan_grid():
    """The grid is the part that silently ruins a map if it is wrong."""
    pts = oscan.parse_axis_spec("0:100:10")
    check("an axis spec includes its stop point",
          len(pts) == 11 and pts[-1] == 100.0,
          f"{len(pts)} points, last {pts[-1]}")
    pts = oscan.parse_axis_spec("0:10:3")
    check("a step that does not divide the span stops short rather than over",
          pts[-1] <= 10.0 + 1e-9, f"last {pts[-1]}")

    for bad, why in (("0:100", "too few fields"),
                     ("0:100:0", "zero step"),
                     ("100:0:10", "stop before start")):
        try:
            oscan.parse_axis_spec(bad)
            ok = False
        except ValueError:
            ok = True
        check(f"axis spec rejects {why}", ok, bad)

    grid = oscan.ScanGrid({"x": oscan.parse_axis_spec("0:20:10"),
                           "y": oscan.parse_axis_spec("0:10:5")})
    pts = list(grid.points())
    check("the grid visits every combination", len(pts) == len(grid) == 9,
          f"{len(pts)} points")

    # The whole reason the scan is not a serpentine: if any axis ever ran
    # backwards, leadscrew backlash would stamp an offset into alternate rows
    # that looks exactly like real field structure.
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    check("the slow axis never reverses",
          all(b >= a for a, b in itertools.pairwise(xs)), f"{xs}")
    rows = [ys[i:i + 3] for i in range(0, 9, 3)]
    check("every row of the fast axis runs the same way",
          all(r == sorted(r) for r in rows), f"{rows}")


def test_scan_guards():
    """run_scan must refuse the two setups that yield a plausible bad map."""
    class FakeStage:
        def __init__(self, name, homed, limit=(0.0, 300.0)):
            self.name, self.serial, self.homed = name, "45000000", homed
            self.position_trusted = homed
            self.distrust_reason = None if homed else "never homed"
            self.limit_mm = limit

    class FakeSet:
        def __init__(self, axes):
            self.axes = axes
            self.names = list(axes)
            self.interlock = ostage.MotionInterlock()

        def __getitem__(self, k):
            return self.axes[k]

    grid = oscan.ScanGrid({"x": oscan.parse_axis_spec("0:10:5")})
    cal = ocal.Calibration()

    try:
        oscan.run_scan(("h",), FakeSet({"y": FakeStage("y", True)}), grid,
                       1.0, cal)
        ok = False
    except ostage.StageError:
        ok = True
    check("a scan naming an axis the stages lack is refused", ok)

    try:
        oscan.run_scan(("h",), FakeSet({"x": FakeStage("x", False)}), grid,
                       1.0, cal)
        ok = False
    except ostage.StageError:
        ok = True
    check("a scan over an unhomed axis is refused",
          ok, "an unhomed counter gives the map no origin")

    # An axis that IS homed but whose count was lost -- an immediate stop, a
    # stall -- is the dangerous one: the controller still says homed, so the
    # old check waved it through and the map came out shifted by however far
    # the count drifted, with nothing on its face to say so.
    lost = FakeStage("x", True)
    lost.position_trusted = False
    lost.distrust_reason = "stopped immediately, may have lost steps"
    try:
        oscan.run_scan(("h",), FakeSet({"x": lost}), grid, 1.0, cal)
        ok = False
    except ostage.StageError as exc:
        ok = "lost steps" in str(exc)
    check("a scan over a homed axis whose count was lost is refused", ok,
          "the homed bit survives a lost count; position_trusted does not")

    # A range outside the working envelope must be refused BEFORE the first
    # move, not discovered as a point failure hours in.
    narrow = FakeSet({"x": FakeStage("x", True, limit=(0.0, 5.0))})
    try:
        oscan.run_scan(("h",), narrow, grid, 1.0, cal)
        ok = False
    except ostage.StageError as exc:
        ok = "allowed" in str(exc)
    check("a scan that leaves the soft limits is refused up front", ok)

    # And a machine that has been stopped stays stopped, whatever asks.
    latched = FakeSet({"x": FakeStage("x", True)})
    latched.interlock.trip("operator pressed the emergency stop")
    try:
        oscan.run_scan(("h",), latched, grid, 1.0, cal)
        ok = False
    except ostage.MotionInterlocked:
        ok = True
    check("a scan is refused while the emergency stop is latched", ok)


def test_scan_survives_failures(workdir):
    """A point that fails must cost that point, not the whole map."""
    print("\nscan resilience")

    class Stage:
        def __init__(self, name):
            self.name, self.serial, self.homed = name, "45000000", True
            self.position_trusted, self.distrust_reason = True, None
            self.limit_mm = (0.0, 300.0)
            self.limit_declared = True

        def frame_note(self):
            return "as mounted"

    class Stages:
        def __init__(self):
            self.axes = {"x": Stage("x")}
            self.names = ["x"]
            self.interlock = ostage.MotionInterlock()
            self.at = 0.0

        def __getitem__(self, k):
            return self.axes[k]

        def move_to(self, settle_s=0.0, **coords):
            self.at = coords["x"]

        def position(self):
            return {"x": self.at}

    grid = oscan.ScanGrid({"x": oscan.parse_axis_spec("0:40:10")})   # 5 points
    cal = ocal.Calibration()
    calls = {"n": 0}

    def flaky(hosts, seconds, cal_, chunk_s=None, drain_s=0.0):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("carrier dropped the stream")
        return np.zeros((16, 3)), {"sem_ut": 0.02, "lost": 0}

    real = oscan.opcap.capture_pose
    oscan.opcap.capture_pose = flaky
    try:
        fm = oscan.run_scan(("h",), Stages(), grid, 1.0, cal, log=lambda m: None)
    finally:
        oscan.opcap.capture_pose = real

    check("one failed point does not lose the whole map", len(fm) == 4,
          f"{len(fm)} of 5 points kept")
    check("the failure is recorded rather than swallowed",
          fm.meta["n_failed"] == 1
          and "carrier dropped" in fm.meta["failures"][0]["error"])
    check("the map still knows how many points were asked for",
          fm.meta["n_requested"] == 5)
    path = fm.save(os.path.join(workdir, "partial"))
    back = oscan.FieldMap.load(path)
    check("a partial map round trips through disk", len(back) == 4)

    # Three in a row is not going to clear on its own; stop, but keep the map.
    calls["n"] = 0

    def always_fails(hosts, seconds, cal_, chunk_s=None, drain_s=0.0):
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("still broken")
        return np.zeros((16, 3)), {"sem_ut": 0.02, "lost": 0}

    oscan.opcap.capture_pose = always_fails
    try:
        fm2 = oscan.run_scan(("h",), Stages(), grid, 1.0, cal, log=lambda m: None)
    finally:
        oscan.opcap.capture_pose = real
    check("a persistent fault stops the scan but keeps what it had",
          len(fm2) == 1 and fm2.meta["n_failed"] == oscan.MAX_CONSECUTIVE_FAILURES,
          f"{len(fm2)} kept, {fm2.meta['n_failed']} failed")


def test_fieldmap_roundtrip(workdir):
    rng = np.random.default_rng(11)
    n = 12
    pos = np.stack([np.arange(n, dtype=float), np.zeros(n)], axis=1)
    fm = oscan.FieldMap(pos, pos + 1e-4, rng.normal(size=(n, 16, 3)),
                        ["x", "y"], meta={"seconds_per_point": 5.0},
                        stats=[{"sem_ut": 0.03, "lost": 0}] * n)
    path = fm.save(os.path.join(workdir, "map_test"))
    back = oscan.FieldMap.load(path)
    check("a field map survives a save/load round trip",
          len(back) == n and np.allclose(back.b_mt, fm.b_mt, atol=1e-6),
          f"{len(back)} points")
    check("the round trip keeps commanded and reached positions apart",
          not np.allclose(back.pos_mm, back.pos_cmd),
          "that difference is the only way a stalled point is visible later")
    check("the field map writes a provenance sidecar",
          os.path.exists(os.path.splitext(path)[0] + ".json"))


def test_group_move_sizes_its_own_wait():
    """A grouped move must size its wait, not hand None to it.

    The bug this pins, seen on the bench as every point of a guided
    calibration pass failing in the same second:

        point 1/41 at y=3.5 x=40 z=220 FAILED (TypeError: unsupported
        operand type(s) for +: 'float' and 'NoneType') -- skipping it

    StageSet.move_to's timeout default became None, meaning "work it out from
    the distance" -- the convention Stage.move_to and move_by follow through
    _timeout_for -- but nothing then worked it out. The None went into
    Stage.wait, where `time.monotonic() + timeout_s` raised before the axis
    had moved. run_scan records a failed point and carries on, so every field
    map and every calibration pass lost all of its points and stopped after
    three in a row, with nothing in the message about stages at all.

    Checked through the REAL Stage._timeout_for, because a fake that sizes the
    wait itself would pass whether or not move_to ever asked for one.
    """
    print("\ngrouped moves size their own wait")

    class FakeAxis:
        """Enough of a Stage for move_to, with the real timeout arithmetic."""

        def __init__(self, name, at):
            self.name, self._at = name, at
            self._vel_cfg, self._accel_cfg = 6.0, 10.0
            self.waited = self.moved = None
            self.stopped = False

        @property
        def position_mm(self):
            return self._at

        def _timeout_for(self, distance_mm, timeout_s):
            return ostage.Stage._timeout_for(self, distance_mm, timeout_s)

        def move_to(self, mm, wait=True):
            self.moved = mm

        def wait(self, timeout_s=180.0, what="move"):
            # Exactly what the real one does with a None it cannot use.
            if timeout_s is None:
                raise ostage.StageError(f"{self.name}: wait() with no timeout")
            self.waited = timeout_s

        def stop(self):
            self.stopped = True

    axes = {"y": FakeAxis("y", 0.0), "x": FakeAxis("x", 40.0)}
    ss = ostage.StageSet.__new__(ostage.StageSet)
    ss.axes = axes

    ss.move_to(y=143.5, x=40.0)
    check("a grouped move with no timeout still waits",
          all(a.waited is not None for a in axes.values()),
          str({n: a.waited for n, a in axes.items()}))
    check("and the wait is sized from how far that axis goes",
          axes["y"].waited == ostage.move_timeout_s(143.5, 6.0, 10.0)
          and axes["y"].waited > axes["x"].waited,
          f"y {axes['y'].waited:.2f} s over 143.5 mm, "
          f"x {axes['x'].waited:.2f} s standing still")

    # An explicit timeout still wins -- that is what the argument is for.
    for a in axes.values():
        a.waited = None
    ss.move_to(timeout_s=42.0, y=0.0)
    check("an explicit timeout is passed through untouched",
          axes["y"].waited == 42.0, str(axes["y"].waited))

    # And the distance is read BEFORE the move: an axis already travelling no
    # longer says where it started, so sizing after the fact reads ~zero.
    check("the distance is taken before the axis is commanded",
          axes["y"].moved == 0.0)


def test_wait_without_a_timeout_says_so():
    """wait(None) must name the axis, not fail inside arithmetic."""
    print("\nwait with no timeout")

    class Stub:
        name = "y"

    try:
        ostage.Stage.wait(Stub(), timeout_s=None)
        check("wait() with no timeout is refused", False)
    except ostage.StageError as e:
        check("wait() with no timeout is refused, by name",
              "y" in str(e) and "_timeout_for" in str(e), str(e))
    except TypeError as e:
        check("wait() with no timeout is refused readably", False, str(e))
