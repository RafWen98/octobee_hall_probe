"""The calibration run itself, driven against stages that misbehave the way
the real ones did.

test_encoder_fit.py pins the arithmetic. This pins the procedure around it:
that it never commands an axis outside its soft limits, that it puts every
axis back where it found it, that it clamps the span to the room available,
and that a stage which stops short of every target no longer poisons the scale
it produces.
"""

import numpy as np

from octobee.gui.workers import (
    EncoderCalibrationWorker,
    EncoderReferenceWorker,
)
from octobee.motion import encoder as oenc
from tests.helpers import (
    check,
)

SCALE = {"x": 14_400.0, "y": 14_400.0, "z": -14_400.0}   # z is wired backwards
COLUMN = {"x": 0, "y": 1, "z": 2}
BASE = {"x": 1.2e6, "y": 4.0e5, "z": 9.0e5}


class FakeAxis:
    """A carriage that stops a fixed distance short of everything it is told.

    `position_mm` reports where it really is, which is what a Thorlabs
    controller does -- it is not lying about the undershoot, nobody was asking
    it. The old calibration never asked; it divided by the number it had
    commanded.
    """

    def __init__(self, start, lo=0.0, hi=300.0, undershoot_mm=0.0):
        self.true_mm = float(start)
        self.limit_mm = (float(lo), float(hi))
        self.undershoot_mm = float(undershoot_mm)
        self.commanded = []

    @property
    def position_mm(self):
        return self.true_mm

    def move_to(self, mm, **kw):
        mm = float(mm)
        lo, hi = self.limit_mm
        # Not a check on the fake: a run that commands outside the soft limits
        # is a run that drives the head into something.
        assert lo - 1e-9 <= mm <= hi + 1e-9, \
            f"commanded {mm:g} mm, outside {lo:g}..{hi:g}"
        self.commanded.append(mm)
        if mm > self.true_mm:
            self.true_mm = mm - self.undershoot_mm
        elif mm < self.true_mm:
            self.true_mm = mm + self.undershoot_mm


class FakeInterlock:
    def require_clear(self, what):
        return True


class FakeStages:
    def __init__(self, axes):
        self.axes = axes
        self.interlock = FakeInterlock()

    @property
    def names(self):
        return list(self.axes)

    def __getitem__(self, name):
        return self.axes[name]


def _counts_from(stages):
    """The counter reading for wherever the carriages really are.

    A fresh array every call, which is what the acquisition tick hands over --
    the worker waits for the object to change to know its reading is not from
    during a move.
    """
    def read():
        row = np.zeros(3)
        for axis, st in stages.axes.items():
            row[COLUMN[axis]] = BASE[axis] + SCALE[axis] * st.true_mm
        return row
    return read


def _run(stages, span_mm=100.0, points=11):
    """Drive the worker synchronously and return (found, error, messages)."""
    worker = EncoderCalibrationWorker(stages, _counts_from(stages),
                                      span_mm=span_mm, points=points)
    # The settle exists so a reading is not taken mid-move. These carriages
    # arrive instantly, so it is only wall-clock time here.
    worker.SETTLE_S = 0.0
    worker.STABLE_TIMEOUT_S = 0.0
    out = {}
    worker.message.connect(lambda m: out.setdefault("msgs", []).append(m))
    worker.done.connect(lambda f, e: out.update(found=f, error=e))
    worker.run()
    return out.get("found"), out.get("error"), out.get("msgs", [])


def test_a_stage_that_stops_short_no_longer_poisons_the_scale():
    """x's failure, run through the whole procedure this time."""
    print("\ncalibration run, undershooting stages")
    starts = {"x": 150.0, "y": 150.0, "z": 150.0}
    stages = FakeStages({a: FakeAxis(starts[a], undershoot_mm=0.0814)
                         for a in ("x", "y", "z")})
    found, error, msgs = _run(stages)

    check("no axis was refused", not error, error)
    check("all three were measured", sorted(found) == ["x", "y", "z"],
          str(sorted(found)))
    for axis in ("x", "y", "z"):
        got = found[axis]["counts_per_mm"]
        check(f"{axis} comes back at its true scale despite the undershoot",
              abs(abs(got) - 14_400.0) < 1.0, f"{got:+,.1f} counts/mm")
        check(f"{axis} found its own column",
              found[axis]["column"] == COLUMN[axis],
              f"column {found[axis]['column']}")
    check("z keeps the sign it is wired with",
          found["z"]["counts_per_mm"] < 0, str(found["z"]["counts_per_mm"]))
    check("and the axes therefore agree with each other",
          oenc.odd_axis_out({a: s["counts_per_mm"]
                             for a, s in found.items()}) is None)

    check("every axis is left where it was found",
          all(abs(stages[a].true_mm - starts[a]) < 0.1 for a in starts),
          str({a: round(stages[a].true_mm, 3) for a in starts}))
    # Every point is approached from below, so a stage that stops short does
    # so by the same amount at both ends and the span it covered is the span
    # it was sent over. (Approached from both sides it lost the undershoot at
    # each end, and the recorded span was 0.16 mm short.)
    check("the run records the span, the points and the passes",
          abs(found["x"]["span_mm"] - 100.0) < 1e-3
          and found["x"]["points"] == 11 and found["x"]["passes"] == 3,
          str(found["x"]))
    check("the log says what each pass measured",
          sum("pass 1/3 ->" in m for m in msgs) == 3
          and sum("pass 3/3 ->" in m for m in msgs) == 3,
          f"{len(msgs)} messages")
    check("a clean run is called repeatable and is not warned about",
          all(found[a]["repeatable"] for a in found)
          and not any("WARNING" in m for m in msgs),
          "; ".join(m for m in msgs if "WARNING" in m) or "no warnings")


def test_the_span_is_clamped_to_the_room_the_axis_has():
    """A rig parked at the end of its travel calibrates rather than refusing.

    FakeAxis asserts on any command outside the soft limits, so this fails
    loudly rather than subtly if the span is ever allowed to run off the end.
    """
    print("\ncalibration span")
    stages = FakeStages({
        "x": FakeAxis(2.0, lo=0.0, hi=300.0),        # hard against the bottom
        "y": FakeAxis(299.0, lo=0.0, hi=300.0),      # hard against the top
        "z": FakeAxis(150.0, lo=140.0, hi=180.0),    # only 40 mm to work in
    })
    found, error, msgs = _run(stages, span_mm=100.0)

    check("all three still calibrate", sorted(found) == ["x", "y", "z"],
          f"{sorted(found)} / {error}")
    check("an axis at the bottom of its travel uses the room above it",
          found["x"]["span_mm"] == 100.0, str(found["x"]["span_mm"]))
    check("an axis at the top uses the room below it",
          found["y"]["span_mm"] == 100.0, str(found["y"]["span_mm"]))
    # 40 mm of travel less the 2 mm kept below the lowest point so that it
    # too can be approached from below.
    check("an axis with less travel than asked for uses what it has",
          abs(found["z"]["span_mm"] - 38.0) < 1e-6, str(found["z"]["span_mm"]))
    check("and says so rather than quietly shortening the run",
          any("room for 38 mm" in m for m in msgs),
          "; ".join(m for m in msgs if "room" in m) or "nothing said")

    lo_cmd = min(stages["x"].commanded)
    hi_cmd = max(stages["y"].commanded)
    check("nothing was commanded below the bottom limit", lo_cmd >= 0.0,
          f"{lo_cmd:g} mm")
    check("nothing was commanded above the top limit", hi_cmd <= 300.0,
          f"{hi_cmd:g} mm")


def test_an_axis_with_no_room_is_refused_rather_than_nudged():
    print("\ncalibration with no room")
    stages = FakeStages({
        "x": FakeAxis(150.0, lo=0.0, hi=300.0),
        "y": FakeAxis(100.5, lo=100.0, hi=101.0),    # 1 mm of travel
        "z": FakeAxis(150.0, lo=0.0, hi=300.0),
    })
    found, error, _ = _run(stages, span_mm=100.0)
    check("the axis that cannot move is left out", "y" not in found,
          str(sorted(found)))
    check("and the reason names it and its travel",
          "y:" in error and "too short" in error, error)
    check("the other two are still measured",
          sorted(found) == ["x", "z"], str(sorted(found)))


def test_a_stream_with_no_counts_is_reported_not_fitted():
    """A 694-only session, or the stream stopped mid-run."""
    print("\ncalibration with no counts")
    stages = FakeStages({"x": FakeAxis(150.0)})
    worker = EncoderCalibrationWorker(stages, lambda: None, span_mm=50.0,
                                      points=5)
    worker.SETTLE_S = 0.0
    worker.STABLE_TIMEOUT_S = 0.0
    out = {}
    worker.done.connect(lambda f, e: out.update(found=f, error=e))
    worker.run()
    check("nothing is fitted from nothing", not out["found"],
          str(out["found"]))
    check("and the message says where the counts come from",
          "acq1001_695" in out["error"], out["error"])
    check("the axis is still put back where it was found",
          abs(stages["x"].true_mm - 150.0) < 1e-9, str(stages["x"].true_mm))


def test_two_axes_on_one_column_is_refused():
    """The check that stops an axis being calibrated against its neighbour."""
    print("\ncalibration with a shared column")
    stages = FakeStages({"x": FakeAxis(150.0)})

    def both_columns():
        # Column 1 tracks x as well, which is what a miswired site looks like.
        p = stages["x"].true_mm
        return np.array([BASE["x"] + 14_400.0 * p,
                         BASE["y"] + 13_000.0 * p,
                         BASE["z"]])

    worker = EncoderCalibrationWorker(stages, both_columns, span_mm=50.0,
                                      points=5)
    worker.SETTLE_S = 0.0
    worker.STABLE_TIMEOUT_S = 0.0
    out = {}
    worker.done.connect(lambda f, e: out.update(found=f, error=e))
    worker.run()
    check("neither column is picked", not out["found"], str(out["found"]))
    check("and the reason is that they cannot be told apart",
          "cannot be told apart" in out["error"], out["error"])


def test_every_reading_is_taken_after_an_approach_from_below():
    """Backlash becomes a constant, not a slope, and the readings can be
    compared with a ruler. Checked from the stage's side: the move that
    brought it to each standstill must have been upward, on the passes that
    go down the points as well as the ones that go up."""
    print("\napproach direction")
    stages = FakeStages({"x": FakeAxis(150.0)})
    ax = stages["x"]
    ax.last_dir = None
    real_move = ax.move_to

    def move_to(mm, **kw):
        before = ax.true_mm
        real_move(mm, **kw)
        ax.last_dir = "up" if ax.true_mm > before else "down"

    ax.move_to = move_to
    seen = []
    base = _counts_from(stages)

    def read():
        seen.append(ax.last_dir)
        return base()

    worker = EncoderCalibrationWorker(stages, read, span_mm=100.0, points=5,
                                      axes=["x"], passes=3)
    worker.SETTLE_S = 0.0
    worker.STABLE_TIMEOUT_S = 0.0
    out = {}
    worker.done.connect(lambda f, e: out.update(found=f, error=e))
    worker.run()
    check("the run succeeded", "x" in out["found"], out["error"])
    check("it read at every one of the 15 standstills",
          len(seen) >= 15, f"{len(seen)} reads")
    check("and every one was reached by moving up",
          set(seen) == {"up"}, str(sorted(set(seen), key=str)))


def test_a_scale_that_changes_between_passes_is_not_called_repeatable():
    """y came back 1.1% apart between two runs and nothing said so."""
    print("\npass spread")
    stages = FakeStages({"x": FakeAxis(150.0)})
    ax = stages["x"]

    def read():
        # 22 moves a pass: a park and a visit for each of 11 points.
        k = (len(ax.commanded) - 1) // 22
        return np.array([BASE["x"] + SCALE["x"] * (1 + 0.002 * k) * ax.true_mm,
                         0.0, 0.0])

    worker = EncoderCalibrationWorker(stages, read, span_mm=100.0, points=11,
                                      axes=["x"], passes=3)
    worker.SETTLE_S = 0.0
    worker.STABLE_TIMEOUT_S = 0.0
    out = {}
    msgs = []
    worker.message.connect(msgs.append)
    worker.done.connect(lambda f, e: out.update(found=f, error=e))
    worker.run()
    spec = out["found"]["x"]
    check("the passes disagree by about 0.4% (2 x 0.2%)",
          3500 < spec["pass_spread_ppm"] < 4500, str(spec["pass_spread_ppm"]))
    check("and the run says it is not repeatable",
          spec["repeatable"] is False, str(spec["repeatable"]))
    check("in words, with the limit",
          any("WARNING" in m and "not a repeatable scale" in m for m in msgs),
          "; ".join(msgs[-2:]))


class RulerAxis(FakeAxis):
    """An axis whose millimetre is 0.4% longer than the controller says."""
    FACTOR = 1.004
    ZERO = 3.0


def test_the_ruler_finds_a_controller_whose_millimetre_is_wrong():
    """What encoder-against-controller cannot see: both are wrong together."""
    print("\nruler check")
    stages = FakeStages({"x": RulerAxis(150.0)})
    ax = stages["x"]
    worker = EncoderReferenceWorker(stages, _counts_from(stages), axis="x",
                                    span_mm=200.0, points=7)
    worker.SETTLE_S = 0.0
    worker.STABLE_TIMEOUT_S = 0.0
    asked = []

    def answer(axis, i, n, ctrl):
        asked.append((i, n, ctrl))
        # The ruler: a different zero, and a millimetre 0.4% longer.
        worker.supply(RulerAxis.ZERO + RulerAxis.FACTOR * ax.position_mm)

    worker.ask.connect(answer)
    out = {}
    worker.done.connect(lambda r, e: out.update(result=r, error=e))
    worker.run()
    r = out["result"]
    check("it finished", r is not None, out["error"])
    check("it asked for every reading, in order",
          [a[0] for a in asked] == list(range(7)) and asked[0][1] == 7,
          str([a[0] for a in asked]))
    check("the controller's millimetre is found to be 0.4% short",
          abs(r["controller_factor"] - 1.004) < 1e-6,
          f"{r['controller_factor']:.6f}")
    # Counts follow the true position, which the controller reports as it is.
    check("and a true millimetre is fewer counts than the controller's",
          abs(r["counts_per_mm"] - 14_400.0 / 1.004) < 0.5,
          f"{r['counts_per_mm']:,.1f}")
    check("the axis is back where it was found",
          abs(ax.true_mm - 150.0) < 1e-9, str(ax.true_mm))


def test_the_ruler_check_stops_when_the_operator_does():
    print("\nruler check, cancelled")
    stages = FakeStages({"x": FakeAxis(150.0)})
    worker = EncoderReferenceWorker(stages, _counts_from(stages), axis="x",
                                    span_mm=100.0, points=5)
    worker.SETTLE_S = 0.0
    worker.STABLE_TIMEOUT_S = 0.0
    worker.ask.connect(lambda *a: worker.supply(None))
    out = {}
    worker.done.connect(lambda r, e: out.update(result=r, error=e))
    worker.run()
    check("nothing is reported", out["result"] is None, str(out["result"]))
    check("and it says it stopped", "stopped" in out["error"], out["error"])
    # Esc closes the question, and Esc is also the emergency stop: a cancel
    # must not be answered with a move back across the span.
    check("the axis is left where it stopped, not driven back",
          abs(stages["x"].true_mm - 100.0) < 1e-9, str(stages["x"].true_mm))
    check("and the operator is told so", "left where it is" in out["error"],
          out["error"])
