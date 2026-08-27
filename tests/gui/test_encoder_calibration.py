"""The calibration run itself, driven against stages that misbehave the way
the real ones did.

test_encoder_fit.py pins the arithmetic. This pins the procedure around it:
that it never commands an axis outside its soft limits, that it puts every
axis back where it found it, that it clamps the span to the room available,
and that a stage which stops short of every target no longer poisons the scale
it produces.
"""

import numpy as np

from octobee.gui.workers import EncoderCalibrationWorker
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
    worker.FRESH_TIMEOUT_S = 0.0
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
    # The span recorded is the one the carriage actually covered, not the one
    # it was sent over -- 100 mm less the undershoot at each end. That is the
    # right number to keep: it is the span the scale was measured across.
    check("the run records the span it really covered, and its point count",
          abs(found["x"]["span_mm"] - (100.0 - 2 * 0.0814)) < 1e-3
          and found["x"]["points"] == 11, str(found["x"]))
    check("the log says what each direction measured",
          sum("out  ->" in m for m in msgs) == 3
          and sum("back ->" in m for m in msgs) == 3,
          f"{len(msgs)} messages")


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
    check("an axis with less travel than asked for uses what it has",
          abs(found["z"]["span_mm"] - 40.0) < 1e-6, str(found["z"]["span_mm"]))
    check("and says so rather than quietly shortening the run",
          any("room for 40 mm" in m for m in msgs),
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
    worker.FRESH_TIMEOUT_S = 0.0
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
    worker.FRESH_TIMEOUT_S = 0.0
    out = {}
    worker.done.connect(lambda f, e: out.update(found=f, error=e))
    worker.run()
    check("neither column is picked", not out["found"], str(out["found"]))
    check("and the reason is that they cannot be told apart",
          "cannot be told apart" in out["error"], out["error"])
