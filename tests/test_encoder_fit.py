"""Measuring counts per millimetre, and knowing whether the measurement worked.

The first calibration run on this rig fitted x at 14,341.4 counts/mm while y
and z came back at 14,399.9 and 14,399.8 -- agreeing with each other to seven
parts per million and disagreeing with x by 0.41%. Three LTS300Cs do not have
different gearing. What differed was the run: the fit divided the counts by the
distance the move was ASKED for, and x ended 81 um short of it.

0.41% is 1.2 mm across the 300 mm axis, in a column labelled X_mm.

Two changes come out of that, and these pin both: fit against the position the
controller reports rather than the one it was commanded, and take enough points
that the fit has a residual to be judged by.
"""

import numpy as np

from octobee.motion import encoder as oenc
from tests.helpers import (
    check,
)

TRUE_SCALE = 14_400.0            # what y and z both measured, to 7 ppm
BASE_COUNTS = 1_234_000.0        # the counter is incremental; it starts somewhere


def _counts(positions_mm, scale=TRUE_SCALE, columns=3, axis_col=0,
            base=BASE_COUNTS, noise_counts=0.0, seed=1):
    """(n, columns) counts for a carriage really at `positions_mm`.

    Every other column holds a counter standing still, which is what the two
    axes that are not being driven actually look like.
    """
    rng = np.random.default_rng(seed)
    pos = np.asarray(positions_mm, dtype=float)
    out = np.tile(np.arange(columns, dtype=float) * 500_000.0, (len(pos), 1))
    out[:, axis_col] = base + scale * pos
    if noise_counts:
        out[:, axis_col] += rng.normal(0, noise_counts, len(pos))
    return out


def test_fitting_against_the_command_is_what_broke_x():
    """The exact failure, reproduced, and then not reproduced."""
    print("\nfit against measured, not commanded")
    undershoot_mm = 0.0814          # what x really did

    # The two-point run as it was: one 20 mm move that lands short.
    start = 150.0
    before = _counts([start])[0]
    after = _counts([start + 20.0 - undershoot_mm])[0]
    _col, old_scale, _note = oenc.fit_scale(before, after, 20.0)
    check("dividing by the commanded distance reproduces x's number",
          abs(old_scale - 14_341.4) < 1.0, f"{old_scale:,.1f} counts/mm")
    check("which is 0.41% low, or 1.2 mm across a 300 mm axis",
          abs((old_scale / TRUE_SCALE - 1) * 100 + 0.406) < 0.01,
          f"{(old_scale / TRUE_SCALE - 1) * 100:+.3f} %")

    # The same move, fitted against where the controller says it ended up.
    fit = oenc.fit_axis([start, start + 20.0 - undershoot_mm],
                        np.vstack([before, after]))
    check("fitting against the measured position gets it exactly right",
          abs(fit.counts_per_mm - TRUE_SCALE) < 0.1,
          f"{fit.counts_per_mm:,.1f} counts/mm")
    check("and two points still admit they have no residual",
          fit.residual_um is None, str(fit.residual_um))


def test_a_stage_that_stops_short_moves_the_residual_not_the_slope():
    """The property that makes more than two points worth taking.

    One bad stop among eleven is a point off the line. With two points it IS
    the line, and there is nothing in the result to say so.
    """
    print("\nmulti-point robustness")
    targets = np.linspace(100.0, 200.0, 11)
    real = targets.copy()
    real[5] -= 0.080                    # one stop 80 um short, as x's was

    fit = oenc.fit_axis(real, _counts(real))
    check("the slope is unmoved by one bad stop",
          abs(fit.counts_per_mm - TRUE_SCALE) < 0.5,
          f"{fit.counts_per_mm:,.1f} counts/mm")

    # Fitting the same counts against the COMMANDED targets is the mistake
    # this replaces, and there the bad stop lands in the residual instead --
    # which is the point: it becomes visible rather than silent.
    against_command = oenc.fit_axis(targets, _counts(real))
    check("fitting against the command shows it as a residual",
          against_command.residual_um > 10.0,
          f"{against_command.residual_um:.1f} um rms")
    check("and names the worst point",
          against_command.worst_um > 50.0,
          f"{against_command.worst_um:.1f} um worst")


def test_a_clean_run_reports_a_small_residual():
    """A good fit has to be distinguishable from a bad one by its number."""
    print("\nresidual on a clean run")
    pos = np.linspace(50.0, 250.0, 11)
    # A quarter of a count of reading noise -- far below one encoder edge.
    fit = oenc.fit_axis(pos, _counts(pos, noise_counts=0.25))
    check("the scale comes back", abs(fit.counts_per_mm - TRUE_SCALE) < 0.5,
          f"{fit.counts_per_mm:,.1f}")
    check("and the residual is sub-micrometre",
          fit.residual_um < 1.0, f"{fit.residual_um:.3f} um rms")
    check("the span and point count are recorded with it",
          abs(fit.span_mm - 200.0) < 1e-6 and fit.n_points == 11,
          f"{fit.span_mm} mm, {fit.n_points} points")
    check("and describe() says all three",
          "14,400" in fit.describe() and "200 mm" in fit.describe()
          and "residual" in fit.describe(), fit.describe())


def test_the_fit_refuses_what_it_cannot_tell_apart():
    """A refusal is worth more than a plausible number."""
    print("\nfit refusals")
    pos = np.linspace(0.0, 100.0, 11)

    still = np.tile(np.array([7.0, 8.0, 9.0]), (len(pos), 1))
    fit = oenc.fit_axis(pos, still)
    check("a stream where nothing moved is refused",
          not fit and "nothing here is wired" in fit.why, fit.why)

    both = _counts(pos)
    both[:, 1] = BASE_COUNTS + TRUE_SCALE * 0.9 * pos     # a second column too
    fit = oenc.fit_axis(pos, both)
    check("two columns moving together is refused, not guessed between",
          not fit and "cannot be told apart" in fit.why, fit.why)

    fit = oenc.fit_axis([10.0] * 5, _counts([10.0] * 5))
    check("an axis that never moved is refused",
          not fit and "never left" in fit.why, fit.why)

    fit = oenc.fit_axis([1.0, 2.0, 3.0], _counts([1.0, 2.0]))
    check("positions and counts that do not pair up are refused",
          not fit and "against" in fit.why, fit.why)

    fit = oenc.fit_axis([1.0], _counts([1.0]))
    check("one standstill is not a line",
          not fit and "at least two" in fit.why, fit.why)

    check("a refused fit carries no scale to be used by accident",
          fit.to_spec() is None and not bool(fit))


def test_the_two_directions_check_each_other():
    """Out and back must agree about the same place, or something is wrong."""
    print("\ndirection agreement")
    out_pos = np.linspace(80.0, 180.0, 11)
    back_pos = out_pos[::-1]

    clean_out = oenc.fit_axis(out_pos, _counts(out_pos, noise_counts=0.25))
    clean_back = oenc.fit_axis(back_pos, _counts(back_pos, noise_counts=0.25,
                                                 seed=2))
    offset = oenc.direction_offset_um(clean_out, clean_back)
    check("two counters geared together agree to well under a micrometre",
          offset < 1.0, f"{offset:.3f} um")

    # A return pass whose readings were all taken 40 um into the move.
    late = oenc.fit_axis(back_pos, _counts(back_pos + 0.040))
    check("a reading taken while still moving shows up as an offset",
          abs(oenc.direction_offset_um(clean_out, late) - 40.0) < 2.0,
          f"{oenc.direction_offset_um(clean_out, late):.1f} um")

    check("comparing two different columns measures nothing, and says so",
          oenc.direction_offset_um(
              clean_out, oenc.fit_axis(out_pos, _counts(out_pos, axis_col=2))
          ) is None)
    check("and a refused fit cannot be compared either",
          oenc.direction_offset_um(clean_out, oenc.AxisFit(why="no")) is None)


def test_the_scale_keeps_the_sign_the_axis_is_wired_with():
    """z runs backwards on this rig, and that is a fact to preserve."""
    print("\nsign")
    pos = np.linspace(20.0, 120.0, 11)
    fit = oenc.fit_axis(pos, _counts(pos, scale=-TRUE_SCALE))
    check("a backwards-wired axis keeps its negative scale",
          abs(fit.counts_per_mm + TRUE_SCALE) < 0.5,
          f"{fit.counts_per_mm:,.1f} counts/mm")
    check("and is still chosen as the column that moved",
          fit.column == 0, str(fit.column))
    check("the spec that gets stored carries the sign",
          fit.to_spec()["counts_per_mm"] < 0, str(fit.to_spec()))
