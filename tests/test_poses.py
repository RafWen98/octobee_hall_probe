"""Indexed 90-degree pose capture."""


import numpy as np

from octobee.acq import carrier as ob
from octobee.calib import convert as ocal
from octobee.calib import roll as opc
from octobee.calib import poses as pcap
from octobee.calib import geometry as pgeom
from tests.helpers import (
    _matching_error_pct,
    _synth_pose_sweeps,
    check,
)



def test_posecap():
    print("\n== indexed 90-degree poses ==")
    geom = pgeom.Geometry.load_or_default()
    n_sens = pgeom.N_SENSORS
    gains = 1.0 + np.random.default_rng(1).normal(0, 0.03, n_sens)

    # ---- the claim the README makes: four poses is enough -----------------
    sweep, S_true, b_true = _synth_pose_sweeps(geom, gains)
    sol = opc.solve_roll([sweep("A", 25.0), sweep("B", -25.0),
                          sweep("C", 60.0)], geom, 49.0)
    err = _matching_error_pct(sol, S_true)
    check("four 90-degree poses match sensors as well as a hand roll",
          err < 0.1, f"max inter-sensor error {err:.4f} %")
    check("four poses still identify the axial column and the gauge",
          sol.identified[:, 2].all() and sol.anisotropy_identified)
    check("four poses recover the offsets",
          np.abs(sol.b - b_true).max() * 1e3 < 0.3,
          f"max {np.abs(sol.b - b_true).max() * 1e3:.3f} uT")

    # A V-block that indexes to 88 degrees must be as good as one that indexes
    # to 90 -- otherwise the whole procedure needs a rotary stage.
    sweep_j, S_j, _ = _synth_pose_sweeps(geom, gains, seed=7)
    solj = opc.solve_roll([sweep_j("A", 25.0, 2.0), sweep_j("B", -25.0, 2.0),
                           sweep_j("C", 60.0, 2.0)], geom, 49.0)
    errj = _matching_error_pct(solj, S_j)
    check("+/-2 degrees of indexing error does not matter",
          abs(errj - err) < 0.02, f"{err:.4f} % -> {errj:.4f} %")

    # The arrangement the bench can actually offer: tube axis never leaves the
    # horizontal, and the only moves are the 90 deg roll and carrying the
    # cradle round to another compass bearing. At the local ~68 deg dip a
    # horizontal axis gets its inclination purely from its bearing.
    dip = np.deg2rad(68.0)
    def bearing_alpha(d):
        """Inclination to the roll plane for a tube pointed at bearing d."""
        return np.degrees(np.arcsin(np.cos(dip) * np.cos(np.deg2rad(d))))
    sweep_h, S_h, b_h = _synth_pose_sweeps(geom, gains, seed=13)
    horiz = opc.solve_roll([sweep_h("A", bearing_alpha(0)),      # north
                            sweep_h("B", bearing_alpha(180)),    # south
                            sweep_h("C", bearing_alpha(90))],    # east
                           geom, 51.0)
    check("three horizontal bearings alone calibrate the probe",
          _matching_error_pct(horiz, S_h) < 0.1
          and horiz.identified[:, 2].all() and horiz.anisotropy_identified,
          f"matching {_matching_error_pct(horiz, S_h):.4f} %, "
          f"offsets {np.abs(horiz.b - b_h).max() * 1e3:.3f} uT")

    # Rolling at ONE bearing and nothing else leaves the axial column and the
    # offset perfectly degenerate -- on this probe that is chip +Y on all 16
    # sensors, a third of the channels. Matching still works, which is the
    # trap: the number you were mostly after looks fine.
    one = opc.solve_roll([sweep_h("A", bearing_alpha(0))], geom, 51.0)
    check("rolling at a single bearing still matches sensors",
          _matching_error_pct(one, S_h) < 0.15,
          f"matching {_matching_error_pct(one, S_h):.4f} %")
    check("...but says the axial column and the gauge are unidentified",
          not one.identified[:, 2].all() and not one.anisotropy_identified)
    check("...and its offsets are worthless, as advertised",
          np.abs(one.b - b_h).max() * 1e3 > 1.0,
          f"max offset error {np.abs(one.b - b_h).max() * 1e3:.1f} uT")

    # An arrangement that LOOKS fine and is not: tube axis east-west, so the
    # end-for-end flip leaves the axial field at zero both times and buys
    # nothing. A third steeply-tilted sweep still spreads bz enough to clear
    # the leverage test, the residual sits on the noise floor, and the matching
    # figure comes out the best of the lot -- while the offsets are 350x wrong.
    # Only the geometry gives it away: every orientation on the same side of
    # level.
    sweep_t, _S_t, b_t = _synth_pose_sweeps(geom, gains, seed=11)
    trap = opc.solve_roll([sweep_t("A", 0.0), sweep_t("B", 0.0),
                           sweep_t("C", 68.0)], geom, 51.0)
    off_err = np.abs(trap.b - b_t).max() * 1e3
    check("the east-west arrangement really does wreck the offsets",
          off_err > 5.0, f"max offset error {off_err:.1f} uT")
    check("...with a residual that gives no hint of it",
          np.median(trap.residual_mt) * 1e3 < 0.1,
          f"{np.median(trap.residual_mt) * 1e3:.4f} uT")
    check("...and is caught anyway, by every orientation sharing a sign",
          "every orientation has the axial field" in trap.report())
    good = opc.solve_roll([sweep_t("A", 22.0), sweep_t("B", -22.0),
                           sweep_t("C", 68.0)], geom, 51.0)
    check("a real end-for-end flip does not trip that warning",
          "every orientation has the axial field" not in good.report()
          and np.abs(good.b - b_t).max() * 1e3 < 0.3,
          f"max offset error {np.abs(good.b - b_t).max() * 1e3:.3f} uT")

    # ---- the recorder ------------------------------------------------------
    # Block means must be the real means: this is the path that keeps a 20 s
    # dwell from being carried to float64 all at once.
    rng = np.random.default_rng(4)
    ai = rng.normal(0, 500, (6400, 32)).astype(np.int16)
    blocks = pcap._block_means(ai, 64)
    check("block means reduce without changing the mean",
          blocks.shape == (64, 32)
          and np.allclose(blocks.mean(axis=0), ai.mean(axis=0, dtype=np.float64)),
          f"{blocks.shape}")
    check("block means survive fewer samples than blocks",
          pcap._block_means(ai[:10], 64).shape[0] == 10)

    # capture_pose against a stubbed carrier: the pose row must be the field
    # that was there, and the drain capture must be thrown away rather than
    # averaged in -- it is the one that can hold the rotation itself.
    R_nom = geom.rotations()
    truth = np.array([49.0e-3 * v for v in (0.3, -0.5, 0.81)])
    state = {"n": 0}

    class _Lay:
        fs_hz, adc_range = 200000.0, "PM10V"
        volts_per_count = 20.0 / 65536

    def fake_capture_all(hosts, seconds, take_over=True, verbose=True):
        state["n"] += 1
        rubbish = state["n"] == 1          # the drain must discard this one
        out = {}
        for bi, h in enumerate(hosts):
            a = np.zeros((max(64, int(2000 * seconds)), 32))
            for s in range(8):
                gi = bi * 8 + s
                m = (R_nom[gi].T @ (truth * (50.0 if rubbish else 1.0)))
                volts = m * 1e-3 * ocal.ob.RANGE_TO_VPT[20.0] + 2.2
                counts = volts / _Lay.volts_per_count
                a[:, s * 4 + 0] = counts[2]      # ch 4k+1 = Bz
                a[:, s * 4 + 1] = counts[1]      # ch 4k+2 = By
                a[:, s * 4 + 2] = counts[0]      # ch 4k+3 = Bx
                a[:, s * 4 + 3] = 2.2 / _Lay.volts_per_count
            out[h] = ({"ai": a, "sam_cnt": np.arange(a.shape[0]),
                       "temp_raw": np.zeros(8, dtype=np.int64)}, _Lay())
        return out

    real = ob.capture_all
    try:
        ob.capture_all = fake_capture_all
        cal = ocal.Calibration(ranges_mt=np.full(n_sens, 20.0))
        row, st = pcap.capture_pose(["a", "b"], 2.0, cal, chunk_s=1.0,
                                    drain_s=1.0)
    finally:
        ob.capture_all = real

    want = np.array([R_nom[i].T @ truth for i in range(n_sens)])
    check("a pose reads back the field that was there",
          np.abs(row - want).max() * 1e3 < 0.5,
          f"max {np.abs(row - want).max() * 1e3:.3f} uT")
    check("the drain capture is discarded, not averaged in",
          np.abs(row).max() < 2 * np.abs(want).max(),
          f"|row| max {np.abs(row).max() * 1e3:.1f} uT vs "
          f"{np.abs(want).max() * 1e3:.1f} uT expected")
    check("a pose reports how many chunks and blocks it averaged",
          st["n_chunks"] == 2 and st["n_blocks"] == 2 * pcap.BLOCKS_PER_CHUNK,
          f"{st['n_chunks']} chunks, {st['n_blocks']} blocks")

    # ---- the location survey ----------------------------------------------
    # Two poses 180 deg apart. Offsets cancel in the difference, so what is
    # left is the field each sensor actually sat in -- and under a uniform
    # field that is one number, whatever each chip's orientation, because a
    # magnitude is rotation-invariant.
    live_all = np.ones(n_sens, bool)
    B_t = np.array([47.3e-3, 0.0, 19.0e-3])
    offs = np.random.default_rng(6).normal(0, 0.3, (n_sens, 3))   # 300 uT
    # pert is peak-to-peak across the 16 sensors, so the survey's
    # (max - min) / median has something exact to be compared against.
    def pose(sign, pert=0.0):
        return np.array(
            [R_nom[i].T @ (sign * B_t * (1.0 + pert * (i - 7.5) / 15.0)) + offs[i]
             for i in range(n_sens)])

    _, med, spread = pcap.survey_uniformity(pose(+1), pose(-1), live_all)
    check("a uniform field surveys flat, whatever the offsets",
          spread < 0.01 and abs(med - 51.0) < 0.5,
          f"median {med:.2f} uT, spread {spread:.4f} %")

    # A 10 % gradient across the head must read back as ~10 %, not be hidden
    # by the offsets it sits on top of.
    _, _med_g, spread_g = pcap.survey_uniformity(pose(+1, 0.10), pose(-1, 0.10),
                                                live_all)
    check("a gradient across the head shows up at its true size",
          9.0 < spread_g < 11.0, f"spread {spread_g:.2f} %")

    # An index short of 180 deg scales every sensor the same way, so it moves
    # the median but must NOT invent or hide non-uniformity.
    c, s = np.cos(np.deg2rad(170.0)), np.sin(np.deg2rad(170.0))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    sloppy = np.array([R_nom[i].T @ (rot @ B_t) + offs[i] for i in range(n_sens)])
    _, med_s, spread_s = pcap.survey_uniformity(pose(+1), sloppy, live_all)
    check("a sloppy index changes the median, not the spread",
          spread_s < 0.01 and med_s < med,
          f"median {med_s:.2f} uT (vs {med:.2f}), spread {spread_s:.4f} %")
