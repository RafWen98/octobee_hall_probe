"""Earth-field roll calibration, graded against a known truth."""

import json
import os
import tempfile

import numpy as np

from octobee.calib import convert as ocal
from octobee.calib import roll as opc
from octobee.calib import geometry as pgeom
from tests.helpers import (
    _matching_error_pct,
    _matching_error_pct_subset,
    _synth_sweeps,
    check,
)



def test_posecal():
    print("\n== Earth-field roll calibration ==")
    geom = pgeom.Geometry.load_or_default()
    n_sens = pgeom.N_SENSORS
    rng = np.random.default_rng(1)
    gains = 1.0 + rng.normal(0, 0.03, n_sens)

    sweep, S_true, b_true = _synth_sweeps(geom, gains)
    A, B, C = sweep("A", 25.0), sweep("B", -25.0), sweep("C", 60.0)

    # ---- the whole point: sensors matched to each other -------------------
    sol = opc.solve_roll([A, B, C], geom, 49.0)
    check("roll solve matches sensors to each other",
          _matching_error_pct(sol, S_true) < 0.1,
          f"max inter-sensor error {_matching_error_pct(sol, S_true):.4f} %")
    check("roll solve recovers the offsets",
          np.abs(sol.b - b_true).max() * 1e3 < 0.5,
          f"max {np.abs(sol.b - b_true).max() * 1e3:.3f} uT")
    check("roll solve recovers the orientations",
          max(np.degrees(np.arccos(np.clip(
              (np.trace(sol.decompose()[0][i] @ geom.rotations()[i].T) - 1) / 2,
              -1, 1))) for i in range(n_sens)) < 0.2)
    check("fit residual sits at the noise floor",
          np.median(sol.residual_mt) * 1e3 < 0.25,
          f"{np.median(sol.residual_mt) * 1e3:.4f} uT")

    # ---- degeneracy guards -----------------------------------------------
    # These are the tests that stop a structurally unidentifiable direction
    # from being quietly fitted to noise. The residual is at the noise floor
    # in EVERY case below, so residual alone can never catch these.
    one = opc.solve_roll([A], geom, 49.0)
    check("one orientation reports the axial column as unidentified",
          not one.identified[:, 2].all())
    check("one orientation still matches sensors to each other",
          _matching_error_pct(one, S_true) < 0.1,
          f"max {_matching_error_pct(one, S_true):.4f} %")

    flip = opc.solve_roll([A, B], geom, 49.0)
    check("an end-for-end flip identifies the axial column",
          flip.identified[:, 2].all())
    check("an end-for-end flip does NOT claim the anisotropy gauge",
          not flip.anisotropy_identified)
    check("an end-for-end flip gives full offset leverage",
          flip.offset_leverage > 0.6, f"{flip.offset_leverage:.2f}")

    two_az = opc.solve_roll([A, C], geom, 49.0)
    check("two same-sign azimuths flag weak offset leverage",
          two_az.offset_leverage < 0.6, f"{two_az.offset_leverage:.2f}")

    check("three orientations identify everything",
          sol.identified[:, 2].all() and sol.anisotropy_identified
          and sol.offset_leverage > 0.6)

    # ---- the anisotropy gauge is real, and only C can see it -------------
    sweep2, S2, _ = _synth_sweeps(geom, gains, axial_ratio=0.95, seed=5)
    A2, B2, C2 = sweep2("A", 25.0), sweep2("B", -25.0), sweep2("C", 60.0)

    def axial_ratio(sv):
        _, S = sv.decompose()
        return float(np.median(0.5 * (S[:, 0, 0] + S[:, 2, 2]))
                     / np.median(S[:, 1, 1]))

    full = opc.solve_roll([A2, B2, C2], geom, 49.0)
    check("a second cradle azimuth recovers axial sensitivity",
          abs(axial_ratio(full) - 0.95) < 0.01,
          f"got {axial_ratio(full):.4f}, truth 0.9500")
    only_flip = opc.solve_roll([A2, B2], geom, 49.0)
    check("without it the axial sensitivity is wrong, and says so",
          abs(axial_ratio(only_flip) - 0.95) > 0.02
          and not only_flip.anisotropy_identified,
          f"got {axial_ratio(only_flip):.4f}")
    assumed = opc.solve_roll([A2, B2], geom, 49.0,
                             anisotropy="assume_isotropic")
    check("assume_isotropic forces isotropy and labels the assumption",
          abs(axial_ratio(assumed) - 1.0) < 0.01
          and "assum" in assumed.notes.lower())

    # ---- the common-mode claim -------------------------------------------
    check("the anisotropy gauge is common mode, so matching survives it",
          _matching_error_pct(only_flip, S2) < 0.1,
          f"max {_matching_error_pct(only_flip, S2):.4f} %")

    # ---- what it says about the geometry ---------------------------------
    ids = opc.identify_faces(sol, geom)
    check("solved face mapping agrees with probe_geometry.json",
          ids["agrees"], f"mismatch {ids['mismatch']}")
    check("a uniform ambient reads as uniform",
          opc.ambient_uniformity(sol) < 0.01,
          f"{opc.ambient_uniformity(sol) * 100:.3f} % of |B|")

    # ---- one dead sensor must not take the other fifteen down ------------
    # S16's analogue ribbon is faulty on this probe, so it fits a response with
    # no rank. Inverting the whole (16,3,3) stack in one call raises on that and
    # loses the entire calibration, which is exactly the wrong failure mode.
    flat = []
    for sw in (A, B, C):
        m = sw.b_mt.copy()
        m[:, 15, :] = 0.0
        flat.append(opc.RollSweep(m, tag=sw.tag))
    try:
        dsol = opc.solve_roll(flat, geom, 49.0)
        ok = True
    except Exception:
        dsol = None
        ok = False
    check("a sensor with no response does not break the solve", ok)
    if dsol is not None:
        check("the unusable sensor is named, not silently averaged in",
              "S16" in dsol.singular, f"singular {dsol.singular}")
        check("the other fifteen still match each other",
              _matching_error_pct_subset(dsol, S_true, range(15)) < 0.1,
              f"max {_matching_error_pct_subset(dsol, S_true, range(15)):.4f} %")
        check("the report calls the dead sensor out",
              "NO RESPONSE" in dsol.report())

    # ---- large sensor offsets must not mirror the solution ---------------
    # Measured on the real probe 2026-08-19: ambient readings run 0.10-0.57 mT
    # while the Earth's field is 0.049 mT, so the offsets are 2-12x the signal.
    # Seeding the field inclination from the raw mean then estimates the OFFSET
    # rather than the field, and the iteration settles into a mirrored frame --
    # matching still looks fine and the residual sits at the noise floor, but
    # det(S) goes negative and the offsets come back wrong by more than their
    # own size. The seed now uses the transverse VARIANCE (which a constant
    # cannot affect) plus differences between orientation means (which cancel
    # the offset exactly), so it has to hold well past anything real.
    for off_mt in (0.02, 0.3, 0.6, 2.0):
        rngo = np.random.default_rng(23)
        go = 1.0 + rngo.normal(0, 0.03, n_sens)
        So = np.array([np.eye(3) / go[i] for i in range(n_sens)])
        Mo = np.array([np.linalg.inv(geom.rotations()[i] @ So[i])
                       for i in range(n_sens)])
        bo = rngo.normal(0, off_mt, (n_sens, 3))

        # rngo/Mo/bo bound as defaults, not captured: this closure is called
        # inside the enclosing loop, so a late-bound name would silently pick
        # up the NEXT iteration's matrices and the test would grade one case
        # against another's truth.
        def osweep(tag, incl, n=2500, rngo=rngo, Mo=Mo, bo=bo):
            a = np.deg2rad(incl)
            bt, bz = 49.0e-3 * np.cos(a), 49.0e-3 * np.sin(a)
            phi = np.sort(rngo.uniform(0, 4 * np.pi, n))
            B = np.column_stack([bt * np.cos(phi), bt * np.sin(phi),
                                 np.full(n, bz)])
            m = np.einsum("sij,tj->tsi", Mo, B) + bo[None]
            return opc.RollSweep(m + rngo.normal(0, 3e-3, m.shape), tag=tag)

        osol = opc.solve_roll([osweep("A", 25.), osweep("B", -25.),
                               osweep("C", 60.)], geom, 49.0)
        _, Sg = osol.decompose()
        dets = np.array([np.linalg.det(Sg[i]) for i in range(n_sens)])
        check(f"offsets of {off_mt * 1e3:.0f} uT still give a proper frame",
              (dets > 0).all(), f"min det {dets.min():.3f}")
        check(f"offsets of {off_mt * 1e3:.0f} uT are recovered",
              np.abs(osol.b - bo).max() * 1e3 < 1.0,
              f"max {np.abs(osol.b - bo).max() * 1e3:.3f} uT")

    # ---- S16's ACTUAL fault, not a convenient version of it --------------
    # ch29/30/32 (Bz, By, VCM) pinned at negative full scale while ch31 (Bx)
    # reads normally. This is nastier than a sensor that saw nothing: the railed
    # channels are CONSTANT, so the model fits them perfectly with a big offset.
    # The residual comes back indistinguishable from a healthy sensor and the
    # condition number stays near 1e4, so neither residual nor conditioning can
    # see it. Only the solved gain gives it away.
    cal20 = ocal.Calibration(ranges_mt=np.full(n_sens, 20.0))
    vpc = 20.0 / 65536
    rng16 = np.random.default_rng(17)
    g16 = 1.0 + rng16.normal(0, 0.03, n_sens)
    R16 = geom.rotations()
    S16t = np.array([np.eye(3) / g16[i] for i in range(n_sens)])
    M16 = np.array([np.linalg.inv(R16[i] @ S16t[i]) for i in range(n_sens)])

    def railed_sweep(tag, incl_deg, fault=True, n=2500):
        a = np.deg2rad(incl_deg)
        bt, bz = 49.0e-3 * np.cos(a), 49.0e-3 * np.sin(a)
        phi = np.sort(rng16.uniform(0, 4 * np.pi, n))
        B = np.column_stack([bt * np.cos(phi), bt * np.sin(phi), np.full(n, bz)])
        v = np.zeros((n, n_sens, 4))
        v[..., :3] = (np.einsum("sij,tj->tsi", M16, B) / 1e3
                      * cal20.volts_per_tesla[None, :, None] + 2.2)
        v[..., 3] = 2.2
        if fault:
            rail = -32768 * vpc
            v[:, 15, 2] = rail        # Bz
            v[:, 15, 1] = rail        # By
            v[:, 15, 3] = rail        # VCM
            #  Bx (index 0) deliberately left alive
        v += rng16.normal(0, 3e-5, v.shape)
        return opc.RollSweep(
            cal20.to_mt(v, apply_zero=False, apply_gain=False, apply_matrix=False),
            tag=tag, ranges_mt=cal20.ranges_mt.copy())

    faulted = [railed_sweep(t, i) for t, i in (("A", 25.), ("B", -25.), ("C", 60.))]
    fsol = opc.solve_roll(faulted, geom, 49.0)          # NOT told about S16
    check("the real S16 railed fault is caught without being told",
          fsol.unusable() == ["S16"], f"flagged {fsol.unusable()}")
    check("neither residual nor conditioning could have caught it",
          abs(fsol.residual_mt[15] - np.median(fsol.residual_mt[:15]))
          < 0.5 * np.median(fsol.residual_mt[:15])
          and np.linalg.cond(fsol.M[15]) < opc.PoseSolution.SINGULAR_COND,
          f"resid {fsol.residual_mt[15]*1e3:.3f} vs "
          f"{np.median(fsol.residual_mt[:15])*1e3:.3f} uT, "
          f"cond {np.linalg.cond(fsol.M[15]):.1e}")
    tr = np.array([np.sqrt(abs(fsol.decompose()[1][i][0, 0]
                               * fsol.decompose()[1][i][2, 2])) for i in range(15)])
    tt = np.array([np.sqrt(S16t[i][0, 0] * S16t[i][2, 2]) for i in range(15)])
    rel = (tr / np.median(tr)) / (tt / np.median(tt))
    check("one railed sensor does not poison the other fifteen",
          np.abs(rel - 1).max() * 100 < 0.5,
          f"S1-S15 match {np.abs(rel - 1).max() * 100:.4f} %")

    clean = [railed_sweep(t, i, fault=False)
             for t, i in (("A", 25.), ("B", -25.), ("C", 60.))]
    check("a healthy probe raises no false alarm",
          opc.solve_roll(clean, geom, 49.0).unusable() == [])

    # ---- refuses to mix ranges -------------------------------------------
    lo = opc.RollSweep(A.b_mt, tag="A", ranges_mt=np.full(n_sens, 40.0))
    hi = opc.RollSweep(B.b_mt, tag="B", ranges_mt=np.full(n_sens, 400.0))
    try:
        opc.solve_roll([lo, hi], geom, 49.0)
        check("refuses to combine sweeps taken at different ranges", False)
    except ValueError:
        check("refuses to combine sweeps taken at different ranges", True)

    # ---- range transfer ---------------------------------------------------
    base = rng.normal(0, 5.0, (200, n_sens, 3))
    ratio_true = 1.82
    r, skipped, notes = opc.range_transfer(base, base * ratio_true,
                                           min_signal_mt=0.5)
    check("range transfer recovers a known gain ratio",
          np.allclose(r[np.abs(base.mean(axis=0)) > 0.5], ratio_true, atol=1e-6),
          f"{ratio_true}x")
    check("range transfer skips channels with too little signal",
          isinstance(skipped, list))
    check("and warns that it was given no zero-field baseline",
          len(notes) == 1 and "offsets are INCLUDED" in notes[0], str(notes))

    # Real captures are field PLUS offset, and the offset is a different
    # number of mT at each gain. A few mT of magnet against 0.4 mT of offset
    # puts the bare ratio out by tens of per cent.
    field = rng.normal(0, 5.0, (n_sens, 3))
    z_lo = rng.normal(0, 0.4, (n_sens, 3))
    z_hi = rng.normal(0, 0.4, (n_sens, 3))
    b_lo = np.tile(field + z_lo, (50, 1, 1))
    b_hi = np.tile(field * 1.02 + z_hi, (50, 1, 1))
    strong = np.abs(field) > 1.0
    r, _skipped, notes = opc.range_transfer(b_lo, b_hi)
    check("without baselines the offsets wreck the ratio, and it says so",
          np.abs(r[strong & (np.abs(field + z_lo) > 1.0)] - 1.02).max() > 0.1
          and notes, f"worst error "
          f"{np.abs(r[strong & (np.abs(field + z_lo) > 1.0)] - 1.02).max():.2f}")
    r, skipped, notes = opc.range_transfer(
        b_lo, b_hi, zero_lo_mt=np.tile(z_lo, (20, 1, 1)), zero_hi_mt=z_hi)
    check("with a zero-field baseline per range the ratio is right",
          np.allclose(r[strong], 1.02, atol=1e-9) and not notes,
          f"worst error {np.abs(r[strong] - 1.02).max():.2e}")
    check("and the signal gate is applied to the field, not field + offset",
          sorted(skipped) == sorted(
              f"S{i + 1}.{opc.AXES[k]}" for i in range(n_sens)
              for k in range(3) if not strong[i, k]))
    _r, _s, notes = opc.range_transfer(b_lo, b_hi, zero_lo_mt=z_lo)
    check("one baseline of the two is still warned about",
          len(notes) == 1 and "high range" in notes[0], str(notes))


def test_posecal_persistence():
    print("\n== calibration file compatibility ==")
    geom = pgeom.Geometry.load_or_default()
    n_sens = pgeom.N_SENSORS
    rng = np.random.default_rng(2)

    # A v1 file has no "version" and no "matrix", and its zero was stored
    # AFTER the gain. Loading it must reproduce v1 numbers exactly, or every
    # capture in captures/ silently changes meaning.
    gain = 1.0 + rng.normal(0, 0.05, (n_sens, 3))
    zero = rng.normal(0, 0.1, (n_sens, 3))
    v1 = {"ranges_mt": [40.0] * n_sens, "zero_mt": zero.tolist(),
          "gain_corr": gain.tolist(), "subtract_vcm": True, "dead": [],
          "notes": "v1"}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "v1.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(v1, f)
        cal = ocal.Calibration.load(path)

        volts = rng.normal(2.2, 0.05, (50, n_sens, 4))
        got = cal.to_mt(volts)
        raw = ((volts[..., :3] - volts[..., 3:4])
               / cal.volts_per_tesla[None, :, None] * 1e3)
        want_v1 = raw * gain[None] - zero[None]        # the old ordering
        check("a v1 calibration.json still converts bit for bit",
              np.allclose(got, want_v1, rtol=0, atol=1e-12),
              f"max diff {np.abs(got - want_v1).max():.2e} mT")

        cal.save(os.path.join(d, "v2.json"))
        back = ocal.Calibration.load(os.path.join(d, "v2.json"))
        check("a v2 calibration.json round trips",
              np.allclose(back.to_mt(volts), got, atol=1e-12)
              and back.VERSION == 2)

    # Applying a pose solution.
    gains = 1.0 + rng.normal(0, 0.03, n_sens)
    sweep, _S_true, _b_true = _synth_sweeps(geom, gains, seed=9)
    sol = opc.solve_roll([sweep("A", 25.0), sweep("B", -25.0),
                          sweep("C", 60.0)], geom, 49.0)
    cal = ocal.Calibration(ranges_mt=np.full(n_sens, 20.0))
    cal.gain_corr = np.full((n_sens, 3), 1.5)
    cal.apply_pose_solution(sol)
    check("applying a pose solution installs the matrix", cal.has_matrix)
    check("applying a pose solution clears the magnet trim",
          np.allclose(cal.gain_corr, 1.0))
    check("applying a pose solution installs the solved offset",
          np.allclose(cal.zero_mt, sol.b))

    # A sensor the solve could not believe must be left alone, not written.
    # S16 flat is the simple version of its real fault: the solve returns a
    # rank-deficient response for it, and installing that -- and wiping its
    # magnet trim on the way -- replaces an uncorrected sensor with a wrong one.
    flat = []
    for tag, incl in (("A", 25.0), ("B", -25.0), ("C", 60.0)):
        m = sweep(tag, incl).b_mt.copy()
        m[:, 15, :] = 0.0
        flat.append(opc.RollSweep(m, tag=tag))
    dsol = opc.solve_roll(flat, geom, 49.0)
    cal_d = ocal.Calibration(ranges_mt=np.full(n_sens, 20.0))
    cal_d.gain_corr = np.full((n_sens, 3), 1.5)
    cal_d.zero_mt[15] = [0.3, -0.2, 0.1]
    cal_d.matrix[15] = np.diag([1.01, 0.99, 1.02])
    lines = cal_d.apply_pose_solution(dsol)
    check("an unusable sensor keeps its zero, trim and matrix",
          dsol.unusable() == ["S16"]
          and np.allclose(cal_d.zero_mt[15], [0.3, -0.2, 0.1])
          and np.allclose(cal_d.gain_corr[15], 1.5)
          and np.allclose(cal_d.matrix[15], np.diag([1.01, 0.99, 1.02])),
          f"zero {cal_d.zero_mt[15]}, gain {cal_d.gain_corr[15]}")
    check("the other fifteen are installed and their trim cleared",
          np.allclose(cal_d.gain_corr[:15], 1.0)
          and np.allclose(cal_d.zero_mt[:15], dsol.b[:15])
          and np.allclose(cal_d.matrix[:15], dsol.decompose()[1][:15]))
    check("and the returned lines name the sensor left alone",
          any("S16" in ln and "untouched" in ln for ln in lines), str(lines))

    sol.ranges_mt = np.full(n_sens, 400.0)
    try:
        cal.apply_pose_solution(sol)
        check("refuses a pose solution solved at another range", False)
    except ValueError:
        check("refuses a pose solution solved at another range", True)

    # The correction must actually undo the sensors it was solved from.
    cal2 = ocal.Calibration(ranges_mt=np.full(n_sens, 20.0))
    sol2 = opc.solve_roll([sweep("A", 25.0), sweep("B", -25.0),
                           sweep("C", 60.0)], geom, 49.0)
    cal2.apply_pose_solution(sol2)
    R, _ = sol2.decompose()
    corrected = np.einsum("sij,tsj->tsi", cal2.matrix,
                          sweep("A", 25.0).b_mt - sol2.b[None])
    tube = np.einsum("sij,tsj->tsi", R, corrected)
    spread = np.std(np.linalg.norm(tube, axis=2), axis=1).mean()
    check("corrected sensors agree on |B| in the tube frame",
          spread / 49.0e-3 < 0.01, f"spread {spread / 49.0e-3 * 100:.3f} % of |B|")


def _reconstruction_ut(sol, sweeps):
    """RMS of data - (M @ B + b), built only from what the solution RETURNS.

    Not sol.residual_mt: the point is to check that M, bt, bz and phis still
    describe the data together after the gauge steps, and a stored residual
    cannot show that.
    """
    acc, n = 0.0, 0
    for sw, phi in zip(sweeps, sol.phis):
        g = sol.tags.index(sw.tag)
        B = np.column_stack([sol.bt[g] * np.cos(phi), sol.bt[g] * np.sin(phi),
                             np.full(len(phi), sol.bz[g])])
        pred = np.einsum("sij,tj->tsi", sol.M, B) + sol.b[None]
        acc += float(((sw.reduced() - pred) ** 2).sum())
        n += pred.size
    return np.sqrt(acc / n) * 1e3


def _median_diag(sol):
    _, S = sol.decompose()
    return np.median(S[:, [0, 1, 2], [0, 1, 2]], axis=0)


def test_roll_absolute_scale():
    """b_earth_ut sets the scale only when the sweeps can support it.

    Every sensor reads 20 % high, so the right correction is 1/1.2 = 0.8333 on
    all three axes. Three inclinations find it. One, or a flip pair, cannot --
    |B| gives one equation for the transverse and axial scales together -- and
    used to hand back some other scale under a docstring that said b_earth_ut
    fixed it.
    """
    print("\n== roll calibration: absolute scale ==")
    geom = pgeom.Geometry.load_or_default()
    n_sens = pgeom.N_SENSORS
    sweep, _S_true, _b = _synth_sweeps(geom, np.full(n_sens, 1.2))
    A, B, C = sweep("A", 25.0), sweep("B", -25.0), sweep("C", 60.0)
    want = 1.0 / 1.2

    full = opc.solve_roll([A, B, C], geom, 49.0)
    check("three inclinations recover the absolute scale from |B|",
          np.abs(_median_diag(full) - want).max() < 0.002
          and full.scale_identified, f"S diag {_median_diag(full).round(4)}")
    check("and the report says where the scale came from",
          "from |B| = 49 uT" in full.report() and "NOMINAL" not in full.report())

    for name, sweeps in (("one sweep", [A]), ("a flip pair", [A, B])):
        sol = opc.solve_roll(sweeps, geom, 49.0)
        check(f"{name} does not claim an absolute scale",
              not sol.scale_identified
              and "NOMINAL -- not derived from b_earth_ut" in sol.report(),
              f"S diag {_median_diag(sol).round(4)}, want {want:.4f}")
        check(f"{name}: and indeed the scale it returns is not the right one",
              abs(_median_diag(sol)[0] - want) > 0.01,
              f"transverse {_median_diag(sol)[0]:.4f}, truth {want:.4f}")

    # What "nominal" means: a probe reading 10 % low everywhere, rolled once,
    # comes back with a transverse correction of 1.00, not the 1.11 it needs.
    low, _S, _b = _synth_sweeps(geom, np.full(n_sens, 0.9), seed=4)
    nominal = opc.solve_roll([low("A", 25.0)], geom, 49.0)
    check("an unidentified scale is the nominal one",
          abs(_median_diag(nominal)[0] - 1.0) < 0.005
          and not nominal.scale_identified,
          f"transverse {_median_diag(nominal)[0]:.4f}, truth {1 / 0.9:.4f}")

    # The flag survives a save and a load, and an older file without it falls
    # back on the gauge flag, which is the same condition.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sol.json")
        opc.solve_roll([A, B], geom, 49.0).save(path)
        back = opc.PoseSolution.load(path)
        check("scale_identified round trips", not back.scale_identified)
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        doc.pop("scale_identified")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        check("a solution saved before the flag reads it off the gauge flag",
              not opc.PoseSolution.load(path).scale_identified)

    # ---- assume_isotropic: a = c is the missing equation -----------------
    # It used to rescale bt and bz to restore |B| without taking the same
    # factor out of M, so M @ B stopped reproducing the data and every gain
    # was out by that factor, under a residual that was never recomputed.
    iso = opc.solve_roll([A, B], geom, 49.0, anisotropy="assume_isotropic")
    check("assume_isotropic with a flip pair recovers the absolute scale",
          np.abs(_median_diag(iso) - want).max() < 0.002
          and iso.scale_identified, f"S diag {_median_diag(iso).round(4)}")
    check("and the returned M, field and angles still reproduce the data",
          _reconstruction_ut(iso, [A, B]) < 0.25,
          f"{_reconstruction_ut(iso, [A, B]):.3f} uT rms")
    check("with a residual that is the returned solution's own",
          abs(np.sqrt(np.mean(iso.residual_mt ** 2)) * 1e3
              - _reconstruction_ut(iso, [A, B])) < 1e-6)
    check("|B| is still the one that was asked for",
          np.allclose(np.hypot(iso.bt, iso.bz), 49.0e-3, rtol=1e-3))
    lone = opc.solve_roll([A], geom, 49.0, anisotropy="assume_isotropic")
    check("assume_isotropic on one sweep still cannot claim a scale",
          not lone.scale_identified and _reconstruction_ut(lone, [A]) < 0.25)


def test_roll_gauge_anchor():
    """The gauge anchor must remove a frame offset, not double it."""
    print("\n== roll calibration: gauge anchor ==")
    geom = pgeom.Geometry.load_or_default()
    n_sens = pgeom.N_SENSORS
    gains = 1.0 + np.random.default_rng(1).normal(0, 0.03, n_sens)
    sweep, _S_true, _b = _synth_sweeps(geom, gains)
    sweeps = [sweep("A", 25.0), sweep("B", -25.0), sweep("C", 60.0)]
    sol = opc.solve_roll(sweeps, geom, 49.0)

    def frame_error_deg(sv):
        R, _ = sv.decompose()
        Rn = geom.rotations()
        return max(np.degrees(np.arccos(np.clip(
            (np.trace(R[i] @ Rn[i].T) - 1) / 2, -1, 1))) for i in range(n_sens))

    # Turn the solved frame 20 deg about the tube axis -- M and the roll
    # angles together, which is the freedom the data cannot see.
    for turn in (20.0, -35.0):
        g = np.deg2rad(turn)
        Rz = np.array([[np.cos(g), -np.sin(g), 0.0],
                       [np.sin(g), np.cos(g), 0.0], [0.0, 0.0, 1.0]])
        sol.M = np.einsum("sij,jk->sik", sol.M, Rz)
        sol.phis = [phi - g for phi in sol.phis]
        offset = frame_error_deg(sol)
        opc._anchor_gauge(sol, geom)
        check(f"a {turn:+.0f} deg frame offset is anchored out",
              abs(offset - abs(turn)) < 0.1 and frame_error_deg(sol) < 0.2,
              f"{offset:.2f} deg before, {frame_error_deg(sol):.2f} deg after")
        check(f"and after {turn:+.0f} deg the roll angles moved with the frame",
              _reconstruction_ut(sol, sweeps) < 0.25,
              f"{_reconstruction_ut(sol, sweeps):.3f} uT rms")


def test_roll_dead_ids():
    """--dead must mean the sensor however it is spelled."""
    print("\n== roll calibration: dead ids ==")
    geom = pgeom.Geometry.load_or_default()
    n_sens = pgeom.N_SENSORS
    gains = 1.0 + np.random.default_rng(1).normal(0, 0.03, n_sens)
    sweep, _S_true, _b = _synth_sweeps(geom, gains)
    rng = np.random.default_rng(12)
    sweeps = []
    for tag, incl in (("A", 25.0), ("B", -25.0), ("C", 60.0)):
        m = sweep(tag, incl).b_mt.copy()
        # A sensor wandering by many times the signal: left in, it gets a
        # vote on every roll angle.
        m[:, 15, :] += np.cumsum(rng.normal(0, 2e-3, m[:, 15, :].shape), axis=0)
        sweeps.append(opc.RollSweep(m, tag=tag))
    told = opc.solve_roll(sweeps, geom, 49.0, dead=["S16"])
    untold = opc.solve_roll(sweeps, geom, 49.0)
    check("excluding the bad sensor changes the solve at all",
          not np.allclose(told.M[:15], untold.M[:15], atol=1e-6))
    for spelling in ("s16", "S16.Bx", " s16.bx "):
        sol = opc.solve_roll(sweeps, geom, 49.0, dead=[spelling])
        check(f"dead={spelling!r} excludes S16 like 'S16' does",
              np.array_equal(sol.M, told.M))
    check("the report marks it excluded whatever the spelling",
          "(excluded)" in told.report(dead=["s16.Bx"])
          or "(NO RESPONSE)" in told.report(dead=["s16.Bx"]))
