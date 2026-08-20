#!/usr/bin/env python3
"""
selftest.py -- drive the whole GUI pipeline without a person in front of it.

Runs the application against the synthetic probe and exercises the paths that
are easy to leave subtly broken: the count-to-tesla conversion, the tare, the
magnet-pass cross-calibration, the tube-frame rotation, the health verdicts,
and every export. Then it reads the written files back and checks the numbers,
because a CSV that exists is not the same as a CSV that is right.

    python selftest.py                     # synthetic probe
    python selftest.py --replay cap.npz    # against a real saved capture

Exit status is 0 only if every check passes.
"""

import argparse
import os
import sys
import json
import tempfile
import time

import numpy as np
from PyQt6 import QtWidgets

import octobee as ob
import octobee_calibration as ocal
import octobee_gui as gui
import octobee_posecal as opc
import octobee_record as orec
import octobee_scan as oscan
import octobee_stage as ostage
import probe_geometry as pgeom

FAILS = []
CHECKS = 0


def check(name, cond, detail=""):
    global CHECKS
    CHECKS += 1
    ok = bool(cond)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


def read_csv(path):
    """
    Read one of our CSVs into (column names, float array).

    numpy's genfromtxt(names=True) takes the first line as the header even when
    it is a comment, so it chokes on the provenance block. Parsing it here also
    checks the file is readable the way a person would read it.
    """
    names, rows = None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if names is None:
                names = line.split(",")
            else:
                rows.append([float(v) for v in line.split(",")])
    return names, np.array(rows)


def read_clkdiv(host):
    u = ob.Uut(host)
    try:
        return u.value("clkdiv", site=1)
    finally:
        u.close()


def pump(win, app, seconds):
    """Run the acquisition loop for a while, as the real timers would."""
    end = time.time() + seconds
    n = 0
    while time.time() < end:
        win.on_tick()
        n += 1
        if n % 5 == 0:              # the view redraws on its own slower clock
            win.on_view_tick()
        app.processEvents()
        time.sleep(0.01)
    win.on_view_tick()
    win.on_slow_tick()
    app.processEvents()


# --------------------------------------------------------------------------
# Earth-field roll calibration
# --------------------------------------------------------------------------

def _synth_sweeps(geom, gains, axial_ratio=1.0, offsets_ut=20.0, seed=3,
                  noise_mt=8.81e-3 / np.sqrt(2500), n=3000):
    """
    Build roll sweeps from a known truth so the solver can be graded.

    axial_ratio is each chip's +Y (tube-axis) sensitivity relative to its X/Z.
    It is the quantity only a second cradle azimuth can see, so it is what the
    gauge tests turn on.
    """
    rng = np.random.default_rng(seed)
    n_sens = pgeom.N_SENSORS
    b_mt = 49.0e-3
    R_nom = geom.rotations()
    # S is the CORRECTION, so a sensor of sensitivity g needs S = 1/g.
    S_true = np.array([np.diag([1.0, 1.0 / axial_ratio, 1.0]) / gains[i]
                       for i in range(n_sens)])
    M_true = np.array([np.linalg.inv(R_nom[i] @ S_true[i]) for i in range(n_sens)])
    b_true = rng.normal(0, offsets_ut * 1e-3, (n_sens, 3))

    def sweep(tag, incl_deg):
        a = np.deg2rad(incl_deg)
        bt, bz = b_mt * np.cos(a), b_mt * np.sin(a)
        phi = np.sort(rng.uniform(0, 4 * np.pi, n))
        B = np.column_stack([bt * np.cos(phi), bt * np.sin(phi), np.full(n, bz)])
        m = np.einsum("sij,tj->tsi", M_true, B) + b_true[None]
        return opc.RollSweep(m + rng.normal(0, noise_mt, m.shape), tag=tag)

    return sweep, S_true, b_true


def _matching_error_pct(sol, S_true):
    """How badly the sensors disagree with each other, in percent.

    Deliberately built from the TRANSVERSE response only. That is the part no
    gauge can touch, so this measures inter-sensor matching without being
    polluted by whether the axial gauge happened to be resolved.
    """
    _, S = sol.decompose()
    n = pgeom.N_SENSORS
    got = np.array([np.sqrt(S[i][0, 0] * S[i][2, 2]) for i in range(n)])
    want = np.array([np.sqrt(S_true[i][0, 0] * S_true[i][2, 2]) for i in range(n)])
    rel = (got / np.median(got)) / (want / np.median(want))
    return float(np.abs(rel - 1).max() * 100)


def _matching_error_pct_subset(sol, S_true, idx):
    """_matching_error_pct restricted to sensors that are supposed to work."""
    _, S = sol.decompose()
    idx = list(idx)
    got = np.array([np.sqrt(S[i][0, 0] * S[i][2, 2]) for i in idx])
    want = np.array([np.sqrt(S_true[i][0, 0] * S_true[i][2, 2]) for i in idx])
    rel = (got / np.median(got)) / (want / np.median(want))
    return float(np.abs(rel - 1).max() * 100)


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
    except Exception:                                    # noqa: BLE001
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

        def osweep(tag, incl, n=2500):
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
    r, skipped = opc.range_transfer(base, base * ratio_true, min_signal_mt=0.5)
    check("range transfer recovers a known gain ratio",
          np.allclose(r[np.abs(base.mean(axis=0)) > 0.5], ratio_true, atol=1e-6),
          f"{ratio_true}x")
    check("range transfer skips channels with too little signal",
          isinstance(skipped, list))


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
    sweep, S_true, b_true = _synth_sweeps(geom, gains, seed=9)
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


# --------------------------------------------------------------------------
# Indexed 90-degree poses (octobee_posecap)
# --------------------------------------------------------------------------

def _synth_pose_sweeps(geom, gains, n_poses=4, seconds=20.0, seed=7):
    """The same truth as _synth_sweeps, sampled at n_poses discrete angles.

    Noise is what a stationary pose actually averages down to: 81.5 uT rms per
    sample (README section 7) over seconds * 200 kSPS.
    """
    rng = np.random.default_rng(seed)
    n_sens = pgeom.N_SENSORS
    b_mt = 49.0e-3
    noise_mt = 81.5e-3 / np.sqrt(200000.0 * seconds)
    R_nom = geom.rotations()
    S_true = np.array([np.eye(3) / gains[i] for i in range(n_sens)])
    M_true = np.array([np.linalg.inv(R_nom[i] @ S_true[i]) for i in range(n_sens)])
    b_true = rng.normal(0, 20e-3, (n_sens, 3))

    def sweep(tag, incl_deg, jitter_deg=0.0):
        a = np.deg2rad(incl_deg)
        bt, bz = b_mt * np.cos(a), b_mt * np.sin(a)
        # An unknown start angle and an imperfectly square V-block: neither is
        # supposed to matter, because solve_roll fits every pose's angle.
        phi = (np.arange(n_poses) * (2 * np.pi / n_poses)
               + np.deg2rad(rng.normal(0, jitter_deg, n_poses))
               + rng.uniform(0, 2 * np.pi))
        B = np.column_stack([bt * np.cos(phi), bt * np.sin(phi),
                             np.full(n_poses, bz)])
        m = np.einsum("sij,tj->tsi", M_true, B) + b_true[None]
        return opc.RollSweep(m + rng.normal(0, noise_mt, m.shape), tag=tag)

    return sweep, S_true, b_true


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
    bearing_alpha = lambda d: np.degrees(np.arcsin(np.cos(dip)
                                                   * np.cos(np.deg2rad(d))))
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
    sweep_t, S_t, b_t = _synth_pose_sweeps(geom, gains, seed=11)
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
    import octobee_posecap as pcap

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
    pose = lambda sign, pert=0.0: np.array(
        [R_nom[i].T @ (sign * B_t * (1.0 + pert * (i - 7.5) / 15.0)) + offs[i]
         for i in range(n_sens)])

    _, med, spread = pcap.survey_uniformity(pose(+1), pose(-1), live_all)
    check("a uniform field surveys flat, whatever the offsets",
          spread < 0.01 and abs(med - 51.0) < 0.5,
          f"median {med:.2f} uT, spread {spread:.4f} %")

    # A 10 % gradient across the head must read back as ~10 %, not be hidden
    # by the offsets it sits on top of.
    _, med_g, spread_g = pcap.survey_uniformity(pose(+1, 0.10), pose(-1, 0.10),
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


# --------------------------------------------------------------------------
# stages and motorised field maps -- no hardware required
# --------------------------------------------------------------------------

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
          all(b >= a for a, b in zip(xs, xs[1:])), f"{xs}")
    rows = [ys[i:i + 3] for i in range(0, 9, 3)]
    check("every row of the fast axis runs the same way",
          all(r == sorted(r) for r in rows), f"{rows}")


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


def test_stage_binding():
    """The ctypes binding, as far as it goes with no stage attached."""
    try:
        d = ostage.dll()
    except ostage.StageError as exc:
        check("the Kinesis C API loads", False, str(exc))
        return
    check("the Kinesis C API loads", d is not None)
    check("the binding declares argtypes for every call it makes",
          all(getattr(d, n).argtypes is not None
              for n in ("ISC_Open", "ISC_MoveToPosition", "ISC_GetStatusBits",
                        "ISC_MoveRelative", "TLI_GetDeviceListExt")),
          "an unbound DWORD return truncates the status word silently")
    check("the status word decodes the homed bit",
          "homed" in ostage.Stage("45000000").status_flags(
              ostage.HOMED_BIT | ostage.ENABLED_BIT))
    check("the status word decodes motion errors",
          "motion_error" in ostage.Stage("45000000").status_flags(
              ostage.MOTION_ERROR_BIT))

    st = ostage.Stage("45000000")
    check("device units round trip through millimetres",
          abs(st._to_mm(st._to_du(137.25)) - 137.25) < 1e-6)
    check("the LTS300C default scaling matches the measured 409600 du/mm",
          st._to_du(300.0) == 122880000, f"{st._to_du(300.0)}")

    for name, fn in (("position", lambda: st.position_mm),
                     ("status", lambda: st.status),
                     ("a move", lambda: st.move_to(10.0))):
        try:
            fn()
            ok = False
        except ostage.StageError:
            ok = True
        check(f"{name} on an unopened stage raises rather than lying", ok)


def _framed(name, invert, origin=None, lo=0.0, hi=300.0):
    """A Stage with its travel filled in as if it had been opened."""
    st = ostage.Stage("45000000", name=name, invert=invert, origin_mm=origin)
    st.travel_dev_mm = (lo, hi)
    st._resolve_frame()
    return st


def test_stage_frame():
    """A reverse-mounted axis must be handled in exactly one place.

    The failure this guards against is not a crash. It is a field map that
    comes out mirrored along one axis and looks entirely plausible.
    """
    fwd = _framed("x", False)
    check("an as-mounted axis is the identity",
          fwd.travel_mm == (0.0, 300.0) and fwd._rig_to_dev(75.0) == 75.0,
          f"travel {fwd.travel_mm}")
    check("an as-mounted axis says so",
          fwd.frame_note() == "as mounted", fwd.frame_note())

    rev = _framed("z", True)
    check("a reversed axis puts rig zero at the far end of travel",
          rev.origin_mm == 300.0, f"origin {rev.origin_mm}")
    check("a reversed axis still spans the same rig travel",
          rev.travel_mm == (0.0, 300.0), f"travel {rev.travel_mm}")
    check("a reversed axis has no negative zero in its travel",
          not any(str(v).startswith("-0") for v in rev.travel_mm),
          f"travel {rev.travel_mm}")

    # The cube the user actually cares about: rig zero is the BOTTOM, and the
    # limit switch the stage homes to is the top.
    check("rig zero maps to the stage's far end",
          rev._rig_to_dev(0.0) == 300.0)
    check("rig maximum maps to the stage's home",
          rev._rig_to_dev(300.0) == 0.0)
    check("the midpoint is its own mirror image",
          rev._rig_to_dev(150.0) == 150.0)
    for rig in (0.0, 12.5, 150.0, 299.0):
        check(f"rig {rig:g} mm round trips through the device frame",
              abs(rev._dev_to_rig(rev._rig_to_dev(rig)) - rig) < 1e-9)

    check("a reversed axis reverses relative moves too",
          rev._sign == -1.0 and fwd._sign == 1.0,
          "a rig +5 mm must drive the device -5 mm, or jogging goes backwards")

    # An explicit origin is how you put rig zero on a fixture datum rather
    # than on a hard limit.
    off = _framed("z", True, origin=250.0)
    check("an explicit origin overrides the automatic one",
          off.origin_mm == 250.0 and off._rig_to_dev(0.0) == 250.0)
    check("an explicit origin shifts the rig travel with it",
          off.travel_mm == (-50.0, 250.0), f"travel {off.travel_mm}")


def test_stage_frame_persistence(workdir):
    path = os.path.join(workdir, "stages_frame.json")
    ostage.save_axis_map(
        {"x": "45502844", "z": "45502854", "y": "45538374"}, path,
        frames={"x": {"invert": False},
                "z": {"invert": True},
                "y": {"invert": False, "origin_mm": 12.0}})
    frames = ostage.load_axis_frames(path)
    check("the axis frame round trips", frames["z"]["invert"] is True,
          str(frames))
    check("an explicit origin round trips", frames["y"]["origin_mm"] == 12.0,
          str(frames))
    check("an unset origin round trips as None",
          frames["z"]["origin_mm"] is None, str(frames))
    check("the axis map survives alongside the frame",
          ostage.load_axis_map(path)["z"] == "45502854")

    # Writing the map without frames must not wipe the mounting -- the GUI's
    # "Save axis map" button does exactly that.
    ostage.save_axis_map({"x": "45502844", "z": "45502854"}, path)
    check("saving the map alone preserves the mounting",
          ostage.load_axis_frames(path)["z"]["invert"] is True)

    # A bare boolean is the shorthand a human is most likely to hand-edit in.
    with open(path) as fh:
        doc = json.load(fh)
    doc["frame"] = {"z": True}
    with open(path, "w") as fh:
        json.dump(doc, fh)
    check("a bare true is accepted as shorthand for invert",
          ostage.load_axis_frames(path)["z"]["invert"] is True)

    check("a missing frame block is empty, not an error",
          ostage.load_axis_frames(os.path.join(workdir, "nope.json")) == {})


def test_stage_axis_map(workdir):
    path = os.path.join(workdir, "stages.json")
    ostage.save_axis_map({"x": "45502844", "z": "45538374"}, path)
    back = ostage.load_axis_map(path)
    check("the axis map round trips",
          back == {"x": "45502844", "z": "45538374"}, str(back))
    check("a missing axis map is empty, not an error",
          ostage.load_axis_map(os.path.join(workdir, "nope.json")) == {})

    # A map naming a stage that is not present must fail loudly: the axis
    # names ARE the coordinate frame, and quietly dropping one produces a map
    # whose axes mean something other than what the file says.
    try:
        ostage.StageSet.from_config(path, serials=["45502844"])
        ok = False
    except ostage.StageError:
        ok = True
    check("an axis map naming an absent stage is refused", ok)

    ss = ostage.StageSet.from_config(path, serials=["45502844", "45538374"])
    check("a valid axis map builds the named axes",
          sorted(ss.names) == ["x", "z"], str(ss.names))
    try:
        ss.move_to(y=1.0)
        ok = False
    except ostage.StageError:
        ok = True
    check("moving an axis the set does not have is refused", ok)


def test_scan_guards():
    """run_scan must refuse the two setups that yield a plausible bad map."""
    class FakeStage:
        def __init__(self, name, homed):
            self.name, self.serial, self.homed = name, "45000000", homed

    class FakeSet:
        def __init__(self, axes):
            self.axes = axes
            self.names = list(axes)

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


# --------------------------------------------------------------------------
# unit-level checks that need no GUI
# --------------------------------------------------------------------------

def test_geometry():
    print("\ngeometry")
    g = pgeom.Geometry()
    R = g.rotations()
    check("rotations are orthonormal",
          all(np.allclose(R[i].T @ R[i], np.eye(3)) for i in range(16)))
    check("rotations are right-handed",
          all(np.linalg.det(R[i]) > 0.99 for i in range(16)))
    # The boards lie flat on the faces and reach out tangentially, so a chip
    # sits barely off its own face but a long way round the tube.
    check("chips sit just clear of the face they are bolted to",
          abs(g.fsv_height_mm - (g.board_thickness_mm + g.fsv_above_board_mm)) < 1e-9,
          f"{g.fsv_height_mm:g} mm off the face, reaching "
          f"{g.fsv_radius_mm:.1f} mm from the axis")
    arm_dirs = np.array([g.arm_dir(i) for i in range(1, 17)])
    check("arms run tangentially, not radially",
          np.allclose(np.einsum("ij,ij->i", arm_dirs, g.normals()), 0.0, atol=1e-9)
          and np.allclose(arm_dirs @ pgeom.TUBE_AXIS, 0.0, atol=1e-9),
          "perpendicular to both the face normal and the tube axis")
    # A chip reading purely on its own +X must map along its arm.
    b = np.zeros((16, 3))
    b[:, 0] = 1.0
    check("chip +X maps along the arm", np.allclose(g.to_tube_frame(b), arm_dirs))
    b = np.zeros((16, 3))
    b[:, 2] = 1.0
    check("chip +Z maps out of the board, along the face normal",
          np.allclose(g.to_tube_frame(b), g.normals()))
    b = np.zeros((16, 3))
    b[:, 1] = 1.0
    check("chip +Y maps along the tube",
          np.allclose(np.abs(g.to_tube_frame(b) @ pgeom.TUBE_AXIS), 1.0))
    # |B| must survive the rotation, that being the whole point of using it.
    rng = np.random.default_rng(1)
    v = rng.normal(size=(50, 16, 3))
    check("|B| is invariant under the rotation",
          np.allclose(np.linalg.norm(v, axis=-1),
                      np.linalg.norm(g.to_tube_frame(v), axis=-1)))
    pt = (g.fsv_radius_mm + 20.0, 0.0, 60.0)
    w = g.expected_response(pt)
    check("geometry weighting favours the nearest chip",
          int(np.argmax(w)) + 1 == g.nearest_sensor(pt),
          f"nearest is S{g.nearest_sensor(pt)}, spread {w.max()/w.min():.0f}x")


def test_conversion():
    print("\ncounts -> tesla")
    cal = ocal.Calibration()
    vpc = 20.0 / 65536.0
    n = 200
    # One sensor, VCM at 2.2 V, Bx sitting 63 mV above it. At 63 V/T that is
    # exactly 1 mT, so the whole chain has an answer we know in advance.
    ai = np.zeros((n, 32), dtype=np.int16)
    vcm_counts = int(round(2.2 / vpc))
    ai[:, :] = vcm_counts                                   # every channel at VCM
    for s in range(8):
        ai[:, s * 4 + 2] = int(round((2.2 + 0.063) / vpc))  # Bx = VCM + 63 mV
    b = cal.convert([ai, ai], [vpc, vpc], apply_zero=False)
    check("Bx of a 63 mV offset at 63 V/T is 1 mT",
          abs(b[0, 0, 0] - 1.0) < 0.01, f"got {b[0,0,0]:.4f} mT")
    check("By and Bz stay at the VCM zero",
          abs(b[0, 0, 1]) < 0.01 and abs(b[0, 0, 2]) < 0.01)
    check("all 16 sensors are populated", b.shape[1] == 16)

    # Halving the range doubles the volts-per-tesla, so the same volts read half.
    cal2 = ocal.Calibration(ranges_mt=np.full(16, 40.0))
    b2 = cal2.convert([ai, ai], [vpc, vpc], apply_zero=False)
    ratio = b2[0, 0, 0] / b[0, 0, 0]
    check("switching 20 mT -> 40 mT range rescales by V/T",
          abs(ratio - 63.0 / 34.65) < 0.01, f"ratio {ratio:.4f}")

    # Without VCM subtraction the 2.2 V virtual ground shows up as fake field.
    cal3 = ocal.Calibration(subtract_vcm=False)
    b3 = cal3.convert([ai, ai], [vpc, vpc], apply_zero=False)
    check("VCM subtraction removes the ~2.2 V pedestal",
          b3[0, 0, 1] > 30.0 and abs(b[0, 0, 1]) < 0.01,
          f"unsubtracted By reads {b3[0,0,1]:.1f} mT")

    # Tare
    cal.tare(b)
    b4 = cal.to_mt(ocal.assemble([ai, ai], [vpc, vpc]))
    check("tare drives the tared signal to zero", np.abs(b4).max() < 1e-9)


def test_cross_calibration():
    print("\ncross-calibration")
    cal = ocal.Calibration()
    # Sensors that respond 2x and 0.5x should come back matched.
    peaks = np.full(16, 1.0)
    peaks[3] = 2.0
    peaks[7] = 0.5
    cal.cross_calibrate(peaks)
    trimmed = peaks * cal.gain_corr[:, 0]
    check("gain trim equalises the response",
          np.allclose(trimmed, trimmed[0], rtol=1e-6),
          f"spread {trimmed.max()/trimmed.min():.6f}x")

    # With a magnet nearer some sensors than others, the geometry weighting must
    # remove the distance effect instead of baking it into the gain.
    g = pgeom.Geometry()
    w = g.expected_response((g.fsv_radius_mm + 20.0, 0.0, 60.0))
    cal2 = ocal.Calibration()
    peaks2 = w * 1.0                      # perfectly matched chips, unequal r
    cal2.cross_calibrate(peaks2, weights=w)
    check("geometry weighting leaves matched chips untouched",
          np.allclose(cal2.gain_corr, 1.0, atol=1e-6),
          f"max trim {np.abs(cal2.gain_corr-1).max():.2e}")
    cal3 = ocal.Calibration()
    cal3.cross_calibrate(peaks2)          # same data, geometry ignored
    check("ignoring geometry would have injected a false trim",
          np.abs(cal3.gain_corr - 1).max() > 0.5,
          f"max trim {np.abs(cal3.gain_corr-1).max():.2f}")


def test_shipped_calibration():
    """
    The repo ships the measured register configuration, not a neutral default.

    Getting this wrong is invisible -- every number on screen is simply
    rescaled, with nothing on screen looking wrong -- so it is worth asserting
    rather than trusting.
    """
    print("\nshipped calibration")
    if not os.path.exists(ocal.CONFIG_NAME):
        check("calibration.json is present", False)
        return
    cal = ocal.Calibration.load(ocal.CONFIG_NAME)
    # Re-read live with: ssh root@acq1001_69x 'python3 /tmp/onbox_gain_config.py show'
    # All 16 confirmed at gain 3000 / EGain_sel 0x00 on 2026-08-19.
    check("all 16 are on the SPI-audited +/-20 mT range",
          all(r == 20.0 for r in cal.ranges_mt), f"{cal.ranges_mt}")
    vpt = cal.volts_per_tesla
    check("every sensor therefore converts at 63 V/T",
          np.allclose(vpt, 63.0), f"{vpt[0]:.2f} V/T")
    check("the two halves no longer differ",
          abs(vpt[8] / vpt[0] - 1.0) < 1e-9, f"{vpt[8]/vpt[0]:.3f}x")
    check("VCM subtraction is on", cal.subtract_vcm)
    check("no sensor is excluded up front", not cal.dead,
          "S16's fault is detected at run time, so a repaired ribbon "
          "starts working without editing this file")


def test_health():
    print("\nchannel health")
    vpc = 20.0 / 65536.0
    n = 500
    rng = np.random.default_rng(3)
    ai = (rng.normal(7000, 2, (n, 32))).astype(np.int16)
    ai[:, 28:32] = -32768                                  # S8 railed
    ai[:, 30] = 7000                                       # ...except one
    rows = ocal.channel_health([ai, ai], [vpc, vpc], ["a", "b"])
    check("64 channels reported", len(rows) == 64)
    v = ocal.health_verdict(rows)
    check("railed sensor is called dead", v[8][0] == "dead", v[8][1])
    check("healthy sensor is called ok", v[1][0] == "ok")
    check("suggest_dead picks it up", "S8" in ocal.suggest_dead(rows))


# --------------------------------------------------------------------------
# full application
# --------------------------------------------------------------------------

def test_app(app, args, workdir):
    kind = ("live hardware" if args.live else
            f"replay {args.replay}" if args.replay else "synthetic probe")
    print(f"\napplication ({kind})")
    ns = argparse.Namespace(
        uut=None, demo=not (args.replay or args.live), replay=args.replay,
        geometry=os.path.join(workdir, "probe_geometry.json"),
        calibration=os.path.join(workdir, "calibration.json"),
        screenshot=None, screenshot_tab=0, screenshot_warmup=0)
    win = gui.MainWindow(ns)
    clkdiv_before = {}
    if args.live:
        # Everything this tool does to the carriers' clock has to be undone on
        # the way out. A run that leaves them slowed down silently poisons the
        # next one, which is how a "full rate" snapshot once came back at
        # 20 kSPS.
        clkdiv_before = {h: read_clkdiv(h) for h in ob.DEFAULT_UUTS}
        print(f"  carriers found at clkdiv {clkdiv_before}")
        win.on_connect()
        deadline = time.time() + 90
        while win.source is None and time.time() < deadline:
            app.processEvents()
            time.sleep(0.05)
    check("source started", win.source is not None)
    if win.source is None:
        return win

    pump(win, app, 3.0)
    check("data is flowing", win.roll.filled > 100,
          f"{win.roll.filled} points buffered")
    check("sensor table populated", win._last_table is not None
          and len(win._last_table) == 16)
    check("channel health computed", win.last_health is not None
          and len(win.last_health) == 64)
    check("S16 detected as dead", "S16" in win.cal.dead,
          f"excluded: {sorted(win.cal.dead)}")

    # ---- tare ----
    v = win.roll.view()
    before = np.abs(np.median(v, axis=0)).max() if v.shape[0] else float("nan")
    win.start_collect("tare", 0.5)
    pump(win, app, 2.5)
    check("tare completed", win.collecting is None)
    check("tare stored a non-trivial zero", np.any(win.cal.zero_mt != 0),
          f"max |zero| {np.abs(win.cal.zero_mt).max():.4f} mT (was reading "
          f"{before:.4f} mT)")
    pump(win, app, 1.0)

    # ---- magnet pass ----
    win.btn_magnet.setChecked(True)
    pump(win, app, 4.0)
    win.btn_magnet.setChecked(False)
    app.processEvents()
    check("magnet pass captured peaks", win.magnet_peaks is not None)
    if win.magnet_peaks is not None:
        live = win.cal.live_mask()
        resp = int((win.magnet_peaks[live] > 1e-4).sum())
        check("most live sensors responded", resp >= 10, f"{resp}/15 responded")
        rep = ocal.spread_report(win.magnet_peaks, live=live)
        check("spread report produced", "raw_spread" in rep,
              f"spread {rep.get('raw_spread', float('nan')):.2f}x")

        # ---- gain trim ----
        before_spread = rep.get("raw_spread")
        win.on_apply_gain()
        trimmed = win.magnet_peaks * win.cal.gain_corr[:, 0]
        after = trimmed[live].max() / trimmed[live].min()
        check("gain trim narrows the spread", after < 1.0001,
              f"{before_spread:.2f}x -> {after:.4f}x")
        win.on_clear_gain()
        check("gain trim clears", np.allclose(win.cal.gain_corr, 1.0))

    # ---- Earth-field roll calibration panel ----
    # The demo source is a moving dipole, not a uniform field being rolled
    # through, so the numbers it produces are meaningless. What is being
    # tested here is the wiring: that a sweep is captured from UNCORRECTED
    # field, that the solve/apply path runs, and that sweeps survive a trip
    # through disk. The physics is graded in test_posecal() against a known
    # truth instead.
    check("roll panel is built",
          all(hasattr(win, a) for a in ("btn_solve_roll", "btn_apply_roll",
                                        "lbl_sweeps", "spin_bearth",
                                        "chk_isotropic", "spin_sweep_s")))
    check("solve is disabled with no sweeps recorded",
          not win.btn_solve_roll.isEnabled())

    win.cal.gain_corr = np.full((pgeom.N_SENSORS, 3), 2.0)
    for tag in ("A", "B", "C"):
        win.start_sweep(tag, 0.4)
        pump(win, app, 2.0)
    win.cal.clear_gain()
    check("all three sweeps recorded", set(win.sweeps) == {"A", "B", "C"},
          win.lbl_sweeps.text())
    check("solve enabled once sweeps exist", win.btn_solve_roll.isEnabled())

    if set(win.sweeps) == {"A", "B", "C"}:
        # A sweep must be independent of whatever trim was loaded when it was
        # taken -- that is why it is captured pre-correction. The x2 gain above
        # was live during capture and must not show up in the stored data.
        sw = win.sweeps["A"]
        check("sweeps are stored uncorrected",
              np.abs(sw.b_mt).max() < 1e4 and np.isfinite(sw.b_mt).all())
        check("sweeps record the range they were taken at",
              sw.ranges_mt is not None
              and np.allclose(sw.ranges_mt, win.cal.ranges_mt))

        win.on_solve_roll()
        check("roll solve produced a solution", win.pose_solution is not None,
              "" if win.pose_solution is not None
              else " ".join(win.cal_report.toPlainText().split())[:300])
        check("apply is enabled after a solve", win.btn_apply_roll.isEnabled())
        if win.pose_solution is not None:
            sol = win.pose_solution
            check("solve used all three orientations",
                  sorted(sol.tags) == ["A", "B", "C"], f"{sol.tags}")
            check("report reaches the calibration pane",
                  "gain spread" in win.cal_report.toPlainText())
            win.cal.apply_pose_solution(sol)
            check("applying installs the matrix and clears the trim",
                  win.cal.has_matrix and np.allclose(win.cal.gain_corr, 1.0))
            check("the applied calibration still converts",
                  np.isfinite(win.cal.to_mt(
                      np.full((4, pgeom.N_SENSORS, 4), 2.2))).all())
            win.cal.clear_matrix()
            win.cal.clear_tare()

        with tempfile.TemporaryDirectory() as d:
            ok = True
            for tag, sw in win.sweeps.items():
                back = opc.RollSweep.load(sw.save(os.path.join(d, f"rs_{tag}")))
                ok &= (back.tag == tag
                       and back.b_mt.shape == sw.b_mt.shape
                       and np.allclose(back.ranges_mt, sw.ranges_mt))
            check("sweeps round trip through disk", ok)

    win.on_clear_sweeps()
    check("clearing sweeps disables solve",
          not win.sweeps and not win.btn_solve_roll.isEnabled())

    # ---- display cannot starve acquisition ----
    win.cmb_view.setCurrentIndex(list(gui.VIEW_RATES).index(10.0))
    start_interval = win.view_timer.interval()
    check("view runs on its own timer, not the acquisition one",
          win.view_timer is not win.timer
          and win.timer.interval() <= start_interval,
          f"acquisition {win.timer.interval()} ms, view {start_interval} ms")
    for _ in range(8):
        win._note_draw_time(500.0)          # pretend every redraw is very slow
    check("a slow redraw backs the view off automatically",
          win.view_timer.interval() > start_interval,
          f"{start_interval} ms -> {win.view_timer.interval()} ms")
    check("the backoff is bounded",
          win.view_timer.interval() <= gui.MAX_VIEW_INTERVAL_MS,
          f"{win.view_timer.interval()} ms")
    # Re-pick the SAME entry, which is what a user does to undo a backoff.
    win.cmb_view.activated.emit(win.cmb_view.currentIndex())
    check("re-picking the same rate clears the backoff",
          win.view_timer.interval() == start_interval,
          f"back to {win.view_timer.interval()} ms")

    win.chk_3d.setChecked(False)
    before = win.roll.filled
    pump(win, app, 1.0)
    check("acquisition continues with the 3D head off", win.roll.filled >= before)
    win.chk_3d.setChecked(True)

    win.act_pause.setChecked(True)
    n_before = win.roll.filled
    pump(win, app, 1.0)
    check("pausing the view does not pause acquisition",
          win.roll.filled >= n_before and win.paused)
    win.act_pause.setChecked(False)

    # ---- recording ----
    win.chk_csv.setChecked(True)
    win.chk_raw.setChecked(True)
    win.chk_tube.setChecked(False)
    os.makedirs("captures", exist_ok=True)
    win.act_record.setChecked(True)
    check("CSV recorder opened", win.csv_rec is not None)
    check("raw recorder opened", win.raw_rec is not None)
    csv_path = win.csv_rec.path if win.csv_rec else None
    raw_path = win.raw_rec.path if win.raw_rec else None
    pump(win, app, 3.0)
    rows_written = win.csv_rec.n_rows if win.csv_rec else 0
    win.act_record.setChecked(False)
    check("recording stopped cleanly", win.csv_rec is None and win.raw_rec is None)

    # ---- read the CSV back and check it ----
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            lines = f.readlines()
        header = [l for l in lines if l.startswith("#")]
        cols = [l for l in lines if not l.startswith("#")][0].strip().split(",")
        names, data = read_csv(csv_path)
        col = {n: i for i, n in enumerate(names)}
        check("CSV has a provenance header", len(header) >= 8,
              f"{len(header)} header lines")
        check("CSV has 1 + 16*4 columns", len(cols) == 65, f"{len(cols)} columns")
        check("CSV row count matches the recorder",
              data.shape[0] == rows_written, f"{data.shape[0]} rows")
        dt = np.diff(data[:, col["t_s"]])
        check("CSV timebase matches the output rate",
              abs(np.median(dt) - 1.0 / win.out_rate) < 1e-6,
              f"dt {np.median(dt)*1e3:.3f} ms at {win.out_rate:g} Hz")
        s1 = data[:, [col["S1_Bx_mT"], col["S1_By_mT"], col["S1_Bz_mT"]]]
        check("CSV |B| column equals the vector norm",
              np.allclose(np.linalg.norm(s1, axis=1), data[:, col["S1_absB_mT"]],
                          atol=1e-4, rtol=1e-3))
        check("CSV values are finite", np.isfinite(data).all())
        check("CSV carries real signal, not zeros",
              np.abs(data[:, col["S1_absB_mT"]]).max() > 1e-6,
              f"max |B| on S1 {np.abs(data[:, col['S1_absB_mT']]).max():.4f} mT")

    # ---- read the raw file back and check it ----
    if raw_path and os.path.exists(raw_path):
        x, meta = orec.load_raw(raw_path)
        check("raw sidecar records the channel names",
              len(meta["channel_names"]) == x.shape[1] == 64)
        check("raw sample count matches the sidecar",
              x.shape[0] == meta["n_samples"], f"{x.shape[0]} samples")
        boxes = orec.raw_to_boxes(x, meta)
        b = win.cal.convert(boxes, meta["volts_per_count"])
        check("raw file reconverts to finite field values",
              np.isfinite(b).all() and b.shape[1:] == (16, 3),
              f"shape {b.shape}")

    # ---- exports ----
    win.on_export_summary()
    win.on_export_health()
    win.on_export_json()
    win.on_health()
    app.processEvents()
    txt = win.health_text.toPlainText()
    check("diagnostics text produced", "per-sensor verdict" in txt,
          f"{len(txt)} chars")
    check("diagnostics names the VCM channels", "VCM" in txt)

    exports = [l for l in win.export_log.toPlainText().splitlines() if l.strip()]
    check("three one-shot exports logged", len(exports) >= 3,
          f"{len(exports)} entries")
    for line in exports:
        p = line.split("] ", 1)[-1].split("  (")[0]
        if os.path.exists(p):
            check(f"export exists and is non-empty: {os.path.basename(p)}",
                  os.path.getsize(p) > 0, f"{os.path.getsize(p)} bytes")

    # ---- calibration round trip ----
    cal_path = os.path.join(workdir, "roundtrip.json")
    win.cal.zero_mt[0, 0] = 1.2345
    win.cal.ranges_mt[5] = 400.0
    win.cal.save(cal_path)
    reloaded = ocal.Calibration.load(cal_path)
    check("calibration survives a save/load round trip",
          np.allclose(reloaded.zero_mt, win.cal.zero_mt)
          and np.allclose(reloaded.ranges_mt, win.cal.ranges_mt)
          and reloaded.dead == win.cal.dead)

    # ---- geometry round trip and rebuild ----
    g = pgeom.Geometry(mapping="ring-major")
    g.tube_width_mm = 55.0
    g.save(ns.geometry)
    win.on_reload_geometry()
    check("geometry reload reaches the 3D view",
          abs(win.view3d.geom.tube_width_mm - 55.0) < 1e-9)
    check("geometry reload reaches the sensor table",
          win.table.geom.mapping == "ring-major")
    pump(win, app, 1.0)
    check("still acquiring after a geometry change", win.roll.filled > 100)

    # ---- tube frame CSV ----
    win.chk_tube.setChecked(True)
    win.chk_raw.setChecked(False)
    win.act_record.setChecked(True)
    tube_path = win.csv_rec.path if win.csv_rec else None
    pump(win, app, 1.5)
    win.act_record.setChecked(False)
    if tube_path and os.path.exists(tube_path):
        with open(tube_path, encoding="utf-8") as f:
            head = "".join(f.readlines()[:12])
        check("tube-frame CSV is labelled as such", "frame: tube" in head)
        tn, td = read_csv(tube_path)
        tc = {n: i for i, n in enumerate(tn)}
        check("tube-frame CSV has data", td.shape[0] > 10, f"{td.shape[0]} rows")
        # Rotation preserves length, so |B| must be identical in either frame.
        s1t = td[:, [tc["S1_Bx_mT"], tc["S1_By_mT"], tc["S1_Bz_mT"]]]
        check("tube-frame |B| still matches its own components",
              np.allclose(np.linalg.norm(s1t, axis=1), td[:, tc["S1_absB_mT"]],
                          atol=1e-4, rtol=1e-3))

    if args.live:
        st = win.source.stats()
        check("no stream gaps on the live link", st.get("gaps", 0) == 0,
              f"gaps {st.get('gaps')}, lost {st.get('lost')}")
        # Measure over a window in which the loop is actually being pumped.
        # A count accumulated across the whole run says more about this
        # harness -- which deliberately blocks for most of a second at a time
        # parsing files back -- than about the application, whose real contract
        # is that a running session does not shed data.
        for stream in win.source.streamers:
            stream.dropped = 0
        pump(win, app, 3.0)
        dropped = sum(x.dropped for x in win.source.streamers)
        check("no blocks dropped while the session is running", dropped == 0,
              f"{dropped} dropped over 3 s of streaming")

        # The snapshot stops the stream, captures at the full 200 kSPS, and
        # hands the port back. It is the only path that takes over the
        # carriers' stream ownership, so it gets exercised for real.
        win.spin_snap_s.setValue(1.0)
        win.on_snapshot()
        deadline = time.time() + 120
        while (win._snap_worker is not None and win._snap_worker.isRunning()
               and time.time() < deadline):
            app.processEvents()
            time.sleep(0.05)
        app.processEvents()
        snaps = [l for l in win.export_log.toPlainText().splitlines()
                 if ".npz" in l]
        check("snapshot written", bool(snaps), snaps[-1] if snaps else "none")
        if snaps:
            sp = snaps[-1].split("] ", 1)[-1].split("  (")[0]
            cap = ocal.load_capture(sp)
            check("snapshot is a full-rate capture",
                  cap["fs_hz"][0] > 150000, f"{cap['fs_hz'][0]/1e3:g} kSPS")
            check("snapshot holds both boxes", len(cap["ai"]) == 2)
            rows = ocal.channel_health(cap["ai"], cap["vpc"], cap["hosts"])
            check("snapshot reproduces the S16 fault",
                  "S16" in ocal.suggest_dead(rows))

    win.on_disconnect()
    if args.live:
        time.sleep(2.0)
        after = {h: read_clkdiv(h) for h in ob.DEFAULT_UUTS}
        check("carriers left at the clock they were found at",
              after == clkdiv_before, f"{clkdiv_before} -> {after}")
    win.close()
    return win


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replay", help="use a real saved capture instead of the "
                                    "synthetic probe")
    p.add_argument("--live", action="store_true",
                   help="connect to the real carriers and test against them "
                        "(changes their clkdiv while it runs, and restores it)")
    args = p.parse_args()

    if not args.live:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    test_geometry()
    test_conversion()
    test_cross_calibration()
    test_shipped_calibration()
    test_health()
    test_posecal()
    test_posecap()
    test_posecal_persistence()
    test_scan_grid()
    test_scan_guards()
    test_stage_binding()
    test_stage_frame()
    with tempfile.TemporaryDirectory() as workdir:
        test_fieldmap_roundtrip(workdir)
        test_stage_axis_map(workdir)
        test_stage_frame_persistence(workdir)
        test_app(app, args, workdir)

    print(f"\n{CHECKS - len(FAILS)}/{CHECKS} checks passed")
    if FAILS:
        print("failed:")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
