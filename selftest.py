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
import ast
import csv
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from PyQt6 import QtCore, QtWidgets

from octobee import paths
from octobee.acq import carrier as ob
from octobee.calib import convert as ocal
from octobee.gui import window as gui
from octobee.gui.crash import CrashHandler
from octobee.calib import roll as opc
from octobee.calib import poses as opcap
from octobee.calib import poses as pcap
from octobee import record as orec
from octobee.motion import scan as oscan
from octobee.motion import stage as ostage
from octobee import help as ohelp
from octobee import machine as omach
from octobee.calib import magnet as omag
from octobee.calib import geometry as pgeom
import itertools

FAILS = []
SKIPS = []
CHECKS = 0


def check(name, cond, detail=""):
    global CHECKS
    CHECKS += 1
    ok = bool(cond)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


def skip(name, why):
    """Record a check that could not run here, without calling it a pass.

    Exactly one thing in this suite needs software a machine may legitimately
    not have: the Thorlabs Kinesis DLL is Windows-only and proprietary, so a
    Linux CI runner cannot load it and never will. Failing there makes the
    quality gate permanently red for a reason that says nothing about the
    code; passing quietly would let a bench run that stopped testing the
    binding look identical to one that did.

    So it is counted separately and named in the summary. A run with skips
    checked less than a full one, and says so.
    """
    SKIPS.append(name)
    print(f"  [skip] {name}  -- {why}")
    return False


def read_csv(path):
    """
    Read one of our CSVs into (column names, float array).

    numpy's genfromtxt(names=True) takes the first line as the header even when
    it is a comment, so it chokes on the provenance block. Parsing it here also
    checks the file is readable the way a person would read it.
    """
    names, rows = None, []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
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

# The measured single-sample noise, averaged down over one solve block. A
# module-level constant rather than an expression in the signature, so it is
# evaluated once and can be referred to by name.
SYNTH_NOISE_MT = 8.81e-3 / np.sqrt(2500)


def _synth_sweeps(geom, gains, axial_ratio=1.0, offsets_ut=20.0, seed=3,
                  noise_mt=SYNTH_NOISE_MT, n=3000):
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


# --------------------------------------------------------------------------
# the coil set the probe measures inside
# --------------------------------------------------------------------------

def _simsopt_file(path, radius=1.0, shift=0.25):
    """A SIMSON file with two coils and two current configurations.

    Written by hand rather than by simsopt, which is the point: the reader in
    octobee_machine parses this format itself, so the test has to be able to
    state exactly what is in the file. Two circles of known radius, one of them
    reached through a RotatedCurve, and the SAME two curves used by both
    BiotSavart objects with different currents -- which is the arrangement that
    would turn two physical coils into four if the de-duplication broke.
    """
    def circle(name, r, z):
        # A circle of radius r at height z: x = r cos(2 pi t), y = r sin(2 pi t)
        return {"@module": "simsopt.geo.curvexyzfourier",
                "@class": "CurveXYZFourier", "@name": name,
                "quadpoints": {"@module": "numpy", "@class": "array",
                               "dtype": "float64",
                               "data": list(np.linspace(0, 1, 32, endpoint=False))},
                "order": 1, "dofs": {"$type": "ref", "value": f"dofs_{name}"}}, {
            "@module": "simsopt._core.optimizable", "@class": "DOFs",
            "@name": f"dofs_{name}",
            "x": {"@module": "numpy", "@class": "array", "dtype": "float64",
                  "data": [0.0, 0.0, r,        # xc(0), xs(1), xc(1)
                           0.0, r, 0.0,        # yc(0), ys(1), yc(1)
                           z, 0.0, 0.0]},      # zc(0), zs(1), zc(1)
            "names": ["xc(0)", "xs(1)", "xc(1)",
                      "yc(0)", "ys(1)", "yc(1)",
                      "zc(0)", "zs(1)", "zc(1)"],
            "free": {"@module": "numpy", "@class": "array", "dtype": "bool",
                     "data": [True] * 9},
            "lower_bounds": {"@module": "numpy", "@class": "array",
                             "dtype": "float64", "data": [-1e30] * 9},
            "upper_bounds": {"@module": "numpy", "@class": "array",
                             "dtype": "float64", "data": [1e30] * 9}}

    objs = {}
    for name, r, z in (("CurveXYZFourier1", radius, 0.0),
                       ("CurveXYZFourier2", radius, shift)):
        curve, dofs = circle(name, r, z)
        objs[name] = curve
        objs[f"dofs_{name}"] = dofs
    # The second coil is reached through a rotation, as a real file does it.
    objs["RotatedCurve1"] = {"@module": "simsopt.geo.curve",
                             "@class": "RotatedCurve", "@name": "RotatedCurve1",
                             "curve": {"$type": "ref",
                                       "value": "CurveXYZFourier2"},
                             "phi": 0.0, "flip": False}
    for i, amps in enumerate((100.0, 250.0, 7.0, -3.0), start=1):
        objs[f"Current{i}"] = {
            "@module": "simsopt.field.coil", "@class": "Current",
            "@name": f"Current{i}", "current": amps,
            "dofs": {"$type": "ref", "value": f"dofs_Current{i}"}}
        objs[f"dofs_Current{i}"] = {
            "@module": "simsopt._core.optimizable", "@class": "DOFs",
            "@name": f"dofs_Current{i}",
            "x": {"@module": "numpy", "@class": "array", "dtype": "float64",
                  "data": [amps]},
            "names": ["x0"],
            "free": {"@module": "numpy", "@class": "array", "dtype": "bool",
                     "data": [True]}}
    # One of them is scaled, because most real files scale every current.
    objs["ScaledCurrent1"] = {"@module": "simsopt.field.coil",
                              "@class": "ScaledCurrent",
                              "@name": "ScaledCurrent1",
                              "current_to_scale": {"$type": "ref",
                                                   "value": "Current1"},
                              "scale": 10.0}
    curves = ["CurveXYZFourier1", "RotatedCurve1"]
    currents = ["ScaledCurrent1", "Current2", "Current3", "Current4"]
    for i in range(4):
        objs[f"Coil{i + 1}"] = {"@module": "simsopt.field.coil",
                                "@class": "Coil", "@name": f"Coil{i + 1}",
                                "curve": {"$type": "ref",
                                          "value": curves[i % 2]},
                                "current": {"$type": "ref",
                                            "value": currents[i]}}
    for i, coils in enumerate((["Coil1", "Coil2"], ["Coil3", "Coil4"]), start=1):
        objs[f"BiotSavart{i}"] = {
            "@module": "simsopt.field.biotsavart", "@class": "BiotSavart",
            "@name": f"BiotSavart{i}",
            "coils": [{"$type": "ref", "value": c} for c in coils]}
    doc = {"@module": "simsopt._core.json", "@class": "SIMSON",
           "@version": "1.8.4.test", "graph": [], "simsopt_objs": objs}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return path


def test_machine_coils(workdir):
    """Reading a coil set, and the volume it takes up."""
    print("\ncoil set")
    path = _simsopt_file(os.path.join(workdir, "coils.json"))
    coils = omach.CoilSet.load(path)

    check("two BiotSavart objects over the same curves are two coils, "
          "not four", len(coils) == 2, f"{len(coils)} coils")
    check("both current configurations are offered",
          coils.configurations == ["BiotSavart1", "BiotSavart2"],
          str(coils.configurations))
    # 100 A through a x10 ScaledCurrent.
    check("a scaled current is resolved through the scale",
          abs(coils["C1"].currents["BiotSavart1"] - 1000.0) < 1e-9,
          f"{coils['C1'].currents['BiotSavart1']} A")
    check("the same coil carries a different current in the other "
          "configuration",
          abs(coils["C1"].currents["BiotSavart2"] - 7.0) < 1e-9,
          f"{coils['C1'].currents['BiotSavart2']} A")

    # The file is in metres and everything downstream is in millimetres. A
    # factor of a thousand here would put the probe a millimetre from a coil
    # that is really a metre away, and the drawing would look plausible.
    r = np.hypot(coils["C1"].points_mm[:, 0], coils["C1"].points_mm[:, 1])
    check("metres become millimetres at the boundary",
          np.allclose(r, 1000.0, atol=1e-6), f"radius {r.mean():.3f} mm")
    check("the centreline comes back closed",
          np.allclose(coils["C1"].points_mm[0], coils["C1"].points_mm[-1]))
    check("a circle's length is 2 pi r",
          abs(coils["C1"].length_mm - 2 * np.pi * 1000.0) < 1.0,
          f"{coils['C1'].length_mm:.1f} mm")
    check("the second coil is where the rotation put it",
          abs(coils["C2"].centroid_mm[2] - 250.0) < 1e-6,
          f"z {coils['C2'].centroid_mm[2]:.1f} mm")

    verts, faces = omach.tube_mesh(coils["C1"].points_mm, 20.0, sides=8)
    n = len(coils["C1"].points_mm) - 1
    d = np.linalg.norm(verts.reshape(n, 8, 3)
                       - coils["C1"].points_mm[:-1][:, None, :], axis=2)
    check("the swept cross-section really is the radius asked for",
          np.allclose(d, 20.0, atol=1e-3), f"{d.min():.4f}..{d.max():.4f} mm")
    edges = {}
    for tri in faces:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            key = (min(a, b), max(a, b))
            edges[key] = edges.get(key, 0) + 1
    check("the tube closes on itself, with no seam and no cap",
          set(edges.values()) == {2},
          f"edge counts {sorted(set(edges.values()))}")

    # ---- clearance, against an answer worked out by hand ----
    # A point on the axis of a circle of radius R, at height h, is
    # sqrt(R^2 + h^2) from the wire.
    probe = np.array([[0.0, 0.0, 300.0]])
    gap = omach.clearance(probe, coils, 20.0, labels=["C1"])
    want = np.hypot(1000.0, 300.0) - 20.0
    # The coil is a polygon through points on the curve, so it lies just
    # inside it: a 256-sided ring of radius 1 m cuts the corner by 75 um. The
    # answer must be short by that and no more -- the module claims a chord
    # error well under a tenth of a millimetre, and this is the claim.
    sag = 1000.0 * (1.0 - np.cos(np.pi / omach.CURVE_POINTS))
    check("clearance is measured to the winding surface, not the centreline",
          -sag <= gap.gap_mm - want <= 0.0,
          f"{gap.gap_mm:.4f} vs {want:.4f} mm, chord error {sag * 1000:.0f} um")
    check("and it names the coil it measured to", gap.coil == "C1", gap.coil)
    check("the closest point it reports is on that coil",
          abs(np.hypot(*gap.coil_point[:2]) - 1000.0) <= sag + 1e-9
          and abs(gap.coil_point[2]) < 1e-9, str(gap.coil_point))

    inside = np.array([[1000.0, 0.0, 5.0]])
    gap = omach.clearance(inside, coils, 20.0)
    check("a point inside a winding is reported as a collision",
          gap.collides and abs(gap.gap_mm + 15.0) < 1e-6, gap.text())

    # A coil that is switched off is still in the way: clearance takes no
    # notice of which ones are energised.
    both = omach.clearance(np.array([[0.0, 0.0, 249.0]]), coils, 20.0)
    check("clearance considers coils that are switched off",
          both.coil == "C2", both.text())

    # The two-pass narrowing inside clearance() is an optimisation, and an
    # optimisation that changes the answer is a bug. Check it against the
    # definition on a cloud of random points.
    rng = np.random.default_rng(4)
    cloud = rng.uniform(-1400, 1400, size=(400, 3))
    fast = omach.clearance(cloud, coils, 20.0)
    slow = np.inf
    for coil in coils:
        a, b = coil.points_mm[:-1], coil.points_mm[1:]
        ab = b - a
        for q in cloud:
            ap = q - a
            t = np.clip((ap * ab).sum(1) / (ab * ab).sum(1), 0.0, 1.0)
            slow = min(slow, np.linalg.norm(ap - t[:, None] * ab,
                                            axis=1).min())
    check("narrowing the search does not change the answer",
          abs(fast.gap_mm - (slow - 20.0)) < 1e-9,
          f"{fast.gap_mm:.9f} vs {slow - 20.0:.9f} mm")

    # ---- the probe body ----
    geom = pgeom.Geometry()
    cloud = omach.probe_cloud(geom)
    reach = np.linalg.norm(cloud, axis=1).max()
    check("the probe cloud covers the whole body, arms included",
          reach > geom.fsv_radius_mm, f"reaches {reach:.1f} mm")
    # Corners alone would miss a coil brushing the middle of the tube.
    along = np.unique(np.round(cloud[:, 1], 3))
    check("and it samples along the tube rather than only its ends",
          len(along) > 20, f"{len(along)} distinct stations along the tube")


def test_machine_placement(workdir):
    """The transform that says where the probe is, and what it remembers."""
    print("\nprobe placement")
    pose = omach.Placement(x_mm=100.0, y_mm=200.0, z_mm=-50.0)
    check("with no rotation the flange is where it was put",
          np.allclose(pose.origin_mm(), [100.0, 200.0, -50.0]))

    # The stages move the probe along the RIG's axes. Turn the assembly a
    # quarter turn about the machine's Z and driving rig x must move it along
    # machine y -- if it moved along machine x regardless, the drawing would
    # be wrong in exactly the way nobody would notice until the head hit
    # something.
    pose = omach.Placement(yaw_deg=90.0)
    moved = pose.origin_mm({"x": 10.0, "y": 0.0, "z": 0.0})
    check("a stage move is applied in the rig's frame, not the machine's",
          np.allclose(moved, [0.0, 10.0, 0.0], atol=1e-9), str(moved))

    pose = omach.Placement(x_mm=5.0, stage_zero_mm={"x": 100.0})
    check("the stage zero is where the pose says the probe is",
          np.allclose(pose.origin_mm({"x": 100.0}), [5.0, 0.0, 0.0]))
    check("and moving off it moves the probe by the difference",
          np.allclose(pose.origin_mm({"x": 130.0}), [35.0, 0.0, 0.0]))

    # Yaw is applied last, so whatever pitch and roll have already done to the
    # assembly, changing yaw turns THAT about the machine's Z. Stated as the
    # identity it has to satisfy rather than as one vector's image, because
    # the vector could come out right for a rotation composed the other way.
    tilted = omach.Placement(pitch_deg=20.0, roll_deg=90.0).rotation()
    turned = omach.Placement(yaw_deg=35.0, pitch_deg=20.0,
                             roll_deg=90.0).rotation()
    rz = omach.rotation_matrix(35.0, 0.0, 0.0)
    check("yaw turns the assembly about the machine's Z, whatever it is doing",
          np.allclose(turned, rz @ tilted, atol=1e-12),
          f"largest disagreement {np.abs(turned - rz @ tilted).max():.2e}")

    corners = omach.Placement(travel_mm={"x": (0.0, 300.0),
                                         "y": (0.0, 100.0),
                                         "z": (0.0, 50.0)}).reach_corners_mm()
    span = corners.max(axis=0) - corners.min(axis=0)
    check("the stage envelope is the travel of each axis",
          np.allclose(span, [300.0, 100.0, 50.0]), str(span))

    # ---- persistence ----
    path = os.path.join(workdir, "machine_cfg.json")
    coil_path = _simsopt_file(os.path.join(workdir, "coils2.json"))
    cfg = omach.MachineConfig(coil_file=coil_path, coil_radius_mm=12.5,
                              configuration="BiotSavart2", current_scale=0.01,
                              energised=["C2"], track_stage=False)
    cfg.pose = omach.Placement(x_mm=1.5, yaw_deg=30.0,
                               stage_zero_mm={"y": 12.0})
    cfg.save(path)
    back = omach.MachineConfig.load(path)
    check("the placement survives a save and a load",
          (back.coil_radius_mm == 12.5 and back.configuration == "BiotSavart2"
           and back.energised == ["C2"] and back.track_stage is False
           and abs(back.pose.yaw_deg - 30.0) < 1e-9
           and abs(back.pose.stage_zero_mm["y"] - 12.0) < 1e-9),
          json.dumps(back.to_dict())[:120])

    coils = omach.CoilSet.load(coil_path)
    check("the saved scale is applied to the file's currents",
          abs(back.current(coils["C2"]) - (-3.0 * 0.01)) < 1e-12,
          f"{back.current(coils['C2'])} A")

    # A file naming coils or a configuration this coil set does not have is
    # the case that must not silently produce a switch for nothing.
    stale = omach.MachineConfig(coil_file=coil_path,
                                configuration="BiotSavart9",
                                energised=["C1", "C9"])
    lost = stale.adopt(coils)
    check("a configuration that is not in the file is dropped, and said so",
          stale.configuration == "BiotSavart1" and len(lost) == 2, str(lost))
    check("so is a coil that is not in the file",
          stale.energised == ["C1"], str(stale.energised))

    fresh = omach.MachineConfig(coil_file=coil_path)
    fresh.adopt(coils)
    check("a placement that has never chosen starts with every coil on",
          fresh.energised == ["C1", "C2"], str(fresh.energised))

    # ---- what a field map carries away with it ----
    meta = back.to_scan_meta(coils, {"x": 10.0, "y": 0.0, "z": 0.0})
    check("a map records which coils were on and at what current",
          meta["coils"]["C2"]["on"] is True
          and meta["coils"]["C1"]["on"] is False
          and abs(meta["coils"]["C2"]["amp_turns"] + 0.03) < 1e-12,
          json.dumps(meta["coils"]))
    check("and where the probe was, in machine coordinates",
          "probe_origin_mm" in meta and meta["pose"]["yaw_deg"] == 30.0,
          str(meta.get("probe_origin_mm")))
    check("and which file the coils came from",
          os.path.samefile(meta["coil_file"], coil_path), meta["coil_file"])

    # The route this metadata takes into a saved map. A rename here would
    # leave the GUI passing a keyword nothing reads, and the map would come
    # back with no machine in it and no complaint from anything.
    params = inspect.signature(oscan.run_scan).parameters
    check("run_scan still takes the metadata a map is annotated with",
          "extra_meta" in params, ", ".join(params))

    missing = omach.MachineConfig.load_or_default(
        os.path.join(workdir, "not_here.json"))
    check("a missing placement file is not an error",
          missing.coil_file == "" and missing.energised is None)
    problems = []
    with open(os.path.join(workdir, "broken.json"), "w",
              encoding="utf-8") as f:
        f.write("{not json")
    omach.MachineConfig.load_or_default(
        os.path.join(workdir, "broken.json"), on_error=problems.append)
    check("a placement file that will not parse is reported, not swallowed",
          len(problems) == 1, str(problems))
    check("and a coil file that will not parse is too",
          omach.CoilSet.load_or_none(os.path.join(workdir, "broken.json"),
                                     on_error=problems.append) is None
          and len(problems) == 2, str(problems[-1:]))


def test_machine_tab(app, workdir):
    """The Machine tab, driven the way a person drives it."""
    print("\nmachine tab")
    coil_path = _simsopt_file(os.path.join(workdir, "tab_coils.json"))
    cfg_path = os.path.join(workdir, "tab_machine.json")
    omach.MachineConfig(coil_file=coil_path, coil_radius_mm=20.0,
                        energised=["C1", "C2"]).save(cfg_path)

    ns = argparse.Namespace(
        uut=None, demo=True, replay=None, no_connect=True,
        geometry=os.path.join(workdir, "tab_geom.json"),
        calibration=os.path.join(workdir, "tab_cal.json"),
        machine=cfg_path,
        out_dir=os.path.join(workdir, "tabcaps"),
        screenshot=None, screenshot_tab=0, screenshot_warmup=0)
    win = gui.MainWindow(ns)
    try:
        titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
        check("the window has a Machine tab", "Machine" in titles, str(titles))
        check("the coil file named in the placement is loaded on startup",
              win.session.coils is not None and len(win.session.coils) == 2,
              "nothing loaded" if win.session.coils is None else win.session.coils.note)
        check("every coil is listed", win.tab_machine.tbl_coils.rowCount() == 2,
              f"{win.tab_machine.tbl_coils.rowCount()} rows")

        # Switch a coil off through the table, as a click does.
        item = win.tab_machine.tbl_coils.item(0, 0)
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        check("unticking a coil switches it off",
              win.session.machine.energised == ["C2"], str(win.session.machine.energised))
        check("and the view is told",
              win.tab_machine.machine_view.energised == {"C2"},
              str(win.tab_machine.machine_view.energised))

        # A coil that is off is still solid: putting the probe on top of the
        # coil that was just switched off must still report a collision.
        on_coil = win.session.coils["C1"].points_mm[0]
        for attr, value in zip(("x_mm", "y_mm", "z_mm"), on_coil):
            win.tab_machine.machine_pose_spins[attr].setValue(float(value))
        win.tab_machine.refresh_machine(force=True)
        check("driving the probe onto a switched-off coil still collides",
              "INSIDE" in win.tab_machine.lbl_clearance.text(), win.tab_machine.lbl_clearance.text())

        win.tab_machine.machine_pose_spins["x_mm"].setValue(float(on_coil[0]) + 4000.0)
        win.tab_machine.refresh_machine(force=True)
        check("and moving it away again clears",
              "clear of" in win.tab_machine.lbl_clearance.text(),
              win.tab_machine.lbl_clearance.text())

        # The pose the tab is showing is the pose that gets written down.
        win.tab_machine.on_machine_save()
        saved = omach.MachineConfig.load(cfg_path)
        check("Save writes the pose that is on screen",
              abs(saved.pose.x_mm - (on_coil[0] + 4000.0)) < 1e-6
              and saved.energised == ["C2"],
              f"{saved.pose.x_mm:.1f} mm, {saved.energised}")

        # What a field map would carry, built the way on_scan_start builds it.
        meta = win.session.machine.to_scan_meta(win.session.coils, None)
        check("a map started now would record the machine around it",
              meta["summary"].startswith("1/2 coils energised"),
              meta["summary"])
    finally:
        win.close()
        app.processEvents()


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
          all(b >= a for a, b in itertools.pairwise(xs)), f"{xs}")
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
        # Not a failure. Kinesis is Windows-only proprietary software that a CI
        # runner does not have and cannot be given, and everything below this
        # point tests the binding AGAINST that DLL -- there is nothing left to
        # check without it. What matters is that the skip is visible, so a
        # bench run that lost its Kinesis install does not look like a pass.
        skip("the Kinesis C API loads", str(exc))
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

    # Soft limits live in the same block, and the same button rewrites it --
    # so the same argument applies with more at stake. A save that dropped
    # limit_mm would remove the only thing keeping the head out of the
    # fixture, during routine housekeeping, silently.
    ostage.save_axis_map({"z": "45502854"}, path,
                         frames={"z": {"invert": True,
                                       "limit_mm": (20.0, 250.0)}})
    check("a soft limit round trips",
          ostage.load_axis_frames(path)["z"]["limit_mm"] == (20.0, 250.0),
          str(ostage.load_axis_frames(path)))
    ostage.save_axis_map({"z": "45502854"}, path,
                         frames={"z": {"invert": True}})
    check("re-saving the frame without limits preserves the soft limit",
          ostage.load_axis_frames(path)["z"]["limit_mm"] == (20.0, 250.0))
    ostage.save_axis_map({"z": "45502854"}, path)
    check("saving the map alone preserves the soft limit",
          ostage.load_axis_frames(path)["z"]["limit_mm"] == (20.0, 250.0))
    check("an axis with no soft limit reports None",
          ostage.load_axis_frames(path).get("x", {}).get("limit_mm") is None)

    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["frame"]["z"]["limit_mm"] = [20.0]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    try:
        ostage.load_axis_frames(path)
        ok = False
    except ostage.StageError:
        ok = True
    check("a malformed soft limit is refused rather than half-applied", ok,
          "a limit_mm that silently became 'no limit' is the worst outcome")
    doc["frame"]["z"]["limit_mm"] = [20.0, 250.0]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)

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


def test_stage_safety():
    """The emergency-stop conventions, without a stage attached.

    All of this is the logic that decides whether a move is allowed to be
    commanded, which is the half that can be checked on a bench with nothing
    plugged in -- and the half where a mistake is a collision rather than an
    exception.
    """
    print("\nmotion interlock and position trust")

    il = ostage.MotionInterlock()
    check("a fresh interlock is clear", il.tripped is None)
    il.trip("z hit the hard limit")
    il.trip("operator pressed the emergency stop")
    check("the FIRST reason to trip is the one kept",
          il.tripped == "z hit the hard limit",
          "the cascade would otherwise overwrite the cause with the symptom")
    try:
        il.require_clear("move x to 10 mm")
        ok = False
    except ostage.MotionInterlocked:
        ok = True
    check("a latched interlock refuses a move", ok)
    check("MotionInterlocked is a StageError",
          issubclass(ostage.MotionInterlocked, ostage.StageError),
          "callers that catch StageError must not miss a stop")
    check("resetting reports what it cleared",
          il.reset() == "z hit the hard limit" and il.tripped is None)

    # One latch for the machine, not one per axis: the axis that faults is not
    # necessarily the one about to hit something.
    ss = ostage.StageSet({"x": ostage.Stage("1", name="x"),
                          "z": ostage.Stage("2", name="z")})
    check("every axis in a set shares one interlock",
          ss["x"].interlock is ss["z"].interlock is ss.interlock)
    ss.emergency_stop("test")
    check("an emergency stop with nothing open still latches",
          ss.interlock.tripped == "test",
          "or a stop pressed while disconnected is forgotten")
    check("and it does not raise on the way", True)

    # Position trust. The whole point is that it is NOT the homed bit.
    st = _framed("x", False)
    check("an unopened, unhomed axis is not trusted", not st.position_trusted)
    st._trusted = True
    check("trust still requires the stage to be open",
          not st.position_trusted,
          "the controller's homed bit cannot even be read on a closed stage")
    check("and neither property raises on a closed stage",
          st.distrust_reason == "the stage is not open",
          "these are read from the stop path, which must not throw")
    st.distrust("stopped immediately")
    st.is_open = True                       # nothing to talk to, just the flag
    check("distrust says why", st.distrust_reason == "stopped immediately")
    st.is_open = False

    # Soft limits: a restriction, never a permission.
    check("no soft limit means the whole travel",
          ostage.Stage._resolve_limit((0.0, 300.0), None) == (0.0, 300.0))
    check("a soft limit narrows the travel",
          ostage.Stage._resolve_limit((0.0, 300.0), (20.0, 280.0))
          == (20.0, 280.0))
    check("a soft limit given the wrong way round is sorted, not refused",
          ostage.Stage._resolve_limit((0.0, 300.0), (280.0, 20.0))
          == (20.0, 280.0))
    check("a soft limit WIDER than the travel is clamped, not obeyed",
          ostage.Stage._resolve_limit((0.0, 300.0), (-50.0, 900.0))
          == (0.0, 300.0),
          "a config that asks for more travel than exists is a typo")
    try:
        ostage.Stage._resolve_limit((0.0, 300.0), (400.0, 500.0))
        ok = False
    except ostage.StageError:
        ok = True
    check("a soft limit entirely outside the travel is refused", ok)

    lim = _framed("z", True)
    lim._limit_cfg = (30.0, 200.0)
    lim._resolve_frame()
    check("the envelope survives the reversed-axis frame",
          lim.limit_mm == (30.0, 200.0) and lim.travel_mm == (0.0, 300.0),
          f"limit {lim.limit_mm}, travel {lim.travel_mm}")

    # "Declared as the whole travel" and "never configured" allow exactly the
    # same movement and are not the same thing: one is a measurement, the
    # other is a gap. Only the gap should still be warning about itself.
    bare = _framed("x", False)
    check("an axis with no limit_mm says so", not bare.limit_declared)
    told = _framed("x", False)
    told._limit_cfg = (0.0, 300.0)
    told._resolve_frame()
    check("an axis declared as its whole travel is still declared",
          told.limit_declared and told.limit_mm == told.travel_mm,
          f"limit {told.limit_mm}, travel {told.travel_mm}")


def test_stage_home_order(workdir):
    """Which axis retracts first is a declared fact, not an emergent one."""
    print("\nhoming order")
    path = os.path.join(workdir, "order.json")
    ostage.save_axis_map({"x": "1", "y": "2", "z": "3"}, path)
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["home_order"] = ["z", "y"]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)

    check("the home order round trips",
          ostage.load_home_order(path) == ("z", "y"))
    check("an axis that is not on the machine is ignored, not an error",
          ostage.load_home_order(path, axes=("x", "z")) == ("z",),
          "unplugging an axis must not invalidate the safe order")
    check("a missing file means no declared order",
          ostage.load_home_order(os.path.join(workdir, "nope.json")) == ())

    ss = ostage.StageSet({n: ostage.Stage(n, name=n) for n in ("x", "y", "z")},
                         home_order=("z", "y"))
    check("undeclared axes follow the declared ones rather than vanishing",
          ss.home_sequence() == ["z", "y", "x"], str(ss.home_sequence()))
    check("with nothing declared the order is the map order",
          ostage.StageSet({n: ostage.Stage(n, name=n)
                           for n in ("x", "y")}).home_sequence() == ["x", "y"])


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


def test_stage_motion(workdir):
    """The motion profile: the maths, the config, and what reaches a Stage."""
    print("\nstage motion profile")
    # A move only reaches the cap if it is long enough to ramp up to it. This
    # is the whole reason a bigger jog step sounds different from a small one.
    check("a short move never reaches the velocity cap",
          abs(ostage.peak_speed_mm_s(1.0, 20.0, 20.0) - 4.472) < 1e-3,
          "1 mm at 20/20 peaks at 4.5 mm/s, not 20")
    check("a long move is capped by velocity, not distance",
          ostage.peak_speed_mm_s(300.0, 8.0, 10.0) == 8.0)
    check("the shipped profile is what makes 5 mm the audible threshold",
          abs(ostage.peak_speed_mm_s(5.0, 20.0, 20.0) - 10.0) < 1e-9
          and ostage.peak_speed_mm_s(20.0, 20.0, 20.0) == 20.0,
          "5 mm -> 10 mm/s but 20 mm -> the full 20")
    check("no step size can outrun the configured cap",
          max(ostage.peak_speed_mm_s(d, ostage.DEFAULT_VEL_MM_S,
                                     ostage.DEFAULT_ACCEL_MM_S2)
              for d in (1, 5, 20, 100, 300)) <= ostage.DEFAULT_VEL_MM_S)

    # ---- the ceiling ----
    # A velocity setting is not a promise: anything that opens the device
    # outside this module leaves the controller at its shipped 20 mm/s. So the
    # cap has to hold at every door into the module, not just the front one.
    check("the default profile is under the ceiling",
          ostage.DEFAULT_VEL_MM_S <= ostage.MAX_VEL_MM_S)
    check("a velocity above the ceiling is clamped",
          ostage.clamp_velocity(50.0) == ostage.MAX_VEL_MM_S
          and ostage.clamp_velocity(2.0) == 2.0)
    check("resolving a profile clamps what it sends to the controller",
          ostage.Stage.resolve_profile(20.0, 20.0, (8.0, 10.0))
          == (ostage.MAX_VEL_MM_S, 20.0),
          "20 mm/s asked for, ceiling sent")
    check("resolving with None keeps what the controller already had",
          ostage.Stage.resolve_profile(None, None, (5.0, 7.0)) == (5.0, 7.0))
    for bad in ((0.0, 5.0), (5.0, 0.0), (-1.0, 5.0)):
        try:
            ostage.Stage.resolve_profile(bad[0], bad[1], (8.0, 10.0))
            ok = False
        except ostage.StageError:
            ok = True
        check(f"a profile of {bad} is refused", ok)
    # Every mover must re-apply the profile, or the ceiling is only true until
    # something else touches the controller.
    src = inspect.getsource(ostage.Stage)
    for mover in ("move_to", "move_by"):
        body = src.split(f"def {mover}(")[1].split("\n    def ")[0]
        check(f"{mover} re-applies the profile before it moves",
              "enforce_profile()" in body)

    # A move's deadline has to scale with the move. A fixed one fires on a
    # long slow traverse that is going perfectly -- and since a timeout now
    # stops the axis and marks its position lost, a spurious one costs a
    # re-home and, mid-raster, the rest of the map.
    check("a short move's time is set by acceleration, not the speed cap",
          abs(ostage.move_time_s(1.0, 6.0, 10.0) - 0.632) < 1e-3,
          f"{ostage.move_time_s(1.0, 6.0, 10.0):.3f} s for 1 mm at 6/10")
    check("a long move's time is set by the speed cap",
          abs(ostage.move_time_s(300.0, 6.0, 10.0) - 50.6) < 0.05,
          f"{ostage.move_time_s(300.0, 6.0, 10.0):.1f} s for the full traverse")
    check("the deadline for a full traverse leaves room but is not open-ended",
          150.0 < ostage.move_timeout_s(300.0, 6.0, 10.0) < 200.0,
          f"{ostage.move_timeout_s(300.0, 6.0, 10.0):.0f} s")
    check("a crawl gets a deadline it can actually meet",
          ostage.move_timeout_s(300.0, 0.1, 10.0) > 3000.0,
          f"{ostage.move_timeout_s(300.0, 0.1, 10.0):.0f} s at 0.1 mm/s -- "
          f"the old fixed 180 s stopped this move at 18 mm")
    check("a tiny move still gets a floor, not milliseconds",
          ostage.move_timeout_s(0.001, 6.0, 10.0) == 30.0)

    path = os.path.join(workdir, "motion.json")
    ostage.save_axis_map({"x": "45502844", "z": "45538374"}, path)
    got = ostage.load_axis_motion(path, ["x", "z"])
    check("a config with no motion block still opens quiet",
          got["x"] == (ostage.DEFAULT_VEL_MM_S, ostage.DEFAULT_ACCEL_MM_S2),
          f"{got['x']} rather than Kinesis's 20/20")
    ostage.save_axis_motion(path, velocity_mm_s=6.0, accel_mm_s2=7.0)
    ostage.save_axis_motion(path, velocity_mm_s=3.0, axis="z")
    got = ostage.load_axis_motion(path, ["x", "z"])
    check("the profile round trips and one axis can override it",
          got == {"x": (6.0, 7.0), "z": (3.0, 7.0)}, str(got))

    # A file written before the ceiling existed, or edited by hand, must not be
    # able to reintroduce the shipped 20 mm/s through the back door.
    hand_edited = os.path.join(workdir, "loud.json")
    with open(hand_edited, "w") as fh:
        json.dump({"axes": {"x": "45502844"},
                   "motion": {"velocity_mm_s": 20.0, "accel_mm_s2": 20.0}}, fh)
    loud = ostage.load_axis_motion(hand_edited, ["x"])
    check("a config asking for 20 mm/s is clamped on the way in",
          loud["x"][0] == ostage.MAX_VEL_MM_S, str(loud["x"]))
    ostage.save_axis_motion(hand_edited, velocity_mm_s=99.0)
    with open(hand_edited) as fh:
        check("and saving one is clamped on the way out",
              json.load(fh)["motion"]["velocity_mm_s"] == ostage.MAX_VEL_MM_S)
    check("saving the profile leaves the axis map alone",
          ostage.load_axis_map(path) == {"x": "45502844", "z": "45538374"})

    # It is no use in the file if it does not reach the hardware, and the one
    # link that cannot be seen from outside is Stage carrying it to open().
    ss = ostage.StageSet.from_config(path, serials=["45502844", "45538374"])
    check("each stage is built with its own profile",
          (ss["x"]._vel_cfg, ss["x"]._accel_cfg) == (6.0, 7.0)
          and (ss["z"]._vel_cfg, ss["z"]._accel_cfg) == (3.0, 7.0))
    check("a stage can still be opened without changing how it moves",
          ostage.Stage("45502844", vel_mm_s=None, accel_mm_s2=None)._vel_cfg
          is None)



def _synth_magnet_run(g, gain, magnet_world, span_mm=260.0, step_mm=4.0,
                      shift_pose=None, eccentric_mm=0.0):
    """A guided run against a 1/r^3 dipole, with a known per-sensor gain.

    shift_pose: if given, that pose is also displaced sideways -- the mistake
    the routine warns about, used here to check it actually shows up.

    eccentric_mm: the head is not concentric with the axis it is turned about,
    by this much. The offset rides WITH the tube, so it points at the magnet in
    one pose and away from it in the opposite one -- which is what makes the
    error separable from gain.
    """
    def roll(deg):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    ys = np.arange(0.0, span_mm + 1e-9, step_mm)
    run = omag.MagnetRun(magnet_note="synthetic dipole")
    for pose in range(omag.N_POSES):
        R = roll(90.0 * pose)
        off = np.array([12.0, 0.0, 0.0]) if pose == shift_pose else np.zeros(3)
        rows = []
        for y in ys:
            b = np.zeros((16, 3))
            for i in range(16):
                p = R @ (pgeom.to_world(g.position(i + 1))
                         + np.array([0.0, 0.0, eccentric_mm])) + off
                p = p + np.array([0.0, y, 0.0])
                r = magnet_world - p
                d = np.linalg.norm(r)
                bw = (r / d) * (50.0 / d) ** 3
                b[i] = gain[i] * (g.rotations()[i].T
                                  @ (pgeom.MOUNT_ROT.T @ (R.T @ bw)))
            rows.append(b)
        run.add(omag.PoseSweep(pose, ys, np.array(rows)))
    return run


def test_magnet_routine():
    """The guided run must recover gain WITHOUT knowing the geometry."""
    print("\nguided magnet calibration")
    g = pgeom.Geometry()
    rng = np.random.default_rng(11)
    gain = rng.normal(1.0, 0.08, 16)
    magnet = np.array([0.0, 200.0, g.fsv_radius_mm + 25.0])
    run = _synth_magnet_run(g, gain, magnet)

    check("a run is complete at four poses", run.complete and len(run) == 4)
    resp, best = run.response()
    # The whole claim: each sensor is loudest in the pose that turned its own
    # face to the magnet, which is what makes the 16 peaks comparable.
    faces = np.array([g.face(i + 1) for i in range(16)])
    same_face_same_pose = all(
        len({int(best[i]) for i in range(16) if faces[i] == f}) == 1
        for f in range(pgeom.N_FACES))
    check("each face answers in exactly one pose", same_face_same_pose,
          f"poses per sensor: {best.tolist()}")

    trim = run.trim()
    recovered = trim * gain
    err = float(np.abs(recovered / np.median(recovered) - 1).max())
    check("the trim recovers the injected gain with no geometry model",
          err < 0.02, f"worst residual {100 * err:.2f} % on a 4 mm sample grid")

    # A fixed magnet at one place cannot do this: the raw peaks of a single
    # pose span the 1/r^3 spread, which is what the extra poses remove.
    one_pose = run.sweeps[0].peaks
    check("one pose alone would have been badly unfair",
          one_pose.max() / one_pose.min() > 5 * (resp.max() / resp.min()),
          f"single pose spreads {one_pose.max() / one_pose.min():.0f}x, "
          f"four poses {resp.max() / resp.min():.2f}x")

    # ---- what the run says about where the sensors are ----
    # The sweep meets the tip first, because the tube's +Z maps to +y and the
    # head is driven the +ve way -- so the far sensor reaches a fixed magnet at
    # a SMALLER stage position. Read off MOUNT_ROT, not written down, so a
    # remounted probe reverses it automatically.
    slots = run.measured_slots()
    check("a correct file measures back exactly as it was written",
          all(slots[sid] == g.slot(sid) for sid in range(1, 17)),
          "the run reproduces the layout it was generated from")

    # Now the case that matters: a file that is wrong the way this rig's was,
    # with one face's sensors numbered backwards along the tube.
    # Built from the same layout the synthetic run was generated against --
    # NOT from the file on disk, which is this rig's real measured geometry and
    # would make the comparison meaningless.
    scrambled = pgeom.Geometry()
    for spec in scrambled.sensors:
        if spec["face"] == 0:
            spec["slot"] = pgeom.SENSORS_PER_FACE - 1 - spec["slot"]
    check("a reversed face is caught", bool(run.check_geometry(scrambled)),
          "; ".join(run.check_geometry(scrambled))[:90])
    changed = run.apply_to_geometry(scrambled)
    check("applying the run fixes exactly the reversed face",
          sorted(sid for sid, _w, _n in changed)
          == [sid for sid in range(1, 17) if g.face(sid) == 0],
          str([(f"S{s}", w, n) for s, w, n in changed]))
    check("and the file then agrees with the run that corrected it",
          run.check_geometry(scrambled) == [])
    check("a corrected file is marked as measured, not as a generated layout",
          scrambled.mapping == pgeom.MEASURED)
    try:
        pgeom._default_sensors(pgeom.MEASURED)
        ok = False
    except ValueError:
        ok = True
    check("and cannot be regenerated back into a guess", ok)

    groups = run.slot_groups()
    check("the sensors pass the magnet in rings of four",
          [len(ids) for _at, ids in groups] == [pgeom.SENSORS_PER_FACE] * 4,
          " | ".join("+".join(f"S{s}" for s in ids) for _at, ids in groups))
    check("a correct geometry file draws no complaint",
          run.check_geometry(g) == [], "; ".join(run.check_geometry(g)))

    # The failure this routine is most likely to suffer in the room.
    bad = _synth_magnet_run(g, gain, magnet, shift_pose=2)
    bad_trim = bad.trim() * gain
    check("a pose that also shifted sideways changes the answer",
          float(np.abs(bad_trim / np.median(bad_trim) - 1).max()) > 0.05,
          "so the wizard's insistence on turning about the axis is not "
          "decoration")

    # ---- the setup fault that looks exactly like gain ----
    bal = run.face_balance(g)
    check("a concentric head shows opposite faces agreeing",
          not bal["notes"],
          "  ".join(f"{pgeom.FACE_NAMES[f]}/{pgeom.FACE_NAMES[o]} {r:.3f}x"
                    for f, o, r, _e in bal["pairs"]))

    # 2 mm off-centre against a magnet ~25 mm away: a few per cent of distance,
    # which 1/r^3 turns into a ten per cent "gain" error on two faces.
    ecc = _synth_magnet_run(g, gain, magnet, eccentric_mm=2.0)
    eb = ecc.face_balance(g)
    check("an off-centre head is caught by the opposite-face check",
          bool(eb["notes"]),
          "  ".join(f"{pgeom.FACE_NAMES[f]}/{pgeom.FACE_NAMES[o]} {r:.3f}x"
                    for f, o, r, _e in eb["pairs"]))
    check("and the report stops calling the spread gain",
          "NOT gain alone" in ecc.report(g))
    # Both pairs move, not just the one the offset points along: the chips sit
    # ~14 mm off the rotation axis, so a sideways offset changes the in-plane
    # separation to first order too. What holds is that the pair the offset
    # points ALONG is the worse of the two.
    by_pair = {f: abs(r - 1.0) for f, _o, r, _e in eb["pairs"]}
    check("the pair the offset points along is the worst affected",
          max(by_pair, key=by_pair.get) == 0,
          "  ".join(f"faces {f}/{f + 2}: {v:.3f}" for f, v in by_pair.items()))
    # It is separable BECAUSE it is antisymmetric: the product of two opposite
    # faces survives the offset even though each of them does not.
    prod = {f: eb["means"][f] * eb["means"][o] for f, o, _r, _e in eb["pairs"]}
    clean = {f: bal["means"][f] * bal["means"][o] for f, o, _r, _e in bal["pairs"]}
    check("the product of opposite faces survives the offset",
          all(abs(prod[f] / clean[f] - 1.0) < 0.02 for f in prod),
          "  ".join(f"{prod[f] / clean[f]:.3f}" for f in prod))

    sideways = omag.MagnetRun(run.sweeps, axis="x")
    try:
        sideways.measured_slots()
        ok = False
    except ValueError:
        ok = True
    check("sweeping across the tube instead of along it is refused", ok,
          "the equal-approach argument does not hold, and neither does the "
          "slot order")

    # A run abandoned early still reports what it established.
    part = omag.MagnetRun(run.sweeps[:2])
    check("two poses still report, and say what is missing",
          not part.complete and "2 of 4 poses" in part.report(g)
          and any("poses recorded" in n for n in part.check_geometry(g)))


def _synth_magnet_passes(g, gain, magnet_world, jitter, standoff_mm=30.0,
                         plane=True, dither=True, seed=4):
    """A guided run with ARM MISPLACEMENT, measured by all three passes.

    `jitter` is (across, along, standoff) rms in mm, applied per sensor in the
    probe's own frame so that it rides with the head through the four poses --
    which is what a real arm does, and what makes the error impossible to
    average away by rolling.
    """
    rng = np.random.default_rng(seed)
    jit = rng.normal(0.0, 1.0, (16, 3)) * np.asarray(jitter, float)

    def roll(deg):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    def field(stage, pose):
        R = roll(90.0 * pose)
        b = np.zeros((16, 3))
        for i in range(16):
            p = R @ (pgeom.to_world(g.position(i + 1)) + jit[i]) + stage
            r = magnet_world - p
            d = np.linalg.norm(r)
            b[i] = gain[i] * (r / d) * (50.0 / d) ** 3
        return b

    across = "x" if plane else None
    normal = "z" if dither else None
    run = omag.MagnetRun(magnet_note="synthetic, misplaced arms",
                         across=across, normal=normal)
    # Ring `slot` meets the magnet at stage y = magnet_y - sensor_y, so the
    # last slot is the first one past it. Start a clearance short of that.
    span, step = omag.suggested_sweep(g)
    last = g.first_sensor_z_mm + (pgeom.SENSORS_PER_FACE - 1) * g.plate_pitch_mm
    ys = np.arange(0.0, span + 1e-9, step) + magnet_world[1] - last - 40.0

    for pose in range(omag.N_POSES):
        pos = np.array([[0.0, y, 0.0] for y in ys])
        rows = np.array([field(p, pose) for p in pos])
        sw = omag.PoseSweep(pose, pos, rows, axis="y", across=across,
                            normal=normal, axes=("x", "y", "z"))
        rings = omag.ring_positions(sw)

        if plane:
            half, xstep = omag.suggested_plane(standoff_mm)
            xs = np.arange(-half, half + 1e-9, xstep)
            pos = np.array([[x, y, 0.0] for y in rings for x in xs])
            rows = np.array([field(p, pose) for p in pos])
            sw.merge(omag.PoseSweep(pose, pos, rows, axis="y", across="x",
                                    normal=normal, axes=("x", "y", "z")))
        if dither:
            xc = 0.0
            if plane:
                vals = sw.peak_across_mm[np.argsort(sw.peaks)[-4:]]
                xc = float(np.mean(vals[np.isfinite(vals)]))
            zs = omag.suggested_dither(standoff_mm)
            for y in rings:
                pos = np.array([[xc, y, z] for z in zs])
                rows = np.array([field(p, pose) for p in pos])
                sw.dithers.append(omag.Dither(zs, rows, at_mm=y))
        run.add(sw)
    return run


def _trim_error_pct(run, gain):
    """How much of the recovered trim is NOT the injected gain."""
    prod = run.trim() * gain
    return 100.0 * float(np.std(prod / np.median(prod)))


def test_magnet_plane():
    """The plane and the dither must beat a bare axial sweep on bent arms.

    The axial-only run is not a straw man -- it is what this routine did until
    the passes were added, and on perfectly built arms it is still right. What
    it cannot do is tell a chip that is a millimetre closer to the magnet from
    a chip with 15 % more gain, and that is what is checked here.
    """
    print("\nguided magnet calibration -- plane and standoff passes")
    g = pgeom.Geometry()
    rng = np.random.default_rng(23)
    gain = rng.normal(1.0, 0.05, 16)
    magnet = np.array([0.0, 200.0, g.fsv_radius_mm + 25.0])

    # --- the fit, on its own, against a known distance
    z = omag.suggested_dither(30.0)
    d, n, _rms, _sigma = omag.fit_falloff(z, 100.0 / np.abs(30.0 - z) ** 3.0)
    check("the dither fit recovers a known standoff", abs(d - 30.0) < 0.2,
          f"got {d:.3f} mm, want 30")
    check("and the exponent that produced it", abs(n - 3.0) < 0.05,
          f"got 1/r^{n:.3f}, want 3")
    check("a dither with no contrast is refused rather than guessed",
          not np.isfinite(omag.fit_falloff(z, np.ones_like(z))[0]))
    check("three points claim no precision, having none to spare",
          not np.isfinite(omag.fit_falloff(
              z[:3], 100.0 / np.abs(30.0 - z[:3]) ** 3.0)[3]))

    # --- a perfectly built probe: the passes must not make it worse
    perfect_axial = _synth_magnet_passes(g, gain, magnet, (0, 0, 0),
                                         plane=False, dither=False)
    perfect_full = _synth_magnet_passes(g, gain, magnet, (0, 0, 0))
    e_axial = _trim_error_pct(perfect_axial, gain)
    e_full = _trim_error_pct(perfect_full, gain)
    check("on straight arms the extra passes cost nothing",
          e_full < max(2.0 * e_axial, 1.0),
          f"axial {e_axial:.2f} % -> all three {e_full:.2f} %")

    # --- a real probe: arms a millimetre out, one direction at a time, so
    # each pass is checked against the error it actually claims to remove
    for label, jit in (("transverse", (1.0, 0.0, 0.0)),
                       ("standoff", (0.0, 0.0, 1.0)),
                       ("all three axes", (1.0, 1.0, 1.0))):
        e_axial = _trim_error_pct(
            _synth_magnet_passes(g, gain, magnet, jit, plane=False,
                                 dither=False), gain)
        e_full = _trim_error_pct(
            _synth_magnet_passes(g, gain, magnet, jit), gain)
        check(f"the passes recover gain through {label} misplacement",
              e_full < 0.4 * e_axial,
              f"{e_axial:.2f} % of fake gain -> {e_full:.2f} %")

    jitter = (1.0, 1.0, 1.0)
    bent_axial = _synth_magnet_passes(g, gain, magnet, jitter,
                                      plane=False, dither=False)
    bent_plane = _synth_magnet_passes(g, gain, magnet, jitter, dither=False)
    bent_full = _synth_magnet_passes(g, gain, magnet, jitter)
    e_axial = _trim_error_pct(bent_axial, gain)
    e_plane = _trim_error_pct(bent_plane, gain)
    e_full = _trim_error_pct(bent_full, gain)
    check("a bare axial sweep turns bent arms into fake gain", e_axial > 3.0,
          f"{e_axial:.2f} % of trim is placement, not gain")

    # The counter-intuitive one, pinned so nobody "optimises" pass C away:
    # peaking over the plane moves every chip to its true closest approach,
    # which is nearer, and that amplifies whatever first-order error is left.
    # The cut is only a win once the dither is there to collect it.
    check("the plane WITHOUT the dither is worse, not better",
          e_plane > e_axial,
          f"{e_axial:.2f} % -> {e_plane:.2f} % -- the cut moves the chips "
          f"closer, so what is left hurts more")
    check("and the two together are what actually pay",
          e_full < 0.4 * e_axial,
          f"{e_axial:.2f} % -> {e_full:.2f} % with all three passes")
    check("the cut is what makes the dither's model true",
          np.nanmedian(bent_full.standoffs())
          < 0.7 * np.nanmedian(_synth_magnet_passes(
              g, gain, magnet, jitter, plane=False).standoffs()),
          f"standoff reads {np.nanmedian(bent_full.standoffs()):.1f} mm under "
          f"the magnet, {np.nanmedian(_synth_magnet_passes(g, gain, magnet, jitter, plane=False).standoffs()):.1f} mm off to one side")

    # --- the measurements the passes expose, not just the trim they feed
    across = bent_full.across_positions()
    check("the plane reports where the arms actually are",
          across is not None and np.isfinite(across).all()
          and float(np.ptp(across)) > 0.5,
          f"transverse peaks span {float(np.ptp(across)):.2f} mm")
    stand = bent_full.standoffs()
    n_fit = int(np.isfinite(stand).sum())
    check("the dither measures a standoff for every sensor", n_fit == 16,
          f"{n_fit} of 16 fitted, median {np.nanmedian(stand):.1f} mm")
    fall = bent_full.falloffs()
    check("and agrees the field falls like a dipole",
          2.5 < float(np.nanmedian(fall)) < 3.5,
          f"measured 1/r^{np.nanmedian(fall):.2f}")

    # --- degradation: a run without pass C must not silently invent one
    corr = bent_plane.standoff_correction()
    check("with no dither the standoff correction is exactly the identity",
          np.allclose(corr, 1.0), f"max deviation {np.max(np.abs(corr - 1)):.2e}")
    check("a one-axis run still analyses after the passes were added",
          np.isfinite(bent_axial.trim()).all()
          and bent_axial.across_positions() is None)


def test_magnet_plane_persistence(workdir):
    g = pgeom.Geometry()
    rng = np.random.default_rng(31)
    run = _synth_magnet_passes(g, rng.normal(1.0, 0.05, 16),
                               np.array([0.0, 200.0, g.fsv_radius_mm + 25.0]),
                               (1.0, 1.0, 1.0))
    path = run.save(os.path.join(workdir, "magcal_plane"))
    back = omag.MagnetRun.load(path)
    check("a plane run round trips through disk",
          len(back) == len(run)
          and np.allclose(back.trim(), run.trim())
          and back.across == "x" and back.normal == "z")
    check("and its dithers survive, standoffs and all",
          all(len(s.dithers) == pgeom.SENSORS_PER_FACE for s in back.sweeps)
          and np.allclose(back.standoffs(), run.standoffs(), equal_nan=True))


def test_magnet_persistence(workdir):
    g = pgeom.Geometry()
    rng = np.random.default_rng(5)
    run = _synth_magnet_run(g, rng.normal(1.0, 0.05, 16),
                            np.array([0.0, 200.0, g.fsv_radius_mm + 25.0]),
                            span_mm=120.0, step_mm=10.0)
    path = run.save(os.path.join(workdir, "magcal_test"))
    back = omag.MagnetRun.load(path)
    check("a guided run round trips through disk",
          len(back) == len(run)
          and np.allclose(back.trim(), run.trim())
          and back.axis == run.axis)


def test_help_index():
    """The Help tab is only as good as its search."""
    print("\nhelp index")
    topics = ohelp.load_topics()
    n_gui = sum(1 for t in topics if t.source == "this window")
    check("the README is indexed into topics", len(topics) > 20,
          f"{len(topics)} topics, {n_gui} of them about the window")
    # The window's own topics exist to cover what a document about the
    # instrument cannot. If they start multiplying, they are being written
    # instead of the README, which is how help text and docs drift apart.
    check("the window's own topics stay a small minority", n_gui < 8,
          f"{n_gui} hand-written topics against {len(topics) - n_gui} indexed")
    check("every topic has a title and a body",
          all(t.title and t.body.strip() for t in topics))

    # Headings inside fenced code are comments, not topics. This one bit:
    # the README is full of bash blocks whose lines start with '#'.
    fenced = ohelp.split_markdown(
        "## Real\ntext\n\n```bash\n## not a heading\necho hi\n```\n\n"
        "## Also real\nmore\n")
    check("a '#' inside a code fence is not a heading",
          [t.title for t in fenced] == ["Real", "Also real"],
          str([t.title for t in fenced]))

    for query, want in (("homing", "Homing, and why the window asks"),
                        ("jog step loud", "Why a bigger jog step is louder, "
                                          "and what to do about it"),
                        ("guided magnet", "Step 5b"),
                        # Typed as a question, which is how anyone actually
                        # reaches for help. Without stop-word filtering this
                        # ranks by document length instead of by relevance.
                        ("why is it so loud", "Everything got slow, or loud"),
                        ("go is greyed out", "Go is greyed out, or a move is "
                                             "refused")):
        hits = ohelp.search(topics, query, limit=3)
        check(f"searching {query!r} finds its topic first",
              bool(hits) and hits[0].title.startswith(want),
              hits[0].title if hits else "nothing matched")
    check("a query of nothing but stop words still answers",
          bool(ohelp.search(topics, "how do I use the")),
          "dropping every term would leave a blank pane")
    check("a query that matches nothing returns nothing",
          ohelp.search(topics, "zzzqqq") == [])
    check("an empty query lists everything",
          len(ohelp.search(topics, "   ", limit=500)) == len(topics))


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


def test_survey_pairing():
    """The opposed pose is i + n/2, which only equals i + 2 at four poses."""
    print("\nlocation survey pairing")
    rng = np.random.default_rng(11)
    offsets = rng.normal(0, 0.4, (16, 3))          # dominate the field, as on the bench
    bt_true = 0.047                                # mT transverse

    def poses(n):
        """n poses of a rigid probe rolled about the tube axis."""
        out = []
        for k in range(n):
            phi = 2 * np.pi * k / n
            field = np.array([bt_true * np.cos(phi), bt_true * np.sin(phi), 0.0])
            out.append(offsets + field[None, :])
        return out

    for n in (4, 6, 8):
        d13, _d24, _ratio, worst = opcap.survey_consistency(poses(n),
                                                          np.ones(16, bool))
        # Opposed poses differ by exactly 2*bt whatever n is, so the recovered
        # transverse field must come back at bt_true -- not bt_true*sin(pi/n).
        check(f"{n} poses recover the true transverse field",
              abs(float(np.median(d13)) - bt_true * 1e3) < 1e-6,
              f"got {np.median(d13):.4f} uT, want {bt_true * 1e3:.4f}")
        check(f"{n} poses agree with themselves", worst < 1e-6,
              f"worst self-disagreement {worst:.3g} %")

    try:
        opcap.survey_consistency(poses(3), np.ones(16, bool))
        ok = False
    except ValueError:
        ok = True
    check("an odd pose count is refused rather than mispaired", ok)


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
    # The bench mounting: drawing-only, but a mirrored one would silently put
    # the probe's tip at the wrong end of the rig in every screenshot.
    check("the mount is a proper rotation",
          np.allclose(pgeom.MOUNT_ROT.T @ pgeom.MOUNT_ROT, np.eye(3))
          and np.linalg.det(pgeom.MOUNT_ROT) > 0.99)
    check("mounted horizontally: the tube axis lies along world +Y",
          np.allclose(pgeom.to_world(pgeom.TUBE_AXIS), [0, 1, 0]))
    pw = pgeom.to_world(g.positions())
    check("the tip end points forward, S16 with it",
          pw[15][1] > pw[12][1] and np.argmax(pw[:, 1]) % 4 == 3,
          f"S16 at y = {pw[15][1]:.0f} mm, S13 at y = {pw[12][1]:.0f} mm")
    check("|B| survives the mount rotation too",
          np.allclose(np.linalg.norm(v, axis=-1),
                      np.linalg.norm(pgeom.to_world(v), axis=-1)))

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

    # A range the part does not have must be refused where it is set, not
    # where it is first used -- a KeyError out of a 20 Hz GUI timer is a long
    # way from the typo in calibration.json that caused it.
    try:
        ocal.Calibration(ranges_mt=np.full(16, 25.0))
        ok = False
    except ValueError:
        ok = True
    check("a range the SENM3Dx does not have is refused at construction", ok)

    # Unipolar ranges put 0 V at count -32768, so the pedestal has to be
    # carried or every channel reads half a span low.
    check("the ADC range table carries volts-at-count-zero",
          ob.ADC_RANGES["+/-10V"][1] == 0.0
          and ob.ADC_RANGES["0-5V"][1] == 2.5)
    lay_uni = ob.Layout(ssb=96, adc_range="0-5V")
    counts = np.full((4, 32), -32768, dtype=np.int16)
    volts = ocal.assemble([counts, counts],
                          [lay_uni.volts_per_count] * 2,
                          [lay_uni.volt_offset] * 2)
    check("a unipolar range converts bottom-of-scale to 0 V, not -2.5 V",
          abs(float(volts[0, 0, 0])) < 1e-9, f"got {volts[0,0,0]:+.3f} V")
    rows = ocal.channel_health([counts, counts],
                               [lay_uni.volts_per_count] * 2,
                               ["a", "b"], [lay_uni.volt_offset] * 2)
    check("and channel_health does not call all 64 channels out of range",
          not any(r["out_of_range"] for r in rows))

    # A wrapping uint32 sample counter is a step of 1, not a 4-billion gap.
    wrap = np.arange(2 ** 32 - 3, 2 ** 32 + 3, dtype=np.int64).astype(np.uint32)
    check("the sample counter wrap is not reported as lost samples",
          ob.check_continuity(wrap) == (0, 0), str(ob.check_continuity(wrap)))
    gap = np.array([10, 11, 20, 21], dtype=np.uint32)
    check("a real gap is still counted", ob.check_continuity(gap) == (1, 8),
          str(ob.check_continuity(gap)))

    # An absent SPAD temperature block is unknown, not -160 degC.
    t = ocal.temperatures_c([np.zeros(8, dtype=np.uint32)] * 2)
    check("a box with no SPAD temperature reports NaN, not -160 C",
          np.isnan(t).all() and t.size == 16)


def test_config_loading():
    """A config that will not parse must be reported, never swallowed."""
    print("\nconfig loading")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "calibration.json")

        # An unknown key is ignored, not fatal: adding a comment field by hand
        # used to drop the whole probe back to built-in +/-20 mT defaults.
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"version": 2, "ranges_mt": [40.0] * 16,
                       "operator": "gh", "site": "bench"}, f)
        cal = ocal.Calibration.load(p)
        check("an unrecognised key does not stop a calibration loading",
              np.allclose(cal.ranges_mt, 40.0))

        # Genuinely broken, though, must be reported rather than passed over.
        with open(p, "w", encoding="utf-8") as f:
            f.write("{not json at all")
        msgs = []
        cal = ocal.Calibration.load_or_default(p, on_error=msgs.append)
        check("a corrupt calibration falls back AND says so",
              len(msgs) == 1 and np.allclose(cal.ranges_mt, 20.0),
              msgs[0][:60] if msgs else "no message")

        g = os.path.join(d, "probe_geometry.json")
        with open(g, "w", encoding="utf-8") as f:
            f.write("[]")
        msgs = []
        pgeom.Geometry.load_or_default(g, on_error=msgs.append)
        check("a corrupt geometry falls back AND says so", len(msgs) == 1)

        msgs = []
        ocal.Calibration.load_or_default(os.path.join(d, "nope.json"),
                                         on_error=msgs.append)
        check("a merely absent file is not an error", not msgs)


def test_raw_survives_a_kill():
    """A recorder that never gets close()d must still leave a readable file."""
    print("\nraw archive durability")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "raw.bin")
        rec = orec.RawRecorder(p, ["a", "b"], [3.0517578125e-4] * 2,
                               [200000.0] * 2)
        check("the sidecar is written at open, not at close",
              os.path.exists(p + ".json"))
        rec.write([np.full((32, 32), 7, dtype=np.int16)] * 2)
        # Deliberately no close(): this is the overnight-log-gets-killed case.
        x, meta = orec.load_raw(p)
        check("an unclosed recording still reads back",
              x.shape == (32, 64) and int(x[0, 0]) == 7, str(x.shape))
        check("and the sidecar already knows the channel layout",
              meta["n_channels"] == 64
              and len(meta["channel_names"]) == 64)
        rec.close()
        x2, meta2 = orec.load_raw(p)
        check("closing adds the final sample count",
              meta2["n_samples"] == 32 and x2.shape == x.shape)


def test_csv_quoting():
    """Free text in a table must not split the row."""
    print("\nCSV quoting")
    with tempfile.TemporaryDirectory() as d:
        # health_verdict() really does produce notes like this one.
        table = [{"sensor": "S1", "state": "fault",
                  "note": "Bx railed, By stuck"},
                 {"sensor": "S2", "state": "noisy",
                  "note": 'VCM noise 9.0 counts -- pickup, not the sensor'},
                 {"sensor": "S3", "state": "ok", "note": None}]
        p = orec.write_sensor_csv(os.path.join(d, "s.csv"), table)
        with open(p, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        check("every row has exactly as many fields as the header",
              all(len(r) == len(rows[0]) for r in rows),
              f"header {len(rows[0])}, rows {[len(r) for r in rows[1:]]}")
        check("the comma survives inside the field, not as a new column",
              rows[1][2] == "Bx railed, By stuck", rows[1][2])
        check("a None renders as empty rather than the word None",
              rows[3][2] == "")


def test_gain_tables():
    """The on-box scripts duplicate octobee's gain tables. Catch any drift."""
    print("\ngain table consistency")
    here = os.path.dirname(os.path.abspath(__file__))
    want_range = {g: r for g, (r, _) in ob.GAIN_TO_RANGE.items()}
    want_vpt = {g: v for g, (_, v) in ob.GAIN_TO_RANGE.items()}
    for name, expect in (("onbox/gain_config.py",
                          {"GAIN_RANGE_MT": want_range, "GAIN_VPT": want_vpt}),
                         ("onbox/sensor_audit.py",
                          {"GAIN_RANGE_MT": want_range})):
        src = open(os.path.join(here, name), encoding="utf-8").read()
        ns = {}
        for line in src.splitlines():
            for key in expect:
                if line.startswith(f"{key} = "):
                    ns[key] = ast.literal_eval(line.split("=", 1)[1].split("#")[0])
        for key, want in expect.items():
            got = ns.get(key)
            check(f"{name}: {key} matches octobee.GAIN_TO_RANGE",
                  got is not None and {int(k): float(v) for k, v in got.items()}
                  == {int(k): float(v) for k, v in want.items()},
                  f"{got} vs {want}")


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
    shipped = paths.config(ocal.CONFIG_NAME)
    if not os.path.exists(shipped):
        check(f"{shipped} is present", False)
        return
    cal = ocal.Calibration.load(shipped)
    # Re-read live with: ssh root@acq1001_69x 'python3 /tmp/gain_config.py show'
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

# Run in a CHILD process: half of what is under test is what happens when the
# process dies, and proving that in-process would take the test runner with it.
_CRASH_PROBE = """
import faulthandler, os, sys
from PyQt6 import QtCore, QtWidgets
from octobee.gui import window as gui
from octobee.gui.crash import CrashHandler

app = QtWidgets.QApplication([])
if {install!r}:
    handler = CrashHandler(path={log!r}).install()
    faulthandler.enable(file=open({log!r}, "a", buffering=1, encoding="utf-8"))

def boom():
    if {native!r}:
        os.abort()                 # stands in for Qt's qFatal
    raise RuntimeError("deliberate fault in a slot")

QtCore.QTimer.singleShot(20, boom)
QtCore.QTimer.singleShot(800, app.quit)
app.exec()
print("SURVIVED")
"""


def test_crash_handler(workdir):
    """Make the program's failures leave a record. Two kinds, two mechanisms.

    A Python exception in a slot does NOT kill this program: pyqtgraph replaces
    sys.excepthook on import specifically to stop PyQt aborting. It prints the
    traceback to stderr instead -- and the desktop icon runs pythonw.exe, where
    stderr goes nowhere. So the failure is survivable and completely invisible,
    which is how a window ends up running in a state nobody knows about.

    A NATIVE fatal error -- an access violation, or Qt calling qFatal() -- is
    the opposite: it kills the process at once and never becomes a Python
    exception at all. All it leaves is a Windows event log line naming
    Qt6Core.dll, which says nothing about which line of Python was running.
    faulthandler is what turns that into a stack trace.
    """
    print("\ncrash handler")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")

    def run(install, log, native=False):
        src = _CRASH_PROBE.format(install=install, log=log,
                                  native=native)
        # check=False on purpose: a non-zero exit IS one of the results.
        return subprocess.run([sys.executable, "-c", src], env=env, check=False,
                              capture_output=True, text=True, timeout=180)

    # ---- a Python exception in a slot ----
    bare_log = os.path.join(workdir, "bare.log")
    bare = run(False, bare_log)
    check("a slot exception does not kill the program (pyqtgraph sees to that)",
          bare.returncode == 0 and "SURVIVED" in bare.stdout,
          f"exit {bare.returncode}")
    check("but nothing is written down, which is the actual problem",
          not os.path.exists(bare_log),
          "under pythonw.exe the traceback goes to a stderr that does not exist")

    log = os.path.join(workdir, "crash_probe.log")
    caught = run(True, log)
    check("with the handler, the program still survives",
          caught.returncode == 0 and "SURVIVED" in caught.stdout,
          f"exit {caught.returncode}")
    check("and now the traceback is on disk",
          os.path.exists(log) and "deliberate fault in a slot" in
          open(log, encoding="utf-8").read())
    check("the log names the line that raised, not just the message",
          "boom" in open(log, encoding="utf-8").read())

    # ---- a native abort, which Python never sees ----
    native_log = os.path.join(workdir, "native.log")
    native = run(True, native_log, native=True)
    check("a native abort does kill the program",
          native.returncode != 0 and "SURVIVED" not in native.stdout,
          f"exit {native.returncode} -- nothing can catch this one")
    dumped = open(native_log, encoding="utf-8").read() if os.path.exists(
        native_log) else ""
    check("faulthandler still leaves a Python stack for it",
          "Fatal Python error" in dumped and "boom" in dumped,
          "which is the difference between 'Qt6Core.dll 0xc0000409' and a line "
          "number")

    # Repeated faults must not queue a dialog each: a fault inside a repainting
    # timer fires several times a second.
    #
    # Counted by looking at the box the handler keeps, NOT by monkeypatching
    # QMessageBox. Two things about that are worth writing down, because both
    # cost an afternoon here:
    #
    #   * the handler must not call QMessageBox.critical(), which blocks on a
    #     click that never comes -- from inside an excepthook that freezes the
    #     program, which is the failure the reporter exists to replace;
    #   * patching QMessageBox.show() to count calls ABORTS the interpreter
    #     (exit 127). It is a C++ virtual, and replacing it out from under Qt
    #     is not a thing sip will survive. Assert on the handler's own state.
    #
    # A parent window is supplied because that is the only state the running
    # program is ever in; with none, the handler logs and shows nothing rather
    # than parenting a box to nowhere.
    parent = QtWidgets.QMainWindow()
    handler = CrashHandler(path=os.path.join(workdir, "twice.log"),
                               window=parent)
    boxes = []
    t0 = time.time()
    for _ in range(3):
        try:
            raise ValueError("again")
        except ValueError:
            handler(*sys.exc_info())
        boxes.append(getattr(handler, "_box", None))
    elapsed = time.time() - t0
    check("reporting never blocks the thread it interrupted", elapsed < 5.0,
          f"{elapsed:.3f} s for three faults -- the blocking dialog never "
          f"returned at all")
    check("one dialog is raised, and the later faults reuse it",
          boxes[0] is not None and boxes[1] is boxes[0]
          and boxes[2] is boxes[0],
          "three boxes for three faults reads as the freeze it replaces")
    parent.close()
    check("repeated faults are all logged", handler.count == 3,
          f"count {handler.count}")
    check("every one of them reaches the log",
          open(handler.path, encoding="utf-8").read().count("again") >= 3)


def test_gui_estop(app, workdir):
    """The stop button: reachable, latching, and it releases nothing by itself.

    Every check here is a property that was missing before, and each one has a
    failure that ends with the head somewhere it should not be:

      the button lives outside the tab stack -- the old STOP was on the Stages
        tab, so during a scan watched from the Live tab it was on a page you
        could not see
      it is enabled with no stages connected -- "not connected" is this
        process's belief, and if that belief were reliable there would be
        nothing to stop
      it latches, and pressing it again does not release -- a person reaching
        for a stop in a hurry may well hit it twice
      the latch survives reconnecting -- otherwise Disconnect/Connect is an
        undocumented way to release a stopped machine
    """
    print("\nemergency stop")

    class FakeStage:
        def __init__(self, name):
            self.name, self.serial, self.model = name, "45000000", "LTS300C"
            self.homed, self.invert, self.is_open = True, False, True
            self.travel_mm = self.limit_mm = (0.0, 300.0)
            self.limit_declared = True
            # An envelope stated is an envelope declared: this double has one,
            # so it must answer the question the window asks at connect.
            self.limit_declared = True
            self.stopped = 0

        @property
        def position_trusted(self):
            return not self.stopped

        @property
        def distrust_reason(self):
            return None if self.position_trusted else "stopped immediately"

        @property
        def position_mm(self):
            return 10.0

        @property
        def vel_params(self):
            return (6.0, 10.0)

    class FakeSet:
        def __init__(self, axes):
            self.axes, self.names = axes, list(axes)
            self.interlock = ostage.MotionInterlock()

        def __getitem__(self, k):
            return self.axes[k]

        def __iter__(self):
            return iter(self.axes.values())

        def emergency_stop(self, reason="emergency stop"):
            for st in self.axes.values():
                st.stopped += 1
            self.interlock.trip(reason)
            return []

        def untrusted(self):
            return [(n, st.distrust_reason) for n, st in self.axes.items()
                    if not st.position_trusted]

        def reset_interlock(self):
            return (self.interlock.reset(), self.untrusted())

        def home_sequence(self):
            return list(self.axes)

        def close(self):
            pass

    ns = argparse.Namespace(
        uut=None, demo=True, replay=None, no_connect=True,
        geometry=os.path.join(workdir, "estop_geom.json"),
        calibration=os.path.join(workdir, "estop_cal.json"),
        machine=os.path.join(workdir, "estop_machine.json"),
        out_dir=os.path.join(workdir, "estopcaps"),
        screenshot=None, screenshot_tab=0, screenshot_warmup=0)
    win = gui.MainWindow(ns)
    try:
        # Reachable from anywhere: it is a toolbar widget, not a tab child.
        tb = win.findChild(QtWidgets.QToolBar)
        widgets = [tb.widgetForAction(a) for a in tb.actions()]
        check("the stop button is on the toolbar, not inside a tab",
              tb.isAncestorOf(win.btn_estop),
              "a stop you have to navigate to is not a stop")
        check("it is the last thing on the toolbar",
              widgets[-2:] == [win.btn_estop, win.btn_estop_reset],
              str([type(w).__name__ for w in widgets[-3:]]))
        check("an expanding spacer pins it to the top right at any width",
              widgets[-3] is not None
              and widgets[-3].sizePolicy().horizontalPolicy()
              == QtWidgets.QSizePolicy.Policy.Expanding)
        check("Escape is bound application-wide",
              win.sc_estop.context()
              == QtCore.Qt.ShortcutContext.ApplicationShortcut,
              "otherwise it stops working the moment the wizard has focus")

        # With nothing connected it must still work, and still latch.
        check("the stop button is live with no stages connected",
              win.session.stages is None and win.btn_estop.isEnabled())
        win.on_estop()
        check("stopping with nothing connected still latches",
              win.motion.reason is not None)
        # isHidden, not isVisible: the window itself is never shown in this
        # test, so isVisible() is False for every child either way and both
        # checks would pass without testing anything.
        check("and the reset button appears only once latched",
              not win.btn_estop_reset.isHidden())
        win.motion._reason = None           # reset without opening the modal
        win._refresh_estop_ui()
        check("clearing the latch hides the reset button",
              win.btn_estop_reset.isHidden())

        stages = FakeSet({"x": FakeStage("x"), "y": FakeStage("y")})
        win.session.stages = stages
        for ax in ("x", "y"):
            win.stage_rows[ax]["present"] = True
        win._sync_stage_controls()
        check("motion controls are live before the stop",
              win.stage_rows["x"]["target"].isEnabled()
              and win.btn_scan_start.isEnabled())

        win.on_estop()
        check("the stop reaches every axis",
              all(st.stopped == 1 for st in stages),
              "one axis left running is the whole failure this prevents")
        check("and latches the machine, not just the axis that was moving",
              stages.interlock.tripped is not None)
        check("motion controls go dead while latched",
              not win.stage_rows["x"]["target"].isEnabled()
              and not win.btn_scan_start.isEnabled())
        try:
            stages.interlock.require_clear("a move")
            ok = False
        except ostage.MotionInterlocked:
            ok = True
        check("and the interlock refuses moves at the point of command", ok,
              "the button cannot reach a thread already past its own check")

        first = win.motion.reason
        win.on_estop()
        check("pressing stop twice does not release the machine",
              win.motion.reason == first and stages.interlock.tripped is not None,
              "a person in a hurry hits it twice")

        # A stopped axis must not accept an absolute move again just because
        # the latch was cleared: an immediate stop can have lost steps.
        check("a stopped axis is no longer trusted",
              not stages["x"].position_trusted)
        was, lost = stages.reset_interlock()
        win.motion._reason = None
        check("resetting the latch clears it", was is not None
              and stages.interlock.tripped is None)
        check("but the axes stay untrusted until they are homed",
              [n for n, _ in lost] == ["x", "y"],
              "clearing a stop is not the same as knowing where the head is")

        # Disconnect/reconnect must not be a back door round a latched stop.
        win.on_estop()
        win.session.stages = None
        win._stage_pending = FakeSet({"x": FakeStage("x")})
        win.on_stage_action_done("connect", "")
        app.processEvents()
        check("reconnecting does not release a latched stop",
              win.session.stages.interlock.tripped is not None,
              "the latch belongs to the rig, not to a StageSet object")
    finally:
        # Teardown, not an API: the latch is deliberately read-only in
        # production and only moves through trigger()/reset().
        win.motion._reason = None
        win.session.stages = None
        win.close()
        app.processEvents()


def test_magnet_wizard_reopens(app, workdir):
    """Closing the wizard must let it be opened again.

    The bug this pins: MagnetWizard.closeEvent() accepted the event without
    calling super(), so QDialog never called reject(), never emitted
    finished(), and the main window kept a reference to a hidden dialog for
    ever. The button then "worked" -- it raised the dead one -- and did nothing
    visible, which from a bug report is indistinguishable from the whole
    program having frozen.
    """
    print("\nguided magnet wizard, reopened")

    class FakeStage:
        def __init__(self, name):
            self.name, self.serial, self.model = name, "45000000", "LTS300C"
            self.homed, self.invert = True, False
            self.travel_mm = (0.0, 300.0)
            self.limit_mm = (0.0, 300.0)
            self.limit_declared = True
            self.limit_declared = True
            self.position_trusted, self.distrust_reason = True, None

        @property
        def position_mm(self):
            return 10.0

        @property
        def vel_params(self):
            return (6.0, 10.0)

    class FakeSet:
        def __init__(self, axes):
            self.axes, self.names = axes, list(axes)
            self.interlock = ostage.MotionInterlock()

        def __getitem__(self, k):
            return self.axes[k]

        def __iter__(self):
            return iter(self.axes.values())

        def home_sequence(self):
            return list(self.axes)

        def untrusted(self):
            return [(n, st.distrust_reason) for n, st in self.axes.items()
                    if not st.position_trusted]

        def close(self):
            pass

    ns = argparse.Namespace(
        uut=None, demo=True, replay=None, no_connect=True,
        geometry=os.path.join(workdir, "reopen_geom.json"),
        calibration=os.path.join(workdir, "reopen_cal.json"),
        machine=os.path.join(workdir, "reopen_machine.json"),
        out_dir=os.path.join(workdir, "reopencaps"),
        screenshot=None, screenshot_tab=0, screenshot_warmup=0)
    win = gui.MainWindow(ns)
    win.session.stages = FakeSet({"x": FakeStage("x"), "y": FakeStage("y")})
    try:
        for attempt in (1, 2, 3):
            win.on_guided_magnet()
            app.processEvents()
            check(f"the wizard opens (attempt {attempt})",
                  win._magnet_wizard is not None
                  and win._magnet_wizard.isVisible())
            win._magnet_wizard.close()
            app.processEvents()
            check(f"and closing it lets go of the handle (attempt {attempt})",
                  win._magnet_wizard is None,
                  "a stuck handle makes the button silently do nothing")

        # Closing must not leave anything modal behind either -- that would
        # block every other window in the program, not just this one.
        check("nothing modal is left blocking the rest of the window",
              QtWidgets.QApplication.activeModalWidget() is None)
        probe = QtWidgets.QDialog(win)
        probe.show()
        app.processEvents()
        check("other windows still open afterwards", probe.isVisible())
        probe.close()

        # An unfinished run is only in the dialog, so closing has to ask.
        win.on_guided_magnet()
        app.processEvents()
        wiz = win._magnet_wizard
        g = win.session.geom
        for sweep in _synth_magnet_run(
                g, np.ones(16),
                np.array([0.0, 200.0, g.fsv_radius_mm + 25.0])).sweeps[:2]:
            wiz.run.add(sweep)
        asked = []
        real_q = QtWidgets.QMessageBox.question
        QtWidgets.QMessageBox.question = staticmethod(
            lambda *a, **k: asked.append(a[2])
            or QtWidgets.QMessageBox.StandardButton.Cancel)
        try:
            wiz.close()
            app.processEvents()
            check("closing on unsaved poses asks before discarding them",
                  bool(asked) and "not saved" in asked[0])
            check("and cancelling keeps the window open", wiz.isVisible())
        finally:
            QtWidgets.QMessageBox.question = real_q
        wiz._finished = True          # pretend it was applied, so it can go
        wiz.close()
        app.processEvents()
    finally:
        win.close()


def test_live_plot_reset(app, workdir):
    """Reset view has to leave the plot AUTO-ranging, not merely re-fitted.

    The trap: ViewBox.autoRange() fits the data and switches auto-ranging off
    as it does it, so the obvious implementation looks right for one frame and
    then freezes. The assertion is therefore on the ViewBox's auto-range state,
    not on the axis limits.
    """
    print("\nlive plot reset view")
    plot = gui.LivePlot(pgeom.Geometry())
    vb = plot.plot.getViewBox()

    rng = np.random.default_rng(4)
    plot.update_data(rng.normal(0, 1, (400, 16, 3)).astype(np.float32), 500.0)
    app.processEvents()
    check("a fresh plot auto-ranges", all(vb.state["autoRange"]),
          str(vb.state["autoRange"]))

    # What a drag and a scroll do, which is what the button has to undo.
    vb.setRange(xRange=(-0.5, -0.4), yRange=(100.0, 200.0))
    vb.setMouseEnabled(x=False, y=True)
    check("zooming turns auto-ranging off, silently",
          not any(vb.state["autoRange"]), str(vb.state["autoRange"]))

    plot.reset_view()
    app.processEvents()
    check("reset view puts BOTH axes back on auto", all(vb.state["autoRange"]),
          str(vb.state["autoRange"]))
    check("and re-enables the mouse on both axes",
          all(vb.state["mouseEnabled"]), str(vb.state["mouseEnabled"]))

    # The flag being set is not the same as the flag WORKING: a view left
    # disabled ignores updateAutoRange() entirely, so recomputing and seeing
    # the range follow ten-times-larger data is what proves the button did
    # something. The explicit call stands in for the repaint that the live
    # window does ten times a second and that a never-shown widget never gets.
    before = vb.viewRange()[1][1]
    plot.update_data(rng.normal(0, 40, (400, 16, 3)).astype(np.float32), 500.0)
    vb.updateAutoRange()
    app.processEvents()
    check("the view follows the data afterwards, not frozen where it was",
          vb.viewRange()[1][1] > before * 2,
          f"y top {before:.2f} -> {vb.viewRange()[1][1]:.2f}")
    plot.deleteLater()


def test_magnet_wizard_saves(app, workdir):
    """Finishing the wizard must leave the trim ON DISK, not just in memory.

    The bug this pins: the wizard applied the gain trim to the running
    calibration and never wrote calibration.json, so closing the window threw
    away a twenty-minute measurement without saying anything. Checking the
    in-memory object would have passed happily -- the assertion has to be
    against the file.
    """
    print("\nguided magnet wizard")
    cal_path = os.path.join(workdir, "wizard_cal.json")
    geom_path = os.path.join(workdir, "wizard_geom.json")
    pgeom.Geometry().save(geom_path)
    ocal.Calibration().save(cal_path)

    ns = argparse.Namespace(
        uut=None, demo=True, replay=None, geometry=geom_path,
        calibration=cal_path, machine=os.path.join(workdir, "wiz_machine.json"),
        out_dir=os.path.join(workdir, "wizcaps"),
        screenshot=None, screenshot_tab=0, screenshot_warmup=0,
        no_connect=True)
    win = gui.MainWindow(ns)

    # Every modal in finish() answered without clicking: warning() returns, and
    # exec() leaving clickedButton() as None is "leave the file alone".
    real_warn, real_exec = QtWidgets.QMessageBox.warning, QtWidgets.QMessageBox.exec
    QtWidgets.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QtWidgets.QMessageBox.exec = lambda self: 0
    try:
        wiz = gui.MagnetWizard(win)
        g = win.session.geom
        rng = np.random.default_rng(19)
        gain = rng.normal(1.0, 0.07, 16)
        magnet = np.array([0.0, 200.0, g.fsv_radius_mm + 25.0])
        for sweep in _synth_magnet_run(g, gain, magnet).sweeps:
            wiz.run.add(sweep)

        before = ocal.Calibration.load(cal_path)
        check("the calibration on disk starts untrimmed",
              np.allclose(before.gain_corr, 1.0))

        wiz.chk_apply.setChecked(True)
        wiz.finish()

        after = ocal.Calibration.load(cal_path)
        check("finishing the wizard writes the trim to calibration.json",
              not np.allclose(after.gain_corr, 1.0),
              f"trim now spans {after.gain_corr.min():.3f}.."
              f"{after.gain_corr.max():.3f} in the FILE")
        check("and the trim it wrote is the one the run measured",
              np.allclose(after.gain_corr[:, 0], win.session.cal.gain_corr[:, 0]))
        check("the file records where the trim came from",
              "guided magnet run" in (after.notes or ""), after.notes or "")
        check("and the run itself is on disk beside it",
              any(f.startswith("magcal_") and f.endswith(".npz")
                  for f in os.listdir(ns.out_dir)),
              str(os.listdir(ns.out_dir)))

        # Unticked, it must leave the file alone -- and say the run is safe.
        ocal.Calibration().save(cal_path)
        win.session.cal = ocal.Calibration.load(cal_path)
        wiz2 = gui.MagnetWizard(win)
        for sweep in _synth_magnet_run(g, gain, magnet).sweeps:
            wiz2.run.add(sweep)
        wiz2.chk_apply.setChecked(False)
        wiz2.finish()
        check("unticked, it does not touch the calibration file",
              np.allclose(ocal.Calibration.load(cal_path).gain_corr, 1.0))
        check("and it says the measurement is still recoverable",
              "nothing about it is lost" in win.log_pane.toPlainText())
        wiz.close()
        wiz2.close()
    finally:
        QtWidgets.QMessageBox.warning = real_warn
        QtWidgets.QMessageBox.exec = real_exec
        win.close()


def test_autoconnect(app, workdir):
    """The window connects itself when it opens -- and shuts up when it can't.

    Hermetic: ConnectWorker and connect_stages are both stubbed, so this never
    touches a carrier or a USB device. What is under test is the wiring --
    whether an attempt is made, and how a failure is reported -- not the
    connecting itself, which test_app covers against real hardware.
    """
    print("\nautomatic connect")

    attempts = []
    modals = []

    class FakeConnectWorker(QtCore.QThread):
        done = QtCore.pyqtSignal(object, object, str)
        progress = QtCore.pyqtSignal(str)

        def __init__(self, hosts, fs):
            super().__init__()
            attempts.append(hosts)

        def start(self):
            # Deferred, not immediate: a real connect takes seconds, and the
            # guard under test only matters while one is in flight. A stub
            # that finishes inside start() would report every guard as broken.
            QtCore.QTimer.singleShot(
                150, lambda: self.done.emit(None, None,
                                            "no route to host (stubbed)"))

    real_worker = gui.ConnectWorker
    real_critical = QtWidgets.QMessageBox.critical
    gui.ConnectWorker = FakeConnectWorker
    QtWidgets.QMessageBox.critical = staticmethod(
        lambda *a, **k: modals.append(a))

    def build(**over):
        ns = argparse.Namespace(
            uut=None, demo=False, replay=None,
            geometry=os.path.join(workdir, "probe_geometry.json"),
            calibration=os.path.join(workdir, "calibration.json"),
            machine=os.path.join(workdir, "machine.json"),
            out_dir=os.path.join(workdir, "captures"),
            screenshot=None, screenshot_tab=0, screenshot_warmup=0,
            no_connect=False)
        for k, v in over.items():
            setattr(ns, k, v)
        w = gui.MainWindow(ns)
        # Before any event is processed, so the deferred connect cannot reach
        # the real stages when it fires.
        w.connect_stages = lambda quiet=False: False
        return w

    def pump(seconds=1.0):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.02)

    try:
        win = build()
        pump()
        check("opening the window starts a connect on its own",
              len(attempts) == 1, f"{len(attempts)} attempt(s)")
        check("a failed automatic connect logs instead of raising a dialog",
              not modals and "connect failed" in win.log_pane.toPlainText(),
              f"{len(modals)} modal(s)")
        check("and it says the rest of the window still works",
              "everything that does not need the carriers" in
              win.log_pane.toPlainText())
        check("Connect is left enabled to try again",
              win.act_connect.isEnabled())
        win.close()

        # A second attempt while one is in flight would open a second stream
        # against the same carriers. --no-connect here so the only attempts
        # counted are the ones this test makes.
        attempts.clear()
        win = build(no_connect=True)
        for _ in range(3):
            win.on_connect()
        check("overlapping connects start exactly one worker",
              len(attempts) == 1, f"{len(attempts)} worker(s) for 3 calls")
        pump(0.4)
        check("and the next attempt is allowed once that one has finished",
              (win.on_connect(), len(attempts))[1] == 2,
              f"{len(attempts)} worker(s) after the first failed")
        pump(0.3)
        win.close()

        attempts.clear()
        win = build(no_connect=True)
        pump(0.6)
        check("--no-connect starts disconnected", not attempts
              and "press Connect" in win.log_pane.toPlainText())
        win.close()

        attempts.clear()
        win = build(demo=True)
        pump(0.6)
        check("a demo window does not go looking for carriers",
              not attempts and win.session.source is not None)
        win.close()
    finally:
        gui.ConnectWorker = real_worker
        QtWidgets.QMessageBox.critical = real_critical


def test_app(app, args, workdir):
    kind = ("live hardware" if args.live else
            f"replay {args.replay}" if args.replay else "synthetic probe")
    print(f"\napplication ({kind})")
    ns = argparse.Namespace(
        uut=None, demo=not (args.replay or args.live), replay=args.replay,
        geometry=os.path.join(workdir, "probe_geometry.json"),
        calibration=os.path.join(workdir, "calibration.json"),
        machine=os.path.join(workdir, "machine.json"),
        # Into the temp dir, not captures/. A test run used to leave ~9 MB of
        # synthetic recordings, reports and health CSVs in the real capture
        # directory, named identically to the bench data beside them and
        # indistinguishable from it afterwards.
        out_dir=os.path.join(workdir, "captures"),
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
        while win.session.source is None and time.time() < deadline:
            app.processEvents()
            time.sleep(0.05)
    check("source started", win.session.source is not None)
    if win.session.source is None:
        return win

    pump(win, app, 3.0)
    check("data is flowing", win.session.roll.filled > 100,
          f"{win.session.roll.filled} points buffered")
    check("sensor table populated", win._last_table is not None
          and len(win._last_table) == 16)
    check("channel health computed", win.session.last_health is not None
          and len(win.session.last_health) == 64)
    check("S16 detected as dead", "S16" in win.session.cal.dead,
          f"excluded: {sorted(win.session.cal.dead)}")

    # ---- tare ----
    v = win.session.roll.view()
    before = np.abs(np.median(v, axis=0)).max() if v.shape[0] else float("nan")
    win.start_collect("tare", 0.5)
    pump(win, app, 2.5)
    check("tare completed", win.session.collecting is None)
    check("tare stored a non-trivial zero", np.any(win.session.cal.zero_mt != 0),
          f"max |zero| {np.abs(win.session.cal.zero_mt).max():.4f} mT (was reading "
          f"{before:.4f} mT)")
    pump(win, app, 1.0)

    # zero_mt is defined BEFORE the gain trim, so the zero a tare stores must
    # not depend on what trim happens to be loaded. It used to: _finish_tare
    # reconstructed the uncorrected field as `data + zero_mt`, which only
    # inverts to_mt() when the trim is 1.0 and the matrix is identity, so
    # taring after a magnet pass or a roll solve stored a zero scaled by the
    # trim -- invisibly, because a uniformly wrong zero looks like a zero.
    #
    # Driven directly rather than through the live source: the demo probe's
    # magnet keeps moving, so two tares taken seconds apart legitimately
    # differ and could not tell "scaled by the trim" from "the field moved".
    # The same fixed block through both trims can.
    zeros_at = {}
    fixed = np.full((64, pgeom.N_SENSORS, 4), 2.2)         # volts, all at VCM
    fixed[:, :, 0] += 0.063                                # Bx = VCM + 63 mV
    for trim in (1.0, 2.0):
        win.session.cal.clear_tare()
        win.session.cal.clear_matrix()
        win.session.cal.gain_corr = np.full((pgeom.N_SENSORS, 3), trim)
        win.session.collecting = {"what": "tare", "blocks": [], "n": 0, "need": 1,
                          "peak": None, "baseline": None, "tag": None,
                          "decim": 1}
        win._collect_block(win.session.cal.to_mt(fixed), fixed)
        check(f"tare at trim {trim:g} completed", win.session.collecting is None)
        zeros_at[trim] = win.session.cal.zero_mt.copy()
    d = float(np.abs(zeros_at[2.0] - zeros_at[1.0]).max())
    check("the stored zero does not scale with the gain trim", d < 1e-9,
          f"trim 1.0 -> {zeros_at[1.0][0]}, trim 2.0 -> {zeros_at[2.0][0]}")
    check("and it is the uncorrected field, not the corrected one",
          abs(zeros_at[1.0][0, 0] - 1.0) < 0.01,
          f"63 mV at 63 V/T should tare at 1 mT, got {zeros_at[1.0][0, 0]:.4f}")
    win.session.cal.clear_gain()
    win.session.cal.clear_tare()
    pump(win, app, 1.0)

    # ---- magnet pass ----
    win.btn_magnet.setChecked(True)
    pump(win, app, 4.0)
    win.btn_magnet.setChecked(False)
    app.processEvents()
    check("magnet pass captured peaks", win.session.magnet_peaks is not None)
    if win.session.magnet_peaks is not None:
        live = win.session.cal.live_mask()
        resp = int((win.session.magnet_peaks[live] > 1e-4).sum())
        check("most live sensors responded", resp >= 10, f"{resp}/15 responded")
        rep = ocal.spread_report(win.session.magnet_peaks, live=live)
        check("spread report produced", "raw_spread" in rep,
              f"spread {rep.get('raw_spread', float('nan')):.2f}x")

        # ---- gain trim ----
        before_spread = rep.get("raw_spread")
        win.on_apply_gain()
        trimmed = win.session.magnet_peaks * win.session.cal.gain_corr[:, 0]
        after = trimmed[live].max() / trimmed[live].min()
        check("gain trim narrows the spread", after < 1.0001,
              f"{before_spread:.2f}x -> {after:.4f}x")
        win.on_clear_gain()
        check("gain trim clears", np.allclose(win.session.cal.gain_corr, 1.0))

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

    win.session.cal.gain_corr = np.full((pgeom.N_SENSORS, 3), 2.0)
    for tag in ("A", "B", "C"):
        win.start_sweep(tag, 0.4)
        pump(win, app, 2.0)
    win.session.cal.clear_gain()
    check("all three sweeps recorded", set(win.session.sweeps) == {"A", "B", "C"},
          win.lbl_sweeps.text())
    check("solve enabled once sweeps exist", win.btn_solve_roll.isEnabled())

    if set(win.session.sweeps) == {"A", "B", "C"}:
        # A sweep must be independent of whatever trim was loaded when it was
        # taken -- that is why it is captured pre-correction. The x2 gain above
        # was live during capture and must not show up in the stored data.
        sw = win.session.sweeps["A"]
        check("sweeps are stored uncorrected",
              np.abs(sw.b_mt).max() < 1e4 and np.isfinite(sw.b_mt).all())
        check("sweeps record the range they were taken at",
              sw.ranges_mt is not None
              and np.allclose(sw.ranges_mt, win.session.cal.ranges_mt))

        win.on_solve_roll()
        check("roll solve produced a solution", win.session.pose_solution is not None,
              "" if win.session.pose_solution is not None
              else " ".join(win.cal_report.toPlainText().split())[:300])
        check("apply is enabled after a solve", win.btn_apply_roll.isEnabled())
        if win.session.pose_solution is not None:
            sol = win.session.pose_solution
            check("solve used all three orientations",
                  sorted(sol.tags) == ["A", "B", "C"], f"{sol.tags}")
            check("report reaches the calibration pane",
                  "gain spread" in win.cal_report.toPlainText())
            win.session.cal.apply_pose_solution(sol)
            check("applying installs the matrix and clears the trim",
                  win.session.cal.has_matrix and np.allclose(win.session.cal.gain_corr, 1.0))
            check("the applied calibration still converts",
                  np.isfinite(win.session.cal.to_mt(
                      np.full((4, pgeom.N_SENSORS, 4), 2.2))).all())
            win.session.cal.clear_matrix()
            win.session.cal.clear_tare()

        with tempfile.TemporaryDirectory() as d:
            ok = True
            for tag, sw in win.session.sweeps.items():
                back = opc.RollSweep.load(sw.save(os.path.join(d, f"rs_{tag}")))
                ok &= (back.tag == tag
                       and back.b_mt.shape == sw.b_mt.shape
                       and np.allclose(back.ranges_mt, sw.ranges_mt))
            check("sweeps round trip through disk", ok)

    win.on_clear_sweeps()
    check("clearing sweeps disables solve",
          not win.session.sweeps and not win.btn_solve_roll.isEnabled())

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
    before = win.session.roll.filled
    pump(win, app, 1.0)
    check("acquisition continues with the 3D head off", win.session.roll.filled >= before)
    win.chk_3d.setChecked(True)

    win.act_pause.setChecked(True)
    n_before = win.session.roll.filled
    pump(win, app, 1.0)
    check("pausing the view does not pause acquisition",
          win.session.roll.filled >= n_before and win.paused)
    win.act_pause.setChecked(False)

    # ---- recording ----
    win.tab_export.chk_csv.setChecked(True)
    win.tab_export.chk_raw.setChecked(True)
    win.tab_export.chk_tube.setChecked(False)
    win.act_record.setChecked(True)
    check("CSV recorder opened", win.session.csv_rec is not None)
    check("raw recorder opened", win.session.raw_rec is not None)
    csv_path = win.session.csv_rec.path if win.session.csv_rec else None
    raw_path = win.session.raw_rec.path if win.session.raw_rec else None
    pump(win, app, 3.0)
    rows_written = win.session.csv_rec.n_rows if win.session.csv_rec else 0
    win.act_record.setChecked(False)
    check("recording stopped cleanly", win.session.csv_rec is None and win.session.raw_rec is None)

    # ---- read the CSV back and check it ----
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            lines = f.readlines()
        header = [ln for ln in lines if ln.startswith("#")]
        cols = next(ln for ln in lines
                    if not ln.startswith("#")).strip().split(",")
        names, data = read_csv(csv_path)
        col = {n: i for i, n in enumerate(names)}
        check("CSV has a provenance header", len(header) >= 8,
              f"{len(header)} header lines")
        check("CSV has 1 + 16*4 columns", len(cols) == 65, f"{len(cols)} columns")
        check("CSV row count matches the recorder",
              data.shape[0] == rows_written, f"{data.shape[0]} rows")
        dt = np.diff(data[:, col["t_s"]])
        check("CSV timebase matches the output rate",
              abs(np.median(dt) - 1.0 / win.session.out_rate) < 1e-6,
              f"dt {np.median(dt)*1e3:.3f} ms at {win.session.out_rate:g} Hz")
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
        b = win.session.cal.convert(boxes, meta["volts_per_count"])
        check("raw file reconverts to finite field values",
              np.isfinite(b).all() and b.shape[1:] == (16, 3),
              f"shape {b.shape}")

    # ---- exports ----
    win.tab_export.export_summary()
    win.tab_health.export_csv()
    win.tab_export.export_json()
    win.tab_health.analyse()
    app.processEvents()
    txt = win.tab_health.health_text.toPlainText()
    check("diagnostics text produced", "per-sensor verdict" in txt,
          f"{len(txt)} chars")
    check("diagnostics names the VCM channels", "VCM" in txt)

    exports = [line for line in win.tab_export.export_log.toPlainText().splitlines()
               if line.strip()]
    check("three one-shot exports logged", len(exports) >= 3,
          f"{len(exports)} entries")
    for line in exports:
        p = line.split("] ", 1)[-1].split("  (")[0]
        if os.path.exists(p):
            check(f"export exists and is non-empty: {os.path.basename(p)}",
                  os.path.getsize(p) > 0, f"{os.path.getsize(p)} bytes")

    # ---- calibration round trip ----
    cal_path = os.path.join(workdir, "roundtrip.json")
    win.session.cal.zero_mt[0, 0] = 1.2345
    win.session.cal.ranges_mt[5] = 400.0
    win.session.cal.save(cal_path)
    reloaded = ocal.Calibration.load(cal_path)
    check("calibration survives a save/load round trip",
          np.allclose(reloaded.zero_mt, win.session.cal.zero_mt)
          and np.allclose(reloaded.ranges_mt, win.session.cal.ranges_mt)
          and reloaded.dead == win.session.cal.dead)

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
    check("still acquiring after a geometry change", win.session.roll.filled > 100)

    # ---- tube frame CSV ----
    win.tab_export.chk_tube.setChecked(True)
    win.tab_export.chk_raw.setChecked(False)
    win.act_record.setChecked(True)
    tube_path = win.session.csv_rec.path if win.session.csv_rec else None
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
        st = win.session.source.stats()
        check("no stream gaps on the live link", st.get("gaps", 0) == 0,
              f"gaps {st.get('gaps')}, lost {st.get('lost')}")
        # Measure over a window in which the loop is actually being pumped.
        # A count accumulated across the whole run says more about this
        # harness -- which deliberately blocks for most of a second at a time
        # parsing files back -- than about the application, whose real contract
        # is that a running session does not shed data.
        for stream in win.session.source.streamers:
            stream.dropped = 0
        pump(win, app, 3.0)
        dropped = sum(x.dropped for x in win.session.source.streamers)
        check("no blocks dropped while the session is running", dropped == 0,
              f"{dropped} dropped over 3 s of streaming")

        # The snapshot stops the stream, captures at the full 200 kSPS, and
        # hands the port back. It is the only path that takes over the
        # carriers' stream ownership, so it gets exercised for real.
        win.tab_export.spin_snap_s.setValue(1.0)
        win.on_snapshot()
        deadline = time.time() + 120
        while (win._snap_worker is not None and win._snap_worker.isRunning()
               and time.time() < deadline):
            app.processEvents()
            time.sleep(0.05)
        app.processEvents()
        snaps = [line for line in win.tab_export.export_log.toPlainText().splitlines()
                 if ".npz" in line]
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
    test_config_loading()
    test_csv_quoting()
    test_raw_survives_a_kill()
    test_gain_tables()
    test_cross_calibration()
    test_shipped_calibration()
    test_health()
    test_posecal()
    test_posecap()
    test_posecal_persistence()
    test_magnet_routine()
    test_magnet_plane()
    test_help_index()
    test_scan_grid()
    test_scan_guards()
    test_survey_pairing()
    test_stage_binding()
    test_stage_frame()
    test_stage_safety()
    with tempfile.TemporaryDirectory() as workdir:
        test_stage_home_order(workdir)
        test_fieldmap_roundtrip(workdir)
        test_scan_survives_failures(workdir)
        test_stage_axis_map(workdir)
        test_stage_motion(workdir)
        test_magnet_persistence(workdir)
        test_magnet_plane_persistence(workdir)
        test_stage_frame_persistence(workdir)
        test_machine_coils(workdir)
        test_machine_placement(workdir)
        test_crash_handler(workdir)
        test_gui_estop(app, workdir)
        test_magnet_wizard_reopens(app, workdir)
        test_live_plot_reset(app, workdir)
        test_magnet_wizard_saves(app, workdir)
        test_autoconnect(app, workdir)
        test_machine_tab(app, workdir)
        test_app(app, args, workdir)

    print(f"\n{CHECKS - len(FAILS)}/{CHECKS} checks passed")
    if SKIPS:
        print(f"{len(SKIPS)} skipped -- this run checked less than a full one:")
        for name in SKIPS:
            print(f"  - {name}")
    if FAILS:
        print("failed:")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
