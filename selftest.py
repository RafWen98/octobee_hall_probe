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
import tempfile
import time

import numpy as np
from PyQt6 import QtWidgets

import octobee as ob
import octobee_calibration as ocal
import octobee_gui as gui
import octobee_record as orec
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
    while time.time() < end:
        win.on_tick()
        app.processEvents()
        time.sleep(0.01)
    win.on_slow_tick()
    app.processEvents()


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
    # A chip reading purely on its own +Z must map to its face's outward normal.
    b = np.zeros((16, 3))
    b[:, 2] = 1.0
    check("chip +Z maps to the face normal",
          np.allclose(g.to_tube_frame(b), g.normals()))
    # |B| must survive the rotation, that being the whole point of using it.
    rng = np.random.default_rng(1)
    v = rng.normal(size=(50, 16, 3))
    check("|B| is invariant under the rotation",
          np.allclose(np.linalg.norm(v, axis=-1),
                      np.linalg.norm(g.to_tube_frame(v), axis=-1)))
    w = g.expected_response((60.0, 0.0, 100.0))
    check("geometry weighting favours the nearest chip",
          int(np.argmax(w)) + 1 == g.nearest_sensor((60.0, 0.0, 100.0)),
          f"nearest is S{g.nearest_sensor((60.0, 0.0, 100.0))}")


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
    w = g.expected_response((60.0, 0.0, 100.0))
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

    Getting this wrong is invisible -- every number on screen is simply scaled
    by 1.82 -- so it is worth asserting rather than trusting.
    """
    print("\nshipped calibration")
    if not os.path.exists(ocal.CONFIG_NAME):
        check("calibration.json is present", False)
        return
    cal = ocal.Calibration.load(ocal.CONFIG_NAME)
    check("S1-S8 are on the SPI-audited +/-40 mT range",
          all(cal.ranges_mt[i] == 40.0 for i in range(8)),
          f"{cal.ranges_mt[:8]}")
    check("S9-S16 are on the SPI-audited +/-20 mT range",
          all(cal.ranges_mt[i] == 20.0 for i in range(8, 16)),
          f"{cal.ranges_mt[8:]}")
    vpt = cal.volts_per_tesla
    check("the two halves differ by the 1.82x the audit predicts",
          abs(vpt[8] / vpt[0] - 1.818) < 0.01, f"{vpt[8]/vpt[0]:.3f}x")
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
        check("no blocks dropped by the GUI", st.get("dropped blocks", 0) == 0,
              f"{st.get('dropped blocks')} dropped")

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
    with tempfile.TemporaryDirectory() as workdir:
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
