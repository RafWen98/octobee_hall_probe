#!/usr/bin/env python3
"""
octobee_cal.py -- calibration / health report for the 16-sensor OCTO-BEE probe.

What it answers
---------------
  * Is every sensor alive, and what is its zero point (VCM)?
  * How much noise does each channel carry?
  * With a magnet passed by the probe: how big is each sensor's response, and
    how much do the sensors disagree with each other?

Why the VCM channel matters
---------------------------
Each SENIS chip outputs Bx/By/Bz as voltages referenced to its own virtual
ground VCM (~2.2 V), and it exports that VCM on the 4th channel of its group.
The VCMs are NOT identical between chips. Any comparison of "spike height"
that does not subtract each sensor's own VCM first is comparing apples to
oranges. This script always subtracts it.

What this script canNOT tell you
--------------------------------
Whether an amplitude difference is a *gain setting* difference or a *sensor
calibration* difference. Gain lives in the SENM3Dx G_CTRL registers, reachable
only over SPI on the carrier itself -- run onbox_sensor_audit.py for that.

Usage
-----
    python octobee_cal.py --seconds 3                 # live capture, then report
    python octobee_cal.py --load octobee_capture.npz  # report on a saved capture
    python octobee_cal.py --seconds 20 --plot         # magnet pass + peak analysis
"""

import argparse

import numpy as np

import octobee as ob


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    hosts = [str(h) for h in z["hosts"]]
    boxes = []
    for bi, h in enumerate(hosts):
        boxes.append({
            "host": h,
            "ai": z[f"ai_{bi}"],
            "temp_raw": z[f"temp_raw_{bi}"],
            "pwr_good": int(z[f"pwr_good_{bi}"]),
            "fs_hz": float(z[f"fs_hz_{bi}"]),
            "vpc": float(z[f"vpc_{bi}"]),
            "adc_range": str(z[f"adc_range_{bi}"]),
            "sam_cnt": z[f"sam_cnt_{bi}"],
        })
    return boxes


def live(hosts, seconds):
    res = ob.capture_all(hosts, seconds)
    boxes = []
    for h in hosts:
        d, lay = res[h]
        boxes.append({
            "host": h,
            "ai": d["ai"],
            "temp_raw": d.get("temp_raw", np.zeros(8)),
            "pwr_good": d.get("pwr_good", 0),
            "fs_hz": lay.fs_hz,
            "vpc": lay.volts_per_count,
            "adc_range": lay.adc_range,
            "sam_cnt": d["sam_cnt"],
        })
    return boxes


def vcm_subtract(ai):
    """
    (nsamples, 32) raw counts -> (nsamples, 24) field channels with each
    sensor's own VCM channel subtracted, plus the list of (sensor_idx, axis).
    """
    out, meta = [], []
    for s in range(ob.SENSORS_PER_BOX):
        vcm = ai[:, s * 4 + 3].astype(np.float64)
        for a, axis in enumerate(("Bz", "By", "Bx")):
            out.append(ai[:, s * 4 + a].astype(np.float64) - vcm)
            meta.append((s, axis))
    return np.column_stack(out), meta


def report(boxes, b_range_mt, do_plot, plot_path):
    v_per_t = ob.RANGE_TO_VPT[b_range_mt]

    print("=" * 78)
    print("OCTO-BEE probe report")
    print("=" * 78)
    for bi, bx in enumerate(boxes):
        n = bx["ai"].shape[0]
        gaps, lost = ob.check_continuity(bx["sam_cnt"])
        print(f"{bx['host']}: {n} samples ({n/bx['fs_hz']:.2f} s @ {bx['fs_hz']/1e3:g} kHz), "
              f"{bx['adc_range']}, {gaps} gap(s)/{lost} lost, "
              f"PWR_GOOD=0x{bx['pwr_good']:08x} (latched at boot)")
    print(f"assumed sensor range: +/-{b_range_mt:g} mT  ({v_per_t:g} V/T)  "
          f"-- confirm with onbox_sensor_audit.py")
    print()

    # ---- per-sensor zero point, noise, temperature -----------------------
    print("-" * 78)
    print("per-sensor zero point and noise      (VCM subtracted; 1 count = "
          f"{boxes[0]['vpc']*1e6:.1f} uV)")
    print("-" * 78)
    print(f"{'sensor':>7} {'VCM [V]':>9} {'T [C]':>7} "
          f"{'off Bz':>9} {'off By':>9} {'off Bx':>9}   "
          f"{'noise Bz':>9} {'noise By':>9} {'noise Bx':>9}   (uT rms)")

    rows = []
    for bi, bx in enumerate(boxes):
        ai = bx["ai"]
        vpc = bx["vpc"]
        temps = ob.temp_c(bx["temp_raw"])
        for s in range(ob.SENSORS_PER_BOX):
            gs = bi * ob.SENSORS_PER_BOX + s + 1          # global sensor number
            vcm_v = ai[:, s * 4 + 3].mean() * vpc
            offs, noise = [], []
            for a in range(3):
                x = (ai[:, s * 4 + a].astype(np.float64) - ai[:, s * 4 + 3]) * vpc
                offs.append(x.mean() / v_per_t * 1e6)      # uT
                noise.append(x.std() / v_per_t * 1e6)      # uT rms
            rows.append({"sensor": gs, "vcm_v": vcm_v, "temp_c": temps[s],
                         "off": offs, "noise": noise})
            print(f"{'S'+str(gs):>7} {vcm_v:9.4f} {temps[s]:7.1f} "
                  f"{offs[0]:9.1f} {offs[1]:9.1f} {offs[2]:9.1f}   "
                  f"{noise[0]:9.2f} {noise[1]:9.2f} {noise[2]:9.2f}")

    vcms = np.array([r["vcm_v"] for r in rows])
    noises = np.array([r["noise"] for r in rows])
    print()
    print(f"VCM spread across the 16 chips: {vcms.min():.4f} .. {vcms.max():.4f} V "
          f"(delta {(vcms.max()-vcms.min())*1e3:.1f} mV = "
          f"{(vcms.max()-vcms.min())/v_per_t*1e6:.0f} uT of apparent field if not subtracted)")
    med_noise = np.median(noises)
    worst = np.argsort(noises.max(axis=1))[::-1][:4]
    print(f"noise: median {med_noise:.2f} uT rms; noisiest sensors " +
          ", ".join(f"S{rows[i]['sensor']} ({noises[i].max():.1f})" for i in worst))
    print()

    # ---- magnet response -------------------------------------------------
    print("-" * 78)
    print("magnet response (peak deviation from each channel's own baseline)")
    print("-" * 78)
    peaks = []
    for bi, bx in enumerate(boxes):
        ai = bx["ai"]
        vpc = bx["vpc"]
        for s in range(ob.SENSORS_PER_BOX):
            gs = bi * ob.SENSORS_PER_BOX + s + 1
            ax_pk = []
            for a in range(3):
                x = (ai[:, s * 4 + a].astype(np.float64) - ai[:, s * 4 + 3]) * vpc
                base = np.median(x)
                dev = x - base
                # signed peak: whichever extreme is larger in magnitude
                i = np.argmax(np.abs(dev))
                ax_pk.append(dev[i] / v_per_t * 1e3)        # mT
            peaks.append({"sensor": gs, "pk": ax_pk,
                          "mag": float(np.linalg.norm(ax_pk))})

    mags = np.array([p["mag"] for p in peaks])
    active = mags > max(5 * med_noise * 1e-3, 1e-3)          # above the noise floor
    print(f"{'sensor':>7} {'pk Bz':>10} {'pk By':>10} {'pk Bx':>10} "
          f"{'|B| peak':>10}   {'rel. to median':>15}")
    ref = np.median(mags[active]) if active.any() else np.nan
    for p, m, act in zip(peaks, mags, active):
        rel = f"{m/ref:8.2f} x" if (act and np.isfinite(ref) and ref > 0) else "     --"
        flag = "" if act else "   (no response above noise)"
        print(f"{'S'+str(p['sensor']):>7} {p['pk'][0]:10.3f} {p['pk'][1]:10.3f} "
              f"{p['pk'][2]:10.3f} {m:10.3f}   {rel:>15}{flag}")
    print(f"\nall values in mT, assuming the +/-{b_range_mt:g} mT range.")
    if active.sum() >= 2:
        spread = mags[active].max() / mags[active].min()
        print(f"responding sensors: {int(active.sum())}/16; "
              f"largest/smallest peak = {spread:.2f}x")
        if spread > 1.3:
            print("  -> that is well beyond sensor-to-sensor calibration tolerance.")
            print("     Check, in this order: (1) amplifier gain register per chip")
            print("     (onbox_sensor_audit.py), (2) whether EEPROM calibration is")
            print("     actually loaded on every chip, (3) magnet distance/alignment.")
    else:
        print("no clear magnet response in this capture -- pass a magnet by the probe")
        print("while capturing (try --seconds 20) to get the amplitude comparison.")

    if do_plot:
        _plot(boxes, peaks, mags, b_range_mt, plot_path)


def _plot(boxes, peaks, mags, b_range_mt, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v_per_t = ob.RANGE_TO_VPT[b_range_mt]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), height_ratios=[2, 1])

    for bi, bx in enumerate(boxes):
        ai = bx["ai"]
        vpc = bx["vpc"]
        fs = bx["fs_hz"]
        dec = max(1, ai.shape[0] // 4000)
        t = np.arange(0, ai.shape[0], dec) / fs
        for s in range(ob.SENSORS_PER_BOX):
            gs = bi * ob.SENSORS_PER_BOX + s + 1
            for a, nm in enumerate(("Bz", "By", "Bx")):
                x = (ai[::dec, s * 4 + a].astype(np.float64) - ai[::dec, s * 4 + 3]) * vpc
                x = (x - np.median(x)) / v_per_t * 1e3
                ax1.plot(t, x, lw=0.7, alpha=0.8,
                         label=f"S{gs} {nm}" if a == 0 and gs % 4 == 1 else None)
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel(f"B [mT]  (VCM subtracted, baseline removed)")
    ax1.set_title(f"All 48 field channels overlaid, 16 sensors "
                  f"(assumed +/-{b_range_mt:g} mT range)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7, ncol=4, loc="upper right")

    idx = np.arange(len(peaks))
    ax2.bar(idx, mags, color=["tab:blue"] * 8 + ["tab:orange"] * 8)
    ref = np.median(mags[mags > 0]) if (mags > 0).any() else 0
    if ref:
        ax2.axhline(ref, color="k", ls="--", lw=1, label=f"median {ref:.3f} mT")
        ax2.legend(fontsize=8)
    ax2.set_xticks(idx)
    ax2.set_xticklabels([f"S{p['sensor']}" for p in peaks], fontsize=8)
    ax2.set_ylabel("|B| peak [mT]")
    ax2.set_title("Peak magnet response per sensor "
                  "(blue = acq1001_694, orange = acq1001_695)")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nwrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uut", action="append", default=None)
    p.add_argument("--seconds", type=float, default=3.0,
                   help="live capture length (ignored with --load)")
    p.add_argument("--load", help="report on a saved .npz instead of capturing")
    p.add_argument("--range", dest="b_range", type=float, default=20.0,
                   choices=sorted(ob.RANGE_TO_VPT), metavar="mT",
                   help="sensor measurement range in mT: 20, 40, 400 or 4000 "
                        "(default 20 = max gain)")
    p.add_argument("--plot", action="store_true", help="also write a PNG")
    p.add_argument("--plot-path", default="octobee_cal.png")
    a = p.parse_args()

    hosts = tuple(a.uut) if a.uut else ob.DEFAULT_UUTS
    boxes = load_npz(a.load) if a.load else live(hosts, a.seconds)
    report(boxes, a.b_range, a.plot, a.plot_path)


if __name__ == "__main__":
    main()
