#!/usr/bin/env python3
"""
octobee/acq/idmap.py -- prove which ACQ423 channel belongs to which physical sensor.

The channel map in the OCTO-BEE guide (ch 4k+1=Bz, 4k+2=By, 4k+3=Bx, 4k+4=VCM)
is documentation, not measurement. Before you trust any calibration you should
confirm it on your own hardware -- especially here, where 16 sensors are mounted
on a square tube pointing outwards, so "sensor 5" on the ribbon is not obviously
"sensor 5" on the tube.

How it works
------------
1. On the carrier, run the sweep:
       ssh root@acq1001_694 'python3 /tmp/sensor_audit.py --id-sweep 4'
   It mutes each sensor's Bx/By/Bz over SPI, one sensor at a time, in position
   order 1..8, holding each for a few seconds.

2. Here, capture the whole sweep and watch which ACQ423 channels go quiet, and
   in what order. The k-th group to change is SPI position k.

This needs no clock synchronisation between the PC and the carrier -- only the
*order* of the events matters.

    python octobee/acq/idmap.py --uut acq1001_694 --seconds 70

Alternative without SPI: --magnet mode. Pass a magnet slowly along the tube,
sensor by sensor, and the same event detector reports the channel groups in the
order they responded. That also tells you the physical order along the tube,
which the SPI sweep cannot.
"""

import argparse

import numpy as np

from octobee.acq import carrier as ob
from octobee import live as ol


def bin_stats(ai, fs, bin_s):
    """(nsamples, 32) -> per-bin mean and std, shape (nbins, 32)."""
    k = max(1, int(fs * bin_s))
    n = (ai.shape[0] // k) * k
    x = ai[:n].astype(np.float64).reshape(-1, k, ai.shape[1])
    return x.mean(axis=1), x.std(axis=1), k / fs


def robust_z(v):
    """Deviation of each bin from the channel's own median, in MAD units."""
    med = np.median(v, axis=0)
    mad = np.median(np.abs(v - med), axis=0)
    mad = np.where(mad < 1e-9, 1e-9, mad)
    return (v - med) / (1.4826 * mad), med


def find_events(score, thresh, min_bins):
    """Cluster bins where any channel is anomalous into contiguous events."""
    hot = (np.abs(score) > thresh).any(axis=1)
    events, start = [], None
    for i, h in enumerate(hot):
        if h and start is None:
            start = i
        elif not h and start is not None:
            if i - start >= min_bins:
                events.append((start, i))
            start = None
    if start is not None and len(hot) - start >= min_bins:
        events.append((start, len(hot)))
    return events


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uut", default=ob.DEFAULT_UUTS[0],
                   help="single carrier to map (run once per box)")
    p.add_argument("--seconds", type=float, default=70.0,
                   help="capture length; must cover the whole sweep")
    p.add_argument("--fs", type=float, default=20000.0,
                   help="ADC rate to run at during the capture (0 = leave alone)")
    p.add_argument("--bin", type=float, default=0.25, help="analysis bin, seconds")
    p.add_argument("--thresh", type=float, default=8.0,
                   help="anomaly threshold in MAD units")
    p.add_argument("--magnet", action="store_true",
                   help="magnet-pass mode: rank groups by response instead of "
                        "looking for muted channels")
    p.add_argument("--save", default=None, help="also save the raw capture to .npz")
    a = p.parse_args()

    prev = None
    try:
        if a.fs:
            prev, actual = ol.set_rate(a.uut, a.fs)
            print(f"{a.uut}: sample rate -> {actual:.0f} Hz (restored on exit)")

        print(f"capturing {a.seconds:.0f} s from {a.uut} ...")
        d, lay = ob.capture_one(a.uut, a.seconds)
    finally:
        if prev:
            ol.restore_rate(a.uut, prev)
            print(f"{a.uut}: sample rate restored")

    ai = d["ai"]
    if a.save:
        np.savez_compressed(a.save, ai=ai, fs_hz=lay.fs_hz,
                            vpc=lay.volts_per_count)
        print(f"saved {a.save}")

    means, stds, bin_s = bin_stats(ai, lay.fs_hz, a.bin)
    print(f"{means.shape[0]} bins of {bin_s:.2f} s\n")

    if a.magnet:
        # rank sensor groups by peak |B| excursion, in time order of their peak
        print("magnet-pass mode: order in which each sensor group responded")
        order = []
        for s in range(ob.SENSORS_PER_BOX):
            vcm = means[:, s * 4 + 3]
            f = means[:, s * 4:s * 4 + 3] - vcm[:, None]
            f = f - np.median(f, axis=0)
            mag = np.sqrt((f ** 2).sum(axis=1))
            i = int(np.argmax(mag))
            order.append((i * bin_s, s, mag[i] * lay.volts_per_count))
        order.sort()
        print(f"{'rank':>5} {'t [s]':>8} {'channels':>12} {'group':>7} {'peak [mV]':>10}")
        for r, (t, s, pk) in enumerate(order, 1):
            print(f"{r:>5} {t:8.2f} {f'{s*4+1}-{s*4+4}':>12} "
                  f"{'S'+str(s+1):>7} {pk*1e3:10.2f}")
        print("\nThat is the physical order along the tube. Compare it with the")
        print("ribbon/port order to build your tube-position -> channel table.")
        return

    z, _med = robust_z(means)
    zs, _ = robust_z(stds)
    score = np.where(np.abs(z) > np.abs(zs), z, zs)
    events = find_events(score, a.thresh, max(1, int(1.0 / bin_s)))

    if not events:
        print("no events detected. Check that the sweep was running during the")
        print("capture, and try a lower --thresh.")
        return

    print(f"{len(events)} event(s) detected\n")
    print(f"{'#':>3} {'t_start':>8} {'t_end':>8}  channels affected -> inferred group")
    inferred = {}
    for k, (i0, i1) in enumerate(events, 1):
        seg = np.abs(score[i0:i1]).max(axis=0)
        chans = [c + 1 for c in range(ob.NCHAN_AI) if seg[c] > a.thresh]
        groups = sorted({(c - 1) // 4 for c in chans})
        gtxt = ", ".join(f"ch{(g*4)+1}-{(g*4)+4}" for g in groups)
        print(f"{k:>3} {i0*bin_s:8.2f} {i1*bin_s:8.2f}  "
              f"{','.join(map(str, chans)):40s} -> {gtxt}")
        if len(groups) == 1:
            inferred[k] = groups[0]

    print("\nverified map for", a.uut)
    print(f"{'SPI position':>13} {'ACQ423 channels':>18} {'guide says':>12}")
    ok = True
    for k in sorted(inferred):
        g = inferred[k]
        expect = k - 1
        mark = "" if g == expect else "   <-- MISMATCH"
        ok &= (g == expect)
        print(f"{k:>13} {f'{g*4+1}-{g*4+4}':>18} "
              f"{f'{expect*4+1}-{expect*4+4}':>12}{mark}")
    if ok and len(inferred) == ob.SENSORS_PER_BOX:
        print("\nchannel map CONFIRMED: SPI position k -> channels 4(k-1)+1 .. 4(k-1)+4")
    elif not ok:
        print("\nchannel map DIFFERS from the guide -- use the measured column.")
    else:
        print(f"\nonly {len(inferred)}/{ob.SENSORS_PER_BOX} positions resolved; "
              f"rerun with a longer dwell.")


if __name__ == "__main__":
    main()
