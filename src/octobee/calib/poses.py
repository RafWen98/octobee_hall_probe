#!/usr/bin/env python3
"""
octobee/calib/poses.py -- record an Earth-field roll sweep as INDEXED POSES rather
than a continuous hand roll.

Why a separate recorder?
------------------------
The GUI's "Record sweep A/B/C" takes one continuous recording off the live
stream, which runs the ADC clock down to 20 kSPS (octobee/live.py --fs) so the
display can keep up. That is the right trade for a hand roll, where you are
buying roll ANGLES and the head never stops moving.

A square tube in a V-block does not roll continuously -- it indexes in 90 deg
jumps. That changes the trade completely: the head is STATIONARY in each pose,
so the only thing limiting you is how long you average, and per README section
7 the sensor noise is white and falls as 1/sqrt(N) essentially perfectly:

    method                                  samples  alias   effective noise
    60 s hand roll off the live stream       1.2 M   x3.16       0.235 uT
    4 poses x 20 s offline, 200 kSPS          16 M   x1.00       0.020 uT

The V-block wins by 11.5x, because a stationary pose can be averaged for as
long as you like at the full rate, where the ADC does its own anti-aliasing.

Four poses is the ideal minimum set, not merely what a square tube gives you:
opposite poses subtract to isolate each transverse column of M and add to
cancel the roll entirely. Graded against synthetic truth at 20 s per pose with
all three mounting orientations, 4 poses match sensors to 0.038 % against
0.039 % for 8 and 0.041 % for 12 -- total averaging time is the limit, not the
number of angles. (More poses do sharpen the offsets, 0.048 -> 0.022 uT from 4
to 8, so add them if offsets are what you are after.)

You do NOT need the 90 deg to be accurate. solve_roll() fits every pose's roll
angle from the data, so a V-block that indexes to 88 deg is as good as one that
indexes to 90; +/-2 deg of pose error changes the result in the fourth decimal
place. What you do need is for the head to be RIGID between poses -- same
cradle, same cable dress, nothing ferrous moving with it.

Usage
-----
One run records one mounting orientation:

    python octobee/calib/poses.py --tag A --seconds 20    # axis horizontal, north
    python octobee/calib/poses.py --tag B --seconds 20    # cradle turned to south
    python octobee/calib/poses.py --tag C --seconds 20    # cradle turned to east

The tube never has to leave the horizontal and never has to come out of the
V-block: carrying the whole cradle round 180 deg is the same thing to the solve
as lifting the tube and reversing it end-for-end, because both send
Bz = B . (tube axis) to -Bz with the transverse magnitude unchanged. Three
bearings N/S/E give 0.041 % matching and 0.041 uT offsets with both gauges
identified -- see README "Which way to point the tube", including the two
bearing patterns that fail silently.

then solve all three together:

    python octobee/calib/roll.py captures/rollsweep_A.npz \
                              captures/rollsweep_B.npz \
                              captures/rollsweep_C.npz \
        --b-earth-ut <yours> --apply calibration.json

Each run writes captures/rollsweep_<TAG>.npz + .json, which is exactly the
format octobee/calib/roll.py and the GUI's "Load sweeps" already read.
"""

import argparse
import os
import sys
import time

import numpy as np

from octobee import paths
from octobee.acq import carrier as ob
from octobee.calib import convert as ocal
from octobee.calib import roll as opc

# A pose is stored as the MEDIAN of its sub-block means, not as a plain mean.
# That costs ~25 % in noise and buys immunity to someone walking past with a
# phone in their pocket -- the failure mode you cannot see at the time and
# cannot undo afterwards. The block spread doubles as a stability figure.
#
# A 20 s capture at 200 kSPS is 4 M samples x 32 channels x 2 boxes; carried to
# float64 through assemble() that is over 2 GB. So the dwell is taken as a
# series of shorter captures, each reduced to block means in integer space
# before anything is converted, which caps peak memory at about 0.7 GB however
# long you average. Each capture costs ~4 s of fixed overhead (socket setup,
# the 2 s stop_live_stream wait, decode), so the chunk wants to be long enough
# that the overhead does not dominate -- measured on the bench, 4 s of data
# took 7.7 s wall clock.
CHUNK_S = 10.0
BLOCKS_PER_CHUNK = 32

# Thrown away at the start of each pose. Opening the stream can hand back
# whatever the carrier had buffered, and immediately after indexing the block
# that means samples taken WHILE THE TUBE WAS MOVING. Those are not noise --
# they are a different pose -- so they are drained rather than averaged in.
DRAIN_S = 2.0

# Below this the capture is not usable as a pose row: either a killed live
# session left the clock down, or a box is misreporting.
MIN_FS_HZ = 190000.0

# Single-sample white noise per axis, microtesla, measured on this probe at the
# +/-20 mT range (README section 7). Averaging N samples divides it by sqrt(N)
# essentially perfectly, which is what makes the noise estimate below a
# prediction rather than a guess.
NOISE_1_SAMPLE_UT = 81.5


def check_rate(hosts):
    """
    Confirm every box is at full rate before an hour of bench work.

    Returns (ok, layouts) -- the layouts because the caller needs the REAL
    sample rate to predict the noise it should see, and assuming 200 kSPS
    there while checking it here would be its own small lie.
    """
    ok = True
    layouts = []
    for h in hosts:
        lay = ob.probe_uut(h)
        layouts.append(lay)
        flag = "" if lay.fs_hz >= MIN_FS_HZ else "   <-- NOT full rate"
        print(f"  {h}: {lay.fs_hz:.0f} Hz, {lay.adc_range}, "
              f"{lay.volts_per_count * 1e6:.1f} uV/count{flag}")
        if lay.fs_hz < MIN_FS_HZ:
            ok = False
    if not ok:
        print("\nA killed live session leaves the ADC clock down. Fix it with:")
        print("    python octobee/acq/carrier.py restore")
    return ok, layouts


def _block_means(ai, n_blocks):
    """
    (n, 32) counts -> (n_blocks, 32) float64 block means, without ever
    materialising the whole array in float64.

    np.mean with an explicit accumulator dtype reduces in buffered chunks, so
    the int16 capture is never copied wholesale.
    """
    n = ai.shape[0]
    factor = max(1, n // max(1, n_blocks))
    n_blocks = min(n_blocks, max(1, n // factor))
    out = np.empty((n_blocks, ai.shape[1]))
    for j in range(n_blocks):
        out[j] = ai[j * factor:(j + 1) * factor].mean(axis=0, dtype=np.float64)
    return out


def _grab(hosts, seconds, verbose=False):
    """One capture -> (per-box block means, vpc, temps, samples, lost)."""
    res = ob.capture_all(hosts, seconds, verbose=verbose)
    blocks, vpc, temps, n_samples, lost_total = [], [], [], None, 0
    for h in hosts:
        d, lay = res[h]
        gaps, lost = ob.check_continuity(d["sam_cnt"])
        lost_total += lost
        if gaps:
            print(f"    note: {h} {gaps} gap(s), {lost} sample(s) lost")
        blocks.append(_block_means(d["ai"], BLOCKS_PER_CHUNK))
        vpc.append(lay.volts_per_count)
        temps.append(ob.temp_c(d.get("temp_raw", np.zeros(8))))
        n_samples = d["ai"].shape[0] if n_samples is None else min(
            n_samples, d["ai"].shape[0])
        del d, res[h]                 # release ~130 MB before the next box
    return blocks, vpc, np.concatenate(temps), n_samples, lost_total


def capture_pose(hosts, seconds, cal, chunk_s=CHUNK_S, drain_s=DRAIN_S):
    """
    One pose -> (row (16,3) mT uncorrected, stats dict).

    "Uncorrected" is what RollSweep wants: VCM subtracted, divided by the
    nominal V/T for the configured range, no tare, no trim, no matrix.
    """
    if drain_s > 0:
        _grab(hosts, drain_s)         # discarded: may contain the rotation

    n_chunks = max(1, int(round(seconds / max(chunk_s, 0.5))))
    per_chunk = seconds / n_chunks

    rows, temps, n_samples, lost_total = [], [], 0, 0
    for _ in range(n_chunks):
        blocks, vpc, t_c, n_s, lost = _grab(hosts, per_chunk)
        n = min(b.shape[0] for b in blocks)
        volts = ocal.assemble([b[:n] for b in blocks], vpc)
        rows.append(cal.to_mt(volts, apply_zero=False, apply_gain=False,
                              apply_matrix=False))
        temps.append(t_c)
        n_samples += n_s
        lost_total += lost

    blocks_mt = np.concatenate(rows)                       # (~N_BLOCKS,16,3)
    row = np.median(blocks_mt, axis=0)
    # Robust spread of the block means, carried to a standard error on the
    # stored row. This is an EMPIRICAL noise figure for this pose in this room,
    # so it catches pickup and drift that a datasheet number never would.
    mad = np.median(np.abs(blocks_mt - row[None]), axis=0) * 1.4826
    sem_ut = float(np.median(mad)) * 1e3 / np.sqrt(blocks_mt.shape[0])

    return row, {"mad_ut": mad * 1e3,
                 "n_samples": int(n_samples),
                 "n_blocks": int(blocks_mt.shape[0]),
                 "n_chunks": n_chunks,
                 "lost": int(lost_total),
                 "sem_ut": sem_ut,
                 "temps_c": np.mean(temps, axis=0),
                 "bmag_ut": np.linalg.norm(row, axis=1) * 1e3}


# How uniform the field has to be across the head before a roll calibration
# means anything. The whole method rests on all 16 sensors sitting in the SAME
# field vector, so any spread between them is error that no averaging removes.
# Chip-to-chip gain is factory-trimmed to a few percent, which is why the
# "good" band is not tighter than that -- below ~2 % you are measuring the
# chips, above ~5 % you are measuring the room.
SURVEY_GOOD_PCT = 2.0
SURVEY_USABLE_PCT = 5.0


def survey_uniformity(row_a, row_b, live):
    """
    (bt per sensor in uT, median over live, spread as % of median).

    A single static capture cannot see a gradient: each sensor's offset is
    indistinguishable from the field it sits in. Rolling 180 deg cancels the
    offset exactly and leaves twice the transverse field, so

        bt_i = |pose_a_i - pose_b_i| / 2

    is an offset-free measurement of the field each sensor actually sat in.
    Under a uniform field every sensor must return the SAME number, whatever
    its orientation, because a magnitude is rotation-invariant. The spread
    between them is the non-uniformity, measured rather than assumed.

    Anything short of 180 deg scales every sensor's bt by the same
    sin(delta/2), so the SPREAD is unchanged and a sloppy index costs nothing
    here -- only the absolute median moves.
    """
    bt = np.linalg.norm(np.asarray(row_a) - np.asarray(row_b), axis=1) / 2.0 * 1e3
    good = bt[np.asarray(live, bool)]
    med = float(np.median(good))
    spread = float(good.max() - good.min()) / max(med, 1e-9) * 100.0
    return bt, med, spread


def survey_consistency(rows, live):
    """
    Two opposed pose pairs -> the one number a gain error cannot fake.

    Each pair is ~180 deg apart, so each yields an independent estimate of the
    same bt for the same sensor. A wrong GAIN scales both estimates identically
    and cancels in their ratio. Only a field that CHANGES AS THE ARM SWINGS can
    make them disagree.

    So `spread` (below) says "these sensors disagree with each other", which
    could be the room or could be the chips, while this ratio says "this sensor
    disagrees with ITSELF between two positions", which can only be the room.

    The opposite of pose i is i + n/2, NOT i + 2. Hard-coding rows[0]-rows[2]
    is only the same thing at exactly four poses: at --poses 6 it differences
    two poses 120 deg apart, which scales every bt by sin(60 deg) and -- worse
    -- scales BOTH pairs identically, so the ratio still comes back at 1.0 and
    reports the room as clean while the premise has quietly failed.
    """
    rows = [np.asarray(r) for r in rows]
    n = len(rows)
    if n < 4 or n % 2:
        raise ValueError(f"need an even number of poses, at least 4, got {n}")
    half = n // 2
    a = rows[0] - rows[half]
    b = rows[1] - rows[1 + half]
    d13 = np.linalg.norm(a, axis=1) / 2.0 * 1e3
    d24 = np.linalg.norm(b, axis=1) / 2.0 * 1e3
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(d24 > 1e-9, d13 / np.where(d24 > 1e-9, d24, 1.0), np.nan)
    sel = np.asarray(live, bool) & np.isfinite(ratio)
    worst = float(np.max(np.abs(ratio[sel] - 1.0))) * 100.0 if sel.any() else np.nan
    return d13, d24, ratio, worst


def survey(hosts, cal, live, seconds, chunk, drain, settle, yes, poses=4):
    """
    Score a location before committing to a full sweep.

    Runs poses+2 captures, because "the numbers disagree" has two very
    different causes and they need separating:

      * capture 1 and 2 are the SAME pose, back to back, nothing touched.
        Their difference is the floor -- noise plus whatever the room is doing
        on its own. Nothing that follows can be better than this.
      * the last capture returns to pose 1 after a full turn. Its difference
        from capture 1 is what the ACT OF ROLLING cost you. If the static
        floor is small and this is large, the room is not the problem: the
        cables, the connector shells or the way the tube re-seats are.
    """
    step = 360 // max(1, poses)
    rows, static_pair = [], None
    n_total = poses + 2
    k = 0
    while k < n_total:
        static = k == 1
        closing = k == n_total - 1
        if static:
            what = ("SAME pose again, nothing touched -- this measures the "
                    "floor")
        elif closing:
            what = (f"CLOSURE: index the last {step} deg, back to pose 1 -- "
                    f"this measures what rolling cost you")
        elif k == 0:
            what = f"pose 1 of {poses}: any starting position"
        else:
            what = f"pose {k} of {poses}: index {step} deg in the V-block"
        print(f"\n[{k + 1}/{n_total}] {what}")
        if not yes and not static:
            try:
                input("        hands off the head, then press Enter "
                      "(Ctrl-C aborts) ... ")
            except (EOFError, KeyboardInterrupt):
                print("\naborted")
                return 1
        time.sleep(max(0.0, settle))
        print(f"        capturing {seconds:.0f} s ...", end="", flush=True)
        t0 = time.time()
        row, st = capture_pose(hosts, seconds, cal, chunk, drain)
        print(f" done in {time.time() - t0:.0f} s "
              f"(noise {st['sem_ut']:.3f} uT)")
        if static:
            static_pair = np.abs(row - rows[0])[np.asarray(live, bool)] * 1e3
            print(f"        static repeat: median "
                  f"{np.median(static_pair):.3f} uT, "
                  f"worst {static_pair.max():.3f} uT")
        elif closing:
            back = np.abs(row - rows[0])[np.asarray(live, bool)] * 1e3
        else:
            rows.append(row)
        k += 1

    bt, med, spread_pct = survey_uniformity(rows[0], rows[poses // 2], live)
    # The self-consistency check needs two independent opposed PAIRS, so it
    # wants an even pose count of at least four. An odd count has no opposite
    # pose at all and gets the single-pair answer above.
    full = poses >= 4 and poses % 2 == 0
    if full:
        d13, d24, ratio, worst_self = survey_consistency(rows, live)
        bt = 0.5 * (d13 + d24)
        med = float(np.median(bt[live]))
    elif poses > 4:
        print(f"\nnote: {poses} poses is odd, so there is no opposed pair to "
              f"cross-check with. Use an even count for the self-disagreement "
              f"column.")

    opp = poses // 2
    hdr = ("  sensor    field it sat in     vs median"
           + (f"     from 1&{1 + opp}   from 2&{2 + opp}   self-disagreement"
              if full else ""))
    print("\n" + hdr)
    for i in range(opc.N_SENSORS):
        mark = "" if live[i] else "   (excluded)"
        extra = (f"   {d13[i]:>8.1f}   {d24[i]:>8.1f}   "
                 f"{abs(ratio[i] - 1) * 100:>8.0f} %" if full else "")
        print(f"   S{i + 1:<3}     {bt[i]:>8.2f} uT      "
              f"{bt[i] / med:>5.2f}x{extra}{mark}")

    good = bt[np.asarray(live, bool)]
    # Robust as well as peak-to-peak: four bad arms out of fifteen make the
    # peak-to-peak useless for answering "is the REST of the head clean?",
    # which is the question that decides whether you move the bench or hunt
    # down one object.
    mad_pct = float(np.median(np.abs(good - med))) / max(med, 1e-9) * 100.0
    print(f"\nmedian transverse field: {med:.2f} uT"
          f"   (Earth alone would give ~{opc.DEFAULT_B_EARTH_UT:.0f} uT here)")
    print(f"spread across the head : {good.min():.2f}-{good.max():.2f} uT "
          f"= {spread_pct if not full else (good.max()-good.min())/med*100:.1f} % "
          f"peak-to-peak, {mad_pct:.1f} % median deviation")
    print("under a uniform field every one of those would be the same number")

    outliers = [i for i in range(opc.N_SENSORS)
                if live[i] and abs(bt[i] - med) / max(med, 1e-9) > 0.15]
    if outliers:
        print("more than 15 % out: "
              + ", ".join(f"S{i + 1} ({bt[i] / med:.2f}x)" for i in outliers))

    if full:
        print("\nself-disagreement (a sensor against ITSELF, two positions):")
        print(f"  worst {worst_self:.0f} %. A wrong gain cancels in this ratio, so")
        print("  anything here is the FIELD changing as that arm swings --")
        print("  it cannot be the chip.")

    print("\nwhere the disagreement comes from:")
    print(f"  standing still      : median {np.median(static_pair):.3f} uT, "
          f"worst {static_pair.max():.3f} uT")
    print(f"  after a full turn   : median {np.median(back):.3f} uT, "
          f"worst {back.max():.3f} uT")
    ratio_turn = np.median(back) / max(np.median(static_pair), 1e-9)
    if ratio_turn > 5:
        print(f"  -> rolling costs {ratio_turn:.0f}x what standing still does. "
              f"The room is not\n     drifting; something about the ROTATION is "
              f"moving field relative to\n     the sensors. Check, in this order: "
              f"cables and connector shells that\n     do not rotate rigidly with "
              f"the head; the tube re-seating differently\n     in the V-block; "
              f"anything ferrous the arms swing past.")
    else:
        print(f"  -> rolling costs about what standing still does "
              f"({ratio_turn:.1f}x), so the\n     mechanics are sound and any "
              f"spread above is the field itself.")

    spread_now = (good.max() - good.min()) / max(med, 1e-9) * 100.0
    if spread_now <= SURVEY_GOOD_PCT:
        print(f"\nVERDICT: good. {spread_now:.1f} % is at or below the chips' own "
              f"gain spread,\n         so what is left to measure IS the chips. "
              f"Record the sweep here.")
    elif spread_now <= SURVEY_USABLE_PCT:
        print(f"\nVERDICT: usable, but {spread_now:.1f} % becomes the floor on "
              f"inter-sensor\n         matching -- you cannot match sensors better "
              f"than the field\n         they disagree about. Try another spot "
              f"first.")
    elif outliers and mad_pct <= SURVEY_USABLE_PCT:
        print(f"\nVERDICT: LOCALISED. The bulk of the head agrees to "
              f"{mad_pct:.1f} %, but\n         "
              + ", ".join(f"S{i + 1}" for i in outliers)
              + " are far out. That is not a room-sized\n"
                "         gradient -- it is something close to those specific "
                "arms.\n         Look at what they swing past before moving the "
                "whole bench.")
    else:
        print(f"\nVERDICT: too non-uniform. At {spread_now:.1f} % the premise of "
              f"the method\n         fails: the sensors are not sitting in the "
              f"same field, so their\n         disagreement is the room, not "
              f"them. Move and re-survey.")
    return 0 if spread_now <= SURVEY_USABLE_PCT else 4


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None,
                    help="mounting orientation: A as mounted, B end-for-end, "
                         "C cradle at another azimuth. Required unless --survey")
    ap.add_argument("--survey", action="store_true",
                    help="score this LOCATION in two poses and write nothing: "
                         "measures how uniform the field is across the head, "
                         "which is the assumption the whole method rests on. "
                         "Run it before committing to a full sweep")
    ap.add_argument("--poses", type=int, default=4,
                    help="poses per turn (default 4, i.e. 90 deg steps)")
    ap.add_argument("--turns", type=int, default=1,
                    help="go round this many times, revisiting the SAME poses "
                         "(default 1). A V-block offers four positions and no "
                         "more, so this is how you buy more data without "
                         "inventing angles it cannot index to")
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="averaging time per pose (default 20 s = 0.041 uT)")
    ap.add_argument("--chunk", type=float, default=CHUNK_S,
                    help=f"split the dwell into captures this long, to bound "
                         f"memory (default {CHUNK_S:g} s)")
    ap.add_argument("--drain", type=float, default=DRAIN_S,
                    help=f"discard this much at the start of each pose, in case "
                         f"the carrier hands back samples taken while the tube "
                         f"was still moving (default {DRAIN_S:g} s)")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="wait this long after you press Enter before reading")
    ap.add_argument("--uut", action="append", default=None,
                    help="carrier hostname/IP, repeatable")
    ap.add_argument("--out-dir", default=paths.captures_dir())
    ap.add_argument("--cal", default=paths.config(ocal.CONFIG_NAME),
                    help="calibration.json, read for the per-sensor RANGES only")
    ap.add_argument("--no-closure", action="store_true",
                    help="skip the repeat of pose 1 that measures session drift")
    ap.add_argument("--dead", default="",
                    help="comma-separated ids to leave out of the on-screen "
                         "summary, e.g. S16")
    ap.add_argument("--yes", action="store_true",
                    help="do not wait for Enter between poses (for a motorised "
                         "or otherwise scripted index)")
    a = ap.parse_args(argv)
    if not a.survey and not a.tag:
        ap.error("--tag is required (or use --survey to score a location first)")

    hosts = a.uut or list(ob.DEFAULT_UUTS)
    dead = {d.strip().upper() for d in a.dead.split(",") if d.strip()}
    cal = ocal.Calibration.load_or_default(a.cal)
    live = np.array([f"S{i + 1}" not in dead for i in range(opc.N_SENSORS)])
    step = 360 // max(1, a.poses)
    n_poses = a.poses * max(1, a.turns)

    if a.survey:
        print(f"=== location survey: {a.poses} poses x {a.seconds:.0f} s "
              f"at full rate")
        ok, _layouts = check_rate(hosts)
        if not ok:
            return 2
        return survey(hosts, cal, live, a.seconds, a.chunk, a.drain,
                      a.settle, a.yes, poses=a.poses)

    print(f"=== roll sweep {a.tag.upper()}: {a.poses} poses"
          + (f" x {a.turns} turns" if a.turns > 1 else "")
          + f" x {a.seconds:.0f} s at full rate")
    print(f"ranges from {a.cal}: "
          f"{', '.join(sorted({f'{r:g} mT' for r in cal.ranges_mt}))}")
    ok, layouts = check_rate(hosts)
    if not ok:
        return 2
    fs_hz = min(lay.fs_hz for lay in layouts)
    expected_ut = NOISE_1_SAMPLE_UT / np.sqrt(fs_hz * a.seconds)
    print(f"expected noise per pose: {expected_ut:.3f} uT per axis "
          f"({expected_ut / opc.DEFAULT_B_EARTH_UT * 100:.3f} % of a "
          f"{opc.DEFAULT_B_EARTH_UT:.0f} uT field)")

    rows, all_stats = [], []
    n_total = n_poses + (0 if a.no_closure else 1)
    for k in range(n_total):
        closing = k == n_poses
        turn, idx = k // a.poses + 1, k % a.poses + 1
        of = (f"{idx} of {a.poses}" if a.turns == 1
              else f"{idx} of {a.poses}, turn {turn} of {a.turns}")
        if closing:
            what = (f"CLOSURE: index the last {step} deg, back to pose 1 "
                    f"(completing the turn)")
        elif k == 0:
            what = (f"pose 1 of {a.poses}: start anywhere -- the roll angle is "
                    f"solved, not assumed")
        elif idx == 1:
            what = (f"pose {of}: index {step} deg -- you are back where you "
                    f"started, going round again for more averaging")
        else:
            what = (f"pose {of}: index {step} deg about the "
                    f"TUBE AXIS, same direction as last time")
        print(f"\n[{k + 1}/{n_total}] {what}")
        if not a.yes:
            try:
                input("        hands off the head, then press Enter "
                      "(Ctrl-C aborts) ... ")
            except (EOFError, KeyboardInterrupt):
                print("\naborted; nothing written")
                return 1
        # A tube just nudged into a V-block is still ringing, and a moving head
        # smears the pose.
        time.sleep(max(0.0, a.settle))
        print(f"        capturing {a.seconds:.0f} s ...", end="", flush=True)
        t0 = time.time()
        try:
            row, st = capture_pose(hosts, a.seconds, cal, a.chunk, a.drain)
        except (RuntimeError, OSError) as exc:
            print(f"\n        capture failed: {exc}")
            print("        nothing was written; fix and re-run this sweep")
            return 3
        print(f" done in {time.time() - t0:.0f} s")

        # These are UNCORRECTED readings, so what dominates them is each
        # sensor's own offset (0.1-0.6 mT on this probe) rather than the 49 uT
        # field. A median near a few hundred uT is normal and is not a fault;
        # what matters is that no sensor reads flat.
        mag = st["bmag_ut"][live]
        print(f"        raw |B| (offset-dominated): median {np.median(mag):.1f} uT, "
              f"{mag.min():.1f}-{mag.max():.1f} uT across sensors")
        note = ("" if st["sem_ut"] < 4 * expected_ut
                else "   <-- 4x expectation: something in the room is moving")
        print(f"        measured noise: {st['sem_ut']:.3f} uT{note}")
        if st["lost"]:
            print(f"        {st['lost']} sample(s) lost across the dwell -- "
                  f"pose still usable, averaging absorbs it")

        if closing:
            # Restricted to live sensors, and read off the MEDIAN rather than
            # the max: one chronically noisy axis (S15 Bx is the live example,
            # 860 uT rms against a 73 uT pack) otherwise sets the number for
            # all 48, and a dead sensor's railed channel sets it at 10^5.
            drift = np.abs(row - rows[0])[live] * 1e3
            hi = float(np.percentile(drift, 90))
            print(f"\nCLOSURE: drift from pose 1 over the whole sweep: "
                  f"median {np.median(drift):.3f} uT, 90th pct {hi:.3f} uT, "
                  f"worst {drift.max():.3f} uT")
            if hi > 5.0:
                print("        > 5 uT on nine channels in ten: the field or the "
                      "head moved. Re-record; do not solve on this.")
            elif hi > 1.0:
                print("        1-5 uT: usable, but this is now your error "
                      "floor, not the noise figure above.")
            else:
                print("        clean: the session held to under a microtesla.")
            print("        a single channel far above the 90th percentile is "
                  "that channel's problem, not the sweep's -- check it in "
                  "octobee/report.py")
            break

        rows.append(row)
        all_stats.append(st)

    sw = opc.RollSweep(np.stack(rows), tag=a.tag,
                       ranges_mt=cal.ranges_mt,
                       temps_c=np.mean([s["temps_c"] for s in all_stats], axis=0),
                       meta={"source": "octobee_posecap",
                             "poses": len(rows),
                             "poses_per_turn": a.poses,
                             "turns": a.turns,
                             "seconds_per_pose": a.seconds,
                             "samples_per_pose": [s["n_samples"] for s in all_stats],
                             "sem_ut": [s["sem_ut"] for s in all_stats],
                             "hosts": list(hosts)})
    os.makedirs(a.out_dir, exist_ok=True)
    path = sw.save(os.path.join(a.out_dir, f"rollsweep_{a.tag.upper()}"))
    amp = sw.amplitudes()[live]
    print(f"\nwrote {path}  ({len(sw)} poses)")
    print(f"transverse swing: median {np.median(amp) * 1e3:.1f} uT across the "
          f"live sensors")
    print("a sensor reading near zero here saw nothing -- check its ribbon "
          "before solving")
    return 0


if __name__ == "__main__":
    sys.exit(main())
