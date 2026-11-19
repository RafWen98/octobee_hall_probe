#!/usr/bin/env python3
"""
tools/rescue_positions.py -- reposition a volume map whose encoder slipped.

    python tools/rescue_positions.py captures/volume_20261118_153037.npz [...]
    python tools/rescue_positions.py --force y captures/volume_....npz

A map stores two records of where the head was: the encoder counts, latched
with every field sample, and the stage controllers' own position, read over
USB a few dozen times a second. The positions in the file come from the
encoders. When an encoder stops following its leadscrew -- on this rig, a
broken coupling that slipped and read 29 % short -- those positions are wrong
and the controller's are not.

For each line this compares where the encoder says an axis ended with where
its controller says it did. Where they differ by more than the tolerance, that
axis on that line is positioned from the controller reads instead. The result
goes to <name>_rescued.npz and .json beside the original, which is opened
read-only and never written.

Positions rebuilt this way carry the USB read's timing rather than the sample
clock's: good to a few tenths of a millimetre at sweep speed.
"""

import argparse
import json
import os
import sys

import numpy as np

from octobee import machine as omach
from octobee.motion import sweep as osweep


def rescue(path, tol_mm=osweep.ENCODER_TOL_MM, force=(), log=print):
    """Write <path>_rescued.npz/.json. Returns (output path or None, report)."""
    base = os.path.splitext(path)[0]
    arrays, side = osweep.SweepLog.load(path)
    if not side:
        log(f"{path}: no .json beside it, so the lines and the encoder "
            f"datums are unknown -- skipped")
        return None, {}
    placement = None
    pose = (side.get("machine") or {}).get("pose")
    if pose:
        placement = omach.Placement.from_dict(pose)
    new, new_side, report = osweep.rebuild_positions(
        arrays, side, placement, tol_mm=tol_mm, force=force)
    if not report:
        log(f"{os.path.basename(path)}: every encoder ended every line where "
            f"its controller did (within {tol_mm:g} mm) -- nothing to rescue")
        return None, report
    for axis, rows in sorted(report.items()):
        short = [100.0 * (1.0 - r["encoder_mm"] / r["controller_mm"])
                 for r in rows if abs(r["controller_mm"]) > 5.0]
        note = (f", encoder short by {np.median(short):.2f} % (median)"
                if short else "")
        log(f"{os.path.basename(path)}: {axis} rebuilt from the controller "
            f"on {len(rows)} line(s){note}")
    new_side["rescued_from"] = os.path.basename(base) + ".npz"
    out = base + "_rescued"
    np.savez_compressed(out + ".part.npz", **new)
    with open(out + ".part.json", "w", encoding="utf-8") as fh:
        json.dump(new_side, fh, indent=2, default=str)
        fh.write("\n")
    os.replace(out + ".part.npz", out + ".npz")
    os.replace(out + ".part.json", out + ".json")
    lost = int((~np.isfinite(new["pos_mm"]).all(axis=1)).sum()
               - (~np.isfinite(arrays["pos_mm"]).all(axis=1)).sum())
    log(f"  -> {out}.npz"
        + (f" ({lost:,} row(s) outside the controller reads now have no "
           f"position)" if lost > 0 else ""))
    return out + ".npz", report


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("maps", nargs="+", help="volume_*.npz files")
    p.add_argument("--tol", type=float, default=osweep.ENCODER_TOL_MM,
                   help="mm an encoder may end a line from its controller "
                        "(default %(default)s)")
    p.add_argument("--force", default="",
                   help="axes to rebuild on every line whatever the "
                        "comparison says, e.g. xy")
    args = p.parse_args(argv)
    for path in args.maps:
        rescue(path, args.tol, tuple(args.force))
    return 0


if __name__ == "__main__":
    sys.exit(main())
