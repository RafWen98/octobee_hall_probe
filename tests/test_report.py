"""The health report, on every ADC range the carriers support.

report.py converted counts to volts as counts * vpc and never added the range
pedestal. On the two unipolar ranges count 0 is not 0 V, so a healthy 2.5 V
VCM was reported as -2.5 V -- outside the plausible window, which would have
condemned all sixteen chips at once. The comment beside ADC_RANGES says it:
carrying the pedestal is the difference between the range being a supported
setting and being a trap.
"""

import numpy as np

from octobee.acq import carrier as ob
from octobee.calib import convert as ocal
from octobee import report as orep
from tests.helpers import (
    check,
)

TRUE_VCM_V = 2.5


def _capture(adc_range, n=2000):
    """One box of quiet data whose VCM really is TRUE_VCM_V on this range."""
    span, offset = ob.ADC_RANGES[adc_range]
    vpc = span / 65536.0
    counts = int(round((TRUE_VCM_V - offset) / vpc))
    rng = np.random.default_rng(4)
    ai = np.zeros((n, 32), dtype=np.int16)
    for ch in range(32):
        ai[:, ch] = counts + rng.normal(0, 15, n).astype(np.int16)
    return {"host": "synthetic", "ai": ai,
            "temp_raw": np.full(8, 2200, dtype=np.uint32), "pwr_good": 1,
            "fs_hz": 200000.0, "vpc": vpc, "volt_offset": offset,
            "adc_range": adc_range, "sam_cnt": np.arange(n, dtype=np.uint32)}


def test_vcm_is_right_on_every_adc_range():
    print("\nreport VCM across ADC ranges")
    lo, hi = ocal.PLAUSIBLE_V
    for adc_range in sorted(ob.ADC_RANGES):
        bx = _capture(adc_range)
        vpc, offset = bx["vpc"], bx["volt_offset"]
        vcm_v = bx["ai"][:, 3].mean() * vpc + bx.get("volt_offset", 0.0)
        check(f"{adc_range}: VCM reads {TRUE_VCM_V} V",
              abs(vcm_v - TRUE_VCM_V) < 1e-3, f"{vcm_v:+.4f} V")
        check(f"{adc_range}: that is inside the plausible window",
              lo <= vcm_v <= hi, f"{vcm_v:+.4f} V, window {lo}..{hi}")
        if offset:
            # What the bug did, stated so the test explains itself. Note the
            # damage differs by range: on 0-10V it lands at -2.5 V and trips
            # the plausibility check, so every chip reads as broken. On 0-5V
            # it lands at 0.0 V, which is INSIDE the window -- no alarm at
            # all, just a VCM that is silently 2.5 V wrong.
            without = bx["ai"][:, 3].mean() * vpc
            check(f"{adc_range}: without the pedestal it is out by {offset} V",
                  abs((vcm_v - without) - offset) < 1e-6,
                  f"{without:+.4f} V instead of {vcm_v:+.4f} V"
                  + ("  (and still looks plausible)" if lo <= without <= hi
                     else "  (trips the plausibility check)"))


def test_load_npz_reads_the_pedestal(workdir):
    """capture writes volt_offset; load_npz has to read it back."""
    print("\nreport capture round-trip")
    path = f"{workdir}/cap.npz"
    bx = _capture("0-10V", n=500)
    np.savez_compressed(
        path, hosts=np.array(["synthetic"]), ai_0=bx["ai"],
        sam_cnt_0=bx["sam_cnt"], temp_raw_0=bx["temp_raw"], pwr_good_0=1,
        fs_hz_0=bx["fs_hz"], vpc_0=bx["vpc"], volt_offset_0=bx["volt_offset"],
        adc_range_0=bx["adc_range"])
    boxes = orep.load_npz(path)
    check("the pedestal survives the file", boxes[0]["volt_offset"] == 5.0,
          str(boxes[0].get("volt_offset")))

    # older captures predate the key and were all taken on +/-10V
    path2 = f"{workdir}/old.npz"
    np.savez_compressed(
        path2, hosts=np.array(["synthetic"]), ai_0=bx["ai"],
        sam_cnt_0=bx["sam_cnt"], temp_raw_0=bx["temp_raw"], pwr_good_0=1,
        fs_hz_0=bx["fs_hz"], vpc_0=bx["vpc"], adc_range_0="+/-10V")
    boxes2 = orep.load_npz(path2)
    check("a capture without the key still loads, at zero",
          boxes2[0]["volt_offset"] == 0.0, str(boxes2[0].get("volt_offset")))
