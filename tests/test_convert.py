"""Counts to tesla: the conversion, the config, and channel health."""

import json
import os
import tempfile

import numpy as np

from octobee import paths
from octobee.acq import carrier as ob
from octobee.calib import convert as ocal
from octobee.calib import geometry as pgeom
from tests.helpers import (
    check,
)



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
