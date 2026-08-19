#!/usr/bin/env python3
"""
octobee_calibration.py -- turn raw ACQ423 counts into calibrated field vectors,
and keep the calibration state that does it.

The conversion chain, in order, is:

    counts  --(ADC volts/count)-->  volts
            --(subtract that chip's own VCM channel)-->  differential volts
            --(divide by the chip's V/T sensitivity)-->  tesla, chip frame
            --(per-axis gain correction)-->  cross-calibrated tesla
            --(subtract ambient zero)-->  field relative to the tare point
            --(rotate by R_i from probe_geometry)-->  tesla, common tube frame

Every step is separately inspectable, because on this hardware every step has
been a suspect at some point:

  * VCM: each chip's virtual ground differs by up to ~90 mV, which is ~1.4 mT of
    apparent field at the 20 mT range if you forget to subtract it.
  * V/T sensitivity: set by the chip's G_CTRL gain register. Gain 3000 and 1500
    are exactly a factor of two apart in response, and the registers have NOT
    been audited on this probe -- so the per-sensor range is settable here,
    not global.
  * gain correction: whatever is left over after the above, measured from a
    magnet pass and divided out.
  * rotation: 16 chips, 16 different orientations.

Nothing here talks to the hardware; it operates on decoded arrays, so it can be
unit-tested and replayed against saved captures.
"""

import json
import os

import numpy as np

import octobee as ob
import probe_geometry as pg

N_SENSORS = pg.N_SENSORS
AXES = ("Bx", "By", "Bz")

# Channel order inside one sensor's group of 4 is Bz, By, Bx, VCM
# (OCTO-BEE guide sec 3.7). Reorder to the friendlier Bx, By, Bz, VCM.
GROUP_REORDER = [2, 1, 0, 3]

CONFIG_NAME = "calibration.json"

ADC_RAIL = 32767            # 16-bit signed full scale

# The SENM3Dx analogue output can only ever sit between 0.125 V and 4.380 V
# (datasheet Table 7). The ADC, on the +/-10 V range, can represent far more
# than that, so a channel resting outside this window is not a working sensor
# output at all -- it is a broken connection, a dead buffer, or a railed input.
# The window is opened up a little to leave room for offset and drift.
PLAUSIBLE_V = (-0.5, 5.5)


# --------------------------------------------------------------------------
# raw assembly
# --------------------------------------------------------------------------

def assemble(ai_by_box, vpc_by_box=None):
    """
    Per-box (n, 32) count arrays -> (n, 16, 4) volts (or counts) ordered
    Bx, By, Bz, VCM per sensor, sensor axis running S1..S16.

    The two carriers free-run on separate oscillators, so the boxes are trimmed
    to the shorter one and only aligned to about the start of the capture.
    That is fine for amplitude work and useless for phase -- see README.
    """
    ai_by_box = [np.asarray(a) for a in ai_by_box]
    n = min(a.shape[0] for a in ai_by_box)
    out = []
    for bi, ai in enumerate(ai_by_box):
        x = ai[:n].astype(np.float64)
        if vpc_by_box is not None:
            x = x * float(vpc_by_box[bi])
        nsens = x.shape[1] // 4
        g = x.reshape(n, nsens, 4)[:, :, GROUP_REORDER]
        out.append(g)
    return np.concatenate(out, axis=1)


def decimate(x, factor):
    """Block-average along axis 0. factor <= 1 is a no-op."""
    factor = int(factor)
    if factor <= 1:
        return x
    n = (x.shape[0] // factor) * factor
    if n == 0:
        return x[:0]
    return x[:n].reshape(n // factor, factor, *x.shape[1:]).mean(axis=1)


# --------------------------------------------------------------------------
# calibration state
# --------------------------------------------------------------------------

class Calibration:
    """
    Everything needed to convert counts to calibrated millitesla.

    ranges_mt   (16,)    per-sensor full-scale range: 20, 40, 400 or 4000 mT.
                         Per-sensor, not global, because the chips' gain
                         registers have not been audited and may well differ --
                         that is the leading suspect for unequal spike heights.
    zero_mt     (16, 3)  ambient offset removed after conversion (the tare).
    gain_corr   (16, 3)  dimensionless residual gain trim, 1.0 = untouched.
    dead        set      "S16" or "S16.Bx" entries excluded from summaries.
    """

    def __init__(self, ranges_mt=None, zero_mt=None, gain_corr=None,
                 subtract_vcm=True, dead=None, notes=""):
        self.ranges_mt = (np.full(N_SENSORS, 20.0) if ranges_mt is None
                          else np.asarray(ranges_mt, float).reshape(N_SENSORS))
        self.zero_mt = (np.zeros((N_SENSORS, 3)) if zero_mt is None
                        else np.asarray(zero_mt, float).reshape(N_SENSORS, 3))
        self.gain_corr = (np.ones((N_SENSORS, 3)) if gain_corr is None
                          else np.asarray(gain_corr, float).reshape(N_SENSORS, 3))
        self.subtract_vcm = bool(subtract_vcm)
        self.dead = set(dead or ())
        self.notes = notes

    # ---- persistence ----------------------------------------------------
    def to_dict(self):
        return {"ranges_mt": self.ranges_mt.tolist(),
                "zero_mt": self.zero_mt.tolist(),
                "gain_corr": self.gain_corr.tolist(),
                "subtract_vcm": self.subtract_vcm,
                "dead": sorted(self.dead),
                "notes": self.notes}

    def save(self, path=CONFIG_NAME):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path=CONFIG_NAME):
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))

    @classmethod
    def load_or_default(cls, path=CONFIG_NAME):
        if path and os.path.exists(path):
            try:
                return cls.load(path)
            except (OSError, ValueError, TypeError):
                pass
        return cls()

    # ---- conversion -----------------------------------------------------
    @property
    def volts_per_tesla(self):
        """(16,) sensitivity, from each sensor's own configured range."""
        return np.array([ob.RANGE_TO_VPT[float(r)] for r in self.ranges_mt])

    def to_mt(self, grouped_volts, apply_zero=True, apply_gain=True):
        """
        (n, 16, 4) volts from assemble() -> (n, 16, 3) millitesla, chip frame,
        axis order Bx, By, Bz.
        """
        v = np.asarray(grouped_volts, float)
        field = v[..., :3]
        if self.subtract_vcm:
            field = field - v[..., 3:4]
        b = field / self.volts_per_tesla[None, :, None] * 1e3      # V -> mT
        if apply_gain:
            b = b * self.gain_corr[None, :, :]
        if apply_zero:
            b = b - self.zero_mt[None, :, :]
        return b

    def convert(self, ai_by_box, vpc_by_box, apply_zero=True, apply_gain=True):
        """Raw per-box counts straight through to (n, 16, 3) millitesla."""
        return self.to_mt(assemble(ai_by_box, vpc_by_box), apply_zero, apply_gain)

    # ---- calibration steps ----------------------------------------------
    def tare(self, b_mt):
        """
        Set the zero point from an ambient capture, i.e. from (n, 16, 3) mT
        computed with apply_zero=False. Median, not mean, so a stray magnet
        wave during the tare does not poison it.
        """
        self.zero_mt = np.median(np.asarray(b_mt, float), axis=0)
        return self.zero_mt

    def clear_tare(self):
        self.zero_mt = np.zeros((N_SENSORS, 3))

    def cross_calibrate(self, peak_mag_mt, weights=None, min_response=None):
        """
        Trim residual per-sensor gain so every sensor reports the same |B| for
        the same physical stimulus.

        peak_mag_mt  (16,)  measured peak |B| per sensor from one magnet pass.
        weights      (16,)  expected relative response from geometry
                            (probe_geometry.expected_response). Supply it if the
                            magnet was NOT equidistant from every sensor --
                            without it you would fold the 1/r^3 distance
                            difference into the gain trim and make things worse.

        A sensor with no usable response keeps its previous correction and is
        reported back in `skipped`.
        """
        p = np.asarray(peak_mag_mt, float).reshape(N_SENSORS)
        w = np.ones(N_SENSORS) if weights is None else np.asarray(weights, float)
        corrected = p / np.where(w > 0, w, np.nan)
        if min_response is None:
            min_response = 0.02 * np.nanmax(corrected)
        ok = np.isfinite(corrected) & (corrected > min_response)
        skipped = [f"S{i+1}" for i in range(N_SENSORS) if not ok[i]]
        if ok.sum() < 2:
            return self.gain_corr, skipped
        ref = np.median(corrected[ok])
        new = self.gain_corr.copy()
        new[ok, :] = self.gain_corr[ok, :] * (ref / corrected[ok])[:, None]
        self.gain_corr = new
        return self.gain_corr, skipped

    def clear_gain(self):
        self.gain_corr = np.ones((N_SENSORS, 3))

    def is_dead(self, sensor_id, axis=None):
        if f"S{sensor_id}" in self.dead:
            return True
        return axis is not None and f"S{sensor_id}.{axis}" in self.dead

    def live_mask(self):
        return np.array([not self.is_dead(i + 1) for i in range(N_SENSORS)])

    def summary(self):
        if np.all(self.ranges_mt == self.ranges_mt[0]):
            ranges = f"ranges: all +/-{self.ranges_mt[0]:g} mT"
        else:
            odd = ", ".join(f"S{i+1}=+/-{r:g}mT"
                            for i, r in enumerate(self.ranges_mt)
                            if r != self.ranges_mt[0])
            ranges = (f"ranges: mostly +/-{self.ranges_mt[0]:g} mT, "
                      f"except {odd}")
        lines = [f"VCM subtraction: {'on' if self.subtract_vcm else 'OFF'}",
                 ranges,
                 f"tare: {'set' if np.any(self.zero_mt) else 'not set'}",
                 f"gain trim: {'applied' if not np.allclose(self.gain_corr, 1) else 'none'}"]
        if self.dead:
            lines.append(f"excluded: {', '.join(sorted(self.dead))}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# analysis on top of the calibration
# --------------------------------------------------------------------------

def magnitude(b_mt):
    """(..., 16, 3) -> (..., 16) rotation-invariant |B|."""
    return np.linalg.norm(np.asarray(b_mt, float), axis=-1)


def peak_response(b_mt, baseline=None):
    """
    Per-sensor magnet response from a capture.

    Returns dict with:
      signed_pk (16,3)  largest excursion of each axis from its own baseline
      mag_pk    (16,)   peak of |B - baseline|, the number to compare across
                        sensors, because it does not care how the chip is turned
      at_index  (16,)   sample index of each sensor's peak
    """
    b = np.asarray(b_mt, float)
    base = np.median(b, axis=0) if baseline is None else np.asarray(baseline, float)
    dev = b - base[None, :, :]
    mag = np.linalg.norm(dev, axis=-1)                 # (n, 16)
    at = np.argmax(mag, axis=0)                        # (16,)
    signed = np.array([dev[at[s], s, :] for s in range(b.shape[1])])
    return {"signed_pk": signed,
            "mag_pk": mag[at, np.arange(b.shape[1])],
            "at_index": at,
            "baseline": base}


def noise_mt(b_mt):
    """(n,16,3) -> (16,3) rms noise per axis, in the same units as b_mt."""
    return np.std(np.asarray(b_mt, float), axis=0)


def channel_health(ai_by_box, vpc_by_box, hosts=ob.DEFAULT_UUTS):
    """
    Per-channel condition of the raw analogue path, before any calibration.

    This is the diagnostic that matters most on this probe right now: it is what
    exposes a railed (dead) sensor and it is what showed that VCM noise rises
    monotonically along the concentrator -- VCM carries no field, so noise on it
    is analogue pickup, i.e. cabling and grounding, not a sensor problem.
    """
    rows = []
    for bi, ai in enumerate(ai_by_box):
        ai = np.asarray(ai)
        vpc = float(vpc_by_box[bi])
        for ch in range(ai.shape[1]):
            x = ai[:, ch].astype(np.float64)
            sensor, axis = ob.channel_label(ch + 1, bi)
            std = float(x.std())
            mn, mx = float(x.min()), float(x.max())
            mean_v = float(x.mean() * vpc)
            railed = (abs(mn) >= ADC_RAIL or abs(mx) >= ADC_RAIL
                      or mn <= -32768 or mx >= 32767)
            stuck = std < 1e-9
            out_of_range = not (PLAUSIBLE_V[0] <= mean_v <= PLAUSIBLE_V[1])
            rows.append({
                "host": hosts[bi] if bi < len(hosts) else f"box{bi}",
                "box": bi, "ch": ch + 1, "sensor": sensor, "axis": axis,
                "sensor_id": int(sensor[1:]),
                "mean_counts": float(x.mean()), "mean_v": mean_v,
                "std_counts": std, "std_uv": float(std * vpc * 1e6),
                "min": mn, "max": mx, "p2p_counts": mx - mn,
                "railed": bool(railed),
                "stuck": bool(stuck),
                "out_of_range": bool(out_of_range),
                "bad": bool(railed or stuck or out_of_range),
                "is_vcm": axis == "VCM",
            })
    return rows


def bad_reason(r):
    """Short description of why channel_health() called a channel bad."""
    if r["stuck"]:
        return "stuck"
    if r["railed"]:
        return "railed"
    if r["out_of_range"]:
        return f"{r['mean_v']:+.2f} V, outside the sensor's output range"
    return ""


def health_verdict(rows):
    """Turn channel_health() rows into short per-sensor status strings."""
    out = {}
    for sid in range(1, N_SENSORS + 1):
        chans = [r for r in rows if r["sensor_id"] == sid]
        if not chans:
            out[sid] = ("unknown", "no data")
            continue
        bad = [r for r in chans if r["bad"]]
        field = [r for r in chans if not r["is_vcm"]]
        vcm = [r for r in chans if r["is_vcm"]]
        if vcm and vcm[0]["bad"]:
            # Every axis on a chip is quoted relative to that chip's own VCM.
            # If the reference itself is broken, all three axes are nonsense --
            # a chip can look "3/4 alive" and still be entirely unusable, which
            # is exactly how S16 presents on this probe.
            out[sid] = ("dead", f"VCM reference {bad_reason(vcm[0])} -- every "
                                f"axis on this chip is referenced to a broken "
                                f"zero")
        elif len(bad) >= 2:
            out[sid] = ("dead", ", ".join(f"{r['axis']} {bad_reason(r)}"
                                          for r in bad))
        elif bad:
            out[sid] = ("fault", ", ".join(f"{r['axis']} {bad_reason(r)}"
                                           for r in bad))
        elif vcm and vcm[0]["std_counts"] > 5:
            out[sid] = ("noisy", f"VCM noise {vcm[0]['std_counts']:.1f} counts "
                                 f"-- analogue pickup, not the sensor")
        elif field and max(r["std_counts"] for r in field) > 50:
            out[sid] = ("noisy", f"up to {max(r['std_counts'] for r in field):.0f} "
                                 f"counts rms on a field channel")
        else:
            out[sid] = ("ok", "")
    return out


def suggest_dead(rows):
    """Sensors that channel_health says should be excluded from statistics."""
    v = health_verdict(rows)
    return {f"S{sid}" for sid, (state, _) in v.items() if state == "dead"}


def temperatures_c(temp_raw_by_box):
    """Per-box SPAD temperature words -> (16,) degC, S1..S16."""
    out = []
    for t in temp_raw_by_box:
        out.append(np.asarray(ob.temp_c(t), float).reshape(-1)[:8])
    return np.concatenate(out) if out else np.full(N_SENSORS, np.nan)


def spread_report(mag_pk, geom=None, magnet_point=None, exponent=3.0,
                  live=None):
    """
    The 'why are the spikes different heights' answer, as data.

    Returns raw spread, and -- if a magnet position is given -- the spread that
    remains once the 1/r^exponent geometry of the tube has been divided out.
    Anything still left over after that is electrical: gain register, EEPROM
    calibration, or Hall bias current.
    """
    m = np.asarray(mag_pk, float)
    mask = np.ones(N_SENSORS, bool) if live is None else np.asarray(live, bool)
    mask = mask & np.isfinite(m) & (m > 0)
    res = {"n_responding": int(mask.sum())}
    if mask.sum() < 2:
        return res
    res["raw_spread"] = float(m[mask].max() / m[mask].min())
    res["raw_median"] = float(np.median(m[mask]))
    if geom is not None and magnet_point is not None:
        w = geom.expected_response(magnet_point, exponent)
        corr = m / np.where(w > 0, w, np.nan)
        res["geometry_spread"] = float(np.nanmax(w[mask]) / np.nanmin(w[mask]))
        res["corrected"] = corr
        res["corrected_spread"] = float(np.nanmax(corr[mask]) / np.nanmin(corr[mask]))
    return res


# --------------------------------------------------------------------------
# loading saved captures
# --------------------------------------------------------------------------

def load_capture(path):
    """Read an octobee.py capture .npz into the shape this module expects."""
    z = np.load(path, allow_pickle=True)
    hosts = [str(h) for h in z["hosts"]]
    ai = [z[f"ai_{i}"] for i in range(len(hosts))]
    vpc = [float(z[f"vpc_{i}"]) for i in range(len(hosts))]
    fs = [float(z[f"fs_hz_{i}"]) for i in range(len(hosts))]
    temp = [z[f"temp_raw_{i}"] for i in range(len(hosts))]
    return {"hosts": hosts, "ai": ai, "vpc": vpc, "fs_hz": fs, "temp_raw": temp,
            "adc_range": [str(z[f"adc_range_{i}"]) for i in range(len(hosts))]}


def main():
    import argparse
    p = argparse.ArgumentParser(description="calibration engine self-check")
    p.add_argument("capture", help="an .npz written by octobee.py capture")
    p.add_argument("--range", type=float, default=20.0, choices=sorted(ob.RANGE_TO_VPT))
    p.add_argument("--tare", action="store_true", help="tare on this capture")
    p.add_argument("--save", nargs="?", const=CONFIG_NAME,
                   help="write the resulting calibration.json")
    a = p.parse_args()

    cap = load_capture(a.capture)
    cal = Calibration(ranges_mt=np.full(N_SENSORS, a.range))
    b = cal.convert(cap["ai"], cap["vpc"], apply_zero=False)
    rows = channel_health(cap["ai"], cap["vpc"], cap["hosts"])
    verdict = health_verdict(rows)
    temps = temperatures_c(cap["temp_raw"])

    print(f"{a.capture}: {b.shape[0]} samples, {len(cap['hosts'])} boxes "
          f"@ {cap['fs_hz'][0]/1e3:g} kHz")
    print(f"{'sensor':>7} {'state':>6} {'|B| [mT]':>10} {'noise [uT rms]':>15} "
          f"{'T [C]':>7}  note")
    n = noise_mt(b)
    mag = magnitude(np.median(b, axis=0)[None])[0]
    for sid in range(1, N_SENSORS + 1):
        st, note = verdict[sid]
        print(f"{'S'+str(sid):>7} {st:>6} {mag[sid-1]:10.3f} "
              f"{n[sid-1].max()*1e3:15.1f} {temps[sid-1]:7.1f}  {note}")

    if a.tare:
        cal.tare(b)
        print(f"\ntared: max |zero| = {np.abs(cal.zero_mt).max():.3f} mT")
    cal.dead = suggest_dead(rows)
    if cal.dead:
        print(f"excluding {', '.join(sorted(cal.dead))} from summaries")
    if a.save:
        print(f"wrote {cal.save(a.save)}")


if __name__ == "__main__":
    main()
