# How sensitive is it really?

Two different questions, often conflated.

### Sensitivity — already maxed out

Sensitivity is the V/T gain, and it is a setting with exactly four values:

| gain | range | sensitivity | notes |
|---|---|---|---|
| **3000** | **±20 mT** | **63 V/T** | **current setting — the most sensitive available** |
| 1500 | ±40 mT | 34.65 V/T | |
| 150 | ±400 mT | 3.78 V/T | |
| 15 | ±4000 mT | 0.378 V/T | |

There is nothing above 3000. You are already at the most sensitive range the
part offers; every other option is coarser.

### Resolution — yes, single-digit µT, set by noise not by gain

Noise is white above ~1 Hz, so it scales as √bandwidth: every 100× of averaging
buys 10× resolution. Measured on this hardware at gain 3000, clean sensors,
using an Allan-style successive-difference estimator (immune to slow drift):

| averaging | S1 | S2 | S11 | S14 | datasheet (0.20 µT/√Hz) |
|---|---|---|---|---|---|
| none, 100 kHz BW | 62.4 | 65.0 | 67.6 | 70.4 | 63.2 |
| 1 kHz | 4.9 | 5.4 | 5.3 | 5.1 | 6.3 |
| 100 Hz | 1.50 | 1.85 | 1.65 | 1.67 | 2.0 |
| 10 Hz | 0.69 | 0.71 | 0.79 | 0.62 | 0.63 |

All µT rms. **Your clean sensors track the datasheet almost exactly** — 62.4
measured against 63.2 predicted at full bandwidth. Nothing is wrong with them.

### Where it stops: ~1 µT

From a 60 s continuous capture, extending below 1 Hz:

| averaging | S1 Bz | S2 Bz | S5 Bz | S8 Bz (bad port) |
|---|---|---|---|---|
| 10 Hz | 1.39 | 1.65 | 3.39 | 7.99 |
| 1 Hz | 0.75 | 0.69 | 0.74 | 2.73 |
| 0.2 Hz | 0.98 | 0.64 | 0.38 | 1.38 |

The curve **flattens around 0.5–1 µT below ~1 Hz** and stops improving. That is
the offset fluctuation and drift floor, datasheet Table 9: 1.10–1.30 µT rms at
the ±20 mT range over 100 s. Averaging longer than about a second buys nothing.

So: **~1 µT per axis per sensor is the practical floor**, which is exactly the
"High magnetic field resolution: 1 µT" on page 1 of the datasheet. To do better
you would need to modulate the field and lock in, or difference two sensors to
reject common drift.

### Useful scale references

- Earth's field is 25–65 µT. At 1 µT you resolve ~2 % of it, and you *will* see
  it — rotating the probe changes every reading by tens of µT.
- 1 µT ≈ 1/20000 of the ±20 mT full scale, i.e. about 14.3 effective bits.
- Sensitivity tempco is <100 ppm/°C, so a 10 °C swing is 0.1 % of reading —
  20 µT at full scale. Above 1 µT resolution, temperature matters; log the chip
  temperatures with the readings.

### What actually limits you today

1. **Bandwidth.** At the raw 200 kSPS you see ~62 µT rms. Decide the bandwidth
   you actually need and average down to it — this is the whole game.
2. **Two bad ports.** S8 and S9 sit 3–4× worse than their neighbours at every
   averaging time (2.7 µT vs 0.7 µT at 1 Hz). That is cabling, not silicon.
3. **S16's open analog lines** — no useful field data at all until fixed.
4. Not the ADC. See the corrected note in §4.

---

---

*Part of the [OCTO-BEE documentation](../README.md).*
