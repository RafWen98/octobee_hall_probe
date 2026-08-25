# The hardware, and what it is telling us

Re-measured **2026-08-21** against both live carriers: 2 s, ~400 k samples per
box, ambient field, no magnet, stage motors powered off. Frame alignment was
verified before trusting any of it -- `sam_cnt` increments by exactly 1 and
`usec_cnt` by exactly 5 across the whole capture.

**S16 is alive.** This section used to record ch29/30/32 pinned at -32768 counts
with a faulty port-8 ribbon. That is repaired:

```
S16  ch29 Bz  mean 7240.7  std 20.41       ch31 Bx  mean 7288.5  std 22.37
     ch30 By  mean 7290.5  std 23.36       ch32 VCM mean 7290.0  std  0.74
```

All four channels read normally with healthy variance and nothing is railed.
`calibration.json` keeps `dead` empty on purpose and the GUI decides at run
time, so there is nothing to change there.

**The ribbon-pickup problem is gone.** VCM carries no field and no amplifier
gain, so its noise is pure analog-path pickup. It used to climb monotonically
with port number, reaching 12.47 counts on S8 and 13.26 on S9. It no longer
does:

```
694 VCM noise [counts]:  S1 0.51  S2 0.51  S3 0.55  S4 1.03
                         S5 0.73  S6 0.86  S7 0.70  S8 1.18
695 VCM noise [counts]:  S9 2.15  S10 0.66  S11 0.54  S12 0.58
                         S13 0.83  S14 0.52  S15 0.54  S16 0.72
```

Every channel now sits between 0.5 and 2.2 counts. Whatever the cabling fault
was, reseating fixed it. **Do not cite the old numbers to justify re-cabling.**

**A dead box looks quiet, not noisy.** When 695's eval kits lost power during
the 2026-08-21 connector work, all 32 of its channels read ~-0.02 V with std
~0.7 counts. That is *below* the SENM3Dx's own noise floor of ~13 counts, which
is the tell: a working Hall sensor cannot be that quiet. Check **VCM** -- it is
the chip's internally generated virtual ground and reads ~2.2 V whenever the
chip has power. At 0 V the chips are unpowered, whatever else the channel looks
like.

**VCM spread.** 2.2360 V (S1) down to 2.1566 V (S4). ~80 mV = ~260 counts =
about 1.2 mT of *apparent* field at the 20 mT range if you plot raw channels.
This shifts baselines, not amplitudes -- but it will wreck any absolute reading.

**ASIC temperatures look wrong.** Several read 8–14 °C, below plausible room
temperature. Either the AMUX channel mapping on the CELF differs from the guide,
or those inputs are not connected. Treat the temperature column as unverified.

**PWR_GOOD is less useful than it looks.** It is constant within a capture, but
it changed on 694 between two captures *with no reboot in between*
(`0x0ff0ffff` -> `0x0f0f0fff`) while that box's sensors were demonstrably
healthy. Do not read the individual nibbles as a per-sensor health map. The one
case where it was decisive: with 695's sensor power disconnected it read
`0x00000000` flat, against a non-zero word on the working box.

**The 1.9× gain split between the boxes is history.** It was real, it was a
register difference rather than cabling, and it is what motivated the
harmonisation below -- all 16 chips have run gain 3000 since 2026-08-19. The
`noise(field) / noise(VCM)` check in `octobee report` is still the right tool if
the two halves ever diverge again.

### Telling a bad chip from a bad cable, without unplugging anything

VCM is the control channel. It leaves the same package, runs the same 20-way
ribbon, crosses the same concentrator port and lands in the same ADC group as
that chip's Bx/By/Bz -- but carries no Hall signal. So:

* noise on the field channels **and** on that chip's VCM -> shared analog path:
  connector, ribbon, grounding. Re-cabling can fix it.
* noise on the field channels and **absent from VCM** -> generated after the
  point where VCM branches off, i.e. inside the chip. Re-cabling cannot.

Second discriminator: compare the same frequency band on *neighbouring* chips. A
real magnetic disturbance couples into everything nearby; a chip oscillating
into its own output does not.

This was validated by experiment on 2026-08-21. S15's Bx carried a 2.44 kHz
square wave -- 81.5 % of its power, std 176 counts -- with a clean VCM (0.57)
and only 0.34 % of that band on every neighbour, so the method predicted "bad
chip, not bad cable". The eval kits on 695 **ports 3 and 7 were then physically
exchanged**:

| 695 port | before the swap | after the swap |
|---|---|---|
| port 3 | Bx std 15.02, 2.44 kHz = 0.36 % | Bx std **163.80**, 2.44 kHz = **82.62 %** |
| port 7 | Bx std **176.61**, 2.44 kHz = **81.53 %** | Bx std 15.06, 2.44 kHz = 0.31 % |

The fault travelled with the chip. The port, ribbon, concentrator channel and
CELF path were all clean under a different chip. **Method confirmed** -- so the
remaining faults can be diagnosed from a capture instead of an afternoon with a
screwdriver.

### The five bad channels

Power in each fault's dominant line, relative to the faulty channel itself:

| chip | axis | dominant line | std [counts] | own VCM | neighbours |
|---|---|---|---|---|---|
| ex-S15 (see above) | Bx | 2.44 kHz | 164 | clean | 0.004 |
| S4 (694) | Bx | 77.5 kHz | 78 | 0.000 | 0.012–0.035 |
| S9 (695) | Bz | 19.8 kHz | 71 | 0.001 | 0.027–0.072 |
| S13 (695) | Bx | 98.2 kHz | 65 | 0.000 | 0.017–0.067 |
| S8 (694) | Bx | 14.3 kHz | 59 | 0.000 | 0.023–0.13 |

Clean channels sit at 13–17 counts for comparison. Every fault is absent from
its own chip's VCM and 15–60× weaker on the neighbours: **all five are chip
faults, and no connector work will fix any of them.**

**Four of the five are on Bx**, which is always `ch 4k+3`. Four independent eval
kits failing on the same axis is a pattern worth raising with the vendor, not
five coincidences.

**S8's Bz is not S8's fault.** Its dominant 79.4 kHz line is **4.3× stronger at
S4** than at S8, and is visible right across the box (S5 1.16, S6 0.39, S7 0.40,
S1–S3 ~0.25). That is a real magnetic field radiating from around S4 which the
other Hall sensors honestly detect. S4 is therefore the highest-value repair on
694 -- worst single channel *and* contaminating its neighbours. Expect S5, S6,
S7 and part of S8 to improve when it is fixed.

**S9's noisy VCM is a red herring.** Its VCM excess sits at 1.03 kHz, and S11,
S12, S13 and S15 all carry the same 1 kHz comb at 0.71–0.83 of S9's level -- a
small box-wide common-mode artefact, unrelated to S9's actual fault at 19.8 kHz.

**These numbers move ~20 % run to run.** S7 By measured 30.2 then 36.6 counts in
two back-to-back identical captures. Do not chase small changes.

> **Probe state, 2026-08-21:** 695 ports 3 and 7 are still **swapped** from the
> diagnostic above, so the software's sensor index does not match physical tube
> position. Every `octobee geometry` rotation, zero and calibration keyed by
> index is wrong until they are exchanged back. Swap them back before taking any
> calibration or field data.

### SPI audit, acq1001_694 (S1–S8)

Measured over SPI on the carrier. **Within this box the configuration is
uniform** — the chips are not set up differently from each other:

```
amplifier gain     1500  on all 8   -> +/-40 mT range, 34.65 V/T
PWM_CTRL              0  on all 8   -> LPF 100 kHz, same on every chip
CHANNEL_CTRL        127  on all 8   -> X, Z, Y and temperature all enabled
EEPROM key         0xa5  on all 8   -> factory calibration IS loaded
STATUS             0x1c  on all 8   -> no checksum error, no invalid key
chip temperature   21.4 .. 32.5 C   -> plausible, unlike the SPAD TCH values
```

The per-chip trims *do* differ, and that is correct — they are the factory
calibration compensating each Hall element:

```
DAC_X (Bx bias)  13..16  =  2.02-2.38 mA   18% spread
DAC_Z (Bz bias)  18..21  =  2.61-2.96 mA   13% spread
DAC_Y (By bias)  13..16  =  2.02-2.38 mA   18% spread
```

Those match the datasheet's nominal 2.2 mA (vertical) / 2.6 mA (horizontal)
bias. Differing values here are what make the chips *agree*, not what makes them
disagree — an earlier version of the audit flagged them as faults, which was
wrong, and the script now separates "configuration that must match" from
"calibration trim that is expected to differ".

Three real anomalies did show up, where a trim has dropped to zero while every
sibling is ~130–205:

```
SENS_Y = 0 on S4      SENS_Z = 0 on S8      SENS_Y = 0 on S8
```

`SENS` bit 7 clear means *auto linearity control* instead of *manual fine gain*,
so those three axes are in a different sensitivity-correction mode from the other
21. The fine-gain trim is only 0.05 %/LSB (a few percent overall), so this is a
small effect — but it is a genuine per-axis calibration difference.

### SPI audit, acq1001_695 (S9–S16) — and the root cause

```
S9 : gain=3000  G_CTRL_X=0  EEkey=0xa5  DAC=[11,16,12]  T=34.5 C
S10: gain=3000  G_CTRL_X=0  EEkey=0xa5  DAC=[12,16,12]  T=31.3 C
S11: gain=3000  G_CTRL_X=0  EEkey=0xa5  DAC=[12,15,12]  T=31.4 C
S12: gain=3000  G_CTRL_X=0  EEkey=0xa5  DAC=[12,16,13]  T=28.8 C
S13: gain=3000  G_CTRL_X=0  EEkey=0xa5  DAC=[12,16,13]  T=31.4 C
S14: gain=3000  G_CTRL_X=0  EEkey=0xa5  DAC=[13,17,13]  T=30.0 C
S15: gain=3000  G_CTRL_X=0  EEkey=0xa5  DAC=[13,16,13]  T=24.5 C
S16: gain=3000  G_CTRL_X=0  EEkey=0xa5  DAC=[12,17,12]  T=29.2 C
```

**The two halves of the probe are running at different gains.**

| carrier | sensors | `G_CTRL` | gain | range | sensitivity |
|---|---|---|---|---|---|
| `acq1001_694` | S1–S8 | 1 | 1500 | ±40 mT | **34.65 V/T** |
| `acq1001_695` | S9–S16 | 0 | 3000 | ±20 mT | **63 V/T** |

63 / 34.65 = **1.82×**. The same magnet at the same distance produces a spike
1.82× taller on S9–S16 than on S1–S8, purely from the gain setting. That matches
the 1.9× the noise-ratio check predicted from the data alone, before any SPI
access — two independent routes to the same number.

This is the main instrument-side answer to "why are the spikes different
heights". It is not a broken calibration: every chip has a valid EEPROM key
(`0xa5`) and sensible per-chip trims. The two boxes were simply configured on
different ranges.

**Also: S16's chip is alive.** It answers on SPI, reports gain 3000, a valid
EEPROM key and 29.2 °C. So the fault behind the railed ch29/30/32 is purely in
the **analog** path — the eval-kit-to-concentrator ribbon for port 8 — since SPI
and analog take different routes. The chip and its power are fine.

#### Fixing it

Pick one range for all 16 and apply it everywhere. The choice depends on the
largest field any sensor will see:

- **±20 mT (gain 3000)** — best resolution, but clips above 20 mT.
- **±40 mT (gain 1500)** — half the sensitivity, twice the headroom.

```bash
# make everything ±40 mT (matches the current 694 setting)
"$PUTTY/plink.exe" -ssh -batch -pw "$PW" -hostkey "$HK" root@acq1001_695 \
    'python3 /tmp/sensor_audit.py --set-gain 1500'
```

This writes registers only, so it is **lost at power cycle**. Make it permanent
by writing the same values to EEPROM and calling `activate_EEPROM_config()`, or
by applying it from `rc.user` at boot (OCTO-BEE guide Appendix B.4.2).

Until they are harmonised, tell the analysis the truth per box:

```bash
octobee report --range 40 --range 20      # 694 then 695, in --uut order
```

### Gain: all 16 now at 3000 (±20 mT), permanently — 2026-08-19

Applied and verified. **The EEPROM is the single source of truth** — there is no
boot script setting gain on either box.

Each chip's EEPROM byte `EGain_sel` (0x10E) = `0x00`, with the key (0x1FE) and
checksum (0x1FF) rewritten via `activate_EEPROM_config()`. At power-up the chip
validates key+checksum, then loads `EGain_sel` *and* the matching per-gain trim
set into its registers. Read live from S1 to prove it:

```
0x10E  EGain_sel = 0x00  ->  X=00 Z=00 Y=00  = gain 3000, selects data set 0

  set 0 (gain 3000, 0x140)  [15, 19, 15, 182, 200, 167, ...]   <- SELECTED
  set 1 (gain 1500, 0x150)  [16, 21, 16, 200, 175, 201, ...]
  live registers 0x11-0x1F  [15, 19, 15, 182, 200, 167, ...]   match = True
```

**The boot-time gain loop was removed from both carriers on 2026-08-19.** 695 had
one (`set-device-gain.py` in a `for` loop in `/mnt/local/rc.user`); 694 briefly
got one too before it was taken back out. It is gone on purpose:

- It forces only the **gain**, never the calibration trims. If an EEPROM went
  invalid the chip would boot to datasheet defaults — `SENS_*`/`OFFSET_*` at 0
  where a calibrated chip holds 128–255 — and the loop would set the gain back to
  3000, making the sensor *look* correct while its calibration was silently
  missing. It masked exactly the fault worth seeing.
- Two mechanisms meant two places to update. That is literally how the boxes came
  to differ: D-TACQ added the loop to 695 to force 3000 because its EEPROM said
  1500, and 694 was left on the EEPROM's 1500.

With no loop, a bad EEPROM shows up immediately as gain 1500 in
`onbox/gain_config.py verify`. `/mnt/local/rc.user` on both boxes now carries a
comment block explaining this. `set-device-gain.py` is still installed on both
(it is a useful manual tool) — it is just not run at boot.

Verified from the data afterwards: the gain-chain noise ratio on 694 moved
14.5 → 26.7 while 695 stayed at 28.0, so **the two boxes now agree to 1.05×**.
The 1.82× mismatch is gone. All 16 report `gain=3000`, `EGain_sel=0x00`,
`key=0xa5`, `valid=True`, and each chip has loaded its own gain-3000 trims.

**Everything is now ±20 mT / 63 V/T**, so use `--range 20` (a single value now
applies to both boxes).

#### Two things to know

**±20 mT clips.** Gain 3000 saturates above 20 mT at the sensor. A small
neodymium magnet a few mm from a chip easily exceeds that. If you see flat-topped
peaks, that is clipping, not saturation of the ADC — drop to gain 1500 (±40 mT).

**Fine-gain trims.** The EEPROM stores a separate trim set per gain, and the
gain-3000 set has more dropped `SENS_*` entries than the gain-1500 set (10 axes
of 48 vs 5). A dropped trim means that axis uses auto-linearity instead of a
manual fine-gain correction — worth ≤ ~4 % on that axis. Not a reason to avoid
gain 3000, but it is the floor on cross-calibration until you measure scale
factors yourself (§4 step 5).

#### Backups and undo

Full 256-byte EEPROM dumps of all 16 chips, taken before any write, are in
`onbox/eeprom/`. Original `rc.user` files are in `onbox/rc.user/`, and
`/mnt/local/rc.user.bak.pre-gain` on 694.

```bash
# revert one carrier's EEPROM exactly as it was
python3 /tmp/gain_config.py restore --in /tmp/eeprom_acq1001_694.json
# or just change gain again
python3 /tmp/gain_config.py set --gain 1500
python3 /tmp/gain_config.py verify --gain 1500
```

If you change gain, **change it in both places** — EEPROM *and* the `rc.user`
loop on both boxes. The rc.user loop wins at boot, so editing only the EEPROM
would silently be overridden. That is exactly how the two boxes came to differ.

One chip needed a retry: 694 sensor 7 first wrote checksum `0xbb` where `0xba`
was correct and reported `valid=False`. Re-running `activate_EEPROM_config()`
fixed it. `onbox/gain_config.py verify` is what catches this — always run it.

### A library bug to know about

`get_meas_range()` returns `'400 mT'` while `get_gain()` returns `1500`, and the
library's own `GainMeasRangeDict` maps `1500 -> '40 mT'`. The datasheet is
unambiguous (Table 22: `G_CTRL` bits 1:0 = 01 → gain 1500; Table 21: gain 1500 →
±40 mT), and the register reads `G_CTRL_* = 1`. **Trust `get_gain()`, not
`get_meas_range()`** — at ±40 mT the sensitivity is 34.65 V/T, a factor of ten
away from what the range string implies. Getting this wrong scales every field
number by 10.

---

---

*Part of the [OCTO-BEE documentation](../README.md).*
