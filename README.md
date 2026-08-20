# OCTO-BEE Hall probe — setup, live view, and calibration

Tooling that replaces the Phoebus Data Browser round-trip for the 16-sensor
3D Hall probe, plus what the hardware actually reports today.

Everything here was verified against the live hardware on 2026-08-19.

---

## 1. What the system actually is

You have **two** carriers, not one. That is why you were only ever seeing 32
channels: Phoebus was pointed at one box.

| carrier | IP | ACQ423 s/n | sensors | stream frame |
|---|---|---|---|---|
| `acq1001_694` | 192.168.1.82 | E42310297 | S1–S8 | 96 B (24 LW), 1 encoder word |
| `acq1001_695` | 192.168.1.83 | E42310298 | S9–S16 | 104 B (26 LW), 3 encoder words |

Both: ACQ423ELF, 32 ch, 16-bit packed, ±10 V, 200 kSPS, SPAD = 7 longwords.
The two boxes have **different frame layouts**, so anything that decodes the raw
stream has to read the geometry off the box rather than hard-code it. The tools
here do that.

The carriers are **not** synchronised — each free-runs on its own oscillator
(measured 200001 Hz vs 199999 Hz). The two halves of the probe drift relative to
each other by about 1 sample per 100 k. Fine for amplitude work, not for phase.

### Channel map

Per box, each SENIS eval kit takes 4 consecutive ACQ423 channels:

```
ch 4k+1 = Bz    ch 4k+2 = By    ch 4k+3 = Bx    ch 4k+4 = VCM
```

so kit 1 → ch 1–4, kit 2 → ch 5–8, … kit 8 → ch 29–32, on each box.

**VCM is not a field channel.** It is that chip's own virtual-ground reference
(~2.2 V), and it must be subtracted from that same chip's Bx/By/Bz before any
field number is quoted. The 16 chips' VCMs differ by up to ~90 mV.

This map comes from the OCTO-BEE guide §3.7. It is documentation, not
measurement — confirm it on your hardware with `octobee_idmap.py` (§4).

---

## 2. Quick start

```bash
python octobee.py info                  # live config + channel map, both boxes
python octobee_live.py                  # LIVE PLOT, all 16 sensors, one window
python octobee_cal.py --seconds 5       # health + calibration report
```

`octobee_live.py` modes:

| flag | view |
|---|---|
| *(default)* `--mode stack` | 48 field traces stacked, labelled `S1 Bz` … `S16 Bx` |
| `--mode overlay` | all traces on one pair of axes — use this to compare spike heights |
| `--mode grid` | one subplot per sensor, with a black \|B\| trace |
| `--show-vcm` | adds the 16 VCM references |
| `--range 20` | y-axis in mT instead of volts |

### Sample rate while streaming live

**Both carriers sustain full 200 kSPS indefinitely.** Measured 2026-08-19:
39.1 MB/s combined, **zero lost samples** over a 150 s run, with the pipeline lag
flat at 2.6 s / 3.1 s — constant latency, not a growing backlog. The 1 Gbps link
runs at about 31 % utilisation.

An earlier version of this file claimed a 9.8 MB/s ceiling. That came from a
5-second throughput test and was wrong: the stream **ramps over the first ~30 s**
(16.5 → 18.9 MB/s), so short measurements badly understate the steady rate.
`octobee_live.py` drops the ADC clock to 20 kSPS while it runs (`--fs`) and puts
the original clock back on exit; measured broadband noise is unchanged, and a
hand-passed magnet is a sub-Hz signal anyway.

Short offline captures at the full 200 kSPS are unaffected — the box buffers and
delivers every sample, just slower than real time. A 2 s capture from both boxes
came back with **zero** lost samples.

If a live session is killed rather than closed, the reduced clock is left in
place. Undo it with:

```bash
python octobee.py restore        # puts both boxes back to 200 kSPS
python octobee.py info           # confirm
```

### Phoebus interaction

Phoebus' *Streaming Capture* button starts a `CONTINUOUS` stream that owns the
data mover; while it runs, port 4210 gives an immediate EOF. All the tools here
stop it automatically. To go back to Phoebus, just press its start button again.

---

## 3. What the hardware is telling us right now

From a 2 s capture of both boxes, ambient field, no magnet:

**Dead sensor.** `acq1001_695` ch29/30/32 (S16 Bz, By, VCM) are pinned at
−32768 counts — negative full scale, zero variance. ch31 (Bx) reads normally.
Three of four lines dead but one alive points at the port-8 ribbon/connector on
that concentrator, not a dead chip. **Fix this before calibrating.**

**Noise rises steadily along the ribbon.** The VCM channel carries no field, so
its noise is pure analog-path pickup. Per box it climbs monotonically with port
number:

```
694 VCM noise [counts]:  S1 0.52  S2 0.54  S3 0.55  S4 1.26
                         S5 0.97  S6 2.00  S7 3.33  S8 12.47
695 worst:               S9 13.26   (S9 field channels: 136–311 counts std)
```

That is a cabling/grounding signature — longer runs and later connector
positions picking up more. It is not a sensor calibration problem, and no amount
of gain trimming will fix it. Worth reseating and checking the ribbon
grounding on the high-numbered ports.

**VCM spread.** 2.2324 V (S1) down to 2.1420 V (S8). ~90 mV = ~295 counts =
about 1.4 mT of *apparent* field at the 20 mT range if you plot raw channels.
This shifts baselines, not amplitudes — but it will wreck any absolute reading.

**ASIC temperatures look wrong.** Several read 8–14 °C, below plausible room
temperature. Either the AMUX channel mapping on the CELF differs from the guide,
or those inputs are not connected. Treat the temperature column as unverified.

**PWR_GOOD** reads `0xff000fff` (694) and `0xffffff00` (695), not `0xffffffff`.
Per the guide it only latches at boot, so it may be stale — but it is worth a
reboot to see whether it agrees with the S16 fault.

**The two boxes' chips are configured differently.** `octobee_cal.py` includes an
SPI-free check for this. The VCM channel is the control: it leaves the same
package, travels the same ribbon into the same ADC, but carries no Hall signal
and no amplifier gain. So `noise(field) / noise(VCM)` isolates the amplifier
chain with the shared cable pickup divided out. On the sensors whose VCM is
clean:

```
acq1001_694  S1 14.9   S2 15.2   S3 14.2   S5  9.3        median 14.5
acq1001_695  S10 20.9  S11 27.2  S12 26.3  S13 38.6  S14 28.4  S15 28.8   median 27.8
```

A consistent **1.9×** split between the boxes, with matching VCM noise. That is a
register difference — amplifier gain (`G_CTRL_X/Z/Y`) and/or low-pass corner
(`PWM_CTRL` bits 5:4) — not cabling. A chip at higher gain gives a proportionally
bigger spike for the same field. Sensors whose VCM is already noisy (S4, S6–S9)
cannot be assessed this way; the report says so rather than guessing.

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
    'python3 /tmp/onbox_sensor_audit.py --set-gain 1500'
```

This writes registers only, so it is **lost at power cycle**. Make it permanent
by writing the same values to EEPROM and calling `activate_EEPROM_config()`, or
by applying it from `rc.user` at boot (OCTO-BEE guide Appendix B.4.2).

Until they are harmonised, tell the analysis the truth per box:

```bash
python octobee_cal.py --range 40 --range 20      # 694 then 695, in --uut order
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
`onbox_gain_config.py verify`. `/mnt/local/rc.user` on both boxes now carries a
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
`eeprom_backup/`. Original `rc.user` files are in `rcuser_backup/`, and
`/mnt/local/rc.user.bak.pre-gain` on 694.

```bash
# revert one carrier's EEPROM exactly as it was
python3 /tmp/onbox_gain_config.py restore --in /tmp/eeprom_acq1001_694.json
# or just change gain again
python3 /tmp/onbox_gain_config.py set --gain 1500
python3 /tmp/onbox_gain_config.py verify --gain 1500
```

If you change gain, **change it in both places** — EEPROM *and* the `rc.user`
loop on both boxes. The rc.user loop wins at boot, so editing only the EEPROM
would silently be overridden. That is exactly how the two boxes came to differ.

One chip needed a retry: 694 sensor 7 first wrote checksum `0xbb` where `0xba`
was correct and reported `valid=False`. Re-running `activate_EEPROM_config()`
fixed it. `onbox_gain_config.py verify` is what catches this — always run it.

### A library bug to know about

`get_meas_range()` returns `'400 mT'` while `get_gain()` returns `1500`, and the
library's own `GainMeasRangeDict` maps `1500 -> '40 mT'`. The datasheet is
unambiguous (Table 22: `G_CTRL` bits 1:0 = 01 → gain 1500; Table 21: gain 1500 →
±40 mT), and the register reads `G_CTRL_* = 1`. **Trust `get_gain()`, not
`get_meas_range()`** — at ±40 mT the sensitivity is 34.65 V/T, a factor of ten
away from what the range string implies. Getting this wrong scales every field
number by 10.

---

## 4. Calibration procedure

### The geometry problem

The 16 sensors are mounted on a square tube, 4 per face, pointing outwards. Two
consequences dominate everything else:

1. **Every chip has a different orientation.** A chip on face 1 and a chip on
   face 3 are rotated 180° about the tube axis, so their `Bx` point in opposite
   lab directions. *Comparing a single axis across sensors is meaningless.*
   Compare `|B| = √(Bx²+By²+Bz²)`, which is rotation-invariant. The tools do.

2. **"Same distance from the probe" is not the same distance from each sensor.**
   A magnet held at a fixed point sits at very different distances from sensors
   on the near face and the far face, and at different distances from the four
   positions along a face. Field from a small magnet falls off roughly as 1/r³,
   so a 10 % distance error is a 30 % amplitude error. **This alone can produce
   the unequal spike heights you are seeing, with perfectly calibrated sensors.**

So the calibration has to present the magnet *per sensor*, normal to that
sensor's own face, at a fixed distance from that sensor's package centre. The
field-sensitive volume is at the centre of the QFN28 package, 100 × 100 × 10 µm.

### Step 0 — fix the hardware first

Repair the S16 port. Reseat the high-numbered ribbons. Re-run
`python octobee_cal.py --seconds 5` and confirm all 16 sensors show VCM noise
below ~1 count before going further. Calibrating over a cabling fault just bakes
the fault into your coefficients.

### Step 1 — make the chips identical (this is the likely culprit)

Gain, Hall bias current and EEPROM calibration status live inside the ASIC and
are reachable only over SPI from the carrier. A chip at gain 1500 gives exactly
half the response of one at 3000, for the same field.

```bash
scp onbox_sensor_audit.py root@acq1001_694:/tmp/
ssh root@acq1001_694 'python3 /tmp/onbox_sensor_audit.py'
ssh root@acq1001_695 'python3 /tmp/onbox_sensor_audit.py'
```

The root password is the one on the printed sheet that shipped with the units
(D-TACQ manual §"The root password is provided on a printed sheet with your
shipment"); both carriers use the same one. OpenSSH on this PC prompts
interactively, which does not work from a script — PuTTY is installed and takes
the password on the command line:

```bash
PUTTY="/c/Program Files/PuTTY"
HK=SHA256:51adMhUSu461QXQICNZfCpfA81hV1nomlMNQPhKaq3M    # acq1001_694 host key
"$PUTTY/pscp.exe"  -batch -pw "$PW" -hostkey "$HK" onbox_sensor_audit.py root@acq1001_694:/tmp/
"$PUTTY/plink.exe" -ssh -batch -pw "$PW" -hostkey "$HK" root@acq1001_694 'python3 /tmp/onbox_sensor_audit.py'
```

Better still, install a key once and drop the password entirely
(D-TACQ manual, "Now ssh logins are automatic, no password required"):

```bash
ssh-keygen -t ed25519            # if you have no key yet
ssh-copy-id root@acq1001_694
ssh-copy-id root@acq1001_695
```

The SPI library lives at `/usr/local/senm3dx/ThreeDHALLInterface.py` on the
carrier, not `/usr/local/CARE` as the OCTO-BEE guide implies; the audit script
puts both on `sys.path`.

The audit takes a couple of minutes per box — `verify_spi_interface(100)` does
100 SPI round trips per chip before anything is read.

Read the `!!` lines. They flag any register that differs between chips:
`gain`, `DAC_X/Z/Y` (bias current = sensitivity), `SENS_*` (fine gain), and
**`EE valid`** — if a chip's EEPROM key/checksum is invalid the factory
calibration is *not loaded* and it runs on datasheet defaults.

To harmonise gain across a box:

```bash
ssh root@acq1001_694 'python3 /tmp/onbox_sensor_audit.py --set-gain 3000'
```

That writes registers only, so it is lost at power cycle. Make it permanent by
writing the same values to EEPROM and calling `activate_EEPROM_config()`.

### Step 2 — verify the channel map

```bash
# terminal 1
ssh root@acq1001_694 'python3 /tmp/onbox_sensor_audit.py --id-sweep 4'
# terminal 2, started right after
python octobee_idmap.py --uut acq1001_694 --seconds 70
```

The sweep mutes each sensor's Bx/By/Bz in turn over SPI; the host tool reports
which ACQ423 channels went quiet, in order, and prints CONFIRMED or the measured
map. No clock sync needed — only the order matters. Repeat for `_695`.

### Step 3 — build the physical map

SPI position ≠ position on the tube. Pass a magnet slowly along one face:

```bash
python octobee_idmap.py --uut acq1001_694 --seconds 60 --magnet
```

It ranks the sensor groups by when they peaked, which is the physical order.
Record the result as a table: *tube face × position → sensor → channels*.

### Step 4 — zero-field offsets

Capture with no magnet nearby and the probe away from anything ferrous:

```bash
python octobee.py capture --seconds 10 -o zero_field.npz
python octobee_cal.py --load zero_field.npz
```

The `off Bz/By/Bx` columns are each axis' residual offset after VCM subtraction.
Store them as your offset table. For a proper zero you want either a zero-gauss
chamber, or the flip method: rotate the probe 180° and average the two readings,
which cancels the ambient field and leaves the offset.

### Step 5 — cross-calibration

With gains harmonised and offsets known, present the *same* magnet to each
sensor in turn, normal to that sensor's face, at a fixed distance — use a
machined spacer or jig so the geometry repeats to well under a millimetre.

```bash
python octobee_cal.py --seconds 20 --range 20 --plot
```

The report gives each sensor's peak `|B|` and its ratio to the median. Those
ratios are your per-sensor scale factors `k_i = B_ref / |B|_i`. After Step 1 they
should land inside a few percent; anything still outside ±10 % means that chip's
EEPROM calibration is suspect or its mounting differs.

### Step 6 — Earth-field roll calibration (the accurate way to match sensors)

Step 5 matches sensors with a magnet, and its accuracy is capped by *geometry*,
not by noise. `cross_calibrate()` has to divide out the 1/r³ weight from
`Geometry.expected_response()`, so a 2 mm error in where you think the magnet
was, at r ≈ 78 mm, becomes 3 × (2/78) ≈ **7.7 % of gain error**. No amount of
averaging helps: you are limited by how well you know the jig.

The Earth's field has no 1/r³ term to divide out. Over a head this size (the FSV
shell is 155 mm across) it is uniform to within nanotesla — its gradient is
nT/m — so **every sensor is guaranteed to be sitting in the same field vector**,
and the only thing that can make two of them disagree is the sensors. That turns
matching from a geometry problem into a noise problem, worth about an order of
magnitude. Against synthetic truth the solver matches sensors to **0.01 %**.

It is also the only field source that covers the whole head at once. A Helmholtz
pair is ~1 % uniform only inside ~0.3 R, so spanning a 77.7 mm FSV radius needs
R ≈ 260 mm coils at ~290 A-turns per mT. A coil is the right tool for absolute
scale on *one* arm tip (Step 7); it is the wrong tool for all sixteen.

#### What rolling can and cannot see

Rolling about the tube axis leaves the axial field component constant, so in
`m_i = M_i·B_tube + b_i` the third column of `M` and the offset `b` are exactly
degenerate. On this probe that is not abstract: with `mount_style: "tangential"`,
`arm_dir` is circumferential and `board_normal` is radial, so `e2 = cross(e3,e1)`
lands on the tube axis and **chip +Y is axial on all 16 sensors**. Roll alone
calibrates chip X and Z and leaves chip Y untouched — a third of the channels.

There is a second, sneakier degeneracy. Scaling the tube frame by
`T = diag(a,a,c)` maps the swept circles onto circles of the same family, so the
solver can trade transverse scale against axial scale for free; fixing |B| leaves
`a²Bt² + c²Bz² = |B|²`, one equation in two unknowns. **An end-for-end flip does
not fix this** — it sends `Bz → −Bz` and leaves `Bz²` alone. Only two genuinely
different *inclinations* do.

Both of these sit at the noise floor in the fit residual while being completely
wrong, which is why the solver reports identifiability separately from residual.

#### Survey the location first — this is not optional

The method rests on one assumption: **all 16 sensors sit in the same field
vector.** Earth's field satisfies it to nanotesla over a head this size. A room
containing steel, monitors, speakers or a stray magnet does not, and when the
assumption fails the solve does not fail loudly — it returns a fitted answer.

A single static capture cannot detect this, because each sensor's own offset
(0.1–0.6 mT here) is indistinguishable from the field it is sitting in. Two
poses 180° apart can: the offset cancels exactly in the difference and what is
left is offset-free.

```
bt_i = |pose1_i − pose2_i| / 2
```

Under a uniform field **every sensor must return the same number**, whatever its
orientation, because a magnitude is rotation-invariant. The spread between them
is the non-uniformity, measured rather than assumed. Two minutes:

```bash
python octobee_posecap.py --survey --seconds 10 --dead S16
```

| spread | verdict |
|---|---|
| ≤ 2 % | good — at or below the chips' own gain spread, so what is left to measure *is* the chips |
| 2–5 % | usable, but that number becomes the floor on inter-sensor matching |
| > 5 % | the sensors are not in the same field; their disagreement is the room, not them |

An index short of 180° scales every sensor's `bt` by the same `sin(Δ/2)`, so it
moves the median and leaves the spread alone — a sloppy survey index costs
nothing.

Measured on the bench 2026-08-20, the first location tried came back at
**83–113 %**, with a median transverse field of 56–65 µT against the ~47 µT
Earth alone would give. The four fitted roll angles were 0° / 95° / 187° / 272°,
so the V-block indexing was fine; per-sensor fit residuals were 3–20 µT against
a 0.14 µT per-pose noise floor, and the fitted response matrices came out up to
37° from orthogonal. That is what a failed uniformity assumption looks like from
the inside — not an error, just wrong numbers.

#### The procedure: three sweeps

Cradle the tube (grip the *tube*, not the arms) in something non-magnetic. A
square tube indexes in 90° jumps in a V-block, and **that is enough** — see
"four poses beat a hand roll" below; you do not need round end-collars. Keep the
cables strain-relieved so they rotate rigidly with the head: a flexing cable
moves ferrous connector shells relative to the sensors and turns a constant
hard-iron term into a varying one.

| sweep | what you do | what it buys |
|---|---|---|
| **A** | as mounted, roll ≥2 turns | transverse response |
| **B** | **reverse the tube axis** relative to the field, roll again | separates offset from axial response |
| **C** | point the tube axis a **third** way, roll again | pins transverse-vs-axial scale |

"Reverse the tube axis" does not mean you have to lift anything. The solve only
ever sees `bz = B · (tube axis)` and `bt = |B − bz·axis|`, so **carrying the
whole cradle round 180° on the bench is identical to lifting the tube out and
putting it back end-for-end** — both send the axis from north to south and both
give `bz: +19.10 → −19.10 µT`, `bt: 47.29 µT` unchanged. The transverse axes
land differently, but the roll angle is a free parameter, so nothing downstream
can tell the two apart. Turn the cradle; it is less disturbance and there is
nothing to re-clamp.

You do not need to know the roll angle, or roll evenly. All 16 sensors see the
same field at every instant, so φ(t) is solved from the data alongside
everything else.

#### Which way to point the tube

What the solver cares about is `α`, the angle between the field and the roll
plane — i.e. how much field lies along the tube axis. The three sweeps want
three different `α`. Here the local dip is ~68°, so a **horizontal** tube axis
gets its `α` purely from its compass bearing:

| tube axis, horizontal, bearing | α | axial | transverse |
|---|---|---|---|
| magnetic north (0°) | +22.0° | +19.1 µT | 47.3 µT |
| 30° off north | +18.9° | +16.6 µT | 48.2 µT |
| 45° off north | +15.4° | +13.5 µT | 49.2 µT |
| magnetic east (90°) | 0° | 0 µT | 51.0 µT |
| magnetic south (180°) | −22.0° | −19.1 µT | 47.3 µT |
| *(vertical, for comparison)* | +68.0° | +47.3 µT | 19.1 µT |

So the whole calibration can be done **without the tube ever leaving the
horizontal**, which is what a V-block on a bench actually offers. Graded on
synthetic truth, 4 poses × 20 s, 16 sensors:

| bearings used | matching | offsets | axial col. | aniso gauge |
|---|---|---|---|---|
| N only (roll and nothing else) | 0.088 % | **6.0 µT** | **no** | **no** |
| N, E | 0.061 % | **25.7 µT** | yes | yes |
| N, S | 0.063 % | 0.046 µT | yes | **no** |
| **N, S, E** | **0.041 %** | **0.041 µT** | yes | yes |
| N, S, E, NE | 0.033 % | 0.118 µT | yes | yes |

**N, S, E is the one to use.** It costs 0.005 % of matching against standing
sweep C vertically (0.036 %) and needs no lifting, no tilting and no fixture.

Two failure modes are worth knowing:

- **Never pair two bearings 90° apart and stop there.** N + E leaves both
  orientations with `bz ≥ 0`; the "flip" separated nothing and the offsets came
  back 25.7 µT wrong. `solve_roll()` now catches this specific case (every
  orientation on the same side of level) and says so, because the leverage
  figure alone did not — it scored 0.18 there but 0.67 in the closely related
  east–west case, which used to sail past the 0.6 threshold.
- **Don't rotate the whole N/S/E pattern by 45°.** That maps it onto bearings
  45/225/135, whose `|α|` are 15.4° three times over, and the anisotropy gauge
  dies. Any other rotation is fine.

**How well must you know north?** Not very. Rotating the whole pattern degrades
the offsets from 0.041 µT at 0° error to 0.068 / 0.162 / 0.315 µT at 10 / 20 /
30°, and matching does not move at all. A phone compass held a couple of metres
from the probe is ample; keep it away from the head while you record.

#### If a single bearing is all you can offer

Rolling at one bearing and never moving the cradle is a legitimate calibration —
it just buys less, and it is worth being precise about which parts you get:

| quantity | single bearing, axis ≈ north | why |
|---|---|---|
| inter-sensor matching | **0.057 %** (25 realisations, 4 poses × 60 s) | roll invariance needs nothing else |
| orientation about the tube axis | **0.03°** | ditto |
| sensor offsets `b` | **1.7 µT** median, 3.3 µT worst | back-derived, see below |
| chip X and Z response | measured | these are the transverse axes |
| **chip Y response** | **not measured** — nominal geometry | +Y is the tube axis on all 16 |
| transverse-vs-axial gauge | not measured, and common mode | harmless for matching |

The offsets deserve care because the honest description cuts both ways. Roll
leaves `M[:,2]·Bz` and `b` *exactly* degenerate, so `solve_roll()` fills the
axial column from `probe_geometry.json` and back-derives `b` from what is left.
The error in `b` is therefore (error in the nominal axial column) × 19 µT of
axial field — **a systematic floor that more averaging does not touch**:
measured 5.16 → 5.19 µT across dwells from 20 s to 120 s in one realisation.

But the alternative is not a better offset, it is *no* offset. This probe's real
offsets run 0.1–0.6 mT, and `calibration.json` currently carries zeros, so the
single-bearing solve still improves them by a **median factor of 426×** (732 µT
→ 1.7 µT). Apply it.

`--anisotropy assume_isotropic` does nothing here — it adjusts the axial gauge,
and the axial column already came from geometry — so don't bother passing it.

A V-block offers four positions and no more, so the way to buy accuracy is dwell
and repeats, not angles. Same four positions, total data is what counts:

| | matching | offsets |
|---|---|---|
| 1 turn × 20 s = 80 s | 0.113 % | 0.99 µT |
| 1 turn × 60 s = 240 s | 0.066 % | 0.99 µT |
| 2 turns × 30 s = 240 s | 0.050 % | 0.99 µT |
| 2 turns × 60 s = 480 s | **0.036 %** | 0.99 µT |

`octobee_posecap.py --turns 2` walks you round twice, prompting for the same
90° index each time; a revisited pose also shows you drift directly, since two
rows that should be identical are sitting in the same file.

In the GUI: *Calibration → 3. Earth-field roll calibration → Record sweep A / B /
C → Solve → Apply*. Or offline:

```bash
python octobee_posecal.py captures/rollsweep_A.npz \
                          captures/rollsweep_B.npz \
                          captures/rollsweep_C.npz \
    --b-earth-ut 48.7 --dead S16 --apply calibration.json
```

`--b-earth-ut` is your local total field from
[NOAA](https://ngdc.noaa.gov/geomag/calculators/magcalc.shtml) or BGS. It sets
**absolute scale only** — matching, offsets and orientation are all solved
without reference to it.

#### Four indexed poses beat a continuous hand roll

An earlier version of this file said to print end-collars so the tube rolls
continuously, on the grounds that ~10⁷ samples over many angles beats four
poses. That reasoning counted samples and ignored where they were taken:

| method | samples | alias penalty | **effective noise** |
|---|---|---|---|
| 60 s hand roll off the live stream, 20 kSPS | 1.2 M | ×3.16 | **0.235 µT** |
| 4 poses × 20 s offline capture, 200 kSPS | 16 M | ×1.00 | **0.020 µT** |

An indexed pose is *stationary*, so you can average it for as long as you like
at the full 200 kSPS, where the ADC does its own anti-aliasing (§7). A hand roll
cannot: the head is always moving, and the live stream it is recorded from runs
the clock down to 20 kSPS so the display keeps up. **The V-block wins by 11.5×.**

Four is also the right number, not merely the number a square tube gives you.
Opposite poses subtract to isolate each transverse column and add to cancel the
roll, so 90° steps are the ideal minimum set. Graded against synthetic truth at
20 s per pose with all three orientations, 4 poses match sensors to **0.038 %**
against 0.039 % for 8 and 0.041 % for 12 — the total averaging time is what
limits you, not the number of angles. More poses do sharpen the offsets
(0.048 → 0.022 µT from 4 to 8), so add them if offsets are what you are after.

**The 90° does not have to be accurate.** `solve_roll()` fits every pose's roll
angle from the data, so ±2° of indexing error moves the answer in the fourth
decimal place. What must be true is that the head is *rigid* between poses.

Record a sweep pose-by-pose with:

```bash
python octobee_posecap.py --tag A --seconds 20      # as mounted
python octobee_posecap.py --tag B --seconds 20      # tube end-for-end
python octobee_posecap.py --tag C --seconds 20      # cradle at a new azimuth
```

It prompts between poses, refuses to start if a killed live session left the ADC
clock down, stores the median of 64 sub-blocks per pose (so one person walking
past cannot poison a pose without being seen), and finishes each sweep with a
**closure** capture back at pose 1. That closure is the honest error bar on the
whole session: under 1 µT is clean, 1–5 µT is your real floor rather than the
noise figure, above 5 µT means something moved and the sweep should be redone.

What it writes is an ordinary `RollSweep`, so `octobee_posecal.py` and the GUI's
*Load sweeps* read it with no special handling.

#### Read the identifiability lines, not just the residual

```
axial column identified:     True
transverse/axial gauge:      measured
offset leverage:             1.24 (good; an end-for-end flip gives the most)
gain spread: 2.31 % peak-to-peak about the median
```

If sweep C is missing you get `NOT MEASURED`, and you can either record it or
pass `--anisotropy assume_isotropic` / tick the box in the GUI, which fixes the
gauge by assuming the median chip has equal sensitivity on all three axes. That
is fair for a monolithic factory-trimmed part, but it is an assumption and is
recorded as one in the calibration notes. **Inter-sensor matching is valid either
way** — that gauge is common mode, because `diag(a,a,c)` commutes with the
rotations about the tube axis that distinguish one face from another.

Two same-side azimuths (A+C without B) are *not* a shortcut: both have `Bz > 0`,
so the offset is poorly pinned and that error leaks into the axial column. The
solver flags this as weak offset leverage.

#### What else falls out for free

- **The sensor→face mapping.** In any pose all 16 chips see the same `B_tube`, so
  chips on different faces report vectors differing by 90° about the tube axis.
  `identify_faces()` clusters them and checks the result against
  `probe_geometry.json`, which currently records that mapping as an assumption.
  It cannot recover *slot* (position along the tube) — every slot on a face
  shares the same rotation — so keep using `octobee_idmap.py --magnet` for that.
  The mapping is also only recovered up to a global rotation of all four faces at
  once, which no amount of Earth's field can pin down.
- **A site survey.** Once the sensors are calibrated, the residual of "all 16 must
  read the same vector" *is* a 16-point map of the ambient gradient across the
  head. `ambient_uniformity()` reports it; much above ~0.5 % of |B| means there is
  ferrous structure close enough to matter and the calibration taken there is not
  trustworthy.

#### Resolution: do NOT reach for the 0–5 V ADC range

It looks like it should help — Earth's field is only ~10 counts at ±10 V — and it
does not. Measured on the bench 2026-08-19, ambient, median over the clean
sensors:

| ADC range | µT/count | noise (counts) | **noise (µT)** | after 1000× averaging |
|---|---|---|---|---|
| ±10 V | 4.84 | 16.8 | **81.5** | 2.845 µT |
| 0–5 V | 1.21 | 62.5 | **75.6** | 2.919 µT |

The LSB really does get 4× finer, and the noise in counts rises 3.7× to cancel it
exactly. **Quantisation was never the limit**: the analog noise is ~67× the
quantisation floor, and a spectrum puts 98.7 % of its power above 1 kHz — broadband
sensor and amplifier noise shaped by the SENM3Dx's 100 kHz low-pass, not pickup
and not mains (50 Hz is 0.0 % of the power). §8 covers why 100 kHz is already the
narrowest corner the part offers.

There is also a code reason to stay bipolar: `Layout.volts_per_count` is
`span/65536` with **no offset term**. That is correct for ±10 V (0 counts = 0 V),
but a unipolar 0–5 V range puts 0 V at −32768, so true volts are
`counts*vpc + 2.5`. Field readings survive — subtracting VCM cancels the missing
pedestal — but every raw channel would read 2.5 V low, and
`PLAUSIBLE_V = (-0.5, 5.5)` in `channel_health()` would call all 64 healthy
channels broken.

What *does* work is averaging: this noise is white, and falls as 1/√N almost
perfectly (81.5 → 27.4 → 8.8 → 2.8 → 1.1 µT at 1/10/100/1000/10000×). So the
lever is sampling **fast** and averaging, not quantising finely — and per §8,
sampling at 200 kSPS and averaging on the host beats dropping the ADC clock by
~3×, because then the ADC does the anti-aliasing for you. Prefer a full-rate
capture over a live-stream recording when taking sweeps.

At the ±400 mT range Earth is 0.6 counts and this whole method is unusable;
carry the calibration up with `range_transfer()` instead (park a magnet to make 15–30 mT, capture at both
gains without moving anything, take the ratio — the magnet's absolute value never
enters, only that it held still).

#### Honest limits

| quantity | how it is fixed | expected |
|---|---|---|
| inter-sensor matching | roll invariance, no 1/r³ term | **~0.01–0.2 %** |
| sensor offset `b` | flip pair, model-free | sub-µT |
| orientation `R_i` | pose solve | ~0.01–0.5° |
| absolute scale | your IGRF/WMM number | 0.2–0.3 % |

Offsets, matching and orientation are all **model-free** — they need no knowledge
of the Earth's field magnitude. Only absolute scale depends on it.

### Step 7 — absolute scale and orientation (when you need lab-frame vectors)

Step 6 gives you sensors matched to each other and a common frame, but its
absolute scale is only as good as the IGRF number you fed it. For traceable
absolute field you need a known applied field —
a Helmholtz coil large enough to swallow the tube, or a calibrated reference
magnetometer:

- **Absolute scale:** apply a known `B`, read each sensor, `k_i = B_known/|B|_i`.
- **Orientation matrix `R_i`:** with the tube in a known uniform field, each
  sensor's (Bx,By,Bz) is `R_i` applied to that field. Repeat for three
  independent field directions (or rotate the tube about its axis to three known
  angles) and solve for `R_i` per sensor. After that every sensor reports in the
  tube frame and the four faces become directly comparable.
- Log `TCH1..8` alongside — sensitivity TC is <100 ppm/°C, so a 10 °C spread is
  a 0.1 % effect, small but free to record.

### The ACQ423 range: not worth changing (earlier note corrected)

The ACQ423 is on ±10 V while the SENIS output only spans 0.125–4.380 V, so an
earlier version of this file said switching to `0-5V` would give "4× better
resolution for free". **That was wrong, and the measurements above disprove it.**

At ±10 V one ADC count is 305.2 µV = **4.84 µT**. But the sensor's own broadband
noise is ~62 µT rms, i.e. about 13 counts. The ADC step is therefore heavily
dithered, and averaging recovers resolution far *below* one count — measured
0.69 µT at 10 Hz, which is one seventh of an LSB. Quantisation noise is
LSB/√12 ≈ 1.4 µT rms against 62 µT of sensor noise: a 2 % contribution in
quadrature, and it averages down alongside everything else.

Switching to `0-5V` would shrink quantisation noise to ~0.35 µT rms, which
changes nothing you can measure. Do it if you want the headroom used sensibly,
but it is not a resolution win and it is not a priority.

---

## 5. Files

| file | runs on | purpose |
|---|---|---|
| `octobee.py` | PC | core library: UUT knobs, frame decode, capture. `info` / `capture` / `restore` CLI |
| `octobee_live.py` | PC | live plot of all 16 sensors, both boxes, one window |
| `octobee_cal.py` | PC | health + calibration report, optional PNG |
| `octobee_idmap.py` | PC | proves the channel↔sensor map (SPI sweep or magnet pass) |
| `onbox_sensor_audit.py` | **carrier** | SPI register audit of all 8 chips; `--id-sweep`, `--set-gain` |
| `octobee_gui.py` | PC | the application: live view, 3D probe head, calibration, exports |
| `probe_geometry.py` | PC | chip positions and per-chip rotation matrices on the tube |
| `probe_view3d.py` | PC | the 3D probe-head widget |
| `octobee_calibration.py` | PC | counts → tesla, zero, gain trim, pose matrix, channel health |
| `octobee_posecal.py` | PC | Earth-field roll calibration: solves per-sensor response, offsets and orientation from hand-rolled sweeps |
| `octobee_posecap.py` | PC | records a roll sweep as indexed 90° poses, one full-rate capture each — 11.5× quieter than a hand roll |
| `octobee_record.py` | PC | CSV / raw / report writers |
| `octobee_stage.py` | PC | Thorlabs LTS300C control over the Kinesis C API — no Kinesis app, no pythonnet |
| `octobee_scan.py` | PC | motorised field map: move, settle, average at full rate, repeat |
| `octobee_launch.pyw` | PC | what the desktop icon runs: no console, but startup failures still reported |
| `octobee.ico` | PC | application icon |
| `selftest.py` | PC | end-to-end verification, offline or against the hardware |

The command-line tools need only `numpy` and `matplotlib`. The GUI additionally
needs `PyQt6`, `pyqtgraph` and `PyOpenGL` — `pip install -r requirements.txt`.
Nothing else in either case: no HAPI install, no Phoebus, no EPICS.

The stage tools need no Python package at all beyond `numpy` — they call the
Kinesis C API through `ctypes`. They do need Kinesis *installed*, for the DLLs.

---

## 6. The GUI

```bash
pip install -r requirements.txt

python octobee_gui.py                       # the two carriers
python octobee_gui.py --demo                # synthetic probe, no hardware
python octobee_gui.py --replay captures/ambient_test.npz
```

### The desktop icon

There is a shortcut, **OCTO-BEE Hall Probe**, on the desktop. It runs
`octobee_launch.pyw` under `pythonw.exe`, so there is no console window.

That wrapper exists because a `.pyw` with no console fails *silently*: a missing
package, a moved checkout or a half-finished `pip` upgrade all look identical
from the desktop, which is to say they look like the icon not working. So it
catches whatever went wrong, writes `octobee_launch_error.log` next to itself,
and reports it in a native message box — native rather than Qt, because if the
thing that failed *was* PyQt6 then a Qt dialog is not available to say so.

It also pins the working directory to the checkout. Shortcuts do not reliably
set one, and `calibration.json`, `probe_geometry.json`, `stages.json` and
`captures/` are all resolved relative to it: launched from the wrong place the
GUI would come up on built-in defaults with nothing on screen to say it had.

To recreate the shortcut, or point it at a checkout somewhere else:

```powershell
$repo = 'C:\Users\3DHall\hall_claude'
$pw   = 'C:\Users\3DHall\AppData\Local\Python\bin\pythonw.exe'
$lnk  = Join-Path ([Environment]::GetFolderPath('Desktop')) 'OCTO-BEE Hall Probe.lnk'
$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
$s.TargetPath       = $pw
$s.Arguments        = '"' + (Join-Path $repo 'octobee_launch.pyw') + '"'
$s.WorkingDirectory = $repo
$s.IconLocation     = (Join-Path $repo 'octobee.ico') + ',0'
$s.Save()
```

One window, built on the same `octobee.py` decode path as the CLI tools. The
left half carries the work — live traces, the per-sensor table, calibration,
diagnostics, exports — and the right half always shows the probe head in 3D
with a peak-|B| bar chart underneath, so the state of all 16 sensors is visible
whatever tab you are on.

![The Live tab against the real carriers](docs/gui-live-hardware.png)

*Live, both carriers, no magnet. S16 is excluded automatically and dropped from
the legend; the bar chart ranks the rest by peak |B| and reproduces the known
noise pattern — S9 worst, then S15, S13 and S8.*

### Why a 3D view rather than more plots

The chips point in 16 different directions, so a stack of 48 traces cannot show
you where a field actually is. The 3D view rotates each chip's vector into the
common tube frame with the matrices from `probe_geometry.py` and draws it where
that chip physically sits. A magnet passing the probe then reads directly: which
face it went past, in which direction the field pointed, and — because every
arrow shares one scale — which sensors answered more strongly than their
neighbours.

Colour and arrow length are |B|, which is rotation-invariant and therefore the
only amplitude that can honestly be compared between chips. Excluded chips are
drawn dark red with no arrow.

![A magnet passing the probe](docs/gui-magnet-pass.png)

*A magnet travelling along the probe (`--demo`). Each sensor answers in turn,
and the arrows on the 3D model track it.*

### The calibration sequence

1. **Connect.** Drops the ADC clock to 20 kSPS so the link keeps up, and puts
   the original `clkdiv` back on disconnect. It waits until *both* carriers are
   actually delivering before reporting success.
2. **Zero (tare)** with no magnet near the probe. Stores each axis of each chip's
   ambient reading as its zero point.
3. **Magnet pass.** Start it, move the magnet along the probe, stop it. The peak
   |B| per sensor is recorded against the baseline from when the pass started.
4. If the magnet was not equidistant from every chip — it never is — tick
   **use geometry weighting** and enter its position. That divides out the
   1/r³ distance term, which on this tube is worth a factor of ~200 on its own.
   Whatever spread survives that is electrical: gain register, EEPROM
   calibration, or Hall bias current.
5. **Apply gain trim** to equalise what is left, then **Save calibration**.

![The Calibration tab](docs/gui-calibration.png)

Each sensor's measurement range is set per row in the Sensors tab rather than
globally, because the range is a per-chip register and the two halves of this
probe genuinely have differed. As of the 2026-08-19 harmonisation (section 3)
all 16 run gain 3000, +/-20 mT, 63 V/T, and the shipped `calibration.json`
records exactly that -- re-read live off both carriers with:

```bash
ssh root@acq1001_694 'python3 /tmp/onbox_gain_config.py show'
ssh root@acq1001_695 'python3 /tmp/onbox_gain_config.py show'
```

Before that harmonisation S1-S8 sat at gain 1500 (+/-40 mT, 34.65 V/T), which
made the same magnet read 1.82x taller on one half than the other. Keep the file
and the registers in step: **if the gain is ever changed, change the ranges in
the same commit.** A wrong range is invisible -- a uniformly rescaled half looks
entirely normal on screen.

If `calibration.json` is missing the GUI falls back to +/-20 mT for everything
and says so in the Log. That default happens to match the probe today, but
nothing checks it at run time.

### Data output

| what | format | rate |
|---|---|---|
| **Record** → calibrated CSV | millitesla, one column per axis plus \|B\|, with a provenance header | the output rate, default 500 Hz |
| **Record** → raw | flat `int16` `.bin` + `.json` sidecar, wire order, nothing subtracted | the full stream rate |
| **Snapshot** | `.npz`, lossless | the carriers' own 200 kSPS |

The raw route exists so that a capture taken today survives a later correction
to the channel map, the VCM handling or the gain registers. The snapshot puts
the carriers' clock back to full rate first, takes the capture, and leaves them
there — a short capture is buffered and delivered complete even though a
sustained one could not be.

Tick **rotate into the common tube frame** to get components that can be
compared between sensors. Unticked, each chip's Bx/By/Bz are its own and only
|B| is comparable.

### Diagnostics stay on screen

The per-sensor table and the Diagnostics tab report all 64 raw channels, and
sensors whose channels are railed or stuck are excluded from every statistic,
scale and export automatically. A chip whose **VCM reference** is broken counts
as dead even when its other channels look plausible, because every axis it
reports is quoted relative to that reference — which is exactly how S16
presents on this probe.

Read the VCM rows first when hunting noise. VCM carries no field, so noise on it
is analogue pickup in the cabling and grounding rather than a sensor problem.

### Where the millitesla come from

The ADC measures volts; nothing measures tesla. The live view says mT because
the chain below is applied, and it is worth knowing which steps are measured
and which are assumed. Worked through on one real channel, S1 Bx:

| step | value | measured or assumed |
|---|---|---|
| raw ADC counts | Bx 7300.4, VCM 7313.7 | measured |
| x volts/count = 20 V / 65536 = 305.176 uV | 2.227914 V, 2.231977 V | the ADC range is read off the box at run time |
| - that chip's own VCM | **-4.0637 mV** | measured; this is what the chip is actually saying |
| / sensitivity 63 V/T | **-0.0645 mT** | **assumed** -- datasheet nominal for gain 3000 |
| x gain trim, - tare | unchanged unless you set them | yours |

So everything up to the differential millivolts is measurement. The last step
divides by a **datasheet nominal sensitivity**, picked per sensor from the
amplifier gain that was actually read out of each chip's `G_CTRL` register over
SPI -- currently 63 V/T on all 16, every chip reporting `G_CTRL = [0,0,0]`,
`EGain_sel = 0x00` and a valid EEPROM key.

That makes the mT scale traceable to the datasheet and to a register read, not
to a reference magnet. Nothing here has been calibrated against a known field.
In practice:

- **relative** comparisons are sound -- between axes of one chip, and between
  chips once the magnet-pass gain trim has equalised them;
- **absolute** accuracy is datasheet nominal, so treat it as a few percent
  rather than metrology until someone puts a known field on it;
- a wrong range setting is **invisible**: it simply rescales the affected
  sensors, and nothing on screen looks wrong. That is why `calibration.json`
  carries the audited registers, and why the selftest asserts them.

A corollary worth remembering: **a raw capture stores counts, not field.**
Reading one back needs the gain the chips were running at the time, and that
lives in a register rather than in the data. Any `.npz` or `.bin` taken before
the 2026-08-19 harmonisation is 1.82x wrong on S1-S8 if converted with today's
settings. The `.bin` sidecar now stamps `ranges_mt` and `volts_per_tesla` into
itself for exactly this reason; the `.npz` snapshot format does not, so date
those by hand.

The **in** selector next to the plot mode walks the chain back so you can look
at any of it directly: `mT`, `uT`, `mV (chip output)` -- the differential volts
above, with VCM subtracted -- or `ADC counts`, what came off the wire. The
factor is per sensor, because the range is a per-chip setting and the halves
have run different gains before
and one field is therefore not one voltage. The peak bars stay in mT on purpose
for the same reason: equal millivolts on the two halves are not equal fields.

### The probe geometry

The chips are not on the tube: each rides at the tip of a 92 mm SENIS eval-kit
PCB standing off radially, so its field sensitive volume sits **89 mm clear of
the face** it is bolted to (3 mm in from the board's tip, 0.55 mm above the
board surface). On the 1 inch (25.4 mm) tube that puts every chip 101.7 mm from
the tube axis and 203 mm from the chip opposite it.

The boards lie perpendicular to the tube axis -- shelves, not fins -- and the
mounting plates repeat every 33 mm along each face (30 mm plate, ~3 mm gap),
four to a face. So the head is four rings of four, and it is wider than it is
long.

That standoff is the single biggest geometric fact about this instrument. A
magnet 20 mm off one chip is ~220 mm from the far side, and 1/r^3 turns that
into a **~1600x** difference in what they read from the same magnet. Any
comparison of spike heights has to divide it out before blaming a register --
which is what *use geometry weighting* on the Calibration tab does.

### The geometry file is an assumption

`probe_geometry.json` holds the tube and arm dimensions and which sensor sits on
which face, and carries that warning in its own `notes` field. The arm
dimensions come from the vendor drawing and are solid; the tube width, the
mounting-plate pitch, the board orientation and above all the sensor-to-face
mapping are not yet confirmed. **The face assignment has not been verified on this hardware.** Only the
3D view's arrangement and the tube-frame rotation depend on it — |B|, the health
diagnostics and the raw exports do not. Fix it with `octobee_idmap.py` and a
slow magnet pass along one face, then press *Reload geometry*.

### When it feels slow: `--profile`

```bash
python octobee_gui.py --profile
```

Adds a **Profile** tab that times every stage separately and keeps a rolling
5 second window. It exists because "the app goes sluggish once data arrives"
has several quite different causes and they need different fixes, and because a
normal Python profiler sees only half of them -- the expensive half often
happens inside Qt's paint, after our code has returned.

It reports:

- each acquisition stage (socket read, counts to tesla, decimate, file writes)
- each drawing stage, *and* the repaints themselves: `GL paint (probe head)`
  and `Qt paint (live plot)` are hooked directly
- **event loop lag** -- how late a timer that asked for 100 ms actually fired.
  This is the honest measure of "is it frozen" as opposed to merely busy
- the OpenGL vendor/renderer strings. A renderer of `GDI Generic`,
  `Microsoft Basic Render Driver` or `llvmpipe` means there is no GPU
  acceleration and every repaint of the probe head is being done on the CPU;
  the app says so in the Log if it sees one

How to read it:

| what is at the top | what to do |
|---|---|
| `GL paint (probe head)` | untick **3D**, or lower **refresh** |
| `Qt paint (live plot)` | shorten **window**, or lower **output rate** |
| `counts -> tesla` | lower the **stream rate** |
| reader queue growing | acquisition is falling behind; recordings will have holes |
| lag large, every row small | something outside this list is blocking |

#### The one that mattered

A live plot repaint was taking **4.6 seconds** on the bench PC and 39 seconds on
the slower machine here, while the code asking for it took 5 ms. It was not the
volume of data. Reproduced and bisected:

| setting | ms per repaint |
|---|---|
| pen width **1.6**, antialiasing on | **39 172** |
| pen width **1** | **74** |
| + antialiasing off | 45 |
| + decimate to the screen | 22 |

Qt strokes a cosmetic pen of non-integer width through a completely different
and vastly slower path. Same data, same point count, 500x. So the live plot now
uses integer pen widths, antialiasing off, and hands each curve about one
min/max pair per four pixels instead of the whole buffer -- which makes the
repaint cost depend on the width of the window in pixels rather than on the
length of the buffer or the output rate.

Note the decimation keeps each bin's minimum AND maximum rather than taking
every Nth sample. A plain stride is cheaper still, but it drops a magnet spike
that happens to fall between the samples it keeps; min/max renders that spike
at full height. The same reasoning applies upstream: the 20 kSPS stream is
block-*averaged* down to the output rate, not strided, because averaging also
suppresses noise instead of aliasing it.

Two more findings, both now fixed:

- pyqtgraph's downsampling has to be set **on the curves**, not just on the
  plot. Without it Qt was rasterising 16 traces x 10 000 points on every
  repaint -- 54 ms on average and 236 ms at worst, the single most expensive
  thing in the application, and invisible to any timing of our own code.
- the synthetic `--demo` source was rebuilding the sensor positions and all 16
  rotation matrices per sample, so profiling `--demo` mostly measured the demo.
  `Geometry.rotations()` is now cached and the demo is vectorised.

For reference, a healthy run on the bench machine (Intel integrated graphics,
both carriers live at 20 kSPS): event loop lag mean 10 ms / worst 28 ms, no
single stage above 6%.

### Verifying it still works

```bash
python selftest.py                     # synthetic probe, no hardware
python selftest.py --replay captures/ambient_test.npz
python selftest.py --live              # against the real carriers
```

It drives the whole pipeline and then reads the written files back to check the
numbers, rather than checking that files merely exist. The live run also asserts
that no blocks were dropped and that the carriers are left at the clock they
were found at.

---

## 7. How sensitive is it really?

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

## 8. Long continuous logging, and the data rate

**You do not need to reduce anything.** Both carriers stream full 200 kSPS
continuously, indefinitely, with zero sample loss. Long continuous measurement at
full bandwidth and full resolution is exactly what this system does.

### Measured, both boxes together

| test | duration | combined rate | gaps | lost |
|---|---|---|---|---|
| 100 kSPS | 90 s | 19.3 MB/s | 0 | 0 |
| 150 kSPS | 60 s | 28.8 MB/s | 0 | 0 |
| 200 kSPS | 180 s | 39.3 MB/s | 0 | 0 |
| 200 kSPS + host decimation to disk | 150 s | 39.1 MB/s | 0 | 0 |

In the 180 s run the lag held flat at 2.78 s (694) and 3.06 s (695) from the
20 s mark onward. Flat lag means keeping up; a growing lag would mean the box's
512 MB buffer pool was filling.

### The shape to use for long logs

Receive at 200 kSPS (the ADC does your anti-aliasing), decimate on the host,
write the decimated stream. Measured over 150 s with both boxes:

```
200 kSPS in  ->  host boxcar average  ->  1 kHz float32 out
   0 gaps, 0 lost, lag flat at 2.6 s / 3.1 s
   453 MB/hour per box  =  10.9 GB/day per box, 21.7 GB/day for both
```

Output size scales with the rate you choose, and the noise improves as √decim:

| output rate | per box | both boxes | noise floor |
|---|---|---|---|
| 1 kHz | 10.9 GB/day | 21.7 GB/day | ~4.9 µT |
| 100 Hz | 1.1 GB/day | 2.2 GB/day | ~1.5 µT |
| 10 Hz | 0.11 GB/day | 0.22 GB/day | ~0.7 µT |
| 1 Hz | 0.011 GB/day | 0.022 GB/day | ~1 µT (drift floor, §7) |

So a month of continuous 10 Hz logging from all 16 sensors is about 6.6 GB. The
binding constraint is host disk, not the instrument.

### If you *do* lower the clock, it costs you

The SENM3Dx analog low-pass is fixed at 100 kHz (`PWM_CTRL` bits 5:4, already at
its narrowest), so sampling below 200 kSPS folds 0–100 kHz into 0–fs/2 and noise
density rises by `sqrt(100kHz/(fs/2))`:

| rate | stream/box | penalty | measured |
|---|---|---|---|
| 200 kSPS | 19.2 MB/s | 1.00× | baseline |
| 50 kSPS | 4.8 MB/s | 2.00× | — |
| 20 kSPS | 1.9 MB/s | 3.16× | **3.1×** |

Note the penalty largely **washes out below ~1 Hz**, where offset drift dominates
instead of white noise: at 1 Hz, 20 kSPS measured 0.75 µT against ~0.7 µT at full
rate. So for very slow logging the aliasing barely matters — but there is no
reason to accept it, since full rate streams fine.

```bash
python octobee.py rate                    # report rate, load, aliasing cost
python octobee.py rate --fs 50000         # set (rarely needed)
python octobee.py restore                 # back to 200 kSPS
```

On the carrier: `/usr/local/CARE/set-sample-rate.sh [hz]`. In the GUI, the
**stream rate** dropdown now defaults to "leave the box alone".

### The FPGA oversampling filter is still missing

D-TACQ's `nacc` accumulate/decimate (manual §13.1) would decimate in the FPGA
with no aliasing and √N less noise. It is inert here: `NACC:DISA` was `TRUE`, and
with it `FALSE` and `nacc=8`/`nacc=32` set, the knob echoes `8,3,0,1`/`32,5,0,1`
while the output rate stays 200 kSPS and the noise stays 63 µT. The FPGA is a
custom `ACQ1001_TOP_09_74_32B-OCTOBEE` bitstream, so the block is probably not in
this build. Worth asking D-TACQ for — but it is now a nice-to-have, not a
blocker, since full-rate streaming works.

### Onboard capture as an alternative

`nbuffers=512 × bufferlen=1 MB` = **512 MB** of dedicated capture RAM, i.e. ~26 s
gapless at 200 kSPS (96 B/sample) with no network involved. Useful for
triggered bursts. Not needed for continuous work. Not yet wired into these tools.

---

## 9. The Thorlabs stages, and motorised field maps

Verified against the hardware on 2026-08-20.

### What is on the bench

Three **LTS300C** long-travel stages, on USB as APT devices:

| serial | axis | mounting | travel | device units |
|---|---|---|---|---|
| 45502844 | **x** | forward | 0–300 mm | 409 600 du/mm |
| 45502854 | **z** | **reversed** | 0–300 mm | 409 600 du/mm |
| 45538374 | **y** | forward | 0–300 mm | 409 600 du/mm |

This is the standard map, recorded in `stages.json`.

The stages report `stepsPerRev=200`, `gearBoxRatio=1.0`, `pitch=1.0 mm`, so one
leadscrew revolution is exactly 1 mm and the controller counts 2048 microsteps
per full step. **There is no linear scale on the carriage** — the position you
read is where the controller counted itself to, not where the platform is.

### z is mounted backwards, and that is handled

The z stage is bolted on in reverse, so **its limit switch is at the top of the
working volume**. Homing parks it at rig z = 300 mm, not 0. In cube terms, the
position it homes to is the top-left corner; rig zero is the bottom-left one.

Left uncorrected this is the nastiest class of bug in the whole system, because
it does not fail — it produces a field map mirrored along one axis that looks
completely reasonable and is wrong by the height of the volume.

So the mounting is declared once, in `stages.json`, and everything public speaks
**rig** millimetres:

```json
{"axes":  {"x": "45502844", "z": "45502854", "y": "45538374"},
 "frame": {"z": {"invert": true}}}
```

    device = origin + sign × rig        sign = −1 when inverted
    origin = the far end of travel when inverted, 0 when not

so rig z=0 drives the stage to device 300, rig z=300 drives it to device 0, and
150 is its own mirror image. Relative moves flip too, so jogging +z always goes
up. `position_dev_mm` keeps the raw device number for when you need to reconcile
against the controller's own display — on a reversed axis the two disagree, and
that is alarming until you know why. The Stages tab shows both, and any field
map records the frame per axis in its `.json` sidecar, so a map stays readable
years later without this README.

To change it — if a bracket gets remounted, or if the reversed axis turns out to
be a different one:

```bash
python octobee_stage.py map --assign x=45502844 --assign z=45502854 \
                           --assign y=45538374 --invert z
python octobee_stage.py map ... --forward z          # clear it again
python octobee_stage.py map ... --origin z=250       # rig zero on a fixture datum
```

Verify it by eye rather than trusting the file: jog +z in the GUI and check the
probe goes **up**. That is a two-second check against an error that averaging,
calibration and every noise figure in this README cannot touch.

### Two things that will waste your afternoon

**Only one process may hold the stages.** They are exclusive-open FTDI/APT
devices. If the Kinesis application is running it owns all three, and the device
list here comes back *empty* — which looks exactly like a cabling fault. Close
Kinesis first. `octobee_stage.py list` says so explicitly rather than letting you
chase it.

**Nothing is homed at power-on,** and an unhomed stage still reports a position.
That number is whatever was left in the counter. Absolute moves and scans refuse
to run until the axis is homed; jogging is relative and does not need it.

### Accuracy: repeatability is not accuracy

| | |
|---|---|
| repeatability (returning to a point) | micrometres — excellent |
| absolute accuracy, no calibration file | **~47 µm** |
| absolute accuracy, with Thorlabs' per-serial calibration file | **<±5 µm** |

**None of the three stages has a calibration file loaded** (checked via
`ISC_GetCalibrationFile`, 2026-08-20). Thorlabs supply them free per serial
number; they map that individual leadscrew's error and the controller applies it
internally from then on. Install with `Stage.set_calibration_file()`.

Whether 47 µm matters is a gradient question, not a preference. A position error
*d* appears in the data as a field error |∇B|·*d*. Against a 1 mT/mm gradient,
47 µm is 47 µT — 2000× the 0.02 µT noise floor `octobee_posecap.py` works so hard
to reach. In a near-uniform field it is irrelevant. Do the arithmetic for the
magnet you are actually mapping before deciding.

### Why the scan stops at every point

Same argument as the pose capture, applied to position instead of roll: the noise
is white and falls as 1/√N, so a point you can sit on averages down in a way a
moving one cannot. Stopping also means the stage's USB position readout is good
enough — a reading with tens of milliseconds of latency is *exact* when the thing
it describes is not moving.

**Every row is traversed in the same direction.** A serpentine raster would save
travel and is the wrong choice: reversing direction costs leadscrew backlash, so
alternate rows would carry a fixed offset that looks exactly like real field
structure with the periodicity of the raster.

Set `--settle` from measurement, not hope. The controller reports "stopped" when
its motion profile ends, which is not when a cantilevered probe stops ringing.
`octobee_scan.py --settle-scan z` moves the axis and watches the field stop
changing, which is the number your rig actually needs. A settle time that is too
short does not look like an error — it looks like a gradient.

### Would the spare rotary encoders help?

They measure the **leadscrew**, which is the same shaft the controller already
counts to a part in 400 000. So they add nothing to absolute accuracy — the 47 µm
is *downstream* of the screw, in the nut's travel, and a rotary encoder on the
screw is blind to it by construction. They are also blind to backlash, since on a
reversal the screw turns and the carriage does not.

What they do buy is the one thing nothing else can: the ACQ400 samples encoder
words **in the data frame at the ADC clock**, so position and field become
synchronous. That is required for measuring *while moving*, and useless
otherwise. Carrier 695 already carries 3 encoder words per frame; 694 carries 1.

If you wire them up: they are incremental, so they need a datum — home the stage
and zero the counter at that instant. The index pulse will not do it on its own,
because with a 1 mm pitch it fires **once per millimetre**, giving 300 identical
marks over the travel. Derive counts/mm from CPR × 4 ÷ pitch rather than dividing
a 300 mm move, which would fold the leadscrew error into the scale factor. Do
compare encoder counts against the controller's microstep count over a long move
as a *check*: both watch the same shaft, so any divergence is a slipped coupling
or lost steps.

Before committing to continuous scanning, measure what the **motors** do to the
probe. Three energised steppers commutating a few tens of mm from a sensor you
are reading to 20 nT is a rotating magnetic dipole with switching drive current,
and that contamination is coherent with position — it will look like real field
structure and no averaging removes it. Park the probe, run an axis at scan speed,
and see. This is a stronger argument for stop-and-capture than the noise
statistics are.

### Usage

```bash
python octobee_stage.py list                     # what is on the bus
python octobee_stage.py identify                 # wiggle each one to see which axis it is
python octobee_stage.py map --assign x=45502844 --assign z=45502854 \
                           --assign y=45538374 --invert z
python octobee_stage.py status
python octobee_stage.py home --axis x            # explicit; asks before moving
python octobee_stage.py moveby --x 5             # relative, works unhomed
python octobee_stage.py moveto --x 100 --y 50    # absolute, needs homing

python octobee_scan.py --settle-scan z           # measure the real settle time
python octobee_scan.py --x 0:100:5 --y 0:100:5 --seconds 5
```

The axis map and the mounting live in `stages.json`. Nothing guesses either: a
wrong guess produces a silently transposed or mirrored coordinate frame, and
such a field map looks entirely plausible.

In the GUI, all of this is the **Stages** tab — find, assign, home, jog, and run
a field map with a progress bar and an abort. A scan takes the carriers off the
live stream and puts them back on their own 200 kSPS clock for the duration, so
the live plot stops; press Connect afterwards to get it back. Output is
`captures/fieldmap_<time>.npz` plus a `.json` sidecar, holding both the commanded
and the reached position for every point — when those disagree, something
stalled, and having both is the difference between noticing and quietly folding a
bad point into the map.
