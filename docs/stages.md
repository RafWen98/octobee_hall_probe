# The stages, and motorised field maps

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
octobee stage map --assign x=45502844 --assign z=45502854 \
                           --assign y=45538374 --invert z
octobee stage map ... --forward z          # clear it again
octobee stage map ... --origin z=250       # rig zero on a fixture datum
```

Verify it by eye rather than trusting the file: jog +z in the GUI and check the
probe goes **up**. That is a two-second check against an error that averaging,
calibration and every noise figure in this README cannot touch.

### Two things that will waste your afternoon

**Only one process may hold the stages.** They are exclusive-open FTDI/APT
devices. If the Kinesis application is running it owns all three, and the device
list here comes back *empty* — which looks exactly like a cabling fault. Close
Kinesis first. `octobee/motion/stage.py list` says so explicitly rather than letting you
chase it.

**Nothing is homed at power-on,** and an unhomed stage still reports a position.
That number is whatever was left in the counter. Absolute moves and scans refuse
to run until the axis is homed; jogging is relative and does not need it.

### Why a bigger jog step is louder, and what to do about it

Kinesis ships the LTS300C at **20 mm/s** with **20 mm/s²** of acceleration. A
trapezoidal move only reaches the velocity cap if it is long enough to ramp up
to it, so what a move actually peaks at is

`v_peak = min(v_max, √(a·d))`

At the shipped numbers that is 4.5 mm/s for a 1 mm jog, 10 mm/s at 5 mm, and
the full 20 mm/s for anything past 20 mm. That is the whole explanation for a
rig that is quiet on small steps and howls on large ones: the setting never
changed, the distance did, and above about 5 mm the axis climbs into the
motor's resonance band. It is not a fault and it is not something to fix by
jogging in smaller steps — cap the speed instead.

The profile is applied when a stage opens, so every mover — the GUI's jog
buttons, `moveto`, a field map — gets the same one:

```bash
octobee stage speed                        # what each axis is set to
octobee stage speed --vel 8 --accel 10 --save
octobee stage speed --axis z --vel 5 --save # just the vertical one
```

`speed` prints peak speed against step size, which is the number that matches
what you can hear. `--save` writes a `motion` block into `stages.json`; without
it the change lasts until the controller is power cycled. The Stages tab has
the same two boxes beside the jog buttons.

The default here is **6 mm/s, 10 mm/s²**, and that number came off the bench
rather than out of the arithmetic above. The first attempt was 8 mm/s, reasoned
from a 5 mm jog already peaking at 10 mm/s and sounding fine — and it was still
audible in use. A jog only touches its peak for an instant; a long absolute
move sits at the cap for tens of seconds, and it is the sustained tone that is
objectionable. **Brief peaks are a poor guide to what a traverse will sound
like.** 6 mm/s is quiet.

It costs time on long moves: a full 300 mm traverse goes from about 16 s at the
shipped settings to about 51 s. Homing has its own velocity (2 mm/s) and none
of this touches it.

### The speed ceiling, and why the profile is re-sent before every move

Above that setting is a hard ceiling — `MAX_VEL_MM_S`, **10 mm/s** — that
nothing in `octobee stage` will exceed. It is enforced at every door into
the module: the GUI spin box will not go higher, `speed --vel 20` is clamped
and says so, and a `stages.json` written by hand or by an older version is
clamped as it is read *and* as it is written.

A velocity setting is not a promise, though. The controller keeps its own
stored settings, and `ISC_LoadSettings` puts them back **every time anything
opens the device** — the Kinesis application, a crashed process, a power cycle.
Set the profile once at connect and an absolute move minutes later can still go
at the shipped 20 mm/s, which is exactly what a `Go` that "goes crazy" looks
like.

So `move_to()` and `move_by()` send the profile again immediately before they
command anything. One extra DLL call against a move that takes seconds, in
exchange for the speed on screen being the speed that will actually be used.

### Stopping the machine

There are two stop buttons and they answer different questions.

**EMERGENCY STOP** — the red button in the **top right of the toolbar**, on
every tab, and **Escape** does the same thing. It is enabled whether or not the
stages are connected.

**Stop moving** — on the Stages tab, beside the jog buttons. Profiled, does not
latch, nothing needs re-homing. "That is far enough."

The red one does three things, and each is deliberate:

| | |
|---|---|
| **immediate, not profiled** | The deceleration ramp is abandoned. At the default 6 mm/s and 10 mm/s² a *profiled* stop still coasts about **1.8 mm** — and if 1.8 mm did not matter, nobody would be pressing the button. |
| **it latches** | All motion, every axis, every thread, until it is reset. Stopping the axis that happens to be moving does nothing about the raster thread that is a fraction of a second from commanding the next point. |
| **it distrusts the position** | An immediate stop can lose steps. Afterwards the count no longer matches the carriage — see below. |

Reset is a separate button that only appears once latched. The stop button
never becomes the start button: someone reaching for a stop in a hurry may
well hit it twice, and the second press must not release the machine.

#### `homed` is not the same as "knows where it is"

This is the one worth internalising. The controller's homed bit means *a
homing cycle completed at some point*. It stays set no matter what happens to
the count afterwards — an immediate stop, a stall, a crash into something that
is not the limit switch, a driver fault. Every one of those leaves the bit set
and the number wrong, and an absolute move then computes its distance from a
position the stage believes and does not have.

So `octobee stage` tracks trust separately from the bit. `Stage.homed` is
the controller's claim; `Stage.position_trusted` is this module's, granted only
by a homing cycle it watched complete and withdrawn by anything that could
have cost steps. Absolute moves require both. The Stages tab shows the
difference: **NOT HOMED** against **POSITION LOST**.

Resetting the emergency stop clears the latch. It does **not** restore trust —
those axes still refuse absolute moves until they are homed again, and the
reset dialog says which.

#### Soft limits

The controller's travel limits describe the leadscrew: 0–300 mm. Everything
this rig can actually collide with — the fixture, the magnet clamp, the cable
dress — is *inside* that travel, so travel limits protect nothing that matters.

`limit_mm` in `stages.json`, per axis, in rig millimetres, is the working
envelope, and it is what every move, every scan range and every wizard pass is
checked against:

```json
"frame": { "z": { "invert": true, "limit_mm": [20.0, 250.0] } }
```

It is a restriction, never a permission — asking for a wider range than the
stage has is clamped to the travel, not obeyed.

All three axes on this rig are declared as their full `[0, 300]`, which is a
deliberate answer and not a restriction: **nothing is currently kept out of
bounds by software.** That is a real state of affairs — the head can traverse
the whole volume without meeting anything — and not the same as leaving the
key out, which is why `limit_declared` distinguishes them. `status` and the
Log say "the whole travel, declared" for an axis that has been checked, and
warn loudly for one that has not.

The moment a fixture, a magnet clamp or a shorter cable dress goes on the
bench, that stops being true and these numbers want narrowing. They are the
only thing that would stop a raster driving into it; the limit switches are at
the ends of the leadscrew, which is behind the obstruction, not in front of it.

#### Homing order

Homing drives the full travel into a hard stop, so if one axis has to retract
before the others can sweep, that has to *happen first* — not at the same
time, which is a race decided by whichever axis is slower today. `home_order`
in `stages.json` declares the sequence and the axes are homed one at a time in
it. It is set to `["z", "y", "x"]`: z is the reverse-mounted axis, so homing
it drives the head to the **top** of the working volume, out of the way, before
anything sweeps horizontally. Axes not named follow, in map order.

#### One axis in trouble stops the machine

The Stages tab polls every axis at 5 Hz for its position anyway. If one reports
a motion error, or ends a move on a hard limit switch, that trips the same
latch the button does. On a stacked rig the axis that reports the fault is not
necessarily the one about to do damage: a commanded move that did not happen
means any other axis still executing its half of a coordinated move is now
going somewhere that was only safe while all three agreed.

#### What this is not

It is a software stop, over USB, from this process. It needs this program
running, the USB link up and the controllers answering — and the failures a
real emergency stop exists for are precisely the ones where they are not. In
those the motors keep doing whatever they were last told.

**If this rig can hurt someone or destroy something irreplaceable, the stop
that matters is a mushroom head in series with the controllers' supply**
(EN ISO 13850, category 0). Nothing above substitutes for it. What is above is
worth having because most of what goes wrong on a bench rig is not a runaway —
it is a raster driving the head into a fixture that moved since the map was set
up — and for that it is exactly the right instrument.

From another terminal, when the window is not the thing running or is not
answering:

```
octobee stage estop
```

That stops the axes but cannot latch: the interlock lives in a process and that
one exits. Stop whatever commanded the move as well.

#### Closing the window while something is moving

`close()` on a stage does not stop it. The move is already in the controller
and it runs to completion whether or not anything is still listening — so
closing the window used to leave the head traversing with nothing watching it
and no stop button anywhere. The window now asks, stops every axis first, and
waits for the worker threads before releasing the devices.

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
47 µm is 47 µT — 2000× the 0.02 µT noise floor `octobee poses` works so hard
to reach. In a near-uniform field it is irrelevant. Do the arithmetic for the
magnet you are actually mapping before deciding.

### Why the scan stops at every point

Same argument as the pose capture, applied to position instead of roll: the noise
is white and falls as 1/√N, so a point you can sit on averages down in a way a
moving one cannot. Stopping also means the stage's USB position readout is good
enough — a reading with tens of milliseconds of latency is *exact* when the thing
it describes is not moving.

**The latency is real and it is measured.** `Stage.open()` calls
`ISC_StartPolling(serial, 100)`, which starts a background thread inside the
Kinesis DLL that asks the controller for status and position every 100 ms over
USB. `ISC_GetPosition` does **not** go to the wire -- it returns the DLL's cached
copy. Timed on the z stage, 2000 calls each:

```
ISC_GetPosition     median 1.20 us   p95 2.00 us   max 86.10 us
ISC_GetStatusBits   median 0.40 us   p95 0.50 us   max  6.90 us
```

A USB full-speed frame alone is 1 ms and an APT request/response is several, so
those timings can only be memory reads. The number you get is therefore 0–100 ms
old, ~50 ms on average. While moving that is `v × Δt` of position error:

| scan speed | mean error (50 ms) | worst (100 ms) |
|---|---|---|
| 20 mm/s (shipped default) | **1.0 mm** | 2.0 mm |
| 5 mm/s | 250 µm | 500 µm |
| 1 mm/s | 50 µm | 100 µm |

Against the ~47 µm uncalibrated accuracy, the default speed makes poll staleness
~20× larger than every other position error combined. Pushing it merely *down to*
47 µm needs under 0.94 mm/s; making it negligible needs under 0.1 mm/s, at which
point a 100 mm row takes 17 minutes and you have reinvented stop-and-average the
slow way.

**And it is jitter, not offset.** A fixed 50 ms lag could be subtracted. This
cannot: the poll thread free-runs on Windows, unsynchronised to anything, so each
reading's true age is random over 0–100 ms. That becomes random position error
*coherent with scan direction*, which does not average away and looks like real
field structure. Underneath it sits a second problem -- there is no common
timebase at all. Field samples are stamped by the carrier's own oscillator (694
measured 200002 Hz, 695 measured 199999 Hz) while the position number is stamped
by the Windows host clock whenever Python happened to call. Nothing relates the
two better than the poll interval, and a ~100 Hz Python loop reads position once
per 2000 field samples.

All of which collapses to zero the moment the carriage is stationary -- hence
stop-and-average. The encoder section below is the only thing that fixes it for a
moving measurement.

**Every row is traversed in the same direction.** A serpentine raster would save
travel and is the wrong choice: reversing direction costs leadscrew backlash, so
alternate rows would carry a fixed offset that looks exactly like real field
structure with the periodicity of the raster.

Set `--settle` from measurement, not hope. The controller reports "stopped" when
its motion profile ends, which is not when a cantilevered probe stops ringing.
`octobee/motion/scan.py --settle-scan z` moves the axis and watches the field stop
changing, which is the number your rig actually needs. A settle time that is too
short does not look like an error — it looks like a gradient. Read the result as
an **upper bound**: each probe is a whole `capture_pose()`, so probes land
seconds apart and a fast rig always reports "settled at the first probe".

**A failed point costs that point, not the map.** A scan at the settings above
runs for hours, and a dropped socket four hours in used to unwind out of
`run_scan` with every completed point still in a local variable. Now the point
is logged, recorded in `meta["failures"]`, and skipped; three consecutive
failures end the scan; and every exit path — including Ctrl-C and a stage
fault — still writes the `FieldMap`. `n_requested` against `n_captured` in the
sidecar says whether what you have is the whole grid.

### The encoders: measured working, 2026-08-21

**The z encoder is wired, counting, and accurate to 300 nm over a 20 mm move.**
It reports on `acq1001_695` site 6 = **ENC_Z = LW 19**, which also confirms the
site ordering that was previously only inferred: site 2 -> ENC_X, site 5 ->
ENC_Y, site 6 -> ENC_Z. `QEN:DI4` on site 6 went from `7` (all three inputs
idling high, nothing driving them) to `2` when the encoder was connected, while
sites 2 and 5 still read `7`. That knob is the quickest "is anything plugged in"
check there is.

```
sam_cnt first=1, 4 163 458 samples, no gaps
ENC_Z  first=0  last=287995  delta=287995
  motion fully bracketed: 5.81 s stationary head, 4.60 s stationary tail
  encoder +19.9997 mm | stage +20.0000 mm | disagreement -0.0003 mm
  per-sample steps: +0:3875458  +1:287997  -1:2
```

**Scale factor: 14400 counts/mm**, which is 3600 PPR x 4 quadrature over the
1 mm leadscrew pitch, exactly as the part number predicts. That is **69.4 nm per
count**. Confirmed independently across four consecutive 5 mm steps (71995,
71999, 72000, 72000 counts) and the 20 mm run above. Derive it this way rather
than by dividing a long move, which would fold the leadscrew error into the
scale factor.

The per-sample steps are only 0 and +1, so the quadrature decoder is not
aliasing, and the count sits **in the same sample as the field, latched by the
same clock**. Latency and jitter are zero by construction, not merely small.
That is the whole point, and it is what the USB readout above can never do.

**Speed range: no limit found.** A sweep of 1, 2, 3, 4, 5, 6, 8 and 12 mm/s over
10 mm each tracked the stage to 0.01-0.02 % in the forward direction, with no
sign of dropped counts even at 12 mm/s (172 800 counts/s, 1.16 ADC samples per
count).

**Reverse moves come up ~21 µm short.** Consistently: 0.0210, 0.0210, 0.0209 mm
across three trials, against ~1 µm for forward moves. That is direction-dependent
lost motion, almost certainly the controller's own backlash compensation, since
an encoder on the screw cannot see nut backlash. It is reproducible, so it is
correctable -- but do not mistake it for encoder error.

#### Two capture-timing traps

**1. Connecting to port 4210 does not start the capture immediately.** There is a
**~2-3.5 s arming delay**, and everything before the capture actually starts is
discarded. A 2 s pre-roll put the capture start *in the middle of a move* and
produced apparent count losses of 8-35 %, which looked convincingly like a
slipping coupling or a speed-dependent decoder limit. It was neither. Use a
generous pre-roll and **verify it worked**: the capture must have a stationary
head *and* a stationary tail around the motion. `enc[0]` should be 0 and the
first samples should show no movement. If `enc[0]` is already non-zero, the move
started before your capture did and the delta is truncated.

**2. Starting a capture zeroes the QEN counters.** Verified with the stage
stationary: `QEN:COUNT` sat at 85132, went to exactly 0 the instant streaming
began, and stayed there for the rest of the capture. `SamCnt` restarts at 1 by
the same event. This is sensible -- each capture's encoder words are referenced
to its own start -- but it means **`QEN:COUNT` read over the command port is not
a datum that survives a capture start.** Read position from the in-frame ENC
words, not from the knob. The knob is for diagnostics (is it connected, is it
counting), not for measurement.

Both of these explain every anomalous number seen while commissioning this: the
"12287 counts/mm", the "35 % loss at 8 mm/s", the "2.93 mm of slip". All one
artefact, none of them real.

#### What they still do not buy you

They measure the **leadscrew**, the same shaft the controller already counts to a
part in 400 000, so they add nothing to absolute accuracy -- the 47 µm is
*downstream* of the screw, in the nut's travel, where a rotary encoder on the
screw is blind by construction. They are equally blind to nut backlash, since on
a reversal the screw turns and the carriage does not. What they buy is
synchronism, and that is the one thing nothing else can provide.

They are incremental, so they need a datum: home the stage and zero the counter
at that instant. The index pulse will not do it on its own, because with a 1 mm
pitch it fires **once per millimetre**, giving 300 identical marks over the
travel. `QEN:ZC:EN` is currently `OFF` on all three channels, so the index is not
even enabled yet. Do compare encoder counts against the controller's microstep
count over a long move as a *check*: both watch the same shaft, so any real
divergence is a slipped coupling or lost steps — but read the two gotchas above
before believing a divergence.

#### Carrier 694 cannot do this at all

Sites 2, 5 and 6 on 695 each expose a full quadrature counter (`QEN:COUNT`,
`QEN:COUNT64`, `QEN:ECOUNT`, `QEN:ZCOUNT`, index-home, programmable triggers)
with phase A and B enabled. 694's site-2 OCTO reports `NCHAN=0` and has **no
`QEN:*` knobs at all**. The FPGA images say why:

```
694:  ACQ1001_TOP_09_74_32B              built 2025/09/01   "NOT in RELEASE at all"
695:  ACQ1001_TOP_09_74_ff_ff_74_74_32B  built 2026/01/29
```

695's personality declares OCTO modules at sites 2, 5 and 6
(`74_ff_ff_74_74`) -- three encoder channels. 694's declares only site 2, with
no quadrature logic, so **694's single extra longword is not an encoder counter**
and nothing plugged into that box can ever appear. The encoder was originally
wired to 694, which is exactly why it read zero. Getting encoder data on 694
means asking D-TACQ to reflash it with the 695 image.

Wiring, if you add the other two axes: `GHB38-08G3600BML5`, line-driver variant,
guide 2.3 -- red VCC (4.5-30 V), black GND, green/brown A/-A, white/grey B/-B,
yellow/orange Z/-Z. Without VCC the line driver is dead and you get exactly the
constant `DI4 = 7` that the unused channels show. Power the system down before
touching encoder connectors; the guide requires it twice.

#### Motor contamination: measured, and it is not the problem here

The standing worry was that three energised steppers a few tens of mm from a
sensor read to 20 nT would be a rotating dipole whose contamination is coherent
with position. Measured on 2026-08-21 by comparing 694 with the motors energised
against the same channels with them powered down:

| channel | motors on | motors off |
|---|---|---|
| S4 Bx | 67.0 | 86.9 |
| S8 Bz / By / Bx | 42.1 / 45.8 / 57.3 | 41.5 / 41.1 / 49.7 |
| S5 Bz | 31.3 | 29.5 |
| S1 Bz (clean reference) | 13.3 | 13.1 |

Unchanged within run-to-run scatter -- S4 Bx even measured worse with the motors
off. **The stepper drivers are not a measurable contributor** on this rig, and
the elevated channels are the chip faults catalogued in section 3. The argument
for stop-and-capture rests on the noise statistics and the position latency
above, not on motor pickup -- and with the encoder now working and synchronous,
continuous scanning is worth revisiting on its merits.

### Usage

```bash
octobee stage list                     # what is on the bus
octobee stage identify                 # wiggle each one to see which axis it is
octobee stage map --assign x=45502844 --assign z=45502854 \
                           --assign y=45538374 --invert z
octobee stage status
octobee stage home --axis x            # explicit; asks before moving
octobee stage moveby --x 5             # relative, works unhomed
octobee stage moveto --x 100 --y 50    # absolute, needs homing
octobee stage stop                     # profiled; nothing is lost
octobee stage estop                    # EMERGENCY: stop now

octobee scan --settle-scan z           # measure the real settle time
octobee scan --x 0:100:5 --y 0:100:5 --seconds 5
```

The axis map and the mounting live in `stages.json`. Nothing guesses either: a
wrong guess produces a silently transposed or mirrored coordinate frame, and
such a field map looks entirely plausible.

In the GUI, all of this is the **Stages** tab — find, assign, home, jog, and run
a field map with a progress bar and an abort. The **emergency stop** is not on
that tab: it is the red button in the top right of the window, on every tab,
and Escape does the same thing. See *Stopping the machine* above. A scan takes the carriers off the
live stream and puts them back on their own 200 kSPS clock for the duration, so
the live plot stops; press Connect afterwards to get it back. Output is
`captures/fieldmap_<time>.npz` plus a `.json` sidecar, holding both the commanded
and the reached position for every point — when those disagree, something
stalled, and having both is the difference between noticing and quietly folding a
bad point into the map.

---

---

*Part of the [OCTO-BEE documentation](../README.md).*
