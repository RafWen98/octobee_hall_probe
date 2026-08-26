# The application

```bash
pip install -r requirements.txt

python octobee/gui/window.py                       # the two carriers
python octobee/gui/window.py --demo                # synthetic probe, no hardware
python octobee/gui/window.py --replay captures/ambient_test.npz
```

### It connects itself when it opens

The window starts connecting as soon as it is on screen — carriers and stages
both, the same as pressing Connect. `--no-connect` starts it disconnected, for
when you only want to read a saved capture or edit a calibration on a bench
with the hardware switched off.

If the automatic attempt fails it says so in the Log and stops there: no dialog
to dismiss. Arriving at this program with the carriers off is a normal thing to
do, and a modal on every launch would only teach you to dismiss modals. Connect
stays enabled and will try again.

### Connect brings up the whole rig

The probe and the stages are one instrument — the carriers read the field, the
stages say where it was read — so **Connect** opens both, and **Disconnect**
closes both. The stages come from the axis map in `stages.json`; if that has no
map yet, or a stage in it is not on the bus, the probe connects anyway and the
Log says what was missing. A bench with no stages plugged in behaves exactly as
it did before.

Anything that can go wrong on the motion side is caught and logged rather than
raised: a stale USB handle or a missing Kinesis install must not stop the probe
half of the window from coming up.

### Homing, and why the window asks

An unhomed stage still reports a position, and that number is **whatever was
left in its counter**. It looks exactly like a real coordinate. So the moment
the stages come up, and only then, the window asks whether to reference them —
before anyone has read a number off the screen and believed it.

It asks rather than simply doing it because homing drives each carriage into
its limit switch across the whole travel, with the probe head and its cabling
mounted. Only the person in the room can say that is clear. Until an axis is
homed, absolute moves and field maps are refused; jogging is relative and works
either way.

### The Help tab

Searchable documentation, indexed from **this README** at startup, plus a
handful of topics about the window itself (shown in blue). Type a few words —
`homing`, `roll sweep`, `why is it loud`, `VCM` — and matching sections are
ranked with heading hits first.

Indexing the README rather than writing separate help text is deliberate: the
reasoning already lives here, and a second copy in a help pane would be wrong
within a month because nobody proof-reads help text against a README. The
consequence is that the help is only as good as this file, which is the right
incentive — a topic that is hard to find in the window is a heading that needs
rewriting here.

### The desktop icon

There is a shortcut, **OCTO-BEE Hall Probe**, on the desktop. It runs
`octobee_launch.pyw` under `pythonw.exe`, so there is no console window.

That wrapper exists because a `.pyw` with no console fails *silently*: a missing
package, a moved checkout or a half-finished `pip` upgrade all look identical
from the desktop, which is to say they look like the icon not working. So it
catches whatever went wrong, writes `octobee_launch_error.log` next to itself,
and reports it in a native message box — native rather than Qt, because if the
thing that failed *was* PyQt6 then a Qt dialog is not available to say so.

It used to pin the working directory to the checkout as well, and no longer
needs to. `calibration.json`, `probe_geometry.json`, `stages.json` and
`captures/` were once resolved relative to the CWD, so a shortcut that set the
wrong one — and shortcuts do not reliably set any — brought the GUI up on
built-in defaults with nothing on screen to say it had. The package locates its
own configuration now (see §5), so every entry point behaves identically from
any directory.

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

One window, built on the same `octobee probe` decode path as the CLI tools. The
left half carries the work — live traces, the per-sensor table, calibration,
diagnostics, exports — and the right half always shows the probe head in 3D
with a peak-|B| bar chart underneath, so the state of all 16 sensors is visible
whatever tab you are on.

Underneath those sits a small **x / y / z panel**: a live position readout, a
jog step with − and +, and a move-to box with **Go**, for each axis. It is
there because moving the head and watching the field answer is one action, and
having the buttons on the Stages tab meant doing it with the plot on a page you
could not see. The panel is a second set of buttons onto the *same* methods the
Stages tab calls — the emergency-stop latch, the one-move-at-a-time guard and
the refusal to move an unreferenced axis absolutely are written once and apply
identically from either place. The readout goes amber while an axis is moving
and red while its counter cannot be believed.

The **peak bars** tick beside the 3D controls turns the bar chart off and gives
its height back to the head and the stage panel. The bars answer "are the
spikes the same height", which is not the question you are asking while driving
the rig somewhere. Nothing else changes: acquisition, recording and the health
check are untouched, and the computation stops with the widget, so the space is
not the only thing you get back.

![The Live tab against the real carriers](images/gui-live-hardware.png)

*Live, both carriers, no magnet. S16 is excluded automatically and dropped from
the legend; the bar chart ranks the rest by peak |B| and reproduces the known
noise pattern — S9 worst, then S15, S13 and S8. Those four are the chip faults
catalogued in section 3; S16 was still excluded when this was taken and is now
healthy.*

### Why a 3D view rather than more plots

The chips point in 16 different directions, so a stack of 48 traces cannot show
you where a field actually is. The 3D view rotates each chip's vector into the
common tube frame with the matrices from `octobee geometry` and draws it where
that chip physically sits. A magnet passing the probe then reads directly: which
face it went past, in which direction the field pointed, and — because every
arrow shares one scale — which sensors answered more strongly than their
neighbours.

Colour and arrow length are |B|, which is rotation-invariant and therefore the
only amplitude that can honestly be compared between chips. Excluded chips are
drawn dark red with no arrow.

The model is drawn the way the probe is mounted rather than the way the maths
writes it: the tube lies horizontally along the rig's **Y** with the tip end
(S4, S8, S12, S16) pointing forward and **Z** up, so an arrow on screen points
the way the field points on the bench. Only the drawing moves. `MOUNT_ROT` in
`octobee geometry` is applied at the last moment before each mesh and arrow
goes to the GPU, so the tube frame that every calibration, pose solve and export
is written in is untouched — remount the probe and it is one matrix to change,
with nothing to recalibrate.

![A magnet passing the probe](images/gui-magnet-pass.png)

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

![The Calibration tab](images/gui-calibration.png)

Each sensor's measurement range is set per row in the Sensors tab rather than
globally, because the range is a per-chip register and the two halves of this
probe genuinely have differed. As of the 2026-08-19 harmonisation (section 3)
all 16 run gain 3000, +/-20 mT, 63 V/T, and the shipped `calibration.json`
records exactly that -- re-read live off both carriers with:

```bash
ssh root@acq1001_694 'python3 /tmp/gain_config.py show'
ssh root@acq1001_695 'python3 /tmp/gain_config.py show'
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

Everything is written under `--out-dir` (default `captures/`).

**Record shows a red dot while a file is actually open**, and stopping puts up
a box naming the files it wrote, the folder they are in and how big they are,
with a button that opens that folder. Both of those exist because the path was
already in two places nobody was looking: the Log, and the Data output tab you
were not on. The dot follows the recorders rather than the button, so pressing
Record with nothing ticked here — which puts the button down for as long as it
takes to find that out — does not light it, and the four other things that stop
a recording (a field map starting, a snapshot, Disconnect, closing the window)
each get the indicator right without doing anything. Closing the window is the
one stop that does not raise the box: a modal put up during teardown is a
window that will not close.

**Changing the calibration while recording rolls the CSV over.** The header is
written once, at open, and carries `calibration_id` plus the whole conversion
state so a file can be matched back to the calibration that produced it. Zero,
Apply gain, Apply roll, a range change or Load calibration therefore close the
current file and start a new one, logging both names and stamping `continues:`
into the new header. One CSV whose rows came from two calibrations, under a
header naming only the first, would be worse than either file alone — nothing
about it would look wrong. The raw `.bin` is unaffected: it holds counts, which
no calibration changes.

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
  carries the audited registers, and why the suite asserts them.

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

### The geometry file is measured now — and one face was backwards

`probe_geometry.json` was an assumption for most of this project's life. As of
**2026-08-24** the arrangement in it is a measurement, from a guided magnet run
(`captures/magcal_20260824_153205.npz`), and the file is marked `"mapping":
"measured"` so it cannot be regenerated back into a guess.

What the run found:

| | |
|---|---|
| faces | **as assumed** — S1–S4, S5–S8, S9–S12, S13–S16, four to a face |
| ring pitch | rings at 39.4, 73.5, 105.0, 136.5 mm → 32.4 mm mean, against the 33 mm the plate spacing predicts |
| slots | **face 0 was reversed**: S1–S4 run the opposite way along the tube from the other three faces |

So S1 sits at the *tip*, beside S8, S12 and S16 — not at the flange where the
file had put it. Twelve of the sixteen were already right, which is exactly why
this was worth measuring rather than eyeballing: a layout that is three-quarters
correct looks entirely correct in a 3D view.

The direction is not a convention anyone chose. The tube's +Z maps to the rig's
+y, and the head is driven the positive way, so the sensor further along the
tube reaches a fixed magnet at a *smaller* stage position and passes it first —
which makes the first ring the tip. `measured_slots()` reads that off
`MOUNT_ROT` rather than hard-coding it, so remounting the probe reverses the
rule automatically. It also cross-checks against the one thing about this probe
established by looking at it: S16 is at the front, and S16 came out in the first
ring.

Still assumptions: the tube width, the mounting-plate pitch (though the ring
spacing above now corroborates it to about 2 %), and each chip's `chip_rot_deg`
and `axis_signs` — the Earth-field roll sweep is what settles those, and the
magnet run says nothing about them.

One thing the magnet **cannot** decide is which face index a group of four
carries. Turning the tube in its cradle renames all four, so the run checks the
*grouping* and leaves the labels alone.

### When it feels slow: `--profile`

```bash
python octobee/gui/window.py --profile
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
pytest                                 # synthetic probe, no hardware
pytest --replay captures/ambient_test.npz
pytest --live                          # against the real carriers

pytest -k stage                        # just the stage tests
pytest -x                              # stop at the first failure
pytest tests/gui -q                    # just the ones that drive the window
```

It drives the whole pipeline and then reads the written files back to check the
numbers, rather than checking that files merely exist. The live run also asserts
that no blocks were dropped and that the carriers are left at the clock they
were found at.

454 checks, none of which need hardware in the default mode. Everything it
writes goes to a temporary directory — a test run leaves `captures/` alone, and
CI fails if the working tree comes back dirty. Run the linter alongside it:

```bash
pip install -e ".[gui,dev]"
ruff check .                           # config and reasons in pyproject.toml
pytest
```

Both run on every push via `.github/workflows/checks.yml`. `pyproject.toml`
gives a reason next to every lint rule that is switched off, so the list stays
a set of decisions rather than a pile of suppressions.

---

---

*Part of the [OCTO-BEE documentation](../README.md).*
