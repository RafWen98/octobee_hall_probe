# Measuring inside the coil set

A field map is a table of vectors at rig millimetres. On its own it says
nothing: 40 mT at (120, 45, 80) describes a useful measurement and a useless
one equally well, and which it was depends on two things that are nowhere in
the file — where the probe was in the machine, and which coils were carrying
current. The **Machine** tab is where both are declared, and it draws the
consequence rather than asking anyone to picture it.

```bash
octobee machine designA_after_scaled.json          # what is in the file
octobee machine designA_after_scaled.json --at 40 0 12   # and where the probe is
```

In the GUI it is the tab between **Stages** and **Data output**: the coil set
on the right in 3D, the declarations on the left, and a clearance number
underneath them that turns red when the probe is drawn somewhere it cannot be.

### The coil file

The tool reads simsopt configuration files — the SIMSON JSON that
`simsopt`'s `save()` writes. It does **not** import simsopt to do it: that
would drag a compiled optimisation stack onto a machine whose job is two
carriers and three stages, so the graph is parsed directly. Only what a coil
set is made of is understood — `CurveXYZFourier`, `CurveRZFourier`,
`RotatedCurve`, `Coil`, `Current`, `ScaledCurrent`, `CurrentSum`, `BiotSavart`
— and everything else in the file, including plasma boundaries and the
optimiser's own bookkeeping, is ignored rather than refused.

Two things about these files are worth knowing before the numbers look wrong.

**A file usually holds the same coils several times over.** `designA` has
eighteen `Coil` objects, but only six distinct curves: three `BiotSavart`
objects, each pairing the same six windings with a different set of currents,
because that is how a multi-configuration optimisation records its current
sets. Taking those at face value would draw every coil three times and offer
eighteen switches for six physical windings. So coils are identified by their
*geometry* — the base curve plus the chain of rotations applied to it — and the
currents hang off that identity, one per configuration. `designA` therefore
loads as **six coils in three current configurations**, which is what is on the
bench:

| | | |
|---|---|---|
| C1–C4 | modular, from `CurveXYZFourier1` | φ = 34°, 146°, 214°, 326° |
| C5, C6 | from `CurveXYZFourier2` | φ = 90°, 270° |

The four modular coils are one shape, rotated and flipped into the other three
positions by the machine's two-field-period stellarator symmetry; the tool
reproduces that from the file and nothing about it is typed in.

**The coil curves are exact, and the sampling in the file is not a limit.**
`quadpoints` is whatever the optimiser happened to be using. The Fourier
coefficients are the geometry, so the curves are re-evaluated here at 256
points each — a chord error of about 75 µm on a 3 m coil, which is far below
anything the clearance number is trusted to.

### The keep-out volume, and the one number that is a guess

Each centreline is swept with a circular cross-section, and that tube is the
volume the probe cannot enter. **No simsopt file contains a conductor
thickness** — the optimiser works with infinitely thin filaments — so the
radius is a setting in the tab, starting at **60 mm**, the pack this machine is
being built around. It is still a number to check against the real conductor
rather than one read out of any file, and correcting it moves every clearance
figure by the same amount.

Clearance is reported to the tube's *surface*, not to the centreline, and it is
measured from the whole probe body — the tube, all sixteen boards and their
chip packages, sampled along every edge rather than at corners alone, because a
coil can pass close to the middle of a 159 mm tube while staying far from both
of its ends.

Two things it deliberately does not do:

* **A coil that is switched off is still solid copper.** Clearance is checked
  against every coil in the file, energised or not. The energised set changes
  the colour and what gets recorded, never what is in the way.
* **It is a drawing, not an interlock.** Nothing here stops a stage. The
  clearance number is only as true as the pose typed in above it, and a pose
  that has not been measured will produce a confident number about nothing.
  The emergency stop and the axis limits in `stages.json` are the things that
  actually protect the head.

### Frames: what the six numbers mean

```
machine frame   the coil file's own coordinates, in millimetres.
                Z is the axis of the torus.
mount frame     the probe's own frame: tube axis along +Y, tip forward,
                +Z up — the rig's axes, which is what the stages move along.
```

The placement is where the probe's **mounting flange** — the end of the tube
the boards count up from — sits in the machine, plus one angle: how far the
whole assembly is turned **about the machine's Z**, which is the axis of the
torus.

This angle is what says **how the assembly is mounted relative to the coils**,
and it turns the probe's whole coordinate system with it: the drawn body, its
own axes on the drag handle, the green stage envelope and the volume to be
mapped all follow, because a stage driven along rig *x* on an assembly turned
90° moves the head along machine *y*.

One angle, because the rig has one. The probe is bolted to a three-axis
cartesian gantry that cannot tilt it: the tube lies horizontal along the rig's
Y and the only freedom is which way round the machine the assembly is pointed.
Pitch and roll used to be settable and were always zero, and a pose box that
can describe a position the rig cannot reach is a way to produce a confident
clearance number about an impossible one. A `machine.json` written before they
went is still read — `yaw` was this same rotation under another name and comes
straight across — and a non-zero pitch or roll in one is dropped **with a
warning in the log** rather than quietly approximated.

The live stage reading is then added **in the mount frame**, before that
rotation. Turn the assembly a quarter turn and driving rig *x* moves the probe
along machine *y*, which is what the rig physically does — a drawing that moved
it along machine *x* regardless would be wrong in exactly the way nobody
notices until something is bent.

*Stage zero is here* takes the current reading as the position the pose
describes. Park the rig where the flange was measured, press it once, and every
later move is drawn relative to that. The green wireframe box is the volume the
flange can be driven through, taken from each axis's allowed travel — the
region this rig can reach without being unbolted and moved.

### Dragging the probe into place

The zero point itself is drawn: a pale ball at the flange, with a red, a green
and a blue arrow and a blue ring round it. **The arrows are the probe's own
axes, not the machine's** — the rig's *x*, *y* and *z*, the ones the stages
drive along, turned by whatever *about Z* is set to. That is the whole point of
them: the machine's axes are already drawn once at the machine's origin, so a
second set of the same axes at the probe would say nothing the ball does not,
where the probe's own frame makes the mounting angle visible as the angle
between the two. Drag an arrow and the probe slides along that rig axis — which
on an assembly mounted at 90° means driving rig *x* moves it along machine *y*,
exactly as that stage would; drag the ring and it turns about Z. Either way the
*x*, *y*, *z* or *about Z* box above follows — it is the same edit, taking the same route through the same
box, so the clearance number, the drawing and what *Save placement* writes all
update exactly as if the number had been typed.

The ring is tested in the plane it lives in rather than in screen pixels. A
circle in the world is an ellipse in the window, and chasing that ellipse in
pixels gets the near and far sides of it wrong at a shallow camera elevation;
casting the pointer onto the plane gives the right answer at any angle. It also
sits outside the arrows' grab band, because an arrow wins a click they both
want, and a ring overlapping the +x arrow would slide the probe when it looked
like it should turn it.

Three things about it are deliberate:

* **It is sized in pixels, not millimetres.** A handle a fixed number of
  millimetres long is invisible with a three-metre coil set in frame and fills
  the window when the camera is on the head. This one stays the same size on
  screen at every zoom, so it is grabbable at both.
* **It is drawn over everything.** Depth testing is off for the handle, so it
  stays visible when the head is behind a winding — which is precisely the
  moment somebody wants to drag it back out.
* **Only the arrows take a click.** Anywhere else in the view still orbits the
  camera, and the offset is measured from where the arrow was grabbed rather
  than accumulated frame by frame, so a long drag cannot creep away from the
  pointer. *drag handle*, under the view, turns it off entirely.

Both are still a declaration, not a measurement. Dragging is a faster way to
type a number, and the number is only as good as the tape it came from.

## Mapping a volume

A field map of a whole volume is a different problem from a field map of a
point, and it needs a different method.

**The settled raster does not scale.** `octobee/motion/scan.py` moves, stops,
settles and averages, and it is right to: the noise is white, so twenty seconds
of stationary averaging at 200 kSPS reaches 0.020 µT where the moving stream
manages 0.235. But the stage envelope is 300 mm cubed, which on a 10 mm grid is
29,791 points, and at the seven-and-a-half seconds a settled point really costs
that is **sixty-two hours**. Halving the grid step multiplies it by eight. That
is not a long experiment; it is an impossible one.

**So a volume is swept.** One axis runs at constant velocity while the stream is
logged continuously; the other two step between lines. The same 29,791 samples
come off a 300 mm line in thirty seconds. What is lost is per-sample noise —
and that is recoverable, because every sample is stored and averaging afterwards
is free. What is *not* recoverable is a point that was never visited, which is
what the settled raster actually costs.

Both are kept. **Stages → Field map** for a small box measured carefully;
**Machine → Volume to map** for a survey.

### Saying which box

*The whole travel* is everything the stages can reach, taken from each axis's
own travel — 300 mm cubed on this rig, and the same box the green stage
envelope already draws. Untick it and type where a smaller box starts and how
big it is. Either way the box is drawn in the machine, so what is typed in rig
millimetres can be seen in coil coordinates.

*Grid step* is how far apart the swept lines are, and the resolution the
reachable region is worked out at. It is **not** how finely the swept axis is
sampled: that is the output rate against the sweep speed, and at 500 Hz and
10 mm/s it is a sample every 20 µm.

*Sweep speed* is capped at **5 mm/s**, which is a measurement limit rather than
a hardware one — the axes will do 10. The head is on the end of a cantilever and
this is the only measurement taken while the thing taking it is moving, so there
is nothing to be gained by hurrying it.

*Outer surface only* maps the six faces of the box instead of filling it, each
swept in its own plane — the two faces across the sweep axis are run along a
different one, which is why a line carries its own sweep axis rather than taking
the volume's. For a field in a current-free region the boundary values determine
the interior, so this is not a poor relation of the filled version. It is also
five or six times fewer lines: the whole envelope on a 10 mm grid is 182 lines
against 961, about **4.6 hours** at 5 mm/s where filling it would be a day.

### Cutting the volume to the shape of the space

With **keep clear of the coils** ticked, every line is cut back to the part the
probe body actually fits in. A line that would pass through a winding becomes
two shorter lines with the coil between them, so what gets mapped is the shape
of the space that is there rather than the largest box that happens to fit
inside it. The lines that survived are drawn in green and the ones that did not
in dim red, because the carved shape only reads as a shape next to what it was
carved out of.

**The whole probe is tested, not the head.** The square tube runs the length of
the probe back to its zero point, and it is what meets a coil first — a plan
that only checked the sensors would drive the tube into a winding while
reporting a comfortable clearance at the tip.

This is not the same calculation as the clearance readout, and it is worth
knowing why. That readout is exact and costs a few milliseconds at one place;
a 300 mm cube on a 10 mm grid has thirty thousand places in it, and asking
exactly costs a minute — long enough that nobody moves the pose and watches the
volume change, which is the whole point of drawing it. Instead: the probe is
rigid and cannot tilt, so the set of flange positions that put some part of it
inside a winding is exactly the forbidden region **dilated by the probe's own
shape**. On a regular grid that is a handful of shifted ORs over a boolean
array, and it takes about half a second.

The price is stated rather than hidden. The probe is snapped to the grid, which
moves any part of it by at most half a grid diagonal, and that distance is
added to the exclusion radius — so the answer can refuse a position that would
in fact have been fine, and **never the reverse**. At a 10 mm step it is 8.7 mm
of extra standoff, and the label says so next to the margin you asked for. The
coils themselves are measured against their decimated outlines first and the
real 256-point curve wherever the decimation cannot settle it, so that
approximation costs time rather than clearance.

### Flying it first

*Fly it* runs the probe along the planned path in the drawing. Nothing moves.
It is the same `set_pose` the stage timer uses, fed a made-up stage reading, so
what is drawn is exactly what would be drawn if the rig really were there —
**clearance readout included**. That is the point: a plan is worth flying
precisely because the number that says it is safe is computed the same way
either way.

### What comes off it

A swept map reads the live stream rather than taking the carriers over, so the
live plot keeps working and there is no clock to put back afterwards. It lands
in `captures/` as `volume_<timestamp>.npz` with a `.json` sidecar:

| | |
|---|---|
| `t_s` | seconds from the start of the sweep |
| `pos_mm` | rig x, y, z — the stage polls interpolated onto `t_s` |
| `pos_machine_mm` | the same point in the coil file's frame |
| `b_mt` | (rows, 16, 3) — Bx, By, Bz for every sensor |
| `line_index` | which swept line each row belongs to |
| `in_span` | row is inside the measured span, not in a ramp |
| `counts` | raw quadrature counts, unwrapped, one row per field sample |
| `poll_t_s`, `poll_mm` | every raw controller read, so the fallback can be redone |

### Where position comes from

Two sources, answering different halves of the question.

**Where.** Neither is better at this. The LTS300C has no scale on the carriage,
and the quadrature encoders are rotary, on the motors, so both count turns of
the same 1 mm leadscrew and inherit the same ~47 µm of absolute pitch error
without Thorlabs' per-serial calibration files. Absolute position comes from
homing either way. The encoders do **not** make the map more accurate.

**When.** Here they are not close, and this is what they are for.

`acq1001_695` aggregates three quadrature encoder modules into its own sample
stream — its `rc.user` puts sites 2, 5 and 6 in the aggregator set alongside the
ADC and sets `phaseA_en`/`phaseB_en` on each — so every frame it emits carries
the encoder counts *for that sample*. The count in sample *n* was latched when
sample *n* was converted, on the ADC's clock, so a position taken from it
belongs to the sample it is written against. There is nothing to interpolate
and no offset to estimate.

**Which carrier is not obvious from the frame.** Both boxes aggregate a
quadrature module in site 2, so both emit a longword between the analogue
channels and the scratchpad:

| | aggregator | encoder longwords | counting |
|---|---|---|---|
| `acq1001_694` | `sites=1,2` | 1 | none — `phaseA_en=0` |
| `acq1001_695` | `sites=1,2,5,6` | 3 | sites 2, 5, 6 |

Picking "the carrier with some encoder longwords" therefore picks the 694, which
is first in `DEFAULT_UUTS`, and its column never moves. That failure is invisible
downstream: a column that does not change looks exactly like an axis that did not
move. So the carrier is chosen on `phaseA_en`, read from the boxes themselves —
which is also what makes a rebuilt box say so rather than being assumed.

Without them the position column is a controller reading taken over USB,
interpolated onto a field sample that arrived through a stream buffer — two wall
clocks with an unknown common offset between them, which at sweep speed is
around a millimetre. Both are still logged, and `pos_source` in the sidecar says
per axis which one was used; an axis with no calibrated encoder simply falls
back. On the bench the difference measures three orders of magnitude.

A quadrature counter is **incremental**: it counts from wherever it was at
power-up. So it is used as a displacement against a datum taken from the
controller at the start of each line, standing still —

```
mm(n) = mm_at_datum + (counts(n) - counts_at_datum) / counts_per_mm
```

— which puts absolute position on the homing and timing on the ADC clock, each
on whichever is actually responsible for it.

**Calibrate encoders** measures the scale rather than asking you to type it. It
drives each axis 20 mm and back, one at a time, and watches which stream column
follows it and by how much. Moving one axis at a time is the method, not
politeness: the column that moved is the column that belongs to the axis that
moved, and two at once makes that unanswerable — it refuses rather than guessing
if two columns move together, or if none moves at all. The result goes into
`stages.json` under `encoders`, beside the rest of the axis facts, and it can be
re-run as a check. A typed-in counts-per-millimetre that is wrong rescales every
position in every map and looks entirely plausible doing it.

Lines still never reverse, for `octobee/motion/scan.py`'s reason: the leadscrew
has backlash and a serpentine would stamp a fixed offset into every other line.
Each line is swept in the +ve direction and the axis returns to the start of the
next one — which on axes capped at 10 mm/s means the return costs as much as
the sweep, so a volume takes twice as long as the sweeping alone suggests. That
is in the estimate rather than discovered overnight. Lines also over-run at both
ends where the travel allows, putting the trapezoidal ramps outside the region
being measured.

### Saying which coils are on

Pick a configuration, tick the coils that are actually energised, and set the
scale. The file's currents are the design point in amp-turns — tens of
kiloamp-turns — and a bench test is that design point scaled down by a factor
of a hundred or more, so a single scale factor keeps the ratios between coils
as optimised while putting the magnitudes where the power supply really is. For
a single-coil test, untick the other five; the scale then simply is that coil's
current as a fraction of its design value.

None of this is a measurement and none of it is inferred from the data: it is a
declaration, and it is written down as one.

### What a field map carries away with it

Starting a scan from the Stages tab records the machine into the map's `.json`
sidecar, under `machine`: the coil file, the configuration, the scale, which
coils were on and at what amp-turns, the pose, the winding radius used, and
where the flange was. A map found next year is then still readable — none of it
can be recovered from the vectors themselves. Caller metadata cannot overwrite
anything the scan measured; the measurement's own keys win.

`machine.json` beside `stages.json` holds all of it between sessions, because
the pose of a probe inside a coil set is measured once, with a tape and some
care, and re-typing it every morning is how it drifts.

### What this is not

It draws no field and predicts none. There is no Biot–Savart here: the tab
answers *where am I and what is live*, and leaves *what should I be reading* to
the measurement.

---

*Part of the [OCTO-BEE documentation](../README.md).*
