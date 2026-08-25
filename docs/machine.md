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
radius is a setting in the tab, starting at **20 mm**, and it is a placeholder
until somebody measures the real winding pack. It is the first number to
correct, and correcting it moves every clearance figure by the same amount.

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
the boards count up from — sits in the machine, plus yaw, pitch and roll.
Yaw is applied last, so it swings the whole assembly about the machine's Z
whatever pitch and roll are already set to; that is what somebody typing into
the box expects it to do.

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
