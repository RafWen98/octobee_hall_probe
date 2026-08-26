# Calibration procedure

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

The cabling work is **done**: S16 is repaired and every VCM now reads between
0.5 and 2.2 counts (section 3). What remains is five chip-level faults that no
amount of reseating will fix, so this step is now a check rather than a job.

Re-run `octobee report --seconds 5` and confirm every sensor still shows
VCM noise below ~2 counts. If one has crept up, that is a cabling fault and
calibrating over it bakes it into your coefficients. If instead a *field*
channel is noisy while its own VCM is clean, that is a chip fault -- see
*Telling a bad chip from a bad cable* in section 3 -- and it cannot be reseated
away. Decide whether to exclude that axis rather than holding up calibration
for it.

**695 ports 3 and 7 are currently swapped** from the diagnostic in section 3.
Exchange them back first, or every geometry matrix, zero and offset keyed by
sensor index will be wrong.

### Step 1 — make the chips identical (this is the likely culprit)

Gain, Hall bias current and EEPROM calibration status live inside the ASIC and
are reachable only over SPI from the carrier. A chip at gain 1500 gives exactly
half the response of one at 3000, for the same field.

```bash
scp onbox/sensor_audit.py root@acq1001_694:/tmp/
ssh root@acq1001_694 'python3 /tmp/sensor_audit.py'
ssh root@acq1001_695 'python3 /tmp/sensor_audit.py'
```

The root password is the one on the printed sheet that shipped with the units
(D-TACQ manual §"The root password is provided on a printed sheet with your
shipment"); both carriers use the same one. OpenSSH on this PC prompts
interactively, which does not work from a script — PuTTY is installed and takes
the password on the command line:

```bash
PUTTY="/c/Program Files/PuTTY"
HK=SHA256:51adMhUSu461QXQICNZfCpfA81hV1nomlMNQPhKaq3M    # acq1001_694 host key
"$PUTTY/pscp.exe"  -batch -pw "$PW" -hostkey "$HK" onbox/sensor_audit.py root@acq1001_694:/tmp/
"$PUTTY/plink.exe" -ssh -batch -pw "$PW" -hostkey "$HK" root@acq1001_694 'python3 /tmp/sensor_audit.py'
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
ssh root@acq1001_694 'python3 /tmp/sensor_audit.py --set-gain 3000'
```

That writes registers only, so it is lost at power cycle. Make it permanent by
writing the same values to EEPROM and calling `activate_EEPROM_config()`.

### Step 2 — verify the channel map

```bash
# terminal 1
ssh root@acq1001_694 'python3 /tmp/sensor_audit.py --id-sweep 4'
# terminal 2, started right after
octobee idmap --uut acq1001_694 --seconds 70
```

The sweep mutes each sensor's Bx/By/Bz in turn over SPI; the host tool reports
which ACQ423 channels went quiet, in order, and prints CONFIRMED or the measured
map. No clock sync needed — only the order matters. Repeat for `_695`.

### Step 3 — build the physical map

SPI position ≠ position on the tube. Pass a magnet slowly along one face:

```bash
octobee idmap --uut acq1001_694 --seconds 60 --magnet
```

It ranks the sensor groups by when they peaked, which is the physical order.
Record the result as a table: *tube face × position → sensor → channels*.

### Step 4 — zero-field offsets

Capture with no magnet nearby and the probe away from anything ferrous:

```bash
octobee probe capture --seconds 10 -o zero_field.npz
octobee report --load zero_field.npz
```

The `off Bz/By/Bx` columns are each axis' residual offset after VCM subtraction.
Store them as your offset table. For a proper zero you want either a zero-gauss
chamber, or the flip method: rotate the probe 180° and average the two readings,
which cancels the ambient field and leaves the offset.

### Step 5 — cross-calibration (superseded — see Step 5b)

> **Superseded by Step 5b.** In the GUI this now lives behind *Calibration →
> show the superseded manual routines*, unticked by default. It is kept because
> a hand pass is still the quickest way to see whether all sixteen channels are
> alive, but it should not be what sets your gain trim: the correction it needs
> and the geometry it needs to compute that correction are the same unknown.

With gains harmonised and offsets known, present the *same* magnet to each
sensor in turn, normal to that sensor's face, at a fixed distance — use a
machined spacer or jig so the geometry repeats to well under a millimetre.

```bash
octobee report --seconds 20 --range 20 --plot
```

The report gives each sensor's peak `|B|` and its ratio to the median. Those
ratios are your per-sensor scale factors `k_i = B_ref / |B|_i`. After Step 1 they
should land inside a few percent; anything still outside ±10 % means that chip's
EEPROM calibration is suspect or its mounting differs.

### Step 5b — the guided magnet run (motorised, and geometry-free)

Step 5 as written above has a weakness it cannot escape: presenting the same
magnet to each sensor *by hand* means a different distance and angle every
time, and correcting for that needs `expected_response()` — i.e. it needs the
geometry file to be right, which at Step 5 it is not yet. The correction and
the thing being corrected are the same unknown.

With the stages, there is a version where the geometry cancels instead of being
modelled. The tube axis lies along the rig's **y**, so driving the *head* along
y past a **fixed** magnet carries S1, S2, S3, S4 of one face past it at the
**same closest approach**, one after another. Turn the head a quarter turn and
the next face gets its turn on the same terms. After four poses all 16 peaks
are directly comparable, and

`trim_i = median(peak) / peak_i`

is a pure gain ratio with no 1/r^n anywhere in it. Against a synthetic dipole
with a known ±8 % gain error injected, the run recovers it to **0.5 %** — the
residual being the sampling grid, not the method — where a single fixed-magnet
pass over the same probe spreads the raw peaks by **277×** from geometry alone.

#### Why one sweep is not enough: the arms are not identical

"Same closest approach" above is doing real work, and it is only true if the
sixteen arms are identical. They are not. Each chip rides at the tip of a 92 mm
board, so a degree of board rotation about its bolt, or a foot seated proud,
puts the chip a millimetre from where the file says — and at 1/r³ and a 20 mm
standoff that is `3 × 1/20` ≈ **15 % of trim per millimetre**, which is larger
than the gain spread being measured.

Resolved along the three directions of the magnet-to-chip line, the
misplacement behaves completely differently, and each direction needs its own
treatment. So each pose is **three passes**, not one:

| pass | what it does | what it fixes |
|---|---|---|
| **A** locate | coarse sweep along the tube | finds where the four rings are |
| **B** cut | fine sweep *across* the face at each ring | each sensor measured at the top of its *own* peak, not at a slice through it |
| **C** dither | a few points *toward and away from* the magnet at each ring | measures each chip's actual distance |

Pass C is not a sweep for a peak, because along the standoff direction there
isn't one — `|B|` just keeps rising as the chip gets closer. What that direction
carries is a slope and a curvature, and those give the distance and the falloff
exponent outright:

`|B| = C·|d−z|^−n` ⟹ `d = slope / curvature`, `n = slope² / curvature`

both read off the dither with no constant assumed and nothing taken from the
geometry file. Correcting every sensor to a common, *measured* standoff removes
the last first-order term. Against a synthetic probe with 1 mm of misplacement
on all three axes, a bare axial sweep records **9.9 %** of fake gain; all three
passes bring it to **1.8 %**. On straight arms the extra passes cost nothing —
the standoff correction shrinks itself toward the identity when the scatter it
would correct is no bigger than the fit's own noise.

The whole thing costs about 5× the points of a bare axial sweep (~130 per pose
here), against the ~39 000 a full volume raster would need to learn the same
thing.

> **B and C are a package — run both, or neither.** Pass C's model is *moving
> straight toward the magnet*, which is only true once B has put the chip
> underneath it; off to one side by `a`, the fit returns `h·r²/(h²−a²)` instead
> of the distance — 51 mm for a chip really 26 mm away, on this rig's geometry.
> And B on its own moves every chip to its true, *nearer* closest approach,
> which amplifies whatever first-order error is left: B alone takes that 9.9 %
> to **12.8 %**. It is only a win once C is there to collect it. The wizard
> refuses the dither without the cut for this reason.

You do not have to park the magnet accurately. Finding each peak is what pass B
is for — that is most of the point of it.

**Calibration tab → Guided magnet calibration (motorised, 4 poses).** The
wizard drives each pass, averages at every point at the full 200 kSPS, and
stops between poses to ask you to index the tube.

It opens on the **standard run** — 145 points a pose, about 11 minutes each and
44 minutes for all four:

| | | | |
|---|---|---|---|
| sweep | 140 mm | cut half-span | 10 mm |
| step | 3.5 mm | cut step | 1 mm |
| per point | 2.0 s | dither half-span | 5 mm |
| settle | 0.5 s | dither points | 5 |
| standoff | 20 mm | | |

Those numbers are fixed on purpose (`magnet.STANDARD_RUN`), so that two runs a
month apart are the same measurement and their trims can be compared rather
than each believed on its own. They are not the only defensible ones, and two
are worth knowing you are choosing:

* **five dither points, not seven.** Seven is what the bench measured as best —
  on a probe misplaced by 1 mm the residual trim error is 2.1 % at seven points
  against 4.2 % at five. The standard buys back eight moves a pose at half the
  standoff accuracy. Raise it for a run whose trim you intend to keep.
* **a 10 mm cut half-span at a 20 mm standoff**, which is half of what the
  sizing rule below asks for, at the same 21 points — the same cost spent
  closer in around the peak.

Change the *standoff* box and both of the other passes resize themselves from
it, because nothing is measured with that number but B and C both have to be
*sized* from it: the cut wants a half-span of about one standoff, because that
is the width of the peak it is hunting, and the dither wants a quarter of one,
because curvature is second order and a ±1 mm dither buries it under the slope.
That hands the run back to the rule and takes it off the standard, which is the
intended direction — a non-standard standoff has no standard sizing.

**Every pose is written to disk as it lands**, to one growing pair of
`captures/magcal_*.npz` / `.json` files, so a crash or a closed window after
pose 3 costs the poses you have not driven yet and nothing else. When the run
finishes it **writes the trim to `calibration.json`** as well as applying it.
Untick the box and it applies nothing, and says where the run is so it can be
applied later without repeating it. Two conditions, which are why
it is guided rather than a button:

* the magnet must not move between poses, and nothing ferrous may — it is the
  common reference for all four;
* the turn must be about the tube's **own axis and nothing else**. A pose that
  also shifts the head sideways breaks the equal-approach argument quietly, and
  the numbers still look plausible afterwards. (The self-test injects exactly
  this mistake and checks that it changes the answer.)

The run also measures what `probe_geometry.json` still assumes: sorting the 16
peak positions gives the physical order along the tube, and grouping by which
pose was loudest sorts the sensors onto faces. It **reports** disagreements
with the file rather than rewriting it. Pass B adds to that — the transverse
position of each sensor's peak is a direct measurement of where its arm
actually put the chip, printed in the report as an `x peak` column, and the
`standoff` column beside it is the same for the radial direction. Read those
two before the trim: if the standoff column is mostly `--`, pass C could not
fit and the distances are assumed rather than measured, and the fix is a longer
average per point.

It does not replace Step 6 — but it takes most of Step 6's job. This fixes the
scalar per-sensor response, says which sensor is where, and now measures the
position error that Step 6 existed to sidestep. What the Earth-field roll sweep
still does alone is pin each chip's **orientation** in three dimensions. Run
this one first regardless: knowing which sensor is on which face makes the roll
solve's per-face report readable.

#### The one setup fault that looks exactly like gain

Everything above rests on the head turning about its **own** axis, so all four
faces meet the magnet at the same distance. If the head is not concentric with
whatever it is turned about, one face comes closer and the opposite one goes
further by the same amount — and 1/r³ turns a sub-millimetre error into a ten
per cent "gain" difference that the trim will happily absorb.

It is separable because it is *anti-symmetric*. The geometric factor multiplies
one face by k and its opposite by 1/k, so the **product** of two opposite
faces' mean responses survives it while each face on its own does not. The run
report therefore prints opposite faces side by side, and says so when they
disagree by more than 5 %:

```
opposite faces (equal unless the head is off-centre):
  +X vs -X:  12.225 vs  12.578 mT  (0.972x, ~0.5% off-centre)
  +Y vs -Y:  13.599 vs  11.406 mT  (1.192x, ~2.9% off-centre)
```

That is the real run of 2026-08-24: one pair clean, the other 19 % apart, so
roughly ±9 % of what that trim calls gain on the ±Y faces is geometry wearing
gain's clothes. The within-face numbers are unaffected — face 0 spans 1.056×
across its four sensors — because those four never move relative to each other.

When it fires, either re-centre the head and run again, or accept that the
face-to-face part of the trim is not purely electrical. The peak-spread line in
the report stops claiming "this is gain only" when it does.

### Step 6 — Earth-field roll calibration (superseded for gain; still the way to pin orientation)

> **Superseded for gain matching by Step 5b**, which measures the position error
> this step was invented to dodge, rather than dodging it. In the GUI it now
> lives behind *Calibration → show the superseded manual routines*, unticked by
> default.
>
> It has not become useless: the roll sweep is still the only thing here that
> pins each chip's **orientation** in three dimensions and settles
> `chip_rot_deg` / `axis_signs`. Run 5b first for gain and the sensor-to-face
> map, then this if you need the orientation solve. The rest of this section is
> written as it was, when it was the accurate way to match sensors.

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
octobee poses --survey --seconds 10 --dead S16
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

`octobee/calib/poses.py --turns 2` walks you round twice, prompting for the same
90° index each time; a revisited pose also shows you drift directly, since two
rows that should be identical are sitting in the same file.

In the GUI: tick *Calibration → show the superseded manual routines*, then
*Superseded: Earth-field roll calibration → Record sweep A / B / C → Solve →
Apply*. Or offline:

```bash
octobee roll captures/rollsweep_A.npz \
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
octobee poses --tag A --seconds 20      # as mounted
octobee poses --tag B --seconds 20      # tube end-for-end
octobee poses --tag C --seconds 20      # cradle at a new azimuth
```

It prompts between poses, refuses to start if a killed live session left the ADC
clock down, stores the median of 64 sub-blocks per pose (so one person walking
past cannot poison a pose without being seen), and finishes each sweep with a
**closure** capture back at pose 1. That closure is the honest error bar on the
whole session: under 1 µT is clean, 1–5 µT is your real floor rather than the
noise figure, above 5 µT means something moved and the sweep should be redone.

What it writes is an ordinary `RollSweep`, so `octobee roll` and the GUI's
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
  shares the same rotation — so keep using `octobee/acq/idmap.py --magnet` for that.
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

There used to be a code reason to stay bipolar as well, and there is not any
more. A unipolar 0–5 V range puts 0 V at −32768, so true volts are
`counts*vpc + 2.5`; `Layout.volts_per_count` carried no such offset term, so
every raw channel would have read 2.5 V low and `PLAUSIBLE_V = (-0.5, 5.5)` in
`channel_health()` would have called all 64 healthy channels broken. Field
readings would have survived — subtracting VCM cancels the pedestal — which is
what made it dangerous: the diagnostics would scream while the numbers looked
fine.

`ADC_RANGES` now carries `(span, volts_at_count_zero)` per range and
`Layout.volt_offset` exposes it; `assemble()` and `channel_health()` take it as
an argument, and every capture records it so an old file still reads back
correctly. `probe_uut()` refuses a range it does not have a table entry for
rather than guessing. So switching the range is now safe — it is still not a
resolution win, for the reasons above.

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

---

*Part of the [OCTO-BEE documentation](../README.md).*
