#!/usr/bin/env python3
"""
octobee_posecal.py -- calibrate the probe against the Earth's magnetic field by
rolling it about the tube axis.

Why bother, when there is already a magnet pass?
------------------------------------------------
`Calibration.cross_calibrate()` matches sensors from one magnet pass, and its
accuracy is limited by GEOMETRY, not by noise: it has to divide out the 1/r^3
weight from `Geometry.expected_response()`, so a 2 mm error in where you think
the magnet was, at r ~ 78 mm, lands as 3 * (2/78) ~ 7.7 % of gain error.

The Earth's field has no 1/r^3 term to divide out. Over a head this size
(FSV shell is 155 mm across) it is uniform to within nanotesla -- its gradient
is nT/m. So every sensor is guaranteed to be sitting in the *same* field
vector, and the only thing that can make two sensors disagree is the sensors.
That turns inter-sensor matching from a geometry problem into a noise problem,
which is worth about an order of magnitude.

It is also the only field source that covers the whole head at once. A
Helmholtz pair is ~1 % uniform only inside ~0.3 R, so spanning a 77.7 mm FSV
radius would need R ~ 260 mm coils at ~290 A-turns per mT. A coil is the right
tool for absolute scale on ONE arm tip; it is the wrong tool for all sixteen.

What rolling can and cannot see
-------------------------------
Rolling about the tube axis leaves the axial component of the field constant.
Writing the per-sensor response as `m_i = M_i @ B_tube + b_i`:

    m_i = M[:,0]*Bx + M[:,1]*By + ( M[:,2]*Bz + b_i )
                                   `-- inseparable --'

The third column of M and the offset are perfectly degenerate, and NO number of
roll angles fixes it. On this probe that is not an abstract worry: with
`mount_style = "tangential"`, `arm_dir` is circumferential and `board_normal`
is radial, so `e2 = cross(e3, e1)` lands exactly on the tube axis and **chip +Y
is axial on all 16 sensors**. Roll alone therefore calibrates chip X and Z and
leaves chip Y untouched -- a third of the channels.

The fix needs no fixture: record a second sweep with the tube lifted out of the
cradle and put back end-for-end. Bz flips sign, so

    b_i        = (sweep_A + sweep_B) / 2
    M[:,2]*Bz  = (sweep_A - sweep_B) / 2

There is a SECOND degeneracy, and it is easy to miss because the fit residual
sits right on the noise floor while the answer is wrong. Scaling the tube frame
by `T = diag(a, a, c)` maps the swept circles onto circles of the same family,
so the solver can trade transverse scale against axial scale for free. Fixing
|B| leaves the constraint

    a^2 * Bt^2 + c^2 * Bz^2 = |B|^2

which is ONE equation in two unknowns. The end-for-end flip does not help:
it sends Bz -> -Bz, and Bz^2 is unchanged, so it hands back the same equation.
Two sweeps at genuinely different INCLINATIONS give two independent equations
and force a = c = 1. In practice that means turning the cradle to point a
different way and rolling again -- the angle does not need to be known, only to
be different.

So the full procedure is three sweeps:

    A   as mounted
    B   lifted out and replaced end-for-end   -> separates offset from axial
    C   cradle turned to another azimuth      -> separates transverse from axial

`solve_roll()` accepts any number of them and *says which columns and which
gauges it actually determined* rather than quietly fitting a degenerate
direction to noise. With only A and B it can fall back on the assumption that
the average chip is isotropic (`anisotropy="assume_isotropic"`), which is
defensible for a monolithic factory-trimmed 3-axis part but is an assumption,
and is labelled as one.

Worth knowing: this second gauge is COMMON MODE. It rescales every sensor the
same way, because diag(a, a, c) commutes with the rotations about the tube axis
that distinguish one face from another. So inter-sensor matching -- the thing
the exercise is mostly for -- is correct even when this gauge is unresolved.
What it corrupts is the absolute axial-vs-transverse sensitivity ratio.

You do not need to know the roll angle
--------------------------------------
All 16 sensors see the same B_tube at every instant, so the roll angle phi(t)
is just another unknown, solved alongside everything else. Hand-rolling
unevenly is fine; stopping halfway and starting again is fine.

What comes out
--------------
The solver returns `M_i` (tube -> chip). Its inverse `C_i = M_i^-1` maps chip
readings to the tube frame, and a polar decomposition splits that into the two
things the rest of the codebase keeps separate:

    C_i = R_i @ S_i        R_i orthogonal -> the geometric rotation
                           S_i symmetric  -> gain + non-orthogonality

so `S_i` goes into `Calibration.matrix` (a chip-frame correction, leaving
`to_mt()` chip-frame as it has always been) and `R_i` is reported against
`probe_geometry.json` as a check on the mapping. Keeping the split means
nothing downstream -- `to_tube_frame()`, the 3D view, the CSV exports -- has to
change its idea of what frame it is in.

Nothing here talks to the hardware. It operates on decoded arrays, so it can be
unit-tested and replayed against saved captures.
"""

import json
import os

import numpy as np

import octobee_calibration as ocal
import probe_geometry as pg

N_SENSORS = pg.N_SENSORS
AXES = ocal.AXES

# The Earth's total field, microtesla. Central Europe is ~49 uT at ~65 deg
# inclination. This is the ONLY place absolute scale enters the solve, so it is
# worth getting from a real model rather than this default: NOAA
# (ngdc.noaa.gov/geomag/calculators/magcalc.shtml) or BGS, for your lat/lon and
# today's date. An error here scales every gain by the same factor; it does NOT
# affect inter-sensor matching, offsets, or orientation, which are all
# determined without reference to it.
DEFAULT_B_EARTH_UT = 49.0

# Solving on 10^7 raw samples is pointless: the information is in the shape of
# the ellipse, not the sample count. Block-average down to this many points.
# At 2 turns over 60 s, 4000 points is 0.18 deg apart, and averaging across
# that smears the amplitude by under a part per million.
SOLVE_POINTS = 4000

# Below this transverse amplitude (mT) a sensor cannot pin the roll angle and
# is not trusted to seed it. 0.005 mT = 5 uT, a tenth of the Earth's field.
MIN_SEED_AMPLITUDE_MT = 0.005


# --------------------------------------------------------------------------
# capture container
# --------------------------------------------------------------------------

class RollSweep:
    """
    One hand-rolled sweep: (n, 16, 3) uncorrected millitesla.

    "Uncorrected" means exactly what `Calibration.to_mt(..., apply_zero=False,
    apply_gain=False)` produces -- VCM subtracted and divided by the nominal
    V/T for the configured range, but with no trim and no tare. That keeps M_i
    dimensionless and close to a rotation, which is what makes the solve
    well conditioned, and it means the solved offset `b` is in mT and directly
    comparable with the old `zero_mt`.

    tag         which MOUNTING ORIENTATION this sweep was rolled in. Sweeps
                sharing a tag share one field inclination and are pooled.
                Convention: "A" as mounted, "B" lifted out and replaced
                end-for-end, "C"/"D"/... the cradle turned to other azimuths.
                Any label works; only distinctness matters.
    ranges_mt   (16,) the ranges in force during the sweep, so a solution can
                never be applied to a capture taken at a different gain.
    temps_c     (16,) ASIC temperatures, recorded but not yet corrected for --
                see README on the <100 ppm/degC sensitivity TC. Logging them now
                means a correction can be fitted later without re-rolling.
    """

    def __init__(self, b_mt, tag="A", ranges_mt=None, temps_c=None, meta=None):
        self.b_mt = np.asarray(b_mt, float)
        if self.b_mt.ndim != 3 or self.b_mt.shape[1:] != (N_SENSORS, 3):
            raise ValueError(f"b_mt must be (n, {N_SENSORS}, 3), got "
                             f"{self.b_mt.shape}")
        self.tag = str(tag).upper()
        self.ranges_mt = (None if ranges_mt is None else
                          np.asarray(ranges_mt, float).reshape(N_SENSORS))
        self.temps_c = (None if temps_c is None else np.asarray(temps_c, float))
        self.meta = dict(meta or {})

    def __len__(self):
        return self.b_mt.shape[0]

    def reduced(self, n_points=SOLVE_POINTS):
        """Block-average down to about n_points rows. Averaging over a small
        arc of roll is harmless; see SOLVE_POINTS."""
        n = len(self)
        factor = max(1, n // max(1, int(n_points)))
        return ocal.decimate(self.b_mt, factor)

    def amplitudes(self):
        """(16,) peak-to-peak-ish transverse amplitude per sensor, mT.

        Used to pick a seed sensor and to spot a sensor that saw nothing --
        a dead ribbon looks exactly like a sensor whose arm never moved."""
        x = self.b_mt - self.b_mt.mean(axis=0, keepdims=True)
        return np.sqrt((x ** 2).sum(axis=2).mean(axis=0)) * np.sqrt(2.0)

    # ---- persistence ----------------------------------------------------
    def save(self, path):
        """Write <path>.npz (bulk) alongside <path>.json (provenance)."""
        base = os.path.splitext(path)[0]
        np.savez_compressed(base + ".npz", b_mt=self.b_mt.astype(np.float32))
        side = {"tag": self.tag,
                "n_samples": int(len(self)),
                "ranges_mt": (None if self.ranges_mt is None
                              else self.ranges_mt.tolist()),
                "temps_c": (None if self.temps_c is None
                            else self.temps_c.tolist()),
                "meta": self.meta}
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(side, f, indent=2)
        return base + ".npz"

    @classmethod
    def load(cls, path):
        base = os.path.splitext(path)[0]
        with np.load(base + ".npz") as z:
            b = z["b_mt"].astype(np.float64)
        with open(base + ".json", encoding="utf-8") as f:
            side = json.load(f)
        return cls(b, tag=side["tag"], ranges_mt=side.get("ranges_mt"),
                   temps_c=side.get("temps_c"), meta=side.get("meta"))


# --------------------------------------------------------------------------
# solution container
# --------------------------------------------------------------------------

class PoseSolution:
    """
    What came out of solve_roll().

    M           (16,3,3)  tube frame -> chip reading
    b           (16,3)    sensor offset, mT, in the chip frame
    identified  (16,3)    which COLUMNS of M the data actually determined.
                          Column 2 is False unless two distinct field
                          inclinations were supplied -- see the module
                          docstring.
    anisotropy_identified
                bool      whether the transverse-vs-axial gauge was pinned by
                          the data. Needs two distinct |inclination| values,
                          which an end-for-end flip alone does NOT provide.
    bt, bz      (G,)      transverse and axial field per orientation, mT
    tags        (G,)      the orientation labels, in the same order
    residual_mt (16,)     RMS fit residual per sensor
    """

    def __init__(self, M, b, identified, bt, bz, residual_mt, phis,
                 b_earth_ut, ranges_mt=None, n_sweeps=1, notes="",
                 tags=None, anisotropy_identified=True,
                 offset_leverage=0.0):
        self.M = np.asarray(M, float).reshape(N_SENSORS, 3, 3)
        self.b = np.asarray(b, float).reshape(N_SENSORS, 3)
        self.identified = np.asarray(identified, bool).reshape(N_SENSORS, 3)
        self.bt = np.atleast_1d(np.asarray(bt, float))
        self.bz = np.atleast_1d(np.asarray(bz, float))
        self.residual_mt = np.asarray(residual_mt, float).reshape(N_SENSORS)
        self.phis = list(phis)
        self.b_earth_ut = float(b_earth_ut)
        self.ranges_mt = (None if ranges_mt is None
                          else np.asarray(ranges_mt, float).reshape(N_SENSORS))
        self.n_sweeps = int(n_sweeps)
        self.tags = list(tags) if tags is not None else ["A"]
        self.anisotropy_identified = bool(anisotropy_identified)
        self.offset_leverage = float(offset_leverage)
        self.notes = notes

    # ---- the physically meaningful split --------------------------------
    # A sensor that saw nothing -- S16's analogue ribbon is the live example --
    # fits an M with no rank, and inverting the whole (16,3,3) stack at once
    # would then raise and take the other fifteen sensors down with it. Invert
    # per sensor, fall back to a pseudo-inverse, and record who failed so the
    # report can say so instead of quietly returning nonsense.
    SINGULAR_COND = 1e6

    def chip_to_tube(self):
        """(16,3,3) C_i = M_i^-1, mapping a chip reading into the tube frame."""
        C = np.empty_like(self.M)
        bad = []
        for i in range(N_SENSORS):
            if np.linalg.cond(self.M[i]) > self.SINGULAR_COND:
                bad.append(f"S{i + 1}")
                C[i] = np.linalg.pinv(self.M[i])
                continue
            try:
                C[i] = np.linalg.inv(self.M[i])
            except np.linalg.LinAlgError:
                bad.append(f"S{i + 1}")
                C[i] = np.linalg.pinv(self.M[i])
        self._singular = bad
        return C

    @property
    def singular(self):
        """Sensors whose response could not be inverted -- treat as unusable."""
        if not hasattr(self, "_singular"):
            self.chip_to_tube()
        return list(self._singular)

    def decompose(self):
        """
        Polar-decompose C_i = R_i @ S_i.

        R_i is the orthogonal (rotation) part -- the geometry. S_i is the
        symmetric positive-definite part -- gain and non-orthogonality, which
        is what belongs in a chip-frame correction. Returns (R, S), both
        (16,3,3).
        """
        C = self.chip_to_tube()
        R = np.empty_like(C)
        S = np.empty_like(C)
        for i in range(N_SENSORS):
            W, sv, Vt = np.linalg.svd(C[i])
            V = Vt.T
            # Keep the rotation proper. A reflection here means the solve
            # settled into a mirrored frame, which is a real (if rare) failure
            # mode of an unanchored fit, not something to silently accept.
            D = np.eye(3)
            if np.linalg.det(W @ Vt) < 0:
                D[2, 2] = -1.0
            R[i] = W @ D @ Vt
            S[i] = V @ D @ np.diag(sv) @ Vt
        return R, S

    def gains(self):
        """
        (16,) sensitivity of each sensor relative to the set median.

        Above 1.0 means the sensor reads high for the same physical field. This
        is the number the whole exercise is about, so it comes from the
        symmetric part only -- a rotation cannot change a gain, and letting one
        leak in is how a geometry error gets misread as a gain error.

        NaN for a sensor whose correction came out with a negative determinant.
        That means the solve put it in a mirrored frame, which is a genuine
        failure (usually a dead or badly wired channel), not a small number to
        be rounded away.
        """
        _, S = self.decompose()
        det = np.array([np.linalg.det(S[i]) for i in range(N_SENSORS)])
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(det > 0, np.abs(det) ** (1.0 / 3.0), np.nan)
            g = 1.0 / scale
        return g / np.nanmedian(g)

    def orthogonality_deg(self):
        """(16,3) departure from 90 deg between each pair of chip axes."""
        _, S = self.decompose()
        out = np.empty((N_SENSORS, 3))
        for i in range(N_SENSORS):
            e = S[i]
            for k, (p, q) in enumerate(((0, 1), (1, 2), (2, 0))):
                cp = e[:, p] / np.linalg.norm(e[:, p])
                cq = e[:, q] / np.linalg.norm(e[:, q])
                out[i, k] = 90.0 - np.degrees(
                    np.arccos(np.clip(abs(cp @ cq), -1.0, 1.0)))
        return out

    @property
    def b_total_mt(self):
        return float(np.hypot(self.bt[0], self.bz[0]))

    @property
    def inclination_deg(self):
        """(G,) angle of the field out of the roll plane, per orientation.

        Not the geomagnetic inclination unless the tube happened to be level
        and pointing magnetic north. It is still the number to look at: two
        orientations whose |inclination| differ are what pins the
        transverse-vs-axial gauge, and an end-for-end flip should come back at
        very nearly minus the original."""
        return np.degrees(np.arctan2(self.bz, self.bt))

    # ---- persistence ----------------------------------------------------
    def to_dict(self):
        return {"M": self.M.tolist(),
                "b": self.b.tolist(),
                "identified": self.identified.tolist(),
                "bt": self.bt.tolist(), "bz": self.bz.tolist(),
                "tags": self.tags,
                "anisotropy_identified": self.anisotropy_identified,
                "offset_leverage": self.offset_leverage,
                "residual_mt": self.residual_mt.tolist(),
                "b_earth_ut": self.b_earth_ut,
                "ranges_mt": (None if self.ranges_mt is None
                              else self.ranges_mt.tolist()),
                "n_sweeps": self.n_sweeps,
                "notes": self.notes}

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["M"], d["b"], d["identified"], d["bt"], d["bz"],
                   d["residual_mt"], [], d["b_earth_ut"],
                   ranges_mt=d.get("ranges_mt"),
                   n_sweeps=d.get("n_sweeps", 1), notes=d.get("notes", ""),
                   tags=d.get("tags"),
                   anisotropy_identified=d.get("anisotropy_identified", True),
                   offset_leverage=d.get("offset_leverage", 0.0))

    def report(self, dead=()):
        g = self.gains()
        ortho = self.orthogonality_deg()
        incl = self.inclination_deg
        lines = [f"sweeps: {self.n_sweeps} in {len(self.tags)} orientation(s) "
                 f"({', '.join(self.tags)})",
                 f"|B| = {self.b_total_mt * 1e3:.2f} uT, inclination to the "
                 f"roll plane per orientation: "
                 + ", ".join(f"{t}={a:+.1f} deg" for t, a in zip(self.tags, incl)),
                 f"axial column identified:     {bool(self.identified[:, 2].all())}",
                 f"transverse/axial gauge:      "
                 f"{'measured' if self.anisotropy_identified else 'NOT MEASURED'}",
                 f"offset leverage:             {self.offset_leverage:.2f} "
                 f"({'good' if self.offset_leverage > 0.6 else 'WEAK'}; "
                 f"an end-for-end flip gives the most)",
                 f"gain spread: {(np.nanmax(g) - np.nanmin(g)) * 100:.2f} % "
                 f"peak-to-peak about the median",
                 ""]
        bad = set(self.singular)
        if bad:
            lines.append(f"UNUSABLE (no invertible response): "
                         f"{', '.join(sorted(bad))}")
            lines.append("")
        lines.append("  sensor   gain    resid_uT   max non-ortho")
        for i in range(N_SENSORS):
            sid = f"S{i + 1}"
            flag = ("  (NO RESPONSE)" if sid in bad else
                    "  (excluded)" if sid in dead else "")
            lines.append(f"  {sid:<6}  {g[i]:6.4f}  {self.residual_mt[i] * 1e3:8.3f}"
                         f"   {np.nanmax(np.abs(ortho[i])):6.2f} deg{flag}")
        if not self.identified[:, 2].all():
            lines += ["",
                      "WARNING: the axial column of M was not identified by the",
                      "data and has been filled in from the nominal geometry.",
                      "On this probe that is chip +Y on every sensor. Record a",
                      "sweep with the tube reversed end-for-end to fix it."]
        if self.identified[:, 2].all() and self.offset_leverage < 0.6:
            lines += ["",
                      "WARNING: the axial field barely changed between",
                      "orientations, so the offset and the axial column are",
                      "only weakly separated and the axial sensitivity will be",
                      "biased. Add a sweep with the tube reversed end-for-end;",
                      "two cradle azimuths on the same side of level are not a",
                      "substitute."]
        if not self.anisotropy_identified:
            lines += ["",
                      "WARNING: transverse-vs-axial sensitivity was not pinned by",
                      "the data. Every sweep shared one |inclination|, and an",
                      "end-for-end flip does not change that. Roll again with the",
                      "cradle turned to another azimuth. This gauge is COMMON",
                      "MODE, so inter-sensor matching below is still valid; what",
                      "is uncertain is the absolute axial-vs-transverse ratio."]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# the solver
# --------------------------------------------------------------------------

def _seed_phi(m, M_nom, live):
    """
    First guess at the roll angle from all live sensors at once.

    Each sensor's own nominal rotation turns its reading into an estimate of
    B_tube; averaging those estimates is far steadier than trusting whichever
    single sensor looks healthiest, and it costs nothing.
    """
    x = m - m.mean(axis=0, keepdims=True)              # (n,16,3)
    # M_nom maps tube -> chip, so its transpose (it is a rotation) maps back.
    est = np.einsum("sji,tsj->tsi", M_nom[live], x[:, live, :])
    mean_est = est.mean(axis=1)                        # (n,3)
    return np.arctan2(mean_est[:, 1], mean_est[:, 0])


def _fit_sensor(design, y):
    """Least squares for one sensor: returns (M_cols, offset)."""
    sol, *_ = np.linalg.lstsq(design, y, rcond=None)
    return sol[:-1].T, sol[-1]


def _update_phi(m, M, b, bt, bz, live, phi):
    """
    Re-estimate the roll angle of every sample, given the sensors.

    The residual for one sample is

        sum_i || u_i cos(phi) + v_i sin(phi) + w_i ||^2

    which flattens to  a cos^2 + b sin^2 + c sin cos + d cos + e sin + const.
    Once M is anywhere near orthogonal, a ~ b and c ~ 0, so the stationary
    point is essentially atan2(-e, -d); a couple of Newton steps on the exact
    derivative mop up the rest. That is much cheaper, and better conditioned,
    than a grid search over every sample.
    """
    u = M[live, :, 0] * bt                              # (L,3)
    v = M[live, :, 1] * bt
    w = M[live, :, 2] * bz + b[live] - m[:, live, :]     # (n,L,3)

    a = float((u * u).sum())
    bb = float((v * v).sum())
    c = 2.0 * float((u * v).sum())
    d = 2.0 * np.einsum("ij,tij->t", u, w)
    e = 2.0 * np.einsum("ij,tij->t", v, w)

    phi = np.arctan2(-e, -d)
    for _ in range(3):
        s2, c2 = np.sin(2 * phi), np.cos(2 * phi)
        s1, c1 = np.sin(phi), np.cos(phi)
        f1 = -(a - bb) * s2 + c * c2 - d * s1 + e * c1
        f2 = -2 * (a - bb) * c2 - 2 * c * s2 - d * c1 - e * s1
        step = np.where(np.abs(f2) > 1e-12, f1 / np.where(np.abs(f2) > 1e-12, f2, 1.0), 0.0)
        phi = phi - np.clip(step, -0.5, 0.5)
    return phi


def solve_roll(sweeps, geometry=None, b_earth_ut=DEFAULT_B_EARTH_UT,
               dead=(), iters=60, n_points=SOLVE_POINTS, tol=1e-10,
               anisotropy="solve"):
    """
    Solve per-sensor response from one or more roll sweeps.

    sweeps      RollSweep, or a list of them. Sweeps sharing a tag are pooled
                and share one field inclination; each distinct tag gets its own.
                Two distinct inclinations identify the axial column, and two
                distinct |inclination| values additionally pin the
                transverse-vs-axial gauge. See the module docstring: the
                end-for-end flip gives the first but NOT the second.
    geometry    probe_geometry.Geometry for the nominal orientations. Used to
                seed the solve, to anchor the leftover rotation gauge, and to
                fill the axial column when it is not identified.
    b_earth_ut  total field magnitude, microtesla. Fixes absolute scale ONLY.
    dead        sensor ids ("S16") to leave out of the shared field estimate.
                They are still fitted, so you get numbers for them, they just
                do not get a vote on where the field is pointing.
    anisotropy  "solve"            require the data to pin the gauge (default);
                                   if it cannot, leave it and flag it.
                "assume_isotropic" fall back on the average chip having equal
                                   sensitivity on all three axes. Reasonable
                                   for a monolithic factory-trimmed part, but
                                   an assumption, and recorded as one.

    Returns a PoseSolution.
    """
    if isinstance(sweeps, RollSweep):
        sweeps = [sweeps]
    sweeps = list(sweeps)
    if not sweeps:
        raise ValueError("need at least one sweep")
    if geometry is None:
        geometry = pg.Geometry.load_or_default()

    ranges = next((s.ranges_mt for s in sweeps if s.ranges_mt is not None), None)
    for s in sweeps:
        if (ranges is not None and s.ranges_mt is not None
                and not np.allclose(ranges, s.ranges_mt)):
            raise ValueError(
                "sweeps were taken at different ranges; the EEPROM holds a "
                "separate factory dataset per gain, so they cannot be combined")

    dead = {str(d) for d in dead}
    live = np.array([f"S{i + 1}" not in dead for i in range(N_SENSORS)])
    if live.sum() < 2:
        raise ValueError("need at least two live sensors to fix the roll angle")

    # Nominal tube -> chip is the transpose of geometry's chip -> tube.
    M_nom = np.transpose(geometry.rotations(), (0, 2, 1)).copy()

    data = [s.reduced(n_points) for s in sweeps]
    tags = [s.tag for s in sweeps]
    groups = sorted(set(tags))
    gidx = [groups.index(t) for t in tags]

    b_total_mt = float(b_earth_ut) * 1e-3
    phis = [_seed_phi(m, M_nom, live) for m in data]

    # Seed each orientation's field split from the nominal geometry: project
    # onto the tube frame and see how much of it never moved during the roll.
    alpha = np.zeros(len(groups))
    for g in range(len(groups)):
        sel = [m for m, gg in zip(data, gidx) if gg == g]
        seed = np.concatenate(
            [np.einsum("sji,tsj->tsi", M_nom[live], m[:, live, :]) for m in sel])
        amp = np.sqrt(np.mean(seed[..., 0].var(axis=0) + seed[..., 1].var(axis=0)))
        alpha[g] = np.arctan2(np.mean(seed[..., 2]), max(amp, 1e-12))

    M = M_nom.copy()
    b = np.zeros((N_SENSORS, 3))
    prev = np.inf

    for _ in range(int(iters)):
        bt = b_total_mt * np.cos(alpha)
        bz = b_total_mt * np.sin(alpha)

        # ---- step A: sensors, given the field -----------------------------
        X = np.concatenate([
            np.column_stack([bt[g] * np.cos(phi), bt[g] * np.sin(phi),
                             np.full(len(phi), bz[g]), np.ones(len(phi))])
            for phi, g in zip(phis, gidx)])
        Y = np.concatenate(data)

        if _axial_rank(bz[np.array(gidx)]) < 2:
            # Every sample shares one axial value, so the axial column is
            # exactly collinear with the intercept. Drop it rather than let
            # lstsq hand back an arbitrary split of the two.
            for i in range(N_SENSORS):
                cols, off = _fit_sensor(X[:, [0, 1, 3]], Y[:, i, :])
                M[i, :, 0], M[i, :, 1] = cols[:, 0], cols[:, 1]
                M[i, :, 2] = M_nom[i, :, 2]
                b[i] = off - M[i, :, 2] * bz[gidx[0]]
        else:
            for i in range(N_SENSORS):
                M[i], b[i] = _fit_sensor(X, Y[:, i, :])

        # ---- step B: field, given the sensors -----------------------------
        phis = [_update_phi(m, M, b, bt[g], bz[g], live, phi)
                for m, g, phi in zip(data, gidx, phis)]

        # Refit each orientation's (cos, sin) pair unconstrained -- only the
        # RATIO is observable -- then restore the magnitude from |B|. That is
        # gauge fixing, not fitting, hence a rescale rather than a constrained
        # solve.
        for g in range(len(groups)):
            pa, qa, ta = [], [], []
            for m, gg, phi in zip(data, gidx, phis):
                if gg != g:
                    continue
                p = (M[live][:, :, 0][None] * np.cos(phi)[:, None, None]
                     + M[live][:, :, 1][None] * np.sin(phi)[:, None, None])
                q = np.broadcast_to(M[live][:, :, 2][None], p.shape)
                t = m[:, live, :] - b[live][None]
                pa.append(p.reshape(-1)); qa.append(q.reshape(-1))
                ta.append(t.reshape(-1))
            A = np.column_stack([np.concatenate(pa), np.concatenate(qa)])
            s, *_ = np.linalg.lstsq(A, np.concatenate(ta), rcond=None)
            alpha[g] = np.arctan2(s[1], s[0])

        # ---- convergence --------------------------------------------------
        rms = float(np.sqrt(np.mean(_residuals(
            data, gidx, phis, M, b,
            b_total_mt * np.cos(alpha), b_total_mt * np.sin(alpha)) ** 2)))
        if abs(prev - rms) < tol * max(rms, 1e-12):
            break
        prev = rms

    bt = b_total_mt * np.cos(alpha)
    bz = b_total_mt * np.sin(alpha)
    axial_ok = _axial_rank(bz[np.array(gidx)]) >= 2
    # How well the offset is separated from the axial column. The two are only
    # told apart by Bz MOVING, so the useful figure is the spread of Bz across
    # orientations as a fraction of |B|. Reversing the tube end-for-end gives
    # the largest possible spread, which is why it is the recommended second
    # orientation: two azimuths on the same side of level both have Bz > 0 and
    # leave the offset poorly pinned, and that error then leaks straight into
    # the axial column.
    offset_leverage = float((bz.max() - bz.min()) / max(b_total_mt, 1e-12))
    # The transverse-vs-axial gauge needs two distinct |inclination| values.
    # An end-for-end flip sends bz -> -bz and leaves bz**2 alone, so it does
    # NOT count here even though it does count for the axial column.
    aniso_ok = _axial_rank(bz[np.array(gidx)] ** 2) >= 2

    identified = np.ones((N_SENSORS, 3), bool)
    identified[:, 2] = axial_ok

    resid = _residuals(data, gidx, phis, M, b, bt, bz)
    sol = PoseSolution(M, b, identified, bt, bz, resid, phis, b_earth_ut,
                       ranges_mt=ranges, n_sweeps=len(sweeps),
                       tags=groups, anisotropy_identified=aniso_ok,
                       offset_leverage=offset_leverage)
    if not aniso_ok and anisotropy == "assume_isotropic":
        _assume_isotropic(sol)
    _anchor_gauge(sol, geometry)
    return sol


def _axial_rank(values, rel_tol=0.05):
    """How many genuinely distinct values are in here.

    Two orientations a degree apart do not identify anything, so 'distinct'
    has to mean 'separated by a useful fraction of the field', not merely
    'not bitwise equal'.
    """
    v = np.asarray(values, float)
    scale = max(np.abs(v).max(), 1e-12)
    keep = []
    for x in v:
        if not any(abs(x - k) < rel_tol * scale for k in keep):
            keep.append(x)
    return len(keep)


def _assume_isotropic(sol):
    """
    Fix the transverse-vs-axial gauge by assuming the average chip is isotropic.

    The SENM3Dx is one monolithic part whose three axes are trimmed together in
    the factory dataset, so "the median chip has the same sensitivity on all
    three axes" is a fair prior. It is still a prior, so it is applied only on
    request and recorded in the solution's notes.

    Chip +Y is the tube axis on every sensor here, so the gauge shows up as the
    median S[1,1] drifting away from the median of S[0,0] and S[2,2]. Rescaling
    the tube frame's axial direction pulls it back.
    """
    for _ in range(12):
        _, S = sol.decompose()
        yy = np.median(S[:, 1, 1])
        xz = np.median(0.5 * (S[:, 0, 0] + S[:, 2, 2]))
        if not np.isfinite(yy) or not np.isfinite(xz) or yy <= 0 or xz <= 0:
            break
        # Stretching the tube frame's axial direction by tau scales the chip's
        # axial correction by 1/tau, and chip +Y is the tube axis here, so
        # S[1,1] -> S[1,1]/k. Setting that equal to the transverse median gives
        # k = yy / xz.
        k = yy / xz
        if abs(k - 1.0) < 1e-12:
            break
        # M maps tube -> chip; stretching the tube frame's axial direction by k
        # is M @ diag(1, 1, k).
        sol.M = sol.M * np.array([1.0, 1.0, k])[None, None, :]
        sol.bz = sol.bz / k
        norm = np.hypot(sol.bt, sol.bz)
        good = norm > 0
        scale = np.where(good, sol.b_earth_ut * 1e-3 / np.where(good, norm, 1.0), 1.0)
        sol.bt = sol.bt * scale
        sol.bz = sol.bz * scale
    sol.notes = (sol.notes + " Transverse-vs-axial gauge was NOT identified by "
                 "the data; it was fixed by assuming the median chip is "
                 "isotropic. Roll in a second cradle azimuth to measure it "
                 "instead.").strip()
    return sol


def _residuals(data, gidx, phis, M, b, bt, bz):
    """(16,) RMS residual per sensor over every sample of every sweep."""
    acc = np.zeros(N_SENSORS)
    n = 0
    for m, g, phi in zip(data, gidx, phis):
        B = np.column_stack([bt[g] * np.cos(phi), bt[g] * np.sin(phi),
                             np.full(len(phi), bz[g])])
        pred = np.einsum("sij,tj->tsi", M, B) + b[None]
        acc += ((m - pred) ** 2).sum(axis=(0, 2))
        n += len(phi) * 3
    return np.sqrt(acc / max(n, 1))


def _anchor_gauge(sol, geometry):
    """
    Remove the one gauge freedom the data cannot see.

    Roll is about the tube axis, so the solved frame's z IS the tube axis --
    that part is pinned by the physics. What is NOT pinned is where the frame's
    x axis points, because it was set by wherever the tube happened to be when
    the first sweep started. That is a free rotation about z, and it shuffles
    the faces.

    Anchoring it to the nominal geometry is a CONVENTION, not a measurement.
    Which sensors share a face, and their order around the tube, are genuinely
    determined by the data; a global relabelling of all four faces at once is
    not, and cannot be, from the Earth's field alone.
    """
    R, _ = sol.decompose()
    R_nom = geometry.rotations()
    # Best single rotation about z aligning solved to nominal: maximise
    # sum_i trace(Rz(g)^T R_nom_i R_i^T) -> closed form in atan2.
    s = c = 0.0
    bad = set(sol.singular)
    for i in range(N_SENSORS):
        if f"S{i + 1}" in bad:
            continue        # a sensor that saw nothing gets no say in the frame
        Q = R_nom[i] @ R[i].T
        c += Q[0, 0] + Q[1, 1]
        s += Q[1, 0] - Q[0, 1]
    gamma = np.arctan2(s, c)
    cg, sg = np.cos(gamma), np.sin(gamma)
    Rz = np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]])
    # M maps tube -> chip, so rotating the tube frame by Rz means M @ Rz.
    sol.M = np.einsum("sij,jk->sik", sol.M, Rz)
    sol.notes = (sol.notes + f" gauge anchored by {np.degrees(gamma):+.2f} deg "
                             "about the tube axis.").strip()
    return sol


# --------------------------------------------------------------------------
# what the solution says about the geometry
# --------------------------------------------------------------------------

def identify_faces(solution, geometry=None):
    """
    Work out which face each sensor is really on.

    All 16 chips see the same B_tube, so two chips on different faces report
    vectors differing by a 90 deg rotation about the tube axis. The solved
    rotations therefore fall into four clusters, one per face, and that is the
    sensor -> face mapping -- which `probe_geometry.json` currently records as
    an assumption.

    Returns a dict with `face` (16,) solved face index, `nominal` (16,),
    `angle_deg` (16,) the solved rotation of each chip about the tube axis,
    `mismatch` list of sensor ids that disagree, and `agrees` bool.

    Two honest limits:
      * Slot (position ALONG the tube) is invisible here -- every slot on a
        face shares the same rotation. Use octobee_idmap.py --magnet for that.
      * The mapping is recovered up to a global rotation of all four faces at
        once; see _anchor_gauge.
    """
    if geometry is None:
        geometry = pg.Geometry.load_or_default()
    R, _ = solution.decompose()
    R_nom = geometry.rotations()

    # Each face differs by a quarter turn about the tube axis. Compare each
    # solved rotation against the nominal rotation of a sensor known to be on
    # each face, and take the closest.
    face_ref = {}
    for i in range(N_SENSORS):
        face_ref.setdefault(int(geometry.face(i + 1)), R_nom[i])

    faces = np.zeros(N_SENSORS, int)
    angles = np.zeros(N_SENSORS)
    for i in range(N_SENSORS):
        best, best_d = 0, np.inf
        for f, Rf in face_ref.items():
            # Frobenius distance is monotonic in rotation angle, so it ranks
            # candidates the same way a geodesic distance would, for less work.
            d = np.linalg.norm(R[i] - Rf)
            if d < best_d:
                best, best_d = f, d
        faces[i] = best
        angles[i] = np.degrees(np.arctan2(R[i][1, 0], R[i][0, 0]))

    nominal = np.array([int(geometry.face(i + 1)) for i in range(N_SENSORS)])
    mismatch = [f"S{i + 1}" for i in range(N_SENSORS) if faces[i] != nominal[i]]
    return {"face": faces, "nominal": nominal, "angle_deg": angles,
            "mismatch": mismatch, "agrees": not mismatch}


def ambient_uniformity(solution):
    """
    How uniform the field was across the head, as a fraction of |B|.

    The residual of "all 16 sensors must read the same vector" is, once the
    sensors themselves are calibrated, a 16-point map of the ambient gradient
    over the 155 mm span of the head. That makes the probe its own site survey
    tool: a spot that comes back much above ~0.005 here has ferrous structure
    close enough to matter, and the calibration taken there is not trustworthy.
    """
    live = solution.residual_mt > 0
    if not live.any():
        return float("nan")
    return float(np.median(solution.residual_mt[live]) / max(solution.b_total_mt, 1e-12))


# --------------------------------------------------------------------------
# carrying a calibration to another range
# --------------------------------------------------------------------------

def range_transfer(b_lo_mt, b_hi_mt, min_signal_mt=1.0):
    """
    Per-sensor, per-axis ratio between two gain settings.

    The Earth's field is 0.6 counts at the 400 mT range -- invisible. So a
    calibration solved at 20/40 mT cannot simply be declared valid up there.
    Park a magnet to make a field big enough to be well resolved at BOTH
    ranges (15-30 mT is the sweet spot), capture without moving anything,
    change gain, capture again.

    The magnet's absolute value never enters: only that it held still. Returns
    (ratio (16,3), skipped list) where ratio multiplies a low-range gain to
    give the high-range one.

    Both arguments are (n,16,3) uncorrected mT, each converted with the
    NOMINAL V/T of the range it was taken at.
    """
    lo = np.asarray(b_lo_mt, float).mean(axis=0)
    hi = np.asarray(b_hi_mt, float).mean(axis=0)
    ok = np.abs(lo) > float(min_signal_mt)
    ratio = np.ones((N_SENSORS, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = hi / lo
    ratio[ok] = r[ok]
    skipped = [f"S{i + 1}.{AXES[k]}"
               for i in range(N_SENSORS) for k in range(3) if not ok[i, k]]
    return ratio, skipped


# --------------------------------------------------------------------------
# CLI -- solve from saved sweeps
# --------------------------------------------------------------------------

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweeps", nargs="+",
                    help="RollSweep .npz files. Three sweeps for a complete "
                         "solve: A as mounted, B with the tube end-for-end "
                         "(separates offset from axial response), C with the "
                         "cradle at another azimuth (pins transverse-vs-axial; "
                         "the flip alone cannot)")
    ap.add_argument("--b-earth-ut", type=float, default=DEFAULT_B_EARTH_UT,
                    help=f"total field magnitude in uT (default {DEFAULT_B_EARTH_UT}); "
                         "look yours up at ngdc.noaa.gov/geomag")
    ap.add_argument("--dead", default="", help="comma-separated ids to ignore, e.g. S16")
    ap.add_argument("--anisotropy", choices=("solve", "assume_isotropic"),
                    default="solve",
                    help="what to do when no second azimuth was recorded: "
                         "'solve' leaves the transverse-vs-axial gauge "
                         "unresolved and says so (default); 'assume_isotropic' "
                         "fixes it by assuming the median chip has equal "
                         "sensitivity on all three axes")
    ap.add_argument("--geometry", default=pg.CONFIG_NAME)
    ap.add_argument("--save", help="write the PoseSolution here as JSON")
    ap.add_argument("--apply", metavar="CALFILE",
                    help="fold the solution into this calibration.json")
    a = ap.parse_args(argv)

    geom = pg.Geometry.load_or_default(a.geometry)
    dead = [d.strip() for d in a.dead.split(",") if d.strip()]
    sweeps = [RollSweep.load(p) for p in a.sweeps]
    sol = solve_roll(sweeps, geom, a.b_earth_ut, dead=dead,
                     anisotropy=a.anisotropy)

    print(sol.report(dead=dead))
    print()
    ids = identify_faces(sol, geom)
    print("face mapping: " + ("agrees with probe_geometry.json"
                              if ids["agrees"]
                              else "DISAGREES at " + ", ".join(ids["mismatch"])))
    print(f"ambient uniformity: {ambient_uniformity(sol) * 100:.3f} % of |B|")

    if a.save:
        sol.save(a.save)
        print(f"\nwrote {a.save}")
    if a.apply:
        cal = ocal.Calibration.load_or_default(a.apply)
        cal.apply_pose_solution(sol)
        cal.save(a.apply)
        print(f"applied to {a.apply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
