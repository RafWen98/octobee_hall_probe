"""
Shared test helpers: the check() reporter, and the synthetic data builders.

check() records rather than raises, and conftest turns a test's collected
failures into one pytest failure at the end of it. That is deliberate: these
tests assert twenty or thirty things about one run of a calibration, and
stopping at the first is how you end up fixing them one per CI round instead
of seeing the shape of what broke.
"""

import json
import time

import numpy as np

from octobee.acq import carrier as ob
from octobee.calib import roll as opc
from octobee.motion import stage as ostage
from octobee.calib import magnet as omag
from octobee.calib import geometry as pgeom

# The measured single-sample noise, averaged down over one solve block. A
# module-level constant rather than an expression in the signature, so it is
# evaluated once and can be referred to by name.
SYNTH_NOISE_MT = 8.81e-3 / np.sqrt(2500)

_FAILS = []
_SKIPS = []
_COUNT = [0]


def begin():
    """Start a fresh test. Called by the autouse fixture in conftest."""
    _FAILS.clear()


def failures():
    """What this test recorded as failed."""
    return list(_FAILS)


def totals():
    """(checks made across the run, names skipped)."""
    return _COUNT[0], list(_SKIPS)


def check(name, cond, detail=""):
    _COUNT[0] += 1
    ok = bool(cond)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        _FAILS.append(name + (f"  -- {detail}" if detail else ""))
    return ok


def skip(name, why):
    """Record a check that could not run here, without calling it a pass.

    Exactly one thing in this suite needs software a machine may legitimately
    not have: the Thorlabs Kinesis DLL is Windows-only and proprietary, so a
    Linux CI runner cannot load it and never will. Failing there makes the
    quality gate permanently red for a reason that says nothing about the
    code; passing quietly would let a bench run that stopped testing the
    binding look identical to one that did.

    So it is counted separately and named in the summary. A run with skips
    checked less than a full one, and says so.
    """
    _SKIPS.append(name)
    print(f"  [skip] {name}  -- {why}")
    return False


def read_csv(path):
    """
    Read one of our CSVs into (column names, float array).

    numpy's genfromtxt(names=True) takes the first line as the header even when
    it is a comment, so it chokes on the provenance block. Parsing it here also
    checks the file is readable the way a person would read it.
    """
    names, rows = None, []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if names is None:
                names = line.split(",")
            else:
                rows.append([float(v) for v in line.split(",")])
    return names, np.array(rows)


def read_clkdiv(host):
    u = ob.Uut(host)
    try:
        return u.value("clkdiv", site=1)
    finally:
        u.close()


def pump(win, app, seconds):
    """Run the acquisition loop for a while, as the real timers would."""
    end = time.time() + seconds
    n = 0
    while time.time() < end:
        win.on_tick()
        n += 1
        if n % 5 == 0:              # the view redraws on its own slower clock
            win.on_view_tick()
        app.processEvents()
        time.sleep(0.01)
    win.on_view_tick()
    win.on_slow_tick()
    app.processEvents()


def _framed(name, invert, origin=None, lo=0.0, hi=300.0):
    """A Stage with its travel filled in as if it had been opened."""
    st = ostage.Stage("45000000", name=name, invert=invert, origin_mm=origin)
    st.travel_dev_mm = (lo, hi)
    st._resolve_frame()
    return st


def _synth_sweeps(geom, gains, axial_ratio=1.0, offsets_ut=20.0, seed=3,
                  noise_mt=SYNTH_NOISE_MT, n=3000):
    """
    Build roll sweeps from a known truth so the solver can be graded.

    axial_ratio is each chip's +Y (tube-axis) sensitivity relative to its X/Z.
    It is the quantity only a second cradle azimuth can see, so it is what the
    gauge tests turn on.
    """
    rng = np.random.default_rng(seed)
    n_sens = pgeom.N_SENSORS
    b_mt = 49.0e-3
    R_nom = geom.rotations()
    # S is the CORRECTION, so a sensor of sensitivity g needs S = 1/g.
    S_true = np.array([np.diag([1.0, 1.0 / axial_ratio, 1.0]) / gains[i]
                       for i in range(n_sens)])
    M_true = np.array([np.linalg.inv(R_nom[i] @ S_true[i]) for i in range(n_sens)])
    b_true = rng.normal(0, offsets_ut * 1e-3, (n_sens, 3))

    def sweep(tag, incl_deg):
        a = np.deg2rad(incl_deg)
        bt, bz = b_mt * np.cos(a), b_mt * np.sin(a)
        phi = np.sort(rng.uniform(0, 4 * np.pi, n))
        B = np.column_stack([bt * np.cos(phi), bt * np.sin(phi), np.full(n, bz)])
        m = np.einsum("sij,tj->tsi", M_true, B) + b_true[None]
        return opc.RollSweep(m + rng.normal(0, noise_mt, m.shape), tag=tag)

    return sweep, S_true, b_true


def _matching_error_pct(sol, S_true):
    """How badly the sensors disagree with each other, in percent.

    Deliberately built from the TRANSVERSE response only. That is the part no
    gauge can touch, so this measures inter-sensor matching without being
    polluted by whether the axial gauge happened to be resolved.
    """
    _, S = sol.decompose()
    n = pgeom.N_SENSORS
    got = np.array([np.sqrt(S[i][0, 0] * S[i][2, 2]) for i in range(n)])
    want = np.array([np.sqrt(S_true[i][0, 0] * S_true[i][2, 2]) for i in range(n)])
    rel = (got / np.median(got)) / (want / np.median(want))
    return float(np.abs(rel - 1).max() * 100)


def _matching_error_pct_subset(sol, S_true, idx):
    """_matching_error_pct restricted to sensors that are supposed to work."""
    _, S = sol.decompose()
    idx = list(idx)
    got = np.array([np.sqrt(S[i][0, 0] * S[i][2, 2]) for i in idx])
    want = np.array([np.sqrt(S_true[i][0, 0] * S_true[i][2, 2]) for i in idx])
    rel = (got / np.median(got)) / (want / np.median(want))
    return float(np.abs(rel - 1).max() * 100)


def _synth_pose_sweeps(geom, gains, n_poses=4, seconds=20.0, seed=7):
    """The same truth as _synth_sweeps, sampled at n_poses discrete angles.

    Noise is what a stationary pose actually averages down to: 81.5 uT rms per
    sample (README section 7) over seconds * 200 kSPS.
    """
    rng = np.random.default_rng(seed)
    n_sens = pgeom.N_SENSORS
    b_mt = 49.0e-3
    noise_mt = 81.5e-3 / np.sqrt(200000.0 * seconds)
    R_nom = geom.rotations()
    S_true = np.array([np.eye(3) / gains[i] for i in range(n_sens)])
    M_true = np.array([np.linalg.inv(R_nom[i] @ S_true[i]) for i in range(n_sens)])
    b_true = rng.normal(0, 20e-3, (n_sens, 3))

    def sweep(tag, incl_deg, jitter_deg=0.0):
        a = np.deg2rad(incl_deg)
        bt, bz = b_mt * np.cos(a), b_mt * np.sin(a)
        # An unknown start angle and an imperfectly square V-block: neither is
        # supposed to matter, because solve_roll fits every pose's angle.
        phi = (np.arange(n_poses) * (2 * np.pi / n_poses)
               + np.deg2rad(rng.normal(0, jitter_deg, n_poses))
               + rng.uniform(0, 2 * np.pi))
        B = np.column_stack([bt * np.cos(phi), bt * np.sin(phi),
                             np.full(n_poses, bz)])
        m = np.einsum("sij,tj->tsi", M_true, B) + b_true[None]
        return opc.RollSweep(m + rng.normal(0, noise_mt, m.shape), tag=tag)

    return sweep, S_true, b_true


def _simsopt_file(path, radius=1.0, shift=0.25):
    """A SIMSON file with two coils and two current configurations.

    Written by hand rather than by simsopt, which is the point: the reader in
    octobee_machine parses this format itself, so the test has to be able to
    state exactly what is in the file. Two circles of known radius, one of them
    reached through a RotatedCurve, and the SAME two curves used by both
    BiotSavart objects with different currents -- which is the arrangement that
    would turn two physical coils into four if the de-duplication broke.
    """
    def circle(name, r, z):
        # A circle of radius r at height z: x = r cos(2 pi t), y = r sin(2 pi t)
        return {"@module": "simsopt.geo.curvexyzfourier",
                "@class": "CurveXYZFourier", "@name": name,
                "quadpoints": {"@module": "numpy", "@class": "array",
                               "dtype": "float64",
                               "data": list(np.linspace(0, 1, 32, endpoint=False))},
                "order": 1, "dofs": {"$type": "ref", "value": f"dofs_{name}"}}, {
            "@module": "simsopt._core.optimizable", "@class": "DOFs",
            "@name": f"dofs_{name}",
            "x": {"@module": "numpy", "@class": "array", "dtype": "float64",
                  "data": [0.0, 0.0, r,        # xc(0), xs(1), xc(1)
                           0.0, r, 0.0,        # yc(0), ys(1), yc(1)
                           z, 0.0, 0.0]},      # zc(0), zs(1), zc(1)
            "names": ["xc(0)", "xs(1)", "xc(1)",
                      "yc(0)", "ys(1)", "yc(1)",
                      "zc(0)", "zs(1)", "zc(1)"],
            "free": {"@module": "numpy", "@class": "array", "dtype": "bool",
                     "data": [True] * 9},
            "lower_bounds": {"@module": "numpy", "@class": "array",
                             "dtype": "float64", "data": [-1e30] * 9},
            "upper_bounds": {"@module": "numpy", "@class": "array",
                             "dtype": "float64", "data": [1e30] * 9}}

    objs = {}
    for name, r, z in (("CurveXYZFourier1", radius, 0.0),
                       ("CurveXYZFourier2", radius, shift)):
        curve, dofs = circle(name, r, z)
        objs[name] = curve
        objs[f"dofs_{name}"] = dofs
    # The second coil is reached through a rotation, as a real file does it.
    objs["RotatedCurve1"] = {"@module": "simsopt.geo.curve",
                             "@class": "RotatedCurve", "@name": "RotatedCurve1",
                             "curve": {"$type": "ref",
                                       "value": "CurveXYZFourier2"},
                             "phi": 0.0, "flip": False}
    for i, amps in enumerate((100.0, 250.0, 7.0, -3.0), start=1):
        objs[f"Current{i}"] = {
            "@module": "simsopt.field.coil", "@class": "Current",
            "@name": f"Current{i}", "current": amps,
            "dofs": {"$type": "ref", "value": f"dofs_Current{i}"}}
        objs[f"dofs_Current{i}"] = {
            "@module": "simsopt._core.optimizable", "@class": "DOFs",
            "@name": f"dofs_Current{i}",
            "x": {"@module": "numpy", "@class": "array", "dtype": "float64",
                  "data": [amps]},
            "names": ["x0"],
            "free": {"@module": "numpy", "@class": "array", "dtype": "bool",
                     "data": [True]}}
    # One of them is scaled, because most real files scale every current.
    objs["ScaledCurrent1"] = {"@module": "simsopt.field.coil",
                              "@class": "ScaledCurrent",
                              "@name": "ScaledCurrent1",
                              "current_to_scale": {"$type": "ref",
                                                   "value": "Current1"},
                              "scale": 10.0}
    curves = ["CurveXYZFourier1", "RotatedCurve1"]
    currents = ["ScaledCurrent1", "Current2", "Current3", "Current4"]
    for i in range(4):
        objs[f"Coil{i + 1}"] = {"@module": "simsopt.field.coil",
                                "@class": "Coil", "@name": f"Coil{i + 1}",
                                "curve": {"$type": "ref",
                                          "value": curves[i % 2]},
                                "current": {"$type": "ref",
                                            "value": currents[i]}}
    for i, coils in enumerate((["Coil1", "Coil2"], ["Coil3", "Coil4"]), start=1):
        objs[f"BiotSavart{i}"] = {
            "@module": "simsopt.field.biotsavart", "@class": "BiotSavart",
            "@name": f"BiotSavart{i}",
            "coils": [{"$type": "ref", "value": c} for c in coils]}
    doc = {"@module": "simsopt._core.json", "@class": "SIMSON",
           "@version": "1.8.4.test", "graph": [], "simsopt_objs": objs}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return path


def _synth_magnet_run(g, gain, magnet_world, span_mm=260.0, step_mm=4.0,
                      shift_pose=None, eccentric_mm=0.0):
    """A guided run against a 1/r^3 dipole, with a known per-sensor gain.

    shift_pose: if given, that pose is also displaced sideways -- the mistake
    the routine warns about, used here to check it actually shows up.

    eccentric_mm: the head is not concentric with the axis it is turned about,
    by this much. The offset rides WITH the tube, so it points at the magnet in
    one pose and away from it in the opposite one -- which is what makes the
    error separable from gain.
    """
    def roll(deg):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    ys = np.arange(0.0, span_mm + 1e-9, step_mm)
    run = omag.MagnetRun(magnet_note="synthetic dipole")
    for pose in range(omag.N_POSES):
        R = roll(90.0 * pose)
        off = np.array([12.0, 0.0, 0.0]) if pose == shift_pose else np.zeros(3)
        rows = []
        for y in ys:
            b = np.zeros((16, 3))
            for i in range(16):
                p = R @ (pgeom.to_world(g.position(i + 1))
                         + np.array([0.0, 0.0, eccentric_mm])) + off
                p = p + np.array([0.0, y, 0.0])
                r = magnet_world - p
                d = np.linalg.norm(r)
                bw = (r / d) * (50.0 / d) ** 3
                b[i] = gain[i] * (g.rotations()[i].T
                                  @ (pgeom.MOUNT_ROT.T @ (R.T @ bw)))
            rows.append(b)
        run.add(omag.PoseSweep(pose, ys, np.array(rows)))
    return run


def _synth_magnet_passes(g, gain, magnet_world, jitter, standoff_mm=30.0,
                         plane=True, dither=True, seed=4):
    """A guided run with ARM MISPLACEMENT, measured by all three passes.

    `jitter` is (across, along, standoff) rms in mm, applied per sensor in the
    probe's own frame so that it rides with the head through the four poses --
    which is what a real arm does, and what makes the error impossible to
    average away by rolling.
    """
    rng = np.random.default_rng(seed)
    jit = rng.normal(0.0, 1.0, (16, 3)) * np.asarray(jitter, float)

    def roll(deg):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    def field(stage, pose):
        R = roll(90.0 * pose)
        b = np.zeros((16, 3))
        for i in range(16):
            p = R @ (pgeom.to_world(g.position(i + 1)) + jit[i]) + stage
            r = magnet_world - p
            d = np.linalg.norm(r)
            b[i] = gain[i] * (r / d) * (50.0 / d) ** 3
        return b

    across = "x" if plane else None
    normal = "z" if dither else None
    run = omag.MagnetRun(magnet_note="synthetic, misplaced arms",
                         across=across, normal=normal)
    # Ring `slot` meets the magnet at stage y = magnet_y - sensor_y, so the
    # last slot is the first one past it. Start a clearance short of that.
    span, step = omag.suggested_sweep(g)
    last = g.first_sensor_z_mm + (pgeom.SENSORS_PER_FACE - 1) * g.plate_pitch_mm
    ys = np.arange(0.0, span + 1e-9, step) + magnet_world[1] - last - 40.0

    for pose in range(omag.N_POSES):
        pos = np.array([[0.0, y, 0.0] for y in ys])
        rows = np.array([field(p, pose) for p in pos])
        sw = omag.PoseSweep(pose, pos, rows, axis="y", across=across,
                            normal=normal, axes=("x", "y", "z"))
        rings = omag.ring_positions(sw)

        if plane:
            half, xstep = omag.suggested_plane(standoff_mm)
            xs = np.arange(-half, half + 1e-9, xstep)
            pos = np.array([[x, y, 0.0] for y in rings for x in xs])
            rows = np.array([field(p, pose) for p in pos])
            sw.merge(omag.PoseSweep(pose, pos, rows, axis="y", across="x",
                                    normal=normal, axes=("x", "y", "z")))
        if dither:
            xc = 0.0
            if plane:
                vals = sw.peak_across_mm[np.argsort(sw.peaks)[-4:]]
                xc = float(np.mean(vals[np.isfinite(vals)]))
            zs = omag.suggested_dither(standoff_mm)
            for y in rings:
                pos = np.array([[xc, y, z] for z in zs])
                rows = np.array([field(p, pose) for p in pos])
                sw.dithers.append(omag.Dither(zs, rows, at_mm=y))
        run.add(sw)
    return run


def _trim_error_pct(run, gain):
    """How much of the recovered trim is NOT the injected gain."""
    prod = run.trim() * gain
    return 100.0 * float(np.std(prod / np.median(prod)))

