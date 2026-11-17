"""The standalone analysis scripts in tools/: faces, sensor choice, the fill.

They are not part of the octobee package, so they are loaded by path. Their
maps are built here from an analytic harmonic field, so every expected number
is known rather than eyeballed from a picture.
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

from octobee import paths
from tests.helpers import check

TOOLS = os.path.join(paths.repo_root(), "tools")
AXES = ("x", "y", "z")
STEP = 10.0
# A box that is neither 250 x 250 x 200 nor at the origin, and whose corner is
# not a multiple of the step: the cases the hard-coded versions got wrong.
LO = np.array([33.0, -17.0, 104.0])
HI = LO + np.array([60.0, 50.0, 40.0])
SRC = np.array([-100.0, -120.0, -80.0])      # the field's source, outside the box


def _load(name):
    # The tools import each other by bare name, as they do when run as scripts.
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    spec = importlib.util.spec_from_file_location(
        f"tools_{name}", os.path.join(TOOLS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tools():
    return {n: _load(n) for n in ("plot_faces", "fill_volume", "plot_in_machine")}


def phi(P, scale=1.0):
    """A harmonic, positive field: the potential of a point source outside."""
    return scale * 100.0 / np.linalg.norm(np.asarray(P, float) - SRC, axis=-1)


def _nodes(k):
    return LO[k] + STEP * np.arange(int(round((HI[k] - LO[k]) / STEP)) + 1)


def build_map(directory, name="vol", scale=1.0, bad_sensor=None, drop_node=None,
              drop_face=None, slices=False, side_extra=None):
    """Write <name>.npz + <name>.json for a shell of the box above.

    Every sensor reads (0, 0, phi) so |B| = phi. `bad_sensor` (1-16) gets a
    channel stuck at -57 mT, beyond the 20 mT range. `drop_node` removes the
    rows at one (i, j, k) node; `drop_face` removes all rows of one face
    (axis index, end).
    """
    rows, lines, li = [], [], []

    def add_line(sweep, fixed, part, face=None):
        a = AXES.index(sweep)
        for v in _nodes(a):
            p = np.zeros(3)
            p[a] = v
            for ax, at in fixed.items():
                p[AXES.index(ax)] = at
            if drop_node is not None and np.allclose(
                    p, LO + STEP * np.array(drop_node)):
                continue
            if face is not None and face == drop_face:
                continue
            rows.append(p)
            li.append(len(lines))
        lines.append({"sweep": sweep, "start_mm": float(LO[a]),
                      "stop_mm": float(HI[a]), "fixed": fixed,
                      "run_in_mm": 0.0, "run_out_mm": 0.0, "part": part})

    for k in range(3):
        a, b = [i for i in range(3) if i != k]
        for end, at in enumerate((LO[k], HI[k])):
            for vb in _nodes(b):
                add_line(AXES[a], {AXES[b]: float(vb), AXES[k]: float(at)},
                         "box", face=(k, end))
    if slices:
        for y in (LO[1] + 2 * STEP, LO[1] + 3 * STEP):
            for z in _nodes(2)[1:-1]:
                add_line("x", {"y": float(y), "z": float(z)}, "slice")

    pos = np.array(rows)
    b = np.zeros((len(pos), 16, 3), np.float32)
    b[:, :, 2] = phi(pos, scale)[:, None]
    if bad_sensor:
        b[:, bad_sensor - 1, 0] = -57.0
    base = os.path.join(str(directory), name)
    np.savez_compressed(base + ".npz", pos_mm=pos, b_mt=b,
                        line_index=np.array(li, np.int32),
                        in_span=np.ones(len(pos), bool))
    side = {"lines": lines}
    side.update(side_extra or {})
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(side, fh)
    return base + ".npz"


def _interior_error(U, vol):
    """Largest |solved - analytic| over the interior, as a share of the mean."""
    idx = np.indices(U.shape).reshape(3, -1).T
    P = vol.lo + STEP * idx
    want = phi(P).reshape(U.shape)
    inner = np.zeros(U.shape, bool)
    inner[1:-1, 1:-1, 1:-1] = True
    return float(np.abs(U - want)[inner].max() / want.mean())


def test_faces_of_an_offset_box(tools, tmp_path):
    print("\nfaces of a box that is not 250 x 250 x 200 and not at the origin")
    pf = tools["plot_faces"]
    m = build_map(tmp_path)
    faces, lo, hi, choice = pf.load_faces(m, STEP, log=lambda s: None)
    check("box low corner is the sidecar's, not rounded to the step",
          np.allclose(lo, LO), f"{lo}")
    check("box high corner", np.allclose(hi, HI), f"{hi}")
    check("six faces found", len(faces) == 6, f"{sorted(faces)}")
    check("ids come from the real coordinates",
          sorted(faces) == sorted(["x33", "x93", "y-17", "y33", "z104", "z144"]),
          f"{sorted(faces)}")
    check("titles carry them", faces["y-17"]["title"] == "y = -17",
          faces["y-17"]["title"])
    bad = 0
    for f in faces.values():
        iu, iv = AXES.index(f["u"]), AXES.index(f["v"])
        ifx = 3 - iu - iv
        U, V = np.meshgrid(f["ue"], f["ve"])
        P = np.zeros((*U.shape, 3))
        P[..., iu], P[..., iv], P[..., ifx] = U, V, f["at"]
        bad += int((~np.isfinite(f["grid"])).sum())
        bad += int(not np.allclose(f["grid"], phi(P), rtol=1e-5))
    check("every face cell holds the field at its own coordinates", bad == 0,
          f"{bad} bad cells")
    check("a 16-sensor mean says so", choice.text == "16-sensor mean", choice.text)


def test_net_layout_keeps_roles(tools):
    print("\nthe net layout is by role")
    pf = tools["plot_faces"]
    spec = pf.face_spec(0, 0, LO, HI)
    check("low x face is the mirrored left end",
          spec["col"] == 0 and spec["flip_u"] and spec["label"] == "left end",
          f"{spec}")
    spec = pf.face_spec(2, 0, LO, HI)
    check("low z face is the floor, flipped in v",
          spec["row"] == 2 and spec["flip_v"], f"{spec}")


def test_fill_reproduces_analytic(tools, tmp_path):
    print("\nthe Laplace fill against an analytic harmonic field")
    fv = tools["fill_volume"]
    m = build_map(tmp_path)
    vol = fv.read_faces(m, STEP, log=lambda s: None)
    check("no shell node missing", vol.missing == 0, f"{vol.missing}")
    U, info = fv.solve(vol.F, log=lambda s: None)
    check("converged", info["converged"], f"{info}")
    err = _interior_error(U, vol)
    check("interior matches the analytic field to 0.5 %", err < 5e-3,
          f"{100 * err:.3f} %")


def test_missing_shell_node_does_not_drag_interior(tools, tmp_path):
    print("\na missing shell node")
    fv = tools["fill_volume"]
    # an interior node of the low-x face, which only that face's one line holds
    m = build_map(tmp_path, drop_node=(0, 3, 2))
    vol = fv.read_faces(m, STEP, log=lambda s: None)
    check("one shell node is missing", vol.missing == 1, f"{vol.missing}")
    n = fv.fill_missing_shell(vol.F)
    check("and one is filled", n == 1 and not np.isnan(vol.F[0, 3, 2]), f"{n}")
    want = phi(vol.lo + STEP * np.array([0, 3, 2]))
    check("from its neighbours, close to the true value",
          abs(vol.F[0, 3, 2] - want) / want < 0.02,
          f"{vol.F[0, 3, 2]:.4f} vs {want:.4f}")
    U, _ = fv.solve(vol.F, log=lambda s: None)
    err = _interior_error(U, vol)
    check("interior stays within 1 % of the analytic field", err < 1e-2,
          f"{100 * err:.3f} %")
    # What the old code did: leave the node free, to relax toward zero.
    F2 = vol.F.copy()
    F2[0, 3, 2] = np.nan
    try:
        fv.solve(F2, log=lambda s: None)
        refused = False
    except ValueError:
        refused = True
    check("solve refuses an unset shell node rather than solving it to zero",
          refused)


def test_missing_face_is_refused_unless_forced(tools, tmp_path, monkeypatch, capsys):
    print("\na face with no data")
    fv = tools["fill_volume"]
    monkeypatch.setattr(fv, "draw_slices", lambda *a, **k: "x")
    monkeypatch.setattr(fv, "draw_3d", lambda *a, **k: "x")
    monkeypatch.setattr(os.path, "getsize", lambda p: 0)
    m = build_map(tmp_path, drop_face=(1, 1))
    rc = fv.main([m, "--out", str(tmp_path / "o")])
    err = capsys.readouterr().err
    check("refused with exit 2", rc == 2, f"rc={rc}")
    check("and says why", "REFUSED" in err and "--force" in err, err)
    rc = fv.main([m, "--out", str(tmp_path / "o"), "--force"])
    check("--force solves anyway", rc == 0, f"rc={rc}")


def test_sensor_exclusion(tools, tmp_path):
    print("\n--exclude, the sidecar's list, and automatic exclusion")
    pf = tools["plot_faces"]
    quiet = []
    log = quiet.append
    m = build_map(tmp_path, bad_sensor=5)
    pos, b, _, side = pf.read_map(m)
    truth = phi(pos)

    c = pf.choose_sensors(b, side, log=log)
    check("a channel stuck beyond the range is excluded automatically",
          c.excluded == [5], f"{c.excluded}")
    check("and named in a warning", any("S5" in q for q in quiet), f"{quiet}")
    check("the mean over the rest is the true field",
          np.allclose(pf.magnitude(b, c), truth, rtol=1e-5))
    check("the label says 15, and which",
          c.text == "15-sensor mean, S5 excluded", c.text)

    c = pf.choose_sensors(b, side, keep_all=True, log=log)
    check("--keep-all keeps it", c.excluded == [] and c.n == 16, f"{c.excluded}")
    check("and the mean is then wrong, as it should be",
          not np.allclose(pf.magnitude(b, c), truth, rtol=1e-2))

    c = pf.choose_sensors(b, side, exclude=pf.parse_sensors("s1, S2"), log=log)
    check("--exclude adds to the automatic ones", c.excluded == [1, 2, 5],
          f"{c.excluded}")

    m2 = build_map(tmp_path, name="rec", side_extra={"excluded": ["S3", "S7.Bx"]})
    _, b2, _, side2 = pf.read_map(m2)
    c = pf.choose_sensors(b2, side2, log=log)
    check("the sidecar's own record is used", c.excluded == [3, 7],
          f"{c.excluded}")
    check("--keep-all overrides it",
          pf.choose_sensors(b2, side2, keep_all=True, log=log).excluded == [])

    for bad in ("17", "S0", "foo", "S9,,x"):
        try:
            pf.parse_sensors(bad)
            ok = False
        except ValueError:
            ok = True
        check(f"parse_sensors rejects {bad!r}", ok)
    check("case-insensitive, and 'S' optional",
          pf.parse_sensors("s9,S10,11") == {9, 10, 11})

    faces, *_ = pf.load_faces(m, STEP, log=log)
    faces_all, *_ = pf.load_faces(m, STEP, keep_all=True, log=log)
    check("--keep-all changes the face values",
          not np.allclose(faces["x33"]["grid"], faces_all["x33"]["grid"],
                          rtol=1e-2))


def test_interior_planes_are_not_faces(tools, tmp_path):
    print("\ninterior x-z planes")
    pf, fv = tools["plot_faces"], tools["fill_volume"]
    m = build_map(tmp_path, slices=True)
    faces, lo, hi, _ = pf.load_faces(m, STEP, log=lambda s: None)
    check("still the same six faces and box",
          len(faces) == 6 and np.allclose(lo, LO) and np.allclose(hi, HI),
          f"{sorted(faces)} {lo} {hi}")
    vol = fv.read_faces(m, STEP, log=lambda s: None)
    n = int(np.isfinite(vol.S).sum())
    check("the planes' interior nodes are held apart from the shell",
          n > 0 and not np.isfinite(vol.F[1:-1, 1:-1, 1:-1]).any()
          and not np.isfinite(vol.S[0]).any(), f"{n}")
    U, _ = fv.solve(vol.F, log=lambda s: None)
    c = fv.slice_check(U, vol.S)
    check("the solve agrees with the held-out planes", c and c["pct"] < 0.5,
          f"{c}")


def test_plot_in_machine_reads_the_map_it_is_given(tools, tmp_path):
    print("\nplot_in_machine.load_faces")
    pim = tools["plot_in_machine"]
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
    # a decoy of the kind it used to read, beside the map
    (a / "faces").mkdir()
    (a / "faces" / "face_x0.csv").write_text("y,z,B\n0,0,99\n", encoding="utf-8")
    ma = build_map(a, scale=1.0)
    mb = build_map(b, scale=2.0)
    fa, lo, hi, _ = pim.load_faces(ma, STEP)
    fb, *_ = pim.load_faces(mb, STEP)
    check("six faces, from the npz", len(fa) == 6 and len(fb) == 6,
          f"{sorted(fa)}")
    check("the box is the map's", np.allclose(lo, LO) and np.allclose(hi, HI))
    ratio = fb["x33"]["grid"] / fa["x33"]["grid"]
    check("a different map gives different faces", np.allclose(ratio, 2.0),
          f"{ratio.min()} {ratio.max()}")
    check("the decoy CSV is ignored", not np.isclose(fa["x33"]["grid"], 99).any())
    check("faces carry what draw() needs",
          {"u", "v", "at", "end", "ue", "ve", "grid"} <= set(fa["x33"]))


def test_non_convergence_is_reported(tools, tmp_path, monkeypatch, capsys):
    print("\nnon-convergence")
    fv = tools["fill_volume"]
    m = build_map(tmp_path)
    vol = fv.read_faces(m, STEP, log=lambda s: None)
    log = []
    _, info = fv.solve(vol.F, max_iter=3, log=log.append)
    check("solve says it did not converge", info["converged"] is False, f"{info}")
    check("and logs it", any("NOT CONVERGED" in s for s in log), f"{log}")

    notes_seen = []
    monkeypatch.setattr(fv, "draw_slices",
                        lambda *a, **k: notes_seen.append(a[-1]) or "x")
    monkeypatch.setattr(fv, "draw_3d", lambda *a, **k: "x")
    monkeypatch.setattr(os.path, "getsize", lambda p: 0)
    out = str(tmp_path / "o")
    rc = fv.main([m, "--out", out, "--max-iter", "3"])
    err = capsys.readouterr().err
    check("exit code 1", rc == 1, f"rc={rc}")
    check("stderr says so", "did not converge" in err, err)
    check("the figure footer says so",
          any("NOT CONVERGED" in n for ns in notes_seen for n in ns),
          f"{notes_seen}")
    rc = fv.main([m, "--out", out, "--max-iter", "3", "--force"])
    check("--force exits 0", rc == 0, f"rc={rc}")
    rc = fv.main([m, "--out", out])
    check("a normal run exits 0", rc == 0, f"rc={rc}")


def test_validation_reports_all_recordings(tools):
    print("\nthe summary of several validation recordings")
    fv = tools["fill_volume"]
    few = [{"name": "a", "pct": 1.0, "n": 10}, {"name": "b", "pct": 4.0, "n": 20}]
    note = fv.check_note(few)
    check("up to three are all listed", "a 1.0" in note and "b 4.0" in note, note)
    many = [{"name": n, "pct": p, "n": 5}
            for n, p in zip("abcde", (1.0, 2.0, 3.0, 9.0, 4.0), strict=True)]
    note = fv.check_note(many)
    check("more give the worst and the median, not the best",
          "worst 9.0" in note and "median 3.0" in note and "1.0 %" not in note,
          note)
