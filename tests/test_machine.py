"""The coil set the probe measures inside."""

import inspect
import json
import os

import numpy as np

from octobee.motion import scan as oscan
from octobee import machine as omach
from octobee.calib import geometry as pgeom
from tests.helpers import (
    _simsopt_file,
    check,
)



def test_machine_coils(workdir):
    """Reading a coil set, and the volume it takes up."""
    print("\ncoil set")
    path = _simsopt_file(os.path.join(workdir, "coils.json"))
    coils = omach.CoilSet.load(path)

    check("two BiotSavart objects over the same curves are two coils, "
          "not four", len(coils) == 2, f"{len(coils)} coils")
    check("both current configurations are offered",
          coils.configurations == ["BiotSavart1", "BiotSavart2"],
          str(coils.configurations))
    # 100 A through a x10 ScaledCurrent.
    check("a scaled current is resolved through the scale",
          abs(coils["C1"].currents["BiotSavart1"] - 1000.0) < 1e-9,
          f"{coils['C1'].currents['BiotSavart1']} A")
    check("the same coil carries a different current in the other "
          "configuration",
          abs(coils["C1"].currents["BiotSavart2"] - 7.0) < 1e-9,
          f"{coils['C1'].currents['BiotSavart2']} A")

    # The file is in metres and everything downstream is in millimetres. A
    # factor of a thousand here would put the probe a millimetre from a coil
    # that is really a metre away, and the drawing would look plausible.
    r = np.hypot(coils["C1"].points_mm[:, 0], coils["C1"].points_mm[:, 1])
    check("metres become millimetres at the boundary",
          np.allclose(r, 1000.0, atol=1e-6), f"radius {r.mean():.3f} mm")
    check("the centreline comes back closed",
          np.allclose(coils["C1"].points_mm[0], coils["C1"].points_mm[-1]))
    check("a circle's length is 2 pi r",
          abs(coils["C1"].length_mm - 2 * np.pi * 1000.0) < 1.0,
          f"{coils['C1'].length_mm:.1f} mm")
    check("the second coil is where the rotation put it",
          abs(coils["C2"].centroid_mm[2] - 250.0) < 1e-6,
          f"z {coils['C2'].centroid_mm[2]:.1f} mm")

    verts, faces = omach.tube_mesh(coils["C1"].points_mm, 20.0, sides=8)
    n = len(coils["C1"].points_mm) - 1
    d = np.linalg.norm(verts.reshape(n, 8, 3)
                       - coils["C1"].points_mm[:-1][:, None, :], axis=2)
    check("the swept cross-section really is the radius asked for",
          np.allclose(d, 20.0, atol=1e-3), f"{d.min():.4f}..{d.max():.4f} mm")
    edges = {}
    for tri in faces:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            key = (min(a, b), max(a, b))
            edges[key] = edges.get(key, 0) + 1
    check("the tube closes on itself, with no seam and no cap",
          set(edges.values()) == {2},
          f"edge counts {sorted(set(edges.values()))}")

    # ---- clearance, against an answer worked out by hand ----
    # A point on the axis of a circle of radius R, at height h, is
    # sqrt(R^2 + h^2) from the wire.
    probe = np.array([[0.0, 0.0, 300.0]])
    gap = omach.clearance(probe, coils, 20.0, labels=["C1"])
    want = np.hypot(1000.0, 300.0) - 20.0
    # The coil is a polygon through points on the curve, so it lies just
    # inside it: a 256-sided ring of radius 1 m cuts the corner by 75 um. The
    # answer must be short by that and no more -- the module claims a chord
    # error well under a tenth of a millimetre, and this is the claim.
    sag = 1000.0 * (1.0 - np.cos(np.pi / omach.CURVE_POINTS))
    check("clearance is measured to the winding surface, not the centreline",
          -sag <= gap.gap_mm - want <= 0.0,
          f"{gap.gap_mm:.4f} vs {want:.4f} mm, chord error {sag * 1000:.0f} um")
    check("and it names the coil it measured to", gap.coil == "C1", gap.coil)
    check("the closest point it reports is on that coil",
          abs(np.hypot(*gap.coil_point[:2]) - 1000.0) <= sag + 1e-9
          and abs(gap.coil_point[2]) < 1e-9, str(gap.coil_point))

    inside = np.array([[1000.0, 0.0, 5.0]])
    gap = omach.clearance(inside, coils, 20.0)
    check("a point inside a winding is reported as a collision",
          gap.collides and abs(gap.gap_mm + 15.0) < 1e-6, gap.text())

    # A coil that is switched off is still in the way: clearance takes no
    # notice of which ones are energised.
    both = omach.clearance(np.array([[0.0, 0.0, 249.0]]), coils, 20.0)
    check("clearance considers coils that are switched off",
          both.coil == "C2", both.text())

    # The two-pass narrowing inside clearance() is an optimisation, and an
    # optimisation that changes the answer is a bug. Check it against the
    # definition on a cloud of random points.
    rng = np.random.default_rng(4)
    cloud = rng.uniform(-1400, 1400, size=(400, 3))
    fast = omach.clearance(cloud, coils, 20.0)
    slow = np.inf
    for coil in coils:
        a, b = coil.points_mm[:-1], coil.points_mm[1:]
        ab = b - a
        for q in cloud:
            ap = q - a
            t = np.clip((ap * ab).sum(1) / (ab * ab).sum(1), 0.0, 1.0)
            slow = min(slow, np.linalg.norm(ap - t[:, None] * ab,
                                            axis=1).min())
    check("narrowing the search does not change the answer",
          abs(fast.gap_mm - (slow - 20.0)) < 1e-9,
          f"{fast.gap_mm:.9f} vs {slow - 20.0:.9f} mm")

    # ---- the probe body ----
    geom = pgeom.Geometry()
    cloud = omach.probe_cloud(geom)
    reach = np.linalg.norm(cloud, axis=1).max()
    check("the probe cloud covers the whole body, arms included",
          reach > geom.fsv_radius_mm, f"reaches {reach:.1f} mm")
    # Corners alone would miss a coil brushing the middle of the tube.
    along = np.unique(np.round(cloud[:, 1], 3))
    check("and it samples along the tube rather than only its ends",
          len(along) > 20, f"{len(along)} distinct stations along the tube")


def test_machine_placement(workdir):
    """The transform that says where the probe is, and what it remembers."""
    print("\nprobe placement")
    pose = omach.Placement(x_mm=100.0, y_mm=200.0, z_mm=-50.0)
    check("with no rotation the flange is where it was put",
          np.allclose(pose.origin_mm(), [100.0, 200.0, -50.0]))

    # The stages move the probe along the RIG's axes. Turn the assembly a
    # quarter turn about the machine's Z and driving rig x must move it along
    # machine y -- if it moved along machine x regardless, the drawing would
    # be wrong in exactly the way nobody would notice until the head hit
    # something.
    pose = omach.Placement(rot_z_deg=90.0)
    moved = pose.origin_mm({"x": 10.0, "y": 0.0, "z": 0.0})
    check("a stage move is applied in the rig's frame, not the machine's",
          np.allclose(moved, [0.0, 10.0, 0.0], atol=1e-9), str(moved))

    pose = omach.Placement(x_mm=5.0, stage_zero_mm={"x": 100.0})
    check("the stage zero is where the pose says the probe is",
          np.allclose(pose.origin_mm({"x": 100.0}), [5.0, 0.0, 0.0]))
    check("and moving off it moves the probe by the difference",
          np.allclose(pose.origin_mm({"x": 130.0}), [35.0, 0.0, 0.0]))

    # The rig has one rotation and it is about the machine's Z. Stated as the
    # properties it has to satisfy rather than as one vector's image, because
    # a vector can come out right for a rotation composed the other way.
    r = omach.Placement(rot_z_deg=35.0).rotation()
    check("the only rotation is about the machine's Z",
          np.allclose(r @ [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], atol=1e-12)
          and np.allclose(r @ r.T, np.eye(3), atol=1e-12)
          and abs(np.linalg.det(r) - 1.0) < 1e-12,
          f"Z maps to {r @ [0.0, 0.0, 1.0]}, det {np.linalg.det(r):.6f}")
    check("and turning twice is the same as turning by the sum",
          np.allclose(omach.rotation_matrix(20.0) @ omach.rotation_matrix(15.0),
                      r, atol=1e-12))

    # A pose written before the tilts were removed. yaw WAS rotation about Z
    # under another name, so it comes across; a pitch or a roll describes a
    # pose this rig has never been able to reach, and being quietly ignored is
    # how it would come back as a confident clearance number.
    notes = []
    old_pose = omach.Placement.from_dict(
        {"x_mm": 3.0, "yaw_deg": 45.0, "pitch_deg": 20.0, "roll_deg": 0.0},
        notes.append)
    check("a pose saved before the tilts went keeps its rotation about Z",
          abs(old_pose.rot_z_deg - 45.0) < 1e-9 and old_pose.x_mm == 3.0,
          f"{old_pose.rot_z_deg} deg")
    check("and a tilt in it is dropped out loud, not silently",
          len(notes) == 1 and "pitch 20" in notes[0], str(notes))
    quiet = []
    omach.Placement.from_dict({"yaw_deg": 10.0, "pitch_deg": 0.0}, quiet.append)
    check("while a file with the tilts at zero says nothing", not quiet,
          str(quiet))

    corners = omach.Placement(travel_mm={"x": (0.0, 300.0),
                                         "y": (0.0, 100.0),
                                         "z": (0.0, 50.0)}).reach_corners_mm()
    span = corners.max(axis=0) - corners.min(axis=0)
    check("the stage envelope is the travel of each axis",
          np.allclose(span, [300.0, 100.0, 50.0]), str(span))

    # ---- persistence ----
    path = os.path.join(workdir, "machine_cfg.json")
    coil_path = _simsopt_file(os.path.join(workdir, "coils2.json"))
    cfg = omach.MachineConfig(coil_file=coil_path, coil_radius_mm=12.5,
                              configuration="BiotSavart2", current_scale=0.01,
                              energised=["C2"], track_stage=False)
    cfg.pose = omach.Placement(x_mm=1.5, rot_z_deg=30.0,
                               stage_zero_mm={"y": 12.0})
    cfg.save(path)
    back = omach.MachineConfig.load(path)
    check("the placement survives a save and a load",
          (back.coil_radius_mm == 12.5 and back.configuration == "BiotSavart2"
           and back.energised == ["C2"] and back.track_stage is False
           and abs(back.pose.rot_z_deg - 30.0) < 1e-9
           and abs(back.pose.stage_zero_mm["y"] - 12.0) < 1e-9),
          json.dumps(back.to_dict())[:120])

    coils = omach.CoilSet.load(coil_path)
    check("the saved scale is applied to the file's currents",
          abs(back.current(coils["C2"]) - (-3.0 * 0.01)) < 1e-12,
          f"{back.current(coils['C2'])} A")

    # A file naming coils or a configuration this coil set does not have is
    # the case that must not silently produce a switch for nothing.
    stale = omach.MachineConfig(coil_file=coil_path,
                                configuration="BiotSavart9",
                                energised=["C1", "C9"])
    lost = stale.adopt(coils)
    check("a configuration that is not in the file is dropped, and said so",
          stale.configuration == "BiotSavart1" and len(lost) == 2, str(lost))
    check("so is a coil that is not in the file",
          stale.energised == ["C1"], str(stale.energised))

    fresh = omach.MachineConfig(coil_file=coil_path)
    fresh.adopt(coils)
    check("a placement that has never chosen starts with every coil on",
          fresh.energised == ["C1", "C2"], str(fresh.energised))

    # ---- what a field map carries away with it ----
    meta = back.to_scan_meta(coils, {"x": 10.0, "y": 0.0, "z": 0.0})
    check("a map records which coils were on and at what current",
          meta["coils"]["C2"]["on"] is True
          and meta["coils"]["C1"]["on"] is False
          and abs(meta["coils"]["C2"]["amp_turns"] + 0.03) < 1e-12,
          json.dumps(meta["coils"]))
    check("and where the probe was, in machine coordinates",
          "probe_origin_mm" in meta and meta["pose"]["rot_z_deg"] == 30.0,
          str(meta.get("probe_origin_mm")))
    check("and which file the coils came from",
          os.path.samefile(meta["coil_file"], coil_path), meta["coil_file"])

    # The route this metadata takes into a saved map. A rename here would
    # leave the GUI passing a keyword nothing reads, and the map would come
    # back with no machine in it and no complaint from anything.
    params = inspect.signature(oscan.run_scan).parameters
    check("run_scan still takes the metadata a map is annotated with",
          "extra_meta" in params, ", ".join(params))

    missing = omach.MachineConfig.load_or_default(
        os.path.join(workdir, "not_here.json"))
    check("a missing placement file is not an error",
          missing.coil_file == "" and missing.energised is None)
    problems = []
    with open(os.path.join(workdir, "broken.json"), "w",
              encoding="utf-8") as f:
        f.write("{not json")
    omach.MachineConfig.load_or_default(
        os.path.join(workdir, "broken.json"), on_error=problems.append)
    check("a placement file that will not parse is reported, not swallowed",
          len(problems) == 1, str(problems))
    check("and a coil file that will not parse is too",
          omach.CoilSet.load_or_none(os.path.join(workdir, "broken.json"),
                                     on_error=problems.append) is None
          and len(problems) == 2, str(problems[-1:]))
