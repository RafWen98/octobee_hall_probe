#!/usr/bin/env python3
"""
probe_geometry.py -- where each sensor actually sits, and which way its chip faces.

The probe is a square tube carrying 16 SENM3Dx evaluation-kit PCBs, 4 per face,
each standing off radially like a bristle on a brush. The chips are NOT on the
tube surface: each one rides at the far tip of a 92 mm board, so the field
sensitive volume sits roughly 89 mm clear of the face it is bolted to.

That standoff dominates everything geometric about this instrument:

  1. Every chip has a DIFFERENT orientation, so comparing a single axis across
     sensors is meaningless. Compare |B| (rotation invariant), or rotate each
     chip's vector into the common tube frame with the matrices here.

  2. A magnet held near the probe is at a very different distance from each
     chip. Field falls off roughly as 1/r^3, so this alone produces large
     amplitude differences before any gain or calibration difference.
     expected_response() quantifies it so it can be divided out.

Dimensions come from the SENIS eval-kit drawing (Figure 2) and the FSV detail
(Figure 3):

    PCB overall length              92 mm
    mounting end                    30 mm wide, first 20 mm of the length
    arm                             12 mm wide
    FSV inset from the far end       3 mm      -> 89 mm from the mounting edge
    FSV depth below the chip lid  0.35 mm
    chip package height            0.9 mm     -> FSV sits 0.55 mm above the
    PCB thickness                  1.2 mm        board's top face

Tube frame convention
---------------------
    +Z  along the tube, away from the mounting flange (toward the tip)
    +X  outward normal of face 0
    +Y  outward normal of face 1
    faces 0,1,2,3 at 0, 90, 180, 270 degrees about the tube axis

Chip local frame (before chip_rot_deg)
--------------------------------------
    local +X  along the arm, pointing away from the tube
    local +Z  out of the PCB surface
    local +Y  completes the right-handed set

WHICH SENSOR IS ON WHICH FACE IS NOT YET VERIFIED on this hardware, and neither
is `board_plane`. The defaults are assumptions -- see probe_geometry.json.
"""

import json
import os

import numpy as np

N_SENSORS = 16
N_FACES = 4
SENSORS_PER_FACE = N_SENSORS // N_FACES

CONFIG_NAME = "probe_geometry.json"

# Face index -> outward normal in the tube frame.
FACE_NORMALS = np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0]])
FACE_NAMES = ("+X", "+Y", "-X", "-Y")

MAPPINGS = ("face-major", "ring-major")

# Which way the flat of the PCB faces once the arm points radially outward.
#   axial           the board plane contains the tube axis (boards stand like
#                   fins along the tube), so the board normal runs around the
#                   circumference
#   circumferential the board plane is perpendicular to the tube axis, so the
#                   board normal runs along the tube
BOARD_PLANES = ("axial", "circumferential")

TUBE_AXIS = np.array([0.0, 0.0, 1.0])

# Eval-kit PCB, from the vendor drawing. Millimetres.
ARM_LENGTH_MM = 92.0
ARM_MOUNT_LENGTH_MM = 20.0
ARM_MOUNT_WIDTH_MM = 30.0
ARM_WIDTH_MM = 12.0
ARM_THICKNESS_MM = 1.2
FSV_FROM_TIP_MM = 3.0
FSV_ABOVE_BOARD_MM = 0.55        # 0.9 mm package height - 0.35 mm FSV depth
CHIP_SIZE_MM = 6.0               # QFN-28 footprint, for drawing only
PLATE_GAP_MM = 3.0               # gap between adjacent mounting plates


def _default_sensors(mapping):
    """sensor id (1..16) -> (face, slot along the tube)."""
    out = []
    for i in range(N_SENSORS):
        if mapping == "ring-major":
            # S1..S4 form the first ring around the tube, S5..S8 the second, ...
            face, slot = i % N_FACES, i // N_FACES
        else:
            # S1..S4 run along face 0, S5..S8 along face 1, ...
            face, slot = i // SENSORS_PER_FACE, i % SENSORS_PER_FACE
        out.append({"id": i + 1, "face": int(face), "slot": int(slot),
                    "chip_rot_deg": 0.0, "axis_signs": [1, 1, 1]})
    return out


class Geometry:
    """Positions and orientations of the 16 chips, loadable from JSON."""

    def __init__(self, tube_width_mm=40.0,
                 arm_length_mm=ARM_LENGTH_MM,
                 fsv_from_tip_mm=FSV_FROM_TIP_MM,
                 fsv_above_board_mm=FSV_ABOVE_BOARD_MM,
                 mount_standoff_mm=0.0,
                 plate_pitch_mm=None,
                 first_sensor_z_mm=30.0,
                 tube_length_mm=None,
                 board_plane="axial",
                 mapping="face-major", sensors=None, notes=""):
        self.notes = notes
        self.tube_width_mm = float(tube_width_mm)
        self.arm_length_mm = float(arm_length_mm)
        self.fsv_from_tip_mm = float(fsv_from_tip_mm)
        self.fsv_above_board_mm = float(fsv_above_board_mm)
        self.mount_standoff_mm = float(mount_standoff_mm)
        self.plate_pitch_mm = float(
            ARM_MOUNT_WIDTH_MM + PLATE_GAP_MM if plate_pitch_mm is None
            else plate_pitch_mm)
        self.first_sensor_z_mm = float(first_sensor_z_mm)
        self.board_plane = board_plane
        self.mapping = mapping
        self.sensors = sensors or _default_sensors(mapping)
        if tube_length_mm is None:
            last = self.first_sensor_z_mm + (SENSORS_PER_FACE - 1) * self.plate_pitch_mm
            tube_length_mm = last + self.first_sensor_z_mm
        self.tube_length_mm = float(tube_length_mm)

    # ---- persistence ----------------------------------------------------
    def to_dict(self):
        return {"tube_width_mm": self.tube_width_mm,
                "arm_length_mm": self.arm_length_mm,
                "fsv_from_tip_mm": self.fsv_from_tip_mm,
                "fsv_above_board_mm": self.fsv_above_board_mm,
                "mount_standoff_mm": self.mount_standoff_mm,
                "plate_pitch_mm": self.plate_pitch_mm,
                "first_sensor_z_mm": self.first_sensor_z_mm,
                "tube_length_mm": self.tube_length_mm,
                "board_plane": self.board_plane,
                "mapping": self.mapping,
                "notes": self.notes,
                "sensors": self.sensors}

    def save(self, path=CONFIG_NAME):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path=CONFIG_NAME):
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))

    @classmethod
    def load_or_default(cls, path=CONFIG_NAME):
        if path and os.path.exists(path):
            try:
                return cls.load(path)
            except (OSError, ValueError, TypeError):
                pass
        return cls()

    # ---- derived quantities ---------------------------------------------
    def face(self, sensor_id):
        return self.sensors[sensor_id - 1]["face"]

    def slot(self, sensor_id):
        return self.sensors[sensor_id - 1]["slot"]

    def normal(self, sensor_id):
        """Outward radial direction of this chip's face -- the arm's direction."""
        return FACE_NORMALS[self.sensors[sensor_id - 1]["face"]].copy()

    def normals(self):
        return np.array([self.normal(i) for i in range(1, N_SENSORS + 1)])

    @property
    def arm_root_radius_mm(self):
        """Distance from the tube axis to where an arm leaves the face."""
        return self.tube_width_mm / 2.0 + self.mount_standoff_mm

    @property
    def fsv_radius_mm(self):
        """Distance from the tube axis to the field sensitive volume."""
        return (self.arm_root_radius_mm
                + self.arm_length_mm - self.fsv_from_tip_mm)

    def board_normal(self, sensor_id):
        """Unit normal of the PCB surface, in the tube frame."""
        n = self.normal(sensor_id)
        if self.board_plane == "circumferential":
            e = TUBE_AXIS.copy()
        else:                                    # 'axial'
            e = np.cross(n, TUBE_AXIS)
        norm = np.linalg.norm(e)
        return e / norm if norm > 0 else TUBE_AXIS.copy()

    def slot_z_mm(self, sensor_id):
        return (self.first_sensor_z_mm
                + self.slot(sensor_id) * self.plate_pitch_mm)

    def arm_root(self, sensor_id):
        """Where the arm meets the tube face, in the tube frame, mm."""
        return (self.normal(sensor_id) * self.arm_root_radius_mm
                + TUBE_AXIS * self.slot_z_mm(sensor_id))

    def position(self, sensor_id):
        """The chip's field sensitive volume in the tube frame, mm."""
        return (self.normal(sensor_id) * self.fsv_radius_mm
                + TUBE_AXIS * self.slot_z_mm(sensor_id)
                + self.board_normal(sensor_id) * self.fsv_above_board_mm)

    def positions(self):
        return np.array([self.position(i) for i in range(1, N_SENSORS + 1)])

    def rotation(self, sensor_id):
        """
        3x3 matrix R with B_tube = R @ B_chip, for B_chip in the chip's own
        (Bx, By, Bz). Sign flips are folded in by rotations().
        """
        e1 = self.normal(sensor_id)              # chip +X: out along the arm
        e3 = self.board_normal(sensor_id)        # chip +Z: out of the board
        e2 = np.cross(e3, e1)                    # chip +Y completes the set
        psi = np.deg2rad(self.sensors[sensor_id - 1].get("chip_rot_deg", 0.0))
        c, s = np.cos(psi), np.sin(psi)
        return np.column_stack([c * e1 + s * e2, -s * e1 + c * e2, e3])

    def axis_signs(self, sensor_id):
        return np.array(self.sensors[sensor_id - 1].get("axis_signs", [1, 1, 1]),
                        dtype=float)

    def rotations(self):
        """(16, 3, 3) stack of R_i with the per-axis sign flips folded in."""
        return np.array([self.rotation(i) * self.axis_signs(i)[None, :]
                         for i in range(1, N_SENSORS + 1)])

    def to_tube_frame(self, b_chip):
        """
        (..., 16, 3) chip-frame vectors -> the same shape in the tube frame.
        einsum, so it works on a single sample or a whole capture.
        """
        return np.einsum("sij,...sj->...si", self.rotations(), np.asarray(b_chip))

    # ---- geometric fairness ---------------------------------------------
    def distances_to(self, point_mm):
        """Distance from each chip's FSV to a point in the tube frame, mm."""
        return np.linalg.norm(self.positions() - np.asarray(point_mm, float), axis=1)

    def expected_response(self, point_mm, exponent=3.0, normalize=True):
        """
        Relative field magnitude each sensor should see from a small magnet at
        `point_mm`, purely from 1/r**exponent geometry.

        Divide a measured peak-amplitude vector by this to remove the "the
        magnet was simply closer to S5" effect before blaming a gain register.
        A point dipole is 1/r^3; a bar magnet close in behaves more like 1/r^2,
        so the exponent is exposed rather than baked in.
        """
        r = np.maximum(self.distances_to(point_mm), 1e-6)
        w = r ** (-float(exponent))
        return w / np.median(w) if normalize else w

    def nearest_sensor(self, point_mm):
        return int(np.argmin(self.distances_to(point_mm))) + 1

    def describe(self):
        lines = [
            f"square tube {self.tube_width_mm:g} mm across, "
            f"{self.tube_length_mm:g} mm long",
            f"arms {self.arm_length_mm:g} mm long, FSV {self.fsv_from_tip_mm:g} mm "
            f"from the tip -> chips sit {self.fsv_radius_mm:g} mm from the tube "
            f"axis ({self.fsv_radius_mm - self.arm_root_radius_mm:g} mm clear of "
            f"the face)",
            f"mounting plates every {self.plate_pitch_mm:g} mm, board plane "
            f"'{self.board_plane}', mapping '{self.mapping}'"]
        for i in range(1, N_SENSORS + 1):
            p = self.position(i)
            lines.append(f"  S{i:<2d} face {FACE_NAMES[self.face(i)]} slot {self.slot(i)}"
                         f"  FSV ({p[0]:7.1f},{p[1]:7.1f},{p[2]:7.1f}) mm")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# meshes for the 3D view
# --------------------------------------------------------------------------

def _box(centre, u, v, w, su, sv, sw):
    """Axis-aligned-in-its-own-frame box -> (verts, faces)."""
    verts = []
    for du in (-su / 2, su / 2):
        for dv in (-sv / 2, sv / 2):
            for dw in (-sw / 2, sw / 2):
                verts.append(centre + u * du + v * dv + w * dw)
    # vertex order: (du, dv, dw) with dw fastest
    f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
         (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
         (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return np.array(verts, dtype=np.float32), np.array(f, dtype=np.int32)


def tube_mesh(geom, cap=True):
    """Vertices and faces of the square tube, mm, tube frame."""
    h = geom.tube_width_mm / 2.0
    z0, z1 = 0.0, geom.tube_length_mm
    corners = [(h, h), (-h, h), (-h, -h), (h, -h)]
    verts = [[x, y, z0] for x, y in corners] + [[x, y, z1] for x, y in corners]
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        faces += [[i, j, j + 4], [i, j + 4, i + 4]]     # side walls
    if cap:
        faces += [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def arm_mesh(geom, sensor_id):
    """
    The eval-kit PCB: a wide mounting foot at the tube, a narrower arm reaching
    out to the chip. Two boxes, merged.
    """
    n = geom.normal(sensor_id)                  # along the arm
    w = geom.board_normal(sensor_id)            # out of the board
    v = np.cross(w, n)                          # across the board
    root = geom.arm_root(sensor_id)
    t = ARM_THICKNESS_MM

    foot_c = root + n * (ARM_MOUNT_LENGTH_MM / 2.0)
    fv, ff = _box(foot_c, n, v, w, ARM_MOUNT_LENGTH_MM, ARM_MOUNT_WIDTH_MM, t)

    rest = geom.arm_length_mm - ARM_MOUNT_LENGTH_MM
    arm_c = root + n * (ARM_MOUNT_LENGTH_MM + rest / 2.0)
    av, af = _box(arm_c, n, v, w, rest, ARM_WIDTH_MM, t)

    return (np.vstack([fv, av]).astype(np.float32),
            np.vstack([ff, af + len(fv)]).astype(np.int32))


def chip_mesh(geom, sensor_id, size_mm=CHIP_SIZE_MM):
    """The chip package itself, sitting on the board at the FSV."""
    n = geom.normal(sensor_id)
    w = geom.board_normal(sensor_id)
    v = np.cross(w, n)
    centre = (geom.position(sensor_id)
              - w * geom.fsv_above_board_mm
              + w * (ARM_THICKNESS_MM / 2.0 + 0.45))
    return _box(centre, n, v, w, size_mm, size_mm, 0.9)


def main():
    import argparse
    p = argparse.ArgumentParser(description="probe geometry helper")
    p.add_argument("--config", default=CONFIG_NAME)
    p.add_argument("--mapping", choices=MAPPINGS,
                   help="rewrite the config with this sensor->face mapping")
    p.add_argument("--board-plane", choices=BOARD_PLANES)
    p.add_argument("--width", type=float, help="tube width across the flats, mm")
    p.add_argument("--pitch", type=float, help="mounting plate pitch along the tube, mm")
    p.add_argument("--arm", type=float, help="PCB length, mm")
    p.add_argument("--magnet", nargs=3, type=float, metavar=("X", "Y", "Z"),
                   help="report the 1/r^3 expected response for a magnet here (mm)")
    a = p.parse_args()

    g = Geometry.load_or_default(a.config)
    changed = False
    if a.mapping:
        g.mapping = a.mapping
        g.sensors = _default_sensors(a.mapping)
        changed = True
    for attr, val in (("board_plane", a.board_plane), ("tube_width_mm", a.width),
                      ("plate_pitch_mm", a.pitch), ("arm_length_mm", a.arm)):
        if val is not None:
            setattr(g, attr, val)
            changed = True
    if changed:
        print(f"wrote {g.save(a.config)}")

    print(g.describe())
    if a.magnet:
        w = g.expected_response(a.magnet)
        d = g.distances_to(a.magnet)
        print(f"\nmagnet at {tuple(a.magnet)} mm -- nearest chip is "
              f"S{g.nearest_sensor(a.magnet)}")
        print(f"{'sensor':>7} {'r [mm]':>9} {'expected |B| rel.':>18}")
        for i in range(N_SENSORS):
            print(f"{'S'+str(i+1):>7} {d[i]:9.1f} {w[i]:18.3f}")
        print(f"\nspread from geometry alone: {w.max()/w.min():.1f}x -- any "
              f"amplitude comparison must divide this out first.")


if __name__ == "__main__":
    main()
