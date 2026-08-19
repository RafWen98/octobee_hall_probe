#!/usr/bin/env python3
"""
probe_geometry.py -- where each sensor sits on the tube, and which way its chip faces.

The probe is a square tube with 16 SENM3Dx chips mounted on it, 4 per face,
each pointing radially outwards ("toilet-brush"). Two consequences drive
everything else in this repo:

  1. Every chip has a DIFFERENT orientation. Chip 3's "Bx" and chip 11's "Bx"
     point in different lab-frame directions, so comparing a single axis across
     sensors is meaningless. Either compare |B| (rotation invariant), or rotate
     each chip's vector into the common tube frame with the matrices here.

  2. A magnet held at a fixed distance FROM THE TUBE is at a different distance
     from each chip, depending on which face it is on and where along the tube.
     Field falls off roughly as 1/r^3, so this alone produces large amplitude
     differences between sensors -- before any gain or calibration difference.
     expected_response() below quantifies that so it can be divided out.

Tube frame convention
---------------------
    +Z  along the tube, pointing away from the mounting flange (toward the tip)
    +X  outward normal of face 0
    +Y  outward normal of face 1
    faces 0,1,2,3 are at 0, 90, 180, 270 degrees about the tube axis

Chip local frame (before any chip_rot_deg is applied)
-----------------------------------------------------
    local +Z  = outward normal of the face it sits on   (radially out of the tube)
    local +X  = along the tube, toward +Z
    local +Y  = local Z cross local X

WHICH SENSOR IS ON WHICH FACE IS NOT YET VERIFIED on this hardware. The default
"face-major" mapping (S1-S4 on face 0, S5-S8 on face 1, ...) is an assumption.
Run octobee_idmap.py with a slow magnet pass along one face to find the real
order, then edit probe_geometry.json -- nothing else needs to change.
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

    def __init__(self, tube_width_mm=40.0, sensor_pitch_mm=60.0,
                 first_sensor_z_mm=40.0, tube_length_mm=None,
                 mapping="face-major", sensors=None, notes=""):
        self.notes = notes
        self.tube_width_mm = float(tube_width_mm)
        self.sensor_pitch_mm = float(sensor_pitch_mm)
        self.first_sensor_z_mm = float(first_sensor_z_mm)
        self.mapping = mapping
        self.sensors = sensors or _default_sensors(mapping)
        if tube_length_mm is None:
            span = self.first_sensor_z_mm + (SENSORS_PER_FACE - 1) * self.sensor_pitch_mm
            tube_length_mm = span + self.first_sensor_z_mm
        self.tube_length_mm = float(tube_length_mm)

    # ---- persistence ----------------------------------------------------
    def to_dict(self):
        return {"tube_width_mm": self.tube_width_mm,
                "sensor_pitch_mm": self.sensor_pitch_mm,
                "first_sensor_z_mm": self.first_sensor_z_mm,
                "tube_length_mm": self.tube_length_mm,
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

    def position(self, sensor_id):
        """Chip centre in the tube frame, mm."""
        s = self.sensors[sensor_id - 1]
        n = FACE_NORMALS[s["face"]]
        z = self.first_sensor_z_mm + s["slot"] * self.sensor_pitch_mm
        return n * (self.tube_width_mm / 2.0) + np.array([0.0, 0.0, z])

    def positions(self):
        return np.array([self.position(i) for i in range(1, N_SENSORS + 1)])

    def normal(self, sensor_id):
        return FACE_NORMALS[self.sensors[sensor_id - 1]["face"]].copy()

    def normals(self):
        return np.array([self.normal(i) for i in range(1, N_SENSORS + 1)])

    def rotation(self, sensor_id):
        """
        3x3 matrix R with B_tube = R @ B_chip, where B_chip is (Bx, By, Bz)
        in that chip's own axes. Sign flips are folded in by rotations().
        """
        s = self.sensors[sensor_id - 1]
        e3 = FACE_NORMALS[s["face"]]                 # outward
        e1 = np.array([0.0, 0.0, 1.0])               # along the tube
        e2 = np.cross(e3, e1)
        psi = np.deg2rad(s.get("chip_rot_deg", 0.0))
        c, sn = np.cos(psi), np.sin(psi)
        return np.column_stack([c * e1 + sn * e2, -sn * e1 + c * e2, e3])

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
        """Distance from each chip to a point in the tube frame, mm."""
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
        lines = [f"square tube {self.tube_width_mm:g} mm across, "
                 f"{self.tube_length_mm:g} mm long, {self.sensor_pitch_mm:g} mm pitch, "
                 f"mapping '{self.mapping}'"]
        for i in range(1, N_SENSORS + 1):
            p = self.position(i)
            lines.append(f"  S{i:<2d} face {FACE_NAMES[self.face(i)]} slot {self.slot(i)}"
                         f"  pos ({p[0]:6.1f},{p[1]:6.1f},{p[2]:6.1f}) mm")
        return "\n".join(lines)


def tube_mesh(geom, cap=True):
    """
    Vertices and faces of the square tube, for the 3D view.
    Returns (verts (n,3), faces (m,3)) in mm, tube frame.
    """
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


def pad_mesh(geom, sensor_id, size_mm=14.0, thick_mm=3.0):
    """A small slab standing proud of the face, marking one chip."""
    n = geom.normal(sensor_id)
    p = geom.position(sensor_id)
    R = geom.rotation(sensor_id)
    u, v = R[:, 0], R[:, 1]                      # in-plane axes of that face
    s = size_mm / 2.0
    verts, faces = [], []
    for d in (0.0, thick_mm):
        base = p + n * d
        verts += [base + u * a + v * b for a, b in
                  ((s, s), (-s, s), (-s, -s), (s, -s))]
    for i in range(4):
        j = (i + 1) % 4
        faces += [[i, j, j + 4], [i, j + 4, i + 4]]
    faces += [[4, 5, 6], [4, 6, 7], [0, 2, 1], [0, 3, 2]]
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def main():
    import argparse
    p = argparse.ArgumentParser(description="probe geometry helper")
    p.add_argument("--config", default=CONFIG_NAME)
    p.add_argument("--mapping", choices=MAPPINGS,
                   help="write a fresh config using this sensor->face mapping")
    p.add_argument("--width", type=float, help="tube width across the flats, mm")
    p.add_argument("--pitch", type=float, help="spacing between chips along a face, mm")
    p.add_argument("--magnet", nargs=3, type=float, metavar=("X", "Y", "Z"),
                   help="report the 1/r^3 expected response for a magnet here (mm)")
    a = p.parse_args()

    if a.mapping or a.width or a.pitch:
        g = Geometry.load_or_default(a.config)
        if a.mapping:
            g = Geometry(g.tube_width_mm, g.sensor_pitch_mm, g.first_sensor_z_mm,
                         None, a.mapping, _default_sensors(a.mapping))
        if a.width:
            g.tube_width_mm = a.width
        if a.pitch:
            g.sensor_pitch_mm = a.pitch
        print(f"wrote {g.save(a.config)}")
    else:
        g = Geometry.load_or_default(a.config)

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
