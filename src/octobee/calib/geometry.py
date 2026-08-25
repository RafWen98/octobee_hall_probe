#!/usr/bin/env python3
"""
octobee/calib/geometry.py -- where each sensor actually sits, and which way its chip faces.

The probe is a square tube carrying 16 SENM3Dx evaluation-kit PCBs, 4 per face.
Each board lies FLAT ON its face -- coplanar with it, bolted down through the
wide foot -- and reaches out sideways past the edge of the tube, tangentially.
The chip rides at the far tip. So the arms are not spokes pointing away from the
axis; they are tangents, and the four on a given ring form a pinwheel.

That matters for three separate things:

  1. Every chip has a DIFFERENT orientation, so comparing a single axis across
     sensors is meaningless. Compare |B| (rotation invariant), or rotate each
     chip's vector into the common tube frame with the matrices here.

  2. A magnet held near the probe is at a very different distance from each
     chip. Field falls off roughly as 1/r^3, so this alone produces large
     amplitude differences before any gain or calibration difference.
     expected_response() quantifies it so it can be divided out.

  3. Because the board lies flat, its 30 mm-wide foot runs ALONG the tube. That
     is what sets the 33 mm plate pitch (30 mm plate + ~3 mm gap), and it is why
     four boards fit in a row on a 25.4 mm face without fouling each other.

Dimensions come from the SENIS eval-kit drawing (Figure 2) and the FSV detail
(Figure 3):

    PCB overall length              92 mm      (tangential, out from the tube)
    mounting foot                   30 mm wide (along the tube), first 20 mm
    arm                             12 mm wide (along the tube)
    PCB thickness                  1.2 mm
    FSV inset from the far end       3 mm      -> 89 mm from the mounting edge
    FSV depth below the chip lid  0.35 mm
    chip package height            0.9 mm     -> FSV sits 0.55 mm above the
                                                 board's top face

Tube frame convention
---------------------
    +Z  along the tube, away from the mounting flange (toward the tip)
    +X  outward normal of face 0
    +Y  outward normal of face 1
    faces 0,1,2,3 at 0, 90, 180, 270 degrees about the tube axis

World frame (bench mounting, used only for drawing)
---------------------------------------------------
The probe is mounted horizontally: the tube axis runs along the rig's Y with
the tip end forward, and Z stays up. MOUNT_ROT / to_world() carry that, and
only the 3D view uses them -- see the note beside MOUNT_ROT.

Chip local frame (before chip_rot_deg)
--------------------------------------
    local +X  along the arm, pointing away from the tube (tangential)
    local +Z  out of the PCB surface, i.e. along the face's outward normal
    local +Y  completes the right-handed set (runs along the tube)

WHICH SENSOR IS ON WHICH FACE IS NOT YET VERIFIED on this hardware -- see the
notes in probe_geometry.json. arm_sense is: the arms reach clockwise off their
faces seen looking toward the tip, which is arm_sense -1.
"""

import argparse
import json
import os

from octobee import paths

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

# The two arrangements this module can GENERATE. A config file may also say
# "measured", which means the sensors list in the file came off the bench and
# is the answer -- see _default_sensors.
MAPPINGS = ("face-major", "ring-major")
MEASURED = "measured"

# How a board is attached to its face.
#   tangential  the board lies flat on the face and reaches out sideways past
#               the tube edge, staying in the plane of the face  (this probe)
#   radial      the board stands off the face like a spoke        (kept so the
#               earlier reading of the build can still be compared against)
MOUNT_STYLES = ("tangential", "radial")

TUBE_AXIS = np.array([0.0, 0.0, 1.0])

# ---- how the probe is actually mounted -----------------------------------
# The tube frame above keeps +Z along the tube because every calibration, pose
# solve and export is written in it, and none of them care which way the rig is
# bolted together. The bench does: the probe hangs horizontally, its axis along
# the rig's Y, tip end (slot 3 -- S4, S8, S12, S16) pointing forward, and Z is
# up. MOUNT_ROT is that mounting and nothing else, so the 3D view can draw the
# probe the way it lies on the bench without any of the maths changing frame.
#
#     tube +Z (toward the tip)  ->  world +Y (forward)
#     tube +X (face 0 normal)   ->  world +X (right)
#     tube +Y (face 1 normal)   ->  world -Z (down)
#
# It is a proper rotation (det +1), so mesh winding and handedness survive it.
MOUNT_ROT = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, -1.0, 0.0]])


def to_world(v):
    """(..., 3) tube-frame points or vectors -> the world frame the view draws."""
    return np.asarray(v, float) @ MOUNT_ROT.T


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
    """sensor id (1..16) -> (face, slot along the tube), for a GENERATED layout.

    Refuses anything it cannot generate rather than falling back to face-major.
    A file marked "measured" carries an arrangement established with a magnet;
    regenerating it would replace a measurement with a guess that looks exactly
    like one -- which is the failure this whole module is written against.
    """
    if mapping not in MAPPINGS:
        raise ValueError(
            f"cannot generate the layout {mapping!r}. "
            + (f"{MEASURED!r} means the sensors list in the file IS the "
               f"result -- keep it, or re-run the guided magnet calibration "
               f"to establish it again."
               if mapping == MEASURED else
               f"Known layouts: {', '.join(MAPPINGS)}."))
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

    def __init__(self, tube_width_mm=25.4,
                 arm_length_mm=ARM_LENGTH_MM,
                 fsv_from_tip_mm=FSV_FROM_TIP_MM,
                 fsv_above_board_mm=FSV_ABOVE_BOARD_MM,
                 board_thickness_mm=ARM_THICKNESS_MM,
                 mount_standoff_mm=0.0,
                 mount_inset_mm=0.0,
                 plate_pitch_mm=None,
                 first_sensor_z_mm=30.0,
                 tube_length_mm=None,
                 mount_style="tangential",
                 arm_sense=-1,
                 mapping="face-major", sensors=None, notes="",
                 board_plane=None):
        self.notes = notes
        self._rot_sig = None
        self._rot_cache = None
        self.tube_width_mm = float(tube_width_mm)
        self.arm_length_mm = float(arm_length_mm)
        self.fsv_from_tip_mm = float(fsv_from_tip_mm)
        self.fsv_above_board_mm = float(fsv_above_board_mm)
        self.board_thickness_mm = float(board_thickness_mm)
        self.mount_standoff_mm = float(mount_standoff_mm)
        self.mount_inset_mm = float(mount_inset_mm)
        self.plate_pitch_mm = float(
            ARM_MOUNT_WIDTH_MM + PLATE_GAP_MM if plate_pitch_mm is None
            else plate_pitch_mm)
        self.first_sensor_z_mm = float(first_sensor_z_mm)
        self.mount_style = mount_style
        self.arm_sense = 1 if float(arm_sense) >= 0 else -1
        # board_plane belonged to the older radial reading of the build; accept
        # it from an old config file so nothing breaks, but it is unused now.
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
                "board_thickness_mm": self.board_thickness_mm,
                "mount_standoff_mm": self.mount_standoff_mm,
                "mount_inset_mm": self.mount_inset_mm,
                "plate_pitch_mm": self.plate_pitch_mm,
                "first_sensor_z_mm": self.first_sensor_z_mm,
                "tube_length_mm": self.tube_length_mm,
                "mount_style": self.mount_style,
                "arm_sense": self.arm_sense,
                "mapping": self.mapping,
                "notes": self.notes,
                "sensors": self.sensors}

    def save(self, path=None):
        path = path or paths.config(CONFIG_NAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    # Unrecognised keys are ignored rather than fatal, so a hand-added note in
    # probe_geometry.json cannot drop the whole file back to defaults. Same
    # reasoning as Calibration._FIELDS: a plausible-looking default geometry is
    # worse than a loud error.
    _FIELDS = frozenset((
        "tube_width_mm", "arm_length_mm", "fsv_from_tip_mm",
        "fsv_above_board_mm", "board_thickness_mm", "mount_standoff_mm",
        "mount_inset_mm", "plate_pitch_mm", "first_sensor_z_mm",
        "tube_length_mm", "mount_style", "arm_sense", "mapping", "sensors",
        "notes", "board_plane"))

    @classmethod
    def load(cls, path=None):
        path = path or paths.config(CONFIG_NAME)
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict):
            raise ValueError(f"{path}: expected a JSON object, got "
                             f"{type(doc).__name__}")
        return cls(**{k: v for k, v in doc.items() if k in cls._FIELDS})

    @classmethod
    def load_or_default(cls, path=None, on_error=None):
        """
        Load `path`, or fall back to the nominal geometry.

        A file that exists but will not parse is reported through `on_error`.
        The default geometry is a guess about which chip is on which face, and
        a wrong one produces a tube-frame export that looks entirely
        plausible -- so silently substituting it is the worst of the options.
        """
        path = path or paths.config(CONFIG_NAME)
        if path and os.path.exists(path):
            try:
                return cls.load(path)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                if on_error is not None:
                    on_error(f"{path} exists but could not be read "
                             f"({type(exc).__name__}: {exc}) -- falling back "
                             f"to the nominal geometry, whose sensor-to-face "
                             f"mapping is an assumption.")
        return cls()

    # ---- per-sensor frame ------------------------------------------------
    def face(self, sensor_id):
        return self.sensors[sensor_id - 1]["face"]

    def slot(self, sensor_id):
        return self.sensors[sensor_id - 1]["slot"]

    def normal(self, sensor_id):
        """Outward normal of the face this board is bolted to."""
        return FACE_NORMALS[self.sensors[sensor_id - 1]["face"]].copy()

    def normals(self):
        return np.array([self.normal(i) for i in range(1, N_SENSORS + 1)])

    def sense(self, sensor_id):
        """Which way round the tube this arm points: +1 or -1.

        -1 on this probe: looking along the tube toward the tip, each board
        reaches clockwise off the face it is bolted to. It is a sign on one
        cross product here, but it mirrors the whole head -- and a mirrored
        head still looks entirely plausible, which is why it is written down.
        """
        return float(self.sensors[sensor_id - 1].get("arm_sense", self.arm_sense))

    def arm_dir(self, sensor_id):
        """Unit vector along the board, from the mounting foot to the chip."""
        n = self.normal(sensor_id)
        if self.mount_style == "radial":
            return n
        # Tangential: perpendicular to both the face normal and the tube axis,
        # so it runs across the face and off the edge.
        t = np.cross(TUBE_AXIS, n) * self.sense(sensor_id)
        return t / np.linalg.norm(t)

    def board_normal(self, sensor_id):
        """Unit normal of the PCB surface, pointing away from the tube."""
        n = self.normal(sensor_id)
        if self.mount_style == "radial":
            e = np.cross(n, TUBE_AXIS)
            return e / np.linalg.norm(e)
        return n                       # the board lies flat on the face

    def board_width_dir(self, sensor_id):
        """Across the board -- along the tube for a tangential mount."""
        return np.cross(self.board_normal(sensor_id), self.arm_dir(sensor_id))

    # ---- positions -------------------------------------------------------
    @property
    def face_radius_mm(self):
        """Distance from the tube axis out to a face."""
        return self.tube_width_mm / 2.0 + self.mount_standoff_mm

    @property
    def fsv_height_mm(self):
        """How far the FSV stands off the board's mounting surface."""
        return self.board_thickness_mm + self.fsv_above_board_mm

    @property
    def fsv_reach_mm(self):
        """Distance from the mounting edge of the board to the FSV."""
        return self.arm_length_mm - self.fsv_from_tip_mm

    def slot_z_mm(self, sensor_id):
        return (self.first_sensor_z_mm
                + self.slot(sensor_id) * self.plate_pitch_mm)

    def mount_edge_offset_mm(self):
        """
        Where the board's mounting edge sits along the arm direction, measured
        from the tube axis. The board starts at the trailing edge of the face
        and runs across it, so that is -half a tube width.
        """
        if self.mount_style == "radial":
            return self.face_radius_mm
        return -self.tube_width_mm / 2.0 + self.mount_inset_mm

    def arm_root(self, sensor_id):
        """The board's mounting edge, in the tube frame, mm."""
        a = self.arm_dir(sensor_id)
        base = TUBE_AXIS * self.slot_z_mm(sensor_id) + a * self.mount_edge_offset_mm()
        if self.mount_style == "radial":
            return base
        return base + self.normal(sensor_id) * self.face_radius_mm

    def position(self, sensor_id):
        """The chip's field sensitive volume in the tube frame, mm."""
        a = self.arm_dir(sensor_id)
        w = self.board_normal(sensor_id)
        if self.mount_style == "radial":
            return (a * (self.face_radius_mm + self.fsv_reach_mm)
                    + TUBE_AXIS * self.slot_z_mm(sensor_id)
                    + w * self.fsv_above_board_mm)
        return (self.normal(sensor_id) * (self.face_radius_mm + self.fsv_height_mm)
                + a * (self.mount_edge_offset_mm() + self.fsv_reach_mm)
                + TUBE_AXIS * self.slot_z_mm(sensor_id))

    def positions(self):
        return np.array([self.position(i) for i in range(1, N_SENSORS + 1)])

    @property
    def fsv_radius_mm(self):
        """Largest distance of any chip from the tube axis -- the probe's reach."""
        p = self.positions()
        return float(np.max(np.hypot(p[:, 0], p[:, 1])))

    # ---- orientation -----------------------------------------------------
    def rotation(self, sensor_id):
        """
        3x3 matrix R with B_tube = R @ B_chip, for B_chip in the chip's own
        (Bx, By, Bz). Sign flips are folded in by rotations().
        """
        e1 = self.arm_dir(sensor_id)             # chip +X: out along the arm
        e3 = self.board_normal(sensor_id)        # chip +Z: out of the board
        e2 = np.cross(e3, e1)                    # chip +Y completes the set
        psi = np.deg2rad(self.sensors[sensor_id - 1].get("chip_rot_deg", 0.0))
        c, s = np.cos(psi), np.sin(psi)
        return np.column_stack([c * e1 + s * e2, -s * e1 + c * e2, e3])

    def axis_signs(self, sensor_id):
        return np.array(self.sensors[sensor_id - 1].get("axis_signs", [1, 1, 1]),
                        dtype=float)

    def _rot_signature(self):
        """Cheap fingerprint of everything rotations() depends on."""
        # Per-sensor arm_sense belongs here as much as the global one does:
        # sense() honours the override, so leaving it out of the fingerprint
        # meant flipping one arm changed arm_dir() and rotation() but never
        # invalidated the cache. The correction appeared to do nothing, and the
        # tube-frame export, the 3D view and identify_faces() all carried on
        # with the old orientation. arm_sense is one of the three things
        # probe_geometry.json still marks UNVERIFIED, so it is exactly the
        # field someone will edit.
        return (self.mount_style, self.arm_sense,
                tuple((s["face"], s.get("chip_rot_deg", 0.0),
                       s.get("arm_sense", self.arm_sense),
                       tuple(s.get("axis_signs", (1, 1, 1))))
                      for s in self.sensors))

    def rotations(self):
        """
        (16, 3, 3) stack of R_i with the per-axis sign flips folded in.

        Cached. This is called on every 3D frame and on every tube-frame
        conversion, and rebuilding 16 matrices from trig and cross products
        each time was costing more than the field maths it serves. The cache
        keys on a fingerprint rather than a dirty flag because Geometry is
        edited by plain attribute assignment.
        """
        sig = self._rot_signature()
        if sig != self._rot_sig:
            self._rot_cache = np.array(
                [self.rotation(i) * self.axis_signs(i)[None, :]
                 for i in range(1, N_SENSORS + 1)])
            self._rot_sig = sig
        return self._rot_cache

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
            f"boards {self.mount_style}: {self.arm_length_mm:g} mm long, FSV "
            f"{self.fsv_from_tip_mm:g} mm from the tip -> chips reach "
            f"{self.fsv_radius_mm:.1f} mm from the tube axis, standing "
            f"{self.fsv_height_mm:g} mm off the face they sit on",
            f"mounting plates every {self.plate_pitch_mm:g} mm along the tube, "
            f"arm sense {self.arm_sense:+d}, mapping '{self.mapping}'"]
        for i in range(1, N_SENSORS + 1):
            p = self.position(i)
            lines.append(f"  S{i:<2d} face {FACE_NAMES[self.face(i)]} slot {self.slot(i)}"
                         f"  FSV ({p[0]:7.1f},{p[1]:7.1f},{p[2]:7.1f}) mm")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# meshes for the 3D view
# --------------------------------------------------------------------------

def _box(centre, u, v, w, su, sv, sw):
    """Box given its own three axes and sizes -> (verts, faces)."""
    verts = []
    for du in (-su / 2, su / 2):
        for dv in (-sv / 2, sv / 2):
            for dw in (-sw / 2, sw / 2):
                verts.append(centre + u * du + v * dv + w * dw)
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
    The eval-kit PCB lying on its face: a wide foot bolted to the tube, then a
    narrower arm running out past the edge to the chip. Two boxes, merged.
    """
    a = geom.arm_dir(sensor_id)                 # along the board
    w = geom.board_normal(sensor_id)            # out of the board
    v = geom.board_width_dir(sensor_id)         # across the board
    root = geom.arm_root(sensor_id)
    t = geom.board_thickness_mm
    # The board's underside rests on the face, so the slab centre sits half a
    # thickness proud of it.
    lift = w * (t / 2.0)

    foot_c = root + a * (ARM_MOUNT_LENGTH_MM / 2.0) + lift
    fv, ff = _box(foot_c, a, v, w, ARM_MOUNT_LENGTH_MM, ARM_MOUNT_WIDTH_MM, t)

    rest = geom.arm_length_mm - ARM_MOUNT_LENGTH_MM
    arm_c = root + a * (ARM_MOUNT_LENGTH_MM + rest / 2.0) + lift
    av, af = _box(arm_c, a, v, w, rest, ARM_WIDTH_MM, t)

    return (np.vstack([fv, av]).astype(np.float32),
            np.vstack([ff, af + len(fv)]).astype(np.int32))


def chip_mesh(geom, sensor_id, size_mm=CHIP_SIZE_MM):
    """The chip package itself, sitting on the board under the FSV."""
    a = geom.arm_dir(sensor_id)
    w = geom.board_normal(sensor_id)
    v = geom.board_width_dir(sensor_id)
    # Centre the package on the board surface beneath the FSV.
    centre = geom.position(sensor_id) - w * (geom.fsv_above_board_mm - 0.45)
    return _box(centre, a, v, w, size_mm, size_mm, 0.9)


def main():
    p = argparse.ArgumentParser(description="probe geometry helper")
    p.add_argument("--config", default=paths.config(CONFIG_NAME))
    p.add_argument("--mapping", choices=MAPPINGS,
                   help="rewrite the config with this sensor->face mapping")
    p.add_argument("--mount-style", choices=MOUNT_STYLES)
    p.add_argument("--arm-sense", type=int, choices=(1, -1),
                   help="which way round the tube the arms point")
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
    for attr, val in (("mount_style", a.mount_style), ("arm_sense", a.arm_sense),
                      ("tube_width_mm", a.width), ("plate_pitch_mm", a.pitch),
                      ("arm_length_mm", a.arm)):
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
