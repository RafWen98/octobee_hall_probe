#!/usr/bin/env python3
"""
hardware/rig_v2/build_rig.py -- parametric CAD for the reworked Hall-probe rig.

Builds every 3D-printed part as an STL (print orientation, mm) and a coloured
assembly (out/rig_v2.glb) of the whole rig for the viewer. Pure CSG on
manifold3d, so it needs no CAD program:

    python -m venv cadenv && cadenv/Scripts/pip install manifold3d trimesh numpy
    cadenv/Scripts/python hardware/rig_v2/build_rig.py

Everything is driven by the parameters below. The ones marked MEASURE are
numbers the photos and vendor drawings could not give -- check them on the
bench before printing the part that uses them.

Probe-head geometry is NOT a parameter here: it comes from
config/probe_geometry.json through octobee.calib.geometry, the same code the
software uses, and the build fails if the printed seats do not put every FSV
where that code says it is. Print the nose and the software needs no change.

Frames
------
tube    +Z along the probe toward the tip, faces 0..3 = +X,+Y,-X,-Y (geometry.py)
world   +X along the X stage, +Y toward the coil (the probe axis), +Z up
groups  fixed / xcar / zcar / ycar: the parts that ride each stage carriage,
        each in its own local frame. The viewer nests them and drives them
        from the stage sliders; GROUP_ORIGIN below is the same arithmetic.
"""

import json
import math
import os
import sys

import numpy as np
import manifold3d as m3
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from octobee.calib.geometry import Geometry          # noqa: E402

OUT = os.path.join(HERE, "out")
STL_DIR = os.path.join(OUT, "stl")
M = m3.Manifold
CS = m3.CrossSection
SEG = 40

# ==========================================================================
# parameters
# ==========================================================================
GEOM = Geometry.load()
TUBE = GEOM.tube_width_mm               # 25.4 -- the printed core keeps it
HALF = TUBE / 2.0
SENSOR_LEN = GEOM.tube_length_mm        # 159 -- sensor section of the nose

# eval-kit PCB (SENIS SENM3Dx-EKv2.4 drawing, Figure 2)
PCB_T = GEOM.board_thickness_mm         # 1.2
PCB_HOLE_U, PCB_HOLE_V = 19.0, 10.0     # 19 mm in from root edge, +/-10 across
PCB_OUTLINE = [(0, -15), (20, -15), (31, -3), (92, -3),
               (92, 3), (31, 3), (20, 15), (0, 15)]

# heat-set inserts -- BRASS only near the head (non-magnetic)
M2_INSERT_D, M2_INSERT_DEPTH = 3.2, 4.5
M3_INSERT_D, M3_INSERT_DEPTH = 4.0, 6.0
M4_INSERT_D, M4_INSERT_DEPTH = 5.6, 6.0
M5_CLEAR = 5.5
QTR20_CLEAR = 6.8                       # 1/4-20 or M6 table screws

# nose <-> coupler interface
SPIGOT = 24.4                           # spigot is 1 mm smaller than the core:
SPIGOT_LEN = 35.0                       # the 0.5 mm shoulder is the axial datum
SOCKET_CLEAR = 0.3
KEY = 4.0                               # corner chamfer, face0/face1 corner
CORE_BORE_R = 6.5

# 2040 boom, 40 mm vertical
EXT_W, EXT_H = 20.0, 40.0
BOOM_LEN = 600.0
SLOT_FROM_EDGE = 10.0                   # 20-series slot centres
COUPLER_NECK = 60.0                     # extra plastic between Al and head
COUPLER_SIDE = 31.0                     # neck / socket square

# MEASURE: the OCTO-BEE PCB. CONC and CELF are the same board (silkscreen
# "OCTO-BEE PCB 2100-5013-001 REV A"), populated differently. No drawing is
# published; these come from photo 1 (CELF, near face-on), scaled off the
# 3.81 mm terminal blocks and the 33 mm 2x10 box headers. Expect +/-5 %.
# Board frame: u across (0 = the edge with the 50-way ANALOG CONNs),
# v along (0 = the encoder-terminal end, OCTO_L = the FFC end).
OCTO_W, OCTO_L, OCTO_T = 90.0, 189.0, 1.6
OCTO_HOLES = [(23.0, 26.0), (85.0, 21.0), (27.0, 71.0), (87.0, 66.0), (35.0, 142.0)]
OCTO_STANDOFF = 10.0

# MEASURE: LTS300C carriage tapped-hole rows. The Y sled stands in the y-z
# plane, so the rows run along y and the slots run up and down (z).
CARRIAGE_ROWS_Y = [-50.8, -25.4, 25.4, 50.8]
CARRIAGE_SLOT_Z = (24.0, 48.0)

# MEASURE: ACQ1001 + OCTO-BEE-CELF in its aluminium U-chassis, standing up as
# it does today: CELF face CAR_W wide (along the boom), CAR_D deep, CAR_H tall.
CAR_W, CAR_D, CAR_H = 110.0, 85.0, 250.0
CARRIER_KG = 1.2                        # MEASURE: weigh one, chassis and all

# Z-axis load. Thorlabs: LTS300 2.5 kg (catalog), rated 4 kg vertical.
LTS300_KG = 2.5
Z_RATED_KG = 4.0
Z_NET_DOWN_KG = 1.0                     # leave this much un-balanced: keeps the
                                        # leadscrew loaded one way (no backlash flip)

# table: 1" grid, 1/4-20 (Newport / Thorlabs imperial breadboards)
GRID = 25.4

# drag chain (bought, not printed): openable, inner >= 15 x 50 mm. With the
# carriers on the boom no ribbon flexes any more: only round cables travel.
CHAIN_INNER_W, CHAIN_INNER_H = 50.0, 15.0
CHAIN_R = 48.0

# where the electronics tower stands on the X platform (xcar frame)
TOWER_XY = (-62.0, -253.0)

# the three chains, for the viewer: fixed point F (in group gF), moving point
# M (in group gM), run direction d, bend offset n (|M-F| along n = 2R), length.
CHAINS = [
    {"id": "x_umbilical", "gF": "fixed", "F": [-250, -630, 14], "gM": "xcar", "M": [-100, -380, 32],
     "d": [1, 0, 0], "n": [0, 0, 1], "R": CHAIN_R, "L": 520, "w": 60, "h": 28,
     "carries": "DC leads (2x ACQ, Y + Z stage, switch, hub), X encoder, 1x USB, 1x Ethernet"},
    {"id": "z_chain", "gF": "xcar", "F": [-38, -253, 40], "gM": "zcar", "M": [60, -253, 0],
     "d": [0, 0, 1], "n": [1, 0, 0], "R": CHAIN_R, "L": 600, "w": 64, "h": 23,
     "carries": "2x Ethernet, 2x ACQ DC, X + Z encoders, Y-stage USB + DC"},
    {"id": "y_chain", "gF": "zcar", "F": [42, -185, 61.5], "gM": "ycar", "M": [-46, 0, 157.5],
     "d": [0, 1, 0], "n": [0, 0, 1], "R": CHAIN_R, "L": 630, "w": 64, "h": 23,
     "carries": "2x Ethernet, 2x ACQ DC, X + Y + Z encoders"},
]

MATERIALS = {
    "printed":   ((232, 119, 34, 255), 0.0, 0.55),
    "printed2":  ((245, 245, 240, 255), 0.0, 0.6),
    "alu":       ((188, 194, 200, 255), 0.85, 0.35),
    "stage":     ((32, 33, 36, 255), 0.3, 0.5),
    "pcb":       ((24, 112, 64, 255), 0.0, 0.6),
    "chip":      ((15, 15, 15, 255), 0.0, 0.4),
    "conn":      ((30, 30, 30, 255), 0.0, 0.6),
    "terminal":  ((90, 200, 120, 255), 0.0, 0.6),
    "ribbon":    ((214, 214, 214, 255), 0.0, 0.8),
    "carrier":   ((200, 203, 208, 255), 0.7, 0.4),
    "box":       ((45, 47, 52, 255), 0.1, 0.6),
    "table":     ((40, 42, 46, 255), 0.4, 0.6),
    "coil":      ((184, 108, 52, 255), 0.8, 0.35),
    "volume":    ((80, 150, 230, 50), 0.0, 1.0),
    "white":     ((236, 236, 236, 255), 0.0, 0.7),
}

# ==========================================================================
# CSG helpers
# ==========================================================================
def box(lo, hi):
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    return M.cube((hi - lo).tolist(), False).translate(lo.tolist())


def rbox(lo, hi, r):
    """Box rounded on its vertical (z) edges."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    h = hi[2] - lo[2]
    cs = []
    for x in (lo[0] + r, hi[0] - r):
        for y in (lo[1] + r, hi[1] - r):
            cs.append(M.cylinder(h, r, r, SEG, False).translate([x, y, lo[2]]))
    return M.batch_hull(cs)


def cyl_between(p0, p1, r, seg=SEG):
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    z = d / length
    helper = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c = M.cylinder(length, r, r, seg, False)
    return c.transform(np.column_stack([x, y, z, p0]))


def cyl_axis(r, center, axis, length, seg=SEG):
    c = np.asarray(center, float)
    a = np.zeros(3)
    a["xyz".index(axis)] = length / 2.0
    return cyl_between(c - a, c + a, r, seg)


def slot(p0, p1, r, h, z0=0.0):
    """Vertical stadium slot from p0 to p1 (xy), height h from z0."""
    a = M.cylinder(h, r, r, SEG, False).translate([p0[0], p0[1], z0])
    b = M.cylinder(h, r, r, SEG, False).translate([p1[0], p1[1], z0])
    return M.batch_hull([a, b])


def prism(poly_xy, z0, z1):
    return M.extrude(CS([poly_xy]), z1 - z0).translate([0, 0, z0])


def union(parts):
    parts = [p for p in parts if p is not None]
    return M.batch_boolean(parts, m3.OpType.Add)


def place(m, R, t):
    return m.transform(np.column_stack([np.asarray(R, float), np.asarray(t, float)]))


def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


I3 = np.eye(3)

# ==========================================================================
# probe-head geometry, straight from octobee.calib.geometry
# ==========================================================================
def board_frame(sid):
    """(R, root): R columns = (arm, across-board, board normal), tube frame."""
    a = GEOM.arm_dir(sid)
    n = GEOM.board_normal(sid)
    w = GEOM.board_width_dir(sid)
    return np.column_stack([a, w, n]), GEOM.arm_root(sid)


def eval_board(with_holes=True):
    pcb = prism(PCB_OUTLINE, 0.0, PCB_T)
    if with_holes:
        for v in (-PCB_HOLE_V, PCB_HOLE_V):
            pcb = pcb - M.cylinder(4, 1.1, 1.1, 24, True).translate([PCB_HOLE_U, v, 0])
    return pcb


def eval_board_parts():
    fsv_u = GEOM.arm_length_mm - GEOM.fsv_from_tip_mm
    chip = box((fsv_u - 2.5, -2.5, PCB_T), (fsv_u + 2.5, 2.5, PCB_T + 0.9))
    header = box((3, -13, PCB_T), (11, 13, PCB_T + 9))
    return eval_board(), chip, header


# ==========================================================================
# printed parts
# ==========================================================================
def nose_core():
    """
    The probe nose: a printed 25.4 mm square core with 32 brass M2 insert
    seats, one pair per eval board, at exactly the positions geometry.py
    assumes. Keyed spigot so it can only go in one way round -- turning the
    tube in its cradle renames every face, and this makes that impossible.
    Tube frame, spigot down: print standing on the spigot.
    """
    core = box((-HALF, -HALF, 0), (HALF, HALF, SENSOR_LEN))
    s = SPIGOT / 2
    spig = box((-s, -s, -SPIGOT_LEN), (s, s, 0.01))
    # key: chamfer the face0 / face1 (+X/+Y) corner of the spigot
    key = prism([(s - KEY, s + 1), (s + 1, s - KEY), (s + 1, s + 1)], -SPIGOT_LEN - 1, 0.0)
    spig = spig - key
    # 0.6 mm tip chamfer so the nose finds the socket
    body = core + spig
    body = body - cyl_axis(CORE_BORE_R, (0, 0, (SENSOR_LEN - SPIGOT_LEN) / 2),
                           "z", SENSOR_LEN + SPIGOT_LEN + 2)
    holes = []
    for sid in range(1, 17):
        R, root = board_frame(sid)
        n = R[:, 2]
        for v in (-PCB_HOLE_V, PCB_HOLE_V):
            p = root + R @ np.array([PCB_HOLE_U, v, 0.0])
            holes.append(cyl_between(p - n * M2_INSERT_DEPTH, p + n * 0.5,
                                     M2_INSERT_D / 2, 24))
    # face-index dimples on the spigot: face k gets k+1 dimples
    for f, nrm in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
        for i in range(f + 1):
            z = -SPIGOT_LEN + 6 + i * 5
            c = np.array([nrm[0] * s, nrm[1] * s, z])
            holes.append(cyl_between(c - np.r_[nrm, 0] * 0.7, c + np.r_[nrm, 0] * 1.0, 1.1, 20))
    return body - union(holes)


def nose_coupler():
    """
    Joins the nose to the 2040 boom. Sleeve over the boom end (4x M5 into the
    side slots, use aluminium or brass T-nuts), a plastic neck that keeps the
    aluminium COUPLER_NECK + 36 mm behind the first board, and a keyed socket
    with one nylon M4 thumbscrew. Boom axis = local +Y; print standing on the
    sleeve end (y = 0 down).
    """
    w2, h2 = EXT_W / 2 + 0.2, EXT_H / 2 + 0.2
    sx, sz = w2 + 6.8, h2 + 6.8
    y_wall = 36.0
    y_neck0 = y_wall + 30.0
    y_sock0 = y_neck0 + COUPLER_NECK
    y_end = y_sock0 + 36.0
    q = COUPLER_SIDE / 2

    sleeve = box((-sx, 0, -sz), (sx, y_wall + 6, sz))
    trans = M.batch_hull([box((-sx, y_wall + 5, -sz), (sx, y_wall + 6, sz)),
                          box((-q, y_neck0, -q), (q, y_neck0 + 1, q))])
    neck = box((-q, y_neck0, -q), (q, y_end, q))
    boss = cyl_axis(6.0, (0, y_sock0 + 18, q + 2.5), "z", 5.0)
    body = union([sleeve, trans, neck, boss])

    cut = [box((-w2, -0.01, -h2), (w2, y_wall, h2)),                 # boom pocket
           cyl_axis(M5_CLEAR / 2, (0, y_wall + 3, 0), "y", 8),        # end M5
           cyl_axis(6.0, (0, (y_wall + y_sock0) / 2 + 3, 0), "y", y_sock0 - y_wall - 2)]
    for z in (-SLOT_FROM_EDGE, SLOT_FROM_EDGE):
        for yy in (11.0, 27.0):
            cut.append(cyl_axis(M5_CLEAR / 2, (0, yy, z), "x", 2 * sx + 2))
            for sgn in (-1, 1):
                cut.append(cyl_axis(5.25, (sgn * (sx + 0.5), yy, z), "x", 3.0))
    # keyed socket (tube frame -> local: X->x, Y->-z, Z->y)
    s = SPIGOT / 2 + SOCKET_CLEAR / 2
    cut.append(box((-s, y_sock0, -s), (s, y_end + 0.1, s)))
    # velcro grooves on the neck, ribbons lie on the neck faces
    for g0 in (y_neck0 + 8, y_neck0 + COUPLER_NECK - 20):
        cut.append(box((-q - 1, g0, -q - 1), (q + 1, g0 + 12, q + 1)) -
                   box((-q + 1.2, g0 - 1, -q + 1.2), (q - 1.2, g0 + 13, q - 1.2)))
    # thumbscrew insert
    cut.append(cyl_axis(M4_INSERT_D / 2, (0, y_sock0 + 18, q + 1), "z", 12))
    body = body - union(cut)
    # key fill: matches the spigot's chamfered face0/face1 corner, which is
    # tube (+X, +Y) = local (+x, -z). Triangle drawn in (x, y'), y' -> -z.
    k = KEY - 0.4
    tri = M.extrude(CS([[(s - k, s), (s, s - k), (s, s)]]), 36.0)
    tri = place(tri, rot_x(90), (0, y_sock0 + 36.0, 0))
    return body + tri


def tower_foot():
    """2040 upright on a 1-inch 1/4-20 breadboard. Slots span one grid pitch."""
    base = rbox((-60, -60, 0), (60, 60, 8), 8)
    sock = rbox((-17, -27, 8), (17, 27, 48), 4)
    gus = []
    tri_y = M.extrude(CS([[(0, 0), (31, 0), (0, 36)]]), 6).translate([0, 0, -3])
    tri_x = M.extrude(CS([[(0, 0), (41, 0), (0, 36)]]), 6).translate([0, 0, -3])
    for sgn in (-1, 1):
        # local x -> outward, local y -> up, local z -> across the wall
        gus.append(place(tri_y, np.array([[0, 0, 1], [sgn, 0, 0], [0, 1, 0]]), (0, sgn * 27, 8)))
        gus.append(place(tri_x, np.array([[sgn, 0, 0], [0, 0, 1], [0, 1, 0]]), (sgn * 17, 0, 8)))
    body = union([base, sock] + gus)
    cut = [box((-10.2, -20.2, 8), (10.2, 20.2, 49))]
    for y in (-10, 10):
        cut.append(cyl_axis(M5_CLEAR / 2, (0, y, 30), "x", 50))
    cut.append(cyl_axis(M5_CLEAR / 2, (0, 0, 30), "y", 70))
    # slots one grid pitch long, 3 pitches apart: always land on 1" holes
    for sx in (-1, 1):
        for sy in (-1, 1):
            cut.append(slot((sx * 1.5 * GRID, sy * 1.0 * GRID), (sx * 1.5 * GRID, sy * 2.0 * GRID),
                            QTR20_CLEAR / 2, 10, -1))
    return body - union(cut)


def device_shelf():
    """Shelf on the tower for the network switch and USB hub. Velcro straps."""
    back = place(rbox((-80, 0, 0), (80, 70, 6), 4), rot_x(90), (0, 6, 0))
    shelf = rbox((-80, 0, 0), (80, 120, 5), 6)
    lip = box((-80, 116, 0), (80, 120, 12))
    gus = []
    for x in (-60, 0, 60):
        tri = M.extrude(CS([[(0, 0), (60, 0), (0, 55)]]), 5).translate([0, 0, -2.5])
        gus.append(place(tri, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), (x, 0, 5)))
    body = union([back, shelf, lip] + gus)
    cut = []
    for x in (-10, 10):
        for z in (20, 55):
            cut.append(cyl_axis(M5_CLEAR / 2, (x, 3, z), "y", 20))
    for y in (25, 85):
        for x0 in (-72, 8):
            cut.append(box((x0, y, -1), (x0 + 3, y + 22, 7)))
            cut.append(box((x0 + 61, y, -1), (x0 + 64, y + 22, 7)))
    return body - union(cut)


def psu_cradle(w=62.0, h=34.0):
    """U-cradle for a laptop-style supply brick; two per brick; 1/4-20 or strap."""
    t = 4.0
    outer = rbox((-w / 2 - t, 0, 0), (w / 2 + t, 30, h * 0.75 + t), 3)
    inner = box((-w / 2, -1, t), (w / 2, 31, h + t + 10))
    flange = rbox((-w / 2 - 22, 0, 0), (w / 2 + 22, 30, t), 4)
    body = (outer + flange) - inner
    cut = [cyl_axis(QTR20_CLEAR / 2, (sgn * (w / 2 + 13), 15, 2), "z", 10) for sgn in (-1, 1)]
    cut += [box((sgn * (w / 2 + 5) - 1.5, 4, -1), (sgn * (w / 2 + 5) + 1.5, 26, t + 1)) for sgn in (-1, 1)]
    return body - union(cut)


def chain_end():
    """L-bracket for a bought drag chain's end piece. Slots fit M5 T-nuts or the table."""
    t = 6.0
    a = rbox((-30, 0, 0), (30, 50, t), 4)
    b = place(rbox((-30, 0, 0), (30, 50, t), 4), rot_x(90), (0, t, 0))
    gus = []
    for x in (-27, 21):
        tri = M.extrude(CS([[(0, 0), (40, 0), (0, 40)]]), 6)
        gus.append(place(tri, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), (x, t, t)))
    body = union([a, b] + gus)
    cut = [slot((-12, 22), (-12, 40), QTR20_CLEAR / 2, 10, -1),
           slot((12, 22), (12, 40), QTR20_CLEAR / 2, 10, -1)]
    for z in (22.0, 38.0):
        cut.append(place(slot((-18, 0), (18, 0), 2.2, 10, -2), rot_x(90), (0, t + 2, z)))
    return body - union(cut)


def rbox_x(lo, hi, r):
    """Box rounded on its edges parallel to x (for plates standing in the y-z plane)."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    cs = []
    for y in (lo[1] + r, hi[1] - r):
        for z in (lo[2] + r, hi[2] - r):
            cs.append(cyl_axis(r, ((lo[0] + hi[0]) / 2, y, z), "x", hi[0] - lo[0]))
    return M.batch_hull(cs)


def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


# Y-carriage frame (ycar): origin on the sled's outer face, centre of the
# carriage; +x away from the stage, +y along travel, +z up. The sled itself
# is x -14..0. Everything below is built in this frame.
PLATE_T = 10.0                          # carriage plate
BOOM_X0 = PLATE_T                       # boom inner face
BOOM_XC = BOOM_X0 + EXT_W / 2           # boom centre line x (z = 0)
SHELF_Z0, SHELF_T = EXT_H / 2, 8.0      # shelf sits on the boom's top face
SHELF_X1 = BOOM_X0 + CAR_D + 16.0
CAR_X0 = BOOM_X0 + 8.0                  # carriers' back face
SHELF_Y = (-155.0, -155.0 + 2 * CAR_W + 15.0)
CONC_Y0 = 140.0                         # rear end of the CONC frame
CONC_PLATE_T = 5.0


def carriage_plate():
    """
    Vertical plate on the Y sled. The boom bolts to its outer face (M5 from the
    inner face into the boom's inner slots, outside the carriage footprint);
    a tab on top carries the Y-chain arm. Print flat (inner face down).
    """
    body = union([rbox_x((0, -125, -60), (PLATE_T, 125, 60), 6),
                  rbox_x((0, -45, 40), (PLATE_T, 15, 100), 6)])
    cut = []
    for y in CARRIAGE_ROWS_Y:
        for sgn in (-1, 1):
            a, b = sgn * CARRIAGE_SLOT_Z[0], sgn * CARRIAGE_SLOT_Z[1]
            seg = place(slot((a, 0), (b, 0), QTR20_CLEAR / 2, PLATE_T + 2, -1), rot_y(-90),
                        (PLATE_T + 1, y, 0))
            cut.append(seg)
            cut.append(place(slot((a, 0), (b, 0), 5.6, 4, 0), rot_y(-90),
                             (PLATE_T + 0.01, y, 0)))                # head pocket, outer face
    for y in (-110.0, -80.0, 80.0, 110.0):
        for z in (-10.0, 10.0):
            cut.append(cyl_axis(M5_CLEAR / 2, (PLATE_T / 2, y, z), "x", PLATE_T + 2))
            cut.append(cyl_axis(5.2, (1.6, y, z), "x", 3.4))          # button head, inner face
    for z in (72.0, 90.0):
        cut.append(cyl_axis(M5_CLEAR / 2, (PLATE_T / 2, -10, z), "x", PLATE_T + 2))
    return body - union(cut)


def carrier_shelf():
    """
    Shelf for both carriers, standing up as today, CELF facing out (+x).
    Sits on the boom's top face (3x M5 into the top slot) and on three
    gussets whose tabs bolt to the boom's outer face. Print upside down.
    """
    y0, y1 = SHELF_Y
    z0, t = SHELF_Z0, SHELF_T
    plate = rbox((BOOM_X0, y0, z0), (SHELF_X1, y1, z0 + t), 5)
    parts = [plate]                     # flat top: the straps locate the carriers
    xo = BOOM_X0 + EXT_W                                               # boom outer face
    for yg in (y0 + 12, (y0 + y1) / 2, y1 - 12):
        web = M.extrude(CS([[(0, 0), (0, -z0 - 8), (SHELF_X1 - xo, 0)]]), 6)   # CCW
        parts.append(place(web, np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]), (xo, yg - 3, z0)))
        parts.append(box((xo, yg - 9, -z0 + 4), (xo + 6, yg + 9, z0)))   # tab on the boom face
    body = union(parts)
    cut = []
    for ys in (y0 + 35, (y0 + y1) / 2 + 20, y1 - 35):
        cut.append(cyl_axis(M5_CLEAR / 2, (BOOM_XC, ys, z0 + t / 2), "z", t + 2))
        cut.append(cyl_axis(5.2, (BOOM_XC, ys, z0 + t - 1.6), "z", 3.4))
    for yg in (y0 + 12, (y0 + y1) / 2, y1 - 12):
        cut.append(cyl_axis(M5_CLEAR / 2, (xo + 3, yg, 10.0), "x", 10))      # into the upper slot
    for yc in (y0 + CAR_W / 2, y1 - CAR_W / 2):                       # strap slots, both sides
        for xs in (BOOM_X0 + 4.5, SHELF_X1 - 5.5):
            cut.append(box((xs - 1.5, yc - 11, z0 - 1), (xs + 1.5, yc + 11, z0 + t + 1)))
        cut.append(box((BOOM_X0 - 1, yc - 12, z0 - 1), (SHELF_X1 + 1, yc + 12, z0 + 2)))   # strap relief
        for xv in np.linspace(BOOM_X0 + 25, SHELF_X1 - 25, 3):            # vents
            cut.append(rbox((xv - 6, yc - 30, z0 - 1), (xv + 6, yc + 30, z0 + t + 1), 5.9))
    return body - union(cut)


def octo_frame():
    """
    Both OCTO-BEE-CONC boards on the boom's outer face, one above and one below,
    ports toward the nose. M3 brass inserts in 10 mm bosses at OCTO_HOLES.
    Print flat on the back. Print octo_gauge first and lay it on a real board.
    """
    L = OCTO_L + 12.0
    H = 2 * OCTO_W + 30.0
    plate = rbox_x((0, 0, -H / 2), (CONC_PLATE_T, L, H / 2), 6)
    parts = [plate]
    cut = []
    for zc in (OCTO_W / 2 + 8, -OCTO_W / 2 - 8):
        for u, v in OCTO_HOLES:
            z = zc - OCTO_W / 2 + u      # component side out, v toward the nose => u = 0 edge down
            y = 6.0 + v
            parts.append(cyl_axis(4.5, (CONC_PLATE_T + OCTO_STANDOFF / 2, y, z), "x", OCTO_STANDOFF))
            cut.append(cyl_axis(M3_INSERT_D / 2, (CONC_PLATE_T + OCTO_STANDOFF - M3_INSERT_DEPTH / 2 + 0.01, y, z),
                                "x", M3_INSERT_DEPTH))
        for y in np.linspace(40, L - 40, 3):                       # lightening + airflow
            cut.append(rbox_x((-1, y - 22, zc - 22), (CONC_PLATE_T + 1, y + 22, zc + 22), 8))
    for y in (25.0, L / 2, L - 25.0):
        for z in (-10.0, 10.0):
            cut.append(cyl_axis(M5_CLEAR / 2, (CONC_PLATE_T / 2, y, z), "x", CONC_PLATE_T + 2))
    for y in np.linspace(20, L - 20, 5):                           # zip-tie pairs on both edges
        for z in (H / 2 - 6, -H / 2 + 3.5):
            cut.append(box((-1, y - 1.5, z - 1.25), (CONC_PLATE_T + 1, y + 1.5, z + 1.25)))
    return union(parts) - union(cut)


def octo_gauge():
    """1.2 mm outline of the OCTO-BEE PCB with its estimated holes: lay it on a board."""
    g = rbox((0, 0, 0), (OCTO_W, OCTO_L, 1.2), 2) - rbox((8, 8, -1), (OCTO_W - 8, OCTO_L - 8, 2), 2)
    pads = [M.cylinder(1.2, 5.5, 5.5, SEG, False).translate([u, v, 0]) for u, v in OCTO_HOLES]
    arms = [box((u - 1.5, v - 1.5, 0), (u + 1.5, v + 1.5, 1.2)) for u, v in OCTO_HOLES]
    struts = []
    for u, v in OCTO_HOLES:                                  # tie each pad to the nearest frame edge
        targets = [(0, v), (OCTO_W, v), (u, 0), (u, OCTO_L)]
        tx, ty = min(targets, key=lambda p: abs(p[0] - u) + abs(p[1] - v))
        struts.append(box((min(u, tx) - 1.5, min(v, ty) - 1.5, 0), (max(u, tx) + 1.5, max(v, ty) + 1.5, 1.2)))
    body = union([g] + pads + arms + struts)
    holes = [cyl_axis(1.7, (u, v, 0.6), "z", 3) for u, v in OCTO_HOLES]
    mark = box((2, OCTO_L / 2 - 10, 0.8), (4, OCTO_L / 2 + 10, 1.4))         # u = 0 edge mark
    return body - union(holes) + mark


def y_chain_arm():
    """Holds the Y-chain's moving end over the Y stage. Bolts to the carriage-plate tab."""
    leg = box((-8, -25, 55), (0, 5, 186))
    arm = box((-80, -25, 178), (0, 5, 186))
    gus = place(M.extrude(CS([[(0, 0), (0, -40), (40, 0)]]), 6), np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
                (-8, -25, 178))
    body = union([leg, arm, gus])
    cut = [cyl_axis(M5_CLEAR / 2, (-4, -10, z), "x", 12) for z in (72.0, 90.0)]
    for x in (-58.0, -34.0):
        cut.append(cyl_axis(2.2, (x, -10, 182), "z", 10))
    return body - union(cut)


# MEASURE: the LTS300C end section the cap slips over (schematic stage is 64 x 100)
STAGE_END_W, STAGE_END_D = 64.0, 100.0
DRUM_R = 25.0


def spring_cap():
    """
    Cap over the top end of the vertical (Z) stage carrying two constant-force
    spring drums on a brass axle. The bands hang down to the Y bracket and take
    most of the Z payload off the leadscrew. Two M5 pinch screws, no holes in
    the stage. Print upside down (open end up).
    """
    w, d, wall, depth = STAGE_END_W + 0.6, STAGE_END_D + 0.6, 5.0, 30.0
    outer = rbox((-w / 2 - wall, -d / 2 - wall, 0), (w / 2 + wall, d / 2 + wall, depth + 8), 4)
    pocket = box((-w / 2, -d / 2, -1), (w / 2, d / 2, depth))
    # three yokes for the drum axle (axle along y), two drums between them.
    # Axle offset so the band leaves the drum's +x side straight down onto
    # the middle of the Z carriage (stage centre x=-48 -> band x=-9).
    ax_x, ax_z = 39.0 - DRUM_R, depth + 8 + DRUM_R + 2
    yokes = []
    for yy in (-d / 2 - wall, -4, d / 2 + wall - 8):
        yokes.append(box((ax_x - 14, yy, depth), (ax_x + 14, yy + 8, ax_z + 10)))
    body = union([outer] + yokes) - pocket
    cut = [cyl_axis(3.2, (ax_x, 0, ax_z), "y", d + 2 * wall + 4)]          # 6 mm brass axle
    for yy in (-25, 25):
        cut.append(cyl_axis(M5_CLEAR / 2, (-w / 2 - wall / 2, yy, depth / 2), "x", wall + 4))
    return body - union(cut), (ax_x, ax_z)


def cable_clip():
    """T-slot P-clip for the round cables (Ethernet, USB, DC, encoders)."""
    tab = rbox((-7, -12, 0), (7, 8, 3.5), 3)
    ring = cyl_axis(7.5, (0, 8 + 5.5, 7.5), "x", 14) - cyl_axis(5.5, (0, 8 + 5.5, 7.5), "x", 16)
    ring = ring - box((-8, 8, -1), (8, 13.5, 7.5))
    body = tab + ring
    return body - cyl_axis(M5_CLEAR / 2, (0, -4, 2), "z", 10)


PRINTED = [
    # (id, builder, qty, material, title, print notes)
    ("nose_core", nose_core, 1, "PETG / ASA",
     "Probe nose: 16 board seats",
     "Stand on the spigot, brim 5 mm, 0.16 mm layers, 5 walls, 40 % gyroid. "
     "32 brass M2 heat-set inserts. Nylon M2x6 screws."),
    ("nose_coupler", nose_coupler, 1, "PETG / ASA",
     "Nose-to-boom coupler",
     "Stand on the sleeve end. 6 walls, 50 % infill. 1 brass M4 insert + nylon M4 thumbscrew; "
     "4x M5x10 into aluminium T-nuts."),
    ("carriage_plate", lambda: place(carriage_plate(), rot_y(90), (0, 0, PLATE_T)), 1, "PETG / PA-CF",
     "Carriage plate on the Y sled",
     "Flat on the outer face. 6 walls, 60 % infill. 8x 1/4-20 (or M6) into the sled; 8x M5 into the boom. "
     "Check CARRIAGE_ROWS_Y first."),
    ("octo_frame", lambda: place(octo_frame(), rot_y(-90), (0, 0, 0)), 1, "PETG",
     "OCTO-BEE-CONC frame (both boards)",
     "Flat on the back. 4 walls, 30 % infill. 10 brass M3 inserts. Print octo_gauge first."),
    ("octo_gauge", octo_gauge, 1, "PLA / PETG",
     "OCTO-BEE hole gauge",
     "10-minute print. Lay it on a CONC: every hole should show a board hole through it. "
     "If not, measure and fix OCTO_HOLES."),
    ("tower_foot", tower_foot, 1, "PETG",
     "2040 tower foot",
     "Flat on its base. 6 walls, 50 % infill. 4x 1/4-20 cap screws."),
    ("device_shelf", device_shelf, 1, "PETG",
     "Switch + USB hub shelf",
     "On its back plate. 4 walls, 25 % infill. Two 20 mm velcro straps."),
    ("psu_cradle", psu_cradle, 4, "PETG",
     "Supply-brick cradle (x4)",
     "Flat. Two per brick; check the 62 x 34 mm default against your bricks."),
    ("chain_end", chain_end, 4, "PETG",
     "Drag-chain end bracket (x4)",
     "On the horizontal leg. 5 walls."),
    ("carrier_shelf", lambda: place(carrier_shelf(), rot_x(180), (0, 0, SHELF_Z0 + SHELF_T)), 1,
     "PETG / PA-CF",
     "Carrier shelf on the boom",
     "Upside down, on the top face. 5 walls, 40 % infill. 3x M5 into the boom's top slot, 3x into its "
     "outer slot. Two velcro straps per carrier. MEASURE CAR_* first."),
    ("y_chain_arm", lambda: place(y_chain_arm(), rot_x(90), (0, 0, 25)), 1, "PETG / PA-CF",
     "Y-chain arm",
     "On its side (the long flat face). 5 walls, 40 % infill. 2x M5 into the carriage-plate tab."),
    ("spring_cap", lambda: spring_cap()[0], 1, "PETG / PA-CF",
     "Z counterbalance cap",
     "Open end up. 6 walls, 60 % infill. 6 mm brass axle, 2x M5 pinch screws. "
     "MEASURE STAGE_END_* first."),
    ("cable_clip", cable_clip, 10, "PETG",
     "T-slot cable clip (x10)",
     "On the tab."),
]


# ==========================================================================
# assembly
# ==========================================================================
def stage_body(length, axis, width=100.0, height=64.0):
    """LTS300C, schematic: body + carriage plate. axis 'x' or 'y' or 'z'."""
    if axis == "x":
        return box((-length / 2, -width / 2, 0), (length / 2, width / 2, height))
    if axis == "y":
        return box((-width / 2, -length / 2, 0), (width / 2, length / 2, height))
    return box((-height / 2, -width / 2, 0), (height / 2, width / 2, length))


def chain_path(F, d, n, m_off, length, R=CHAIN_R, pitch=30.0):
    """Link centres + tangents for a U-chain; same maths as the viewer."""
    F, d, n = (np.asarray(v, float) for v in (F, d, n))
    a = (length - math.pi * R + m_off) / 2.0
    b = length - math.pi * R - a
    pts = []
    s = 0.0
    while s <= length:
        if s <= a:
            p, t = F + d * s, d
        elif s <= a + math.pi * R:
            th = (s - a) / R
            c = F + d * a + n * R
            p = c - n * R * math.cos(th) + d * R * math.sin(th)
            t = d * math.cos(th) + n * math.sin(th)
        else:
            u = s - a - math.pi * R
            p, t = F + d * a + n * 2 * R - d * u, -d
        pts.append((p, t))
        s += pitch
    return pts, a, b


# group origins as functions of the rig positions (mm) -- mirrored in the viewer
def group_origin(xs=150.0, ys=150.0, zs=150.0):
    xcar = np.array([-150.0 + xs, -250.0, 78.0])
    zcar = np.array([-2.0, 0.0, 140.0 + zs])
    ycar = np.array([88.0, -50.0 + ys, 0.0])
    return {"xcar": xcar, "zcar": zcar, "ycar": ycar}


NOSE_IN_YCAR = None      # set in build_assembly


def build_assembly():
    """List of (group, name, category, material, manifold) in group-local frames."""
    A = []

    def add(group, name, cat, mat, m):
        A.append((group, name, cat, mat, m))

    # ---------------- fixed: table, X stage, power, umbilical anchor -------
    add("fixed", "breadboard", "bench", "table", box((-520, -760, -20), (520, 900, 0)))
    add("fixed", "x_stage", "stage", "stage", stage_body(540, "x").translate([0, -250, 0]))
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        add("fixed", f"leg_{sx}_{sy}", "bench", "alu",
            cyl_axis(30, (sx * 470, -250 + sy * 0 + (sy * 560 + 70), -380), "z", 720))
    # power tray: strip + bricks in cradles at the rear-left table edge
    add("fixed", "power_strip", "power", "white", box((-480, -740, 0), (-160, -690, 45)))
    for i in range(2):
        x0 = -460 + i * 75
        add("fixed", f"acq_psu_{i}", "power", "box", box((x0, -680, 4), (x0 + 62, -560, 38)))
        for yy in (-675, -595):
            add("fixed", f"cradle_{i}_{yy}", "printed", "printed",
                place(psu_cradle(), I3, (x0 + 31, yy, 0)))
    for i in range(3):
        add("fixed", f"lts_psu_{i}", "power", "box",
            box((-300 + i * 45, -735, 45), (-262 + i * 45, -695, 95)))
    add("fixed", "x_chain_fixed_end", "printed", "printed",
        place(chain_end(), rot_z(-90), (-262, -630, 0)))

    # ---------------- xcar: platform, Z stage, carriers, tower --------------
    add("xcar", "x_carriage", "stage", "stage", box((-60, -50, -14), (60, 50, 0)))
    add("xcar", "platform", "bench", "table", box((-225, -330, 0), (225, 312, 12.7)))
    add("xcar", "z_bracket", "stage", "stage", box((-90, -60, 12.7), (-6, 60, 25)))
    add("xcar", "z_stage", "stage", "stage", stage_body(540, "z").translate([-48, 0, 25]))
    # Z counterbalance: cap on the top end of the Z stage, two constant-force
    # spring drums; the bands drop onto the Z carriage (band clip in zcar)
    cap, (ax_x, ax_z) = spring_cap()
    z_top = 25.0 + 540.0
    add("xcar", "spring_cap", "printed", "printed", cap.translate([-48, 0, z_top - 30]))
    for yy in (-30.0, 26.0):
        add("xcar", f"spring_drum_{int(yy)}", "counterbalance", "alu",
            cyl_axis(DRUM_R, (-48 + ax_x, yy + 13, z_top - 30 + ax_z), "y", 24))
    # tower: 2040 upright on the printed foot; switch + hub shelf on its far side
    tx, ty = TOWER_XY
    add("xcar", "tower_foot", "printed", "printed", place(tower_foot(), I3, (tx, ty, 12.7)))
    add("xcar", "tower_2040", "extrusion", "alu",
        box((tx - 10, ty - 20, 20.7), (tx + 10, ty + 20, 720)))
    add("xcar", "device_shelf", "printed", "printed",
        place(device_shelf(), rot_z(90), (tx - 16, ty, 470)))
    add("xcar", "gbe_switch", "network", "box", box((tx - 130, ty - 75, 475), (tx - 30, ty - 8, 502)))
    add("xcar", "usb_hub", "network", "box", box((tx - 120, ty + 8, 475), (tx - 30, ty + 70, 497)))
    add("xcar", "z_chain_fixed_end", "printed", "printed",
        place(chain_end(), rot_z(-90), (tx + 10, ty, 12.7)))
    add("xcar", "x_chain_moving_end", "printed", "printed",
        place(chain_end(), rot_z(180), (-100, -300, 12.7)))
    for k, z in enumerate((150, 300, 600)):
        add("xcar", f"tower_clip_{k}", "printed", "printed",
            place(cable_clip(), rot_z(90), (tx - 10, ty, z)))

    # ---------------- zcar: Y stage on the Z carriage, sled facing out -----
    # The Y stage is turned 90 deg about its own axis: body 64 wide (x) by
    # 100 tall (z), sled on its +x face, standing in the y-z plane.
    add("zcar", "z_carriage", "stage", "stage", box((-14, -50, -60), (0, 50, 60)))
    add("zcar", "spring_band_clip", "counterbalance", "alu", box((-13, -40, 60), (-1, 40, 66)))
    add("zcar", "y_bracket", "stage", "stage", box((0, -55, -55), (10, 55, 55)))
    add("zcar", "y_stage", "stage", "stage", box((10, -200, -50), (74, 340, 50)))
    add("zcar", "z_chain_moving_end", "printed", "printed",
        place(chain_end(), rot_z(180), (60, -206, -25)))
    add("zcar", "y_chain_fixed_end", "printed", "printed",
        place(chain_end(), np.eye(3), (42, -215, 50)))

    # ---------------- ycar: carriage plate, boom, shelf, carriers, CONCs ----
    add("ycar", "y_carriage", "stage", "stage", box((-14, -60, -50), (0, 60, 50)))
    add("ycar", "carriage_plate", "printed", "printed", carriage_plate())
    add("ycar", "y_chain_arm", "printed", "printed", y_chain_arm())
    boom_y0, boom_y1 = -150.0, -150.0 + BOOM_LEN
    add("ycar", "boom_2040", "extrusion", "alu",
        box((BOOM_X0, boom_y0, -EXT_H / 2), (BOOM_X0 + EXT_W, boom_y1, EXT_H / 2)))
    cpl = nose_coupler()
    cpl_y0 = boom_y1 - 36.0
    add("ycar", "nose_coupler", "printed", "printed", cpl.translate([BOOM_XC, cpl_y0, 0]))
    y_shoulder = cpl_y0 + 36.0 + 30.0 + COUPLER_NECK + 36.0
    R_tube = rot_x(-90)                                       # tube Z -> world +Y
    t_tube = np.array([BOOM_XC, y_shoulder, 0.0])
    global NOSE_IN_YCAR
    NOSE_IN_YCAR = (R_tube, t_tube)
    add("ycar", "nose_core", "printed", "printed2", place(nose_core(), R_tube, t_tube))
    pcb, chip, header = eval_board_parts()
    for sid in range(1, 17):
        R, root = board_frame(sid)
        Rw, tw = R_tube @ R, R_tube @ root + t_tube
        add("ycar", f"S{sid}_pcb", "head", "pcb", place(pcb, Rw, tw))
        add("ycar", f"S{sid}_chip", "head", "chip", place(chip, Rw, tw))
        add("ycar", f"S{sid}_idc", "head", "conn", place(header, Rw, tw))

    # both CONCs on one frame on the boom's outer face: 694 above, 695 below
    xo = BOOM_X0 + EXT_W
    add("ycar", "octo_frame", "printed", "printed", octo_frame().translate([xo, CONC_Y0, 0]))
    xb = xo + CONC_PLATE_T + OCTO_STANDOFF
    for side, zc in (("694", OCTO_W / 2 + 8), ("695", -OCTO_W / 2 - 8)):
        y0 = CONC_Y0 + 6.0
        zb = zc - OCTO_W / 2
        add("ycar", f"conc_{side}", "electronics", "pcb",
            box((xb, y0, zb), (xb + OCTO_T, y0 + OCTO_L, zb + OCTO_W)))
        # 50-way / 34-way analog + digital headers along the u = 0 (lower) edge
        for k, (v0, v1) in enumerate(((20, 92), (100, 172))):
            add("ycar", f"conc_{side}_idc{k}", "electronics", "conn",
                box((xb + OCTO_T, y0 + v0, zb + 3), (xb + OCTO_T + 9, y0 + v1, zb + 12)))
        # eight 20-way sensor ports in two blocks of four
        for k in range(8):
            v = 75 + (k % 4) * 13 + (k // 4) * 62
            add("ycar", f"conc_{side}_port{k}", "electronics", "conn",
                box((xb + OCTO_T, y0 + v, zb + 40), (xb + OCTO_T + 9, y0 + v + 9, zb + 73)))
    # 20-way ribbon bundles: one per face, along the core, over the neck
    for f in range(4):
        n = np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)][f], float)
        w = np.cross([0, 0, 1.0], n)
        z0, z1 = -96.0, 20.0
        strip = box((-12.7, 0, 0), (12.7, 3.0, z1 - z0))
        Rr = np.column_stack([w, n, [0, 0, 1.0]])
        org = n * (COUPLER_SIDE / 2 + 0.3) + np.array([0, 0, z0])
        add("ycar", f"ribbons_face{f}", "ribbon", "ribbon",
            place(strip, R_tube @ Rr, R_tube @ org + t_tube))
    add("ycar", "ribbons_to_conc", "ribbon", "ribbon",
        box((xo + 1, CONC_Y0 + 150, -14), (xo + 15, cpl_y0 + 60, 14)))

    # carriers standing on the shelf, CELF facing out, 694 behind 695
    add("ycar", "carrier_shelf", "printed", "printed", carrier_shelf())
    zc0 = SHELF_Z0 + SHELF_T
    ys0 = SHELF_Y[0]
    for side, ya in (("694", ys0 + 1.0), ("695", ys0 + CAR_W + 9.0)):
        add("ycar", f"carrier_{side}", "electronics", "carrier",
            box((CAR_X0, ya, zc0), (CAR_X0 + CAR_D, ya + CAR_W, zc0 + CAR_H)))
        xf = CAR_X0 + CAR_D
        add("ycar", f"celf_{side}", "electronics", "pcb",
            box((xf, ya + (CAR_W - OCTO_W) / 2, zc0 + 30), (xf + OCTO_T, ya + (CAR_W + OCTO_W) / 2, zc0 + 30 + OCTO_L)))
        for k in range(3):
            add("ycar", f"enc_term_{side}_{k}", "electronics", "terminal",
                box((xf + OCTO_T, ya + 14 + k * 28, zc0 + 30 + OCTO_L - 24),
                    (xf + 12, ya + 38 + k * 28, zc0 + 30 + OCTO_L - 6)))
        add("ycar", f"idc_{side}", "electronics", "conn",
            box((xf + OCTO_T, ya + (CAR_W - OCTO_W) / 2 - 2, zc0 + 40), (xf + 10, ya + (CAR_W - OCTO_W) / 2 + 10, zc0 + 150)))
        # CELF -> CONC: down the CELF's analog edge, under the shelf, forward to the CONC
        add("ycar", f"ribbons_celf_{side}", "ribbon", "ribbon",
            box((xf + 10, ya + 6, zc0 + 30), (xf + 13, ya + 18, zc0 + 120)))
    add("ycar", "ribbons_under_shelf", "ribbon", "ribbon",
        box((xo + 30, SHELF_Y[0] + 20, -16), (xo + 70, CONC_Y0 + 20, -12)))
    for k, yy in enumerate((110.0,)):
        add("ycar", f"boom_clip_{k}", "printed", "printed",
            place(cable_clip(), np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ rot_z(90),
                  (BOOM_XC, yy, EXT_H / 2)))
    return A

def coil_meshes(center):
    """designA filaments as tubes, centred on `center` (pose is not measured)."""
    try:
        sys.path.insert(0, os.path.join(REPO, "src"))
        from octobee import machine as mach
        cfg = json.load(open(os.path.join(REPO, "config", "machine.json")))
        cls = [getattr(mach, n) for n in dir(mach)
               if hasattr(getattr(mach, n), "load_or_none")][0]
        cs = cls.load(cfg["coil_file"], n_points=160)
    except Exception as exc:                                  # file is on OneDrive
        print("coil file not loaded:", exc)
        return []
    out = []
    for c in cs.coils:
        pts = c.points_mm + center
        segs = [cyl_between(pts[i], pts[i + 1], 18.0, 10) for i in range(len(pts) - 1)]
        out.append((c.label, union(segs)))
    return out


# ==========================================================================
# checks
# ==========================================================================
def check_fsv():
    """Every printed seat must put its FSV where geometry.py says it is."""
    worst = 0.0
    fsv_u = GEOM.arm_length_mm - GEOM.fsv_from_tip_mm
    for sid in range(1, 17):
        R, root = board_frame(sid)
        p = root + R @ np.array([fsv_u, 0.0, GEOM.fsv_height_mm])
        worst = max(worst, float(np.linalg.norm(p - GEOM.position(sid))))
    assert worst < 1e-6, f"FSV mismatch {worst} mm"
    return worst


def check_chains():
    """Every chain must close (|M-F| along n = 2R) and keep both runs >= 0 over full travel."""
    report = []
    for c in CHAINS:
        d, n = np.array(c["d"], float), np.array(c["n"], float)
        lo_run, hi_ext = 1e9, -1e9
        for xs in (0, 300):
            for ys in (0, 300):
                for zs in (0, 300):
                    o = group_origin(xs, ys, zs)
                    w = {"fixed": np.zeros(3), "xcar": o["xcar"]}
                    w["zcar"] = w["xcar"] + o["zcar"]
                    w["ycar"] = w["zcar"] + o["ycar"]
                    F = w[c["gF"]] + c["F"]
                    Mp = w[c["gM"]] + c["M"]
                    off = float((Mp - F) @ n)
                    assert abs(off - 2 * c["R"]) < 1e-6, (c["id"], off)
                    m = float((Mp - F) @ d)
                    a = (c["L"] - math.pi * c["R"] + m) / 2
                    b = c["L"] - math.pi * c["R"] - a
                    lo_run = min(lo_run, a, b)
                    hi_ext = max(hi_ext, float(F @ d) + a + c["R"])
        assert lo_run >= 0, (c["id"], lo_run)
        report.append(f"{c['id']}: shortest run {lo_run:.0f} mm, bend reaches {hi_ext:.0f} along d")
    return report


def _world_aabb(parts, group, origins):
    lo, hi = np.full(3, 1e9), np.full(3, -1e9)
    off = {"fixed": np.zeros(3), "xcar": origins["xcar"]}
    off["zcar"] = off["xcar"] + origins["zcar"]
    off["ycar"] = off["zcar"] + origins["ycar"]
    for m in parts:
        b = np.array(m.bounding_box())
        lo, hi = np.minimum(lo, b[:3] + off[group]), np.maximum(hi, b[3:] + off[group])
    return lo, hi


def check_boom_electronics(assembly):
    """
    The deck and carriers ride the Y carriage. Over the whole cube they must
    stay clear of the Z stage, the spring cap, the tower and both small chains.
    Returns the smallest gap found (mm, AABB vs AABB / link spheres).
    """
    by = {(g, n): m for g, n, _c, _m, m in assembly}
    riders = [k for k in by if k[0] == "ycar" and k[1].startswith(
        ("carrier_", "celf_", "idc_", "enc_term_", "ribbons_celf", "ribbons_under", "conc_",
         "octo_frame", "carriage_plate", "boom_2040", "y_chain_arm"))]
    obstacles = [k for k in by if (k[0] == "xcar" and
                 (k[1] in ("z_stage", "spring_cap", "tower_2040", "device_shelf", "gbe_switch", "usb_hub")
                  or k[1].startswith("spring_drum")))
                 or (k[0] == "zcar" and k[1] in ("y_stage", "y_chain_fixed_end", "z_chain_moving_end"))]
    worst = (1e9, None)
    steps = (0, 100, 200, 300)
    for xs in steps:
        for ys in steps:
            for zs in steps:
                o = group_origin(xs, ys, zs)
                boxes = {k: _world_aabb([by[k]], k[0], o) for k in riders + obstacles}
                for r in riders:
                    rl, rh = boxes[r]
                    for ob in obstacles:
                        ol, oh = boxes[ob]
                        gap = max(np.max(ol - rh), np.max(rl - oh))
                        if gap < worst[0]:
                            worst = (gap, (r[1], ob[1], (xs, ys, zs)))
                    # chain links as boxes: w across (d x n), h across the run, 10 mm along it
                    w = {"fixed": np.zeros(3), "xcar": o["xcar"]}
                    w["zcar"] = w["xcar"] + o["zcar"]
                    w["ycar"] = w["zcar"] + o["ycar"]
                    for c in CHAINS:
                        if c["id"] == "x_umbilical" or (c["id"] == "y_chain" and r[1] == "y_chain_arm"):
                            continue
                        d, n = np.array(c["d"], float), np.array(c["n"], float)
                        e_w = np.cross(d, n)
                        F = w[c["gF"]] + c["F"]
                        Mp = w[c["gM"]] + c["M"]
                        m_off = float((Mp - F) @ d)
                        pts, _a, _b = chain_path(F, d, n, m_off, c["L"], pitch=10.0)
                        for p, t in pts:
                            e_h = np.cross(t, e_w)
                            half = (np.abs(e_w) * c["w"] / 2 + np.abs(e_h) * c["h"] / 2
                                    + np.abs(t) * 5.0)
                            gap = max(np.max((p - half) - rh), np.max(rl - (p + half)))
                            if gap < worst[0]:
                                worst = (gap, (r[1], c["id"], (xs, ys, zs)))
    return worst


def mass_budget(catalogue):
    """What the Z leadscrew lifts, before and after the carriers move on board."""
    g = {p["id"]: p["est_g_each"] * p["qty"] / 1000.0 for p in catalogue}
    boom_kg = BOOM_LEN / 1000.0 * 0.75                        # 2040, ~0.75 kg/m
    items = [
        ("Y stage (LTS300C, Thorlabs catalog)", LTS300_KG),
        ("Y bracket + fasteners (estimate)", 0.35),
        ("2040 boom", round(boom_kg, 2)),
        ("printed: carriage plate, coupler, nose, CONC frame, chain arm", round(sum(g[k] for k in
         ("carriage_plate", "nose_coupler", "nose_core", "octo_frame", "y_chain_arm")), 2)),
        ("2x CONC + 16 eval kits + ribbons (estimate)", 0.45),
        ("Y-chain cables, half their weight (estimate)", 0.25),
    ]
    before = sum(v for _k, v in items)
    after_items = items + [
        (f"2x ACQ1001 + CELF in brackets (MEASURE, {CARRIER_KG} kg each)", 2 * CARRIER_KG),
        ("carrier shelf (printed)", round(g["carrier_shelf"], 2)),
    ]
    after = sum(v for _k, v in after_items)
    springs_n = (after - Z_NET_DOWN_KG) * 9.81
    return {"items": [{"what": k, "kg": round(v, 2)} for k, v in after_items],
            "z_load_before_kg": round(before, 2), "z_load_after_kg": round(after, 2),
            "z_rated_kg": Z_RATED_KG, "net_down_kg": Z_NET_DOWN_KG,
            "spring_total_n": round(springs_n), "spring_each_n": round(springs_n / 2)}


def check_clearances(core):
    pcb = eval_board(with_holes=False)
    boards = []
    for sid in range(1, 17):
        R, root = board_frame(sid)
        boards.append(place(pcb, R, root + R[:, 2] * 0.02))
    worst = []
    for i in range(16):
        for j in range(i + 1, 16):
            if (boards[i] ^ boards[j]).volume() > 1e-6:
                worst.append((i + 1, j + 1))
    hit = [i + 1 for i, b in enumerate(boards) if (b ^ core).volume() > 1e-6]
    return worst, hit


# ==========================================================================
# export
# ==========================================================================
def to_trimesh(m):
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3]
    f = np.asarray(mesh.tri_verts)
    return trimesh.Trimesh(v, f, process=False)


def material(name):
    rgba, metal, rough = MATERIALS[name]
    return trimesh.visual.material.PBRMaterial(
        name=name, baseColorFactor=list(rgba), metallicFactor=metal,
        roughnessFactor=rough, alphaMode="BLEND" if rgba[3] < 255 else "OPAQUE",
        doubleSided=rgba[3] < 255)


def main():
    os.makedirs(STL_DIR, exist_ok=True)
    print("FSV check: max error", check_fsv(), "mm")
    for line in check_chains():
        print("chain", line)

    catalogue = []
    built = {}
    for pid, fn, qty, mat, title, notes in PRINTED:
        m = fn()
        built[pid] = m
        tm = to_trimesh(m)
        tm.export(os.path.join(STL_DIR, f"{pid}.stl"))
        bb = m.bounding_box()
        size = [round(bb[3] - bb[0], 1), round(bb[4] - bb[1], 1), round(bb[5] - bb[2], 1)]
        grams = m.volume() / 1000.0 * 1.27 * 0.55              # PETG, ~55 % solid
        catalogue.append({"id": pid, "title": title, "qty": qty, "material": mat,
                          "notes": notes, "size_mm": size,
                          "est_g_each": round(grams), "watertight": bool(tm.is_watertight),
                          "genus": m.genus()})
        print(f"{pid:14s} {size} mm  ~{round(grams)} g  watertight={tm.is_watertight}")

    overlaps, hit_core = check_clearances(built["nose_core"])
    print("board-board overlaps:", overlaps or "none", "| boards cutting the core:", hit_core or "none")

    assembly = build_assembly()
    gap, where = check_boom_electronics(assembly)
    print(f"boom electronics: smallest gap {gap:.1f} mm ({where})")
    assert gap > 3.0, ("boom electronics collide", gap, where)
    budget = mass_budget(catalogue)
    print(f"Z load: {budget['z_load_before_kg']} kg before, {budget['z_load_after_kg']} kg with "
          f"carriers on the boom (rated {Z_RATED_KG}); springs {budget['spring_total_n']} N total")

    scene = trimesh.Scene()
    nodes = []
    for group, name, cat, mat, m in assembly:
        if m.is_empty():
            continue
        tm = to_trimesh(m)
        tm.visual = trimesh.visual.TextureVisuals(material=material(mat))
        node = f"{group}|{name}|{cat}"
        scene.add_geometry(tm, node_name=node, geom_name=node)
        nodes.append(node)

    o = group_origin()
    R_t, t_t = NOSE_IN_YCAR
    mid = R_t @ np.array([0, 0, 80.0]) + t_t
    vol_lo = o["xcar"] + o["zcar"] + o["ycar"] - np.array([150, 150, 150]) + mid
    vol_c = vol_lo + 150
    for label, m in coil_meshes(vol_c):
        tm = to_trimesh(m)
        tm.visual = trimesh.visual.TextureVisuals(material=material("coil"))
        node = f"fixed|coil_{label}|coil"
        scene.add_geometry(tm, node_name=node, geom_name=node)
    vol = box(vol_lo, vol_lo + 300)
    tm = to_trimesh(vol)
    tm.visual = trimesh.visual.TextureVisuals(material=material("volume"))
    scene.add_geometry(tm, node_name="fixed|scan_volume|volume", geom_name="fixed|scan_volume|volume")

    glb = os.path.join(OUT, "rig_v2.glb")
    scene.export(glb)
    meta = {"printed": catalogue, "nose_in_ycar": {"R": R_t.tolist(), "t": t_t.tolist()},
            "fsv_world_mid": mid.tolist(), "scan_volume_lo": vol_lo.tolist(),
            "chain": {"R": CHAIN_R, "inner": [CHAIN_INNER_W, CHAIN_INNER_H]},
            "chains": CHAINS,
            "z_budget": budget,
            "boom_electronics_gap_mm": round(gap, 1),
            "fsv_radius_mm": GEOM.fsv_radius_mm,
            "sensors": [{"id": i, "face": GEOM.sensors[i - 1]["face"],
                         "slot": GEOM.sensors[i - 1]["slot"],
                         "fsv_tube_mm": [round(float(v), 2) for v in GEOM.position(i)]}
                        for i in range(1, 17)]}
    with open(os.path.join(OUT, "parts.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", glb, os.path.getsize(glb) // 1024, "kB")


if __name__ == "__main__":
    main()
