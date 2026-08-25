#!/usr/bin/env python3
"""
octobee/gui/widgets/machine3d.py -- the probe drawn where it actually is, inside the coils.

The probe view next to the live plot answers "which chip is reading what". This
one answers the other question, the one that decides whether a field map means
anything: where is the head, relative to the windings, and which of those
windings is carrying current.

Everything is in machine millimetres -- the coil file's own frame, scaled from
metres -- so the numbers typed into the placement boxes are the numbers drawn.
The probe's own geometry arrives in the mount frame and is carried into the
machine by a single transform on each item, rather than by rebuilding its
meshes: the stage moves ten times a second while it is jogging, and re-uploading
a few thousand vertices at that rate would make the view the slowest thing in
the window.

Colour is the whole legend:

    amber        a coil with current in it
    slate        a coil that is switched off -- still solid, still in the way
    green line   the closest approach between probe and winding
    red line     the same, when the probe is inside a coil
"""

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6 import QtGui

from octobee import machine as omach
from octobee import profile as oprof
from octobee.calib import geometry as pgeom

COIL_ON_COLOR = (0.95, 0.62, 0.18, 1.0)
COIL_OFF_COLOR = (0.58, 0.63, 0.72, 1.0)
COIL_ON_EDGE = (1.0, 0.80, 0.45, 0.85)
COIL_OFF_EDGE = (0.45, 0.50, 0.60, 0.8)
TUBE_COLOR = (0.62, 0.66, 0.72, 0.55)
ARM_COLOR = (0.55, 0.36, 0.20, 1.0)
CHIP_COLOR = (0.30, 0.75, 0.95, 1.0)
REACH_COLOR = (0.35, 0.75, 0.45, 0.55)
CLEAR_COLOR = (0.35, 0.95, 0.45, 1.0)
HIT_COLOR = (1.0, 0.25, 0.25, 1.0)

# Facets around a coil. Twelve is enough that a 20 mm winding reads as round at
# machine scale, and keeps six coils under 40k triangles.
COIL_SIDES = 12


class MachineView3D(gl.GLViewWidget):
    """Coils, keep-out volume, stage envelope and the probe inside them."""

    def __init__(self, geom=None, parent=None, profiler=None):
        super().__init__(parent)
        self.profiler = profiler or oprof.Profiler(enabled=False)
        self.geom = geom or pgeom.Geometry()
        self.coils = None
        self.radius_mm = omach.DEFAULT_COIL_RADIUS_MM
        self.energised = set()
        self.show_reach = True
        self.show_labels = True
        self._coil_items = {}           # label -> (mesh, centreline, text)
        self._probe_items = []
        self._reach_items = []
        self._static = []
        self._clear_line = None
        self._hit = None                # the coil the probe is currently inside
        self.setBackgroundColor(pg.mkColor(16, 18, 24))
        self._build()

    def paintGL(self, *args, **kwargs):
        with self.profiler.time("GL paint (machine)"):
            super().paintGL(*args, **kwargs)

    # ---- construction ----------------------------------------------------
    def _build(self):
        self._add_probe()
        self._add_axes()
        self._clear_line = gl.GLLinePlotItem(pos=np.zeros((2, 3)), width=2.5,
                                             antialias=True,
                                             color=(0, 0, 0, 0))
        self.addItem(self._clear_line)
        self.reset_camera()

    def _add_axes(self):
        """The machine's own axes at its origin, so the drawing can be read.

        Length is fixed at a quarter of a metre rather than scaled to the
        machine: it doubles as a ruler.
        """
        L = 250.0
        for vec, col, name in ((np.array([L, 0, 0]), (1, .35, .35, 1), "X"),
                               (np.array([0, L, 0]), (.35, 1, .35, 1), "Y"),
                               (np.array([0, 0, L]), (.45, .6, 1, 1), "Z")):
            line = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], vec]), color=col,
                                     width=2.0, antialias=True)
            self.addItem(line)
            self._static.append(line)
            txt = gl.GLTextItem(pos=vec * 1.1, text=name,
                                color=QtGui.QColor.fromRgbF(*col))
            self.addItem(txt)
            self._static.append(txt)

    def _add_probe(self):
        """The probe body, once, in the mount frame. A transform moves it."""
        for item in self._probe_items:
            self.removeItem(item)
        self._probe_items = []
        g = self.geom

        verts, faces = pgeom.tube_mesh(g)
        self._probe_items.append(gl.GLMeshItem(
            meshdata=gl.MeshData(vertexes=pgeom.to_world(verts).astype(np.float32),
                                 faces=faces),
            smooth=False, drawEdges=True, edgeColor=(0.7, 0.75, 0.85, 0.9),
            color=TUBE_COLOR, shader="balloon", glOptions="translucent"))

        for sid in range(1, pgeom.N_SENSORS + 1):
            averts, afaces = pgeom.arm_mesh(g, sid)
            self._probe_items.append(gl.GLMeshItem(
                meshdata=gl.MeshData(
                    vertexes=pgeom.to_world(averts).astype(np.float32),
                    faces=afaces),
                smooth=False, drawEdges=False, color=ARM_COLOR,
                shader="shaded", glOptions="opaque"))
            cverts, cfaces = pgeom.chip_mesh(g, sid)
            self._probe_items.append(gl.GLMeshItem(
                meshdata=gl.MeshData(
                    vertexes=pgeom.to_world(cverts).astype(np.float32),
                    faces=cfaces),
                smooth=False, drawEdges=False, color=CHIP_COLOR,
                shader="shaded", glOptions="opaque"))

        for item in self._probe_items:
            self.addItem(item)

    def set_geometry(self, geom):
        self.geom = geom
        self._add_probe()

    # ---- coils -----------------------------------------------------------
    def set_coils(self, coilset, radius_mm=None, energised=None):
        """Draw a coil set from scratch. Cheap enough to call on any change."""
        with self.profiler.time("build coil meshes"):
            for items in self._coil_items.values():
                for item in items:
                    if item is not None:
                        self.removeItem(item)
            self._coil_items = {}
            for item in self._static[:]:
                if isinstance(item, gl.GLGridItem):
                    self.removeItem(item)
                    self._static.remove(item)

            self.coils = coilset
            if radius_mm is not None:
                self.radius_mm = float(radius_mm)
            if energised is not None:
                self.energised = set(energised)
            if coilset is None or not len(coilset):
                return

            for coil in coilset:
                verts, faces = omach.tube_mesh(coil.points_mm, self.radius_mm,
                                               sides=COIL_SIDES)
                mesh = gl.GLMeshItem(
                    meshdata=gl.MeshData(vertexes=verts, faces=faces),
                    smooth=True, drawEdges=False, shader="shaded",
                    glOptions="opaque")
                line = gl.GLLinePlotItem(pos=coil.points_mm.astype(np.float32),
                                         width=1.5, antialias=True)
                txt = gl.GLTextItem(pos=coil.label_anchor_mm, text=coil.label,
                                    color=(215, 220, 235, 255))
                for item in (mesh, line, txt):
                    self.addItem(item)
                self._coil_items[coil.label] = (mesh, line, txt)
            self._add_floor()
            self._apply_coil_colors()

    def _add_floor(self):
        """A grid under the machine, on a 100 mm pitch, as a sense of scale."""
        lo, hi = self.coils.bounds_mm()
        span = float(np.max(hi[:2] - lo[:2])) * 1.25
        grid = gl.GLGridItem()
        grid.setSize(span, span)
        grid.setSpacing(100.0, 100.0)
        grid.translate(0.0, 0.0, lo[2] - 150.0)
        grid.setColor((60, 66, 82, 100))
        self.addItem(grid)
        self._static.append(grid)

    def set_radius(self, radius_mm):
        if abs(float(radius_mm) - self.radius_mm) < 1e-9:
            return
        self.radius_mm = float(radius_mm)
        self.set_coils(self.coils)

    def set_energised(self, labels):
        self.energised = set(labels or ())
        self._apply_coil_colors()

    def _apply_coil_colors(self):
        for label, (mesh, line, _txt) in self._coil_items.items():
            if label == self._hit:
                mesh.setColor(HIT_COLOR)
                line.setData(color=HIT_COLOR, width=3.0)
                continue
            on = label in self.energised
            mesh.setColor(COIL_ON_COLOR if on else COIL_OFF_COLOR)
            line.setData(color=COIL_ON_EDGE if on else COIL_OFF_EDGE,
                         width=2.5 if on else 1.0)

    def set_labels_visible(self, on):
        self.show_labels = bool(on)
        for label, (_m, _l, txt) in self._coil_items.items():
            txt.setData(text=label if on else "")

    # ---- placement -------------------------------------------------------
    def set_pose(self, placement, stage_mm=None):
        """Move the probe to a Placement, with the stages where they are."""
        rot = placement.rotation()
        origin = placement.origin_mm(stage_mm)
        tr = pg.Transform3D()
        tr.setRow(0, QtGui.QVector4D(rot[0, 0], rot[0, 1], rot[0, 2], origin[0]))
        tr.setRow(1, QtGui.QVector4D(rot[1, 0], rot[1, 1], rot[1, 2], origin[1]))
        tr.setRow(2, QtGui.QVector4D(rot[2, 0], rot[2, 1], rot[2, 2], origin[2]))
        tr.setRow(3, QtGui.QVector4D(0.0, 0.0, 0.0, 1.0))
        for item in self._probe_items:
            item.setTransform(tr)
        self._draw_reach(placement)

    def _draw_reach(self, placement):
        """The box the flange can be driven through, as twelve edges.

        Drawn from the stage travel rather than the current position, because
        the question it answers is asked before the move: can this pose reach
        the region worth measuring at all, without the head meeting a coil on
        the way.
        """
        for item in self._reach_items:
            self.removeItem(item)
        self._reach_items = []
        if not self.show_reach:
            return
        c = placement.reach_corners_mm()
        # Corner i has bits (x, y, z) in that order, so an edge joins corners
        # differing in exactly one bit.
        for i in range(8):
            for bit in (4, 2, 1):
                j = i ^ bit
                if j < i:
                    continue
                line = gl.GLLinePlotItem(pos=np.array([c[i], c[j]]),
                                         color=REACH_COLOR, width=1.2,
                                         antialias=True)
                self.addItem(line)
                self._reach_items.append(line)

    def set_reach_visible(self, on, placement=None):
        self.show_reach = bool(on)
        if placement is not None:
            self._draw_reach(placement)
        elif not on:
            for item in self._reach_items:
                self.removeItem(item)
            self._reach_items = []

    def set_clearance(self, gap):
        """Draw the closest approach found by octobee_machine.clearance.

        A collision also turns the coil itself red. The contact line alone is
        not enough: when the probe is inside a winding the line is inside it
        too, so the one drawing that most needs to be alarming is the one the
        depth buffer hides.
        """
        if gap is None or gap.coil is None or gap.probe_point is None:
            self._clear_line.setData(pos=np.zeros((2, 3)), color=(0, 0, 0, 0))
            hit = None
        else:
            self._clear_line.setData(
                pos=np.array([gap.probe_point, gap.coil_point]),
                color=HIT_COLOR if gap.collides else CLEAR_COLOR,
                width=3.0 if gap.collides else 2.0)
            hit = gap.coil if gap.collides else None
        if hit != self._hit:
            self._hit = hit
            self._apply_coil_colors()

    # ---- camera ----------------------------------------------------------
    def reset_camera(self):
        """Frame the whole machine, looking down on it from outside."""
        if self.coils is None or not len(self.coils):
            self.opts["center"] = pg.Vector(0, 0, 0)
            self.setCameraPosition(distance=1200, elevation=25, azimuth=45)
            return
        lo, hi = self.coils.bounds_mm()
        centre = pg.Vector(*(0.5 * (lo + hi)))
        self.opts["center"] = centre
        self.setCameraPosition(pos=centre,
                               distance=float(np.linalg.norm(hi - lo)) * 1.15,
                               elevation=28, azimuth=35)

    def look_at_probe(self, placement, stage_mm=None):
        """Zoom to the head: at machine scale it is otherwise a speck."""
        origin = placement.origin_mm(stage_mm)
        centre = origin + placement.rotation() @ np.array(
            [0.0, self.geom.tube_length_mm / 2.0, 0.0])
        self.opts["center"] = pg.Vector(*centre)
        self.setCameraPosition(pos=pg.Vector(*centre),
                               distance=max(4.0 * self.geom.fsv_radius_mm,
                                            500.0),
                               elevation=22, azimuth=35)
