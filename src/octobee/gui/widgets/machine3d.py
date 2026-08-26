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

The probe's zero point -- its mounting flange, the origin the boards are
measured from -- carries a drag handle: a ball with three arrows along the
machine axes. Dragging an arrow slides the probe along that axis and writes the
number straight into the placement box, because reading a clearance off the
drawing and then guessing which of six boxes to nudge is how a pose gets typed
in wrong. The handle is sized in pixels rather than millimetres, so it stays
the same size on screen whether the camera is framing a three-metre coil set or
sitting on the head itself.
"""

import math

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL.GL import (GL_BLEND, GL_CULL_FACE, GL_DEPTH_TEST,
                       GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA)
from PyQt6 import QtCore, QtGui

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

# Facets around a coil. Twelve is enough that a 60 mm winding reads as round at
# machine scale, and keeps six coils under 40k triangles.
COIL_SIDES = 12

# The drag handle at the probe's zero point. Red/green/blue for x/y/z, matching
# the machine's own axes at its origin, because the arrow and the box it writes
# into have to be recognisably the same axis.
GIZMO_AXIS_COLOR = ((1.00, 0.38, 0.38, 1.0),
                    (0.38, 1.00, 0.45, 1.0),
                    (0.48, 0.62, 1.00, 1.0))
GIZMO_HOT_COLOR = (1.00, 0.92, 0.35, 1.0)
GIZMO_BALL_COLOR = (0.93, 0.95, 1.00, 1.0)

# Sized on screen, not in the machine: a handle measured in millimetres is
# either invisible when the whole coil set is in frame or fills the window when
# it is not, and it has to be grabbable at both.
GIZMO_SHAFT_PX = 96.0
GIZMO_HEAD_PX = 26.0
GIZMO_BALL_PX = 7.0
GIZMO_GRAB_PX = 12.0

# Depth testing off, drawn last: the handle is a control, and a control that
# disappears when the head is behind a winding is worse than useless -- that is
# exactly the moment somebody wants to drag it back out.
GIZMO_GL = {GL_DEPTH_TEST: False, GL_BLEND: True, GL_CULL_FACE: False,
            "glBlendFunc": (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)}

# +Z onto each machine axis, as (angle, x, y, z) for Transform3D.rotate.
GIZMO_ROTATION = ((90.0, 0.0, 1.0, 0.0), (-90.0, 1.0, 0.0, 0.0),
                  (0.0, 0.0, 0.0, 1.0))
GIZMO_LABEL = ("x", "y", "z")

# The rotation handle: a ring in the machine's XY plane about the zero point,
# which is the plane the only rotation this rig has turns in. Blue, because
# that is the Z axis's colour and Z is what it turns about.
# Clear of the arrows: their tips reach GIZMO_SHAFT_PX + GIZMO_HEAD_PX
# and grab within GIZMO_GRAB_PX of that, and an arrow wins a click the
# ring also wants. Overlapping bands would make the handle nearest +x
# and +y slide the probe when it looks like it should turn it.
GIZMO_RING_PX = 165.0
GIZMO_RING_SEGMENTS = 72
GIZMO_RING_COLOR = (0.48, 0.62, 1.00, 0.85)

# The volume to be mapped, and the path through it. Cyan for the box because
# nothing else in the scene is; the path is drawn in two colours so that what
# was carved away by the coils is visible as absence AND as the thing it was
# carved out of.
VOLUME_COLOR = (0.35, 0.85, 0.95, 0.75)
PATH_COLOR = (0.45, 0.95, 0.60, 0.9)
PATH_DROPPED_COLOR = (0.55, 0.30, 0.34, 0.55)


def _cone_mesh(sides=16):
    """A unit cone: base ring of radius 1 at z=0, tip at z=1."""
    th = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = np.stack([np.cos(th), np.sin(th), np.zeros(sides)], axis=1)
    verts = np.vstack([ring, [[0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0]]])
    tip, base = sides, sides + 1
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces.append([i, j, tip])
        faces.append([j, i, base])
    return verts.astype(np.float32), np.array(faces, dtype=np.int32)


def _segment_distance_px(p, a, b):
    """Distance from screen point p to the segment a-b, in pixels."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-9:
        return float(np.hypot(*(p - a)))
    t = min(1.0, max(0.0, float((p - a) @ ab) / denom))
    return float(np.hypot(*(p - (a + t * ab))))


class MachineView3D(gl.GLViewWidget):
    """Coils, keep-out volume, stage envelope and the probe inside them."""

    #: The flange was dragged to these machine millimetres. Carries the whole
    #: position rather than a delta, so a dropped signal cannot accumulate.
    pose_dragged = QtCore.pyqtSignal(float, float, float)

    #: The ring was dragged: the assembly is now turned this many degrees about
    #: the machine's Z. Absolute, for the same reason.
    pose_turned = QtCore.pyqtSignal(float)

    def __init__(self, geom=None, parent=None, profiler=None):
        super().__init__(parent)
        self.profiler = profiler or oprof.Profiler(enabled=False)
        self.geom = geom or pgeom.Geometry()
        self.coils = None
        self.radius_mm = omach.DEFAULT_COIL_RADIUS_MM
        self.energised = set()
        self.show_reach = True
        self.show_labels = True
        self.show_gizmo = True
        self._coil_items = {}           # label -> (mesh, centreline, text)
        self._probe_items = []
        self._reach_items = []
        self._static = []
        self._clear_line = None
        self._hit = None                # the coil the probe is currently inside
        self._placement = None          # the Placement last drawn
        self._gizmo_items = []          # ball, then (shaft, head, label) x 3
        self._gizmo_origin = None       # the zero point, machine mm
        self._gizmo_len_mm = 1.0        # ball to arrow tip, machine mm
        self._gizmo_hot = None          # the axis under the mouse, or dragged
        self._ring = None               # the rotation handle
        self._ring_hot = False
        self._ring_mm = 1.0             # its radius, machine mm
        self._drag = None               # (axis, grab px, step px, len, base)
        self._turn = None               # (grabbed angle, angle at grab)
        self._volume_item = None        # the box to be mapped
        self._path_item = None          # the lines that will be swept
        self._dropped_item = None       # the ones the coils took away
        self.setBackgroundColor(pg.mkColor(16, 18, 24))
        self.setMouseTracking(True)     # so an arrow can light up under the pointer
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
        self._add_volume()
        self._add_gizmo()
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

    # ---- the volume to be mapped, and the path through it ----------------
    def _add_volume(self):
        """The box, the lines that will be swept, and the ones that were not.

        All three are one line item apiece, drawn in `lines` mode: a sweep of a
        300 mm cube is nearly a thousand segments, and a thousand GL items is
        a thousand draw calls per frame for a picture that never changes
        between plans.
        """
        self._volume_item = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), color=VOLUME_COLOR, width=1.6,
            antialias=True, mode="lines")
        self._path_item = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), color=PATH_COLOR, width=1.8,
            antialias=True, mode="lines")
        self._dropped_item = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), color=PATH_DROPPED_COLOR, width=1.0,
            antialias=True, mode="lines")
        for item in (self._volume_item, self._dropped_item, self._path_item):
            item.setVisible(False)
            self.addItem(item)

    def set_volume(self, corners_mm):
        """Draw the box to be mapped, from its eight corners in machine mm.

        Corner i has bits (x, y, z) in that order, as Volume.corners_mm builds
        them, so an edge joins corners differing in exactly one bit.
        """
        if corners_mm is None:
            self._volume_item.setVisible(False)
            return
        c = np.asarray(corners_mm, dtype=float).reshape(8, 3)
        segs = [c[[i, i ^ bit]] for i in range(8) for bit in (4, 2, 1)
                if (i ^ bit) > i]
        self._volume_item.setData(pos=np.vstack(segs))
        self._volume_item.setVisible(True)

    def set_path(self, swept_mm, dropped_mm=None):
        """Draw the planned sweep. Each is (2N, 3): pairs of segment ends."""
        for item, pts in ((self._path_item, swept_mm),
                          (self._dropped_item, dropped_mm)):
            if pts is None or not len(pts):
                item.setVisible(False)
                continue
            item.setData(pos=np.asarray(pts, dtype=float).reshape(-1, 3))
            item.setVisible(True)

    def clear_plan(self):
        for item in (self._volume_item, self._path_item, self._dropped_item):
            if item is not None:
                item.setVisible(False)

    # ---- the zero point, and its drag handle -----------------------------
    def _add_gizmo(self):
        """The handle at the probe's zero point: a ball and three arrows.

        Built once at unit size and moved by a transform, like the probe
        itself: it is re-sized on every camera change to stay a constant number
        of pixels across, and rebuilding four meshes at that rate for the sake
        of a handle would be absurd.
        """
        ball = gl.GLMeshItem(meshdata=gl.MeshData.sphere(rows=10, cols=18,
                                                         radius=1.0),
                             smooth=True, drawEdges=False, shader="shaded",
                             color=GIZMO_BALL_COLOR)
        self._gizmo_items = [ball]
        verts, faces = _cone_mesh()
        for axis in range(3):
            colour = GIZMO_AXIS_COLOR[axis]
            shaft = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=colour,
                                      width=3.0, antialias=True)
            head = gl.GLMeshItem(meshdata=gl.MeshData(vertexes=verts,
                                                      faces=faces),
                                 smooth=False, drawEdges=False,
                                 shader="shaded", color=colour)
            label = gl.GLTextItem(pos=np.zeros(3), text=GIZMO_LABEL[axis],
                                  color=QtGui.QColor.fromRgbF(*colour))
            self._gizmo_items += [shaft, head, label]
        self._ring = gl.GLLinePlotItem(pos=np.zeros((2, 3)),
                                       color=GIZMO_RING_COLOR, width=2.5,
                                       antialias=True)
        self._gizmo_items.append(self._ring)
        for item in self._gizmo_items:
            item.setGLOptions(GIZMO_GL)
            item.setDepthValue(10)
            item.setVisible(False)
            self.addItem(item)

    def _place_gizmo(self):
        """Put the handle at the zero point, at its fixed size on screen."""
        origin = self._gizmo_origin
        if not self.show_gizmo or origin is None:
            for item in self._gizmo_items:
                item.setVisible(False)
            return
        mm_per_px = float(self.pixelSize(pg.Vector(*origin)))
        if not np.isfinite(mm_per_px) or mm_per_px <= 0.0:
            return
        shaft_mm = GIZMO_SHAFT_PX * mm_per_px
        head_mm = GIZMO_HEAD_PX * mm_per_px
        self._gizmo_len_mm = shaft_mm + head_mm

        ball = pg.Transform3D()
        ball.translate(*origin)
        ball.scale(*((GIZMO_BALL_PX * mm_per_px,) * 3))
        self._gizmo_items[0].setTransform(ball)

        for axis in range(3):
            shaft, head, label = self._gizmo_items[1 + axis * 3:4 + axis * 3]
            direction = np.eye(3)[axis]
            shaft.setData(pos=np.array([origin, origin + direction * shaft_mm]))
            tr = pg.Transform3D()
            tr.translate(*origin)
            tr.rotate(*GIZMO_ROTATION[axis])
            tr.translate(0.0, 0.0, shaft_mm)
            tr.scale(head_mm * 0.34, head_mm * 0.34, head_mm)
            head.setTransform(tr)
            label.setData(pos=origin + direction
                          * (self._gizmo_len_mm + 6.0 * mm_per_px))

        self._ring_mm = GIZMO_RING_PX * mm_per_px
        th = np.linspace(0.0, 2.0 * np.pi, GIZMO_RING_SEGMENTS + 1)
        self._ring.setData(pos=origin + self._ring_mm * np.stack(
            [np.cos(th), np.sin(th), np.zeros_like(th)], axis=1))

        for item in self._gizmo_items:
            item.setVisible(True)
        self._paint_gizmo()

    def _paint_gizmo(self):
        """Light whatever is under the pointer, so it is clear what will move."""
        for axis in range(3):
            shaft, head, _label = self._gizmo_items[1 + axis * 3:4 + axis * 3]
            hot = axis == self._gizmo_hot
            colour = GIZMO_HOT_COLOR if hot else GIZMO_AXIS_COLOR[axis]
            shaft.setData(color=colour, width=5.0 if hot else 3.0)
            head.setColor(colour)
        if self._ring is not None:
            self._ring.setData(
                color=GIZMO_HOT_COLOR if self._ring_hot else GIZMO_RING_COLOR,
                width=4.5 if self._ring_hot else 2.5)

    def _set_hot(self, axis, ring=False):
        if axis == self._gizmo_hot and bool(ring) == self._ring_hot:
            return
        self._gizmo_hot = axis
        self._ring_hot = bool(ring)
        self._paint_gizmo()
        self.setCursor(
            QtCore.Qt.CursorShape.SizeAllCursor if axis is not None
            else QtCore.Qt.CursorShape.OpenHandCursor if ring
            else QtCore.Qt.CursorShape.ArrowCursor)

    def set_gizmo_visible(self, on):
        self.show_gizmo = bool(on)
        if not self.show_gizmo:
            self._drag = self._turn = None
            self._set_hot(None)
        self._place_gizmo()

    def _ring_at(self, pos):
        """True when the pointer is on the rotation ring rather than inside it.

        Tested in the machine's own XY plane rather than on screen: the ring is
        a circle in the world and an ellipse in the window, and chasing that
        ellipse in pixels gets the near and far sides wrong at a shallow
        elevation. Casting the pointer onto the plane the ring lives in gives
        the right answer at any camera angle.
        """
        if (not self.show_gizmo or self._gizmo_origin is None
                or self._gizmo_len_mm <= 0.0):
            return False
        where = self._on_z_plane(pos, float(self._gizmo_origin[2]))
        if where is None:
            return False
        radius = float(np.hypot(*(where[:2] - self._gizmo_origin[:2])))
        # Half the arrows' grab width, in world units, so the ring and an
        # arrow lying under it cannot both claim the same pixel.
        band = max(self._ring_mm * 0.12, 1e-6)
        return abs(radius - self._ring_mm) <= band

    def _on_z_plane(self, pos, z_mm):
        """Where the pointer's ray crosses a horizontal plane, machine mm."""
        eye = np.array(self.cameraPosition())
        viewport = self.getViewport()
        mvp = self.projectionMatrix(viewport, viewport) * self.viewMatrix()
        inverted, ok = mvp.inverted()
        if not ok:
            return None
        w, h = float(self.width()), float(self.height())
        ndc_x = 2.0 * pos.x() / max(w, 1.0) - 1.0
        ndc_y = 1.0 - 2.0 * pos.y() / max(h, 1.0)
        far = inverted.map(QtGui.QVector4D(ndc_x, ndc_y, 1.0, 1.0))
        if abs(far.w()) < 1e-12:
            return None
        target = np.array([far.x() / far.w(), far.y() / far.w(),
                           far.z() / far.w()])
        ray = target - eye
        if abs(ray[2]) < 1e-9:              # looking along the plane
            return None
        t = (z_mm - eye[2]) / ray[2]
        if t <= 0.0:                        # the plane is behind the camera
            return None
        return eye + t * ray

    def _angle_at(self, pos):
        """The pointer's bearing about the zero point, degrees, or None."""
        where = self._on_z_plane(pos, float(self._gizmo_origin[2]))
        if where is None:
            return None
        d = where[:2] - self._gizmo_origin[:2]
        if float(d @ d) < 1e-12:
            return None
        return math.degrees(math.atan2(d[1], d[0]))

    def _to_screen(self, points_mm):
        """Machine millimetres -> widget pixels; NaN for anything behind."""
        viewport = self.getViewport()
        mvp = self.projectionMatrix(viewport, viewport) * self.viewMatrix()
        w, h = float(self.width()), float(self.height())
        out = np.full((len(points_mm), 2), np.nan)
        for i, p in enumerate(points_mm):
            v = mvp.map(QtGui.QVector4D(float(p[0]), float(p[1]),
                                        float(p[2]), 1.0))
            if v.w() <= 1e-9:
                continue
            out[i] = ((v.x() / v.w() + 1.0) * 0.5 * w,
                      (1.0 - v.y() / v.w()) * 0.5 * h)
        return out

    def _gizmo_screen(self):
        """The handle as four screen points: the ball, then the three tips."""
        if not self.show_gizmo or self._gizmo_origin is None:
            return None
        pts = np.vstack([self._gizmo_origin,
                         self._gizmo_origin + np.eye(3) * self._gizmo_len_mm])
        screen = self._to_screen(pts)
        return None if not np.isfinite(screen).all() else screen

    def _axis_at(self, pos):
        """Which arrow the pointer is on, or None."""
        screen = self._gizmo_screen()
        if screen is None:
            return None
        p = np.array([pos.x(), pos.y()])
        best, best_px = None, GIZMO_GRAB_PX
        for axis in range(3):
            d = _segment_distance_px(p, screen[0], screen[1 + axis])
            if d < best_px:
                best, best_px = axis, d
        return best

    # ---- dragging --------------------------------------------------------
    def mousePressEvent(self, ev):
        """Take a left click on an arrow as a drag; anything else orbits."""
        if (ev.button() == QtCore.Qt.MouseButton.LeftButton
                and self._placement is not None):
            pos = ev.position()
            axis = self._axis_at(pos)
            screen = self._gizmo_screen() if axis is not None else None
            if screen is not None:
                step = screen[1 + axis] - screen[0]
                # An axis pointing nearly at the camera has no usable screen
                # direction -- a pixel of mouse would be metres of probe. Leave
                # it to the other two rather than lurch.
                if float(step @ step) >= 16.0:
                    self.mousePos = pos
                    self._drag = (axis, np.array([pos.x(), pos.y()]), step,
                                  self._gizmo_len_mm,
                                  np.array([self._placement.x_mm,
                                            self._placement.y_mm,
                                            self._placement.z_mm]))
                    self._set_hot(axis)
                    return
            if axis is None and self._ring_at(pos):
                grabbed = self._angle_at(pos)
                if grabbed is not None:
                    self.mousePos = pos
                    self._turn = (grabbed, self._placement.rot_z_deg)
                    self._set_hot(None, ring=True)
                    return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        """Slide along the grabbed axis, or let the base class orbit.

        The offset is measured from where the arrow was grabbed rather than
        accumulated frame by frame, so the pose cannot drift away from the
        pointer over a long drag -- and rounding in the spin box it feeds
        cannot creep.
        """
        pos = ev.position()
        if self._drag is not None:
            axis, grabbed, step, length, base = self._drag
            moved = np.array([pos.x(), pos.y()]) - grabbed
            along = float(moved @ step) / float(step @ step) * length
            where = base.copy()
            where[axis] += along
            self.mousePos = pos
            self.pose_dragged.emit(*(float(v) for v in where))
            return
        if self._turn is not None:
            grabbed, was = self._turn
            now = self._angle_at(pos)
            self.mousePos = pos
            if now is not None:
                self.pose_turned.emit(was + (now - grabbed))
            return
        super().mouseMoveEvent(ev)
        if ev.buttons():
            self._place_gizmo()     # the camera moved, so the handle resizes
        else:
            axis = self._axis_at(pos)
            self._set_hot(axis, ring=axis is None and self._ring_at(pos))

    def mouseReleaseEvent(self, ev):
        if self._drag is not None or self._turn is not None:
            self._drag = self._turn = None
            axis = self._axis_at(ev.position())
            self._set_hot(axis, ring=axis is None and self._ring_at(ev.position()))
            return
        super().mouseReleaseEvent(ev)

    def wheelEvent(self, ev):
        super().wheelEvent(ev)
        self._place_gizmo()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._place_gizmo()

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
        self._placement = placement
        self._gizmo_origin = np.asarray(origin, dtype=float)
        tr = pg.Transform3D()
        tr.setRow(0, QtGui.QVector4D(rot[0, 0], rot[0, 1], rot[0, 2], origin[0]))
        tr.setRow(1, QtGui.QVector4D(rot[1, 0], rot[1, 1], rot[1, 2], origin[1]))
        tr.setRow(2, QtGui.QVector4D(rot[2, 0], rot[2, 1], rot[2, 2], origin[2]))
        tr.setRow(3, QtGui.QVector4D(0.0, 0.0, 0.0, 1.0))
        for item in self._probe_items:
            item.setTransform(tr)
        self._draw_reach(placement)
        self._place_gizmo()

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
            self._place_gizmo()
            return
        lo, hi = self.coils.bounds_mm()
        centre = pg.Vector(*(0.5 * (lo + hi)))
        self.opts["center"] = centre
        self.setCameraPosition(pos=centre,
                               distance=float(np.linalg.norm(hi - lo)) * 1.15,
                               elevation=28, azimuth=35)
        self._place_gizmo()

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
        self._place_gizmo()
