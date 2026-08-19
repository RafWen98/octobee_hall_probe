#!/usr/bin/env python3
"""
probe_view3d.py -- the little 3D model of the probe head.

Draws the square tube with all 16 chips on it and, on each chip, the field
vector it is currently reporting. The point is not decoration: this probe is the
one case where a plot of 48 traces genuinely hides what is going on, because
every chip is turned a different way. Once each chip's vector is rotated into
the tube frame and drawn where the chip physically sits, a magnet passing the
probe reads immediately -- you can see which face it went past and in which
direction the field pointed, which no stack of time series shows.

Colour encodes |B| (rotation invariant, so it is comparable between chips);
arrow direction is the tube-frame field direction; arrow length is |B| against
a shared scale so chip-to-chip amplitude differences are visible at a glance.
Dead or excluded chips are drawn dark red with no arrow.
"""

from collections import deque

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6 import QtGui

import probe_geometry as pg_geom

# Turbo runs blue -> green -> red with even perceptual steps, so a small change
# near zero is as visible as one near full scale.
try:
    import matplotlib.cm as _cm
    _CMAP = _cm.get_cmap("turbo") if hasattr(_cm, "get_cmap") else None
except Exception:                                   # noqa: BLE001
    _CMAP = None
if _CMAP is None:
    try:
        from matplotlib import colormaps as _cmaps
        _CMAP = _cmaps["turbo"]
    except Exception:                               # noqa: BLE001
        _CMAP = None


def color_for(frac):
    """0..1 -> RGBA tuple."""
    f = float(np.clip(frac, 0.0, 1.0))
    if _CMAP is not None:
        return tuple(_CMAP(f))
    # linear blue -> cyan -> yellow -> red fallback
    stops = [(0.0, (0.15, 0.25, 0.85)), (0.35, (0.10, 0.75, 0.80)),
             (0.70, (0.95, 0.85, 0.15)), (1.0, (0.90, 0.15, 0.10))]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if f <= b:
            t = (f - a) / (b - a)
            return tuple(ca[i] + t * (cb[i] - ca[i]) for i in range(3)) + (1.0,)
    return stops[-1][1] + (1.0,)


DEAD_COLOR = (0.45, 0.10, 0.10, 1.0)
TUBE_COLOR = (0.62, 0.66, 0.72, 0.35)
ARM_COLOR = (0.55, 0.36, 0.20, 1.0)      # the eval-kit PCBs, bare board colour


class ProbeView3D(gl.GLViewWidget):
    """Interactive 3D view of the probe head. Drag to orbit, wheel to zoom."""

    def __init__(self, geom=None, parent=None):
        super().__init__(parent)
        self.geom = geom or pg_geom.Geometry()
        self.arrow_scale_mm = None        # full-scale arrow length; None = auto
        self.full_scale_mt = 1.0          # |B| that maps to full scale
        self.auto_scale = True
        # Rolling maximum over the last few seconds of frames. A plain
        # frame-by-frame maximum renormalises every frame, so a magnet would
        # look the same size at every sensor -- which is the one comparison this
        # view exists to make. A decaying maximum instead latches onto whatever
        # transient came first (a railed sensor reads ~195 mT) and takes minutes
        # to let go. A bounded window does neither.
        self._recent_max = deque(maxlen=90)
        self.show_labels = True
        self.show_arrows = True
        self.dead = set()
        self._items = []
        self.setBackgroundColor(pg.mkColor(18, 20, 26))
        self.rebuild()

    # ---- construction ----------------------------------------------------
    def rebuild(self, geom=None):
        """Tear down and redraw everything. Called when the geometry changes."""
        if geom is not None:
            self.geom = geom
        for it in self._items:
            self.removeItem(it)
        self._items = []
        g = self.geom
        n = pg_geom.N_SENSORS

        self._add_grid()
        self._add_tube()
        self._add_axes()

        self.pads, self.arrows, self.tips, self.labels = [], [], [], []
        for sid in range(1, n + 1):
            # The arm is structure, not data: draw it once in board colour so
            # the eye can see how far off the tube each chip really sits.
            averts, afaces = pg_geom.arm_mesh(g, sid)
            self._register(gl.GLMeshItem(
                meshdata=gl.MeshData(vertexes=averts, faces=afaces),
                smooth=False, drawEdges=False, color=ARM_COLOR,
                shader="shaded", glOptions="opaque"))

            verts, faces = pg_geom.chip_mesh(g, sid)
            md = gl.MeshData(vertexes=verts, faces=faces)
            pad = gl.GLMeshItem(meshdata=md, smooth=False, drawEdges=True,
                                edgeColor=(0.1, 0.1, 0.12, 1.0),
                                color=color_for(0.0), shader="shaded",
                                glOptions="opaque")
            self._register(pad)
            self.pads.append(pad)

            arr = gl.GLLinePlotItem(pos=np.zeros((2, 3)), width=3.0,
                                    antialias=True, color=(1, 1, 1, 1))
            self._register(arr)
            self.arrows.append(arr)

            tip = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=8.0,
                                       color=(1, 1, 1, 1), pxMode=True)
            self._register(tip)
            self.tips.append(tip)

            p = g.position(sid) + g.normal(sid) * 8.0
            txt = gl.GLTextItem(pos=p, text=f"S{sid}",
                                color=(210, 215, 225, 255))
            self._register(txt)
            self.labels.append(txt)

        self.reset_camera()
        self.update_fields(np.zeros((n, 3)))

    def _register(self, item):
        self.addItem(item)
        self._items.append(item)

    def _add_tube(self):
        verts, faces = pg_geom.tube_mesh(self.geom)
        md = gl.MeshData(vertexes=verts, faces=faces)
        body = gl.GLMeshItem(meshdata=md, smooth=False, drawEdges=True,
                             edgeColor=(0.55, 0.60, 0.70, 0.9),
                             color=TUBE_COLOR, shader="balloon",
                             glOptions="translucent")
        self._register(body)

    @property
    def extent_mm(self):
        """Overall size of the probe: the arms reach far past the tube itself."""
        return max(self.geom.tube_length_mm, 2.0 * self.geom.fsv_radius_mm)

    def _add_grid(self):
        span = self.extent_mm * 1.15
        grid = gl.GLGridItem()
        grid.setSize(span, span)
        grid.setSpacing(span / 14.0, span / 14.0)
        grid.translate(0, 0, -self.geom.fsv_radius_mm * 0.6)
        grid.setColor((70, 76, 90, 110))
        self._register(grid)

    def _add_axes(self):
        L = self.geom.fsv_radius_mm * 0.55
        origin = np.array([0.0, 0.0, -self.geom.fsv_radius_mm * 0.25])
        for vec, col, name in ((np.array([L, 0, 0]), (1, .35, .35, 1), "X"),
                               (np.array([0, L, 0]), (.35, 1, .35, 1), "Y"),
                               (np.array([0, 0, L]), (.45, .6, 1, 1), "Z")):
            self._register(gl.GLLinePlotItem(
                pos=np.array([origin, origin + vec]), color=col, width=2.0,
                antialias=True))
            self._register(gl.GLTextItem(
                pos=origin + vec * 1.12, text=name,
                color=QtGui.QColor.fromRgbF(*col)))

    def reset_camera(self):
        self.opts["center"] = pg.Vector(0, 0, self.geom.tube_length_mm / 2.0)
        # Side-on and slightly above: the tube runs across the view, so all four
        # faces and the whole length are readable at once. Frame the arms, not
        # the tube -- the chips sit far outside it.
        self.setCameraPosition(distance=self.extent_mm * 1.5,
                               elevation=16, azimuth=-65)

    # ---- live update -----------------------------------------------------
    def reset_scale(self):
        """Forget the auto-scale history, e.g. after a calibration change."""
        self._recent_max.clear()

    def set_dead(self, dead):
        dead = set(dead or ())
        if dead != self.dead:
            # The excluded set just changed, so the scale was computed over a
            # different population -- start it again rather than carry it over.
            self.reset_scale()
        self.dead = dead

    def update_fields(self, b_chip_mt):
        """
        b_chip_mt: (16, 3) instantaneous field per sensor in its OWN chip axes.
        Rotation into the tube frame happens here, so callers never have to
        think about it.
        """
        b = np.asarray(b_chip_mt, float).reshape(pg_geom.N_SENSORS, 3)
        b_tube = self.geom.to_tube_frame(b)
        mag = np.linalg.norm(b_tube, axis=1)

        live = np.array([f"S{i+1}" not in self.dead
                         for i in range(pg_geom.N_SENSORS)])
        vis = mag[live] if live.any() else mag
        if self.auto_scale and vis.size:
            self._recent_max.append(float(np.max(vis)))
            self.full_scale_mt = max(max(self._recent_max), 1e-4)
        fs = max(self.full_scale_mt, 1e-9)

        for i in range(pg_geom.N_SENSORS):
            sid = i + 1
            dead = not live[i]
            if dead:
                self.pads[i].setColor(DEAD_COLOR)
                self.arrows[i].setData(pos=np.zeros((2, 3)), color=(0, 0, 0, 0))
                self.tips[i].setData(pos=np.zeros((1, 3)), color=(0, 0, 0, 0))
                continue
            frac = mag[i] / fs
            col = color_for(frac)
            self.pads[i].setColor(col)
            if not self.show_arrows or mag[i] <= 0:
                self.arrows[i].setData(pos=np.zeros((2, 3)), color=(0, 0, 0, 0))
                self.tips[i].setData(pos=np.zeros((1, 3)), color=(0, 0, 0, 0))
                continue
            p0 = self.geom.position(sid)
            direction = b_tube[i] / mag[i]
            scale = (self.arrow_scale_mm if self.arrow_scale_mm
                     else 0.45 * self.geom.arm_length_mm)
            length = scale * min(frac, 1.0)
            p1 = p0 + direction * length
            self.arrows[i].setData(pos=np.array([p0, p1]), color=col, width=3.0)
            self.tips[i].setData(pos=p1[None, :], color=col, size=9.0)

        if self.show_labels:
            hot = int(np.argmax(np.where(live, mag, -np.inf))) if live.any() else -1
            for i, txt in enumerate(self.labels):
                mark = " <" if i == hot and mag[i] > 0 else ""
                txt.setData(text=f"S{i+1}{mark}")
        return fs

    def set_labels_visible(self, on):
        self.show_labels = bool(on)
        for i, txt in enumerate(self.labels):
            txt.setData(text=f"S{i+1}" if on else "")

    def set_arrows_visible(self, on):
        self.show_arrows = bool(on)
