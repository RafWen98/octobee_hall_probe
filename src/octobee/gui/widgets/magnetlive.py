"""The guided magnet run, drawn while it happens.

A pose of the standard run is 145 points at 2.5 s each -- eleven minutes in
which the only thing on screen used to be a progress bar and a count. That is
long enough to sit through a run that is already wrong: the magnet clamped
somewhere the sweep never reaches, a face that answers at a tenth of the
others, a peak walking off the end of the travel, a channel on the rail. All
four are obvious in the first two minutes of a picture and invisible in a
count of points.

So this draws the pass as it is measured -- |B| per sensor against the axis
being moved -- and marks the two things that decide whether the pass is worth
keeping:

    last pose's rings           vertical lines, drawn on pass A only -- they
                                are positions along the TUBE and mean nothing
                                on an axis measured in x or z. The head is
                                only turned between poses, never moved along
                                it, so this pose's four peaks must land on
                                last pose's four lines; a pose that also
                                shifted the head sideways is the one setup
                                fault that breaks the equal-approach argument
                                quietly, and this is what it looks like
    full scale                  a horizontal line. A trace touching it is a
                                clipped channel, which reads LOW and flat and
                                would be trimmed UP to compensate -- the one
                                failure in this routine that quietly makes the
                                calibration worse. See MagnetRun.range_check.

|B| rather than the components, for the same reason the whole routine works on
|B|: sixteen chips are turned sixteen ways and a magnitude is the only thing
directly comparable between them. The four sensors on the face that is turned
toward the magnet are drawn bright and the other twelve faint, recomputed as
the pass goes -- in 1/r^3 terms the far face is metres away, and without that
the picture is four signals and twelve flat lines all competing for the eye.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from octobee.gui.constants import N_SENSORS
from octobee.gui.palette import sensor_colors

# How many sensors to draw bright: one face's worth, because in any pose that
# is how many are actually near the magnet.
LOUD = 4
PEN_LOUD, PEN_QUIET = 2.0, 1.0
ALPHA_QUIET = 70


class MagnetPassPlot(QtWidgets.QWidget):
    """|B| per sensor against the axis a pass is sweeping, as it arrives."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.colors = sensor_colors()
        self._x = []
        self._y = [[] for _ in range(N_SENSORS)]
        self._rings = []
        self._fs_line = None

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("left", "|B|", units="mT")
        self.plot.setLabel("bottom", "position", units="mm")
        # Named at creation but NOT added to the legend: sixteen entries is a
        # block of text across the top quarter of the plot, over the traces it
        # is labelling, and twelve of the sixteen are the faint ones nobody is
        # reading. _emphasise() puts the four that are answering in it and
        # takes them out again when they stop.
        self.legend = self.plot.addLegend(colCount=4, labelTextSize="7pt")
        self.curves, self._named = [], set()
        for s in range(N_SENSORS):
            c = self.plot.plot(pen=pg.mkPen(self.colors[s], width=PEN_QUIET))
            c.opts["name"] = f"S{s + 1}"
            self.curves.append(c)

        self.lbl = QtWidgets.QLabel("")
        self.lbl.setTextFormat(QtCore.Qt.TextFormat.PlainText)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        lay.addWidget(self.plot, 1)
        lay.addWidget(self.lbl)
        self.begin("", "position")

    # ---- the pass ---------------------------------------------------------
    def begin(self, title, axis_name):
        """Start a new pass. Clears the traces; the caller owns the markers.

        Ring markers are NOT cleared here, because whether they still mean
        anything depends on what the next pass is measuring and only the
        caller knows that -- see set_rings.
        """
        self._x = []
        self._y = [[] for _ in range(N_SENSORS)]
        for c in self.curves:
            c.setData([], [])
        for s in sorted(self._named):
            self.legend.removeItem(self.curves[s])
        self._named = set()
        self.plot.setTitle(title)
        self.plot.setLabel("bottom", axis_name, units="mm")
        self.lbl.setText("waiting for the first point")

    def set_full_scale(self, mt):
        """Draw the line a trace must not touch."""
        if self._fs_line is not None:
            self.plot.removeItem(self._fs_line)
            self._fs_line = None
        if mt is None or not np.isfinite(mt):
            return
        self._fs_line = pg.InfiniteLine(
            pos=float(mt), angle=0,
            pen=pg.mkPen((220, 80, 70), width=1.5,
                         style=QtCore.Qt.PenStyle.DashLine),
            label=f"full scale {float(mt):g} mT",
            labelOpts={"position": 0.05, "color": (220, 80, 70),
                       "fill": (0, 0, 0, 0)})
        # ignoreBounds, or the ceiling sets the floor. A 20 mT full scale on a
        # run peaking at 3 mT drags the y range up to 20 and squashes every
        # trace into the bottom seventh of the plot -- the marker that exists
        # to make one failure visible would have hidden everything else. Left
        # out of the auto-range it appears exactly when the data climbs near
        # it, which is the only time it has anything to say.
        self.plot.addItem(self._fs_line, ignoreBounds=True)

    def set_rings(self, positions):
        """Vertical markers, in the units of whatever axis is on the bottom.

        Pass an empty list on any pass that is not sweeping the tube axis.
        Ring positions are measured along the tube, and a line drawn at y=73
        on an axis running 30..50 mm of x is not a wrong-looking line -- it is
        an invisible one, or worse, a plausible one in the wrong place.
        """
        for item in self._rings:
            self.plot.removeItem(item)
        self._rings = []
        for p in positions or []:
            line = pg.InfiniteLine(
                pos=float(p), angle=90,
                pen=pg.mkPen((150, 150, 150), width=1,
                             style=QtCore.Qt.PenStyle.DotLine))
            self.plot.addItem(line)
            self._rings.append(line)

    def add(self, x_mm, b_row):
        """One measured point: the stage coordinate and its (16, 3) mT."""
        b = np.asarray(b_row, float)
        if b.shape != (N_SENSORS, 3):
            return
        mag = np.linalg.norm(b, axis=1)
        self._x.append(float(x_mm))
        for s in range(N_SENSORS):
            self._y[s].append(float(mag[s]))
        x = np.asarray(self._x)
        for s in range(N_SENSORS):
            self.curves[s].setData(x, np.asarray(self._y[s]))
        self._emphasise()
        loud = int(np.argmax([max(y) for y in self._y]))
        self.lbl.setText(
            f"{len(self._x)} points -- loudest S{loud + 1} at "
            f"{max(self._y[loud]):.3f} mT, this point "
            f"{mag.max():.3f} mT on S{int(np.argmax(mag)) + 1}")

    def _emphasise(self):
        """Bright for the face that is answering, faint for the other twelve.

        On the running maximum rather than the latest point, so a sensor stays
        bright once its peak has gone by instead of flicking back to faint the
        moment the magnet moves off it.
        """
        peak = np.array([max(y) if y else 0.0 for y in self._y])
        loud = {int(i) for i in np.argsort(peak)[-LOUD:]}
        for s in range(N_SENSORS):
            col = pg.mkColor(self.colors[s])
            if s in loud:
                self.curves[s].setPen(pg.mkPen(col, width=PEN_LOUD))
            else:
                col.setAlpha(ALPHA_QUIET)
                self.curves[s].setPen(pg.mkPen(col, width=PEN_QUIET))
        if loud == self._named:
            return
        for s in sorted(self._named - loud):
            self.legend.removeItem(self.curves[s])
        for s in sorted(loud - self._named):
            self.legend.addItem(self.curves[s], f"S{s + 1}")
        self._named = loud
