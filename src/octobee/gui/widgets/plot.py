"""The rolling trace plot, and a PlotWidget that times its own repaint."""


import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from octobee import profile as oprof
from octobee.gui.constants import N_SENSORS, PLOT_PEN_WIDTH, PLOT_TARGET_MULT
from octobee.gui.palette import sensor_colors

class ProfiledPlot(pg.PlotWidget):
    """
    A PlotWidget that times its own repaint.

    Without this the table has a hole in it: asking a curve to setData is
    cheap, and the expensive part -- Qt actually rasterising 16 or 48
    polylines -- happens later in the event loop, where it would show up only
    as unexplained lag. Timing it here means every row of the profile adds up
    to something, and "none of these is big but the loop still stalls" becomes
    a real conclusion rather than a gap in the measurement.
    """

    def __init__(self, label, profiler=None, **kw):
        super().__init__(**kw)
        self._label = label
        self.profiler = profiler or oprof.Profiler(enabled=False)

    def paintEvent(self, ev):
        with self.profiler.time(self._label):
            super().paintEvent(ev)


class LivePlot(QtWidgets.QWidget):
    """Rolling traces. |B| per sensor by default, since that is comparable."""

    MODES = ("|B| per sensor", "all axes (chip frame)", "all axes (tube frame)")
    # What the y axis shows. The pipeline always works in millitesla; these
    # walk that back down the conversion chain so the electrical signal can be
    # inspected directly -- useful when you want to know whether a number is
    # the sensor talking or the conversion assuming.
    UNITS = ("mT", "uT", "mV (chip output)", "ADC counts")

    def __init__(self, geom, profiler=None):
        super().__init__()
        self.profiler = profiler or oprof.Profiler(enabled=False)
        self.target_mult = PLOT_TARGET_MULT
        self.unit_scale = np.ones(N_SENSORS)
        self.unit_name = "mT"
        self.geom = geom
        self.mode = self.MODES[0]
        self.visible = set(range(N_SENSORS))
        self.dead = set()
        self.colors = sensor_colors()

        self.plot = ProfiledPlot("Qt paint (live plot)", profiler)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "time", units="s")
        self.plot.setLabel("left", "B", units="mT")
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.setClipToView(True)
        self.legend = self.plot.addLegend(colCount=4, labelTextSize="7pt")

        self.mag_curves, self.axis_curves = [], []
        for s in range(N_SENSORS):
            c = self.plot.plot(
                pen=pg.mkPen(self.colors[s], width=PLOT_PEN_WIDTH),
                name=f"S{s+1}")
            self._thin(c)
            self.mag_curves.append(c)
            row = []
            for _a, style in enumerate((QtCore.Qt.PenStyle.SolidLine,
                                       QtCore.Qt.PenStyle.DashLine,
                                       QtCore.Qt.PenStyle.DotLine)):
                cc = self.plot.plot(pen=pg.mkPen(
                    self.colors[s], width=PLOT_PEN_WIDTH, style=style))
                self._thin(cc)
                row.append(cc)
            self.axis_curves.append(row)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plot)
        self.set_mode(self.mode)

    @staticmethod
    def _thin(curve):
        """
        Make each curve downsample itself to the screen.

        A 20 s window at 500 Hz is 10 000 points per trace, and there are up to
        48 traces. Setting this on the PlotItem is not enough -- it has to be on
        the curves -- and without it Qt rasterises every one of those points
        into a plot around a thousand pixels wide. Measured, that was the single
        most expensive thing in the whole application, tens of milliseconds per
        repaint, and invisible to any timing of our own code because it happens
        inside Qt's paint. 'peak' keeps the extremes of each bin, so a magnet
        spike still shows at full height.
        """
        curve.setDownsampling(auto=True, method="peak")
        curve.setClipToView(True)

    def set_mode(self, mode):
        self.mode = mode
        self._relabel()
        self._apply_visibility()

    def set_units(self, scale, name):
        """Per-sensor scale factor taking millitesla to the displayed unit."""
        self.unit_scale = np.asarray(scale, float).reshape(N_SENSORS)
        self.unit_name = name
        self._relabel()

    def _relabel(self):
        mag = self.mode == self.MODES[0]
        self.plot.setLabel("left", "|B|" if mag else "B",
                           units=self.unit_name)

    def reset_view(self):
        """Undo any panning or zooming and go back to following the data.

        pyqtgraph switches auto-ranging OFF the instant you drag or scroll a
        plot, and nothing on screen says so: the traces simply stop filling the
        axes, which reads as the signal having changed rather than the view
        having been moved. A scroll over the axis can also leave the mouse
        enabled on one axis only, which is even harder to spot.

        Order matters. ViewBox.autoRange() fits once and disables auto-ranging
        as it goes -- it calls setRange(), which defaults to
        disableAutoRange=True -- so enabling has to come last or the button
        would leave the plot frozen at exactly the moment it was pressed.
        """
        vb = self.plot.getViewBox()
        vb.setMouseEnabled(x=True, y=True)
        vb.enableAutoRange(x=True, y=True)
        vb.updateAutoRange()

    def set_visible_sensors(self, sensors):
        self.visible = set(sensors)
        self._apply_visibility()

    def set_dead(self, dead):
        """
        Hide railed sensors. A stuck channel sits at -32768 counts, which is
        ~195 mT of nonsense -- left on the plot it pins the y axis and every
        real trace collapses onto the zero line.
        """
        dead = {int(d[1:]) - 1 for d in dead if d.startswith("S") and d[1:].isdigit()}
        if dead != self.dead:
            self.dead = dead
            self._apply_visibility()

    def _apply_visibility(self):
        mag = self.mode == self.MODES[0]
        for s in range(N_SENSORS):
            on = s in self.visible and s not in self.dead
            self.mag_curves[s].setVisible(on and mag)
            for c in self.axis_curves[s]:
                c.setVisible(on and not mag)
        for s in range(N_SENSORS):
            if s >= len(self.legend.items):
                continue
            on = s in self.visible and s not in self.dead
            for part in self.legend.items[s]:
                if part is not None:
                    part.setVisible(on)

    @staticmethod
    def _to_screen(t, Y, target):
        """
        Reduce (n, ncurves) to about `target` points while keeping the extremes.

        Each output bin contributes its minimum and its maximum, so a magnet
        spike one sample wide still reaches full height -- which a plain stride
        would throw away. Sharing one x array across all curves keeps this fully
        vectorised, at the cost of placing each pair at the bin edges rather
        than at the exact sample; one bin is about one pixel, so that is not
        visible.

        Doing this ourselves rather than leaving it to the plotting library
        makes the repaint cost depend on the width of the window in pixels
        instead of on the length of the buffer, so a longer time window or a
        higher output rate no longer costs anything to draw.
        """
        n = Y.shape[0]
        if n <= target:
            return t, Y
        bins = max(2, target // 2)
        k = n // bins
        m = bins * k
        Yb = Y[:m].reshape(bins, k, Y.shape[1])
        out = np.empty((bins * 2, Y.shape[1]), dtype=Y.dtype)
        out[0::2] = Yb.min(axis=1)
        out[1::2] = Yb.max(axis=1)
        tb = t[:m:k]
        x = np.repeat(tb, 2)
        x[1::2] = tb + (t[-1] - t[0]) / bins * 0.5
        return x, out

    def update_data(self, b_mt, fs_out):
        n = b_mt.shape[0]
        if n < 2:
            return
        t = (np.arange(n) - n) / fs_out
        shown = self.visible - self.dead
        self.profiler.note("curves drawn", len(shown) * (1 if self.mode ==
                                                         self.MODES[0] else 3))
        self.profiler.note("points buffered", n)
        target = max(256, int(self.plot.width() * self.target_mult))
        k = self.unit_scale
        if self.mode == self.MODES[0]:
            # |k B| == k |B| for positive k, so scaling the magnitude is the
            # same as scaling the components first.
            mag = np.linalg.norm(b_mt, axis=-1) * k[None, :]
            x, y = self._to_screen(t, mag, target)
            for s in shown:
                self.mag_curves[s].setData(x, y[:, s])
        else:
            b = self.geom.to_tube_frame(b_mt) if "tube" in self.mode else b_mt
            b = b * k[None, :, None]
            x, y = self._to_screen(t, b.reshape(n, -1), target)
            for s in shown:
                for a in range(3):
                    self.axis_curves[s][a].setData(x, y[:, s * 3 + a])
        self.profiler.note("points drawn per curve", len(x))
