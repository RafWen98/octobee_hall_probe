"""The 16 sensors as a bar chart and as a table."""

from typing import ClassVar

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from octobee.acq import carrier as ob
from octobee.calib import geometry as pgeom
from octobee.gui.constants import N_SENSORS
from octobee.gui.palette import sensor_colors
from octobee.gui.widgets.plot import ProfiledPlot

class SensorBars(QtWidgets.QWidget):
    """
    Peak |B| per sensor since the last refresh -- the "are the spikes the same
    height" view, and the reason this window exists.

    Excluded sensors are drawn as a flat red stub rather than at their real
    value: a railed channel reads nearly 200 mT, which would flatten every real
    bar against the axis.
    """

    def __init__(self, profiler=None):
        super().__init__()
        self.plot = ProfiledPlot("Qt paint (peak bars)", profiler)
        self.plot.showGrid(y=True, alpha=0.25)
        self.plot.setLabel("left", "peak |B|", units="mT")
        self.window_s = 0.5
        self.plot.setTitle(f"peak |B| per sensor, last {self.window_s:g} s",
                           size="9pt")
        ax = self.plot.getAxis("bottom")
        ax.setTicks([[(i, f"{i+1}") for i in range(N_SENSORS)]])
        self.plot.setLabel("bottom", "sensor")
        self.colors = sensor_colors()
        self.bars = pg.BarGraphItem(x=np.arange(N_SENSORS), height=np.zeros(N_SENSORS),
                                    width=0.72, brushes=self.colors)
        self.plot.addItem(self.bars)
        self.median_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("w", style=QtCore.Qt.PenStyle.DashLine))
        self.plot.addItem(self.median_line)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plot)
        self.dead = set()

    def update_values(self, mag, dead=()):
        self.dead = set(dead)
        live = np.array([f"S{i+1}" not in self.dead for i in range(N_SENSORS)])
        h = np.array(mag, dtype=float)
        h[~np.isfinite(h)] = 0.0
        vis = h[live]
        med = float(np.median(vis)) if vis.size else 0.0
        stub = max(med * 0.04, (float(vis.max()) if vis.size else 1.0) * 0.02)
        brushes = []
        for i in range(N_SENSORS):
            if not live[i]:
                h[i] = stub                       # keep the axis on the real data
                brushes.append(pg.mkBrush(140, 40, 40))
            else:
                brushes.append(pg.mkBrush(self.colors[i]))
        self.bars.setOpts(height=h, brushes=brushes)
        self.median_line.setValue(med)


class SensorTable(QtWidgets.QTableWidget):
    """Per-sensor state, and where the per-sensor measurement range is set."""

    COLS = ("sensor", "face", "state", "|B| mT", "Bx", "By", "Bz",
            "noise uT", "VCM V", "T degC*", "range", "gain trim")
    # The star on T degC: octobee.temp_c is uncalibrated and each chip's TA
    # output carries its own offset, so the spread between sensors is not a
    # gradient. Read per chip, over time.
    COL_TIPS: ClassVar[dict] = {"T degC*": ob.TEMP_UNCALIBRATED_NOTE}
    STATE_COLORS: ClassVar[dict] = {
        "ok": (40, 120, 60), "noisy": (150, 110, 20),
        "fault": (150, 70, 20), "dead": (140, 30, 30),
        "unknown": (70, 70, 80)}

    range_changed = QtCore.pyqtSignal(int, float)

    def __init__(self, geom):
        super().__init__(N_SENSORS, len(self.COLS))
        self.geom = geom
        self.setHorizontalHeaderLabels(self.COLS)
        for name, tip in self.COL_TIPS.items():
            self.horizontalHeaderItem(self.COLS.index(name)).setToolTip(tip)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

        self._combos = []
        for r in range(N_SENSORS):
            for c in range(len(self.COLS)):
                it = QtWidgets.QTableWidgetItem("")
                it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.setItem(r, c, it)
            self.item(r, 0).setText(f"S{r+1}")
            combo = QtWidgets.QComboBox()
            for v in sorted(ob.RANGE_TO_VPT):
                combo.addItem(f"+/-{v:g} mT", v)
            combo.setCurrentIndex(0)
            combo.currentIndexChanged.connect(
                lambda _i, row=r, cb=combo: self.range_changed.emit(
                    row, float(cb.currentData())))
            self.setCellWidget(r, self.COLS.index("range"), combo)
            self._combos.append(combo)
        self.refresh_geometry(geom)

    def refresh_geometry(self, geom):
        self.geom = geom
        for r in range(N_SENSORS):
            f = pgeom.FACE_NAMES[geom.face(r + 1)]
            self.item(r, 1).setText(f"{f}/{geom.slot(r+1)}")

    def set_ranges(self, ranges):
        for r, v in enumerate(ranges):
            cb = self._combos[r]
            i = cb.findData(float(v))
            if i >= 0 and i != cb.currentIndex():
                cb.blockSignals(True)
                cb.setCurrentIndex(i)
                cb.blockSignals(False)

    def update_rows(self, table):
        for r, row in enumerate(table):
            state = row["state"]
            self.item(r, 2).setText(state)
            self.item(r, 2).setBackground(
                QtGui.QColor(*self.STATE_COLORS.get(state, (70, 70, 80))))
            self.item(r, 2).setToolTip(row.get("note", "") or "")
            vals = [("|B| mT", f"{row['absB_mT']:.3f}"),
                    ("Bx", f"{row['Bx_mT']:.3f}"),
                    ("By", f"{row['By_mT']:.3f}"),
                    ("Bz", f"{row['Bz_mT']:.3f}"),
                    ("noise uT", f"{row['noise_uT_rms']:.0f}"),
                    ("VCM V", f"{row['vcm_v']:.4f}"),
                    ("T degC*", "--" if row["temp_c"] is None
                     else f"{row['temp_c']:.1f}"),
                    ("gain trim", f"{row['gain_trim']:.3f}")]
            for name, txt in vals:
                self.item(r, self.COLS.index(name)).setText(txt)
