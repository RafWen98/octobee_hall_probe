"""
octobee/gui/tabs/live.py -- the two surfaces that show the field as it arrives.

Live is not shaped like the other tabs, and forcing it to be would have made it
worse. It is two widgets in two places:

    LiveTab        the rolling trace plot and its controls, in the tab strip
    ProbeHeadPane  the 3D head, its controls and the peak bars, in the
                   right-hand pane -- which stays on screen whichever tab is
                   selected, because "what is the probe reading right now" is
                   the question you want answered while you are doing something
                   else

Neither owns the acquisition tick. The tick fills the session's rolling buffer
that every tab reads, so it belongs to the window rather than to whichever
widget happens to draw it, and both of these are handed their samples instead
of reaching for them. The automatic redraw backoff stays with the window too,
for the same reason: it adjusts the window's own view timer.
"""

import time

import numpy as np
from PyQt6 import QtWidgets

from octobee.gui.constants import N_SENSORS
from octobee.gui.widgets.plot import LivePlot
from octobee.gui.widgets.probe3d import ProbeView3D
from octobee.gui.widgets.sensors import SensorBars


class LiveTab(QtWidgets.QWidget):
    """The rolling trace plot: what to show, in what units, for which sensors."""

    def __init__(self, session, parent=None):
        """The rolling trace plot, and what it shows."""
        super().__init__(parent)
        self.session = session
        self.plot = LivePlot(session.geom, profiler=session.prof)
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("show"))
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(LivePlot.MODES)
        self.cmb_mode.currentTextChanged.connect(self.plot.set_mode)
        top.addWidget(self.cmb_mode)
        top.addWidget(QtWidgets.QLabel(" in "))
        self.cmb_units = QtWidgets.QComboBox()
        self.cmb_units.addItems(LivePlot.UNITS)
        self.cmb_units.setToolTip(
            "The pipeline always works in millitesla. These walk that back "
            "down the conversion chain so you can see the electrical signal "
            "itself: 'mV (chip output)' is the chip's own output with its VCM "
            "reference subtracted, and 'ADC counts' is what came off the wire. "
            "Any tare or gain trim in force is still applied.")
        self.cmb_units.currentTextChanged.connect(self.refresh_units)
        top.addWidget(self.cmb_units)
        top.addSpacing(12)
        self.chk_sensors = []
        for s in range(N_SENSORS):
            cb = QtWidgets.QCheckBox(f"{s+1}")
            cb.setChecked(True)
            cb.toggled.connect(self.on_sensor_toggle)
            self.chk_sensors.append(cb)
            top.addWidget(cb)
        btn_all = QtWidgets.QPushButton("all")
        btn_all.clicked.connect(lambda: self.set_all_sensors(True))
        btn_none = QtWidgets.QPushButton("none")
        btn_none.clicked.connect(lambda: self.set_all_sensors(False))
        top.addWidget(btn_all)
        top.addWidget(btn_none)
        top.addStretch(1)
        btn_reset = QtWidgets.QPushButton("reset view")
        btn_reset.setToolTip(
            "Put the axes back on auto after a drag or a scroll. Zooming a "
            "pyqtgraph plot silently stops it auto-ranging, so the traces "
            "stop filling the axes and it looks like the signal changed "
            "rather than the view.")
        btn_reset.clicked.connect(self.plot.reset_view)
        top.addWidget(btn_reset)
        lay.addLayout(top)
        lay.addWidget(self.plot)

    def refresh_units(self, *_):
        """
        Build the per-sensor factor that takes millitesla to the chosen unit.

        All 16 chips currently run gain 3000, so the factor is the same for
        every sensor. It stays per-sensor anyway because the range is a
        per-sensor setting and the two halves genuinely have differed -- they
        were 34.65 and 63 V/T until the 2026-08-19 harmonisation. Under a split
        gain one field is not one voltage, and a single global factor would
        quietly misreport half the probe.
        """
        name = self.cmb_units.currentText()
        vpt = self.session.cal.volts_per_tesla                     # V/T, per sensor
        if name == "mT":
            k = np.ones(N_SENSORS)
        elif name == "uT":
            k = np.full(N_SENSORS, 1e3)
        elif name.startswith("mV"):
            k = vpt                                        # mT * V/T -> mV
        else:                                              # ADC counts
            vpc = np.array([(self.session.source.vpc[0] if s < 8 else self.session.source.vpc[-1])
                            if self.session.source else 20.0 / 65536.0
                            for s in range(N_SENSORS)])
            k = vpt * 1e-3 / vpc                           # mT -> volts -> counts
        self.plot.set_units(k, {"uT": "µT"}.get(name, name))

    def on_sensor_toggle(self, _=None):
        self.plot.set_visible_sensors(
            {i for i, cb in enumerate(self.chk_sensors) if cb.isChecked()})

    def set_all_sensors(self, on):
        for cb in self.chk_sensors:
            cb.blockSignals(True)
            cb.setChecked(on)
            cb.blockSignals(False)
        self.on_sensor_toggle()

    def mark_dead(self, dead):
        for s, cb in enumerate(self.chk_sensors):
            bad = f"S{s+1}" in dead
            cb.setStyleSheet("color:#c05050;" if bad else "")
            cb.setToolTip("excluded: railed or stuck channels" if bad else "")

    def draw(self, recent):
        """Paint the traces from the samples we are handed."""
        with self.session.prof.time("  live plot setData"):
            self.plot.update_data(recent, self.session.out_rate)

    def units(self):
        return self.cmb_units.currentText()

    def set_geometry(self, geom):
        self.plot.geom = geom

    def set_dead(self, dead):
        self.plot.set_dead(dead)


class ProbeHeadPane(QtWidgets.QWidget):
    """The 3D probe head, its controls and the peak bars.

    In the right-hand pane rather than a tab: the field the probe is reading
    now is what you want on screen while you are working in one of the other
    tabs, not something to switch to.
    """

    def __init__(self, session, parent=None):
        """The head, its controls and the peak bars, in the right-hand pane."""
        super().__init__(parent)
        self.session = session
        self.view3d = ProbeView3D(session.geom, profiler=session.prof)
        self.bars = SensorBars(profiler=session.prof)
        self._draw_ms = 0.0

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        head = QtWidgets.QLabel("Probe head — colour and arrow length are |B|, "
                                "arrows are the tube-frame field direction")
        head.setWordWrap(True)
        head.setStyleSheet("color:#9aa3b2; font-size:11px;")
        lay.addWidget(head)
        lay.addWidget(self.view3d, 3)
        lay.addWidget(self._controls())
        lay.addWidget(self.bars, 1)

    def _controls(self):
        w = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        self.chk_auto = QtWidgets.QCheckBox("auto scale")
        self.chk_auto.setChecked(True)
        self.chk_auto.toggled.connect(
            lambda v: setattr(self.view3d, "auto_scale", v))
        row.addWidget(self.chk_auto)
        row.addWidget(QtWidgets.QLabel("full scale"))
        self.spin_fs = QtWidgets.QDoubleSpinBox()
        self.spin_fs.setRange(0.001, 4000.0)
        self.spin_fs.setDecimals(3)
        self.spin_fs.setValue(1.0)
        self.spin_fs.setSuffix(" mT")
        self.spin_fs.valueChanged.connect(
            lambda v: setattr(self.view3d, "full_scale_mt", v))
        row.addWidget(self.spin_fs)
        self.chk_3d = QtWidgets.QCheckBox("3D")
        self.chk_3d.setChecked(True)
        self.chk_3d.setToolTip(
            "Draw the probe head. This is the most expensive thing in the "
            "window -- untick it if the display cannot keep up. Nothing else "
            "changes: acquisition, calibration and recording are unaffected.")
        self.chk_3d.toggled.connect(self.on_3d_toggle)
        row.addWidget(self.chk_3d)
        chk_arrows = QtWidgets.QCheckBox("arrows")
        chk_arrows.setChecked(True)
        chk_arrows.toggled.connect(self.view3d.set_arrows_visible)
        row.addWidget(chk_arrows)
        chk_lbl = QtWidgets.QCheckBox("labels")
        chk_lbl.setChecked(True)
        chk_lbl.toggled.connect(self.view3d.set_labels_visible)
        row.addWidget(chk_lbl)
        self.chk_bars = QtWidgets.QCheckBox("peak bars")
        self.chk_bars.setChecked(True)
        self.chk_bars.setToolTip(
            "The bar chart underneath. Untick it to give the height back to "
            "the head and to the stage controls below — the bars answer "
            "\"are the spikes the same height\", which is not a question you "
            "are asking while driving the rig somewhere. Nothing else "
            "changes: acquisition, calibration and recording carry on, and "
            "the health check still uses every channel.")
        self.chk_bars.toggled.connect(self.on_bars_toggle)
        row.addWidget(self.chk_bars)
        btn = QtWidgets.QPushButton("reset view")
        btn.clicked.connect(self.view3d.reset_camera)
        row.addWidget(btn)
        row.addStretch(1)
        return w

    def on_3d_toggle(self, on):
        self.view3d.setVisible(bool(on))
        self._draw_ms = 0.0
        self.session.log("3D head on" if on else
                     "3D head off -- everything else carries on unchanged")

    def on_bars_toggle(self, on):
        """Hide the peak bars, and stop computing them.

        The peak is a max over half a second of the rolling buffer across all
        16 sensors, every refresh. A hidden widget still gets its setData
        honoured, so leaving the update running would give back the screen
        space and none of the time.
        """
        self.bars.setVisible(bool(on))
        self._draw_ms = 0.0
        self.session.log("peak bars on" if on else
                     "peak bars off -- the head and the stage controls take "
                     "the space; nothing else changes")

    def draw(self, recent):
        """Paint the head and the bars. Returns how long it took, in ms."""
        t0 = time.perf_counter()
        if self.chk_3d.isChecked():
            with self.session.prof.time("  3D head: build vectors"):
                k = min(8, recent.shape[0])
                fs = self.view3d.update_fields(recent[-k:].mean(axis=0))
            if (self.chk_auto.isChecked()
                    and abs(fs - self.spin_fs.value()) > 1e-4):
                self.spin_fs.blockSignals(True)
                self.spin_fs.setValue(max(fs, 0.001))
                self.spin_fs.blockSignals(False)
        if self.chk_bars.isChecked():
            with self.session.prof.time("  peak bars"):
                # Peak over a trailing half second, not the instantaneous
                # value: a magnet passed by hand is over well inside one
                # refresh, so sampling one point would miss most passes.
                n = max(2, min(recent.shape[0],
                               int(self.session.out_rate * self.bars.window_s)))
                self.bars.update_values(
                    np.linalg.norm(recent[-n:], axis=-1).max(axis=0),
                    self.session.cal.dead)
        return (time.perf_counter() - t0) * 1000.0

    def gl_info(self):
        """The live OpenGL context's info, or None while the head is hidden."""
        return self.view3d.gl_info() if self.view3d.isVisible() else None

    def set_geometry(self, geom):
        self.view3d.rebuild(geom)

    def set_dead(self, dead):
        self.view3d.set_dead(dead)

    def reset_scale(self):
        self.view3d.reset_scale()
