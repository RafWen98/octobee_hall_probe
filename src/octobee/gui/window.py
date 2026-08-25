#!/usr/bin/env python3
"""
octobee/gui/window.py -- control, live view, calibration and data export for the
16-sensor OCTO-BEE Hall probe, in one window.

    python octobee/gui/window.py                       # talk to the two carriers
    python octobee/gui/window.py --demo                # synthetic probe, no hardware
    python octobee/gui/window.py --replay capture.npz  # play back a saved capture

What it is for
--------------
The probe is 16 three-axis chips on a square tube, split across two carriers
that are configured differently and are not synchronised. Reading it through
Phoebus means one box at a time, PV names typed by hand, and a CSV round trip.
This does the whole job in one place: connect, watch, zero, cross-calibrate,
and write the data out.

The three things it insists on, because each of them has silently corrupted a
measurement on this bench already:

  * every sensor's own VCM is subtracted from that sensor's own axes;
  * amplitudes are compared as |B|, never as a single axis, because the chips
    point in 16 different directions;
  * per-channel health (railed, stuck, noisy) is always on screen, because one
    sensor is physically dead and the analogue path is noisier at one end of
    the concentrator than the other.
"""

import contextlib
import os
import time
from collections import deque

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from octobee.acq import carrier as ob
from octobee.calib import convert as ocal
from octobee import live as olive
from octobee.calib import roll as opc
from octobee import record as orec
from octobee.motion import scan as oscan
from octobee.motion import stage as ostage
from octobee.calib import geometry as pgeom
from octobee.gui.widgets.probe3d import ProbeView3D
from octobee.gui.estop import MotionControl, is_running
from octobee.gui.constants import (
    DEFAULT_VIEW_HZ,
    HEALTH_PERIOD_S,
    HEALTH_WINDOW_S,
    MAX_VIEW_INTERVAL_MS,
    N_SENSORS,
    OUT_RATES,
    RAW_HISTORY_S,
    STREAM_RATES,
    VIEW_RATES,
)
from octobee.gui.dialogs.magnet import MagnetWizard
from octobee.gui.rolling import Rolling
from octobee.gui.session import Session
from octobee.gui.tabs.export import ExportTab
from octobee.gui.tabs.health import HealthTab
from octobee.gui.tabs.help import HelpTab
from octobee.gui.tabs.machine import MachineTab
from octobee.gui.tabs.profile import ProfileTab
from octobee.gui.sources import DemoSource, LiveSource, ReplaySource
from octobee.gui.widgets.log import LogPane
from octobee.gui.widgets.plot import LivePlot
from octobee.gui.widgets.sensors import SensorBars, SensorTable
from octobee.gui.workers import (
    ConnectWorker,
    ScanWorker,
    SnapshotWorker,
    StageWorker,
    _stage_set_for,
)

# ==========================================================================
# main window
# ==========================================================================

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, args):
        super().__init__()
        self.session = Session(args)

        # ---- this window's own state ---------------------------------------
        # Everything below belongs to the window or to exactly one of its tabs.
        # What more than one tab reads lives on the session above.
        self.window_s = 20.0
        self.session.roll = Rolling(int(self.session.out_rate * self.window_s))
        self.raw_hist = None
        self.raw_hist_n = 0
        self.paused = False
        self._draw_ms = 0.0
        self._last_table = None
        self._last_health_t = 0.0
        self._last_dropped = 0
        self._machine_key = None
        self._machine_quiet = False
        self._machine_travel_taken = False
        self._connect_worker = None
        self._snap_worker = None
        self._connect_was_automatic = False
        self._connecting = False
        self.stage_combos = {}
        self._stage_pending = None
        self._stage_worker = None
        self._magnet_wizard = None
        self._scan_worker = None
        self._scan_t0 = 0.0

        # Every thread that can command motion, and the latch that
        # overrides all of them. Built before _build_ui: the toolbar's
        # stop button and the Esc shortcut are live from the moment the
        # window exists, and both go through this.
        self.motion = MotionControl(self.session, parent=self)
        self.motion.changed.connect(self._refresh_estop_ui)
        self._state_before_estop = None

        self.setWindowTitle("OCTO-BEE Hall probe")
        self.resize(1720, 980)
        self._build_ui()
        self._apply_dark()

        # Three clocks, fastest first: acquisition must never be blocked by
        # drawing, drawing should be smooth but is the user's to trade away,
        # and the diagnostics are slow and expensive.
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(50)
        self.view_timer = QtCore.QTimer(self)
        self.view_timer.timeout.connect(self.on_view_tick)
        self.view_timer.start(int(1000 / DEFAULT_VIEW_HZ))
        self.slow = QtCore.QTimer(self)
        self.slow.timeout.connect(self.on_slow_tick)
        self.slow.start(1000)
        # Asks for exactly 100 ms and records how late it actually gets it.
        # That difference is the honest measure of "is the app blocked".
        self.lag_timer = QtCore.QTimer(self)
        self.lag_timer.timeout.connect(self.session.lag.tick)
        self.lag_timer.start(100)
        self.prof_timer = QtCore.QTimer(self)
        self.prof_timer.timeout.connect(self.tab_profile.refresh)
        self.prof_timer.start(1000)
        # The stage positions come out of the Kinesis DLL's own polling
        # cache, so this reads memory rather than talking to USB -- it
        # can run fast enough to make jogging feel direct without
        # competing with acquisition.
        self.stage_timer = QtCore.QTimer(self)
        self.stage_timer.timeout.connect(self.refresh_stage_table)
        # Same clock as the stage table, and for the same reason: what the
        # machine view draws is where the stages say the head is, so the two
        # must not be able to disagree on screen.
        self.stage_timer.timeout.connect(self.tab_machine.refresh_machine)
        self.stage_timer.start(200)

        for msg in self.session.config_errors:
            self.session.log(f"WARNING: {msg}")

        self.tab_machine.announce()

        if args.demo:
            self._set_source(DemoSource(self.session.geom), "demo")
        elif args.replay:
            self._set_source(ReplaySource(args.replay), f"replay {args.replay}")
        elif getattr(args, "no_connect", False):
            self.session.log("ready -- press Connect to take over the stream from "
                         "Phoebus and start reading both carriers")
        else:
            # Deferred rather than called here: connecting wants a running
            # event loop to report progress into, and __init__ is finishing
            # inside show(). A few hundred milliseconds also means the window
            # is painted before the first "connecting..." line arrives, so a
            # slow carrier looks like a slow connect rather than a slow start.
            self.session.log("connecting automatically -- run with --no-connect "
                         "to start disconnected")
            QtCore.QTimer.singleShot(300, self._auto_connect)
        self._report_calibration_source()

    def _magnet_geometry_correction(self):
        """The magnet-pass geometry correction, or None if it is not wanted.

        Handed to the Data output tab so a report can carry it. The controls
        belong to the Calibration tab; what a report needs is the answer.
        """
        if not self.chk_geom.isChecked():
            return None
        return ([self.spin_mx.value(), self.spin_my.value(),
                 self.spin_mz.value()], self.spin_exp.value())

    def _report_calibration_source(self):
        if self.session.cal_from_file:
            self.session.log(f"calibration loaded from {self.session.args.calibration}: "
                         + self.session.cal.summary().replace("\n", "; "))
            if self.session.cal.notes:
                first = self.session.cal.notes.split(". ")[0]
                self.session.log(f"  {first}. (full note in "
                             f"{self.session.args.calibration})")
        else:
            self.session.log(
                f"no {self.session.args.calibration} found -- using built-in defaults, "
                f"which put every sensor on +/-20 mT / 63 V/T. That matches "
                f"the probe as audited on 2026-08-19, when all 16 chips were "
                f"harmonised to gain 3000, but nothing here checks it at run "
                f"time. If the gain has been changed since, set the range per "
                f"sensor in the Sensors tab and save the calibration -- a "
                f"wrong range is invisible on screen and simply rescales "
                f"everything.")

    # ---- construction ----------------------------------------------------
    def _build_ui(self):
        self._build_toolbar()
        self.log_pane = LogPane()
        # Everything that wants to say what it did goes through the
        # session from here on; nothing else needs the widget.
        self.session.attach_log(self.log_pane)

        self.view3d = ProbeView3D(self.session.geom, profiler=self.session.prof)
        self.bars = SensorBars(profiler=self.session.prof)
        self.plot = LivePlot(self.session.geom, profiler=self.session.prof)
        self.table = SensorTable(self.session.geom)
        self.table.range_changed.connect(self.on_range_changed)
        self.table.set_ranges(self.session.cal.ranges_mt)

        left = QtWidgets.QTabWidget()
        left.addTab(self._live_tab(), "Live")
        left.addTab(self.table, "Sensors")
        left.addTab(self._calib_tab(), "Calibration")
        self.tab_health = HealthTab(self.session, self._raw_arrays)
        self.tab_health.wants_focus.connect(
            lambda: self.tabs.setCurrentWidget(self.tab_health))
        left.addTab(self.tab_health, "Diagnostics")
        left.addTab(self._stage_tab(), "Stages")
        self.tab_machine = MachineTab(self.session)
        left.addTab(self.tab_machine, "Machine")
        self.tab_export = ExportTab(
            self.session,
            sensor_table=lambda: self._last_table,
            magnet_geometry=self._magnet_geometry_correction)
        self.tab_export.snapshot_requested.connect(self.on_snapshot)
        self.tab_health.exported.connect(self.tab_export.note)
        left.addTab(self.tab_export, "Data output")
        self.tab_profile = ProfileTab(
            self.session,
            gl_info=lambda: (self.view3d.gl_info()
                             if self.view3d.isVisible() else None))
        left.addTab(self.tab_profile, "Profile")
        left.addTab(self.log_pane, "Log")
        self.tab_help = HelpTab(self.session)
        left.addTab(self.tab_help, "Help")
        left.setCurrentIndex(0)
        self.tabs = left

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)
        head = QtWidgets.QLabel("Probe head — colour and arrow length are |B|, "
                                "arrows are the tube-frame field direction")
        head.setWordWrap(True)
        head.setStyleSheet("color:#9aa3b2; font-size:11px;")
        rl.addWidget(head)
        rl.addWidget(self.view3d, 3)
        rl.addWidget(self._view3d_controls())
        rl.addWidget(self.bars, 1)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([1000, 720])
        self.setCentralWidget(split)

        self.status = self.statusBar()
        self.lbl_state = QtWidgets.QLabel("disconnected")
        self.lbl_rate = QtWidgets.QLabel("")
        self.lbl_rec = QtWidgets.QLabel("")
        for w in (self.lbl_state, self.lbl_rate, self.lbl_rec):
            self.status.addPermanentWidget(w)

    def _build_toolbar(self):
        tb = QtWidgets.QToolBar("main")
        tb.setIconSize(QtCore.QSize(16, 16))
        tb.setMovable(False)
        self.addToolBar(tb)

        self.act_connect = QtGui.QAction("Connect", self)
        self.act_connect.triggered.connect(self.on_connect)
        tb.addAction(self.act_connect)
        self.act_disconnect = QtGui.QAction("Disconnect", self)
        self.act_disconnect.triggered.connect(self.on_disconnect)
        self.act_disconnect.setEnabled(False)
        tb.addAction(self.act_disconnect)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel(" stream rate "))
        self.cmb_rate = QtWidgets.QComboBox()
        for label, fs in STREAM_RATES.items():
            self.cmb_rate.addItem(label, fs)
        self.cmb_rate.setToolTip(
            "Each carrier produces 19.2 MB/s at 200 kSPS but its stream path "
            "delivers only ~10-15 MB/s (the Ethernet is 1 Gbps and not the "
            "limit -- reading localhost:4210 on the box itself tops out at "
            "15 MB/s), so a sustained full-rate stream falls behind.\n\n"
            "Lowering the clock is a trade, not a free win: the sensor's analog "
            "low-pass is fixed at 100 kHz, so anything below 200 kSPS aliases "
            "and the noise density rises by sqrt(100kHz/(fs/2)) -- 2x at "
            "50 kSPS, 3.2x at 20 kSPS (measured 3.1x).\n\n"
            "For short captures prefer 200 kSPS: the box buffers and delivers "
            "every sample, just slower than real time (a 3 s capture from both "
            "boxes came back with zero lost samples). The original clkdiv is "
            "restored on disconnect.")
        tb.addWidget(self.cmb_rate)

        tb.addWidget(QtWidgets.QLabel("  output rate "))
        self.cmb_out = QtWidgets.QComboBox()
        for r in OUT_RATES:
            self.cmb_out.addItem(f"{r:g} Hz", r)
        self.cmb_out.setCurrentIndex(list(OUT_RATES).index(500.0))
        self.cmb_out.currentIndexChanged.connect(self.on_out_rate)
        tb.addWidget(self.cmb_out)

        tb.addWidget(QtWidgets.QLabel("  refresh "))
        self.cmb_view = QtWidgets.QComboBox()
        for r in VIEW_RATES:
            self.cmb_view.addItem(f"{r:g} Hz", r)
        self.cmb_view.setCurrentIndex(list(VIEW_RATES).index(DEFAULT_VIEW_HZ))
        self.cmb_view.setToolTip(
            "How often the plot, the bars and the 3D head are redrawn. This is "
            "purely cosmetic: acquisition, recording and the calibration all "
            "run on their own clock, so turning it down costs smoothness and "
            "nothing else. Drop it to 2 Hz if the window feels heavy.")
        self.cmb_view.currentIndexChanged.connect(self.on_view_rate)
        # 'activated' as well as 'currentIndexChanged': after an automatic
        # backoff the combo still reads the rate you asked for, so re-picking
        # that same entry is exactly how you would expect to restore it -- and
        # currentIndexChanged stays silent when the index has not moved.
        self.cmb_view.activated.connect(self.on_view_rate)
        tb.addWidget(self.cmb_view)

        self.act_pause = QtGui.QAction("Pause view", self)
        self.act_pause.setCheckable(True)
        self.act_pause.setToolTip(
            "Stop redrawing entirely. Acquisition and recording carry on -- "
            "use this while recording if you want every cycle going to the data.")
        self.act_pause.toggled.connect(self.on_pause)
        tb.addAction(self.act_pause)

        tb.addWidget(QtWidgets.QLabel("  window "))
        self.spin_window = QtWidgets.QDoubleSpinBox()
        self.spin_window.setRange(1.0, 120.0)
        self.spin_window.setValue(self.window_s)
        self.spin_window.setSuffix(" s")
        self.spin_window.valueChanged.connect(self.on_window)
        tb.addWidget(self.spin_window)
        tb.addSeparator()

        self.act_tare = QtGui.QAction("Zero (tare)", self)
        self.act_tare.setToolTip("Take 2 s of ambient data and store it as the "
                                 "zero point of every axis of every sensor.")
        self.act_tare.triggered.connect(lambda: self.start_collect("tare", 2.0))
        tb.addAction(self.act_tare)

        self.act_record = QtGui.QAction("Record", self)
        self.act_record.setCheckable(True)
        self.act_record.toggled.connect(self.on_record)
        tb.addAction(self.act_record)

        self.act_snapshot = QtGui.QAction("Snapshot", self)
        self.act_snapshot.setToolTip("Pause streaming and take a lossless "
                                     "full-rate capture to .npz.")
        self.act_snapshot.triggered.connect(self.on_snapshot)
        tb.addAction(self.act_snapshot)

        self._build_estop(tb)

    def _build_estop(self, tb):
        """The stop button, top right, always there and always live.

        Everything about this is deliberate.

        WHERE. Top right of the toolbar, pushed there by an expanding spacer
        so it stays in the corner at any window width. It is outside the tab
        stack because the tab stack is exactly the problem: the STOP that used
        to be the only one lived on the Stages tab, which meant that while the
        head was traversing and you were watching the live plot -- the normal
        way to run a scan -- the stop button was on a page you could not see.
        A stop you have to navigate to is not a stop.

        ALWAYS ENABLED. It is not greyed out when the stages are disconnected,
        because "the stages are disconnected" is this process's belief, and if
        that belief were reliable there would be nothing to stop. It also
        aborts the scan and the wizard, which can be running when `stages` is
        in any state at all.

        KEY. Escape, application-wide, so it works while the guided-magnet
        window has focus. Esc is what a person hits, and it costs nothing to
        honour it. Note the limit: a MODAL dialog eats its own Esc, so while
        one of the confirmation boxes is up the key goes to the box. Those all
        appear before motion starts rather than during it, and the button is
        still there behind them.

        RESET IS SEPARATE. The stop button never becomes the start button.
        Pressing stop twice must not be a way to release the machine, and a
        person reaching for a stop button in a hurry may well hit it twice.
        """
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                             QtWidgets.QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        self.btn_estop = QtWidgets.QPushButton("■  EMERGENCY STOP")
        self.btn_estop.setStyleSheet(
            "QPushButton { background:#c1121f; color:#fff; font-weight:bold; "
            "font-size:13px; padding:6px 18px; border:2px solid #7a0b12; "
            "border-radius:4px; }"
            "QPushButton:hover { background:#e01e2a; }"
            "QPushButton:pressed { background:#7a0b12; }")
        self.btn_estop.setToolTip(
            "Stop every axis immediately and refuse all further motion until "
            "it is reset (Esc).\n\n"
            "Immediate, not profiled: the deceleration ramp is abandoned, so "
            "steps can be lost and every axis needs re-homing before it will "
            "accept an absolute move again. That is the trade — 1.8 mm of "
            "coasting is what a profiled stop costs at the default profile.\n\n"
            "This is a software stop over USB. It needs this program, the USB "
            "link and the controllers all working. It is not a substitute for "
            "a hardware emergency stop in series with the motor supply.")
        self.btn_estop.clicked.connect(self.on_estop)
        tb.addWidget(self.btn_estop)

        self.btn_estop_reset = QtWidgets.QPushButton("Reset")
        self.btn_estop_reset.setToolTip(
            "Clear the latch and allow motion again. Deliberately a separate "
            "button: releasing a machine is an act, not the absence of one.")
        self.btn_estop_reset.setVisible(False)
        self.btn_estop_reset.clicked.connect(self.motion.reset)
        tb.addWidget(self.btn_estop_reset)

        self.sc_estop = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self)
        self.sc_estop.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self.sc_estop.activated.connect(self.on_estop)

    def _view3d_controls(self):
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
        btn = QtWidgets.QPushButton("reset view")
        btn.clicked.connect(self.view3d.reset_camera)
        row.addWidget(btn)
        row.addStretch(1)
        return w

    def _live_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
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
        self.cmb_units.currentTextChanged.connect(self.on_units)
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
        btn_all.clicked.connect(lambda: self._set_all_sensors(True))
        btn_none = QtWidgets.QPushButton("none")
        btn_none.clicked.connect(lambda: self._set_all_sensors(False))
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
        return w

    def _calib_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        g1 = QtWidgets.QGroupBox("1. Zero point")
        f1 = QtWidgets.QHBoxLayout(g1)
        self.spin_tare_s = QtWidgets.QDoubleSpinBox()
        self.spin_tare_s.setRange(0.2, 30.0)
        self.spin_tare_s.setValue(2.0)
        self.spin_tare_s.setSuffix(" s")
        b_tare = QtWidgets.QPushButton("Take zero from ambient")
        b_tare.clicked.connect(
            lambda: self.start_collect("tare", self.spin_tare_s.value()))
        b_clear = QtWidgets.QPushButton("Clear zero")
        b_clear.clicked.connect(self.on_clear_tare)
        f1.addWidget(QtWidgets.QLabel("average over"))
        f1.addWidget(self.spin_tare_s)
        f1.addWidget(b_tare)
        f1.addWidget(b_clear)
        f1.addStretch(1)
        lay.addWidget(g1)

        g_guided = QtWidgets.QGroupBox(
            "2. Guided magnet calibration — measure and equalise response")
        fg = QtWidgets.QVBoxLayout(g_guided)
        blurb = QtWidgets.QLabel(
            "Clamp one magnet, then drive the HEAD past it under motor "
            "control, a quarter turn at a time. Each pose sweeps the plane the "
            "sensors lie in and dithers the standoff, so every sensor is "
            "measured at the top of its own peak and at a measured distance — "
            "no 1/r³ model, and a millimetre of arm placement no longer looks "
            "like 15 % of gain. The run also reports which sensor is really on "
            "which face.")
        blurb.setWordWrap(True)
        fg.addWidget(blurb)
        self.btn_guided = QtWidgets.QPushButton(
            "Guided magnet calibration (motorised, 4 poses)…")
        self.btn_guided.clicked.connect(self.on_guided_magnet)
        row_g = QtWidgets.QHBoxLayout()
        row_g.addWidget(self.btn_guided)
        b_cleargain2 = QtWidgets.QPushButton("Clear gain trim")
        b_cleargain2.clicked.connect(self.on_clear_gain)
        row_g.addWidget(b_cleargain2)
        row_g.addStretch(1)
        fg.addLayout(row_g)
        lay.addWidget(g_guided)

        # ---- superseded routines, built but not shown -----------------------
        # Both of these are replaced by the guided run above and are hidden by
        # default. They are still CONSTRUCTED, and that is deliberate: the
        # calibration report, the export and the collector all reach into these
        # widgets, so deleting them would mean chasing those references for two
        # routines that are still occasionally worth a look. Hidden, not
        # removed -- and the checkbox says why rather than just offering a
        # toggle.
        self.chk_superseded = QtWidgets.QCheckBox(
            "show the superseded manual routines (hand magnet pass, "
            "Earth-field roll)")
        self.chk_superseded.setChecked(False)
        self.chk_superseded.setToolTip(
            "Both were ways of getting at what the guided run above now "
            "measures directly.\n\n"
            "The hand magnet pass holds a magnet near the probe by hand: every "
            "chip is then at a different distance, so it needs a 1/r^n model "
            "of the geometry to divide that out -- and the geometry is exactly "
            "what is not yet established.\n\n"
            "The Earth-field roll sweep was the way round that: a uniform "
            "field every sensor must read identically. It still does something "
            "the magnet run does not -- it pins chip ORIENTATION in three "
            "dimensions -- but for gain it has been superseded.\n\n"
            "Nothing recorded with either is lost by leaving this unticked.")
        self.chk_superseded.toggled.connect(self.on_show_superseded)
        lay.addWidget(self.chk_superseded)

        g2 = QtWidgets.QGroupBox("Superseded: hand magnet pass")
        f2 = QtWidgets.QGridLayout(g2)
        self.btn_magnet = QtWidgets.QPushButton("Start magnet pass")
        self.btn_magnet.setCheckable(True)
        self.btn_magnet.toggled.connect(self.on_magnet_pass)
        f2.addWidget(self.btn_magnet, 0, 0)
        self.lbl_magnet = QtWidgets.QLabel("no pass recorded")
        f2.addWidget(self.lbl_magnet, 0, 1, 1, 4)

        f2.addWidget(QtWidgets.QLabel("magnet position in the tube frame (mm), "
                                      "for dividing out 1/r^n distance:"), 1, 0, 1, 5)
        self.spin_mx = QtWidgets.QDoubleSpinBox()
        self.spin_my = QtWidgets.QDoubleSpinBox()
        self.spin_mz = QtWidgets.QDoubleSpinBox()
        for sp, v in ((self.spin_mx, 60.0), (self.spin_my, 0.0),
                      (self.spin_mz, 100.0)):
            sp.setRange(-1000.0, 1000.0)
            sp.setValue(v)
            sp.setSuffix(" mm")
        self.spin_exp = QtWidgets.QDoubleSpinBox()
        self.spin_exp.setRange(1.0, 4.0)
        self.spin_exp.setValue(3.0)
        self.spin_exp.setSingleStep(0.1)
        self.spin_exp.setPrefix("1/r^")
        self.chk_geom = QtWidgets.QCheckBox("use geometry weighting")
        self.chk_geom.setChecked(False)
        self.chk_geom.setToolTip(
            "A magnet at a fixed distance from the TUBE is at a different "
            "distance from each CHIP. Tick this and give the magnet position "
            "to divide that out, so what is left is electrical.")
        pos_row = QtWidgets.QWidget()
        pl = QtWidgets.QHBoxLayout(pos_row)
        pl.setContentsMargins(0, 0, 0, 0)
        for lbl, sp in (("x", self.spin_mx), ("y", self.spin_my),
                        ("z", self.spin_mz)):
            pl.addWidget(QtWidgets.QLabel(lbl))
            pl.addWidget(sp)
        pl.addSpacing(12)
        pl.addWidget(self.spin_exp)
        pl.addWidget(self.chk_geom)
        pl.addStretch(1)
        f2.addWidget(pos_row, 2, 0, 1, 5)
        superseded = QtWidgets.QLabel(
            "Superseded by the guided run above, which needs no 1/r^n model "
            "because it measures each sensor at its own peak and at a measured "
            "distance. Kept for comparison — a hand pass is still the quickest "
            "way to see whether all sixteen channels are alive.")
        superseded.setWordWrap(True)
        f2.addWidget(superseded, 4, 0, 1, 5)

        self.btn_apply_gain = QtWidgets.QPushButton("Apply gain trim from this pass")
        self.btn_apply_gain.setEnabled(False)
        self.btn_apply_gain.clicked.connect(self.on_apply_gain)
        b_cleargain = QtWidgets.QPushButton("Clear gain trim")
        b_cleargain.clicked.connect(self.on_clear_gain)
        btn_row = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(btn_row)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.btn_apply_gain)
        bl.addWidget(b_cleargain)
        bl.addStretch(1)
        f2.addWidget(btn_row, 6, 0, 1, 5)
        lay.addWidget(g2)


        g3 = QtWidgets.QGroupBox(
            "Superseded: Earth-field roll calibration")
        f3 = QtWidgets.QGridLayout(g3)
        blurb = QtWidgets.QLabel(
            "The Earth's field is uniform to nanotesla across the whole head, so "
            "every sensor must read the SAME vector. This was the way round the "
            "1/r³ position error of the hand magnet pass; the guided run now "
            "measures that position error instead of dodging it. What this "
            "still does and the magnet run does not is pin chip ORIENTATION in "
            "three dimensions.\n"
            "Roll the tube steadily in its cradle through ≥2 turns per sweep.  "
            "A: as mounted.  B: lifted out and replaced end-for-end (separates "
            "offset from axial response).  C: cradle turned to another azimuth "
            "(pins transverse-vs-axial — the flip alone cannot).")
        blurb.setWordWrap(True)
        f3.addWidget(blurb, 0, 0, 1, 6)

        self.spin_sweep_s = QtWidgets.QDoubleSpinBox()
        self.spin_sweep_s.setRange(5.0, 300.0)
        self.spin_sweep_s.setValue(60.0)
        self.spin_sweep_s.setSuffix(" s")
        f3.addWidget(QtWidgets.QLabel("sweep length"), 1, 0)
        f3.addWidget(self.spin_sweep_s, 1, 1)
        col = 2
        for tag, tip in (("A", "as mounted"),
                         ("B", "tube lifted out and replaced end-for-end"),
                         ("C", "cradle turned to another azimuth")):
            b = QtWidgets.QPushButton(f"Record sweep {tag}")
            b.setToolTip(tip)
            b.clicked.connect(lambda _, t=tag: self.start_sweep(
                t, self.spin_sweep_s.value()))
            f3.addWidget(b, 1, col)
            col += 1

        self.lbl_sweeps = QtWidgets.QLabel("no sweeps recorded")
        f3.addWidget(self.lbl_sweeps, 2, 0, 1, 6)

        self.spin_bearth = QtWidgets.QDoubleSpinBox()
        self.spin_bearth.setRange(20.0, 70.0)
        self.spin_bearth.setValue(opc.DEFAULT_B_EARTH_UT)
        self.spin_bearth.setSuffix(" uT")
        self.spin_bearth.setDecimals(2)
        self.spin_bearth.setToolTip(
            "Total field at your location, from ngdc.noaa.gov/geomag or BGS. "
            "This sets ABSOLUTE scale only — matching, offsets and "
            "orientation are all solved without it.")
        f3.addWidget(QtWidgets.QLabel("|B| here"), 3, 0)
        f3.addWidget(self.spin_bearth, 3, 1)
        self.chk_isotropic = QtWidgets.QCheckBox("assume the average chip is isotropic")
        self.chk_isotropic.setToolTip(
            "Only used when no second azimuth was recorded. Fixes the "
            "transverse-vs-axial gauge by assuming the median chip has equal "
            "sensitivity on all three axes. Fair for a monolithic part, but an "
            "assumption — record sweep C to measure it instead.")
        f3.addWidget(self.chk_isotropic, 3, 2, 1, 3)

        self.btn_solve_roll = QtWidgets.QPushButton("Solve")
        self.btn_solve_roll.setEnabled(False)
        self.btn_solve_roll.clicked.connect(self.on_solve_roll)
        self.btn_apply_roll = QtWidgets.QPushButton("Apply to calibration")
        self.btn_apply_roll.setEnabled(False)
        self.btn_apply_roll.clicked.connect(self.on_apply_roll)
        row = QtWidgets.QWidget()
        rl = QtWidgets.QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.btn_solve_roll)
        rl.addWidget(self.btn_apply_roll)
        for text, slot in (("Clear sweeps", self.on_clear_sweeps),
                           ("Save sweeps", self.on_save_sweeps),
                           ("Load sweeps", self.on_load_sweeps)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            rl.addWidget(b)
        rl.addStretch(1)
        f3.addWidget(row, 4, 0, 1, 6)
        lay.addWidget(g3)

        self._superseded_boxes = (g2, g3)
        for box in self._superseded_boxes:
            box.setVisible(False)

        g4 = QtWidgets.QGroupBox("3. Calibration file and geometry")
        f4 = QtWidgets.QHBoxLayout(g4)
        for text, slot in (("Save calibration", self.on_save_cal),
                           ("Load calibration", self.on_load_cal),
                           ("Edit geometry", self.on_edit_geometry),
                           ("Reload geometry", self.on_reload_geometry)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            f4.addWidget(b)
        self.chk_vcm = QtWidgets.QCheckBox("subtract VCM")
        self.chk_vcm.setChecked(self.session.cal.subtract_vcm)
        self.chk_vcm.setToolTip(
            "Each chip's own virtual ground. The 16 differ by up to ~90 mV, "
            "which is ~1.4 mT of fake field at the 20 mT range. Leave this on.")
        self.chk_vcm.toggled.connect(self.on_vcm_toggle)
        f4.addWidget(self.chk_vcm)
        f4.addStretch(1)
        lay.addWidget(g4)

        self.cal_report = QtWidgets.QPlainTextEdit()
        self.cal_report.setReadOnly(True)
        self.cal_report.setFont(QtGui.QFont("Consolas", 9))
        lay.addWidget(self.cal_report, 1)
        self.refresh_cal_report()
        return w

    # ---- stages ----------------------------------------------------------
    def _stage_tab(self):
        """Thorlabs LTS300C control and motorised field mapping.

        Laid out in the order you have to do things: find the stages, tell the
        software which one is which axis, reference them, then scan. Homing sits
        behind a confirmation because it drives the carriage into a limit switch
        at speed and the software has no idea what is bolted to it.
        """
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        g1 = QtWidgets.QGroupBox("1. Stages")
        f1 = QtWidgets.QGridLayout(g1)
        note = QtWidgets.QLabel(
            "The stages are exclusive-open USB devices: if the Thorlabs "
            "Kinesis application is running it owns all of them and none will "
            "appear here. Close Kinesis first.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#9aa3b2; font-size:11px;")
        f1.addWidget(note, 0, 0, 1, 4)

        self.stage_table = QtWidgets.QTableWidget(0, 5)
        self.stage_table.setHorizontalHeaderLabels(
            ["serial", "model", "axis", "position", "state"])
        self.stage_table.verticalHeader().setVisible(False)
        self.stage_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.stage_table.horizontalHeader().setStretchLastSection(True)
        self.stage_table.setMaximumHeight(140)
        f1.addWidget(self.stage_table, 1, 0, 1, 4)

        self.btn_stage_scan = QtWidgets.QPushButton("Find stages")
        # Not inlinable: clicked() would pass its `checked` bool straight into
        # `quiet`, and a button press must never be the silent variant.
        self.btn_stage_scan.clicked.connect(
            lambda: self.on_stage_find())               # noqa: PLW0108
        self.btn_stage_connect = QtWidgets.QPushButton("Connect")
        self.btn_stage_connect.clicked.connect(self.on_stage_connect)
        self.btn_stage_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_stage_disconnect.clicked.connect(self.on_stage_disconnect)
        self.btn_stage_disconnect.setEnabled(False)
        self.btn_stage_savemap = QtWidgets.QPushButton("Save axis map")
        self.btn_stage_savemap.setToolTip(
            "Records which serial number is which axis in stages.json, so the "
            "next session does not have to be told again.")
        self.btn_stage_savemap.clicked.connect(self.on_stage_save_map)
        f1.addWidget(self.btn_stage_scan, 2, 0)
        f1.addWidget(self.btn_stage_connect, 2, 1)
        f1.addWidget(self.btn_stage_disconnect, 2, 2)
        f1.addWidget(self.btn_stage_savemap, 2, 3)
        lay.addWidget(g1)

        g2 = QtWidgets.QGroupBox("2. Manual control")
        f2 = QtWidgets.QGridLayout(g2)
        f2.addWidget(QtWidgets.QLabel("jog step"), 0, 1)
        f2.addWidget(QtWidgets.QLabel("move to"), 0, 4)
        self.stage_rows = {}
        for r, ax in enumerate(("x", "y", "z"), start=1):
            lbl = QtWidgets.QLabel(ax)
            step = QtWidgets.QDoubleSpinBox()
            step.setRange(0.001, 100.0)
            step.setDecimals(3)
            step.setValue(1.0)
            step.setSuffix(" mm")
            minus = QtWidgets.QPushButton("−")
            plus = QtWidgets.QPushButton("+")
            minus.setMaximumWidth(36)
            plus.setMaximumWidth(36)
            target = QtWidgets.QDoubleSpinBox()
            target.setRange(0.0, 300.0)
            target.setDecimals(3)
            target.setSuffix(" mm")
            go = QtWidgets.QPushButton("Go")
            home = QtWidgets.QPushButton("Home")
            home.setToolTip("Drives this axis into its limit switch. Make sure "
                            "the probe and its cabling are clear first.")
            minus.clicked.connect(
                lambda _, a=ax, s=step: self.on_stage_jog(a, -s.value()))
            plus.clicked.connect(
                lambda _, a=ax, s=step: self.on_stage_jog(a, +s.value()))
            go.clicked.connect(
                lambda _, a=ax, t=target: self.on_stage_goto(a, t.value()))
            home.clicked.connect(lambda _, a=ax: self.on_stage_home([a]))
            for c, wdg in enumerate((lbl, step, minus, plus, target, go, home)):
                f2.addWidget(wdg, r, c)
            self.stage_rows[ax] = {"step": step, "target": target,
                                   "present": False,
                                   "widgets": (lbl, step, minus, plus, target,
                                               go, home)}

        # A jog only reaches full speed if it is long enough to ramp up to it,
        # so the same setting is quiet at 2 mm and loud at 20. This is where
        # that ceiling lives, next to the buttons that run into it.
        f2.addWidget(QtWidgets.QLabel("speed"), 4, 0)
        self.spin_stage_vel = QtWidgets.QDoubleSpinBox()
        # The box cannot ask for more than the module will do, so the number on
        # screen is always the number that will be used.
        self.spin_stage_vel.setRange(0.1, ostage.MAX_VEL_MM_S)
        self.spin_stage_vel.setDecimals(2)
        self.spin_stage_vel.setValue(ostage.DEFAULT_VEL_MM_S)
        self.spin_stage_vel.setSuffix(" mm/s")
        self.spin_stage_vel.setToolTip(
            "Top speed any move is allowed to reach. Kinesis ships these "
            "stages at 20 mm/s, which puts anything past about a 5 mm step "
            "into the motor's resonance -- that is the noise. Lower this, not "
            f"the step size. Hard ceiling {ostage.MAX_VEL_MM_S:g} mm/s: every "
            "move re-applies this profile first, so nothing that touched the "
            "controller in between can make a move faster than this.")
        self.spin_stage_acc = QtWidgets.QDoubleSpinBox()
        self.spin_stage_acc.setRange(0.1, 50.0)
        self.spin_stage_acc.setDecimals(2)
        self.spin_stage_acc.setValue(ostage.DEFAULT_ACCEL_MM_S2)
        self.spin_stage_acc.setSuffix(" mm/s²")
        self.spin_stage_acc.setToolTip(
            "How hard it ramps. Also sets what a SHORT move peaks at: a move "
            "too brief to reach the speed cap tops out at sqrt(accel × "
            "distance) instead.")
        self.btn_stage_speed = QtWidgets.QPushButton("Apply")
        self.btn_stage_speed.setToolTip(
            "Send this profile to every connected axis and record it in "
            "stages.json, so later sessions and the field map use it too. "
            "Homing has its own speed and is not affected.")
        self.btn_stage_speed.clicked.connect(self.on_stage_speed)
        f2.addWidget(self.spin_stage_vel, 4, 1)
        f2.addWidget(self.spin_stage_acc, 4, 2)
        f2.addWidget(self.btn_stage_speed, 4, 3)
        self.lbl_stage_peak = QtWidgets.QLabel("")
        self.lbl_stage_peak.setStyleSheet("color:#9aa3b2;")
        f2.addWidget(self.lbl_stage_peak, 4, 4, 1, 3)
        for sp in (self.spin_stage_vel, self.spin_stage_acc):
            sp.valueChanged.connect(self._update_peak_label)
        self._update_peak_label()

        self.btn_stage_homeall = QtWidgets.QPushButton("Home all axes")
        self.btn_stage_homeall.clicked.connect(lambda: self.on_stage_home(None))
        # Not the emergency stop -- that one is the red button in the top
        # right of the toolbar, where it is reachable from every tab. This is
        # the graded version: profiled, so nothing loses steps, and it does
        # not latch. Two buttons because they answer different questions, and
        # a person who wants "that is far enough" should not have to re-home
        # three axes to get it.
        self.btn_stage_stop = QtWidgets.QPushButton("Stop moving")
        self.btn_stage_stop.setStyleSheet(
            "background:#7a1f1f; color:#fff; font-weight:bold;")
        self.btn_stage_stop.setToolTip(
            "End the move in progress and stop any running field map, without "
            "latching the machine off. Profiled, so positions stay "
            "trustworthy and nothing needs re-homing.\n\n"
            "For 'something is wrong', use EMERGENCY STOP in the top right "
            "(or Esc) — that one is immediate and refuses all further motion "
            "until it is reset.")
        self.btn_stage_stop.clicked.connect(self.motion.stop)
        f2.addWidget(self.btn_stage_homeall, 5, 0, 1, 3)
        f2.addWidget(self.btn_stage_stop, 5, 5, 1, 2)
        lay.addWidget(g2)

        g3 = QtWidgets.QGroupBox("3. Field map")
        f3 = QtWidgets.QGridLayout(g3)
        blurb = QtWidgets.QLabel(
            "Move, stop, average, repeat. Each point is averaged at the "
            "carriers' own 200 kSPS with the stream released, so a 5 s point "
            "reaches roughly 0.04 µT — the same argument as the pose capture, "
            "applied to position instead of roll.\n"
            "Every row is traversed in the same direction so leadscrew "
            "backlash cannot stamp a comb into the map.")
        blurb.setWordWrap(True)
        f3.addWidget(blurb, 0, 0, 1, 6)
        for c, head in enumerate(("", "start", "stop", "step"), start=0):
            f3.addWidget(QtWidgets.QLabel(head), 1, c)
        self.scan_rows = {}
        for r, ax in enumerate(("x", "y", "z"), start=2):
            chk = QtWidgets.QCheckBox(ax)
            spins = []
            for val, lo, hi in ((0.0, 0.0, 300.0), (10.0, 0.0, 300.0),
                                (1.0, 0.001, 300.0)):
                sp = QtWidgets.QDoubleSpinBox()
                sp.setRange(lo, hi)
                sp.setDecimals(3)
                sp.setValue(val)
                sp.setSuffix(" mm")
                spins.append(sp)
            f3.addWidget(chk, r, 0)
            for c, sp in enumerate(spins, start=1):
                f3.addWidget(sp, r, c)
            chk.toggled.connect(self._update_scan_estimate)
            for sp in spins:
                sp.valueChanged.connect(self._update_scan_estimate)
            self.scan_rows[ax] = {"chk": chk, "spins": spins}

        self.spin_scan_s = QtWidgets.QDoubleSpinBox()
        self.spin_scan_s.setRange(0.5, 120.0)
        self.spin_scan_s.setValue(5.0)
        self.spin_scan_s.setSuffix(" s")
        self.spin_scan_s.valueChanged.connect(self._update_scan_estimate)
        self.spin_scan_settle = QtWidgets.QDoubleSpinBox()
        self.spin_scan_settle.setRange(0.0, 30.0)
        self.spin_scan_settle.setValue(oscan.DEFAULT_SETTLE_S)
        self.spin_scan_settle.setSuffix(" s")
        self.spin_scan_settle.setToolTip(
            "Wait after the stage stops before averaging. The controller says "
            "'stopped' when its profile ends, which is not when a cantilevered "
            "probe stops ringing. Too short does not look like an error — it "
            "looks like a field gradient.")
        self.spin_scan_settle.valueChanged.connect(self._update_scan_estimate)
        f3.addWidget(QtWidgets.QLabel("average per point"), 5, 0, 1, 2)
        f3.addWidget(self.spin_scan_s, 5, 2)
        f3.addWidget(QtWidgets.QLabel("settle"), 5, 3)
        f3.addWidget(self.spin_scan_settle, 5, 4)

        self.lbl_scan_est = QtWidgets.QLabel("")
        self.lbl_scan_est.setStyleSheet("color:#9aa3b2;")
        f3.addWidget(self.lbl_scan_est, 6, 0, 1, 6)

        self.btn_scan_start = QtWidgets.QPushButton("Start field map")
        self.btn_scan_start.clicked.connect(self.on_scan_start)
        self.btn_scan_abort = QtWidgets.QPushButton("Abort")
        self.btn_scan_abort.setEnabled(False)
        self.btn_scan_abort.clicked.connect(self.on_scan_abort)
        self.bar_scan = QtWidgets.QProgressBar()
        self.bar_scan.setTextVisible(True)
        f3.addWidget(self.btn_scan_start, 7, 0, 1, 2)
        f3.addWidget(self.btn_scan_abort, 7, 2)
        f3.addWidget(self.bar_scan, 7, 3, 1, 3)
        lay.addWidget(g3)

        lay.addStretch(1)
        self._update_scan_estimate()
        self._set_stage_controls_enabled(False)
        return w

    def _set_stage_controls_enabled(self, on):
        """Connected or not. Everything finer than that is _sync_stage_controls."""
        for row in self.stage_rows.values():
            row["present"] = on and row["present"]
        self.btn_stage_disconnect.setEnabled(on)
        self.btn_stage_connect.setEnabled(not on)
        self._sync_stage_controls()

    def on_stage_find(self, quiet=False):
        """Fill the stage table from the bus. Returns True if any were found.

        `quiet` is for the automatic connect, which must not throw a modal in
        front of someone who only asked to connect the probe: a rig with no
        stages plugged in is a normal way to use this window.
        """
        try:
            serials = ostage.list_devices()
        except ostage.StageError as exc:
            self.session.log(f"stages: {exc}")
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "Stages", str(exc))
            return False
        if not serials:
            msg = ("No stages on the bus. The Kinesis application is running "
                   "and holds all of them — close it and try again."
                   if ostage.kinesisis_running() else
                   "No stages on the bus. Check power and USB.")
            self.session.log(f"stages: {msg}")
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "Stages", msg)
            return False

        saved = {v: k for k, v in ostage.load_axis_map().items()}
        self.stage_table.setRowCount(len(serials))
        self.stage_combos = {}
        for r, s in enumerate(serials):
            # The description is decoration. TLI_GetDeviceInfo fails while
            # another process still has the device open, and letting that kill
            # the whole listing would mean a stale handle somewhere else on the
            # machine takes the Stages tab down with it.
            try:
                desc = ostage.device_info(s)["description"]
            except ostage.StageError as exc:
                desc = "—"
                self.session.log(f"stages: {s} description unavailable ({exc})")
            self.stage_table.setItem(r, 0, QtWidgets.QTableWidgetItem(s))
            self.stage_table.setItem(r, 1, QtWidgets.QTableWidgetItem(desc))
            combo = QtWidgets.QComboBox()
            combo.addItems(["—", "x", "y", "z"])
            if s in saved:
                combo.setCurrentText(saved[s])
            self.stage_table.setCellWidget(r, 2, combo)
            self.stage_combos[s] = combo
            self.stage_table.setItem(r, 3, QtWidgets.QTableWidgetItem("—"))
            self.stage_table.setItem(r, 4, QtWidgets.QTableWidgetItem("closed"))
        self.stage_table.resizeColumnsToContents()
        # Sized for the widest thing each column will ever hold, not for what
        # is in it right now: the position and state cells are rewritten five
        # times a second, and re-fitting the columns on every poll makes the
        # whole table twitch.
        for col, width in ((0, 90), (1, 90), (2, 70), (3, 120)):
            self.stage_table.setColumnWidth(col, width)
        self.session.log(f"stages: found {len(serials)} controller(s): "
                     f"{', '.join(serials)}")
        if not saved:
            self.session.log(
                "stages: no axis map yet. Assign x/y/z in the table — the "
                "software cannot work out which stage is which physical "
                "direction, and guessing would silently transpose the "
                "coordinate frame of every map you take. Use the CLI's "
                "'octobee/motion/stage.py identify' to wiggle each one if unsure.")
        return True

    def _axis_map_from_table(self):
        mapping = {}
        for serial, combo in getattr(self, "stage_combos", {}).items():
            name = combo.currentText()
            if name != "—":
                if name in mapping:
                    raise ostage.StageError(
                        f"axis '{name}' is assigned to two stages")
                mapping[name] = serial
        return mapping

    def on_stage_save_map(self):
        try:
            mapping = self._axis_map_from_table()
        except ostage.StageError as exc:
            QtWidgets.QMessageBox.warning(self, "Axis map", str(exc))
            return
        if not mapping:
            QtWidgets.QMessageBox.information(
                self, "Axis map", "Assign at least one axis first.")
            return
        ostage.save_axis_map(mapping)
        self.session.log(f"stages: wrote {ostage.AXIS_CONFIG} — "
                     + ", ".join(f"{k}={v}" for k, v in mapping.items()))

    def on_stage_connect(self):
        if not getattr(self, "stage_combos", None):
            self.on_stage_find()
            if not getattr(self, "stage_combos", None):
                return
        try:
            mapping = self._axis_map_from_table()
        except ostage.StageError as exc:
            QtWidgets.QMessageBox.warning(self, "Axis map", str(exc))
            return
        if not mapping:
            QtWidgets.QMessageBox.information(
                self, "Stages",
                "Assign at least one stage to an axis before connecting.")
            return
        # The mounting comes from stages.json, not from the table: which way a
        # bracket runs is not something the axis dropdown can express, and
        # building the stages without it would silently ignore a reversed axis
        # and mirror every map taken from this window.
        stages = _stage_set_for(mapping)
        self.btn_stage_connect.setEnabled(False)
        self.session.log("stages: opening " + ", ".join(
            f"{k}={v}" for k, v in mapping.items()))

        def work(emit):
            stages.open()
            for st in stages:
                emit(f"{st.name}: {st.model}, travel "
                     f"{st.travel_mm[0]:g}..{st.travel_mm[1]:g} mm"
                     + ("" if st.homed else ", NOT HOMED"))
                if st.invert:
                    emit(f"{st.name}: mounted REVERSED — rig zero is at the "
                         f"far end of its travel, so homing parks it at "
                         f"{st.travel_mm[1]:g} mm, not 0")
                if st.calibration_file() is None:
                    emit(f"{st.name}: no Thorlabs calibration file loaded — "
                         f"on-axis accuracy is ~47 um rather than <5 um")

        self._stage_pending = stages
        self._stage_action("connect", work)

    def on_stage_disconnect(self):
        if self._scan_worker is not None and self._scan_worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, "Stages", "A field map is running. Abort it first.")
            return
        if self.session.stages is not None:
            self.session.stages.close()
            self.session.stages = None
        self._set_stage_controls_enabled(False)
        for r in range(self.stage_table.rowCount()):
            self.stage_table.setItem(r, 3, QtWidgets.QTableWidgetItem("—"))
            self.stage_table.setItem(r, 4, QtWidgets.QTableWidgetItem("closed"))
        self.session.log("stages: closed")

    def _stage_action(self, what, fn):
        """Run one blocking stage command in a worker, one at a time."""
        if self.motion.latched:
            self.session.log(f"stages: {what} refused — emergency stop is latched "
                         f"({self.motion.reason}). Reset it first.")
            return
        if self.motion.busy():
            self.session.log("stages: busy — wait for the current move to finish")
            return
        self._stage_worker = StageWorker(what, fn)
        self._stage_worker.progress.connect(self.session.log)
        self._stage_worker.done.connect(self.on_stage_action_done)
        self.motion.register(self._stage_worker)
        self._stage_worker.start()
        self._sync_stage_controls()

    def on_stage_action_done(self, what, error):
        self.motion.retire(self._stage_worker)
        if error:
            self.session.log(f"stages: {what} failed — {error}")
            if what == "connect":
                self._stage_pending = None
                self.btn_stage_connect.setEnabled(True)
            self._sync_stage_controls()
            if self.motion.latched:
                # The move failed because the machine was stopped, which the
                # operator already knows -- they pressed the button, and the
                # status bar is red. A modal on top of that is noise, and with
                # several axes in flight it is several modals.
                return
            QtWidgets.QMessageBox.warning(self, "Stages", error)
            return
        if what == "connect":
            self.session.stages = self._stage_pending
            self._stage_pending = None
            if self.motion.latched:
                # Latched while disconnected, or latched and then reconnected
                # to clear it. Neither may release the machine: the interlock
                # belongs to the rig, not to whichever StageSet object happens
                # to be alive.
                self.session.stages.interlock.trip(self.motion.reason)
                self.session.log("stages: connected, but motion stays latched off "
                             f"— {self.motion.reason}")
            for ax, row in self.stage_rows.items():
                have = ax in self.session.stages.names
                row["present"] = have
                if have:
                    # The soft limit, not the travel: a spin box that offers a
                    # number the axis is not allowed to go to is an invitation
                    # to find that out by pressing Go.
                    lo, hi = self.session.stages[ax].limit_mm
                    row["target"].setRange(lo, hi)
            self._set_stage_controls_enabled(True)
            for ax, row in self.scan_rows.items():
                have = ax in self.session.stages.names
                row["chk"].setEnabled(have)
                if not have:
                    row["chk"].setChecked(False)
                else:
                    lo, hi = self.session.stages[ax].limit_mm
                    for sp in row["spins"][:2]:     # start and stop, not step
                        sp.setRange(lo, hi)
                for sp in row["spins"]:
                    sp.setEnabled(have)
            vel, acc = next(iter(self.session.stages)).vel_params
            self.spin_stage_vel.setValue(vel)
            self.spin_stage_acc.setValue(acc)
            self.session.log(f"stages: connected — {vel:.3g} mm/s, "
                         f"{acc:.3g} mm/s² profile")
            self._report_stage_envelope()
            # Deferred: this opens a modal, and doing that from inside a
            # worker's completion signal blocks the thread's own teardown.
            QtCore.QTimer.singleShot(0, self._prompt_home_if_needed)
        else:
            self.session.log(f"stages: {what} done")
        self._sync_stage_controls()

    def _report_stage_envelope(self):
        """Say what will actually stop the head, once, at connect.

        An axis with no soft limit is not obviously different from one that
        has them: both move, both report a position, and the difference only
        shows up the first time a scan range is set a little too wide. Saying
        it at connect is the cheapest place to put that.
        """
        bare, whole = [], []
        for name in self.session.stages.names:
            st = self.session.stages[name]
            lo, hi = st.limit_mm
            if st.limit_mm != st.travel_mm:
                self.session.log(f"stages: {name} may use {lo:g}..{hi:g} mm "
                             f"(soft limit inside {st.travel_mm[0]:g}.."
                             f"{st.travel_mm[1]:g} mm of travel)")
            elif st.limit_declared:
                # Declared as the full travel: someone looked and there is
                # nothing in the way. Same movement as an axis nobody has
                # configured, but it is an answer rather than a gap, so it is
                # reported once and does not nag.
                whole.append(name)
            else:
                bare.append(name)
        if whole:
            self.session.log(f"stages: {', '.join(whole)} may use their whole "
                         f"travel — declared in {ostage.AXIS_CONFIG}, not "
                         f"merely unset")
        if bare:
            self.session.log(
                f"stages: {', '.join(bare)} have NO soft limit — the whole "
                f"travel is allowed, and the only thing that will stop the "
                f"head short of the fixture is the limit switch. Set "
                f'"limit_mm" per axis in {ostage.AXIS_CONFIG}.')
        self.session.log("stages: home order "
                     + " → ".join(self.session.stages.home_sequence()))

    def _update_peak_label(self):
        """Say what the current profile means for the step sizes in the boxes.

        The setting is a ceiling, not a speed: what a jog actually reaches
        depends on how far it goes. Showing both makes the connection between
        "I raised the step size" and "it got loud" visible before it happens.
        """
        vel = self.spin_stage_vel.value()
        acc = self.spin_stage_acc.value()
        parts = []
        for d in (1.0, 5.0, 20.0):
            parts.append(f"{d:g} mm → "
                         f"{ostage.peak_speed_mm_s(d, vel, acc):.1f}")
        self.lbl_stage_peak.setText("peak " + ",  ".join(parts) + " mm/s")

    def on_stage_speed(self):
        """Apply the profile to every open axis, and remember it."""
        if self.session.stages is None:
            return
        vel = self.spin_stage_vel.value()
        acc = self.spin_stage_acc.value()

        def apply(emit):
            for st in self.session.stages:
                st.set_vel_params(vel, acc)
            ostage.save_axis_motion(velocity_mm_s=vel, accel_mm_s2=acc)
            emit(f"stages: {vel:g} mm/s, {acc:g} mm/s² on "
                 f"{', '.join(self.session.stages.names)} — saved to stages.json")

        self._stage_action("set motion profile", apply)

    def on_stage_jog(self, axis, delta):
        if self.session.stages is None or axis not in self.session.stages.names:
            return
        st = self.session.stages[axis]
        self._stage_action(f"jog {axis} {delta:+g} mm",
                           lambda emit: st.move_by(delta))

    def on_stage_goto(self, axis, mm):
        if self.session.stages is None or axis not in self.session.stages.names:
            return
        st = self.session.stages[axis]
        if not st.position_trusted:
            QtWidgets.QMessageBox.warning(
                self, "Position cannot be trusted",
                f"Axis {axis}: {st.distrust_reason}.\n\n"
                f"Its position counter cannot be believed, so an absolute "
                f"move would go somewhere arbitrary. Home it first, or use "
                f"the jog buttons, which are relative and do not need a "
                f"reference.")
            return
        self._stage_action(f"move {axis} to {mm:g} mm",
                           lambda emit: st.move_to(mm))

    def on_stage_home(self, axes):
        if self.session.stages is None:
            return
        wanted = set(axes or self.session.stages.names)
        # Ordered even for a subset: home_sequence() is the order that is safe
        # on this rig, and "the two axes you happened to tick" is not.
        names = [n for n in self.session.stages.home_sequence() if n in wanted]
        if not names:
            return
        one_at_a_time = ("" if len(names) == 1 else
                        f"\n\nThey go one at a time, in this order: "
                        f"{' → '.join(names)}.")
        reply = QtWidgets.QMessageBox.warning(
            self, "Home",
            f"Homing drives {', '.join(names)} into the limit switch at full "
            f"homing speed, across the whole travel — past any soft limit, "
            f"which does not apply to homing.{one_at_a_time}\n\n"
            f"Is the probe head — and its cabling — clear of the entire range "
            f"of movement?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel)
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            self.session.log("stages: homing cancelled")
            return
        self._home_axes(names)

    def _home_axes(self, names):
        """Home these axes, one at a time.

        The caller has already established it is safe. Sequential, where this
        used to start all of them together: with three carriages sweeping
        their full travel at once, whether they foul each other depends on
        which one happens to be slower, and that is not a property anything
        here can check. See StageSet.home_all.
        """
        stages = [self.session.stages[n] for n in names]

        def work(emit):
            for st in stages:
                # Between axes, not only within one: an emergency stop raises
                # out of home() on its own, but the plain "Stop moving" does
                # not latch, so without this it would halt one carriage and
                # then send the next one off across its whole travel.
                if self._stage_worker.aborted:
                    emit(f"homing stopped before {st.name}")
                    return
                st.home(wait=False)
                st.wait(timeout_s=300.0, what="homing")
                st.trust_after_homing()
                emit(f"{st.name}: homed, at {st.position_mm:.4f} mm")

        self._stage_action(f"home {', '.join(names)}", work)

    def _prompt_home_if_needed(self):
        """Offer to reference the axes, once, when they first come up.

        An unhomed stage still reports a position, and that number is whatever
        was left in the counter -- so the one moment this is worth raising is
        now, before anyone has read a coordinate off the screen and believed
        it. It stays a question rather than an automatic move because homing
        drives the carriage into the limit switch across the whole travel,
        with the probe head and its cable dress mounted; only the person in
        the room can say that is clear.
        """
        if self.session.stages is None:
            return
        unhomed = [n for n in self.session.stages.names
                   if not self.session.stages[n].position_trusted]
        if not unhomed:
            self.session.log("stages: every axis is already referenced")
            return
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Stages are not referenced")
        box.setText(f"Axes {', '.join(unhomed)} have not been homed.")
        box.setInformativeText(
            "Until they are, their position readings are whatever was left in "
            "the counter, absolute moves are refused and a field map has no "
            "origin. Jogging is relative and works either way.\n\n"
            "Homing drives each axis into its limit switch across the whole "
            "travel. Is the probe head — and its cabling — clear of the "
            "entire range of movement?")
        yes = box.addButton("Home all axes now",
                            QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Not yet", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is yes:
            self._home_axes(self.session.stages.names)
        else:
            self.session.log(f"stages: {', '.join(unhomed)} left unreferenced — "
                         f"home from the Stages tab before mapping")

    # ---- stopping -------------------------------------------------------

    def on_estop(self, _checked=False):
        """The button, the Esc key and the watchdog all come through here."""
        self.motion.trigger("operator pressed the emergency stop")

    def _refresh_estop_ui(self):
        """Make the latched state impossible to miss or to mistake."""
        latched = self.motion.latched
        self.btn_estop_reset.setVisible(latched)
        self.btn_estop.setText("■  STOPPED" if latched
                               else "■  EMERGENCY STOP")
        if latched:
            if self._state_before_estop is None:
                self._state_before_estop = self.lbl_state.text()
            self.lbl_state.setText(f"EMERGENCY STOP — {self.motion.reason}")
            self.lbl_state.setStyleSheet(
                "color:#fff; background:#c1121f; font-weight:bold; padding:1px 8px;")
        else:
            self.lbl_state.setStyleSheet("")
            # Put back what the status bar said before, rather than leaving
            # "EMERGENCY STOP" sitting there unstyled -- which reads as a
            # machine that is still stopped.
            self.lbl_state.setText(self._state_before_estop or "disconnected")
            self._state_before_estop = None
        self._sync_stage_controls()

    def _sync_stage_controls(self):
        """Motion controls are live only when this window may command motion.

        Two things gate them, and only one of them used to. The interlock is
        the new one. The other is that something else is already driving: a
        field map or a guided-magnet run owns the axes for its duration, and
        the jog, Go and Home buttons stayed enabled behind it -- so a Home
        during a raster would start a second thread commanding the same axis
        through the same DLL, the scan's wait() would watch the homing move as
        if it were its own, and the map would carry on being written with
        every remaining point taken somewhere other than where it says.
        """
        live = (self.session.stages is not None and not self.motion.latched
                and not self.motion.busy())
        for row in self.stage_rows.values():
            for wdg in row["widgets"]:
                wdg.setEnabled(live and row.get("present", True))
        self.btn_stage_homeall.setEnabled(live)
        self.btn_stage_speed.setEnabled(live)
        self.btn_scan_start.setEnabled(live)
        # Stopping stays available in every state that is not "no stages".
        self.btn_stage_stop.setEnabled(self.session.stages is not None)

    def refresh_stage_table(self):
        """Position and state per stage, off the DLL's own polling cache."""
        if self.session.stages is None or not self.stage_table.rowCount():
            return
        by_serial = {st.serial: st for st in self.session.stages}
        for r in range(self.stage_table.rowCount()):
            item = self.stage_table.item(r, 0)
            st = by_serial.get(item.text() if item else None)
            if st is None or not st.is_open:
                continue
            try:
                snap = st.snapshot()
            except ostage.StageError:
                continue
            self.motion.watchdog(snap)
            # Before opening, all the bus can say is "APT Stepper Motor
            # Controller"; the actual model only arrives with the settings.
            if snap["model"] and self.stage_table.item(r, 1).text() != snap["model"]:
                self.stage_table.setItem(
                    r, 1, QtWidgets.QTableWidgetItem(snap["model"]))
            self.stage_table.setItem(
                r, 3, QtWidgets.QTableWidgetItem(f"{snap['position_mm']:.4f} mm"))
            state = "moving" if snap["moving"] else "idle"
            if not snap["trusted"]:
                # "NOT HOMED" was the old wording and it was too narrow: the
                # counter can be untrustworthy on an axis whose homed bit is
                # still set, which is the case that actually hurts.
                state += (", NOT HOMED" if not snap["homed"]
                          else ", POSITION LOST")
            if snap["error"]:
                state += ", MOTION ERROR"
            if snap["at_hard_limit"]:
                state += ", ON HARD LIMIT"
            if snap["interlocked"]:
                state += ", STOPPED"
            if snap["invert"]:
                # Worth carrying in the always-visible row: the rig number and
                # the number on the controller's own display disagree on a
                # reversed axis, and that is alarming until you know why.
                state += f"  [reversed, device {snap['position_dev_mm']:.3f}]"
            self.stage_table.setItem(r, 4, QtWidgets.QTableWidgetItem(state))

    # ---- field map -------------------------------------------------------
    def _scan_grid(self):
        axes = {}
        for ax, row in self.scan_rows.items():
            if not row["chk"].isChecked():
                continue
            start, stop, step = (sp.value() for sp in row["spins"])
            if stop < start:
                raise ValueError(
                    f"{ax}: stop ({stop:g}) is before start ({start:g}). The "
                    f"scan always runs in the +ve direction so that every "
                    f"point is approached from the same side and backlash "
                    f"cannot bias alternate rows.")
            axes[ax] = oscan.parse_axis_spec(f"{start}:{stop}:{step}")
        if not axes:
            raise ValueError("tick at least one axis to scan")
        return oscan.ScanGrid(axes)

    def _update_scan_estimate(self):
        try:
            grid = self._scan_grid()
        except ValueError as exc:
            self.lbl_scan_est.setText(str(exc))
            return
        n = len(grid)
        # Per point: the average itself, the settle, and the move. The move is
        # a guess -- it depends on step size and velocity -- so this is a floor
        # to plan around, not a promise.
        per = self.spin_scan_s.value() + self.spin_scan_settle.value() + 2.0
        total = n * per
        self.lbl_scan_est.setText(
            f"{n} points — {grid.describe()} — roughly "
            f"{total / 60:.0f} min ({total / 3600:.1f} h) at {per:.1f} s/point")

    def on_scan_start(self):
        if self.session.stages is None:
            return
        if self.motion.busy():
            return
        if self.motion.latched:
            QtWidgets.QMessageBox.warning(
                self, "Field map",
                f"The emergency stop is latched: {self.motion.reason}.\n\n"
                f"Reset it before starting a map.")
            return
        try:
            grid = self._scan_grid()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Field map", str(exc))
            return
        unhomed = [(n, self.session.stages[n].distrust_reason) for n in grid.names
                   if not self.session.stages[n].position_trusted]
        if unhomed:
            QtWidgets.QMessageBox.warning(
                self, "Field map",
                "These axes' position counters cannot be believed, so the map "
                "would have no origin. Home them first.\n\n  "
                + "\n  ".join(f"{n}: {why}" for n, why in unhomed))
            return
        # Checked here as well as inside every move, because a range that is
        # out of bounds should be a refusal before a six-hour job starts, not
        # a point failure two hours in.
        outside = []
        for name, pts in grid.axes.items():
            lo, hi = self.session.stages[name].limit_mm
            first, last = float(min(pts)), float(max(pts))
            if not (lo <= first and last <= hi):
                outside.append(f"{name}: asks for {first:g}..{last:g} mm, "
                               f"allowed {lo:g}..{hi:g} mm")
        if outside:
            QtWidgets.QMessageBox.warning(
                self, "Field map",
                "The scan range goes outside what these axes are allowed to "
                "use:\n\n  " + "\n  ".join(outside)
                + f"\n\nEither shorten the range, or change \"limit_mm\" in "
                  f"{ostage.AXIS_CONFIG} — having measured that the head "
                  f"really can go there.")
            return

        n = len(grid)
        per = self.spin_scan_s.value() + self.spin_scan_settle.value() + 2.0
        reply = QtWidgets.QMessageBox.question(
            self, "Field map",
            f"{n} points over {grid.describe()}.\n"
            f"Roughly {n * per / 60:.0f} minutes.\n\n"
            f"This takes the carriers off the live stream and puts them back "
            f"on their own 200 kSPS clock for the duration — the live plot "
            f"will stop.\n\nStart?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel)
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        if self.act_record.isChecked():
            self.act_record.setChecked(False)
        if isinstance(self.session.source, LiveSource):
            self.session.source.stop()
        self.session.source = None
        self.lbl_state.setText("field map in progress...")
        self.bar_scan.setRange(0, n)
        self.bar_scan.setValue(0)
        self.btn_scan_abort.setEnabled(True)
        self.act_snapshot.setEnabled(False)

        self._scan_worker = ScanWorker(
            self.session.hosts, self.session.stages, grid, self.spin_scan_s.value(),
            self.session.cal, self.spin_scan_settle.value(), self.session.prev_clkdiv,
            # Which coils were on, at what current, and where the probe was
            # bolted. None of it can be recovered from the numbers later, and
            # a map without it is a table of vectors at nowhere in particular.
            extra_meta={"machine": self.session.machine.to_scan_meta(
                self.session.coils, self.tab_machine.stage_mm(ignore_tracking=True))})
        # As with the snapshot: the worker restores the clock itself, so
        # clearing this now means a failed scan cannot leave a stale value
        # for disconnect to apply on top.
        self.session.prev_clkdiv = {}
        self._scan_worker.message.connect(self.session.log)
        self._scan_worker.progress.connect(self.on_scan_progress)
        self._scan_worker.done.connect(self.on_scan_done)
        self._scan_t0 = time.time()
        self.motion.register(self._scan_worker)
        self._scan_worker.start()
        self._sync_stage_controls()

    def on_scan_progress(self, i, n, where, sem_ut):
        self.bar_scan.setValue(i)
        elapsed = time.time() - self._scan_t0
        eta = elapsed / max(i, 1) * (n - i)
        self.bar_scan.setFormat(f"%v / %m — {eta / 60:.0f} min left")
        self.lbl_state.setText(f"field map {i}/{n}: {where}")
        if i == 1 or i % 10 == 0 or i == n:
            self.session.log(f"  point {i}/{n} at {where}: noise {sem_ut:.3f} uT")

    def on_scan_abort(self):
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._scan_worker.abort()
            self.btn_scan_abort.setEnabled(False)
            self.session.log("field map: stopping after the point in flight")

    def on_scan_done(self, fm, error):
        self.motion.retire(self._scan_worker)
        self.btn_scan_abort.setEnabled(False)
        self.act_snapshot.setEnabled(True)
        self.bar_scan.setFormat("%v / %m")
        if not self.motion.latched:
            self.lbl_state.setText("disconnected")
        self._sync_stage_controls()
        if error:
            self.session.log(f"field map failed: {error}")
            QtWidgets.QMessageBox.warning(self, "Field map", error)
        if fm is None or not len(fm):
            return
        path = fm.save(os.path.join(
            self.session.out_dir, time.strftime("fieldmap_%Y%m%d_%H%M%S")))
        sem = np.array([s.get("sem_ut", np.nan) for s in fm.stats])
        self.session.log(
            f"field map: {len(fm)} of {fm.meta['n_requested']} points, "
            f"noise median {np.nanmedian(sem):.3f} uT / worst "
            f"{np.nanmax(sem):.3f} uT -> {path}")
        self.session.log("field map: the carriers are still off the live stream — "
                     "press Connect to go back to the live view.")

    # ---- the machine around the probe ------------------------------------
    # ---- machine: loading and current ------------------------------------
    # ---- machine: placement ----------------------------------------------
    def _apply_dark(self):
        self.setStyleSheet("""
            QWidget { background:#12141a; color:#d5d9e2; }
            QGroupBox { border:1px solid #2a2e3a; border-radius:4px;
                        margin-top:9px; padding-top:8px; }
            QGroupBox::title { subcontrol-origin: margin; left:8px;
                               color:#8f98a8; }
            QTabBar::tab { background:#1a1d26; padding:6px 14px; }
            QTabBar::tab:selected { background:#262b38; }
            QTableWidget { gridline-color:#2a2e3a;
                           alternate-background-color:#171a22; }
            QHeaderView::section { background:#1a1d26; border:0;
                                   padding:4px; color:#8f98a8; }
            QPushButton { background:#232836; border:1px solid #333a4a;
                          padding:5px 10px; border-radius:3px; }
            QPushButton:hover { background:#2c3244; }
            QPushButton:checked { background:#3a5070; }
            QPlainTextEdit { background:#0d0f14; border:1px solid #262b38; }
            QToolBar { background:#171a22; border-bottom:1px solid #262b38;
                       spacing:4px; padding:3px; }
        """)

    # ---- connection ------------------------------------------------------
    def _auto_connect(self):
        """The connect the window makes for itself when it opens.

        Marked so that a failure logs rather than throwing a modal across the
        window: the carriers being off is a normal way to arrive at this
        program -- to look at a saved capture, to edit a calibration -- and a
        dialog to dismiss on every launch would train you to dismiss dialogs.
        """
        if self.session.source is not None:
            return
        self._connect_was_automatic = True
        self.on_connect()

    def on_connect(self):
        if self.session.source is not None:
            return
        # A second attempt while the first is still in flight would start a
        # second worker against the same carriers -- and the automatic connect
        # means there is now something in flight that nobody pressed. Tracked
        # with a flag rather than by asking the thread: isRunning() is false
        # for the moment between start() and the thread actually running, and
        # that moment is exactly when a second call arrives.
        if self._connecting:
            return
        fs = float(self.cmb_rate.currentData())
        self.act_connect.setEnabled(False)
        self.lbl_state.setText("connecting...")
        self.session.log(f"connecting to {', '.join(self.session.hosts)}"
                     + (f", setting {fs/1000:g} kSPS" if fs else ""))
        self._connect_worker = ConnectWorker(self.session.hosts, fs)
        self._connect_worker.done.connect(self.on_connected)
        self._connect_worker.progress.connect(self.on_connect_progress)
        self._connecting = True
        self._connect_worker.start()
        # The rig is one instrument: the probe reads the field, the stages say
        # where it was read. Connecting one and not the other is a state the
        # operator has to remember to fix, and forgetting it does not announce
        # itself -- it just means the field map button is greyed out for no
        # visible reason. The two run in separate workers, so a bench with no
        # stages plugged in connects the probe exactly as before.
        self.connect_stages(quiet=True)

    def connect_stages(self, quiet=False):
        """Open the stages named in stages.json, without touching the probe.

        Returns False and logs if there is nothing to connect. Never raises
        into the caller: the probe half of the window has to come up whatever
        the motion hardware is doing.
        """
        try:
            return self._connect_stages(quiet)
        except Exception as exc:
            # Deliberately broad. This runs inside Connect, and the probe half
            # of the window must come up whatever the motion hardware, the
            # Kinesis install or a stale USB handle is doing.
            self.session.log(f"stages: not connected — {type(exc).__name__}: {exc}")
            self._stage_pending = None
            self.btn_stage_connect.setEnabled(True)
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "Stages", str(exc))
            return False

    def _connect_stages(self, quiet):
        if self.session.stages is not None or self._stage_pending is not None:
            return False
        if self._stage_worker is not None and self._stage_worker.isRunning():
            return False
        mapping = ostage.load_axis_map()
        if not mapping:
            self.session.log(
                "stages: no axis map in stages.json, so nothing was connected "
                "automatically. Assign x/y/z on the Stages tab once and every "
                "later session picks it up.")
            return False
        if not self.on_stage_find(quiet=quiet):
            return False
        missing = [ser for ser in mapping.values()
                   if ser not in (self.stage_combos or {})]
        if missing:
            self.session.log(f"stages: {', '.join(missing)} is in stages.json but "
                         f"not on the bus — connect from the Stages tab once "
                         f"it is back")
            return False
        stages = _stage_set_for(mapping)
        self._stage_pending = stages
        self.btn_stage_connect.setEnabled(False)
        self._stage_action("connect", lambda emit: stages.open())
        return True

    def on_connect_progress(self, msg):
        # Connecting involves several seconds of deliberate settling delays.
        # Without this the window sits on "connecting..." long enough to look
        # like it has hung, which is exactly how it was first reported.
        self.lbl_state.setText(f"connecting -- {msg}")
        self.session.log(msg)

    def on_connected(self, source, prev, error):
        self._connecting = False
        self.act_connect.setEnabled(True)
        was_automatic = self._connect_was_automatic
        self._connect_was_automatic = False
        if error:
            self.lbl_state.setText("disconnected")
            self.session.log(f"connect failed: {error}")
            if was_automatic:
                self.session.log(
                    "the automatic connect is the only thing that failed -- "
                    "everything that does not need the carriers still works, "
                    "and Connect will try again")
            else:
                QtWidgets.QMessageBox.critical(
                    self, "Connect failed",
                    f"{error}\n\nIf this says the stream closed immediately, "
                    f"something else owns port 4210 -- usually a Phoebus "
                    f"'Streaming Capture' still running on that box.")
            for h, p in (prev or {}).items():
                with contextlib.suppress(Exception):
                    olive.restore_rate(h, p)
            return
        self.session.prev_clkdiv = prev or {}
        self._set_source(source, "live")

    def _set_source(self, source, kind):
        self.session.source = source
        # Measure the running state, not the startup: building the window and
        # bringing up the GL context stalls the loop once, and leaving that in
        # the numbers makes a healthy session look blocked.
        self.session.prof.reset()
        self.session.lag.reset()
        # counts-per-volt is read off the box, so the counts scale is only
        # known once a source exists.
        if hasattr(self, "cmb_units"):
            self.on_units()
        self.act_disconnect.setEnabled(True)
        self.act_connect.setEnabled(kind != "live")
        self.lbl_state.setText(f"{kind}: {', '.join(source.hosts)} "
                               f"@ {source.fs_hz/1000:g} kSPS")
        self._reset_buffers()
        self.session.log(f"{kind} source running at {source.fs_hz/1000:g} kSPS, "
                     f"decimating to {self.session.out_rate:g} Hz for display and CSV")

    def on_disconnect(self):
        if self.act_record.isChecked():
            self.act_record.setChecked(False)
        if self.session.source is not None:
            self.session.source.stop()
            self.session.source = None
        # Symmetry with Connect: one button owns the whole rig. Leaving the
        # stages open would hold the USB devices against the Kinesis app and
        # against the next run of this window, which looks like a cabling
        # fault rather than a stale handle.
        if self.session.stages is not None:
            self.on_stage_disconnect()
        for h, p in self.session.prev_clkdiv.items():
            try:
                olive.restore_rate(h, p)
                self.session.log(f"{h}: clkdiv restored to {p}")
            except Exception as e:
                self.session.log(f"{h}: could not restore clkdiv: {e}")
        self.session.prev_clkdiv = {}
        self.act_disconnect.setEnabled(False)
        self.act_connect.setEnabled(True)
        self.lbl_state.setText("disconnected")

    def _reset_buffers(self):
        self.session.roll = Rolling(max(4, int(self.session.out_rate * self.window_s)))
        self.raw_hist = None
        self.raw_hist_n = 0
        self._last_health_t = 0.0
        self._last_dropped = 0

    def _calibration_changed(self, what):
        """
        Drop the buffered history whenever the conversion changes.

        The rolling buffer holds millitesla, not counts, so after a tare or a
        range change the points already in it were computed a different way.
        Left in place they poison exactly the things that read the buffer:
        plot autoscaling, the peak bars, and -- worst -- the baseline of a
        magnet pass, which would then measure the calibration step rather than
        the magnet.

        A running CSV recording is rolled over to a new file for the same
        reason, one level up. CsvRecorder stamps calibration_id and the whole
        conversion state into its header once, at open, so that a file can be
        matched back to the calibration that produced it. Carrying on writing
        into that file after the conversion changed would produce one CSV whose
        rows came from two different calibrations, under a header naming only
        the first -- which is worse than either file alone, because nothing
        about it looks wrong.
        """
        self.session.roll.clear()
        self.view3d.reset_scale()
        if hasattr(self, "cmb_units"):
            self.on_units()
        if self.session.collecting is not None and self.session.collecting["what"] == "magnet":
            self.btn_magnet.setChecked(False)
            self.session.log(f"magnet pass abandoned: {what} changed mid-pass")
        self._roll_over_csv(what)

    def _roll_over_csv(self, what):
        """Close the CSV in progress and open a fresh one with a new header."""
        if self.session.csv_rec is None:
            return
        old = self.session.csv_rec
        rows = old.n_rows
        old.close()
        path = orec.default_name("octobee", "csv", self.session.out_dir)
        self.session.csv_rec = orec.CsvRecorder(
            path, self.session.out_rate, self.session.cal, self.session.geom,
            tube_frame=self.tab_export.tube_frame(),
            meta={"hosts": ",".join(self.session.source.hosts) if self.session.source else "",
                  "stream_rate_hz": self.session.source.fs_hz if self.session.source else 0.0,
                  "continues": os.path.basename(old.path),
                  "rolled_over_because": f"{what} changed"})
        self.session.log(
            f"{what} changed while recording: closed {os.path.basename(old.path)} "
            f"at {rows} rows and started {os.path.basename(path)}, so each file "
            f"matches the calibration named in its own header. The raw .bin is "
            f"unaffected -- it holds counts, which no calibration changes.")

    @property
    def decim(self):
        if self.session.source is None:
            return 1
        return max(1, int(round(self.session.source.fs_hz / self.session.out_rate)))

    # ---- the acquisition tick --------------------------------------------
    def on_tick(self):
        if self.session.source is None:
            return
        with self.session.prof.time("acquisition tick (total)"):
            with self.session.prof.time("  socket read + decode"):
                blocks = self.session.source.read()
            if not blocks or blocks[0].shape[0] == 0:
                if getattr(self.session.source, "error", None):
                    self.session.log(f"stream error: {self.session.source.error}")
                    self.session.source.error = None
                return
            self.session.prof.note("samples per acquisition tick", blocks[0].shape[0])

            with self.session.prof.time("  keep raw history"):
                self._keep_raw(blocks)
            if self.session.raw_rec is not None:
                with self.session.prof.time("  write raw file"):
                    self.session.raw_rec.write(blocks)

            with self.session.prof.time("  counts -> tesla"):
                grouped = ocal.assemble(blocks, self.session.source.vpc,
                                        self.session.source.volt_offset)  # (n,16,4) V
                b = self.session.cal.to_mt(grouped)                        # (n,16,3) mT
            with self.session.prof.time("  decimate + buffer"):
                bd = ocal.decimate(b, self.decim)
                if bd.shape[0] == 0:
                    return
                self.session.roll.push(bd.astype(np.float32))

            if self.session.csv_rec is not None:
                with self.session.prof.time("  write CSV"):
                    self.session.csv_rec.write(bd)

            if self.session.collecting is not None:
                with self.session.prof.time("  magnet/tare collect"):
                    self._collect_block(b, grouped)

    # Drawing is deliberately NOT done here. This tick has to keep draining the
    # reader queues or the carriers overrun and the recording gets holes in it,
    # and repainting a 3D scene of 16 boards costs far more than the arithmetic
    # above. Everything visual runs on its own timer.

    def on_view_tick(self):
        """Redraw the live view. Its rate is the user's to choose."""
        if self.session.source is None or self.paused:
            return
        recent = self.session.roll.view()
        if recent.shape[0] < 2:
            return
        t0 = time.perf_counter()
        with self.session.prof.time("view tick (total)"):
            if self.chk_3d.isChecked():
                with self.session.prof.time("  3D head: build vectors"):
                    k = min(8, recent.shape[0])
                    fs = self.view3d.update_fields(recent[-k:].mean(axis=0))
                if (self.chk_auto.isChecked()
                        and abs(fs - self.spin_fs.value()) > 1e-4):
                    self.spin_fs.blockSignals(True)
                    self.spin_fs.setValue(max(fs, 0.001))
                    self.spin_fs.blockSignals(False)
            with self.session.prof.time("  live plot setData"):
                self.plot.update_data(recent, self.session.out_rate)
            with self.session.prof.time("  peak bars"):
                # Peak over a trailing half second, not the instantaneous
                # value: a magnet passed by hand is over well inside one
                # refresh, so sampling one point would miss most passes.
                n = max(2, min(recent.shape[0],
                               int(self.session.out_rate * self.bars.window_s)))
                self.bars.update_values(
                    np.linalg.norm(recent[-n:], axis=-1).max(axis=0),
                    self.session.cal.dead)
        self._note_draw_time((time.perf_counter() - t0) * 1000.0)

    def _note_draw_time(self, ms):
        """
        Watch how long a redraw costs and back off if it cannot keep up.

        A machine without working GPU acceleration falls back to software
        OpenGL, where painting the 3D head can take seconds. The symptom is a
        window that seems fine until data starts arriving and then locks solid.
        Rather than leave that to be diagnosed by hand, measure it and slow the
        redraw down -- acquisition and recording are on another clock and are
        not affected either way.
        """
        self._draw_ms = 0.75 * self._draw_ms + 0.25 * ms if self._draw_ms else ms
        interval = self.view_timer.interval()
        if self._draw_ms > 0.7 * interval and interval < MAX_VIEW_INTERVAL_MS:
            new = min(MAX_VIEW_INTERVAL_MS, interval * 2)
            self.view_timer.setInterval(new)
            self.session.log(
                f"a redraw is taking {self._draw_ms:.0f} ms, more than the "
                f"{interval} ms budget -- slowing the view to "
                f"{1000.0/new:.1f} Hz. Acquisition and recording are unaffected. "
                f"If this keeps happening, untick '3D': on a machine without "
                f"GPU acceleration the 3D head is by far the most expensive "
                f"thing on screen.")

    def _keep_raw(self, blocks):
        """Keep the last few seconds of raw counts for the health analysis."""
        cap = int(self.session.source.fs_hz * RAW_HISTORY_S)
        if self.raw_hist is None:
            self.raw_hist = [deque() for _ in blocks]
        for i, blk in enumerate(blocks):
            self.raw_hist[i].append(blk)
        self.raw_hist_n += blocks[0].shape[0]
        while self.raw_hist_n > cap and len(self.raw_hist[0]) > 1:
            drop = self.raw_hist[0][0].shape[0]
            for d in self.raw_hist:
                d.popleft()
            self.raw_hist_n -= drop

    def _raw_arrays(self, seconds=None):
        """
        Raw counts from the tail of the history, per box.

        `seconds` matters for speed, not just convenience: concatenating the
        whole 5 s history is tens of megabytes of copying, and doing that on
        every refresh starves the reader threads until the box's queues
        overflow. The periodic health check asks for a short window; the
        Diagnostics button asks for everything.
        """
        if not self.raw_hist or not self.raw_hist[0]:
            return None
        if seconds is None:
            return [np.concatenate(list(d), axis=0) for d in self.raw_hist]
        want = max(1, int(self.session.source.fs_hz * seconds))
        out = []
        for d in self.raw_hist:
            blocks, n = [], 0
            for blk in reversed(d):
                blocks.append(blk)
                n += blk.shape[0]
                if n >= want:
                    break
            if not blocks:
                return None
            out.append(np.concatenate(blocks[::-1], axis=0)[-want:])
        return out

    def on_slow_tick(self):
        if self.session.source is None:
            return
        recent = self.session.roll.view()
        if recent.shape[0] < 2:
            return

        # Health first: it decides which sensors are excluded, and everything
        # that follows scales its axes around that decision. It runs on a slower
        # clock than the display, because scanning 64 channels is the most
        # expensive thing in this loop and the answer changes only when a
        # connector does.
        now = time.time()
        if now - self._last_health_t >= HEALTH_PERIOD_S:
            raw = self._raw_arrays(HEALTH_WINDOW_S)
            if raw is not None:
                self._last_health_t = now
                with self.session.prof.time("  channel health scan"):
                    rows = ocal.channel_health(
                        raw, self.session.source.vpc, self.session.source.hosts,
                        self.session.source.volt_offset)
                self.session.last_health = rows
                if self.tab_health.auto_exclude_dead():
                    dead = ocal.suggest_dead(rows)
                    if dead != self.session.cal.dead:
                        self.session.cal.dead = dead
                        self._mark_dead_checkboxes(dead)
                        if dead:
                            self.session.log(
                                f"excluding {', '.join(sorted(dead))}: "
                                + "; ".join(
                                    f"{k} {v[1]}" for k, v in
                                    ocal.health_verdict(rows).items()
                                    if v[0] == "dead"))
        if self.session.last_health:
            table = orec.sensor_table(self.session.cal, recent.astype(np.float64),
                                      self.session.last_health,
                                      self.session.source.temperatures(), geom=self.session.geom)
            self.table.update_rows(table)
            self._last_table = table

        self.plot.set_dead(self.session.cal.dead)
        self.view3d.set_dead(self.session.cal.dead)

        # A dropped block is a hole in whatever is being recorded, so say so
        # rather than leaving it as a number in the corner of the status bar.
        dropped = self.session.source.stats().get("dropped blocks", 0)
        if dropped > self._last_dropped:
            if self.session.csv_rec is not None or self.session.raw_rec is not None:
                self.session.log(f"WARNING: {dropped - self._last_dropped} block(s) "
                             f"dropped while recording -- the file has a gap "
                             f"there. Lower the output rate or the stream rate.")
            self._last_dropped = dropped

        st = dict(self.session.source.stats())
        if self._draw_ms:
            st["draw ms"] = round(self._draw_ms, 1)
        eff = 1000.0 / max(self.view_timer.interval(), 1)
        want = float(self.cmb_view.currentData())
        # Show the rate actually being achieved, marked when the automatic
        # backoff has taken it below what was asked for.
        st["view Hz"] = f"{eff:.1f}" + (" (auto)" if eff < want * 0.95 else "")
        self.lbl_rate.setText("  |  ".join(
            f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
            for k, v in st.items()))
        self._update_rec_label()

    # ---- collection (tare, magnet pass) ----------------------------------
    def start_collect(self, what, seconds):
        if self.session.source is None:
            self.session.log("not connected")
            return
        self.session.collecting = {"what": what, "blocks": [], "n": 0,
                           "need": int(seconds * self.session.source.fs_hz),
                           "peak": None, "baseline": None, "tag": None,
                           "decim": self.decim}
        self.session.log(f"collecting {seconds:g} s for {what}...")

    def _collect_block(self, b, grouped=None):
        """
        Accumulate one acquisition block into whatever collection is running.

        b        (n,16,3) fully calibrated mT -- offset, gain and matrix applied
        grouped  (n,16,4) volts straight off assemble(), nothing applied

        Which of the two a mode wants is not a detail: anything that MEASURES a
        correction has to work from `grouped`, or it fits on top of the
        correction already loaded and the answer depends on where you started.
        Only the magnet pass, which reports a field rather than deriving a
        calibration, legitimately uses `b`.
        """
        c = self.session.collecting
        if c["what"] == "magnet":
            # Deviation from the field that was there when the pass STARTED.
            # Re-deriving the baseline from the rolling window each block would
            # let the magnet drag the baseline along with it and shrink the peak.
            dev = np.linalg.norm(b - c["baseline"][None, :, :], axis=-1)
            best = dev.max(axis=0)
            c["peak"] = best if c["peak"] is None else np.maximum(c["peak"], best)
            c["n"] += b.shape[0]
            return

        # Both remaining modes measure a calibration, so both take the
        # uncorrected field. Decimate hard while we are at it: for a sweep the
        # information is in the shape of the ellipse, and a couple of hundred
        # hertz resolves a hand roll a thousand times over.
        if grouped is None:
            return
        raw = self.session.cal.to_mt(grouped, apply_zero=False, apply_gain=False,
                             apply_matrix=False)
        c["blocks"].append(ocal.decimate(raw, max(1, c["decim"])))
        c["n"] += b.shape[0]
        if c["n"] < c["need"]:
            return

        data = np.concatenate(c["blocks"], axis=0)
        what, tag = c["what"], c["tag"]
        self.session.collecting = None
        if what == "sweep":
            self._finish_sweep(data, tag)
        else:
            self._finish_tare(data)

    def _finish_tare(self, data):
        """
        Set the zero from an UNCORRECTED capture.

        `data` is what to_mt(apply_zero=False, apply_gain=False,
        apply_matrix=False) produced, which is what zero_mt is defined against:
        Calibration applies offset, then gain, then matrix, so the zero is a
        pre-gain quantity.

        This used to reconstruct it instead, as `data + zero_mt`, from the
        fully corrected buffer. That inverts the chain only when the gain trim
        is 1.0 and the matrix is the identity -- so after a magnet-pass trim or
        an applied roll calibration the stored zero came out scaled by the
        trim, silently. Collecting the uncorrected field in the first place
        means there is nothing to invert.
        """
        z = self.session.cal.tare(data)
        self.session.log(f"zeroed on {data.shape[0]} points; "
                     f"largest offset removed {np.abs(z).max():.4f} mT "
                     f"(S{int(np.argmax(np.abs(z).max(axis=1)))+1})")
        self._calibration_changed("the zero point")
        self.refresh_cal_report()


    # ---- Earth-field roll calibration ------------------------------------
    def start_sweep(self, tag, seconds):
        """Record one hand-rolled sweep in the mounting orientation `tag`."""
        if self.session.source is None:
            self.session.log("not connected")
            return
        fs = self.session.source.fs_hz
        self.session.collecting = {"what": "sweep", "tag": tag, "blocks": [], "n": 0,
                           "need": int(seconds * fs), "peak": None,
                           "baseline": None,
                           # aim for a few hundred Hz, whatever the ADC is at
                           "decim": max(1, int(fs / 200))}
        self.session.log(f"rolling sweep {tag}: roll the tube steadily through at "
                     f"least two full turns over the next {seconds:g} s")

    def _finish_sweep(self, data, tag):
        sw = opc.RollSweep(data, tag=tag,
                           ranges_mt=self.session.cal.ranges_mt.copy(),
                           temps_c=self._last_temps())
        self.session.sweeps[tag] = sw
        amp = sw.amplitudes()
        quiet = [f"S{i+1}" for i in range(N_SENSORS)
                 if amp[i] < opc.MIN_SEED_AMPLITUDE_MT
                 and not self.session.cal.is_dead(i + 1)]
        self.session.log(f"sweep {tag}: {len(sw)} points, median transverse swing "
                     f"{np.median(amp)*1e3:.2f} uT"
                     + (f"; SAW ALMOST NOTHING: {', '.join(quiet)}" if quiet else ""))
        self._refresh_sweep_label()

    def _last_temps(self):
        t = getattr(self, "last_temps_c", None)
        return None if t is None else np.asarray(t, float)

    def _refresh_sweep_label(self):
        if not self.session.sweeps:
            self.lbl_sweeps.setText("no sweeps recorded")
            self.btn_solve_roll.setEnabled(False)
            return
        bits = [f"{t} ({len(s)} pts)" for t, s in sorted(self.session.sweeps.items())]
        self.lbl_sweeps.setText("recorded: " + ", ".join(bits))
        self.btn_solve_roll.setEnabled(True)

    def on_clear_sweeps(self):
        self.session.sweeps.clear()
        self.session.pose_solution = None
        self.btn_apply_roll.setEnabled(False)
        self._refresh_sweep_label()
        self.session.log("roll sweeps cleared")

    def on_solve_roll(self):
        if not self.session.sweeps:
            return
        try:
            sol = opc.solve_roll(list(self.session.sweeps.values()), self.session.geom,
                                 self.spin_bearth.value(),
                                 dead=sorted(self.session.cal.dead),
                                 anisotropy=("assume_isotropic"
                                             if self.chk_isotropic.isChecked()
                                             else "solve"))
        except (ValueError, np.linalg.LinAlgError) as e:
            # Deliberately not a modal. Solving is a diagnostic step you may run
            # repeatedly while getting a sweep right, and a dialog you have to
            # dismiss in the middle of a live acquisition is worse than useless.
            self.session.log(f"roll solve failed: {e}")
            self.cal_report.setPlainText(f"Roll solve failed:\n\n{e}")
            self.session.pose_solution = None
            self.btn_apply_roll.setEnabled(False)
            return
        self.session.pose_solution = sol
        ids = opc.identify_faces(sol, self.session.geom)
        text = [sol.report(dead=self.session.cal.dead), ""]
        text.append("face mapping: " + ("agrees with probe_geometry.json"
                                        if ids["agrees"] else
                                        "DISAGREES at " + ", ".join(ids["mismatch"])))
        text.append(f"ambient uniformity: {opc.ambient_uniformity(sol)*100:.3f} "
                    "% of |B|  (a dirty spot shows up here, not in the residual)")
        self.cal_report.setPlainText("\n".join(text))
        self.btn_apply_roll.setEnabled(True)
        self.session.log("roll solve done; review the report before applying")

    def on_apply_roll(self):
        if self.session.pose_solution is None:
            return
        sol = self.session.pose_solution
        warn = []
        if not sol.identified[:, 2].all():
            warn.append("The axial column (chip +Y on every sensor here) was "
                        "NOT identified by the data and will be taken from the "
                        "nominal geometry.")
        if not sol.anisotropy_identified:
            warn.append("Transverse-vs-axial sensitivity was not pinned. "
                        "Inter-sensor matching is still valid; the absolute "
                        "axial ratio is not.")
        if sol.offset_leverage < 0.6:
            warn.append("The axial field barely changed between orientations, "
                        "so offsets and axial response are poorly separated.")
        if warn:
            r = QtWidgets.QMessageBox.question(
                self, "Apply roll calibration",
                "\n\n".join(warn) + "\n\nApply anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No)
            if r != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        try:
            self.session.cal.apply_pose_solution(sol)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Apply roll calibration", str(e))
            return
        g = sol.gains()
        self.session.log("roll calibration applied: matrix + offsets installed, "
                     f"magnet gain trim cleared; gain spread was "
                     f"{(np.nanmax(g)-np.nanmin(g))*100:.2f} %")
        self.chk_vcm.setChecked(self.session.cal.subtract_vcm)
        self._calibration_changed("the roll calibration")
        self.refresh_cal_report()

    def on_save_sweeps(self):
        if not self.session.sweeps:
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Save roll sweeps into", self.session.out_dir)
        if not d:
            return
        for tag, sw in sorted(self.session.sweeps.items()):
            path = sw.save(os.path.join(d, f"rollsweep_{tag}"))
            self.session.log(f"wrote {path}")

    def on_load_sweeps(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Load roll sweeps", self.session.out_dir, "Roll sweeps (*.npz)")
        for f in files:
            try:
                sw = opc.RollSweep.load(f)
            except (OSError, ValueError, KeyError) as e:
                self.session.log(f"could not load {f}: {e}")
                continue
            self.session.sweeps[sw.tag] = sw
            self.session.log(f"loaded sweep {sw.tag} from {os.path.basename(f)}")
        self._refresh_sweep_label()

    def on_clear_tare(self):
        self.session.cal.clear_tare()
        self.session.log("zero cleared")
        self._calibration_changed("the zero point")
        self.refresh_cal_report()

    def on_magnet_pass(self, on):
        if on:
            if self.session.source is None:
                self.btn_magnet.setChecked(False)
                return
            recent = self.session.roll.view()
            if recent.shape[0] < 2:
                self.btn_magnet.setChecked(False)
                self.lbl_magnet.setText("no data yet -- wait a moment and retry")
                return
            base = np.median(recent, axis=0).astype(np.float64)
            self.session.collecting = {"what": "magnet", "blocks": [], "n": 0,
                               "need": 0, "peak": None, "baseline": base,
                               "tag": None, "decim": self.decim}
            self.btn_magnet.setText("Stop magnet pass")
            self.lbl_magnet.setText("recording -- pass the magnet along the probe, "
                                    "then press stop")
        else:
            c = self.session.collecting
            self.session.collecting = None
            self.btn_magnet.setText("Start magnet pass")
            if not c or c.get("peak") is None:
                self.lbl_magnet.setText("no data captured")
                return
            self.session.magnet_peaks = np.asarray(c["peak"], float)
            live = self.session.cal.live_mask()
            rep = ocal.spread_report(self.session.magnet_peaks, live=live)
            n = rep.get("n_responding", 0)
            spread = rep.get("raw_spread")
            self.lbl_magnet.setText(
                f"{n} sensors responded, peak spread "
                f"{spread:.2f}x" if spread else f"{n} sensors responded")
            self.btn_apply_gain.setEnabled(True)
            self.session.log("magnet pass: peak |B| per sensor = "
                         + ", ".join(f"S{i+1}={v:.3f}"
                                     for i, v in enumerate(self.session.magnet_peaks)))
            self.refresh_cal_report()

    def on_apply_gain(self):
        if self.session.magnet_peaks is None:
            return
        w = None
        if self.chk_geom.isChecked():
            pt = (self.spin_mx.value(), self.spin_my.value(), self.spin_mz.value())
            w = self.session.geom.expected_response(pt, self.spin_exp.value())
        _corr, skipped = self.session.cal.cross_calibrate(self.session.magnet_peaks, weights=w)
        note = (f"kept their previous trim (no usable response): "
                f"{', '.join(skipped)}") if skipped else "every live sensor trimmed"
        self.session.log(f"gain trim applied using "
                     f"{'geometry-weighted' if w is not None else 'raw'} peaks; "
                     f"{note}")
        self.lbl_magnet.setText(f"gain trim applied -- {note}")
        self._calibration_changed("the gain trim")
        self.refresh_cal_report()

    def on_guided_magnet(self):
        """Open the guided routine, or explain what it needs first."""
        if self.session.stages is None:
            QtWidgets.QMessageBox.information(
                self, "Guided magnet calibration",
                "This routine drives the head past a fixed magnet, so it "
                "needs the stages. Press Connect — it brings up the carriers "
                "and the stages together — or connect them from the Stages "
                "tab.\n\nWithout motion, the hand magnet pass above is the "
                "alternative; it needs geometry weighting to be fair, which "
                "this one does not.")
            return
        # Visibility, not the reference, decides whether one is already open.
        # Belt and braces against the failure above: any future path that
        # hides this dialog without finishing it would otherwise wedge the
        # button, and the symptom -- a button that does nothing at all -- is
        # about as hard to diagnose from a bug report as it gets.
        if self._magnet_wizard is not None:
            if self._magnet_wizard.isVisible():
                self._magnet_wizard.raise_()
                self._magnet_wizard.activateWindow()
                return
            self._magnet_wizard.deleteLater()
            self._magnet_wizard = None
        self._magnet_wizard = MagnetWizard(self)
        self._magnet_wizard.finished.connect(
            lambda _: setattr(self, "_magnet_wizard", None))
        self._magnet_wizard.show()

    def on_show_superseded(self, on):
        for box in self._superseded_boxes:
            box.setVisible(bool(on))

    def on_clear_gain(self):
        self.session.cal.clear_gain()
        self.session.log("gain trim cleared")
        self._calibration_changed("the gain trim")
        self.refresh_cal_report()

    # ---- calibration state -----------------------------------------------
    def on_range_changed(self, row, value):
        self.session.cal.ranges_mt[row] = value
        self.session.log(f"S{row+1} range set to +/-{value:g} mT "
                     f"({ob.RANGE_TO_VPT[value]:g} V/T)")
        self._calibration_changed(f"the S{row+1} range")
        self.refresh_cal_report()

    def on_vcm_toggle(self, on):
        self.session.cal.subtract_vcm = bool(on)
        if not on:
            self.session.log("WARNING: VCM subtraction off -- readings now include "
                         "each chip's ~2.2 V virtual ground offset")
        self._calibration_changed("VCM subtraction")
        self.refresh_cal_report()

    def refresh_cal_report(self):
        lines = [self.session.cal.summary(), ""]
        z = self.session.cal.zero_mt
        g = self.session.cal.gain_corr
        lines.append(f"{'sensor':>7} {'range':>9} {'zero Bx':>9} {'zero By':>9} "
                     f"{'zero Bz':>9} {'gain trim':>10}")
        for s in range(N_SENSORS):
            lines.append(f"{'S'+str(s+1):>7} {self.session.cal.ranges_mt[s]:8.0f}mT "
                         f"{z[s,0]:9.4f} {z[s,1]:9.4f} {z[s,2]:9.4f} "
                         f"{g[s].mean():10.4f}")
        if self.session.magnet_peaks is not None:
            lines += ["", "last magnet pass, peak |B| per sensor [mT]:"]
            live = self.session.cal.live_mask()
            for s in range(N_SENSORS):
                tag = "" if live[s] else "   (excluded)"
                lines.append(f"{'S'+str(s+1):>7} {self.session.magnet_peaks[s]:10.4f}{tag}")
            rep = ocal.spread_report(self.session.magnet_peaks, live=live)
            if "raw_spread" in rep:
                lines.append(f"\nraw spread across responding sensors: "
                             f"{rep['raw_spread']:.2f}x")
            if self.chk_geom.isChecked():
                pt = (self.spin_mx.value(), self.spin_my.value(),
                      self.spin_mz.value())
                rep2 = ocal.spread_report(self.session.magnet_peaks, self.session.geom, pt,
                                          self.spin_exp.value(), live)
                if "corrected_spread" in rep2:
                    lines.append(
                        f"expected spread from 1/r^{self.spin_exp.value():g} "
                        f"geometry alone: {rep2['geometry_spread']:.2f}x")
                    lines.append(
                        f"spread left after removing geometry: "
                        f"{rep2['corrected_spread']:.2f}x  <- this part is "
                        f"electrical (gain register, EEPROM calibration, "
                        f"Hall bias), not mounting")
        self.cal_report.setPlainText("\n".join(lines))

    def on_save_cal(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save calibration", self.session.args.calibration, "JSON (*.json)")
        if path:
            self.session.cal.notes = f"saved from octobee_gui {time.strftime('%Y-%m-%d %H:%M')}"
            self.session.cal.save(path)
            self.session.log(f"calibration written to {path}")

    def on_load_cal(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load calibration", ".", "JSON (*.json)")
        if not path:
            return
        try:
            cal = ocal.Calibration.load(path)
        except (OSError, ValueError, TypeError) as exc:
            # Deliberately not load_or_default: an explicit Load that quietly
            # substituted +/-20 mT defaults would be the worst outcome here.
            # Keep the calibration already in force and say what went wrong.
            self.session.log(f"could not load {path}: {type(exc).__name__}: {exc}")
            QtWidgets.QMessageBox.warning(
                self, "Load calibration",
                f"{path} could not be read:\n\n{type(exc).__name__}: {exc}\n\n"
                f"The calibration already loaded is unchanged.")
            return
        self.session.cal = cal
        self.table.set_ranges(self.session.cal.ranges_mt)
        self.chk_vcm.setChecked(self.session.cal.subtract_vcm)
        self._calibration_changed("the whole calibration")
        self.refresh_cal_report()
        self.session.log(f"calibration loaded from {path}")

    def on_edit_geometry(self):
        path = self.session.args.geometry
        if not os.path.exists(path):
            self.session.geom.save(path)
        QtWidgets.QMessageBox.information(
            self, "Probe geometry",
            f"The tube layout lives in:\n\n{os.path.abspath(path)}\n\n"
            f"Edit it to match the real probe -- tube width, chip pitch, and "
            f"which sensor is on which face. The default face assignment has "
            f"NOT been verified on this hardware. Then press "
            f"'Reload geometry'.")

    def on_reload_geometry(self):
        self.session.geom = pgeom.Geometry.load_or_default(self.session.args.geometry)
        self.view3d.rebuild(self.session.geom)
        # The body that has to clear the coils just changed shape.
        self.tab_machine.set_geometry(self.session.geom)
        self.session.probe_cloud = None
        self.plot.geom = self.session.geom
        self.table.refresh_geometry(self.session.geom)
        if isinstance(self.session.source, DemoSource):
            self.session.source.geom = self.session.geom
        self.session.log(f"geometry reloaded from {self.session.args.geometry}")

    # ---- diagnostics ------------------------------------------------------
    # ---- data output ------------------------------------------------------
    def on_record(self, on):
        if on:
            if self.session.source is None:
                self.act_record.setChecked(False)
                return
            if self.tab_export.csv_enabled():
                p = orec.default_name("octobee", "csv", self.session.out_dir)
                self.session.csv_rec = orec.CsvRecorder(
                    p, self.session.out_rate, self.session.cal, self.session.geom,
                    tube_frame=self.tab_export.tube_frame(),
                    meta={"hosts": ",".join(self.session.source.hosts),
                          "stream_rate_hz": self.session.source.fs_hz})
                self.session.log(f"recording CSV to {p} at {self.session.out_rate:g} Hz")
            if self.tab_export.raw_enabled():
                p = orec.default_name("octobee", "bin", self.session.out_dir)
                self.session.raw_rec = orec.RawRecorder(
                    p, self.session.source.hosts, self.session.source.vpc,
                    [self.session.source.fs_hz] * len(self.session.source.hosts),
                    cal=self.session.cal)
                self.session.log(f"recording raw counts to {p}")
            if self.session.csv_rec is None and self.session.raw_rec is None:
                self.session.log("nothing selected to record -- see the Data output tab")
                self.act_record.setChecked(False)
        else:
            for rec, kind in ((self.session.csv_rec, "CSV"), (self.session.raw_rec, "raw")):
                if rec is not None:
                    p = rec.close()
                    size = os.path.getsize(p) / 1e6 if os.path.exists(p) else 0
                    self.tab_export.note(f"{p}  ({size:.2f} MB, {kind})")
            self.session.csv_rec = None
            self.session.raw_rec = None
            self.tab_export.set_recording_text("")

    def _update_rec_label(self):
        parts = []
        if self.session.csv_rec is not None:
            parts.append(f"CSV {self.session.csv_rec.n_rows} rows "
                         f"{self.session.csv_rec.size_bytes/1e6:.1f} MB")
        if self.session.raw_rec is not None:
            parts.append(f"raw {self.session.raw_rec.n_samples} samples "
                         f"{self.session.raw_rec.size_bytes/1e6:.1f} MB")
        txt = "  |  ".join(parts) if parts else ""
        self.lbl_rec.setText(("REC  " + txt) if parts else "")
        self.tab_export.set_recording_text(txt)

    def on_snapshot(self):
        if not isinstance(self.session.source, LiveSource):
            QtWidgets.QMessageBox.information(
                self, "Snapshot",
                "A full-rate snapshot needs the live hardware.")
            return
        if self.act_record.isChecked():
            self.act_record.setChecked(False)
        secs = self.tab_export.snapshot_seconds()
        path = orec.default_name("snapshot", "npz", self.session.out_dir)
        self.session.log(f"snapshot: stopping the stream, restoring the carriers' "
                     f"own clock, and capturing {secs:g} s losslessly")
        self.session.source.stop()
        self.session.source = None
        self.act_snapshot.setEnabled(False)
        self.lbl_state.setText("snapshot in progress...")
        self._snap_worker = SnapshotWorker(self.session.hosts, secs, path,
                                           self.session.prev_clkdiv)
        # The worker puts the clock back itself, so there is nothing left for
        # disconnect to restore. Clearing it here rather than on completion
        # means a failed snapshot cannot leave a stale value behind either.
        self.session.prev_clkdiv = {}
        self._snap_worker.done.connect(self.on_snapshot_done)
        self._snap_worker.start()

    def on_snapshot_done(self, path, error, fs_hz):
        self.act_snapshot.setEnabled(True)
        if error:
            self.session.log(f"snapshot failed: {error}")
            QtWidgets.QMessageBox.critical(self, "Snapshot failed", error)
        else:
            size = os.path.getsize(path) / 1e6
            self.tab_export.note(f"{path}  ({size:.2f} MB raw, {fs_hz/1e3:g} kSPS)")
        self.lbl_state.setText("disconnected -- press Connect to resume")
        self.act_disconnect.setEnabled(False)
        self.act_connect.setEnabled(True)

    # ---- misc UI ----------------------------------------------------------
    def _mark_dead_checkboxes(self, dead):
        for s, cb in enumerate(self.chk_sensors):
            bad = f"S{s+1}" in dead
            cb.setStyleSheet("color:#c05050;" if bad else "")
            cb.setToolTip("excluded: railed or stuck channels" if bad else "")

    def on_units(self, *_):
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

    def _set_all_sensors(self, on):
        for cb in self.chk_sensors:
            cb.blockSignals(True)
            cb.setChecked(on)
            cb.blockSignals(False)
        self.on_sensor_toggle()

    def on_3d_toggle(self, on):
        self.view3d.setVisible(bool(on))
        self._draw_ms = 0.0
        self.session.log("3D head on" if on else
                     "3D head off -- everything else carries on unchanged")

    def on_view_rate(self):
        hz = float(self.cmb_view.currentData())
        self.view_timer.setInterval(int(1000 / hz))
        self._draw_ms = 0.0
        self.session.log(f"view redraw rate {hz:g} Hz "
                     f"(acquisition and recording are unaffected)")

    def on_pause(self, on):
        self.paused = bool(on)
        self.session.log("view paused -- acquisition and recording continue"
                     if on else "view resumed")

    def on_out_rate(self):
        self.session.out_rate = float(self.cmb_out.currentData())
        self.session.roll.resize(max(4, int(self.session.out_rate * self.window_s)))
        self.session.log(f"output rate {self.session.out_rate:g} Hz "
                     f"(decimation {self.decim}x from the stream)")

    def on_window(self, v):
        self.window_s = float(v)
        self.session.roll.resize(max(4, int(self.session.out_rate * self.window_s)))

    def closeEvent(self, ev):
        if self.session.prof.enabled:
            print(self.session.prof.text())
            print(f"\nevent loop lag: mean {self.session.lag.mean_ms:.1f} ms, "
                  f"worst {self.session.lag.max_ms:.0f} ms -- {self.session.lag.verdict()}")
        if self.act_record.isChecked():
            self.act_record.setChecked(False)
        if self.motion.busy():
            # Ask before walking away from a moving machine. Closing used to
            # go straight through to stages.close(), which does not stop
            # anything -- the move is already in the controller and it runs to
            # completion whether or not this window still exists. So the
            # window would vanish and the head would carry on traversing with
            # nothing watching it and no stop button anywhere.
            reply = QtWidgets.QMessageBox.warning(
                self, "Something is still moving",
                "A stage job is still running.\n\n"
                "Closing stops every axis first — the move in progress is "
                "abandoned where it is, so re-home before trusting a "
                "position afterwards.\n\nClose and stop?",
                QtWidgets.QMessageBox.StandardButton.Close
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel)
            if reply != QtWidgets.QMessageBox.StandardButton.Close:
                ev.ignore()
                return
            self.motion.trigger("the window was closed while a job was running")
        for w in [*self._motion_workers, self._stage_worker]:
            if is_running(w):
                # Bounded: a worker wedged inside a DLL call must not stop the
                # window closing, or the only way out is Task Manager -- which
                # is the one exit that leaves the stages held and moving.
                w.wait(30000)
        self.stage_timer.stop()
        if self.session.stages is not None:
            # These are exclusive-open USB devices: leaving them held means the
            # Kinesis application will not start until this process dies.
            # close() stops each axis on the way out; see Stage.close.
            self.session.stages.close()
            self.session.stages = None
        self.on_disconnect()
        super().closeEvent(ev)
