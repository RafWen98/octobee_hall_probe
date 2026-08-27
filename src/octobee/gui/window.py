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

from octobee.calib import convert as ocal
from octobee import live as olive
from octobee import record as orec
from octobee.gui.estop import MotionControl, is_running
from octobee.gui.constants import (
    DEFAULT_VIEW_HZ,
    HEALTH_PERIOD_S,
    HEALTH_WINDOW_S,
    MAX_VIEW_INTERVAL_MS,
    OUT_RATES,
    RAW_HISTORY_S,
    STREAM_RATES,
    VIEW_RATES,
)
from octobee.gui.rolling import Rolling
from octobee.gui.session import Session
from octobee.gui.tabs.calibration import CalibrationTab
from octobee.gui.tabs.export import ExportTab
from octobee.gui.tabs.health import HealthTab
from octobee.gui.tabs.help import HelpTab
from octobee.gui.tabs.live import LiveTab, ProbeHeadPane
from octobee.gui.tabs.machine import MachineTab
from octobee.gui.tabs.stages import StagesTab
from octobee.gui.tabs.profile import ProfileTab
from octobee.gui.sources import DemoSource, LiveSource, ReplaySource
from octobee.gui.widgets.log import LogPane
from octobee.gui.widgets.sensors import SensorTable
from octobee.gui.workers import (
    ConnectWorker,
    SnapshotWorker,
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
        self._magnet_wizard = None
        # The "here is where it went" box, and the one state in which it must
        # not appear: closing the window stops a running recording on the way
        # out, and a dialog raised during teardown is a window that will not
        # close.
        self._saved_box = None
        self._closing = False

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
        self.stage_timer.timeout.connect(self.tab_stages.refresh_stage_table)
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
        self.tab_calib.report_source()

    def _stage_mm(self, **kw):
        """Where the head is, in stage millimetres.

        A method rather than a bound reference because the Machine tab is
        built after the Stages tab that asks it.
        """
        return self.tab_machine.stage_mm(**kw)

    def _geometry_changed(self):
        """The probe geometry was reloaded; rebuild everything that draws it.

        Four widgets across three tabs show where the sensors are. The
        Calibration tab is where geometry gets reloaded, but it has no business
        knowing which of them exist -- it says the geometry moved and this
        decides what that means.
        """
        self.probe_pane.set_geometry(self.session.geom)
        self.tab_live.set_geometry(self.session.geom)
        self.table.refresh_geometry(self.session.geom)
        self.tab_machine.set_geometry(self.session.geom)

    def _set_status(self, text):
        """Put a line in the status bar.

        A method, not a bound setText: the status bar is built at the end of
        _build_ui, after the tabs that report into it.
        """
        self.lbl_state.setText(text)

    def _stop_recording(self):
        """Take the Record button down. The toolbar is the window's."""
        if self.act_record.isChecked():
            self.act_record.setChecked(False)

    # ---- construction ----------------------------------------------------
    def _build_ui(self):
        self._build_toolbar()
        self.log_pane = LogPane()
        # Everything that wants to say what it did goes through the
        # session from here on; nothing else needs the widget.
        self.session.attach_log(self.log_pane)

        self.probe_pane = ProbeHeadPane(self.session)
        self.tab_live = LiveTab(self.session)
        self.table = SensorTable(self.session.geom)
        # connected once the Calibration tab exists
        self.table.set_ranges(self.session.cal.ranges_mt)

        left = QtWidgets.QTabWidget()
        left.addTab(self.tab_live, "Live")
        left.addTab(self.table, "Sensors")
        self.tab_calib = CalibrationTab(self.session,
                                        set_ranges=self.table.set_ranges)
        self.tab_calib.calibration_changed.connect(self._calibration_changed)
        self.tab_calib.geometry_changed.connect(self._geometry_changed)
        self.table.range_changed.connect(self.tab_calib.on_range_changed)
        left.addTab(self.tab_calib, "Calibration")
        self.tab_health = HealthTab(self.session, self._raw_arrays)
        self.tab_health.wants_focus.connect(
            lambda: self.tabs.setCurrentWidget(self.tab_health))
        left.addTab(self.tab_health, "Diagnostics")
        self.tab_stages = StagesTab(
            self.session, self.motion,
            stage_mm=self._stage_mm,
            set_status=self._set_status,
            stop_recording=self._stop_recording,
            set_snapshot_enabled=self.act_snapshot.setEnabled)
        left.addTab(self.tab_stages, "Stages")
        self.tab_machine = MachineTab(self.session)
        # Built after the Stages tab, so it cannot be handed to it. What the
        # Machine tab needs to know is only that the stages came or went --
        # its "Calibrate encoders" button is dead without them, and deciding
        # that once at build time left it dead for the whole session.
        self.tab_stages.stages_changed.connect(self.tab_machine.refresh_encoders)
        left.addTab(self.tab_machine, "Machine")
        self.tab_export = ExportTab(
            self.session,
            sensor_table=lambda: self._last_table,
            magnet_geometry=self.tab_calib.magnet_geometry_correction)
        self.tab_export.snapshot_requested.connect(self.on_snapshot)
        self.tab_health.exported.connect(self.tab_export.note)
        left.addTab(self.tab_export, "Data output")
        self.tab_profile = ProfileTab(
            self.session,
            gl_info=self.probe_pane.gl_info)
        left.addTab(self.tab_profile, "Profile")
        left.addTab(self.log_pane, "Log")
        self.tab_help = HelpTab(self.session)
        left.addTab(self.tab_help, "Help")
        left.setCurrentIndex(0)
        self.tabs = left

        # The right-hand column: what the probe is reading, and the three
        # axes that decide where it reads it. Both are things you want while
        # working in some other tab -- moving the head 2 mm and watching the
        # field answer is one action, and it used to cost a trip to the
        # Stages tab and back to see the result.
        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right.addWidget(self.probe_pane)
        right.addWidget(self.tab_stages.jog_pane)
        right.setStretchFactor(0, 1)
        right.setStretchFactor(1, 0)
        # Collapsible: the jog pane is the one thing here somebody may never
        # use, on a bench with no stages on it.
        right.setSizes([760, 190])

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
        # Icons beside the text, not instead of it: the only icon in here is
        # the recording dot, and a toolbar of unlabelled buttons would be a
        # steep price for one indicator.
        tb.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
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
        self.act_tare.triggered.connect(lambda: self.tab_calib.start_collect("tare", 2.0))
        tb.addAction(self.act_tare)

        self.act_record = QtGui.QAction("Record", self)
        self.act_record.setCheckable(True)
        self.act_record.setToolTip(
            "Start writing what is arriving to disk, in whichever formats are "
            "ticked on the Data output tab. A red dot appears here while a "
            "file is actually open, and stopping says where it was written.")
        self.act_record.toggled.connect(self.on_record)
        self._refresh_record_dot()
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

    # ---- stages ----------------------------------------------------------
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
        self.tab_stages.sync_controls()

    # ---- field map -------------------------------------------------------
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
        self.tab_stages.connect_stages(quiet=True)

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
                # This hint used to be appended to EVERY connect failure,
                # which is how a carrier answering "no such knob" a minute
                # after a reboot sent an operator hunting for a Phoebus
                # capture that was not running. A hint shown unconditionally
                # carries no information and costs the reader time. The other
                # two stream failures name port 4210 in their own message; a
                # stream that closed on open is the one that does not.
                hint = ("\n\nThe stream closed as soon as it opened, so "
                        "something else owns port 4210 on that box -- "
                        "usually a Phoebus 'Streaming Capture' left running."
                        if "stream closed" in error else "")
                QtWidgets.QMessageBox.critical(
                    self, "Connect failed", f"{error}{hint}")
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
        self.tab_live.refresh_units()
        self.act_disconnect.setEnabled(True)
        self.act_connect.setEnabled(kind != "live")
        self.lbl_state.setText(f"{kind}: {', '.join(source.hosts)} "
                               f"@ {source.fs_hz/1000:g} kSPS")
        self._reset_buffers()
        # The other half of what the encoder line says: which columns are
        # arriving, and from which carrier. Both are properties of the source,
        # so both are unknown until there is one.
        self.tab_machine.refresh_encoders()
        self.session.log(f"{kind} source running at {source.fs_hz/1000:g} kSPS, "
                     f"decimating to {self.session.out_rate:g} Hz for display and CSV")

    def on_disconnect(self):
        if self.act_record.isChecked():
            self.act_record.setChecked(False)
        if self.session.source is not None:
            self.session.source.stop()
            self.session.source = None
        self.tab_machine.refresh_encoders()
        # Symmetry with Connect: one button owns the whole rig. Leaving the
        # stages open would hold the USB devices against the Kinesis app and
        # against the next run of this window, which looks like a cabling
        # fault rather than a stale handle.
        if self.session.stages is not None:
            self.tab_stages.on_stage_disconnect()
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
        # The carry belongs to the old stream. Keeping it would splice
        # samples from two different sample rates into one output row.
        self.session.new_decimator()
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
        self.probe_pane.reset_scale()
        self.tab_live.refresh_units()
        self._roll_over_csv(what)

    def _new_csv_recorder(self, path, meta):
        """One CsvRecorder, however the recording came to be started.

        Both callers -- pressing Record, and rolling over because the
        calibration changed under a running one -- have to hand it the same
        encoder state, or the second file would quietly lose the position
        columns the first one had.
        """
        return orec.CsvRecorder(
            path, self.session.out_rate, self.session.cal, self.session.geom,
            tube_frame=self.tab_export.tube_frame(),
            meta=meta, samples_per_row=self.session.decim(),
            encoders=self.session.encoders,
            enc_datum=self.session.enc_datum,
            enc_columns=int(
                getattr(self.session.source, "enc_columns", 0) or 0))

    def _roll_over_csv(self, what):
        """Close the CSV in progress and open a fresh one with a new header."""
        if self.session.csv_rec is None:
            return
        old = self.session.csv_rec
        rows = old.n_rows
        old.close()
        path = orec.default_name("octobee", "csv", self.session.out_dir)
        # The datum is deliberately NOT retaken here. The encoder stream is
        # not reset by a calibration change, so the counts carry straight on;
        # keeping the original datum is what makes the position columns of the
        # two files continue each other rather than restart at the same place.
        self.session.csv_rec = self._new_csv_recorder(
            path,
            {"hosts": ",".join(self.session.source.hosts) if self.session.source else "",
             "stream_rate_hz": self.session.source.fs_hz if self.session.source else 0.0,
             "continues": os.path.basename(old.path),
             "rolled_over_because": f"{what} changed"})
        self.session.log(
            f"{what} changed while recording: closed {os.path.basename(old.path)} "
            f"at {rows} rows and started {os.path.basename(path)}, so each file "
            f"matches the calibration named in its own header. The raw .bin is "
            f"unaffected -- it holds counts, which no calibration changes.")

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
                bd = self.session.decimator.push(b)
                # Before the early return, not after: the encoder decimator
                # only stays in step with the field one if it is fed on every
                # tick the field one is. Skipping it on a tick that produced
                # no rows would put the counts a block behind for the rest of
                # the session, and nothing downstream could tell.
                enc = self._tick_encoders(blocks[0].shape[0])
                if enc is not None and len(enc):
                    self.session.enc_counts_now = np.asarray(enc[-1], float)
                    self.tab_machine.note_encoder_counts(enc)
                if bd.shape[0] == 0:
                    return
                self.session.roll.push(bd.astype(np.float32))

            if self.session.csv_rec is not None:
                with self.session.prof.time("  write CSV"):
                    # Same rows, same decimation factor, same tick: the counts
                    # go into the file beside the field they were latched with.
                    self.session.csv_rec.write(bd, enc)

            # A volume sweep logs from here rather than owning the carriers, so
            # the live plot and any recording carry on through it. `logging` is
            # set by the motion thread and goes down before the return move, so
            # what lands here is the swept line and not the way back.
            runner = self.session.volume_runner
            if runner is not None and runner.logging:
                with self.session.prof.time("  log volume sweep"):
                    # Stamped from the moment the block arrived, counting
                    # backwards at the output rate. That clock is only used
                    # for the fallback -- to line these rows up against stage
                    # polls read over USB. Where the encoders are calibrated
                    # the counts below carry their own position, latched when
                    # the sample was converted, and no clock comes into it.
                    n = bd.shape[0]
                    self.session.volume_log.add_block(
                        bd, time.time() - n / max(self.session.out_rate, 1e-9),
                        runner.line_index, counts=enc)

            if self.session.collecting is not None:
                with self.session.prof.time("  magnet/tare collect"):
                    self.tab_calib.collect_block(b, grouped)

    def _tick_encoders(self, n_samples):
        """Continuous, decimated encoder counts for this tick's rows, or None.

        Returns an array whose row count matches the field's, because both
        decimators were built with the same factor and have now been fed the
        same number of samples. Where the source has no encoders -- acq1001_694
        alone, a replay, the synthetic probe -- it returns None and everything
        downstream falls back to asking the controllers where they are.
        """
        source, stream = self.session.source, self.session.enc_stream
        raw = getattr(source, "enc", None)
        if stream is None or raw is None:
            return None
        if raw.shape[0] != n_samples:
            # The source promises these are the same rows; if that ever stops
            # being true the honest thing is to drop them rather than log a
            # position against the wrong sample.
            self.session.log(
                f"WARNING: {raw.shape[0]} encoder rows against {n_samples} "
                f"analogue rows on one tick -- encoder positions dropped")
            return None
        with self.session.prof.time("  unwrap + decimate encoders"):
            return self.session.enc_decimator.push(stream.push(raw))

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
            self.probe_pane.draw(recent)
            self.tab_live.draw(recent)
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
                        self.tab_live.mark_dead(dead)
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

        self.tab_live.set_dead(self.session.cal.dead)
        self.probe_pane.set_dead(self.session.cal.dead)

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
    # ---- Earth-field roll calibration ------------------------------------
    # ---- calibration state -----------------------------------------------
    # ---- diagnostics ------------------------------------------------------
    # ---- data output ------------------------------------------------------
    def on_record(self, on):
        if on:
            if self.session.source is None:
                self.act_record.setChecked(False)
                return
            if self.tab_export.csv_enabled():
                p = orec.default_name("octobee", "csv", self.session.out_dir)
                self.session.enc_datum = self._encoder_datum()
                self.session.csv_rec = self._new_csv_recorder(
                    p, {"hosts": ",".join(self.session.source.hosts),
                        "stream_rate_hz": self.session.source.fs_hz})
                self.session.log(f"recording CSV to {p} at {self.session.out_rate:g} Hz")
                self._log_encoder_columns()
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
            saved = []
            csv_rec = self.session.csv_rec
            if (csv_rec is not None and csv_rec.enc_columns
                    and csv_rec.n_unpaired):
                # Not fatal and not silent: the field in those rows is good,
                # but their position columns are empty, and that is worth
                # knowing before the file is handed to someone.
                self.session.log(
                    f"WARNING: {csv_rec.n_unpaired} of {csv_rec.n_rows} rows "
                    f"were written with no encoder counts against them -- "
                    f"their position columns are blank")
            for rec, kind in ((self.session.csv_rec, "CSV"), (self.session.raw_rec, "raw")):
                if rec is not None:
                    p = rec.close()
                    size = os.path.getsize(p) / 1e6 if os.path.exists(p) else 0
                    self.tab_export.note(f"{p}  ({size:.2f} MB, {kind})")
                    saved.append((kind, p, size))
            self.session.csv_rec = None
            self.session.raw_rec = None
            self.tab_export.set_recording_text("")
            self._report_saved(saved)
        self._refresh_record_dot()

    def _encoder_datum(self):
        """Pair the counts arriving now with the controllers, axis by axis.

        Returns {axis: (counts, mm)} for every calibrated axis that can be
        anchored, which makes that axis's column in the CSV an absolute
        position; an axis left out gets travel from the first row instead, and
        the column name says so.

        Taken once, here, and standing still is not guaranteed -- Record can be
        pressed with the rig moving. That is not a reason to refuse a datum:
        the error it costs is the USB latency on one controller read, tens of
        milliseconds of motion, which is a common offset on that axis and not
        a per-sample one. Every sample after it is still spaced by counts.

        An axis whose counter is not trusted is skipped rather than used. The
        controller will answer position_mm regardless -- steps lost to a stall
        or an immediate stop leave the homed bit set and the number wrong --
        and anchoring to that produces a column that is confidently somewhere
        the head never was.
        """
        counts = self.session.enc_counts_now
        stages = self.session.stages
        if counts is None or not self.session.encoders or stages is None:
            return {}
        out = {}
        for axis, spec in self.session.encoders.axes.items():
            st = stages.axes.get(axis)
            if st is None or spec["column"] >= len(counts):
                continue
            try:
                if not st.position_trusted:
                    continue
                out[axis] = (float(counts[spec["column"]]),
                             float(st.position_mm))
            except Exception as e:      # a controller that has gone away
                self.session.log(f"no encoder datum for {axis}: {e}")
        return out

    def _log_encoder_columns(self):
        """Say what the position columns in the new file mean, at open."""
        rec = self.session.csv_rec
        if rec is None or not rec.enc_columns:
            return
        # Both taken from the columns the file actually has, not from what was
        # asked for: a datum for an axis whose column is not in the stream
        # produces no column, and saying otherwise here would be the one place
        # the operator was told about it.
        absolute = [a for a in rec.enc_axes if a in rec.enc_datum]
        relative = [a for a in rec.enc_axes if a not in rec.enc_datum]
        parts = [f"{rec.enc_columns} encoder count column(s)"]
        if absolute:
            names = "/".join(a.upper() for a in absolute)
            parts.append(f"absolute {names}_mm anchored to the controllers")
        if relative:
            parts.append(f"{'/'.join(a.upper() for a in relative)}_rel_mm as "
                         f"travel from the first row (no trusted controller "
                         f"position to anchor to -- home the stages before "
                         f"recording if you want absolute)")
        if not rec.enc_axes:
            parts.append("no counts/mm scale fitted, so counts only -- run "
                         "the encoder calibration on the Machine tab for "
                         "millimetres")
        self.session.log("recording position: " + "; ".join(parts))

    def _refresh_record_dot(self):
        """A red dot on the Record button while a file is actually open.

        Driven by the recorders, not by the button's checked state, and those
        are not the same thing: pressing Record with nothing ticked on the
        Data output tab puts the button down for as long as it takes to find
        that out, and a dot in that moment would be a lie about where the data
        went. Recording is also stopped from four other places -- a field map
        starting, a snapshot, Disconnect, closing the window -- and this way
        each of them gets the indicator right by doing nothing.
        """
        live = (self.session.csv_rec is not None
                or self.session.raw_rec is not None)
        self.act_record.setIcon(self._dot_icon("#e5383b") if live
                                else QtGui.QIcon())
        self.act_record.setText("Recording" if live else "Record")

    @staticmethod
    def _dot_icon(colour, size=12):
        """A filled circle, drawn rather than shipped as a file.

        Twelve pixels of solid colour is not worth an asset, a path to resolve
        and a way for the installed copy to be missing it.
        """
        pm = QtGui.QPixmap(size, size)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(colour))
        p.drawEllipse(1, 1, size - 2, size - 2)
        p.end()
        return QtGui.QIcon(pm)

    def _report_saved(self, saved):
        """Say where the recording went, in a box that has to be dismissed.

        The Log and the Data output tab both already carry the path, and both
        were being missed: a capture ends, the operator goes looking for the
        file, and the one place they are not looking is the tab they were not
        on. A file that has just been written is worth one modal.

        Non-blocking (open() rather than exec()) on purpose. Stopping a
        recording is also the first step of taking a snapshot and of starting
        a field map, and neither of those may sit and wait for someone to
        press OK.
        """
        if not saved or self._closing:
            return
        where = os.path.dirname(os.path.abspath(saved[0][1])) or "."
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Information)
        box.setWindowTitle("Recording saved")
        box.setText(f"{len(saved)} file{'s' if len(saved) > 1 else ''} written "
                    f"to\n{where}")
        box.setInformativeText("\n".join(
            f"{os.path.basename(p)}   ({size:.2f} MB, {kind})"
            for kind, p, size in saved))
        btn_open = box.addButton("Open folder",
                                 QtWidgets.QMessageBox.ButtonRole.ActionRole)
        btn_ok = box.addButton(QtWidgets.QMessageBox.StandardButton.Ok)
        box.setDefaultButton(btn_ok)
        # Named explicitly. With two buttons and neither of them a Cancel,
        # QMessageBox finds no escape button and then quietly ignores its own
        # close event -- so Esc and the title-bar cross would both do nothing.
        box.setEscapeButton(btn_ok)
        box.finished.connect(
            lambda _r, b=box, w=where: self._saved_box_done(b, btn_open, w))
        # Held on the window: a QMessageBox opened without exec() is only kept
        # alive by something referring to it.
        self._saved_box = box
        box.open()

    def _saved_box_done(self, box, btn_open, where):
        if box.clickedButton() is btn_open:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(where))
        if self._saved_box is box:
            self._saved_box = None
        box.deleteLater()

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
        self.session.new_decimator()
        self.session.log(f"output rate {self.session.out_rate:g} Hz "
                     f"(decimation {self.session.decim()}x from the stream)")

    def on_window(self, v):
        self.window_s = float(v)
        self.session.roll.resize(max(4, int(self.session.out_rate * self.window_s)))

    def closeEvent(self, ev):
        if self.session.prof.enabled:
            print(self.session.prof.text())
            print(f"\nevent loop lag: mean {self.session.lag.mean_ms:.1f} ms, "
                  f"worst {self.session.lag.max_ms:.0f} ms -- {self.session.lag.verdict()}")
        # Before the recording is stopped, not after: stopping it is what
        # would otherwise raise the "saved to" box, and that box would then be
        # asking to be dismissed by a window that is already going away. The
        # path is still in the Log and on the Data output tab.
        self._closing = True
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
                # Still open, and still a window someone will record from.
                self._closing = False
                ev.ignore()
                return
            self.motion.trigger("the window was closed while a job was running")
        for w in self.motion.all_workers():
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
