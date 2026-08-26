"""
octobee/gui/tabs/stages.py -- the three translators, and the field map.

The map lives here rather than in a tab of its own because it is not a separate
activity: a raster is the stages being driven to a list of points, and every
guard that applies to a jog applies to it. Splitting them would have put the
envelope check, the homing prompt and the interlock on one side of a boundary
and the thing that needs them on the other.

What this tab does NOT own: the emergency stop and the motion-worker registry
(octobee/gui/estop.py -- they have to reach further than any tab), and the
toolbar. What it needs from those arrives as four callables, named at the top
of __init__.

Like Live, this module puts a second widget somewhere else:

    StagesTab      find, reference, jog, and the field map, in the tab strip
    StageJogPane   the three axes and nothing else, in the right-hand pane --
                   because "nudge it 2 mm and watch what the field does" is a
                   thing you do while looking at the Live plot or the Machine
                   view, and it should not cost a trip through the tab strip

The pane is a second face on this tab, not a second implementation. Every
button on it calls the method the tab's own button calls, so the interlock,
the busy check and the not-homed refusal are written once.
"""

import os
import time

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from octobee.gui.sources import LiveSource
from octobee.gui.workers import ScanWorker, StageWorker, _stage_set_for
from octobee.motion import scan as oscan
from octobee.motion import stage as ostage


class StagesTab(QtWidgets.QWidget):
    def __init__(self, session, motion, stage_mm, set_status,
                 stop_recording, set_snapshot_enabled, parent=None):
        """Thorlabs LTS300C control and motorised field mapping.

            Laid out in the order you have to do things: find the stages, tell the
            software which one is which axis, reference them, then scan. Homing sits
            behind a confirmation because it drives the carriage into a limit switch
            at speed and the software has no idea what is bolted to it.
            """
        super().__init__(parent)
        self.session = session
        self.motion = motion
        self._stage_mm = stage_mm
        self._set_status = set_status
        self._stop_recording = stop_recording
        self._set_snapshot_enabled = set_snapshot_enabled

        self.stage_combos = {}
        self._stage_pending = None
        self._stage_worker = None
        self._scan_worker = None
        self._scan_t0 = 0.0

        lay = QtWidgets.QVBoxLayout(self)

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

        # Lives in the window's right-hand pane, not in this layout. Built
        # here anyway so that everything which guards a move -- stage_action,
        # the not-homed refusal, sync_controls -- has one owner.
        self.jog_pane = StageJogPane(jog=self.on_stage_jog,
                                     goto=self.on_stage_goto,
                                     home=lambda: self.on_stage_home(None),
                                     stop=self.motion.stop)

        self._update_scan_estimate()
        self._set_stage_controls_enabled(False)

    def _set_stage_controls_enabled(self, on):
        """Connected or not. Everything finer than that is sync_controls."""
        for row in self.stage_rows.values():
            row["present"] = on and row["present"]
        self.btn_stage_disconnect.setEnabled(on)
        self.btn_stage_connect.setEnabled(not on)
        self.sync_controls()

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
        self.stage_action("connect", work)

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
        for ax in self.stage_rows:
            self.jog_pane.set_present(ax, False)
        self.session.log("stages: closed")

    def stage_action(self, what, fn):
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
        self.sync_controls()

    def on_stage_action_done(self, what, error):
        self.motion.retire(self._stage_worker)
        if error:
            self.session.log(f"stages: {what} failed — {error}")
            if what == "connect":
                self._stage_pending = None
                self.btn_stage_connect.setEnabled(True)
            self.sync_controls()
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
                self.jog_pane.set_present(ax, have,
                                          self.session.stages[ax].limit_mm
                                          if have else None)
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
        self.sync_controls()

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

        self.stage_action("set motion profile", apply)

    def on_stage_jog(self, axis, delta):
        if self.session.stages is None or axis not in self.session.stages.names:
            return
        st = self.session.stages[axis]
        self.stage_action(f"jog {axis} {delta:+g} mm",
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
        self.stage_action(f"move {axis} to {mm:g} mm",
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

        self.stage_action(f"home {', '.join(names)}", work)

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

    def sync_controls(self):
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
        # The right-hand pane is the same three axes under the same gate: it
        # must never offer a move this tab is refusing.
        self.jog_pane.set_live(live, {ax: row.get("present", False)
                                      for ax, row in self.stage_rows.items()})
        self.jog_pane.set_stop_enabled(self.session.stages is not None)

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
            self.jog_pane.set_axis_state(st.name, snap)
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

        self._stop_recording()
        if isinstance(self.session.source, LiveSource):
            self.session.source.stop()
        self.session.source = None
        self._set_status("field map in progress...")
        self.bar_scan.setRange(0, n)
        self.bar_scan.setValue(0)
        self.btn_scan_abort.setEnabled(True)
        self._set_snapshot_enabled(False)

        self._scan_worker = ScanWorker(
            self.session.hosts, self.session.stages, grid, self.spin_scan_s.value(),
            self.session.cal, self.spin_scan_settle.value(), self.session.prev_clkdiv,
            # Which coils were on, at what current, and where the probe was
            # bolted. None of it can be recovered from the numbers later, and
            # a map without it is a table of vectors at nowhere in particular.
            extra_meta={"machine": self.session.machine.to_scan_meta(
                self.session.coils, self._stage_mm(ignore_tracking=True))})
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
        self.sync_controls()

    def on_scan_progress(self, i, n, where, sem_ut):
        self.bar_scan.setValue(i)
        elapsed = time.time() - self._scan_t0
        eta = elapsed / max(i, 1) * (n - i)
        self.bar_scan.setFormat(f"%v / %m — {eta / 60:.0f} min left")
        self._set_status(f"field map {i}/{n}: {where}")
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
        self._set_snapshot_enabled(True)
        self.bar_scan.setFormat("%v / %m")
        if not self.motion.latched:
            self._set_status("disconnected")
        self.sync_controls()
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
        self.stage_action("connect", lambda emit: stages.open())
        return True


class StageJogPane(QtWidgets.QWidget):
    """The three axes, in the right-hand pane, next to what they move.

    Deliberately smaller than the Stages tab. There is no device list, no axis
    map and no field map here -- those are set up once and then not touched
    again. What does get touched constantly is "nudge it 2 mm and watch what
    the field does", and until now that meant leaving whichever view you were
    reading to do it.

    Nothing is duplicated but the widgets. Every button calls straight back
    into StagesTab, so the emergency-stop latch, the one-move-at-a-time guard
    and the refusal to move an unreferenced axis absolutely are each still
    written exactly once, in the tab.
    """

    AXES = ("x", "y", "z")

    def __init__(self, jog, goto, home, stop, parent=None):
        super().__init__(parent)
        self._jog = jog
        self._goto = goto

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 4)
        lay.setSpacing(4)

        head = QtWidgets.QLabel(
            "Stages — the same three axes as the Stages tab, and the same "
            "guards on them.")
        head.setWordWrap(True)
        head.setStyleSheet("color:#9aa3b2; font-size:11px;")
        lay.addWidget(head)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(3)
        for col, text in ((3, "step"), (6, "move to")):
            hdr = QtWidgets.QLabel(text)
            hdr.setStyleSheet("color:#9aa3b2; font-size:10px;")
            hdr.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
            grid.addWidget(hdr, 0, col)

        self.rows = {}
        for r, ax in enumerate(self.AXES, start=1):
            lbl = QtWidgets.QLabel(ax)
            lbl.setStyleSheet("font-weight:bold;")
            pos = QtWidgets.QLabel("—")
            pos.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                             | QtCore.Qt.AlignmentFlag.AlignVCenter)
            pos.setMinimumWidth(84)
            # Fixed-width digits: the readout ticks five times a second while
            # an axis is moving, and in a proportional font the whole row
            # jitters as the digits change under it.
            pos.setFont(QtGui.QFontDatabase.systemFont(
                QtGui.QFontDatabase.SystemFont.FixedFont))
            step = QtWidgets.QDoubleSpinBox()
            step.setRange(0.001, 100.0)
            step.setDecimals(3)
            step.setValue(1.0)
            step.setSuffix(" mm")
            step.setMinimumWidth(104)
            minus = QtWidgets.QPushButton("−")
            plus = QtWidgets.QPushButton("+")
            for b in (minus, plus):
                b.setMaximumWidth(30)
            target = QtWidgets.QDoubleSpinBox()
            target.setRange(0.0, 300.0)
            target.setDecimals(3)
            target.setSuffix(" mm")
            target.setMinimumWidth(104)
            go = QtWidgets.QPushButton("Go")
            go.setMaximumWidth(40)
            here = QtWidgets.QPushButton("⇣")
            here.setMaximumWidth(26)
            here.setToolTip("Put the current reading in the box next door, as "
                            "a starting point to edit.")

            minus.clicked.connect(
                lambda _, a=ax, s=step: self._jog(a, -s.value()))
            plus.clicked.connect(
                lambda _, a=ax, s=step: self._jog(a, +s.value()))
            go.clicked.connect(
                lambda _, a=ax, t=target: self._goto(a, t.value()))
            here.clicked.connect(lambda _, a=ax: self.take_position(a))

            for c, wdg in enumerate((lbl, pos, minus, step, plus, here, target,
                                     go)):
                grid.addWidget(wdg, r, c)
            self.rows[ax] = {"pos": pos, "step": step, "target": target,
                             "present": False, "mm": None, "trusted": True,
                             "widgets": (minus, step, plus, here, target, go)}
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.btn_home = QtWidgets.QPushButton("Home all")
        self.btn_home.setToolTip(
            "Drives every axis into its limit switch, one at a time, across "
            "the whole travel. It asks first — the probe and its cabling "
            "have to be clear of the entire range of movement.")
        self.btn_home.clicked.connect(lambda: home())      # noqa: PLW0108
        self.btn_stop = QtWidgets.QPushButton("Stop moving")
        self.btn_stop.setStyleSheet(
            "background:#7a1f1f; color:#fff; font-weight:bold;")
        self.btn_stop.setToolTip(
            "End the move in progress without latching the machine off. "
            "Profiled, so positions stay trustworthy and nothing needs "
            "re-homing.\n\n"
            "For 'something is wrong', use EMERGENCY STOP in the top right "
            "(or Esc).")
        self.btn_stop.clicked.connect(lambda: stop())       # noqa: PLW0108
        row.addWidget(self.btn_home)
        row.addWidget(self.btn_stop)
        lay.addLayout(row)

        self.lbl_note = QtWidgets.QLabel("stages not connected")
        self.lbl_note.setWordWrap(True)
        self.lbl_note.setStyleSheet("color:#9aa3b2; font-size:11px;")
        lay.addWidget(self.lbl_note)

        self.set_live(False)
        self.set_stop_enabled(False)

    def take_position(self, axis):
        """Copy the live reading into that axis's target box."""
        mm = self.rows[axis]["mm"]
        if mm is not None:
            self.rows[axis]["target"].setValue(mm)

    def set_present(self, axis, present, limits=None):
        """Say whether this axis exists, and what it is allowed to reach.

        The soft limit rather than the travel, exactly as on the tab: a box
        that offers a number the axis will refuse is an invitation to find
        that out by pressing Go.
        """
        row = self.rows.get(axis)
        if row is None:
            return
        row["present"] = bool(present)
        if present and limits is not None:
            row["target"].setRange(float(limits[0]), float(limits[1]))
        if not present:
            row["pos"].setText("—")
            row["pos"].setStyleSheet("")
            row["mm"] = None
            row["trusted"] = True

    def set_live(self, live, present=None):
        """Enable the move controls, per axis, under the tab's own gate."""
        for ax, row in self.rows.items():
            if present is not None:
                row["present"] = bool(present.get(ax, row["present"]))
            for wdg in row["widgets"]:
                wdg.setEnabled(bool(live) and row["present"])
        self.btn_home.setEnabled(bool(live))
        self._note()

    def set_stop_enabled(self, on):
        self.btn_stop.setEnabled(bool(on))

    def set_axis_state(self, axis, snap):
        """One axis's line of the stage table, off the DLL's polling cache."""
        row = self.rows.get(axis)
        if row is None:
            return
        mm = float(snap["position_mm"])
        row["mm"] = mm
        row["trusted"] = bool(snap["trusted"])
        row["pos"].setText(f"{mm:9.3f} mm")
        # Amber while it is moving, red while the counter cannot be believed.
        # The number is shown either way -- an unreferenced axis still has a
        # reading, and blanking it would hide exactly the situation someone is
        # trying to understand -- but it must not look like a coordinate.
        row["pos"].setStyleSheet(
            "color:#e07a5f;" if not snap["trusted"]
            else "color:#e8c547;" if snap["moving"] else "")
        self._note()

    def _note(self):
        """The one line under the buttons: what is stopping a move."""
        present = [ax for ax, row in self.rows.items() if row["present"]]
        if not present:
            self.lbl_note.setText("stages not connected")
            return
        unref = [ax for ax in present if not self.rows[ax]["trusted"]]
        self.lbl_note.setText(
            f"{', '.join(unref)} not referenced — jogging is relative and "
            f"works, Go is refused until it is homed" if unref else "")
