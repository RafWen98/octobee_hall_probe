"""
octobee/gui/tabs/machine.py -- where the head is inside the coil set, and
which coils are live.

A field map is a table of vectors at rig millimetres, and rig millimetres mean
nothing on their own: the same numbers describe a useful measurement and a
useless one depending on where the probe was bolted and what was switched on.
This tab is where both are declared, and it draws the consequence rather than
asking anyone to picture it.

It reads the stage positions when asked to track them, and answers stage_mm()
for the scanner, which needs to know the clearance before it commands a move.
"""

import os

from PyQt6 import QtCore, QtGui, QtWidgets

from octobee import machine as omach
from octobee.gui.widgets.machine3d import MachineView3D
from octobee.motion import stage as ostage


class MachineTab(QtWidgets.QWidget):
    def __init__(self, session, parent=None):
        """Where the head is inside the coil set, and which coils are live.

            A field map is a table of vectors at rig millimetres, and rig
            millimetres mean nothing on their own: the same numbers describe a
            useful measurement and a useless one depending on where the probe was
            bolted and what was switched on. This tab is where both are declared,
            and it draws the consequence rather than asking anyone to picture it.

            Laid out in the order the answers arrive: which machine, which coils
            are carrying current, then where the probe sits in it.
            """
        super().__init__(parent)
        self.session = session
        self._machine_key = None
        self._machine_quiet = False
        self._machine_travel_taken = False
        self.machine_view = MachineView3D(session.geom,
                                          profiler=session.prof)

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(split)

        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)

        note = QtWidgets.QLabel(
            "The coil file is in machine coordinates; the probe is placed by "
            "saying where its mounting flange sits in them. The stage reading "
            "is added along the rig's own axes, so the drawn head follows a "
            "jog. Nothing here is measured — it is the frame the measurement "
            "is written down in, and it is only as good as the numbers typed "
            "into it.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#9aa3b2; font-size:11px;")
        lay.addWidget(note)

        # ---- 1. the coil set ----
        g1 = QtWidgets.QGroupBox("1. Coil set")
        f1 = QtWidgets.QGridLayout(g1)
        self.ed_coil_file = QtWidgets.QLineEdit(self.session.machine.coil_file)
        self.ed_coil_file.setPlaceholderText(
            "a simsopt configuration file, e.g. designA_after_scaled.json")
        btn_browse = QtWidgets.QPushButton("Browse…")
        btn_browse.clicked.connect(self.on_machine_browse)
        btn_load = QtWidgets.QPushButton("Load")
        btn_load.clicked.connect(lambda: self.on_machine_load())  # noqa: PLW0108
        f1.addWidget(self.ed_coil_file, 0, 0, 1, 2)
        f1.addWidget(btn_browse, 0, 2)
        f1.addWidget(btn_load, 0, 3)

        self.lbl_coil_summary = QtWidgets.QLabel("no coil set loaded")
        self.lbl_coil_summary.setWordWrap(True)
        self.lbl_coil_summary.setStyleSheet("color:#9aa3b2; font-size:11px;")
        f1.addWidget(self.lbl_coil_summary, 1, 0, 1, 4)

        f1.addWidget(QtWidgets.QLabel("winding radius"), 2, 0)
        self.spin_coil_radius = QtWidgets.QDoubleSpinBox()
        self.spin_coil_radius.setRange(0.5, 500.0)
        self.spin_coil_radius.setDecimals(1)
        self.spin_coil_radius.setSuffix(" mm")
        self.spin_coil_radius.setValue(self.session.machine.coil_radius_mm)
        self.spin_coil_radius.setToolTip(
            "The circular cross-section swept along each coil centreline — "
            "the volume the probe cannot enter. A simsopt file has no such "
            "number in it: the optimiser works with infinitely thin "
            f"filaments, so this starts at {omach.DEFAULT_COIL_RADIUS_MM:g} mm "
            "and is a guess until the real conductor has been measured. "
            "Clearance is reported to this surface, not to the centreline.")
        self.spin_coil_radius.valueChanged.connect(self.on_machine_radius)
        f1.addWidget(self.spin_coil_radius, 2, 1)
        lay.addWidget(g1)

        # ---- 2. currents ----
        g2 = QtWidgets.QGroupBox("2. Which coils are carrying current")
        f2 = QtWidgets.QGridLayout(g2)
        f2.addWidget(QtWidgets.QLabel("configuration"), 0, 0)
        self.cmb_coil_config = QtWidgets.QComboBox()
        self.cmb_coil_config.setToolTip(
            "A simsopt file usually holds the same coils several times over, "
            "once per current set the optimiser was working with. The "
            "geometry is identical; only the currents differ.")
        self.cmb_coil_config.currentIndexChanged.connect(self.on_machine_config)
        f2.addWidget(self.cmb_coil_config, 0, 1)
        f2.addWidget(QtWidgets.QLabel("× scale"), 0, 2)
        self.spin_coil_scale = QtWidgets.QDoubleSpinBox()
        self.spin_coil_scale.setRange(-1000.0, 1000.0)
        self.spin_coil_scale.setDecimals(5)
        self.spin_coil_scale.setSingleStep(0.01)
        self.spin_coil_scale.setValue(self.session.machine.current_scale)
        self.spin_coil_scale.setToolTip(
            "Every current in the chosen configuration is multiplied by this. "
            "The file's numbers are the design point in amp-turns; a bench "
            "test at a few hundred amp-turns is that design point scaled down "
            "by a factor of a hundred or more. The ratios between coils stay "
            "as optimised, which is what makes a single number enough.")
        self.spin_coil_scale.valueChanged.connect(self.on_machine_scale)
        f2.addWidget(self.spin_coil_scale, 0, 3)

        self.tbl_coils = QtWidgets.QTableWidget(0, 4)
        self.tbl_coils.setHorizontalHeaderLabels(
            ["coil", "where", "amp-turns", "curve"])
        self.tbl_coils.verticalHeader().setVisible(False)
        self.tbl_coils.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_coils.horizontalHeader().setStretchLastSection(True)
        # Tall enough that a six-coil machine needs no scrolling: the point
        # of the table is to be able to see at a glance what is switched on.
        self.tbl_coils.setMaximumHeight(230)
        self.tbl_coils.itemChanged.connect(self.on_machine_coil_toggled)
        f2.addWidget(self.tbl_coils, 1, 0, 1, 4)

        btn_all_on = QtWidgets.QPushButton("All on")
        btn_all_on.clicked.connect(lambda: self.on_machine_all(True))
        btn_all_off = QtWidgets.QPushButton("All off")
        btn_all_off.clicked.connect(lambda: self.on_machine_all(False))
        hint = QtWidgets.QLabel(
            "A coil that is switched off is still solid copper: clearance is "
            "checked against every coil, energised or not.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9aa3b2; font-size:11px;")
        f2.addWidget(btn_all_on, 2, 0)
        f2.addWidget(btn_all_off, 2, 1)
        f2.addWidget(hint, 2, 2, 1, 2)
        lay.addWidget(g2)

        # ---- 3. placement ----
        g3 = QtWidgets.QGroupBox("3. Where the probe is")
        f3 = QtWidgets.QGridLayout(g3)
        self.machine_pose_spins = {}
        fields = (("x_mm", "x", " mm", -20000.0, 20000.0),
                  ("y_mm", "y", " mm", -20000.0, 20000.0),
                  ("z_mm", "z", " mm", -20000.0, 20000.0),
                  ("yaw_deg", "yaw", " °", -360.0, 360.0),
                  ("pitch_deg", "pitch", " °", -360.0, 360.0),
                  ("roll_deg", "roll", " °", -360.0, 360.0))
        for i, (attr, label, suffix, lo, hi) in enumerate(fields):
            row, col = divmod(i, 3)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(1)
            spin.setSuffix(suffix)
            spin.setValue(getattr(self.session.machine.pose, attr))
            spin.valueChanged.connect(self.on_machine_pose_edited)
            f3.addWidget(QtWidgets.QLabel(label), row * 2, col)
            f3.addWidget(spin, row * 2 + 1, col)
            self.machine_pose_spins[attr] = spin
        self.machine_pose_spins["x_mm"].setToolTip(
            "The probe's mounting flange — the end of the tube the boards "
            "count up from — in machine millimetres, with every stage at its "
            "zero. Yaw turns the assembly about the machine's Z, which is the "
            "axis of the torus; it is applied last, so it swings the whole "
            "probe round the machine whatever pitch and roll are set to.")

        self.chk_track_stage = QtWidgets.QCheckBox("follow the stages")
        self.chk_track_stage.setChecked(self.session.machine.track_stage)
        self.chk_track_stage.setToolTip(
            "Add the live stage reading to the pose, so the drawn head moves "
            "as the rig does. Off, the drawing stays where it was put — which "
            "is what you want when planning with the stages disconnected.")
        self.chk_track_stage.toggled.connect(self.on_machine_track)
        btn_zero = QtWidgets.QPushButton("Stage zero is here")
        btn_zero.setToolTip(
            "Take the stages' current reading as the position the pose above "
            "describes. Press it with the rig parked where the flange was "
            "measured, and every later move is drawn relative to that.")
        btn_zero.clicked.connect(self.on_machine_zero_stage)
        f3.addWidget(self.chk_track_stage, 4, 0)
        f3.addWidget(btn_zero, 4, 1, 1, 2)

        self.lbl_machine_stage = QtWidgets.QLabel("stages not connected")
        self.lbl_machine_stage.setWordWrap(True)
        self.lbl_machine_stage.setStyleSheet("color:#9aa3b2; font-size:11px;")
        f3.addWidget(self.lbl_machine_stage, 5, 0, 1, 3)
        lay.addWidget(g3)

        self.lbl_clearance = QtWidgets.QLabel("no coil set loaded")
        self.lbl_clearance.setWordWrap(True)
        self.lbl_clearance.setStyleSheet(
            "font-size:15px; padding:6px; border:1px solid #2a2e3a;")
        lay.addWidget(self.lbl_clearance)

        row = QtWidgets.QHBoxLayout()
        btn_save = QtWidgets.QPushButton("Save placement")
        btn_save.setToolTip(f"Writes all of this to {self.session.args.machine}, so "
                            f"the next session opens where this one left off.")
        btn_save.clicked.connect(self.on_machine_save)
        btn_fit = QtWidgets.QPushButton("Fit machine")
        # Not inlinable: clicked() would pass its `checked` bool into a
        # method that takes no arguments.
        btn_fit.clicked.connect(
            lambda: self.machine_view.reset_camera())   # noqa: PLW0108
        btn_zoom = QtWidgets.QPushButton("Zoom to probe")
        btn_zoom.clicked.connect(
            lambda: self.machine_view.look_at_probe(self.session.machine.pose,
                                                    self.stage_mm()))
        for b in (btn_save, btn_fit, btn_zoom):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        chk_reach = QtWidgets.QCheckBox("stage envelope")
        chk_reach.setChecked(True)
        chk_reach.setToolTip(
            "The box the flange can be driven through, from each axis's "
            "allowed travel — the volume this rig can actually reach without "
            "being unbolted and moved.")
        chk_reach.toggled.connect(
            lambda on: self.machine_view.set_reach_visible(on,
                                                           self.session.machine.pose))
        chk_names = QtWidgets.QCheckBox("coil labels")
        chk_names.setChecked(True)
        chk_names.toggled.connect(self.machine_view.set_labels_visible)
        row2.addWidget(chk_reach)
        row2.addWidget(chk_names)
        row2.addStretch(1)
        lay.addLayout(row2)
        lay.addStretch(1)

        split.addWidget(panel)
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(2, 2, 2, 2)
        legend = QtWidgets.QLabel(
            "amber: current flowing   ·   slate: switched off, still in the "
            "way   ·   green line: closest approach   ·   red: the probe is "
            "inside a winding")
        legend.setWordWrap(True)
        legend.setStyleSheet("color:#9aa3b2; font-size:11px;")
        rl.addWidget(legend)
        rl.addWidget(self.machine_view, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([440, 900])

    def announce(self):
        """Say what was loaded, then draw it.

        Separate from __init__ because the log pane does not exist until the
        whole tab strip has been built, and a configuration warning must
        appear above the coil set it is probably about.
        """
        session = self.session
        if session.coils is not None:
            session.log(f"coil set: {session.coils.note} — "
                        f"{session.machine.coil_file}")
            session.log("machine: "
                        + session.machine.energised_summary(session.coils))
            self.machine_view.set_coils(session.coils,
                                        session.machine.coil_radius_mm,
                                        session.machine.energised)
            self.machine_view.reset_camera()
        self._refresh_machine_controls()
        self.refresh_machine(force=True)

    def on_machine_browse(self):
        start = os.path.dirname(self.ed_coil_file.text()) or os.getcwd()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Coil configuration", start, "simsopt JSON (*.json);;"
                                               "All files (*)")
        if path:
            self.ed_coil_file.setText(path)
            self.on_machine_load()

    def on_machine_load(self, path=None, quiet=False):
        """Read a coil file and hand it to the view.

        A failure is reported and then dropped: the rest of the window works
        perfectly well without a coil set, and the file most likely to fail is
        one that was moved, which is a thing to fix rather than a reason to
        lose the session.
        """
        path = path or self.ed_coil_file.text().strip()
        if not path:
            return
        problems = []
        coils = omach.CoilSet.load_or_none(path, on_error=problems.append)
        for msg in problems:
            self.session.log(f"WARNING: {msg}")
        if coils is None:
            if not quiet:
                QtWidgets.QMessageBox.warning(
                    self, "Coil set", "\n\n".join(problems)
                    or f"{path} could not be read.")
            return
        self.session.coils = coils
        self.session.machine.coil_file = path
        for lost in self.session.machine.adopt(coils):
            self.session.log(f"coil set: {lost}")
        self.session.log(f"coil set: {coils.note} — {path}")
        self._refresh_machine_controls()
        self.machine_view.set_coils(coils, self.session.machine.coil_radius_mm,
                                    self.session.machine.energised)
        self.machine_view.reset_camera()
        self.refresh_machine(force=True)

    def _refresh_machine_controls(self):
        """Push the config into the widgets without echoing back out again."""
        self._machine_quiet = True
        try:
            self.ed_coil_file.setText(self.session.machine.coil_file)
            self.cmb_coil_config.clear()
            if self.session.coils is not None:
                self.cmb_coil_config.addItems(self.session.coils.configurations)
                if self.session.machine.configuration in self.session.coils.configurations:
                    self.cmb_coil_config.setCurrentIndex(
                        self.session.coils.configurations.index(
                            self.session.machine.configuration))
                self.lbl_coil_summary.setText(self.session.coils.note)
            else:
                self.lbl_coil_summary.setText("no coil set loaded")
            self._fill_coil_table()
        finally:
            self._machine_quiet = False

    def _fill_coil_table(self):
        rows = list(self.session.coils) if self.session.coils is not None else []
        self.tbl_coils.setRowCount(len(rows))
        for r, coil in enumerate(rows):
            name = QtWidgets.QTableWidgetItem(coil.label)
            name.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable
                          | QtCore.Qt.ItemFlag.ItemIsEnabled)
            name.setCheckState(
                QtCore.Qt.CheckState.Checked if self.session.machine.is_on(coil.label)
                else QtCore.Qt.CheckState.Unchecked)
            name.setData(QtCore.Qt.ItemDataRole.UserRole, coil.label)
            self.tbl_coils.setItem(r, 0, name)
            amps = self.session.machine.current(coil)
            cells = (coil.where(),
                     f"{amps:+,.0f}" if abs(amps) < 10000 else
                     f"{amps / 1000:+,.2f} k",
                     coil.description)
            for c, text in enumerate(cells, start=1):
                item = QtWidgets.QTableWidgetItem(text)
                if not self.session.machine.is_on(coil.label):
                    item.setForeground(QtGui.QColor("#6a7182"))
                self.tbl_coils.setItem(r, c, item)
        self.tbl_coils.resizeColumnsToContents()

    def on_machine_config(self, _index):
        if self._machine_quiet or self.session.coils is None:
            return
        self.session.machine.configuration = self.cmb_coil_config.currentText()
        self._machine_quiet = True
        try:
            self._fill_coil_table()
        finally:
            self._machine_quiet = False
        self.session.log("coil set: " + self.session.machine.energised_summary(self.session.coils))

    def on_machine_scale(self, value):
        if self._machine_quiet:
            return
        self.session.machine.current_scale = float(value)
        self._machine_quiet = True
        try:
            self._fill_coil_table()
        finally:
            self._machine_quiet = False

    def on_machine_coil_toggled(self, item):
        if self._machine_quiet or item.column() != 0:
            return
        label = item.data(QtCore.Qt.ItemDataRole.UserRole)
        on = item.checkState() == QtCore.Qt.CheckState.Checked
        energised = [c for c in (self.session.machine.energised or []) if c != label]
        if on:
            energised.append(label)
        # Keep the file's own order, so the list in machine.json reads the same
        # way the table does however the boxes were clicked.
        order = self.session.coils.labels if self.session.coils is not None else energised
        self.session.machine.energised = [c for c in order if c in energised]
        self.machine_view.set_energised(self.session.machine.energised)
        self._machine_quiet = True
        try:
            self._fill_coil_table()
        finally:
            self._machine_quiet = False
        self.session.log("coil set: " + self.session.machine.energised_summary(self.session.coils))

    def on_machine_all(self, on):
        if self.session.coils is None:
            return
        self.session.machine.energised = list(self.session.coils.labels) if on else []
        self.machine_view.set_energised(self.session.machine.energised)
        self._machine_quiet = True
        try:
            self._fill_coil_table()
        finally:
            self._machine_quiet = False
        self.session.log("coil set: " + self.session.machine.energised_summary(self.session.coils))

    def on_machine_radius(self, value):
        if self._machine_quiet:
            return
        self.session.machine.coil_radius_mm = float(value)
        self.machine_view.set_radius(value)
        self.refresh_machine(force=True)

    def on_machine_pose_edited(self, _value=None):
        if self._machine_quiet:
            return
        for attr, spin in self.machine_pose_spins.items():
            setattr(self.session.machine.pose, attr, float(spin.value()))
        self.refresh_machine(force=True)

    def on_machine_track(self, on):
        self.session.machine.track_stage = bool(on)
        self.refresh_machine(force=True)

    def on_machine_zero_stage(self):
        stage = self.stage_mm(ignore_tracking=True)
        if not stage:
            QtWidgets.QMessageBox.information(
                self, "Stage zero",
                "The stages are not connected, so there is no reading to "
                "take. Connect them in the Stages tab first.")
            return
        self.session.machine.pose.stage_zero_mm.update(
            {ax: float(v) for ax, v in stage.items()})
        self.session.log("machine: stage zero taken at "
                     + ", ".join(f"{k}={v:.3f} mm"
                                 for k, v in sorted(stage.items())))
        self.refresh_machine(force=True)

    def on_machine_save(self):
        path = self.session.machine.save(self.session.args.machine)
        self.session.log(f"machine: placement written to {path} — "
                     + self.session.machine.pose.describe())
        self.session.log("machine: " + self.session.machine.energised_summary(self.session.coils))

    def stage_mm(self, ignore_tracking=False):
        """The live stage reading, or None if there is nothing to read.

        Position is taken even from an axis whose counter is not trusted: this
        drawing is a picture, not an interlock, and refusing to draw an unhomed
        rig would hide exactly the situation someone is trying to understand.
        The label says so.
        """
        if self.session.stages is None:
            return None
        if not (ignore_tracking or self.session.machine.track_stage):
            return None
        out = {}
        for name in self.session.stages.names:
            st = self.session.stages[name]
            if not st.is_open:
                continue
            try:
                out[name] = float(st.position_mm)
            except ostage.StageError:
                continue
        return out or None

    def _machine_adopt_travel(self):
        """Take the stage envelope from the axes themselves, once they exist."""
        if self.session.stages is None:
            return
        for name in self.session.stages.names:
            if name not in omach.Placement.AXES:
                continue
            try:
                lo, hi = self.session.stages[name].limit_mm
            except (ostage.StageError, TypeError, ValueError):
                continue
            self.session.machine.pose.travel_mm[name] = (float(lo), float(hi))

    def refresh_machine(self, force=False):
        """Redraw the head where it is now, and re-measure the clearance.

        Runs off the stage timer, so it must be cheap when nothing has moved:
        the pose and the stage reading are hashed and the whole thing skipped
        when they are unchanged, which is most ticks.
        """
        if getattr(self, "machine_view", None) is None:
            return
        stage = self.stage_mm()
        pose = self.session.machine.pose
        key = (pose.x_mm, pose.y_mm, pose.z_mm, pose.yaw_deg, pose.pitch_deg,
               pose.roll_deg, self.session.machine.coil_radius_mm,
               tuple(sorted(pose.stage_zero_mm.items())),
               None if stage is None else tuple(sorted(
                   (k, round(v, 3)) for k, v in stage.items())),
               id(self.session.coils))
        if key == self._machine_key and not force:
            return
        self._machine_key = key

        if self.session.stages is not None and not self._machine_travel_taken:
            self._machine_adopt_travel()
            self._machine_travel_taken = True

        with self.session.prof.time("machine view"):
            self.machine_view.set_pose(pose, stage)
            if self.session.probe_cloud is None:
                self.session.probe_cloud = omach.probe_cloud(self.session.geom)
            if self.session.coils is not None and len(self.session.coils):
                gap = omach.clearance(pose.to_machine(self.session.probe_cloud, stage),
                                      self.session.coils, self.session.machine.coil_radius_mm)
                self.machine_view.set_clearance(gap)
                self.lbl_clearance.setText(gap.text())
                self.lbl_clearance.setStyleSheet(
                    "font-size:15px; padding:6px; border:1px solid "
                    + ("#8a2020; background:#2a1416; color:#ff8a8a;"
                       if gap.collides else "#2a2e3a;"))
            else:
                self.machine_view.set_clearance(None)

        if stage is None:
            self.lbl_machine_stage.setText(
                "not following the stages — the head is drawn at the pose "
                "above" if self.session.stages is not None
                else "stages not connected — the head is drawn at the pose "
                     "above")
        else:
            where = ", ".join(f"{k}={stage[k]:.3f}" for k in sorted(stage))
            origin = pose.origin_mm(stage)
            untrusted = [n for n in sorted(stage)
                         if not self.session.stages[n].position_trusted]
            note = ("  (position not trusted on "
                    + ", ".join(untrusted) + " — home first)") if untrusted else ""
            self.lbl_machine_stage.setText(
                f"stage {where} mm  →  flange at ({origin[0]:+.0f}, "
                f"{origin[1]:+.0f}, {origin[2]:+.0f}) mm{note}")

    def set_geometry(self, geom):
        """Adopt a reloaded probe geometry and redraw."""
        self.machine_view.set_geometry(geom)
        self.refresh_machine(force=True)
