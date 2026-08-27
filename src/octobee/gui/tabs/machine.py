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

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from octobee import machine as omach
from octobee import record as orec
from octobee.gui.widgets.machine3d import MachineView3D
from octobee.gui.workers import (
    EncoderCalibrationWorker,
    PlanWorker,
    SweepWorker,
)
from octobee.motion import encoder as oenc
from octobee.motion import stage as ostage
from octobee.motion import sweep as osweep


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
        self._reachable = None          # the last computed reachable grid
        self._plan = None               # the last computed SweepPlan
        self._plan_worker = None
        self._sweep_worker = None
        self._preview = None            # [points, index] while animating
        self._preview_at = None          # the made-up stage reading it implies
        self._preview_timer = None
        self._run_hz = 0.0              # the rate a sweep in flight is timed at
        self._enc_worker = None
        self.machine_view = MachineView3D(session.geom,
                                          profiler=session.prof)
        self.machine_view.pose_dragged.connect(self.on_machine_pose_dragged)
        self.machine_view.pose_turned.connect(self.on_machine_pose_turned)

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
                  ("rot_z_deg", "about Z", " °", -360.0, 360.0))
        for i, (attr, label, suffix, lo, hi) in enumerate(fields):
            row, col = divmod(i, 3)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(1)
            spin.setSuffix(suffix)
            spin.setWrapping(attr == "rot_z_deg")
            spin.setValue(getattr(self.session.machine.pose, attr))
            spin.valueChanged.connect(self.on_machine_pose_edited)
            f3.addWidget(QtWidgets.QLabel(label), row * 2, col)
            f3.addWidget(spin, row * 2 + 1, col)
            self.machine_pose_spins[attr] = spin
        self.machine_pose_spins["x_mm"].setToolTip(
            "The probe's mounting flange — the end of the tube the boards "
            "count up from — in machine millimetres, with every stage at its "
            "zero.\n\n"
            "x, y and z can also be dragged: the ball in the drawing is the "
            "probe's zero point, and pulling one of its arrows slides the "
            "probe along that machine axis and writes the number here.")
        self.machine_pose_spins["rot_z_deg"].setToolTip(
            "Which way round the machine the whole assembly is turned, about "
            "the machine's Z — the axis of the torus. It is the only "
            "rotation this rig has: the probe is bolted to a cartesian gantry "
            "that cannot tilt it, so pitch and roll are gone rather than "
            "sitting there able to describe a pose the head cannot reach.\n\n"
            "Draggable too: the ring round the zero point in the drawing.")

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

        # ---- 4. the volume to map ----
        g4 = QtWidgets.QGroupBox("4. Volume to map")
        f4 = QtWidgets.QGridLayout(g4)
        blurb = QtWidgets.QLabel(
            "Swept, not stepped: one axis runs at constant velocity while the "
            "stream is logged, the other two step between lines. Every sample "
            "carries the rig xyz, the same point in the coil file's frame, and "
            "all sixteen sensors' Bx, By and Bz. The whole envelope settled "
            "point by point would be sixty-two hours; swept it is an evening.")
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color:#9aa3b2; font-size:11px;")
        f4.addWidget(blurb, 0, 0, 1, 4)

        self.chk_vol_all = QtWidgets.QCheckBox("the whole travel")
        self.chk_vol_all.setChecked(True)
        self.chk_vol_all.setToolTip(
            "Everything the stages can reach, taken from each axis's own "
            "travel — 300 mm cubed on this rig. Untick to map a smaller box "
            "and type where it starts and how big it is.")
        self.chk_vol_all.toggled.connect(self.on_volume_whole)
        f4.addWidget(self.chk_vol_all, 1, 0, 1, 4)

        self.vol_spins = {}
        for r, (key, label, default) in enumerate(
                (("from", "from", (0.0, 0.0, 0.0)),
                 ("size", "size", (300.0, 300.0, 300.0))), start=2):
            f4.addWidget(QtWidgets.QLabel(label), r, 0)
            row = []
            for c, (ax, val) in enumerate(zip("xyz", default), start=1):
                sp = QtWidgets.QDoubleSpinBox()
                sp.setRange(0.0 if key == "size" else -1000.0, 1000.0)
                sp.setDecimals(1)
                sp.setValue(val)
                sp.setPrefix(f"{ax} ")
                sp.setSuffix(" mm")
                sp.valueChanged.connect(self.on_volume_edited)
                f4.addWidget(sp, r, c)
                row.append(sp)
            self.vol_spins[key] = row

        f4.addWidget(QtWidgets.QLabel("grid step"), 4, 0)
        self.spin_vol_step = QtWidgets.QDoubleSpinBox()
        self.spin_vol_step.setRange(1.0, 100.0)
        self.spin_vol_step.setDecimals(1)
        self.spin_vol_step.setValue(10.0)
        self.spin_vol_step.setSuffix(" mm")
        self.spin_vol_step.setToolTip(
            "How far apart the swept lines are, and the resolution the "
            "reachable region is worked out at. It does NOT set how finely the "
            "swept axis is sampled — that is the output rate and the sweep "
            "speed, and at 500 Hz and 10 mm/s it is a sample every 20 µm.")
        self.spin_vol_step.valueChanged.connect(self.on_volume_edited)
        f4.addWidget(self.spin_vol_step, 4, 1)
        self.cmb_vol_axis = QtWidgets.QComboBox()
        self.cmb_vol_axis.addItems(["x", "y", "z"])
        self.cmb_vol_axis.setToolTip(
            "Which axis runs continuously. The other two step. Sweeping the "
            "axis with the longest run through the volume is what makes this "
            "quick, because every line costs a return move as well.")
        self.cmb_vol_axis.currentIndexChanged.connect(self.on_volume_edited)
        f4.addWidget(QtWidgets.QLabel("sweep"), 5, 2)
        f4.addWidget(self.cmb_vol_axis, 5, 3)

        f4.addWidget(QtWidgets.QLabel("sweep speed"), 5, 0)
        self.spin_vol_speed = QtWidgets.QDoubleSpinBox()
        self.spin_vol_speed.setRange(0.1, osweep.MAX_SPEED_MM_S)
        self.spin_vol_speed.setDecimals(1)
        self.spin_vol_speed.setValue(osweep.DEFAULT_SPEED_MM_S)
        self.spin_vol_speed.setSuffix(" mm/s")
        self.spin_vol_speed.setToolTip(
            f"How fast the swept axis runs. Capped at "
            f"{osweep.MAX_SPEED_MM_S:g} mm/s, which is a measurement limit "
            f"rather than a hardware one — the axes will do 10, but the head "
            f"is on the end of a cantilever and this is the one measurement "
            f"taken while it is moving.")
        self.spin_vol_speed.valueChanged.connect(self.on_volume_edited)
        f4.addWidget(self.spin_vol_speed, 5, 1)

        self.chk_vol_shell = QtWidgets.QCheckBox("outer surface only")
        self.chk_vol_shell.setChecked(True)
        self.chk_vol_shell.setToolTip(
            "Map the six faces of the box rather than filling it — each swept "
            "in its own plane, so the two faces across the sweep axis are run "
            "along a different one. For a field in a current-free region the "
            "boundary determines the interior, so this is not a poor relation "
            "of the filled version; it is five or six times fewer lines, "
            "which is the difference between an evening and a week.")
        self.chk_vol_shell.toggled.connect(self.on_volume_edited)
        f4.addWidget(self.chk_vol_shell, 4, 2, 1, 2)

        self.chk_vol_avoid = QtWidgets.QCheckBox("keep clear of the coils")
        self.chk_vol_avoid.setChecked(True)
        self.chk_vol_avoid.setToolTip(
            "Cut every line back to the part the probe body actually fits in, "
            "so a line that would pass through a winding becomes two shorter "
            "lines with the coil between them. The whole probe is tested, not "
            "the head: the square tube runs the length of the probe back to "
            "its zero point, and that is what hits things first.")
        self.chk_vol_avoid.toggled.connect(self.on_volume_edited)
        f4.addWidget(self.chk_vol_avoid, 6, 0, 1, 2)
        self.spin_vol_margin = QtWidgets.QDoubleSpinBox()
        self.spin_vol_margin.setRange(0.0, 500.0)
        self.spin_vol_margin.setDecimals(1)
        self.spin_vol_margin.setValue(10.0)
        self.spin_vol_margin.setPrefix("margin ")
        self.spin_vol_margin.setSuffix(" mm")
        self.spin_vol_margin.setToolTip(
            "How much daylight to leave between the probe and every winding, "
            "energised or not. It is not a safety factor on top of a computed "
            "number — it IS the number, and it wants to be at least as big as "
            "the pose is uncertain by, because a clearance computed from a "
            "pose measured with a tape is only as good as the tape was.")
        self.spin_vol_margin.valueChanged.connect(self.on_volume_edited)
        f4.addWidget(self.spin_vol_margin, 6, 2, 1, 2)

        self.lbl_encoders = QtWidgets.QLabel("")
        self.lbl_encoders.setWordWrap(True)
        self.lbl_encoders.setStyleSheet("color:#9aa3b2; font-size:11px;")
        self.btn_vol_encoders = QtWidgets.QPushButton("Calibrate encoders")
        self.btn_vol_encoders.setToolTip(
            "Drive each axis a known distance and watch the quadrature counts "
            "that arrive with the field samples, to find which stream column "
            "belongs to which axis and how many counts make a millimetre.\n\n"
            "Worth doing because those counts are latched by the ADC clock: a "
            "position taken from them belongs to the sample it is written "
            "against, where one polled over USB is a few milliseconds stale "
            "and nothing can say how many. It does not make the position more "
            "accurate — the encoders are on the motors, so they count the same "
            "leadscrew — it makes it belong to the right instant.")
        self.btn_vol_encoders.clicked.connect(self.on_volume_calibrate)
        self.spin_enc_span = QtWidgets.QDoubleSpinBox()
        self.spin_enc_span.setRange(oenc.MIN_CALIBRATION_SPAN_MM, 290.0)
        self.spin_enc_span.setValue(oenc.CALIBRATION_SPAN_MM)
        self.spin_enc_span.setDecimals(0)
        self.spin_enc_span.setSuffix(" mm span")
        self.spin_enc_span.setToolTip(
            "How far each axis is stepped, out and back.\n\n"
            "Longer is better and costs only time: every fixed error in the "
            "run — a stage that stops a little short of each target, a "
            "reading taken a moment early — is divided by this number when it "
            "reaches the scale. The first run on this rig used 20 mm and put "
            "0.41% into x that way.\n\n"
            "Clamped to whatever room the axis actually has, from where it is "
            "standing. The run says so if it had to shorten it.")
        f4.addWidget(self.btn_vol_encoders, 7, 0)
        f4.addWidget(self.spin_enc_span, 7, 1)
        f4.addWidget(self.lbl_encoders, 7, 2, 1, 2)

        self.lbl_volume = QtWidgets.QLabel("")
        self.lbl_volume.setWordWrap(True)
        self.lbl_volume.setStyleSheet("color:#9aa3b2; font-size:11px;")
        f4.addWidget(self.lbl_volume, 8, 0, 1, 4)

        self.btn_vol_plan = QtWidgets.QPushButton("Work out the path")
        self.btn_vol_plan.clicked.connect(self.on_volume_plan)
        self.btn_vol_preview = QtWidgets.QPushButton("Fly it")
        self.btn_vol_preview.setEnabled(False)
        self.btn_vol_preview.setToolTip(
            "Run the probe along the planned path in the drawing, without "
            "moving anything. The clearance readout follows it, so a plan that "
            "grazes a winding says so before the rig does.")
        self.btn_vol_preview.clicked.connect(self.on_volume_preview)
        self.btn_vol_start = QtWidgets.QPushButton("Start volume map")
        self.btn_vol_start.setEnabled(False)
        self.btn_vol_start.clicked.connect(self.on_volume_start)
        self.btn_vol_abort = QtWidgets.QPushButton("Abort")
        self.btn_vol_abort.setEnabled(False)
        self.btn_vol_abort.clicked.connect(self.on_volume_abort)
        f4.addWidget(self.btn_vol_plan, 9, 0)
        f4.addWidget(self.btn_vol_preview, 9, 1)
        f4.addWidget(self.btn_vol_start, 9, 2)
        f4.addWidget(self.btn_vol_abort, 9, 3)

        self.bar_volume = QtWidgets.QProgressBar()
        self.bar_volume.setTextVisible(True)
        f4.addWidget(self.bar_volume, 10, 0, 1, 4)
        lay.addWidget(g4)

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
        chk_gizmo = QtWidgets.QCheckBox("drag handle")
        chk_gizmo.setChecked(True)
        chk_gizmo.setToolTip(
            "The probe's zero point, with an arrow along each machine axis. "
            "Drag an arrow to slide the probe along it; the x, y and z boxes "
            "above follow. Turn it off to orbit the view without catching it.")
        chk_gizmo.toggled.connect(self.machine_view.set_gizmo_visible)
        row2.addWidget(chk_reach)
        row2.addWidget(chk_names)
        row2.addWidget(chk_gizmo)
        row2.addStretch(1)
        lay.addLayout(row2)
        lay.addStretch(1)

        # The panel is four groups tall now and does not fit a laptop screen.
        # Scrolled rather than squeezed: the alternative is spin boxes that
        # shrink until the units no longer fit in them, on the tab whose whole
        # job is getting numbers typed in correctly.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        # Wide enough for the panel's own idea of itself plus the scrollbar,
        # so the vertical scroll never costs a horizontal one.
        scroll.setMinimumWidth(panel.sizeHint().width() + 28)
        split.addWidget(scroll)
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(2, 2, 2, 2)
        legend = QtWidgets.QLabel(
            "amber: current flowing   ·   slate: switched off, still in the "
            "way   ·   green line: closest approach   ·   red: the probe is "
            "inside a winding   ·   the ball is the probe's zero point — "
            "drag an arrow to move it")
        legend.setWordWrap(True)
        legend.setStyleSheet("color:#9aa3b2; font-size:11px;")
        rl.addWidget(legend)
        rl.addWidget(self.machine_view, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([panel.sizeHint().width() + 28, 900])

        self.on_volume_whole(self.chk_vol_all.isChecked())
        self.refresh_encoders()

    def announce(self):
        """Say what was loaded, then draw it.

        Separate from __init__ because the log pane does not exist until the
        whole tab strip has been built, and a configuration warning must
        appear above the coil set it is probably about.
        """
        session = self.session
        for note in session.machine.pose_notes:
            session.log(f"WARNING: machine: {note}")
        session.machine.pose_notes.clear()
        if session.encoders:
            session.log(f"encoders: {session.encoders.describe()} "
                        f"— from {session.stages_path}")
            self._warn_if_axes_disagree(session.encoders.to_dict(),
                                        "loaded from file")
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

    def on_machine_pose_dragged(self, x_mm, y_mm, z_mm):
        """An arrow on the zero point was dragged: take the new position."""
        self._write_pose(x_mm=x_mm, y_mm=y_mm, z_mm=z_mm)

    def on_machine_pose_turned(self, deg):
        """The ring round the zero point was dragged: take the new angle."""
        self._write_pose(rot_z_deg=(float(deg) + 180.0) % 360.0 - 180.0)

    def _write_pose(self, **values):
        """Put dragged numbers into the spin boxes, which own the pose.

        The boxes are the one place the pose is written, so a drag goes through
        them and everything that already watches them -- the redraw, the
        clearance, the volume, what Save writes -- follows without knowing a
        mouse was involved.
        """
        for attr, value in values.items():
            spin = self.machine_pose_spins[attr]
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.on_machine_pose_edited()

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

    # ==================================================================
    # the volume to map
    # ==================================================================
    def volume(self):
        """The box currently described by the controls, in rig millimetres."""
        step = float(self.spin_vol_step.value())
        axis = self.cmb_vol_axis.currentText()
        if self.chk_vol_all.isChecked():
            return osweep.Volume.whole_travel(
                self.session.machine.pose.travel_mm, step_mm=step, sweep=axis)
        lo = [sp.value() for sp in self.vol_spins["from"]]
        size = [sp.value() for sp in self.vol_spins["size"]]
        return osweep.Volume(lo, [a + b for a, b in zip(lo, size)],
                             step_mm=step, sweep=axis)

    def on_volume_whole(self, on):
        """Whole travel or a typed box: show the numbers either way.

        The boxes are filled in with the travel rather than disabled and left
        stale, so unticking gives the whole envelope to trim rather than
        whatever was last typed into a box nobody was looking at.
        """
        travel = self.session.machine.pose.travel_mm
        if on:
            for i, ax in enumerate(omach.Placement.AXES):
                lo, hi = travel[ax]
                self.vol_spins["from"][i].blockSignals(True)
                self.vol_spins["size"][i].blockSignals(True)
                self.vol_spins["from"][i].setValue(float(lo))
                self.vol_spins["size"][i].setValue(float(hi - lo))
                self.vol_spins["from"][i].blockSignals(False)
                self.vol_spins["size"][i].blockSignals(False)
        for row in self.vol_spins.values():
            for sp in row:
                sp.setEnabled(not on)
        self.on_volume_edited()

    def on_volume_edited(self, _value=None):
        """Anything that changes the volume invalidates the plan made for it.

        Silently keeping an old path and starting a rig on it is the failure
        this guards: the drawing would show the lines that were planned and the
        stages would run the ones that were, and only the map would say so.
        """
        self._reachable = None
        self._plan = None
        self.btn_vol_preview.setEnabled(False)
        self.btn_vol_start.setEnabled(False)
        self._stop_preview()
        self.machine_view.set_path(None, None)
        try:
            vol = self.volume()
        except ValueError as exc:
            self.machine_view.set_volume(None)
            self.lbl_volume.setText(str(exc))
            return
        self.machine_view.set_volume(
            self.session.machine.pose.flange_path_mm(vol.corners_mm()))
        rough = osweep.plan(vol, speed_mm_s=self.spin_vol_speed.value(),
                            shell=self.chk_vol_shell.isChecked())
        self.lbl_volume.setText(
            f"{vol.describe()} — not planned yet. Ignoring the coils it "
            f"would be {rough.describe()}.")

    def on_volume_plan(self):
        """Work out what can actually be swept, then draw it."""
        if self._plan_worker is not None and self._plan_worker.isRunning():
            return
        try:
            vol = self.volume()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Volume", str(exc))
            return
        if not self.chk_vol_avoid.isChecked() or self.session.coils is None:
            self._finish_plan(vol, None)
            return
        if self.session.probe_cloud is None:
            self.session.probe_cloud = omach.probe_cloud(self.session.geom)
        self.lbl_volume.setText(f"{vol.describe()} — working out where the "
                                f"probe fits…")
        self.btn_vol_plan.setEnabled(False)
        self.bar_volume.setRange(0, 100)
        self.bar_volume.setValue(0)
        self._plan_worker = PlanWorker(
            vol, self.session.machine.pose, self.session.probe_cloud,
            self.session.coils, self.session.machine.coil_radius_mm,
            self.spin_vol_margin.value())
        self._plan_worker.progress.connect(
            lambda i, n: self.bar_volume.setValue(int(100 * i / max(n, 1))))
        self._plan_worker.done.connect(
            lambda grid, err: self._on_plan_done(vol, grid, err))
        self._plan_worker.start()

    def _on_plan_done(self, vol, grid, err):
        self.btn_vol_plan.setEnabled(True)
        self.bar_volume.setValue(0)
        if err:
            self.session.log(f"WARNING: volume: {err}")
            QtWidgets.QMessageBox.warning(self, "Volume", err)
            self.lbl_volume.setText(f"{vol.describe()} — {err}")
            return
        self._finish_plan(vol, grid)

    def _finish_plan(self, vol, reachable):
        """Turn a reachable grid into lines, draw them, and say what it costs."""
        self._reachable = reachable
        self._plan = osweep.plan(
            vol, reachable=reachable,
            speed_mm_s=self.spin_vol_speed.value(),
            log_hz=self.session.out_rate,
            travel_mm=self.session.machine.pose.travel_mm,
            shell=self.chk_vol_shell.isChecked())
        pose = self.session.machine.pose
        dropped = self._dropped_points(vol)
        self.machine_view.set_volume(pose.flange_path_mm(vol.corners_mm()))
        self.machine_view.set_path(
            pose.flange_path_mm(self._path_points()),
            None if dropped is None else pose.flange_path_mm(dropped))
        extra = omach.grid_standoff_mm(vol.step_mm)
        note = self._plan.describe()
        if reachable is not None:
            note += (f". Clearance held is {self.spin_vol_margin.value():g} mm "
                     f"plus up to {extra:.1f} mm from working it out on a "
                     f"{vol.step_mm:g} mm grid — the grid can refuse a "
                     f"position that would in fact have been fine, never the "
                     f"other way round.")
        self.lbl_volume.setText(note)
        self.session.log(f"volume: {vol.describe()} — {self._plan.describe()}")
        self.btn_vol_preview.setEnabled(bool(self._plan.lines))
        self.btn_vol_start.setEnabled(
            bool(self._plan.lines) and self.session.stages is not None)

    def _path_points(self):
        """The planned lines as pairs of end points, rig mm."""
        out = []
        for line in self._plan.lines:
            for where in (line.start_mm, line.stop_mm):
                out.append([line.fixed.get(a, 0.0) if a != line.sweep else where
                            for a in omach.Placement.AXES])
        return np.array(out).reshape(-1, 3) if out else np.zeros((0, 3))

    def _dropped_points(self, vol):
        """What a plan ignoring the coils would have swept, as a faint ghost.

        Drawn so the carved shape reads as a shape: the lines that survive mean
        much more next to the ones that did not than they do alone.
        """
        if self._reachable is None:
            return None
        whole = osweep.plan(vol, speed_mm_s=self.spin_vol_speed.value(),
                            shell=self.chk_vol_shell.isChecked())
        kept = {(line.sweep, tuple(sorted(line.fixed.items())),
                 round(line.start_mm, 6), round(line.stop_mm, 6))
                for line in self._plan.lines}
        out = []
        for line in whole.lines:
            key = (line.sweep, tuple(sorted(line.fixed.items())),
                   round(line.start_mm, 6), round(line.stop_mm, 6))
            if key in kept:
                continue
            for where in (line.start_mm, line.stop_mm):
                out.append([line.fixed.get(a, 0.0) if a != line.sweep else where
                            for a in omach.Placement.AXES])
        return np.array(out).reshape(-1, 3) if out else None

    # ---- the encoders ----------------------------------------------------
    def refresh_encoders(self):
        """Say what the position column of the next map will be made of.

        Called whenever either input changes -- the source, or the stages --
        because both are None while this tab is being built and neither is
        what it will be by the time anyone presses the button.
        """
        running = (self._enc_worker is not None
                   and self._enc_worker.isRunning())
        enc = self.session.encoders
        cols = int(getattr(self.session.source, "enc_columns", 0) or 0)
        if enc:
            missing = [a for a in oenc.AXES if a not in enc]
            text = "encoders: " + enc.describe()
            text += (f" — {', '.join(missing)} still from the controller"
                     if missing else " — all three on the sample clock")
        elif cols:
            host = getattr(self.session.source, "enc_host", None) or "a carrier"
            sites = getattr(self.session.source, "enc_sites", ())
            where = f" (sites {', '.join(str(v) for v in sites)})" if sites else ""
            text = (f"{cols} encoder column(s) arriving from {host}{where}, "
                    f"none calibrated — positions come from the controllers "
                    f"over USB until they are.")
        else:
            text = ("no counting encoders in this stream — a carrier has to "
                    "aggregate quadrature sites AND have phaseA_en set on "
                    "them. Positions come from the controllers, read over USB.")
        # Not while a calibration is in flight: the label is that run's
        # progress, and a stage connecting or a source arriving mid-run must
        # not overwrite it with a description of what is being measured.
        if not running:
            self.lbl_encoders.setText(text)
        self.btn_vol_encoders.setEnabled(
            self.session.stages is not None and not running)

    def _counts_now(self):
        """The most recent encoder counts, or None. Read from other threads.

        Out of the session rather than a copy kept here. The acquisition tick
        publishes one row per tick and the live recorder anchors to that same
        row; a second copy on this tab was one more thing that could be a tick
        behind the first, for no gain.
        """
        return self.session.enc_counts_now

    def note_encoder_counts(self, counts):
        """Called from the acquisition tick with this tick's last count row.

        Only to hand the sweep in flight something fresh to anchor a line to.
        The tick has already put the same row on the session.
        """
        if counts is None or not len(counts):
            return
        runner = self.session.volume_runner
        if runner is not None:
            runner.counts_now = self.session.enc_counts_now

    def on_volume_calibrate(self):
        if self._enc_worker is not None and self._enc_worker.isRunning():
            return
        if self.session.stages is None:
            QtWidgets.QMessageBox.warning(
                self, "Encoders",
                "The stages are not connected. Connect them in the Stages tab.")
            return
        if not int(getattr(self.session.source, "enc_columns", 0) or 0):
            QtWidgets.QMessageBox.warning(
                self, "Encoders",
                "No encoder counts are arriving with the field.\n\n"
                "Only acq1001_695 aggregates the quadrature sites, and only "
                "while the stream is running — press Connect first. If it is "
                "connected and this still says so, check that its rc.user "
                "still has sites 2, 5 and 6 in the aggregator set.")
            return
        span = float(self.spin_enc_span.value())
        points = oenc.CALIBRATION_POINTS
        reply = QtWidgets.QMessageBox.question(
            self, "Encoders",
            f"Each axis will be stepped along {span:g} mm and back, one at a "
            f"time, stopping at {points} places each way, to see which stream "
            f"column follows it and by how many counts per millimetre.\n\n"
            f"The span is centred on where the axis is standing and clamped "
            f"to the room it has. Every axis is returned to where it was "
            f"found.\n\nMake sure the head is clear over that range.\n\nGo?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel)
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.btn_vol_encoders.setEnabled(False)
        self.lbl_encoders.setText("calibrating…")
        self._enc_worker = EncoderCalibrationWorker(
            self.session.stages, self._counts_now, span_mm=span,
            points=points)
        self._enc_worker.message.connect(self.session.log)
        self._enc_worker.progress.connect(self._on_enc_progress)
        self._enc_worker.done.connect(self.on_volume_calibrated)
        self._enc_worker.start()

    def _on_enc_progress(self, done, total):
        """Standstills taken so far. A long run needs to look like it is one."""
        if done < total:
            self.lbl_encoders.setText(
                f"calibrating… {done}/{total} standstills")

    @staticmethod
    def _encoder_note(found):
        """One line for stages.json recording how good the fit was.

        The scale alone cannot be told apart from a wrong scale later, and
        this file is the only thing that outlives the session it was measured
        in. So the residual goes in beside it.
        """
        parts = []
        for axis, spec in sorted(found.items()):
            bits = [f"{spec['counts_per_mm']:+,.1f} c/mm"]
            if spec.get("span_mm"):
                bits.append(f"{spec['span_mm']:g} mm x {spec.get('points')} pts")
            if spec.get("residual_um") is not None:
                bits.append(f"residual {spec['residual_um']:.1f} um")
            if spec.get("direction_spread_ppm") is not None:
                bits.append(f"out/back {spec['direction_spread_ppm']:.0f} ppm")
            if spec.get("direction_offset_um") is not None:
                bits.append(
                    f"out/back agree to {spec['direction_offset_um']:.1f} um")
            parts.append(f"{axis}: " + ", ".join(bits))
        return ("multi-point fit against the controller's measured position; "
                + "; ".join(parts))

    def on_volume_calibrated(self, found, err):
        """Adopt whatever was measured, and say plainly what was not."""
        if found:
            merged = dict(self.session.encoders.to_dict())
            merged.update(found)
            self.session.encoders = oenc.EncoderMap(merged)
            path = self.session.encoders.save(
                self.session.stages_path, note=self._encoder_note(found))
            self.session.log(f"encoders: {self.session.encoders.describe()} "
                             f"-> written to {path}")
            self._warn_if_axes_disagree(found, "this run")
        if err:
            self.session.log(f"WARNING: encoders: {err}")
        self.refresh_encoders()
        if err:
            QtWidgets.QMessageBox.warning(
                self, "Encoders",
                "Not every axis could be calibrated:\n\n  "
                + "\n  ".join(err.split("; "))
                + "\n\nWhatever was measured has been kept; the rest fall "
                  "back to the controller's own position.")

    def _warn_if_axes_disagree(self, found, source="this run"):
        """Say so when one axis's scale is the odd one out.

        A warning rather than a refusal: two axes of genuinely different pitch
        would trip this legitimately, and the operator knows what is bolted to
        the rig where the software does not. See encoder.odd_axis_out for why
        the axes are each other's only check here.

        Run on the way IN as well as on the way out of a calibration. A scale
        read back from stages.json is used exactly as hard as one just
        measured, and it outlives the session that measured it -- so a bad one
        would otherwise be caught once, on the day, and never again.
        """
        scales = {a: s.get("counts_per_mm") for a, s in found.items()}
        odd = oenc.odd_axis_out(scales)
        if odd is None:
            if len(scales) >= 3:
                self.session.log(
                    f"encoders ({source}): all {len(scales)} axes agree on "
                    f"their counts per millimetre to within 0.1% -- which is "
                    f"what identical stages should do")
            return
        axis, frac = odd
        spec = found[axis]
        detail = ""
        if spec.get("residual_um") is not None:
            detail = f" Its own fit residual was {spec['residual_um']:.1f} um."
        # In millimetres as well as per cent, because a percentage of a
        # gearing ratio is not a quantity anyone can picture and the length of
        # the axis is what it actually costs.
        span = max(hi - lo for lo, hi
                   in self.session.machine.pose.travel_mm.values())
        self.session.log(
            f"WARNING: encoders ({source}): {axis} is {frac * 100:+.2f}% from "
            f"the other axes ({abs(spec['counts_per_mm']):,.1f} counts/mm), "
            f"which is {abs(frac) * span:.1f} mm across a {span:g} mm axis. "
            f"Identical stages share a gearing ratio, so this is more likely a "
            f"bad run on {axis} than a real difference.{detail} Re-run it, "
            f"with a longer span if there is room.")

    # ---- flying the path -------------------------------------------------
    def on_volume_preview(self):
        """Walk the probe along the planned path in the drawing.

        Nothing moves. It is the same set_pose the stage timer uses, fed a made
        up stage reading, so what is drawn is exactly what would be drawn if
        the rig really were there -- clearance readout included, which is the
        point: a plan is worth flying precisely because the number that says it
        is safe is computed the same way either way.
        """
        if self._preview is not None:
            self._stop_preview()
            return
        if self._plan is None or not self._plan.lines:
            return
        points = self._preview_points()
        if not len(points):
            return
        self._preview = [points, 0]
        self.btn_vol_preview.setText("Stop")
        if self._preview_timer is None:
            self._preview_timer = QtCore.QTimer(self)
            self._preview_timer.timeout.connect(self._preview_step)
        self._preview_timer.start(40)

    def _preview_points(self):
        """The whole path as rig positions, at the grid step, in run order."""
        step = max(self._plan.volume.step_mm, 1.0)
        out = []
        for line in self._plan.lines:
            n = max(2, int(round(line.span_mm / step)) + 1)
            for t in np.linspace(line.start_mm, line.stop_mm, n):
                out.append([line.fixed.get(a, 0.0) if a != line.sweep else t
                            for a in omach.Placement.AXES])
        return np.array(out).reshape(-1, 3) if out else np.zeros((0, 3))

    def _preview_step(self):
        if self._preview is None:
            return
        points, i = self._preview
        if i >= len(points):
            self._stop_preview()
            return
        # Straight into refresh_machine's own path, so the clearance and the
        # drawing come out of the same code they always do.
        self._preview_at = {a: float(points[i][k]) for k, a
                            in enumerate(omach.Placement.AXES)}
        self._preview[1] = i + max(1, len(points) // 400)
        self.bar_volume.setRange(0, len(points))
        self.bar_volume.setValue(min(i, len(points)))
        self.refresh_machine(force=True)

    def _stop_preview(self):
        if self._preview_timer is not None:
            self._preview_timer.stop()
        self._preview = None
        self._preview_at = None
        self.btn_vol_preview.setText("Fly it")
        self.bar_volume.setRange(0, 100)
        self.bar_volume.setValue(0)
        self.refresh_machine(force=True)

    # ---- running it ------------------------------------------------------
    def on_volume_start(self):
        if self._sweep_worker is not None and self._sweep_worker.isRunning():
            return
        if self._plan is None or not self._plan.lines:
            return
        if self.session.stages is None:
            QtWidgets.QMessageBox.warning(
                self, "Volume map",
                "The stages are not connected. Connect them in the Stages tab.")
            return
        if self.session.source is None:
            QtWidgets.QMessageBox.warning(
                self, "Volume map",
                "There is no live stream to log. A swept map reads the stream "
                "that is already running rather than taking the carriers over, "
                "so it needs one -- press Connect first.")
            return
        self._stop_preview()
        hours = self._plan.duration_s() / 3600.0
        reply = QtWidgets.QMessageBox.question(
            self, "Volume map",
            f"{self._plan.describe()}.\n\n"
            f"Roughly {hours:.1f} hours, logging at "
            f"{self.session.out_rate:g} Hz.\n\n"
            # Said before the run rather than discovered in the sidecar
            # after it. Which axes are on the sample clock is the
            # difference between a position column good to a micrometre
            # and one good to a millimetre, and it is a five-minute fix
            # beforehand.
            f"{self._position_plan()}\n\n"
            f"The stages will run unattended over the whole path. The "
            f"clearance was computed from the pose above -- if that pose is "
            f"not measured, nothing here protects the head.\n\nStart?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel)
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        meta = {"machine": self.session.machine.to_scan_meta(
                    self.session.coils, self.stage_mm(ignore_tracking=True)),
                "plan": self._plan.to_dict(),
                "geometry": self.session.geom.to_dict(),
                "calibration_id": orec.calibration_id(self.session.cal)}
        log = osweep.SweepLog(self.session.geom, meta, self.session.encoders)
        runner = osweep.SweepRunner(self.session.stages, self._plan, log,
                                    log_fn=self.session.log)
        self.session.volume_log = log
        self.session.volume_runner = runner
        # The rate the rows are actually written at. Captured rather than read
        # back at the end, because the output-rate box is still live during a
        # sweep and a map assembled at a rate the rows were not written at
        # would have every position wrong by a growing amount.
        self._run_hz = float(self.session.out_rate)
        self.bar_volume.setRange(0, len(self._plan.lines))
        self.bar_volume.setValue(0)
        self.btn_vol_start.setEnabled(False)
        self.btn_vol_plan.setEnabled(False)
        self.btn_vol_abort.setEnabled(True)
        self._sweep_worker = SweepWorker(runner)
        self._sweep_worker.message.connect(self.session.log)
        self._sweep_worker.progress.connect(self.on_volume_progress)
        self._sweep_worker.done.connect(self.on_volume_done)
        self._sweep_worker.start()

    def on_volume_abort(self):
        if self._sweep_worker is not None:
            self._sweep_worker.abort()
            self.btn_vol_abort.setEnabled(False)

    def on_volume_progress(self, i, n, where):
        self.bar_volume.setValue(i)
        self.bar_volume.setFormat(f"line {i}/{n}  ({where})  "
                                  f"%p%")

    def on_volume_done(self, lines, err):
        """Stop logging, write what was measured, and say where it went.

        Everything here comes off the runner rather than the tab: the volume
        controls stay live during a sweep, so `self._plan` may already describe
        a different volume than the one that was just run.
        """
        runner = self.session.volume_runner
        log = self.session.volume_log
        self.session.volume_runner = None
        self.session.volume_log = None
        if float(self.session.out_rate) != self._run_hz:
            self.session.log(
                f"WARNING: volume map: the output rate changed from "
                f"{self._run_hz:g} to {self.session.out_rate:g} Hz during the "
                f"sweep. Rows are timed at {self._run_hz:g} Hz, so anything "
                f"logged after the change is mistimed -- treat the positions "
                f"in this map with suspicion.")
        self.btn_vol_abort.setEnabled(False)
        self.btn_vol_plan.setEnabled(True)
        self.btn_vol_start.setEnabled(bool(self._plan and self._plan.lines))
        self.bar_volume.setFormat("%p%")
        if err:
            self.session.log(f"WARNING: volume map: {err}")
        if log is None or not log.n_rows:
            self.session.log("volume map: nothing was logged")
            if err:
                QtWidgets.QMessageBox.warning(self, "Volume map", err)
            return
        # Written even after a failure, for the same reason a partial field map
        # is: the lines that did complete are perfectly good data, and an
        # overnight run that ends with nothing on disk is the worst outcome
        # available.
        ran = runner.plan if runner is not None else None
        path = orec.default_name("volume", "npz", self.session.out_dir)
        try:
            written = log.save(path, self._run_hz, self.session.machine.pose,
                               sync_note=self._sync_note(log, ran))
        except OSError as exc:
            self.session.log(f"WARNING: volume map: could not be written ({exc})")
            QtWidgets.QMessageBox.warning(self, "Volume map", str(exc))
            return
        done = 0 if runner is None else runner.lines_done
        self.session.log(
            f"volume map: {log.n_rows:,} rows over {done} lines "
            f"-> {written}" + (" (aborted)" if err or (
                runner is not None and runner.aborted) else ""))

    def _position_plan(self):
        """One line on where this map's positions will come from, per axis."""
        enc = self.session.encoders
        on_clock = [a for a in ("x", "y", "z") if a in enc]
        if not on_clock:
            return ("Positions: from the controllers over USB, interpolated "
                    "-- no axis has a measured encoder scale. Calibrate first "
                    "if you want them on the sample clock.")
        polled = [a for a in ("x", "y", "z") if a not in on_clock]
        out = (f"Positions: {', '.join(on_clock)} from the encoders, latched "
               f"with each field sample")
        if polled:
            out += (f"; {', '.join(polled)} from the controllers over USB, "
                    f"interpolated")
        return out + "."

    def _sync_note(self, log, ran):
        """What the position column in this map is worth, per axis.

        Written per map rather than once in the docs because it depends on
        what was actually calibrated when this one ran, and a map read next
        year cannot go and look.
        """
        speed = 0.0 if ran is None else ran.speed_mm_s
        swept = "the swept axis" if ran is None else ran.volume.sweep
        on_clock = sorted(log.encoders.calibrated)
        polled = [a for a in ("x", "y", "z") if a not in on_clock]
        bits = []
        if on_clock:
            bits.append(
                f"{', '.join(on_clock)}: taken from the quadrature counts "
                f"aggregated into acq1001_695's own sample stream, so each "
                f"position was latched by the same clock as the field sample "
                f"it is written against. Nothing is interpolated and there is "
                f"no offset to estimate. Absolute accuracy is still the "
                f"leadscrew's -- these encoders are on the motors.")
        if polled:
            bits.append(
                f"{', '.join(polled)}: no calibrated encoder, so these come "
                f"from the controller over USB, interpolated onto samples "
                f"stamped when their block reached the application. Both are "
                f"wall clocks with an unknown offset between them; it is the "
                f"same on every line, because every line is swept the same "
                f"way, so it shifts the map rigidly along {swept} rather than "
                f"distorting it, and 100 ms of it is {speed * 0.1:.1f} mm at "
                f"{speed:g} mm/s. Sweep one line backwards and halve the "
                f"apparent shift to measure it.")
        return "  ".join(bits)

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
        # "The whole travel" now means something different from what the boxes
        # are showing, so re-read it rather than leaving the fallback on screen.
        if self.chk_vol_all.isChecked():
            self.on_volume_whole(True)

    def refresh_machine(self, force=False):
        """Redraw the head where it is now, and re-measure the clearance.

        Runs off the stage timer, so it must be cheap when nothing has moved:
        the pose and the stage reading are hashed and the whole thing skipped
        when they are unchanged, which is most ticks.
        """
        if getattr(self, "machine_view", None) is None:
            return
        # Flying the planned path wins over the live reading: the whole point
        # is to see where the rig WOULD be, and a tracked stage sitting still
        # would otherwise pull the drawing back every tick.
        stage = self._preview_at or self.stage_mm()
        pose = self.session.machine.pose
        key = (pose.x_mm, pose.y_mm, pose.z_mm, pose.rot_z_deg,
               self.session.machine.coil_radius_mm,
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
            # A flown position is a made-up stage reading and must not be
            # dressed up as one: there may be no stages at all, and asking
            # them whether they are homed is both wrong and a crash.
            if self._preview_at is not None:
                note = "  (flying the planned path — nothing is moving)"
            else:
                untrusted = [n for n in sorted(stage)
                             if not self.session.stages[n].position_trusted]
                note = ("  (position not trusted on " + ", ".join(untrusted)
                        + " — home first)") if untrusted else ""
            self.lbl_machine_stage.setText(
                f"stage {where} mm  →  flange at ({origin[0]:+.0f}, "
                f"{origin[1]:+.0f}, {origin[2]:+.0f}) mm{note}")

    def set_geometry(self, geom):
        """Adopt a reloaded probe geometry and redraw."""
        self.machine_view.set_geometry(geom)
        self.refresh_machine(force=True)
