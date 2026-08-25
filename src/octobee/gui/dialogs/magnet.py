"""The guided single-magnet calibration wizard."""

import os
import time

import numpy as np
from PyQt6 import QtGui, QtWidgets

from octobee.calib import magnet as omag
from octobee.motion import scan as oscan
from octobee.calib import geometry as pgeom
from octobee.gui.sources import LiveSource
from octobee.gui.workers import ScanWorker

NO_AXIS = "(none)"


METHOD_FULL = "Full — plane sweep and standoff dither  (recommended)"


METHOD_AXIAL = "Axial only — one sweep per pose  (quick)"


METHOD_CUSTOM = "Custom — set the axes by hand"


# What each method is for, in the words someone choosing between them needs.
# Deliberately includes what each one CANNOT do: the axial run is not a
# degraded version of the full one, it is the right answer when there is only
# one stage, and the numbers are what make that judgeable rather than a matter
# of picking the option with "recommended" after it.
METHOD_NOTES = {
    "full": (
        "Three passes per pose: find the rings, cut across each one, then "
        "dither the standoff. Every sensor is measured at the top of its own "
        "peak and at a measured distance, so a millimetre of arm placement "
        "stops looking like 15 % of gain. On a synthetic probe misplaced by "
        "1 mm on all three axes this recovers gain to 1.8 %, against 9.9 % "
        "for the axial run. Needs three stages, and about five times the "
        "points."),
    "axial": (
        "One sweep along the tube per pose — the original routine. Every "
        "sensor still passes the same fixed magnet, so the trim still needs "
        "no 1/r³ model, but each chip is measured wherever its arm happens to "
        "have put it: 1 mm of misplacement is up to 15 % of trim at a 20 mm "
        "standoff. The right choice when only the tube axis is motorised, or "
        "for a quick check."),
    "custom": (
        "The axis boxes below are yours. One combination is not offered above "
        "and is worth knowing about: a transverse cut with NO standoff dither "
        "is worse than doing neither, because peaking over the plane moves "
        "every chip to its true, nearer approach and amplifies whatever error "
        "is left. The run will refuse a dither without a cut for the "
        "matching reason — the dither's model is only true once the cut has "
        "put the chip under the magnet."),
}


# What each pass is doing, for the header while it runs. The names matter more
# than they look: the three passes take very different lengths of time and an
# operator who cannot tell which one is running cannot tell a slow dither from
# a stalled stage.
PASS_NAMES = {
    "locate": "pass A, finding the rings",
    "cut": "pass B, cutting across each ring",
    "dither": "pass C, measuring the standoff",
}


class MagnetWizard(QtWidgets.QDialog):
    """Guided single-magnet calibration: four sweeps, one per quarter turn.

    The routine is in octobee/calib/magnet.py; this is the part that has to be a
    dialog, because between poses the instrument cannot do anything until a
    person has turned the head and said so. That pause is the whole reason
    this is guided rather than a button: the run is only valid if the magnet
    and the cradle stay put across all four poses, and nothing in software can
    check that.

    Each pose is an ordinary one-axis scan, run through the same ScanWorker as
    a field map -- move, settle, average at the full 200 kSPS, repeat -- so the
    per-point noise argument is identical and there is one mover, not two.
    """

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Guided magnet calibration")
        self.setModal(False)
        self.resize(760, 640)
        self.run = omag.MagnetRun(axis="y")
        self._worker = None
        self._t0 = 0.0
        self._start_mm = None
        self._finished = False
        self._park = {}          # across/normal positions, taken once
        self._setting_method = False   # guards the method -> axis-box writes
        self._pass = None        # 'locate' | 'cut' | 'dither'
        self._pending = None     # the PoseSweep the passes are building up
        self._rings = None       # where pass A found this pose's four rings
        self._released = False   # have the carriers been taken off live yet

        lay = QtWidgets.QVBoxLayout(self)
        self.lbl_step = QtWidgets.QLabel()
        f = self.lbl_step.font()
        f.setPointSize(f.pointSize() + 3)
        f.setBold(True)
        self.lbl_step.setFont(f)
        lay.addWidget(self.lbl_step)

        self.txt = QtWidgets.QTextBrowser()
        self.txt.setMaximumHeight(250)
        lay.addWidget(self.txt)

        names = list(win.session.stages.names) if win.session.stages else ["y"]

        # ---- which run this is ---------------------------------------------
        # A named choice rather than "set these two axis boxes and hope",
        # because the passes are not independent options: plane-without-dither
        # is measurably worse than doing neither, and it was previously
        # reachable by setting one combo and not the other. The methods here
        # are the combinations that are actually worth running.
        self.box_method = QtWidgets.QGroupBox("Which run")
        ml = QtWidgets.QVBoxLayout(self.box_method)
        self.cmb_method = QtWidgets.QComboBox()
        for label, tag in ((METHOD_FULL, "full"),
                           (METHOD_AXIAL, "axial"),
                           (METHOD_CUSTOM, "custom")):
            self.cmb_method.addItem(label, tag)
        self.lbl_method = QtWidgets.QLabel()
        self.lbl_method.setWordWrap(True)
        ml.addWidget(self.cmb_method)
        ml.addWidget(self.lbl_method)
        lay.addWidget(self.box_method)

        self.box_setup = QtWidgets.QGroupBox("Sweep")
        gl = QtWidgets.QGridLayout(self.box_setup)

        # ---- pass A: the axial locate
        self.cmb_axis = QtWidgets.QComboBox()
        self.cmb_axis.addItems(names)
        if "y" in names:
            self.cmb_axis.setCurrentText("y")
        self.cmb_axis.setToolTip(
            "The axis the tube lies along. The whole argument for this "
            "routine is that every sensor passes the magnet at the same "
            "approach, which is only true along the tube axis.")
        span, step = omag.suggested_sweep(win.session.geom)
        self.spin_span = QtWidgets.QDoubleSpinBox()
        self.spin_span.setRange(10.0, 300.0)
        self.spin_span.setValue(span)
        self.spin_span.setSuffix(" mm")
        self.spin_span.setToolTip(
            "How far to drive from where the head is parked now. Long enough "
            "to carry every ring past the magnet and out the other side.")
        self.spin_step = QtWidgets.QDoubleSpinBox()
        self.spin_step.setRange(0.1, 50.0)
        self.spin_step.setValue(step)
        self.spin_step.setSuffix(" mm")
        self.spin_step.setToolTip(
            "Coarse on purpose. This pass only has to find WHERE the four "
            "rings are; the transverse cut below is what measures them.")
        self.spin_secs = QtWidgets.QDoubleSpinBox()
        self.spin_secs.setRange(0.2, 30.0)
        self.spin_secs.setValue(1.0)
        self.spin_secs.setSuffix(" s")
        self.spin_secs.setToolTip(
            "Averaging time per point. The peak needs little, but the "
            "standoff dither reads a CURVATURE out of seven points and that "
            "is where the noise actually costs something -- if the standoff "
            "column comes back mostly '--', raise this before anything else.")
        self.spin_settle = QtWidgets.QDoubleSpinBox()
        self.spin_settle.setRange(0.0, 10.0)
        self.spin_settle.setValue(oscan.DEFAULT_SETTLE_S)
        self.spin_settle.setSuffix(" s")

        # ---- the standoff, which sizes both of the other passes
        self.spin_standoff = QtWidgets.QDoubleSpinBox()
        self.spin_standoff.setRange(3.0, 200.0)
        self.spin_standoff.setValue(20.0)
        self.spin_standoff.setSuffix(" mm")
        self.spin_standoff.setToolTip(
            "Roughly how far the magnet is from the chips. Nothing is "
            "measured with this -- pass C measures the real distance, "
            "per sensor -- but both of the other passes have to be SIZED "
            "from it:\n\n"
            "  - the transverse cut needs a half-span of about one standoff, "
            "because that is the width of the peak it is looking for. Much "
            "less and the cut is flat and the peak cannot be placed; much "
            "more and the extra points buy nothing.\n"
            "  - the dither needs a quarter of it, for the same reason in "
            "reverse.\n\n"
            "Within a factor of two is close enough. Changing it resizes "
            "both.")

        for col, (lbl, wdg) in enumerate((("tube axis", self.cmb_axis),
                                          ("standoff ~", self.spin_standoff),
                                          ("sweep", self.spin_span),
                                          ("step", self.spin_step),
                                          ("per point", self.spin_secs),
                                          ("settle", self.spin_settle))):
            gl.addWidget(QtWidgets.QLabel(lbl), 0, col)
            gl.addWidget(wdg, 1, col)

        # ---- pass B: the transverse cut
        self.cmb_across = QtWidgets.QComboBox()
        self.cmb_across.addItem(NO_AXIS)
        self.cmb_across.addItems([n for n in names if n != "y"])
        if "x" in names:
            self.cmb_across.setCurrentText("x")
        self.cmb_across.setToolTip(
            "The transverse axis -- across the face, the way the arms reach. "
            "Sweeping it as well as the tube axis is what lets each sensor be "
            "measured at ITS OWN peak instead of at a slice through it.\n\n"
            "Set to none and this is the old one-axis routine: still valid, "
            "but a millimetre of arm placement then costs up to 15% of trim "
            "if the magnet is not exactly on the chip line.")
        self.spin_across_half = QtWidgets.QDoubleSpinBox()
        self.spin_across_half.setRange(1.0, 100.0)
        self.spin_across_half.setSuffix(" mm")
        self.spin_across_step = QtWidgets.QDoubleSpinBox()
        self.spin_across_step.setRange(0.05, 20.0)
        self.spin_across_step.setSuffix(" mm")

        # ---- pass C: the standoff dither
        self.cmb_normal = QtWidgets.QComboBox()
        self.cmb_normal.addItem(NO_AXIS)
        self.cmb_normal.addItems([n for n in names if n != "y"])
        if "z" in names:
            self.cmb_normal.setCurrentText("z")
        self.cmb_normal.setToolTip(
            "The standoff axis -- toward the magnet. This one is NOT swept "
            "for a peak, because |B| has no maximum in that direction; it "
            "just keeps rising as the chip gets closer. A few points along it "
            "measure the distance itself, which is the last first-order error "
            "in the trim and the one the plane cannot touch.\n\n"
            "Set to none to skip it. The run is still better than a bare "
            "axial sweep, but each chip's distance stays assumed.")
        self.spin_dither_half = QtWidgets.QDoubleSpinBox()
        self.spin_dither_half.setRange(0.2, 50.0)
        self.spin_dither_half.setSuffix(" mm")
        self.spin_dither_half.setToolTip(
            "Half-span of the dither. Bigger than instinct says: the standoff "
            "comes out of the CURVATURE of the field along this axis, which "
            "is second order, so a +-1 mm dither at a 20 mm standoff buries "
            "it 0.4% under the slope and the fit returns noise.\n\n"
            "The near end of the dither is also the closest the chips get to "
            "the magnet. At a quarter of the standoff that is 2.4x the peak "
            "field -- keep the peak under about a third of the range, or the "
            "dither clips and the fit reads a flat top.")
        self.spin_dither_pts = QtWidgets.QSpinBox()
        self.spin_dither_pts.setRange(3, 21)
        self.spin_dither_pts.setValue(omag.DITHER_POINTS)
        self.spin_dither_pts.setToolTip(
            "Three is the minimum that can separate a slope from a curvature, "
            "and it leaves nothing over to judge the fit by. Seven is where "
            "the bench stopped improving.")

        for col, (lbl, wdg) in enumerate(
                (("transverse axis", self.cmb_across),
                 ("cut half-span", self.spin_across_half),
                 ("cut step", self.spin_across_step),
                 ("standoff axis", self.cmb_normal),
                 ("dither half-span", self.spin_dither_half),
                 ("dither points", self.spin_dither_pts))):
            gl.addWidget(QtWidgets.QLabel(lbl), 2, col)
            gl.addWidget(wdg, 3, col)

        self._resize_passes()
        self.spin_standoff.valueChanged.connect(self._resize_passes)

        self.lbl_points = QtWidgets.QLabel("")
        self.lbl_points.setWordWrap(True)
        gl.addWidget(self.lbl_points, 4, 0, 1, 6)
        for sp in (self.spin_span, self.spin_step, self.spin_secs,
                   self.spin_settle, self.spin_across_half,
                   self.spin_across_step, self.spin_dither_half):
            sp.valueChanged.connect(self._update_estimate)
        self.spin_dither_pts.valueChanged.connect(self._update_estimate)
        lay.addWidget(self.box_setup)

        # Wired after the axis combos exist, and in this order: the method sets
        # the combos, and the combos falling back to Custom is how a hand edit
        # is noticed. Connecting the second one first would make _apply_method's
        # own writes look like hand edits and knock it straight to Custom.
        self.cmb_method.currentIndexChanged.connect(self._apply_method)
        # The tube axis feeds into which axes the method may use, so changing
        # it has to re-derive them -- otherwise picking x as the tube axis
        # leaves the transverse cut still pointed at x.
        self.cmb_axis.currentTextChanged.connect(self._apply_method)
        for cb in (self.cmb_across, self.cmb_normal):
            cb.currentTextChanged.connect(self._method_edited)
        self._method_default(names)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setMinimumHeight(22)
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        self.bar.setFormat("no sweep running")
        lay.addWidget(self.bar)

        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setFont(QtGui.QFont("Consolas", 9))
        self.report.setPlaceholderText(
            "After the first pose, this fills with each sensor's peak, where "
            "along the sweep it happened, and the trim it implies.\n\n"
            "It is rewritten after every pose, so you can see the faces "
            "arrive one at a time — and stop early if something is obviously "
            "wrong rather than paying for all four.")
        lay.addWidget(self.report, 1)

        row = QtWidgets.QHBoxLayout()
        self.chk_apply = QtWidgets.QCheckBox(
            "apply the gain trim and save it to calibration.json")
        self.chk_apply.setChecked(True)
        self.chk_apply.setToolTip(
            "Folds the measured per-sensor response into the calibration's "
            "gain trim -- without any geometry weighting, because every "
            "sensor was measured at the top of its own peak and scaled to a "
            "common, measured standoff.\n\nIt SAVES as well as applies. An hour of "
            "measurement that only exists in memory until someone remembers "
            "to press Save calibration is an hour waiting to be lost.")
        row.addWidget(self.chk_apply)
        row.addStretch(1)
        self.btn_primary = QtWidgets.QPushButton()
        self.btn_primary.clicked.connect(self.on_primary)
        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        row.addWidget(self.btn_primary)
        row.addWidget(self.btn_close)
        lay.addLayout(row)

        self._update_estimate()
        self._refresh()

    # ---- state -----------------------------------------------------------
    @property
    def busy(self):
        return self._worker is not None and self._worker.isRunning()

    # ---- which run ------------------------------------------------------
    def _method(self):
        return self.cmb_method.currentData()

    def _method_default(self, names):
        """Pick the best method this rig can actually run, and say so.

        Full needs three axes. Offering it on a one-stage rig and then
        refusing at Start is the failure this combo exists to remove, so an
        unrunnable method is disabled in the list rather than left as a trap.
        """
        # One rule for "can this rig run the full method", shared with
        # _apply_method: two axes free of the tube axis. Counting stages
        # separately here would let the two disagree the moment the tube axis
        # moved, and the one that decides what actually gets scanned is the
        # other one.
        can_full = self._full_axes() is not None
        item = self.cmb_method.model().item(0)
        item.setEnabled(can_full)
        if not can_full:
            self.cmb_method.setItemText(
                0, METHOD_FULL.replace("(recommended)",
                                       "(needs three stages)"))
        self.cmb_method.setCurrentIndex(0 if can_full else 1)
        self._apply_method()

    def _full_axes(self):
        """(transverse, standoff) for the full run, or None if it cannot run.

        Picked against the tube axis rather than hardcoded, because the tube
        axis is a combo the operator can change. Sweeping x and cutting across
        x is the same axis twice: the grid is keyed by axis name, so the two
        collapse into one and the run silently becomes something other than
        what the panel says -- which is the failure mode this whole routine is
        built to avoid, arriving through the door marked convenience.
        """
        sweep = self.cmb_axis.currentText()
        free = [self.cmb_across.itemText(i)
                for i in range(self.cmb_across.count())]
        free = [n for n in free if n not in (NO_AXIS, sweep)]
        if len(free) < 2:
            return None
        # This rig's convention when it is available -- x across the face, z
        # toward the magnet -- and otherwise just two distinct axes.
        across = "x" if "x" in free else free[0]
        rest = [n for n in free if n != across]
        return (across, "z" if "z" in rest else rest[0])

    def _apply_method(self):
        """Drive the axis boxes from the method, and lock them unless Custom."""
        method = self._method()
        full = self._full_axes()
        # The tube axis can change under a chosen method and leave it with
        # nowhere to put the other two passes. Say so by disabling it, the
        # same way a rig with too few stages does, rather than by quietly
        # running a different scan.
        self.cmb_method.model().item(0).setEnabled(full is not None)
        if method == "full" and full is None:
            self.cmb_method.setCurrentIndex(1)      # falls back to axial
            return
        self.lbl_method.setText(METHOD_NOTES.get(method, ""))
        custom = method == "custom"
        for w in (self.cmb_across, self.cmb_normal):
            w.setEnabled(custom)
        if not custom:
            want = {"full": full, "axial": (NO_AXIS, NO_AXIS)}[method]
            self._setting_method = True
            try:
                for cb, name in zip((self.cmb_across, self.cmb_normal), want):
                    if cb.findText(name) >= 0:
                        cb.setCurrentText(name)
            finally:
                self._setting_method = False
        # Sizing only matters to the passes that exist; hiding it would move
        # the layout about, so it greys out instead.
        for w in (self.spin_across_half, self.spin_across_step):
            w.setEnabled(self._across_name() is not None)
        for w in (self.spin_dither_half, self.spin_dither_pts,
                  self.spin_standoff):
            w.setEnabled(self._normal_name() is not None
                         or self._across_name() is not None)
        self._update_estimate()

    def _method_edited(self):
        """A hand edit to an axis box means this is no longer a named method."""
        if getattr(self, "_setting_method", False):
            return
        if self._method() != "custom":
            self.cmb_method.setCurrentIndex(self.cmb_method.count() - 1)
        else:
            self._apply_method()

    def _across_name(self):
        n = self.cmb_across.currentText()
        return None if n == NO_AXIS else n

    def _normal_name(self):
        n = self.cmb_normal.currentText()
        return None if n == NO_AXIS else n

    def _resize_passes(self):
        """Re-size the cut and the dither from the nominal standoff.

        Both of them have a natural size set by the distance to the magnet and
        no natural size otherwise, so leaving the operator to pick millimetres
        out of the air is how a run comes back with a flat cut and an
        unfittable dither. They stay editable; this only moves them when the
        standoff changes.
        """
        d = self.spin_standoff.value()
        half, step = omag.suggested_plane(d)
        self.spin_across_half.setValue(half)
        self.spin_across_step.setValue(step)
        self.spin_dither_half.setValue(float(omag.suggested_dither(d)[-1]))

    def _positions(self):
        start = self._start_mm
        if start is None:
            start = self.win.session.stages[self.cmb_axis.currentText()].position_mm
        return oscan.parse_axis_spec(
            f"{start}:{start + self.spin_span.value()}:{self.spin_step.value()}")

    def _across_offsets(self):
        half, step = self.spin_across_half.value(), self.spin_across_step.value()
        return oscan.parse_axis_spec(f"{-half}:{half}:{step}")

    def _dither_offsets(self):
        half = self.spin_dither_half.value()
        return np.linspace(-half, half, int(self.spin_dither_pts.value()))

    def _pass_sizes(self):
        """(locate, cut, dither) point counts for one pose."""
        rings = pgeom.SENSORS_PER_FACE
        # Through parse_axis_spec rather than span/step + 1, so the estimate is
        # the number of points the scan will actually visit: the spec drops a
        # final point that floating point put past the stop, and an estimate
        # that is one out every time trains you to distrust it.
        n_loc = len(oscan.parse_axis_spec(
            f"0:{self.spin_span.value()}:{self.spin_step.value()}"))
        n_cut = rings * len(self._across_offsets()) if self._across_name() else 0
        n_dit = rings * int(self.spin_dither_pts.value()) \
            if self._normal_name() else 0
        return n_loc, n_cut, n_dit

    def _update_estimate(self):
        n_loc, n_cut, n_dit = self._pass_sizes()
        n = n_loc + n_cut + n_dit
        per = self.spin_secs.value() + self.spin_settle.value() + 2.0
        bits = [f"locate {n_loc}"]
        bits.append(f"cut {n_cut}" if n_cut else "no transverse cut")
        bits.append(f"dither {n_dit}" if n_dit else "no standoff dither")
        self.lbl_points.setText(
            f"{' + '.join(bits)} = {n} points per pose, about "
            f"{n * per / 60:.1f} min each — {omag.N_POSES * n * per / 60:.0f} "
            f"min for all {omag.N_POSES} poses")

    def _refresh(self):
        done = len(self.run)
        # Both boxes, not just the sweep one: changing the method between poses
        # would make pose 3 a different measurement from poses 1 and 2, and the
        # run has no way to say so afterwards.
        self.box_method.setEnabled(done == 0 and not self.busy)
        self.box_setup.setEnabled(done == 0 and not self.busy)
        self.btn_primary.setEnabled(not self.busy)
        if self.busy:
            self.lbl_step.setText(
                f"Pose {done + 1} of {omag.N_POSES} — "
                f"{PASS_NAMES.get(self._pass, 'sweeping')}")
            self.btn_primary.setText("Sweeping...")
            return
        if done == 0:
            self.lbl_step.setText("Before you start")
            self.txt.setMarkdown(SETUP_TEXT)
            self.btn_primary.setText("Start pose 1")
        elif done < omag.N_POSES:
            self.lbl_step.setText(f"Turn the head — pose {done + 1} of "
                                  f"{omag.N_POSES}")
            self.txt.setMarkdown(TURN_TEXT.format(n=done + 1))
            self.btn_primary.setText(f"Start pose {done + 1}")
        else:
            self.lbl_step.setText("All four poses recorded")
            self.txt.setMarkdown(DONE_TEXT)
            self.btn_primary.setText("Apply and save")

    # ---- running ---------------------------------------------------------
    def on_primary(self):
        if self.busy:
            return
        if len(self.run) >= omag.N_POSES:
            self.finish()
            return
        self.start_pose()

    def _check_travel(self, name, lo_off, hi_off, park):
        """True if [park+lo_off, park+hi_off] fits in this axis's envelope."""
        st = self.win.session.stages[name]
        # limit_mm, not travel_mm: this is the check that decides whether a
        # 40-minute unattended run is about to drive the head somewhere, so it
        # has to be against what the axis is allowed to use, not against the
        # length of the leadscrew.
        lo, hi = st.limit_mm
        if hi <= lo:
            return True
        if park + lo_off < lo or park + hi_off > hi:
            envelope = ("travel" if st.limit_mm == st.travel_mm
                        else "allowed range")
            QtWidgets.QMessageBox.warning(
                self, "Guided magnet calibration",
                f"The run needs {name} between {park + lo_off:.1f} and "
                f"{park + hi_off:.1f} mm, which is outside its "
                f"{lo:g}..{hi:g} mm {envelope}. Re-park the head, or shorten "
                f"that pass.")
            return False
        return True

    def start_pose(self):
        win = self.win
        if win.session.stages is None:
            QtWidgets.QMessageBox.warning(
                self, "Guided magnet calibration",
                "The stages are not connected. This routine drives the head "
                "past the magnet; without motion there is nothing to guide.")
            return
        axis = self.cmb_axis.currentText()
        across, normal = self._across_name(), self._normal_name()
        named = [n for n in (axis, across, normal) if n]
        if len(set(named)) != len(named):
            # The grid is a dict keyed by axis name, so naming one axis twice
            # does not fail -- the second silently replaces the first and the
            # run becomes a different scan from the one on the panel, with a
            # pass missing and nothing in the output to say so. Custom is the
            # way in: the named methods derive their axes and cannot collide.
            QtWidgets.QMessageBox.warning(
                self, "Guided magnet calibration",
                f"The same axis is named twice: tube {axis}, transverse "
                f"{across or 'none'}, standoff {normal or 'none'}.\n\n"
                f"Each pass has to move a different axis. One axis doing two "
                f"jobs does not fail — it quietly drops a pass and returns a "
                f"result that looks like a complete run.")
            return
        if normal and not across:
            # Not a fussy pairing rule -- the dither's model is "move straight
            # toward the magnet", and the transverse cut is the only thing that
            # makes that true. Off to one side by a, with the magnet h away
            # along the dither axis, the fit returns h*r^2/(h^2 - a^2) instead
            # of the distance: on this rig's geometry, 51 mm for a chip that is
            # really 26 mm away. Correcting with that number is worse than not
            # correcting.
            QtWidgets.QMessageBox.warning(
                self, "Guided magnet calibration",
                "The standoff dither needs the transverse cut.\n\n"
                "The dither measures distance by assuming it is moving "
                "straight toward the magnet, and the cut is what puts each "
                "chip under the magnet so that it is. Without it the fit "
                "returns a number that is not the distance — on this "
                "geometry it reads 51 mm for a chip 26 mm away — and "
                "correcting with that is worse than not correcting at all.\n\n"
                "Either pick a transverse axis as well, or set the standoff "
                "axis to none and run the plain axial sweep.")
            return
        if win.motion.latched:
            QtWidgets.QMessageBox.warning(
                self, "Guided magnet calibration",
                f"The emergency stop is latched: {win.motion.reason}.\n\n"
                f"Reset it before driving the head.")
            return
        for name in (axis, across, normal):
            if name and not win.session.stages[name].position_trusted:
                QtWidgets.QMessageBox.warning(
                    self, "Guided magnet calibration",
                    f"Axis {name}: {win.session.stages[name].distrust_reason}.\n\n"
                    f"Where it says it is and where it is are unrelated. The "
                    f"peaks would still line up with each other, but nothing "
                    f"else could be compared with them afterwards. Home it "
                    f"first.")
                return

        # The park is taken once, on the first pose, and reused: the whole
        # routine rests on all four poses being measured against the same
        # fixed magnet, so re-reading the stage each time would let a nudged
        # axis redefine the origin halfway through without saying so.
        if self._start_mm is None:
            self._start_mm = win.session.stages[axis].position_mm
            self._park = {n: win.session.stages[n].position_mm
                          for n in (across, normal) if n}
        if not self._check_travel(axis, 0.0, self.spin_span.value(),
                                  self._start_mm):
            self._start_mm = None
            return
        if across and not self._check_travel(
                across, -self.spin_across_half.value(),
                self.spin_across_half.value(), self._park[across]):
            return
        if normal and not self._check_travel(
                normal, -self.spin_dither_half.value(),
                self.spin_dither_half.value(), self._park[normal]):
            return

        self._pending = None
        self._rings = None
        self._run_pass("locate")

    def _grid_for(self, kind):
        """The ScanGrid for one pass.

        Every pass names the same axes in the same order, even the ones it does
        not move. That is what lets the passes be concatenated afterwards: a
        FieldMap only records the columns its grid named, so a locate that
        named one axis and a cut that named three would come back as two point
        clouds in different spaces with no way to say where the first one was
        on the axes it left out.
        """
        axis = self.cmb_axis.currentText()
        across, normal = self._across_name(), self._normal_name()
        if kind == "locate":
            along = self._positions()
        else:
            along = np.asarray(self._rings, float)

        axes = {axis: along}
        if across:
            axes[across] = (self._park[across] + self._across_offsets()
                            if kind == "cut" else [self._cut_centre()])
        if normal:
            axes[normal] = (self._park[normal] + self._dither_offsets()
                            if kind == "dither" else [self._park[normal]])
        return oscan.ScanGrid(axes)

    def _cut_centre(self):
        """Where to park the transverse axis for the dither.

        The mean of the four rings' transverse peaks, not each ring's own. A
        ring whose arm sits a millimetre off the others is a millimetre off the
        top of a peak that is flat to second order there, which costs 0.4 % of
        field at a 20 mm standoff -- far below what the dither can resolve, and
        worth trading for a dither that is one grid instead of four.
        """
        park = self._park.get(self._across_name())
        if self._pending is None or not self._pending.is_plane:
            return park
        across = self._pending.peak_across_mm
        loud = np.argsort(self._pending.peaks)[-pgeom.SENSORS_PER_FACE:]
        vals = across[loud]
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if len(vals) else park

    def _run_pass(self, kind):
        win = self.win
        self._pass = kind
        grid = self._grid_for(kind)

        # First pass of the first pose only: the carriers are still on the live
        # stream, and the worker is what puts them back on their own clock.
        # Everything after finds them already released, so passing the saved
        # clkdiv again would restore a rate that is no longer the one in force.
        first = not self._released
        if first:
            self._released = True
            if win.act_record.isChecked():
                win.act_record.setChecked(False)
            if isinstance(win.session.source, LiveSource):
                win.session.source.stop()
            win.session.source = None
            win.lbl_state.setText("guided magnet calibration in progress...")
        restore = win.session.prev_clkdiv if first else {}
        if first:
            win.session.prev_clkdiv = {}

        self.bar.setRange(0, len(grid))
        self.bar.setValue(0)
        self.bar.setFormat("%v / %m")
        self._t0 = time.time()
        self._worker = ScanWorker(win.session.hosts, win.session.stages, grid,
                                  self.spin_secs.value(), win.session.cal,
                                  self.spin_settle.value(), restore)
        self._worker.message.connect(win.session.log)
        self._worker.progress.connect(self.on_progress)
        self._worker.done.connect(self.on_pass_done)
        # Without this the main window's stop button does not know this thread
        # exists, and stopping the axes mid-pass would leave it to carry on to
        # the next one -- see MainWindow.register_motion_worker.
        win.motion.register(self._worker)
        win.session.log(f"guided magnet calibration: pose {len(self.run) + 1} of "
                    f"{omag.N_POSES}, {PASS_NAMES[kind]}, {len(grid)} points "
                    f"-- {grid.describe()}")
        self._worker.start()
        self._refresh()

    def on_progress(self, i, n, where, sem_ut):
        self.bar.setValue(i)
        elapsed = time.time() - self._t0
        self.bar.setFormat(f"%v / %m — {elapsed / max(i, 1) * (n - i):.0f} s left")

    def _sweep_from(self, fm, pose):
        return omag.PoseSweep.from_fieldmap(
            pose, fm, axis=self.cmb_axis.currentText(),
            across=self._across_name(), normal=self._normal_name(),
            note=f"pose {pose + 1}")

    def _dithers_from(self, fm):
        """Split the dither pass into one Dither per ring.

        Grouped by which ring each row is nearest rather than by counting rows
        off in blocks: run_scan drops a point that failed and carries on, so
        the blocks are not guaranteed to be the length the grid implies, and
        counting would silently slice one ring's dither across two.
        """
        axis, normal = self.cmb_axis.currentText(), self._normal_name()
        pos = np.asarray(fm.pos_mm, float)
        names = list(fm.axes)
        if normal is None or normal not in names or not len(pos):
            return []
        ai, ni = names.index(axis), names.index(normal)
        rings = np.asarray(self._rings, float)
        which = np.argmin(np.abs(pos[:, ai][:, None] - rings[None, :]), axis=1)
        out = []
        for k in range(len(rings)):
            sel = which == k
            if int(sel.sum()) < 3:
                continue
            z = pos[sel, ni]
            out.append(omag.Dither(z - z.mean(), fm.b_mt[sel],
                                   at_mm=float(np.mean(pos[sel, ai]))))
        return out

    def on_pass_done(self, fm, error):
        worker, self._worker = self._worker, None
        self.win.motion.retire(worker)
        self.bar.setFormat("no sweep running")
        kind, self._pass = self._pass, None
        self.win.tab_stages.sync_controls()
        if self.win.motion.latched:
            # An aborted pass comes back with no error and a partial FieldMap
            # -- that is deliberate, it is how a scan keeps the points it did
            # take. But this routine chains: locate starts cut, cut starts
            # dither. Without this the stop would end one pass and immediately
            # start the next, which is not what anyone pressing it meant, and
            # the pose would be built out of half a sweep.
            self.win.session.log("guided magnet calibration: stopped by the "
                             "emergency stop — this pose is abandoned")
            self._pending = None
            self._rings = None
            self._refresh()
            return
        if error:
            self.win.session.log(f"guided magnet calibration: {error}")
            QtWidgets.QMessageBox.warning(self, "Guided magnet calibration",
                                          error)
            self._refresh()
            return
        if fm is None or not len(fm):
            self._refresh()
            return
        pose = len(self.run)

        if kind == "locate":
            self._pending = self._sweep_from(fm, pose)
            self._rings = omag.ring_positions(self._pending)
            self.win.session.log(
                "guided magnet calibration: rings at "
                + ", ".join(f"{v:.1f}" for v in self._rings) + " mm")
            if self._across_name():
                self._run_pass("cut")
                return
        elif kind == "cut":
            self._pending.merge(self._sweep_from(fm, pose))
        elif kind == "dither":
            self._pending.dithers.extend(self._dithers_from(fm))
            self._finish_pose()
            return

        if self._normal_name():
            self._run_pass("dither")
            return
        self._finish_pose()

    def _finish_pose(self):
        self.run.across = self._across_name()
        self.run.normal = self._normal_name()
        self.run.add(self._pending)
        sw, self._pending = self._pending, None
        peaks = sw.peaks
        loud = int(np.argmax(peaks)) + 1
        extra = ""
        if sw.dithers:
            d = sw.standoff_mm[np.argsort(peaks)[-pgeom.SENSORS_PER_FACE:]]
            d = d[np.isfinite(d)]
            extra = (f", standoff {np.mean(d):.1f} mm" if len(d)
                     else ", standoff not fittable -- try a longer average")
        self.win.session.log(
            f"guided magnet calibration: pose {len(self.run)} done, loudest "
            f"S{loud} at {peaks.max():.3f} mT{extra}")
        self.report.setPlainText(self.run.report(self.win.session.geom))
        self._refresh()

    # ---- finishing -------------------------------------------------------
    def finish(self):
        win = self.win
        base = os.path.join(win.session.out_dir,
                            time.strftime("magcal_%Y%m%d_%H%M%S"))
        path = self.run.save(base)
        win.session.log(f"guided magnet calibration: saved {path}")
        # Said before the trim is touched, because it decides what the trim
        # is worth: opposite faces that disagree mean part of it is geometry.
        balance = self.run.face_balance(win.session.geom)
        for note in balance["notes"]:
            win.session.log(f"guided magnet calibration: {note}")

        if self.chk_apply.isChecked():
            resp, _best = self.run.response()
            _corr, skipped = win.session.cal.cross_calibrate(resp)
            note = (f"kept their previous trim: {', '.join(skipped)}"
                    if skipped else "every live sensor trimmed")
            passes = ["axial"]
            if any(s.is_plane for s in self.run.sweeps):
                passes.append("plane")
            if any(s.dithers for s in self.run.sweeps):
                passes.append("standoff dither")
            win.session.cal.notes = (f"gain trim from the guided magnet run of "
                             f"{time.strftime('%Y-%m-%d %H:%M')} "
                             f"({os.path.basename(path)}); passes: "
                             f"{', '.join(passes)}; no geometry weighting -- "
                             f"every sensor was measured at its own peak")
            cal_path = win.session.cal.save(win.session.args.calibration)
            win.session.log(f"gain trim applied from the guided run "
                        f"(no geometry weighting needed); {note} -> "
                        f"{cal_path}")
            win._calibration_changed("the gain trim")
            win.refresh_cal_report()
            if balance["notes"]:
                QtWidgets.QMessageBox.warning(
                    self, "Gain trim applied, with a caveat",
                    "The trim is applied and saved, but this run's opposite "
                    "faces disagree:\n\n  - "
                    + "\n  - ".join(balance["notes"])
                    + "\n\nThe within-face numbers are unaffected — those "
                      "four sensors never moved relative to each other. It is "
                      "the face-to-face part of the trim that is carrying "
                      "geometry as well as gain.")
        else:
            win.session.log(f"gain trim NOT applied. The run is on disk at {path} "
                        f"and nothing about it is lost -- it can be applied "
                        f"later without repeating the measurement.")
        notes = self.run.check_geometry(win.session.geom)
        if notes:
            self.offer_geometry_update(notes)
        win.session.log("guided magnet calibration: the carriers are still off "
                    "the live stream — press Connect to go back to live.")
        self._finished = True
        self.btn_primary.setEnabled(False)
        self.lbl_step.setText("Finished")

    def offer_geometry_update(self, notes):
        """Report what the run disagrees with, and offer to write it down.

        The offer is deliberately explicit about what changes and what does
        not. Slots are a measurement: which sensor passed the magnet first is
        not a matter of opinion. Face NUMBERS are not -- turning the tube in
        its cradle renames all four -- so the grouping is checked and the
        labels are left alone.
        """
        win = self.win
        try:
            slots = self.run.measured_slots()
        except ValueError as exc:
            QtWidgets.QMessageBox.information(
                self, "Geometry",
                "The run disagrees with probe_geometry.json:\n\n  - "
                + "\n  - ".join(notes)
                + f"\n\nIt cannot say what the right answer is, though: "
                  f"{exc}\n\nNothing has been changed.")
            return
        diff = [(sid, win.session.geom.slot(sid), slot)
                for sid, slot in sorted(slots.items())
                if win.session.geom.slot(sid) != slot]
        if not diff:
            return
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        box.setWindowTitle("Geometry")
        box.setText("The run disagrees with probe_geometry.json.")
        box.setInformativeText(
            "  - " + "\n  - ".join(notes)
            + "\n\nThe magnet says these sensors sit elsewhere along the "
              "tube:\n\n"
            + "\n".join(f"    S{sid}:  slot {was}  →  slot {now}"
                         for sid, was, now in diff)
            + "\n\nOnly the slots change. Which face index a group of four "
              "carries is a naming choice the magnet cannot make — turning "
              "the tube renames them all — so the face labels are left as "
              "they are.")
        write = box.addButton("Write it to probe_geometry.json",
                              QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Leave the file alone",
                      QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is not write:
            win.session.log("guided magnet calibration: geometry left as it was")
            return
        changed = self.run.apply_to_geometry(win.session.geom)
        path = win.session.geom.save(win.session.args.geometry)
        win.session.log("geometry updated from the magnet run: "
                    + ", ".join(f"S{sid} slot {was}->{now}"
                                for sid, was, now in changed)
                    + f" -> {path}")
        # Re-read it rather than refreshing by hand: Reload geometry already
        # knows every place a position reaches -- the 3D view, the plot, the
        # sensor table, the demo source -- and a second copy of that list here
        # would be one item short the first time somebody adds a fifth.
        win.on_reload_geometry()

    def closeEvent(self, event):
        """Stop cleanly, and let QDialog finish the job.

        The super() call is not decoration. QDialog.closeEvent() is what calls
        reject() and therefore emits finished(), which is the signal the main
        window uses to forget this dialog. Accepting the event and returning
        looks identical on screen and leaves the window holding a reference to
        a hidden dialog for ever -- after which the menu item that opens this
        quietly does nothing, because it raises the dead one instead.
        """
        if self.busy:
            reply = QtWidgets.QMessageBox.question(
                self, "Guided magnet calibration",
                "A sweep is running. Stop it and close?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel)
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._worker.abort()
            self._worker.wait(30000)
        if self.run.sweeps and not self._finished:
            # Poses recorded but never applied: closing here throws away the
            # only copy, since the run is written to disk by finish().
            reply = QtWidgets.QMessageBox.question(
                self, "Guided magnet calibration",
                f"{len(self.run)} pose(s) recorded and not saved.\n\n"
                f"They are only in this window — closing discards them, and "
                f"the sweeps would have to be driven again. Close anyway?",
                QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel)
            if reply != QtWidgets.QMessageBox.StandardButton.Discard:
                event.ignore()
                return
            self.win.session.log(f"guided magnet calibration: {len(self.run)} "
                             f"unsaved pose(s) discarded")
        super().closeEvent(event)


SETUP_TEXT = """
**One magnet, clamped once, and the head driven past it four times.**

Before pose 1:

1. **Clamp the magnet** beside the head's travel, level with one face and a few
   centimetres clear of it. It must not move again until all four poses are
   done — it is the common reference for every sensor.
2. **Park the head** so the magnet is just clear of the first ring of sensors,
   and roughly centred on them on the other two axes. The run starts from
   wherever the head is now.
3. Note the rough **standoff** — how far the magnet is from the chips — into the
   box below. Nothing is measured with it, but it sizes the other two passes.
4. Nothing ferrous may move nearby during a sweep, including you.

Pick the run at the top. **Full** is three passes per pose, and they do
different jobs:

* **A, locate** — a coarse run along the tube. Finds where the four rings are.
* **B, cut** — a fine sweep *across* the face at each ring. Every sensor gets
  measured at the top of its own peak instead of at a slice through it, which
  is what stops a millimetre of arm placement becoming 15 % of fake gain.
* **C, dither** — a few points *toward and away from* the magnet at each ring.
  |B| has no maximum in that direction, so there is no peak to find; what it
  measures is the distance itself, which is the last error the plane can't
  reach.

**Axial only** runs pass A alone — the original routine, and the right choice
when only the tube axis is motorised.

You do not have to park perfectly. Finding each peak is what pass B is for —
that is most of the point of it.
"""


TURN_TEXT = """
**Index the tube one quarter turn** in its cradle, to pose {n}.

* About the tube's **own axis and nothing else**. A pose that also shifts the
  head sideways breaks the equal-approach argument quietly — the numbers still
  look plausible afterwards.
* The exact angle does not matter. Four faces getting their turn does.
* Do not touch the magnet, the cradle position, or the cable dress.

Then start pose {n}.
"""


DONE_TEXT = """
All four faces have had their turn at the magnet. The table below is the
result: each sensor's peak, taken from the pose where its own face was toward
the magnet, at the top of its own transverse peak, and scaled to a common
standoff — so the spread is gain and nothing else.

Read the **standoff** column before the trim. If it is mostly `--`, pass C
could not fit and the distances are assumed rather than measured; raise the
averaging time and run again. If the spread of standoffs is large, that is the
arms, and it is exactly what would otherwise have shown up as gain.

**Apply and save** writes the run to the captures folder, folds the trim into
the calibration if the box is ticked, and reports anything the run disagrees
with in `probe_geometry.json` — it never rewrites the geometry itself.
"""
