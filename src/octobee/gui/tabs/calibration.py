"""
octobee/gui/tabs/calibration.py -- the calibration procedure, in the order it
has to be done.

Zero point, then a magnet pass to equalise response, then the Earth-field roll
sweep that matches the sensors properly, then the file itself. The tab is laid
out in that order because doing them out of order produces numbers that look
fine and are wrong -- a magnet pass taken before the tare measures the offset
as if it were signal.

Two signals rather than reaching across the window:

    calibration_changed(what)  the conversion moved, so anything holding
                               millitesla computed the old way is stale
    geometry_changed()         the probe geometry was reloaded, so everything
                               that draws it has to be rebuilt
"""

import os
import time

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from octobee.calib import convert as ocal
from octobee.calib import geometry as pgeom
from octobee.acq import carrier as ob
from octobee.calib import roll as opc
from octobee.gui.constants import N_SENSORS
from octobee.gui.dialogs.magnet import MagnetWizard
from octobee.gui.sources import DemoSource


class CalibrationTab(QtWidgets.QWidget):
    """Tare, magnet pass, roll sweep, and the calibration file."""

    calibration_changed = QtCore.pyqtSignal(str)
    geometry_changed = QtCore.pyqtSignal()

    def __init__(self, session, set_ranges, parent=None):
        """The calibration procedure, in the order it has to be done."""
        super().__init__(parent)
        self.session = session
        self._set_ranges = set_ranges
        self._magnet_wizard = None
        lay = QtWidgets.QVBoxLayout(self)

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

        # A calibration change abandons a magnet pass in progress: the
        # baseline it was measuring against just moved, so the rest of the
        # pass would measure the change rather than the magnet.
        self.calibration_changed.connect(self._abandon_magnet_pass)

    def report_source(self):
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

    def start_collect(self, what, seconds):
        if self.session.source is None:
            self.session.log("not connected")
            return
        self.session.collecting = {"what": what, "blocks": [], "n": 0,
                           "need": int(seconds * self.session.source.fs_hz),
                           "peak": None, "baseline": None, "tag": None,
                           "decim": self.session.decim()}
        self.session.log(f"collecting {seconds:g} s for {what}...")

    def collect_block(self, b, grouped=None):
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
        self.calibration_changed.emit("the zero point")
        self.refresh_cal_report()

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
        self.calibration_changed.emit("the roll calibration")
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
        self.calibration_changed.emit("the zero point")
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
                               "tag": None, "decim": self.session.decim()}
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
        self.calibration_changed.emit("the gain trim")
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
        self._magnet_wizard = MagnetWizard(self.window())
        self._magnet_wizard.finished.connect(
            lambda _: setattr(self, "_magnet_wizard", None))
        self._magnet_wizard.show()

    def on_show_superseded(self, on):
        for box in self._superseded_boxes:
            box.setVisible(bool(on))

    def on_clear_gain(self):
        self.session.cal.clear_gain()
        self.session.log("gain trim cleared")
        self.calibration_changed.emit("the gain trim")
        self.refresh_cal_report()

    def on_range_changed(self, row, value):
        self.session.cal.ranges_mt[row] = value
        self.session.log(f"S{row+1} range set to +/-{value:g} mT "
                     f"({ob.RANGE_TO_VPT[value]:g} V/T)")
        self.calibration_changed.emit(f"the S{row+1} range")
        self.refresh_cal_report()

    def on_vcm_toggle(self, on):
        self.session.cal.subtract_vcm = bool(on)
        if not on:
            self.session.log("WARNING: VCM subtraction off -- readings now include "
                         "each chip's ~2.2 V virtual ground offset")
        self.calibration_changed.emit("VCM subtraction")
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
            kept = self.session.cal.archived_to
            self.session.log(
                f"calibration written to {path}"
                + (f", archived as {os.path.basename(kept)}" if kept else ""))

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
        self._set_ranges(self.session.cal.ranges_mt)
        self.chk_vcm.setChecked(self.session.cal.subtract_vcm)
        self.calibration_changed.emit("the whole calibration")
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
        self.session.probe_cloud = None
        # The 3D views, the plot and the sensor table all draw geometry.
        self.geometry_changed.emit()

        if isinstance(self.session.source, DemoSource):
            self.session.source.geom = self.session.geom
        self.session.log(f"geometry reloaded from {self.session.args.geometry}")

    def magnet_geometry_correction(self):
        """The magnet-pass geometry correction, or None if it is not wanted.

        Handed to the Data output tab so a report can carry it. The controls
        belong to the Calibration tab; what a report needs is the answer.
        """
        if not self.chk_geom.isChecked():
            return None
        return ([self.spin_mx.value(), self.spin_my.value(),
                 self.spin_mz.value()], self.spin_exp.value())

    def _abandon_magnet_pass(self, what):
        collecting = self.session.collecting
        if collecting is not None and collecting["what"] == "magnet":
            self.btn_magnet.setChecked(False)
            self.session.log(f"magnet pass abandoned: {what} changed mid-pass")
