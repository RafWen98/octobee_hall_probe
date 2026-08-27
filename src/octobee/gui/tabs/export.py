"""
octobee/gui/tabs/export.py -- what gets written, and a record of what was.

Two quite different things share this tab. Continuous recording is started from
the toolbar and runs while you work; the controls here decide what it writes.
One-shot exports are finished the moment you press them.

The tab owns the choices and the receipt. It does not own the recording
lifecycle itself -- that belongs to the window, because starting a recording
means taking over the stream, and stopping one means the toolbar button has to
come back up whether the close succeeded or not.
"""

import time

from PyQt6 import QtCore, QtGui, QtWidgets

from octobee import record as orec
from octobee.calib import convert as ocal


class ExportTab(QtWidgets.QWidget):
    """Recording options, one-shot exports, and the list of files written."""

    snapshot_requested = QtCore.pyqtSignal()

    def __init__(self, session, sensor_table, magnet_geometry=None,
                 parent=None):
        """
        `sensor_table` returns the most recent per-sensor summary, or None.
        `magnet_geometry` returns the geometry correction to fold into a
        report -- (point_mm, exponent) -- or None if it is not wanted. Both
        belong to other tabs; this one only needs their answers.
        """
        super().__init__(parent)
        self.session = session
        self._sensor_table = sensor_table
        self._magnet_geometry = magnet_geometry or (lambda: None)

        lay = QtWidgets.QVBoxLayout(self)

        g1 = QtWidgets.QGroupBox("Continuous recording (the Record button)")
        f1 = QtWidgets.QGridLayout(g1)
        self.chk_csv = QtWidgets.QCheckBox("calibrated CSV (millitesla)")
        self.chk_csv.setChecked(True)
        self.chk_csv.setToolTip(
            "One row per output sample: t_s, then each sensor's Bx/By/Bz and "
            "|B|.\n\n"
            "Where the stream carries encoder counts -- acq1001_695 only -- "
            "the counts for those same rows are appended, and with them the "
            "X/Y/Z travel in millimetres for every axis whose counts/mm has "
            "been fitted on the Machine tab. Those positions were latched by "
            "the ADC clock in the samples the row was averaged from, so they "
            "need no interpolation against the field.\n\n"
            "A column named X_mm is absolute, anchored to the controller when "
            "Record was pressed; one named X_rel_mm is travel from the first "
            "row, which is what you get when the stages are not connected or "
            "have not been homed.")
        self.chk_raw = QtWidgets.QCheckBox(
            "raw counts, full stream rate (.bin + .json sidecar)")
        self.chk_tube = QtWidgets.QCheckBox(
            "rotate into the common tube frame")
        self.chk_tube.setToolTip(
            "Chip-frame axes point 16 different ways, so only |B| is "
            "comparable between sensors. Tube frame makes the components "
            "comparable too -- at the cost of depending on the geometry file "
            "being right.")
        f1.addWidget(self.chk_csv, 0, 0)
        f1.addWidget(self.chk_tube, 0, 1)
        f1.addWidget(self.chk_raw, 1, 0, 1, 2)
        self.lbl_recinfo = QtWidgets.QLabel("not recording")
        f1.addWidget(self.lbl_recinfo, 2, 0, 1, 2)
        lay.addWidget(g1)

        g2 = QtWidgets.QGroupBox("One-shot exports")
        f2 = QtWidgets.QHBoxLayout(g2)
        for text, slot in (
                ("Snapshot to .npz (full rate)", self._request_snapshot),
                ("Sensor summary CSV", self.export_summary),
                ("Full report JSON", self.export_json)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            f2.addWidget(b)
        self.spin_snap_s = QtWidgets.QDoubleSpinBox()
        self.spin_snap_s.setRange(0.2, 30.0)
        self.spin_snap_s.setValue(3.0)
        self.spin_snap_s.setSuffix(" s")
        f2.addWidget(QtWidgets.QLabel("snapshot length"))
        f2.addWidget(self.spin_snap_s)
        f2.addStretch(1)
        lay.addWidget(g2)

        self.export_log = QtWidgets.QPlainTextEdit()
        self.export_log.setReadOnly(True)
        self.export_log.setFont(QtGui.QFont("Consolas", 9))
        lay.addWidget(self.export_log, 1)

    # ---- what the recorder asks us ----------------------------------------

    def csv_enabled(self):
        return self.chk_csv.isChecked()

    def raw_enabled(self):
        return self.chk_raw.isChecked()

    def tube_frame(self):
        return self.chk_tube.isChecked()

    def set_recording_text(self, txt):
        self.lbl_recinfo.setText(txt or "not recording")

    # ---- the receipt ------------------------------------------------------

    def note(self, what):
        """Record that a file was written, here and in the log."""
        self.export_log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {what}")
        self.session.log(f"wrote {what}")

    # ---- handlers ---------------------------------------------------------

    def snapshot_seconds(self):
        return self.spin_snap_s.value()

    def _request_snapshot(self):
        self.snapshot_requested.emit()

    def export_summary(self):
        table = self._sensor_table()
        if not table:
            self.session.log("no sensor data yet")
            return
        path = orec.default_name("sensor_summary", "csv", self.session.out_dir)
        orec.write_sensor_csv(path, table)
        self.note(path)

    def export_json(self):
        table = self._sensor_table()
        if not table:
            self.session.log("no sensor data yet")
            return
        s = self.session
        live = s.cal.live_mask()
        source = s.source
        payload = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hosts": list(source.hosts) if source else [],
            "stream_rate_hz": source.fs_hz if source else None,
            "output_rate_hz": s.out_rate,
            "calibration": s.cal.to_dict(),
            "geometry": s.geom.to_dict(),
            "sensors": table,
            "channel_health": s.last_health or [],
        }
        if s.magnet_peaks is not None:
            payload["magnet_pass"] = {
                "peak_absB_mT": s.magnet_peaks,
                "spread": ocal.spread_report(s.magnet_peaks, live=live),
            }
            corr = self._magnet_geometry()
            if corr is not None:
                point_mm, exponent = corr
                payload["magnet_pass"]["magnet_point_mm"] = point_mm
                payload["magnet_pass"]["geometry_corrected"] = (
                    ocal.spread_report(s.magnet_peaks, s.geom, point_mm,
                                       exponent, live))
        path = orec.default_name("octobee_report", "json", s.out_dir)
        orec.write_report_json(path, payload)
        self.note(path)
