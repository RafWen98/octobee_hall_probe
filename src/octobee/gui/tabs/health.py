"""
octobee/gui/tabs/health.py -- railed, stuck and noisy channels.

Reports every one of the 64 raw channels rather than the 48 that carry field.
The VCM rows are the point: VCM carries no field, so noise there is analogue
pickup in the cabling and grounding rather than anything a calibration can fix,
and on this probe it climbs steadily along each concentrator -- which is a
wiring fault stated plainly, instead of a slightly disappointing noise figure
spread across sixteen sensors.
"""

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from octobee import record as orec
from octobee.calib import convert as ocal
from octobee.gui.constants import N_SENSORS


class HealthTab(QtWidgets.QWidget):
    """Per-channel diagnostics for the last few seconds of raw data."""

    exported = QtCore.pyqtSignal(str)      # a file was written
    wants_focus = QtCore.pyqtSignal()      # bring this tab to the front

    def __init__(self, session, raw_arrays, parent=None):
        """
        `raw_arrays` is a callable returning the recent raw counts per box, or
        None when nothing is connected. It belongs to the acquisition path;
        this tab only needs the answer.
        """
        super().__init__(parent)
        self.session = session
        self._raw_arrays = raw_arrays

        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Analyse the last few seconds")
        b.clicked.connect(self.analyse)
        top.addWidget(b)
        b2 = QtWidgets.QPushButton("Export channel health CSV")
        b2.clicked.connect(self.export_csv)
        top.addWidget(b2)
        self.chk_autodead = QtWidgets.QCheckBox(
            "auto-exclude dead sensors from statistics")
        self.chk_autodead.setChecked(True)
        top.addWidget(self.chk_autodead)
        top.addStretch(1)
        lay.addLayout(top)

        self.health_text = QtWidgets.QPlainTextEdit()
        self.health_text.setReadOnly(True)
        self.health_text.setFont(QtGui.QFont("Consolas", 9))
        self.health_text.setPlainText(
            "Run this with no magnet near the probe.\n\n"
            "It reports every one of the 64 raw channels: mean, noise, and "
            "whether it is railed or stuck.\n\n"
            "Read the VCM rows first. VCM carries no field, so noise there is "
            "analogue pickup in the cabling and grounding, not a sensor "
            "problem -- and on this probe it grows steadily along each "
            "concentrator, which is a wiring fault, not a calibration one.")
        lay.addWidget(self.health_text)

    # ---- what the Live tab asks us ----------------------------------------

    def auto_exclude_dead(self):
        """Whether statistics elsewhere should drop sensors flagged dead."""
        return self.chk_autodead.isChecked()

    # ---- handlers ---------------------------------------------------------

    def analyse(self):
        raw = self._raw_arrays()
        if raw is None:
            self.health_text.setPlainText("no data -- connect first")
            return
        source = self.session.source
        rows = ocal.channel_health(raw, source.vpc, source.hosts,
                                   source.volt_offset)
        self.session.last_health = rows
        verdict = ocal.health_verdict(rows)
        n = raw[0].shape[0]
        out = [f"{n} samples per box at {source.fs_hz/1e3:g} kSPS "
               f"({n/source.fs_hz:.2f} s), 1 count = "
               f"{source.vpc[0]*1e6:.1f} uV", ""]
        out.append(f"{'host':>13} {'ch':>3} {'signal':>9} {'mean [V]':>10} "
                   f"{'noise [counts]':>15} {'p-p':>8}  flag")
        for r in rows:
            flag = ocal.bad_reason(r).upper()
            out.append(f"{r['host']:>13} {r['ch']:3d} "
                       f"{r['sensor']+' '+r['axis']:>9} {r['mean_v']:10.4f} "
                       f"{r['std_counts']:15.2f} {r['p2p_counts']:8.0f}  {flag}")
        out += ["", "per-sensor verdict:"]
        for sid in range(1, N_SENSORS + 1):
            st, note = verdict[sid]
            out.append(f"  S{sid:<3d} {st:<8s} {note}")

        vcm = [r for r in rows if r["is_vcm"]]
        out += ["", "VCM reference channels (these carry NO field, so any noise "
                    "on them is analogue pickup in the cabling and grounding):"]
        for r in vcm:
            out.append(f"  {r['sensor']:>4} {r['std_counts']:7.2f} counts rms "
                       f"({r['std_uv']:8.1f} uV)   VCM = {r['mean_v']:.4f} V")
        vs = np.array([r["mean_v"] for r in vcm])
        if vs.size:
            spread_v = vs.max() - vs.min()
            vpt = self.session.cal.volts_per_tesla.mean()
            out.append(f"\nVCM spread across the chips: {spread_v*1e3:.1f} mV "
                       f"= {spread_v/vpt*1e3:.2f} mT of apparent field if it "
                       f"were not subtracted.")
        trend = [r["std_counts"] for r in vcm]
        if len(trend) >= 8 and trend[7] > 3 * max(trend[0], 0.1):
            out.append("VCM noise climbs steadily along the first concentrator "
                       "-- that pattern is a cabling/ground problem, not a "
                       "sensor calibration problem.")
        self.health_text.setPlainText("\n".join(out))
        self.wants_focus.emit()

    def export_csv(self):
        if not self.session.last_health:
            self.analyse()
        if not self.session.last_health:
            return
        path = orec.default_name("channel_health", "csv", self.session.out_dir)
        orec.write_health_csv(path, self.session.last_health)
        self.exported.emit(path)
