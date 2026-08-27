"""Pressing Record must write where the head was, not only what the field was.

The pieces already existed -- the 695 puts encoder counts in every frame, the
acquisition tick unwraps and decimates them for the sweep -- but the Record
button's CSV never saw them, so a recording made while the stage travelled
described a field with no position against it.

This drives the whole path: a source that carries counts, the tick that pairs
them with the field, the datum taken from the controllers, and the file that
comes out.
"""

import argparse
import os

import numpy as np

from octobee.calib import geometry as pgeom
from octobee.gui import window as gui
from octobee.gui.sources import DemoSource
from octobee.motion import encoder as oenc
from tests.helpers import (
    check,
    pump,
    read_csv,
)

COUNTS_PER_MM = 2000.0


class EncoderDemoSource(DemoSource):
    """The demo probe, plus a 695's three quadrature columns.

    x advances at a steady 10 mm/s, y and z stand still. The counts are raw
    uint32 as they come off the wire -- x is started ten millimetres below the
    wrap, so a recording of any length crosses it and the file has to come out
    the other side counting up rather than four billion counts to the left.
    """

    def __init__(self, geom, **kw):
        super().__init__(geom, **kw)
        self.enc_columns = 3
        self.enc_host = self.hosts[1]
        self._counts = np.array([2 ** 32 - 10.0 * COUNTS_PER_MM,
                                 400_000.0, 900_000.0])
        self.mm_per_sample = 10.0 / self.fs_hz
        self.wraps = 0

    def read(self):
        out = super().read()
        if out is None:
            self.enc = None
            return None
        n = out[0].shape[0]
        step = np.arange(1, n + 1) * COUNTS_PER_MM * self.mm_per_sample
        enc = np.column_stack([self._counts[0] + step,
                               np.full(n, self._counts[1]),
                               np.full(n, self._counts[2])])
        if enc[-1, 0] >= 2 ** 32:
            self.wraps += 1
            enc[:, 0] -= 2 ** 32 * (enc[:, 0] >= 2 ** 32)
        self._counts[0] = enc[-1, 0]
        self.enc = enc.astype(np.uint32)
        return out


class _Stage:
    """Just enough of a Stage to be anchored to: a position, and whether it
    can be believed.

    A stand-in rather than a real one because opening a Stage needs the
    Kinesis DLL, and because `trusted` is exactly what is under test -- the
    controller answers position_mm either way, and the whole point of checking
    is that a counter which has lost steps must not anchor a column.
    """

    def __init__(self, mm, trusted=True):
        self.position_mm = float(mm)
        self.position_trusted = trusted


class _Stages:
    """Just enough StageSet for a datum: named axes with a position each."""

    def __init__(self, axes):
        self.axes = axes

    @property
    def names(self):
        return list(self.axes)

    def close(self):
        pass


def _window(workdir, **over):
    ns = argparse.Namespace(
        uut=None, demo=True, replay=None, no_connect=True,
        calibration=os.path.join(workdir, "cal.json"),
        geometry=os.path.join(workdir, "geom.json"),
        machine=os.path.join(workdir, "machine.json"),
        out_dir=os.path.join(workdir, "captures"),
        screenshot=None, screenshot_tab=0, screenshot_warmup=0)
    for k, v in over.items():
        setattr(ns, k, v)
    win = gui.MainWindow(ns)
    win.tab_stages.connect_stages = lambda quiet=False: False
    return win


def _record(win, app, seconds=1.5):
    win.tab_export.chk_csv.setChecked(True)
    win.tab_export.chk_raw.setChecked(False)
    win.act_record.setChecked(True)
    path = win.session.csv_rec.path if win.session.csv_rec else None
    pump(win, app, seconds)
    win.act_record.setChecked(False)
    if win._saved_box is not None:
        win._saved_box.close()
        app.processEvents()
    return path


def test_record_writes_encoder_positions(app, workdir):
    """Counts and millimetres, on the same rows as the field."""
    print("\nrecorded position")
    win = _window(workdir)
    try:
        win._set_source(EncoderDemoSource(pgeom.Geometry.load_or_default()),
                        "demo with encoders")
        win.session.encoders = oenc.EncoderMap(
            {"x": {"column": 0, "counts_per_mm": COUNTS_PER_MM},
             "y": {"column": 1, "counts_per_mm": COUNTS_PER_MM}})
        win.session.stages = _Stages({"x": _Stage(40.0),
                                      "y": _Stage(12.5, trusted=False)})
        pump(win, app, 0.2)              # counts have to be arriving already
        wraps_before = win.session.source.wraps
        path = _record(win, app)

        check("a file was written", path and os.path.exists(path), str(path))
        names, data = read_csv(path)
        col = {n: i for i, n in enumerate(names)}
        check("the counts are in the file", "enc0_counts" in col, str(names[-6:]))
        check("a trusted axis gets an absolute position column",
              "X_mm" in col and "X_rel_mm" not in col, str(names[-6:]))
        check("an axis whose counter cannot be believed does not",
              "Y_rel_mm" in col and "Y_mm" not in col, str(names[-6:]))

        x = data[:, col["X_mm"]]
        check("every row carries a position", np.isfinite(x).all(),
              f"{int((~np.isfinite(x)).sum())} of {len(x)} blank")
        check("the recording starts where the controller said it was",
              abs(x[0] - 40.0) < 0.05, f"{x[0]:.3f} mm")
        check("the position advances with the counts, never backwards",
              np.all(np.diff(x) >= -1e-9)
              and x[-1] - x[0] > 1.0, f"{x[0]:.2f} -> {x[-1]:.2f} mm")
        # The demo x column is started ten millimetres below 2**32, so the
        # counter rolls over partway through this recording. Handled or not is
        # the difference between 15 mm of travel and a 2000 km jump.
        check("the counter really did wrap during the recording",
              win.session.source.wraps > wraps_before,
              f"{win.session.source.wraps - wraps_before} wraps")
        check("and a 32-bit wrap does not teleport the stage",
              float(np.abs(np.diff(x)).max()) < 1.0,
              f"biggest step {float(np.abs(np.diff(x)).max()):.3f} mm")
        check("a stationary axis stays where it is",
              float(np.abs(data[:, col["Y_rel_mm"]]).max()) < 1e-6)

        head = "".join(ln for ln in open(path, encoding="utf-8")
                       if ln.startswith("#"))
        check("the header says what the millimetres are measured from",
              "encoder_datum:" in head and "40.0000 mm" in head,
              head[head.find("# encoder_datum"):][:120].strip())
        check("and the Log says which columns the operator is getting",
              "recording position:" in win.log_pane.toPlainText())
    finally:
        win.close()


def test_record_without_a_datum_is_named_relative(app, workdir):
    """No stages connected is a normal bench state, not a reason to stop.

    What must not happen is a column called X_mm holding travel from wherever
    the head happened to be -- that reads as an absolute position and is not
    one.
    """
    print("\nrecorded position, no controllers")
    win = _window(workdir)
    try:
        win._set_source(EncoderDemoSource(pgeom.Geometry.load_or_default()),
                        "demo with encoders")
        win.session.encoders = oenc.EncoderMap(
            {"x": {"column": 0, "counts_per_mm": COUNTS_PER_MM}})
        win.session.stages = None
        pump(win, app, 0.5)
        path = _record(win, app)

        names, data = read_csv(path)
        col = {n: i for i, n in enumerate(names)}
        check("the column is named for what it actually is",
              "X_rel_mm" in col and "X_mm" not in col, str(names[-4:]))
        check("travel is still measured, from the first row",
              abs(data[0, col["X_rel_mm"]]) < 1e-9
              and data[-1, col["X_rel_mm"]] > 1.0,
              f"{data[0, col['X_rel_mm']]:.3f} -> "
              f"{data[-1, col['X_rel_mm']]:.3f} mm")
        check("the counts themselves are recorded regardless",
              np.isfinite(data[:, col["enc0_counts"]]).all())
    finally:
        win.close()


def test_a_probe_with_no_encoders_records_as_before(app, workdir):
    """acq1001_694 alone, a replay, the demo probe: no columns invented."""
    print("\nrecorded position, no encoders")
    win = _window(workdir)
    try:
        win._set_source(DemoSource(pgeom.Geometry.load_or_default()), "demo")
        pump(win, app, 0.3)
        path = _record(win, app, seconds=0.8)
        names, _ = read_csv(path)
        check("no encoder columns appear",
              not any("enc" in n or n.endswith("_mm") for n in names),
              str(names[-4:]))
    finally:
        win.close()


def test_the_encoder_calibration_button_follows_the_stages(app, workdir):
    """A control whose precondition arrives later must be re-decided later.

    "Calibrate encoders" is enabled only when the stages are open, and that
    was worked out once while the tab was being built -- before anything is
    connected, when session.stages is always None. Nothing re-ran it when the
    stages actually opened, so the button stayed dead for the whole session
    and pressing it did nothing at all. The encoder scale could not be
    measured, so no recording could ever carry millimetres.
    """
    print("\nencoder calibration button")
    win = _window(workdir)
    try:
        win._set_source(EncoderDemoSource(pgeom.Geometry.load_or_default()),
                        "demo with encoders")
        check("the button is dead before the stages are connected",
              not win.tab_machine.btn_vol_encoders.isEnabled())

        # What connecting does, as on_stage_action_done does it.
        win.session.stages = _Stages({"x": _Stage(40.0), "y": _Stage(12.5),
                                      "z": _Stage(80.0)})
        win.tab_stages.stages_changed.emit()
        app.processEvents()
        check("connecting the stages brings it to life",
              win.tab_machine.btn_vol_encoders.isEnabled())

        # And the label describes the stream that is actually running, which
        # is also only known once there is one.
        text = win.tab_machine.lbl_encoders.text()
        check("and the tab says which columns are arriving",
              "3 encoder column(s)" in text and "none calibrated" in text,
              text)

        win.tab_stages.on_stage_disconnect()
        app.processEvents()
        check("disconnecting takes it away again",
              not win.tab_machine.btn_vol_encoders.isEnabled())
    finally:
        win.close()
