"""The window around the tabs: recording, connection, closing, and what it refuses.

Everything here is driven headless against the synthetic probe. Nothing
touches a carrier or a USB device: workers that would are replaced by small
QThreads that sleep, which is all the behaviour under test depends on.
"""

import argparse
import os
import time

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from octobee.calib import convert as ocal
from octobee.gui import session as gsession
from octobee.gui import window as gui
from octobee.gui.sources import DemoSource
from octobee.gui.tabs import stages as stages_mod

from tests.helpers import check, pump


def _args(workdir, name, **over):
    ns = argparse.Namespace(
        uut=None, demo=True, replay=None, no_connect=True,
        stages=os.path.join(workdir, f"{name}_stages.json"),
        geometry=os.path.join(workdir, f"{name}_geom.json"),
        calibration=os.path.join(workdir, f"{name}_cal.json"),
        machine=os.path.join(workdir, f"{name}_machine.json"),
        out_dir=os.path.join(workdir, f"{name}_caps"),
        screenshot=None, screenshot_tab=0, screenshot_warmup=0)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _window(workdir, name, **over):
    return gui.MainWindow(_args(workdir, name, **over))


def _log(win):
    return win.log_pane.toPlainText()


class _Sleeper(QtCore.QThread):
    """A thread that is 'running' for a while and then ends."""

    done = QtCore.pyqtSignal(object, object, str)

    def __init__(self, seconds, result=(None, {}, "")):
        super().__init__()
        self.seconds = seconds
        self.result = result

    def run(self):
        time.sleep(self.seconds)
        self.done.emit(*self.result)


def _start_csv(win):
    win.tab_export.chk_csv.setChecked(True)
    win.tab_export.chk_raw.setChecked(False)
    win.act_record.setChecked(True)


# ---------------------------------------------------------------- item 1

# (the crash handler is covered in test_crash.py)


# ---------------------------------------------------------------- item 2

def test_output_rate_change_rolls_the_csv(app, workdir):
    """A new output rate must not be written into a file opened at the old one."""
    print("\noutput rate during a recording")
    win = _window(workdir, "rate")
    try:
        _start_csv(win)
        pump(win, app, 0.5)
        first = win.session.csv_rec
        check("recording at 500 Hz", first is not None and first.fs_out == 500.0)
        win.cmb_out.setCurrentIndex(0)          # 100 Hz
        second = win.session.csv_rec
        check("the CSV was rolled over to a new file",
              second is not None and second is not first
              and second.path != first.path)
        check("whose header describes the new rate and decimation",
              second.fs_out == 100.0
              and second.samples_per_row == win.session.decim(),
              f"{second.fs_out} Hz, {second.samples_per_row} per row")
        pump(win, app, 0.3)
        with open(second.path, encoding="utf-8") as fh:
            head = fh.read(4000)
        check("and names the file it continues, and why",
              os.path.basename(first.path) in head and "output rate" in head)
        check("rows keep arriving in the new file", second.n_rows > 0)

        # A volume sweep running: refuse, put the combo back, say why.
        win.session.volume_runner = object()
        win.cmb_out.setCurrentIndex(4)          # 2000 Hz
        check("a rate change during a volume map is refused",
              win.session.out_rate == 100.0 and win.session.csv_rec is second)
        check("the combo is put back to what is really running",
              win.cmb_out.currentData() == 100.0)
        check("and the log says why", "refused" in _log(win)
              and "volume map" in _log(win))
        win.session.volume_runner = None
    finally:
        win.close()
        app.processEvents()


# ---------------------------------------------------------------- item 3

def test_a_failed_stream_stops_the_recording(app, workdir):
    """An errored or silent stream must end the recording, once and loudly."""
    print("\nfailed stream during a recording")
    win = _window(workdir, "stream")
    try:
        _start_csv(win)
        pump(win, app, 0.3)
        src = win.session.source
        src.error = "acq1001_694: connection reset"
        win.on_tick()
        check("a source error stops the recording through the normal path",
              win.session.csv_rec is None and not win.act_record.isChecked())
        check("and raises a visible, non-blocking alarm",
              "ALARM" in win.status.currentMessage()
              and "ALARM" in _log(win))
        check("the error is logged once", _log(win).count("stream error") == 1)

        # The same error arriving every tick must not flood the log.
        for _ in range(50):
            src.error = "acq1001_694: connection reset"
            win.on_tick()
        check("a repeating error is still logged only once per connection",
              _log(win).count("stream error") == 1,
              f"{_log(win).count('stream error')} lines for 51 ticks")

        # Silence: no rows for longer than the limit while recording.
        win.status.clearMessage()
        _start_csv(win)
        src.read = lambda: None                  # the stream has gone quiet
        win.on_tick()
        check("a recording survives a short gap", win.session.csv_rec is not None)
        # The window itself busy for longer than the limit is not the stream
        # stalling: one late tick may add no more than STALL_TICK_S.
        win._last_tick_t -= win.STALL_S + 1
        win.on_tick()
        check("and survives this window being busy for longer than the limit",
              win.session.csv_rec is not None, f"{win._quiet_s:.1f} s counted")
        # Quiet tick after quiet tick is, and those add up.
        win._quiet_s = win.STALL_S
        win.on_tick()
        check("but is stopped after the stall limit, with an alarm",
              win.session.csv_rec is None and "no data" in _log(win)
              and "ALARM" in win.status.currentMessage())
    finally:
        win.close()
        app.processEvents()


# ---------------------------------------------------------------- item 4

def test_close_asks_before_it_stops_anything(app, workdir, monkeypatch):
    """Cancel on the 'still moving' question must leave the recording alone."""
    print("\nclosing")
    win = _window(workdir, "close")
    try:
        _start_csv(win)
        pump(win, app, 0.2)
        monkeypatch.setattr(win.motion, "busy", lambda: True)
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "warning",
            staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Cancel))
        ev = QtGui.QCloseEvent()
        win.closeEvent(ev)
        check("Cancel refuses the close", not ev.isAccepted())
        check("and the recording is still running",
              win.session.csv_rec is not None and win.act_record.isChecked())
        check("the window is not left in its closing state", not win._closing)
    finally:
        monkeypatch.undo()
        win.close()
        app.processEvents()


def test_close_waits_for_connect_and_snapshot(app, workdir):
    """Closing mid-connect or mid-snapshot must not strand either."""
    print("\nclosing during a connect")
    win = _window(workdir, "closeconn")
    try:
        demo = DemoSource(win.session.geom)
        win._connecting = True
        win._connect_worker = _Sleeper(0.3, (demo, {}, ""))
        win._connect_worker.done.connect(win.on_connected)
        win._connect_worker.start()
        win.closeEvent(QtGui.QCloseEvent())
        check("the connect result was received before closing",
              not win._connecting)
        check("and the source it delivered was shut down again",
              win.session.source is None)
    finally:
        win.close()
        app.processEvents()

    print("\nclosing during a snapshot")
    win = _window(workdir, "closesnap")
    try:
        snap = _Sleeper(0.4)
        win._snap_worker = snap
        snap.start()
        win.closeEvent(QtGui.QCloseEvent())
        check("a running snapshot thread is waited for, not destroyed",
              snap.isFinished())
    finally:
        win.close()
        app.processEvents()


# ---------------------------------------------------------------- item 5

def test_stage_connect_survives_a_refusal(app, workdir, monkeypatch):
    """A refused connect must give the Connect button back."""
    print("\nstage connect after an e-stop")
    win = _window(workdir, "estop")
    try:
        tab = win.tab_stages
        tab.stage_combos = {"serial": object()}
        monkeypatch.setattr(tab, "_axis_map_from_table", lambda: {"x": "serial"})
        monkeypatch.setattr(stages_mod, "_stage_set_for", lambda m: object())
        win.motion.trigger("test e-stop")
        tab.on_stage_connect()
        check("a refused connect leaves nothing pending",
              tab._stage_pending is None)
        check("and Connect is enabled again", tab.btn_stage_connect.isEnabled())
        check("stage_action reports the refusal",
              tab.stage_action("x", lambda emit: None) is False)
    finally:
        monkeypatch.undo()
        win.close()
        app.processEvents()


# ---------------------------------------------------------------- item 6

def test_toolbar_is_gated_against_motion(app, workdir, monkeypatch):
    print("\ntoolbar while something is moving")
    win = _window(workdir, "gate")
    try:
        # --- a job is running, nothing connected
        win.session.source = None
        win.act_connect.setEnabled(False)       # as a live connection leaves it
        busy = _Sleeper(0.5)
        busy.start()
        win.motion.register(busy)
        started = []
        monkeypatch.setattr(
            gui, "ConnectWorker", lambda *a, **k: started.append(a))
        win.on_connect()
        check("Connect is refused during a job, and says what is running",
              not started and "connect refused" in _log(win))
        app.processEvents()
        check("Connect is shown as unavailable", not win.act_connect.isEnabled())
        win.motion.retire(busy)
        busy.wait(2000)
        app.processEvents()
        check("Connect comes back by itself when the job ends",
              win.act_connect.isEnabled())

        # --- Disconnect with a source, during a job
        win._set_source(DemoSource(win.session.geom), "demo")
        busy = _Sleeper(0.5)
        busy.start()
        win.motion.register(busy)
        app.processEvents()
        win.on_disconnect()
        check("Disconnect is refused during a job",
              win.session.source is not None and "disconnect refused" in _log(win))
        check("and the button is disabled", not win.act_disconnect.isEnabled())
        # closing the stages under a job
        closed = []
        win.session.stages = type("S", (), {"close": lambda self: closed.append(1)})()
        win.tab_stages.on_stage_disconnect()
        check("the stages are not closed under a running job",
              not closed and win.session.stages is not None)
        win.session.stages = None
        win.motion.retire(busy)
        busy.wait(2000)

        # --- Snapshot during a volume map or a collection
        win.session.volume_runner = object()
        win.on_snapshot()
        check("Snapshot is refused during a volume map",
              win.session.source is not None and "snapshot refused" in _log(win))
        win.session.volume_runner = None
        win.session.collecting = {"what": "tare"}
        win.on_snapshot()
        check("and during a tare",
              win.session.source is not None
              and _log(win).count("snapshot refused") == 2)
        win.session.collecting = None

        # --- the partial map is saved before any dialog
        order = []

        class FM:
            def __init__(self):
                self.stats = [{"sem_ut": 1.0}]
                self.meta = {"n_requested": 4}

            def __len__(self):
                return 1

            def save(self, path):
                order.append("save")
                raise OSError("disk full")

        monkeypatch.setattr(
            QtWidgets.QMessageBox, "warning",
            staticmethod(lambda *a, **k: order.append("dialog")))
        busy = _Sleeper(0.0)
        win.tab_stages._scan_worker = busy
        busy.start()
        win.motion.register(busy)
        win.tab_stages.on_scan_done(FM(), "the stage stalled")
        check("a failed scan's partial map is saved before the warning",
              order == ["save", "dialog"], str(order))
        check("a save that raises is logged, not fatal",
              "COULD NOT SAVE" in _log(win))
    finally:
        monkeypatch.undo()
        win.close()
        app.processEvents()


def test_a_started_job_stays_registered(app, workdir):
    """The job must still count as busy after the change signal has gone round.

    With the stages connected, the Stages tab's own handler of motion.changed
    asks motion.busy(); busy() drops a registered worker that is not running
    yet, so registering before start() lost the job -- the e-stop and closeEvent
    then did not know about it.
    """
    print("\nregistering a job")
    win = _window(workdir, "reg")
    try:
        win.session.stages = object()           # sync_controls only asks "is there one"
        win.tab_stages.stage_action("sleep", lambda emit: time.sleep(0.4))
        check("a job started by stage_action is counted as busy",
              win.motion.busy())
        deadline = time.time() + 5
        while win.motion.busy() and time.time() < deadline:
            app.processEvents()
            time.sleep(0.02)
        win.session.stages = None
    finally:
        win.close()
        app.processEvents()


# ---------------------------------------------------------------- item 7

class _Rec:
    def __init__(self, path, fail=False):
        self.path = path
        self.fail = fail
        self.closed = False
        self.n_rows = 3
        self.enc_columns = 0
        self.n_unpaired = 0
        self.size_bytes = 0

    def close(self):
        self.closed = True
        if self.fail:
            raise OSError("no space left on device")
        return self.path


def test_recording_stop_is_exception_safe(app, workdir, monkeypatch):
    print("\nstopping a recording that fails to close")
    win = _window(workdir, "safe")
    try:
        csv, raw = _Rec("a.csv", fail=True), _Rec("a.bin")
        win.session.csv_rec, win.session.raw_rec = csv, raw
        win._closing = True                     # no box
        win.on_record(False)
        check("the raw recorder is closed although the CSV close raised",
              raw.closed)
        check("both are cleared from the session",
              win.session.csv_rec is None and win.session.raw_rec is None)
        check("the failure is in the log", "could not close the CSV" in _log(win))
        win._closing = False

        # a rollover whose new file cannot be opened
        _start_csv(win)
        pump(win, app, 0.2)
        monkeypatch.setattr(
            win, "_new_csv_recorder",
            lambda *a, **k: (_ for _ in ()).throw(OSError("read-only volume")))
        win._roll_over_csv("the zero point")
        win._update_rec_label()
        check("a failed rollover clears csv_rec instead of pointing at a closed file",
              win.session.csv_rec is None)
        check("the status no longer claims REC", win.lbl_rec.text() == "")
        check("the Record button is released", not win.act_record.isChecked())
        check("and the failure is announced", "ALARM" in _log(win))
        win.on_tick()                           # must not raise on a dead recorder
    finally:
        monkeypatch.undo()
        win.close()
        app.processEvents()


# ---------------------------------------------------------------- item 8

def test_calibration_is_frozen_during_a_volume_map(app, workdir):
    print("\ncalibration changes during a volume map")
    win = _window(workdir, "volcal")
    try:
        cal = win.session.cal
        tab = win.tab_calib
        win.session.volume_runner = object()
        snapshot = (cal.zero_mt.copy(), cal.gain_corr.copy(),
                    cal.ranges_mt.copy(), cal.subtract_vcm)
        emitted = []
        tab.calibration_changed.connect(emitted.append)
        cal.zero_mt[:] = 0.5
        tab.on_clear_tare()
        check("clearing the zero is refused", np.all(cal.zero_mt == 0.5))
        tab.start_collect("tare", 1.0)
        check("a tare is not started", win.session.collecting is None)
        win.session.magnet_peaks = np.ones(16)
        tab.on_apply_gain()
        tab.on_clear_gain()
        check("gain trim changes are refused",
              np.all(cal.gain_corr == snapshot[1]))
        other = 50.0 if cal.ranges_mt[2] != 50.0 else 20.0
        before = float(cal.ranges_mt[2])
        tab.on_range_changed(2, other)
        check("a range change is refused",
              float(cal.ranges_mt[2]) == before)
        check("and the Sensors table is put back",
              win.table._combos[2].currentData() == before)
        tab.chk_vcm.setChecked(not cal.subtract_vcm)
        check("VCM subtraction cannot be toggled",
              cal.subtract_vcm == snapshot[3]
              and tab.chk_vcm.isChecked() == snapshot[3])
        check("nothing announced a calibration change", not emitted,
              str(emitted))
        check("and each refusal said a volume map is running",
              _log(win).count("volume map is running") >= 6)
        win.session.volume_runner = None
        tab.on_clear_tare()
        check("the same actions work again afterwards",
              np.all(cal.zero_mt == 0.0) and emitted == ["the zero point"])
    finally:
        win.close()
        app.processEvents()


# ---------------------------------------------------------------- item 9

def test_encoder_mismatch_resynchronises(app, workdir):
    print("\nencoder rows that do not match the field rows")
    win = _window(workdir, "enc")
    try:
        _start_csv(win)
        src = win.session.source
        src.enc_columns = 3
        win.session.new_decimator()             # picks the stream up
        check("encoder stream in place", win.session.enc_stream is not None)
        pump(win, app, 0.2)
        real_read = src.read
        first_csv = win.session.csv_rec
        before = (win.session.decimator, win.session.enc_decimator)

        def bad_read():
            blocks = real_read()
            if blocks:
                src.enc = np.zeros((1, 3), dtype=np.uint32)
            return blocks
        src.read = bad_read
        deadline = time.time() + 3
        while win.session.decimator is before[0] and time.time() < deadline:
            win.on_tick()
            time.sleep(0.02)
        check("both decimators are rebuilt together",
              win.session.decimator is not before[0]
              and win.session.enc_decimator is not before[1])
        check("an open CSV is rolled over, its datum belonged to the old stream",
              win.session.csv_rec is not first_csv)
        win.on_tick()
        check("the mismatch is logged once per connection",
              _log(win).count("restarting the encoder") == 1)
    finally:
        win.close()
        app.processEvents()


# ---------------------------------------------------------------- item 10

def test_reload_geometry_keeps_the_old_one_on_a_typo(app, workdir):
    print("\nreloading a broken geometry file")
    win = _window(workdir, "geom")
    try:
        old = win.session.geom
        with open(win.session.args.geometry, "w", encoding="utf-8") as fh:
            fh.write("{ this is not json")
        win.tab_calib.on_reload_geometry()
        check("the geometry in use is kept", win.session.geom is old)
        check("the log does not claim a reload",
              "geometry reloaded" not in _log(win)
              and "NOT reloaded" in _log(win))
        old.save(win.session.args.geometry)
        win.tab_calib.on_reload_geometry()
        check("a good file reloads", "geometry reloaded" in _log(win))
    finally:
        win.close()
        app.processEvents()


# ---------------------------------------------------------------- item 11

def test_auto_exclude_keeps_manual_entries(app, workdir, monkeypatch):
    print("\nautomatic dead-sensor exclusion")
    path = os.path.join(workdir, "dead_cal.json")
    cal = ocal.Calibration()
    cal.dead = {"S3", "S16.Bx"}
    cal.save(path)
    win = _window(workdir, "dead", calibration=path)
    try:
        suggested = [set()]
        monkeypatch.setattr(ocal, "suggest_dead", lambda rows: set(suggested[0]))
        monkeypatch.setattr(ocal, "health_verdict", lambda rows: {})
        cal = win.session.cal
        check("hand-entered entries were loaded", cal.dead == {"S3", "S16.Bx"})
        win._update_auto_dead([])
        check("a scan that finds nothing keeps them",
              cal.dead == {"S3", "S16.Bx"}, str(cal.dead))
        suggested[0] = {"S7"}
        win._update_auto_dead([])
        check("one scan of a railed sensor does not exclude it",
              "S7" not in cal.dead)
        win._update_auto_dead([])
        check("two in a row do, beside the manual ones",
              cal.dead == {"S3", "S16.Bx", "S7"}, str(cal.dead))
        suggested[0] = set()
        win._update_auto_dead([])
        check("one healthy scan does not re-include it yet", "S7" in cal.dead)
        win._update_auto_dead([])
        check("two do, and the manual entries are untouched",
              cal.dead == {"S3", "S16.Bx"}, str(cal.dead))
    finally:
        monkeypatch.undo()
        win.close()
        app.processEvents()


# ---------------------------------------------------------------- item 12

def test_diagnostics_after_disconnect(app, workdir):
    print("\ndiagnostics with nothing connected")
    win = _window(workdir, "diag")
    try:
        pump(win, app, 0.5)
        check("history exists while connected", win.raw_hist is not None)
        win.on_disconnect()
        check("disconnect clears the raw history", win.raw_hist is None)
        win.raw_hist = [[np.zeros((4, 32))]]    # even if something left it
        win.tab_health.analyse()                # must not raise
        check("analyse with no source says to connect",
              "connect first" in win.tab_health.health_text.toPlainText())
    finally:
        win.close()
        app.processEvents()


# ---------------------------------------------------------------- item 13

def test_small_fixes(app, workdir, monkeypatch):
    print("\nsmall fixes")
    # calibration reported as found even when the geometry file is bad
    cal_path = os.path.join(workdir, "ok_cal.json")
    ocal.Calibration().save(cal_path)
    geom_path = os.path.join(workdir, "bad_geom.json")
    with open(geom_path, "w", encoding="utf-8") as fh:
        fh.write("not json")
    sess = gsession.Session(_args(workdir, "s", calibration=cal_path,
                                  geometry=geom_path))
    check("a bad geometry file is still reported as an error",
          any("geometry" in e or "bad_geom" in e for e in sess.config_errors))
    check("but does not make the calibration look unloaded", sess.cal_from_file)

    win = _window(workdir, "small", calibration=cal_path)
    try:
        # a collection started on the old stream is abandoned
        win.session.collecting = {"what": "tare", "blocks": [], "n": 0,
                                  "need": 10**9, "decim": 1}
        win.on_disconnect()
        check("a tare in progress is abandoned by Disconnect",
              win.session.collecting is None and "abandoned" in _log(win))
        win.session.collecting = {"what": "tare", "blocks": [], "n": 0,
                                  "need": 10**9, "decim": 1}
        win._set_source(DemoSource(win.session.geom), "demo")
        check("and by a new stream arriving", win.session.collecting is None)

        # loading a calibration rolls the CSV over once, not twice
        _start_csv(win)
        pump(win, app, 0.2)
        rolls = []
        real = win._roll_over_csv
        monkeypatch.setattr(
            win, "_roll_over_csv", lambda what, **k: (rolls.append(what),
                                                      real(what, **k))[1])
        monkeypatch.setattr(
            QtWidgets.QFileDialog, "getOpenFileName",
            staticmethod(lambda *a, **k: (cal_path, "")))
        win.tab_calib.chk_vcm.blockSignals(True)
        win.tab_calib.chk_vcm.setChecked(False)  # differs from the file's True
        win.tab_calib.chk_vcm.blockSignals(False)
        win.tab_calib.on_load_cal()
        check("loading a calibration rolls the CSV over exactly once",
              len(rolls) == 1, str(rolls))
    finally:
        monkeypatch.undo()
        win.close()
        app.processEvents()
