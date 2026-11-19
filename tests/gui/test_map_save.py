"""Writing a finished volume map must not stop the GUI thread draining the stream.

On 2026-11-19 a 567k-row map took ~5 s to assemble and compress on the GUI
thread; the carriers' queues held 1.3 s, and 648 blocks (6.6 s of stream) were
dropped from the recording running beside it.
"""

import time

from PyQt6 import QtCore, QtWidgets

from octobee.gui import workers
from octobee.motion import encoder as oenc
from tests.gui.test_status import _window
from tests.helpers import check, pump


class _SlowLog:
    """As much of a SweepLog as the end of a map touches, slow to save."""

    def __init__(self, fail=False):
        self.n_rows = 500_000
        self.encoders = oenc.EncoderMap()
        self.parts_written = []
        self.fail = fail
        self.saved_on_gui_thread = None
        self.dropped = False

    def finish_parts(self):
        pass

    def checkpoint(self, force=False):
        pass

    def drop_parts(self):
        self.dropped = True

    def save(self, path, log_hz, pose, sync_note=""):
        self.saved_on_gui_thread = (QtCore.QThread.currentThread()
                                    is QtCore.QCoreApplication.instance()
                                    .thread())
        time.sleep(1.5)
        if self.fail:
            raise OSError("disk full")
        return path


def _finish(win, log):
    m = win.tab_machine
    win.session.volume_log = log
    win.session.volume_runner = None
    m._run_hz = float(win.session.out_rate)
    m._run_pose = None
    m._run_path = win.session.out_dir + "/volume_test.npz"
    m.on_volume_done(155, "")
    return m


def test_the_map_is_written_off_the_gui_thread(app, workdir):
    print("\nwriting a finished map")
    win = _window(workdir, "mapsave")
    try:
        finished = []
        win.tab_machine.map_finished.connect(
            lambda p, e: finished.append((p, e)))
        log = _SlowLog()
        t0 = time.monotonic()
        m = _finish(win, log)
        returned = time.monotonic() - t0
        check("the handler returns at once instead of waiting for the save",
              returned < 0.5, f"{returned:.2f} s")
        check("a map still being written refuses a new start",
              m.saving_map())
        # The GUI thread keeps turning while it is written: this pump is
        # the tick that drains the carriers' queues.
        ticks, end = 0, time.monotonic() + 1.0
        while time.monotonic() < end:
            pump(win, app, 0.05)
            ticks += 1
        check("the event loop keeps running during the save", ticks >= 10,
              f"{ticks} ticks in 1 s")
        deadline = time.monotonic() + 10
        while not finished and time.monotonic() < deadline:
            pump(win, app, 0.05)
        check("the save ran on a thread of its own",
              log.saved_on_gui_thread is False)
        check("and the map is reported written once it is",
              finished and finished[0][0].endswith("volume_test.npz")
              and log.dropped, str(finished))
        check("and a new map can be started again", not m.saving_map())
    finally:
        win.close()


def test_a_failed_save_keeps_the_map_in_memory(app, workdir, monkeypatch):
    print("\na map that cannot be written")
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: None))
    win = _window(workdir, "mapsavefail")
    try:
        finished = []
        win.tab_machine.map_finished.connect(
            lambda p, e: finished.append((p, e)))
        log = _SlowLog(fail=True)
        _finish(win, log)
        deadline = time.monotonic() + 10
        while not finished and time.monotonic() < deadline:
            pump(win, app, 0.05)
        check("the failure is reported", finished and finished[0][0] == ""
              and "disk full" in finished[0][1], str(finished))
        check("and the rows stay reachable to save again",
              win.session.volume_log is log and not log.dropped)
    finally:
        win.close()


def test_the_stream_queue_holds_seconds_not_one(app):
    held_s = workers.STREAM_QUEUE_BLOCKS * 2048 / 200_000.0
    check("each carrier's queue holds ~10 s of stream at 200 kHz",
          held_s >= 8.0, f"{held_s:.1f} s")
