"""Application entry point: argument parsing, QApplication, the window."""

import contextlib
import faulthandler
import argparse
import os
import sys
import time

from PyQt6 import QtCore, QtGui, QtWidgets

from octobee import paths
from octobee.acq import carrier as ob
from octobee.calib import convert as ocal
from octobee import machine as omach
from octobee.calib import geometry as pgeom
from octobee.gui.crash import CrashHandler
from octobee.gui.window import MainWindow

ICON_NAME = "octobee.ico"


def _apply_app_icon(app):
    """Put the probe-head icon on the window and the taskbar button.

    The AppUserModelID is the non-obvious half. Without it Windows attributes
    the window to pythonw.exe, so the taskbar shows the generic Python icon and
    groups this with every other Python process -- leaving the desktop shortcut
    as the only place the application looks like itself.
    """
    path = paths.asset(ICON_NAME)
    if not os.path.exists(path):
        return
    if sys.platform == "win32":
        try:
            import ctypes  # noqa: PLC0415  (Windows-only, cosmetic)
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "harrer.octobee.hallprobe.1")
        except Exception:
            pass                                      # cosmetic only
    app.setWindowIcon(QtGui.QIcon(path))


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uut", action="append", default=None,
                   help=f"carrier hostname, repeat for both "
                        f"(default: {' '.join(ob.DEFAULT_UUTS)})")
    p.add_argument("--demo", action="store_true",
                   help="synthetic probe, no hardware needed")
    p.add_argument("--replay", help="play back a saved .npz capture")
    p.add_argument("--geometry", default=paths.config(pgeom.CONFIG_NAME))
    p.add_argument("--calibration", default=paths.config(ocal.CONFIG_NAME))
    p.add_argument("--machine", default=paths.config(omach.CONFIG_NAME),
                   help="where the coil set, the energised coils and the "
                        "probe's placement in the machine are remembered "
                        f"(default: {omach.CONFIG_NAME})")
    p.add_argument("--out-dir", default=paths.captures_dir(),
                   help="where recordings, snapshots and exports are written "
                        f"(default: {paths.captures_dir()})")
    p.add_argument("--no-connect", action="store_true",
                   help="start disconnected. Without this the window connects "
                        "to the carriers and the stages as soon as it opens")
    p.add_argument("--profile", action="store_true",
                   help="start with the Profile tab measuring where the time "
                        "goes, and print a summary on exit")
    p.add_argument("--screenshot", help="render one frame to this PNG and exit "
                                        "(for headless checks)")
    p.add_argument("--screenshot-tab", type=int, default=0,
                   help="which tab to show in the screenshot")
    p.add_argument("--screenshot-warmup", type=float, default=3.0,
                   help="seconds of data to collect before the screenshot")
    a = p.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    crash = CrashHandler().install()
    # The other half of the same job. A Qt fatal error or an access violation
    # never becomes a Python exception -- the process simply dies, and all the
    # Windows event log records is "Qt6Core.dll, 0xc0000409", which names no
    # Python at all. faulthandler catches the abort signal itself and dumps the
    # Python stack of every thread as it goes, which is the difference between
    # a crash report that says where to look and one that says nothing.
    # Line-buffered and never closed on purpose: it has to be usable from
    # inside a signal handler, at a moment when the process is already dying.
    with contextlib.suppress(OSError):
        _fault_log = open(crash.path, "a", buffering=1, encoding="utf-8")
        _fault_log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                         f"session started =====\n")
        faulthandler.enable(file=_fault_log)
    _apply_app_icon(app)
    app.setApplicationName("OCTO-BEE Hall probe")
    win = MainWindow(a)
    crash.window = win
    win.show()

    if a.screenshot:
        def shoot():
            win.tabs.setCurrentIndex(a.screenshot_tab)
            if win.source is None:
                win.on_connect()
                deadline = time.time() + 60
                while win.source is None and time.time() < deadline:
                    app.processEvents()
                    time.sleep(0.05)
                if win.source is None:
                    print("could not connect", file=sys.stderr)
            t_end = time.time() + a.screenshot_warmup
            while time.time() < t_end:
                win.on_tick()
                app.processEvents()
                time.sleep(0.02)
            win.on_view_tick()
            win.on_slow_tick()
            app.processEvents()
            time.sleep(0.2)
            app.processEvents()
            win.grab().save(a.screenshot)
            print(f"wrote {a.screenshot}")
            # close(), not quit(): closeEvent stops the recorders and puts the
            # boxes' clkdiv back. Quitting straight out would leave both
            # carriers running at 20 kSPS for whoever connects next.
            win.close()
            app.quit()
        QtCore.QTimer.singleShot(1200, shoot)

    sys.exit(app.exec())
