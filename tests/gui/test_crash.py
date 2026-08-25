"""A fatal fault becomes a file and a dialog, not a vanished window."""

import os
import subprocess
import sys
import time

from PyQt6 import QtWidgets

from octobee.gui.crash import CrashHandler

from tests.helpers import (
    check,
)


# Run in a CHILD process: half of what is under test is what happens when the
# process dies, and proving that in-process would take the test runner with it.
_CRASH_PROBE = """
import faulthandler, os, sys
from PyQt6 import QtCore, QtWidgets
from octobee.gui import window as gui
from octobee.gui.crash import CrashHandler

app = QtWidgets.QApplication([])
if {install!r}:
    handler = CrashHandler(path={log!r}).install()
    faulthandler.enable(file=open({log!r}, "a", buffering=1, encoding="utf-8"))

def boom():
    if {native!r}:
        os.abort()                 # stands in for Qt's qFatal
    raise RuntimeError("deliberate fault in a slot")

QtCore.QTimer.singleShot(20, boom)
QtCore.QTimer.singleShot(800, app.quit)
app.exec()
print("SURVIVED")
"""


def test_crash_handler(app, workdir):
    """Make the program's failures leave a record. Two kinds, two mechanisms.

    A Python exception in a slot does NOT kill this program: pyqtgraph replaces
    sys.excepthook on import specifically to stop PyQt aborting. It prints the
    traceback to stderr instead -- and the desktop icon runs pythonw.exe, where
    stderr goes nowhere. So the failure is survivable and completely invisible,
    which is how a window ends up running in a state nobody knows about.

    A NATIVE fatal error -- an access violation, or Qt calling qFatal() -- is
    the opposite: it kills the process at once and never becomes a Python
    exception at all. All it leaves is a Windows event log line naming
    Qt6Core.dll, which says nothing about which line of Python was running.
    faulthandler is what turns that into a stack trace.
    """
    print("\ncrash handler")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")

    def run(install, log, native=False):
        src = _CRASH_PROBE.format(install=install, log=log,
                                  native=native)
        # check=False on purpose: a non-zero exit IS one of the results.
        # stdin=DEVNULL because pytest's capture leaves this process's
        # stdin without a real OS handle, and the child cannot inherit
        # what does not exist. It never reads stdin either way.
        return subprocess.run([sys.executable, "-c", src], env=env, check=False,
                              stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=180)

    # ---- a Python exception in a slot ----
    bare_log = os.path.join(workdir, "bare.log")
    bare = run(False, bare_log)
    check("a slot exception does not kill the program (pyqtgraph sees to that)",
          bare.returncode == 0 and "SURVIVED" in bare.stdout,
          f"exit {bare.returncode}")
    check("but nothing is written down, which is the actual problem",
          not os.path.exists(bare_log),
          "under pythonw.exe the traceback goes to a stderr that does not exist")

    log = os.path.join(workdir, "crash_probe.log")
    caught = run(True, log)
    check("with the handler, the program still survives",
          caught.returncode == 0 and "SURVIVED" in caught.stdout,
          f"exit {caught.returncode}")
    check("and now the traceback is on disk",
          os.path.exists(log) and "deliberate fault in a slot" in
          open(log, encoding="utf-8").read())
    check("the log names the line that raised, not just the message",
          "boom" in open(log, encoding="utf-8").read())

    # ---- a native abort, which Python never sees ----
    native_log = os.path.join(workdir, "native.log")
    native = run(True, native_log, native=True)
    check("a native abort does kill the program",
          native.returncode != 0 and "SURVIVED" not in native.stdout,
          f"exit {native.returncode} -- nothing can catch this one")
    dumped = open(native_log, encoding="utf-8").read() if os.path.exists(
        native_log) else ""
    check("faulthandler still leaves a Python stack for it",
          "Fatal Python error" in dumped and "boom" in dumped,
          "which is the difference between 'Qt6Core.dll 0xc0000409' and a line "
          "number")

    # Repeated faults must not queue a dialog each: a fault inside a repainting
    # timer fires several times a second.
    #
    # Counted by looking at the box the handler keeps, NOT by monkeypatching
    # QMessageBox. Two things about that are worth writing down, because both
    # cost an afternoon here:
    #
    #   * the handler must not call QMessageBox.critical(), which blocks on a
    #     click that never comes -- from inside an excepthook that freezes the
    #     program, which is the failure the reporter exists to replace;
    #   * patching QMessageBox.show() to count calls ABORTS the interpreter
    #     (exit 127). It is a C++ virtual, and replacing it out from under Qt
    #     is not a thing sip will survive. Assert on the handler's own state.
    #
    # A parent window is supplied because that is the only state the running
    # program is ever in; with none, the handler logs and shows nothing rather
    # than parenting a box to nowhere.
    parent = QtWidgets.QMainWindow()
    handler = CrashHandler(path=os.path.join(workdir, "twice.log"),
                               window=parent)
    boxes = []
    t0 = time.time()
    for _ in range(3):
        try:
            raise ValueError("again")
        except ValueError:
            handler(*sys.exc_info())
        boxes.append(getattr(handler, "_box", None))
    elapsed = time.time() - t0
    check("reporting never blocks the thread it interrupted", elapsed < 5.0,
          f"{elapsed:.3f} s for three faults -- the blocking dialog never "
          f"returned at all")
    check("one dialog is raised, and the later faults reuse it",
          boxes[0] is not None and boxes[1] is boxes[0]
          and boxes[2] is boxes[0],
          "three boxes for three faults reads as the freeze it replaces")
    parent.close()
    check("repeated faults are all logged", handler.count == 3,
          f"count {handler.count}")
    check("every one of them reaches the log",
          open(handler.path, encoding="utf-8").read().count("again") >= 3)
