"""Turning a fatal fault into a file and a dialog rather than a vanished window."""

import contextlib
import os
import sys
import time
import traceback

from PyQt6 import QtCore, QtWidgets

from octobee import paths

CRASH_LOG = "octobee_crash.log"


class CrashHandler:
    """Write unhandled exceptions down, because otherwise nobody ever sees one.

    Bare PyQt aborts the process when an exception escapes a slot -- but this
    program imports pyqtgraph, which replaces sys.excepthook with its own
    override precisely to stop that. Measured here: the same raising slot exits
    127 without pyqtgraph and 0 with it. So exceptions in slots are already
    survivable; what they are not is VISIBLE. pyqtgraph prints the traceback to
    stderr, and the desktop icon runs pythonw.exe, where stderr goes nowhere at
    all. The program carries on in a state nobody knows about.

    So this is a recorder, not a rescue: the traceback goes to a file, the Log
    tab gets a line, and one dialog says the window may now be inconsistent.
    The previous hook is still called, so pyqtgraph keeps doing whatever it
    does.

    It does NOT catch a native crash -- an access violation, or Qt calling
    qFatal(). Those never reach Python. faulthandler, enabled beside this in
    main(), is what leaves a record of those.
    """

    def __init__(self, path=None, window=None):
        self.path = path or os.path.join(
            paths.repo_root() or os.getcwd(), CRASH_LOG)
        self.window = window
        self.count = 0
        self.previous = None

    def install(self):
        self.previous = sys.excepthook
        sys.excepthook = self
        return self

    def __call__(self, exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        self.count += 1
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # The file first, and in its own try: everything after this can fail
        # (there may be no window, Qt may be half torn down) and the whole
        # point is that the traceback survives regardless.
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(f"\n===== {stamp} =====\n{text}")
        except OSError:
            pass
        with contextlib.suppress(Exception):
            if self.window is not None:
                self.window.log.log(
                    f"INTERNAL ERROR ({exc_type.__name__}: {exc}) -- written "
                    f"to {os.path.basename(self.path)}. The program is still "
                    f"running but whatever was in progress did not finish.")
        # One dialog, not one per occurrence: a fault inside a repainting
        # timer fires several times a second, and a queue of identical boxes is
        # indistinguishable from the freeze this is meant to replace.
        #
        # NON-modal, and show() rather than QMessageBox.critical(). The
        # convenience call is the blocking one: it spins a nested event loop
        # and does not return until somebody clicks. Doing that from inside an
        # excepthook stops the program dead in the middle of whatever was
        # interrupted -- measured here, with no window to parent it to, the
        # call simply never returned. A reporter that freezes the application
        # is worse than no reporter at all.
        if self.count == 1 and self.window is not None:
            with contextlib.suppress(Exception):
                box = QtWidgets.QMessageBox(
                    QtWidgets.QMessageBox.Icon.Critical, "Internal error",
                    f"{exc_type.__name__}: {exc}\n\n"
                    f"The full traceback is in {self.path}\n\n"
                    f"The program is still running, but whatever it was doing "
                    f"when this happened did not finish \u2014 save anything "
                    f"you care about and restart. Further errors will be "
                    f"logged without another dialog.",
                    parent=self.window)
                box.setModal(False)
                box.setAttribute(
                    QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
                # Held on self: a box with nothing referencing it is collected
                # before it is ever painted.
                self._box = box
                box.show()
        # Whatever was installed before us -- pyqtgraph's override, normally --
        # still gets its turn, so nothing it relies on is quietly removed.
        if self.previous is not None and self.previous is not self:
            with contextlib.suppress(Exception):
                self.previous(exc_type, exc, tb)
