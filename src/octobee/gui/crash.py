"""Turning a fatal fault into a file and a dialog rather than a vanished window."""

import contextlib
import os
import sys
import time
import traceback

from PyQt6 import QtCore, QtWidgets

from octobee import paths

CRASH_LOG = "octobee_crash.log"
# Rotated to .1 at session start when it has grown past this.
MAX_LOG_BYTES = 20 * 1024 * 1024
# How often a repeating fault may add its one-line tally to the file.
REPEAT_NOTE_S = 60.0


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
        # One entry per distinct fault, keyed on what raised and where. A bug
        # inside a 50 ms or 200 ms timer raises on every tick; with one full
        # traceback per occurrence that grew this file to 3.5 GB over two and
        # a half months, and the one traceback worth reading was somewhere in
        # the middle of it. The first occurrence gets everything; the rest are
        # counted and summarised on a slow clock.
        self._seen = {}
        self._clock = time.monotonic          # a seam for the tests

    def rotate(self):
        """Move an oversized log aside, keeping exactly one previous copy.

        Done at session start, not per write: the file is append-only and the
        dedup below bounds what one session can add, so the only thing that
        needs capping is what earlier sessions left behind.
        """
        with contextlib.suppress(OSError):
            if os.path.getsize(self.path) > MAX_LOG_BYTES:
                os.replace(self.path, self.path + ".1")

    def install(self):
        self.rotate()
        self.previous = sys.excepthook
        sys.excepthook = self
        return self

    @staticmethod
    def _site(exc_type, tb):
        """(exception type, innermost file:line): what identifies one bug."""
        where = "?"
        while tb is not None and tb.tb_next is not None:
            tb = tb.tb_next
        if tb is not None:
            where = f"{tb.tb_frame.f_code.co_filename}:{tb.tb_lineno}"
        return exc_type.__name__, where

    def _write(self, text):
        # The file first, and in its own try: everything after this can fail
        # (there may be no window, Qt may be half torn down) and the whole
        # point is that the traceback survives regardless.
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(text)
        except OSError:
            pass

    def _note_repeats(self, key, entry, final=False):
        """One line saying how many times a fault has recurred since the last."""
        n = entry["count"] - entry["noted"]
        if n <= 0:
            return
        entry["noted"] = entry["count"]
        entry["last_note"] = self._clock()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"===== {stamp} ===== {key[0]} at {key[1]} repeated {n} "
                    f"more time(s) ({entry['count']} in all)"
                    f"{' [session end]' if final else ''}\n")

    def finish(self):
        """Flush pending repeat counts at session end. Never raises."""
        with contextlib.suppress(Exception):
            for key, entry in self._seen.items():
                self._note_repeats(key, entry, final=True)

    def __call__(self, exc_type, exc, tb):
        # An exception handler that raises is the one thing worse than no
        # handler: Python then prints the second traceback and the first is
        # gone.
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        with contextlib.suppress(Exception):
            self._handle(exc_type, exc, tb)
        if self.previous is not None and self.previous is not self:
            with contextlib.suppress(Exception):
                self.previous(exc_type, exc, tb)

    def _handle(self, exc_type, exc, tb):
        self.count += 1
        key = self._site(exc_type, tb)
        entry = self._seen.get(key)
        if entry is not None:
            entry["count"] += 1
            if self._clock() - entry["last_note"] >= REPEAT_NOTE_S:
                self._note_repeats(key, entry)
            return
        self._seen[key] = {"count": 1, "noted": 1, "last_note": self._clock()}
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"\n===== {stamp} =====\n{text}")
        # Once per distinct fault, not per repeat: the Log pane keeps 4000
        # lines and a fault in a timer would flush all of them in minutes.
        # The session owns the log; the window has never had a `log` of its
        # own, which is why this line used to vanish into the suppress.
        with contextlib.suppress(Exception):
            log = getattr(getattr(self.window, "session", None), "log", None)
            if log is not None:
                log(f"INTERNAL ERROR ({exc_type.__name__}: {exc}) -- written "
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
        if len(self._seen) == 1 and self.window is not None:
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
