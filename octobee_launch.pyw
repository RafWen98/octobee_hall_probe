#!/usr/bin/env pythonw
"""
octobee_launch.pyw -- what the desktop icon runs.

Why this exists rather than pointing the shortcut straight at the GUI
--------------------------------------------------------------------
A .pyw runs under pythonw.exe, which has no console. That is what you want for
a desktop application -- but it also means anything that fails before the
window opens fails *silently*: no traceback, no window, nothing. Missing PyQt6,
a moved checkout, a half-finished pip upgrade all look identical from the
desktop, which is to say they look like the icon not working.

So this wrapper catches whatever went wrong, writes it to a log next to this
file, and puts it on screen in a native message box -- native because if the
failure was PyQt6 itself, a Qt dialog is not available to report it with.

It no longer needs to pin the working directory. It used to: the GUI resolved
calibration.json, probe_geometry.json, stages.json and captures/ relative to
the CWD, so a shortcut that set the wrong one -- and shortcuts do not reliably
set any -- brought the window up on built-in defaults with no visible sign. The
package locates its own configuration now (octobee/paths.py), so every entry
point behaves the same way from anywhere, and this wrapper is back to doing the
one thing it is for: saying what went wrong.
"""

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "octobee_launch_error.log")


def report(title, body):
    """
    Native MessageBox: works even when the failure is Qt itself.

    The dialog goes FIRST, and stderr is only attempted afterwards and only if
    it exists. Under pythonw.exe -- which is the entire reason this file
    exists -- sys.stderr is None, so writing to it before showing the box
    raised AttributeError inside an exception handler and the user got no
    dialog at all: exactly the silent failure this wrapper is here to prevent.
    """
    try:
        import ctypes
        MB_ICONERROR = 0x10
        ctypes.windll.user32.MessageBoxW(None, body, title, MB_ICONERROR)
    except Exception:
        pass
    if sys.stderr is not None:
        try:
            sys.stderr.write(body + "\n")
        except Exception:
            pass


def main():
    try:
        from octobee.gui import app
    except Exception:
        tb = traceback.format_exc()
        try:
            with open(LOG, "w", encoding="utf-8") as fh:
                fh.write(tb)
        except Exception:
            pass
        report("OCTO-BEE could not start",
               "The application failed to load.\n\n"
               f"{tb.strip().splitlines()[-1]}\n\n"
               "Most often this is a missing package — from the checkout, run:\n"
               "    pip install -r requirements.txt\n\n"
               f"Full details: {LOG}")
        return 1

    try:
        app.main()
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        try:
            with open(LOG, "w", encoding="utf-8") as fh:
                fh.write(tb)
        except Exception:
            pass
        report("OCTO-BEE stopped",
               f"{tb.strip().splitlines()[-1]}\n\nFull details: {LOG}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
