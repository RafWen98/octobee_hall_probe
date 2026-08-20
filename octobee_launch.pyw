#!/usr/bin/env pythonw
"""
octobee_launch.pyw -- what the desktop icon runs.

Why this exists rather than pointing the shortcut straight at octobee_gui.py
---------------------------------------------------------------------------
A .pyw runs under pythonw.exe, which has no console. That is what you want for
a desktop application -- but it also means anything that fails before the
window opens fails *silently*: no traceback, no window, nothing. Missing PyQt6,
a moved checkout, a half-finished pip upgrade all look identical from the
desktop, which is to say they look like the icon not working.

So this wrapper catches whatever went wrong, writes it to a log next to this
file, and puts it on screen in a native message box -- native because if the
failure was PyQt6 itself, a Qt dialog is not available to report it with.

It also pins the working directory to the checkout. Shortcuts do not reliably
set one, and the GUI resolves calibration.json, probe_geometry.json, stages.json
and captures/ relative to the CWD: launched from the wrong place it would come
up with built-in defaults and no visible sign that it had.
"""

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "octobee_launch_error.log")


def report(title, body):
    """Native MessageBox: works even when the failure is Qt itself."""
    sys.stderr.write(body + "\n")
    try:
        import ctypes
        MB_ICONERROR = 0x10
        ctypes.windll.user32.MessageBoxW(None, body, title, MB_ICONERROR)
    except Exception:
        pass


def main():
    os.chdir(HERE)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    try:
        import octobee_gui
    except Exception:
        tb = traceback.format_exc()
        try:
            with open(LOG, "w") as fh:
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
        octobee_gui.main()
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        try:
            with open(LOG, "w") as fh:
                fh.write(tb)
        except Exception:
            pass
        report("OCTO-BEE stopped",
               f"{tb.strip().splitlines()[-1]}\n\nFull details: {LOG}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
