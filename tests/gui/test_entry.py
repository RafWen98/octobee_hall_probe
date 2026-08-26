"""The application entry point, which nothing else exercises.

octobee/gui/app.py had 0% coverage, and that is where a refactor break went
unnoticed: --screenshot read win.source after `source` had moved onto the
Session. PyQt6 prints an unhandled exception in a slot to stderr and carries
on -- it does not propagate, so no test failed -- and under pythonw, which is
what the desktop shortcut runs, stderr is None and it is invisible entirely.

These call the entry point for real rather than trusting that it works.
"""

import os
import pathlib
import subprocess
import sys

from tests.helpers import (
    check,
)


def test_app_is_runnable_as_a_module():
    """`python -m octobee.gui.app --help` must do something.

    Every other entry point in the package has a __main__ guard. app.py did
    not, so running it as a module imported it, ran nothing and exited 0 --
    indistinguishable from the application starting and closing at once.
    """
    print("\napp module entry")
    r = subprocess.run([sys.executable, "-m", "octobee.gui.app", "--help"],
                       capture_output=True, text=True, timeout=120,
                       check=False, stdin=subprocess.DEVNULL,
                       env=dict(os.environ, QT_QPA_PLATFORM="offscreen"))
    check("python -m octobee.gui.app --help exits cleanly", r.returncode == 0,
          f"rc={r.returncode} {r.stderr.strip()[:120]}")
    check("it prints its usage", "--screenshot" in r.stdout,
          r.stdout[:160] or "(no stdout)")


def test_screenshot_renders_a_frame(workdir):
    """--screenshot must produce a PNG, not hang.

    Run in a child process: it builds its own QApplication, and the suite
    already has one.
    """
    print("\nheadless screenshot")
    out = pathlib.Path(workdir) / "shot.png"
    r = subprocess.run(
        [sys.executable, "-m", "octobee.gui.app", "--demo", "--no-connect",
         "--screenshot", str(out), "--screenshot-warmup", "0.5"],
        capture_output=True, text=True, timeout=300,
        check=False, stdin=subprocess.DEVNULL,
        env=dict(os.environ, QT_QPA_PLATFORM="offscreen"))

    check("the screenshot run exits cleanly", r.returncode == 0,
          f"rc={r.returncode}")
    # The specific regression: an AttributeError inside the QTimer callback.
    check("no attribute went missing while it ran",
          "AttributeError" not in r.stderr,
          [ln for ln in r.stderr.splitlines() if "Error" in ln][:2])
    check("a PNG was written", out.exists() and out.stat().st_size > 0,
          f"{out} {'exists' if out.exists() else 'missing'}")
    if out.exists():
        # PNG magic, so a zero-byte or truncated file does not pass
        check("the file really is a PNG",
              out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n")
