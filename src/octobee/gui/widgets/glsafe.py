"""
octobee/gui/widgets/glsafe.py -- keep an OpenGL error in a 3D view from taking
the GUI down.

PyOpenGL checks glGetError after every call it makes, and glGetError returns
whatever error is pending -- including one left by a call PyOpenGL did NOT make:
Qt's own shader and buffer binds, QPainter text, the driver. The first checked
call of the next frame then raises for it. In pyqtgraph that call is
glClearColor, with perfectly good arguments:

    GLError: err = 1281, invalid value, baseOperation = glClearColor

and because the cause is outside the frame, it raised again on every frame
after -- 120 times a minute -- through the crash dialog, until the session was
unusable. Seen on Intel graphics, 2026-11-18, on a session three hours old.

The 3D views are pictures. Acquisition, recording and calibration never depend
on them. So a GL error here is cleared, the frame is skipped, the scene is
rebuilt if it keeps happening, and as a last resort the view is switched off
with a line in the log -- never an exception out of paintGL.
"""

from OpenGL import GL
from OpenGL import error as glerror
from PyQt6 import QtCore

# Where messages go. The window points this at the session log; until then,
# and in tests, print.
log_fn = print


def drain():
    """Clear every pending GL error. Returns the codes cleared."""
    seen = []
    for _ in range(32):                 # the spec allows one per error flag
        try:
            code = GL.glGetError()
        except Exception:
            break
        if code == GL.GL_NO_ERROR:
            break
        seen.append(int(code))
    return seen


class SafePaint:
    """Runs one view's paint, absorbing GL errors.

    rebuild    called (on the next event-loop turn) after `rebuild_after`
               failed frames in a row: fresh GL resources for the scene
    give_up    called after `give_up_after` failed frames in a row
    """

    def __init__(self, name, rebuild=None, give_up=None,
                 rebuild_after=3, give_up_after=30):
        self.name = name
        self.rebuild = rebuild
        self.give_up = give_up
        self.rebuild_after = rebuild_after
        self.give_up_after = give_up_after
        self.failed_in_a_row = 0
        self.failed_total = 0
        self.stale_cleared = 0
        self.disabled = False

    def paint(self, do_paint):
        if self.disabled:
            return
        # An error pending before the frame starts is not this frame's: it was
        # left by a call PyOpenGL does not check. Clear it, or the frame's
        # first call takes the blame and raises.
        stale = drain()
        if stale:
            self.stale_cleared += 1
            if self.stale_cleared == 1:
                log_fn(f"{self.name}: OpenGL had an error pending from outside "
                       f"the frame ({', '.join(hex(c) for c in stale)}) -- "
                       f"cleared; the data is unaffected")
        try:
            do_paint()
        except glerror.GLError as exc:
            drain()
            self.failed_in_a_row += 1
            self.failed_total += 1
            if self.failed_total == 1:
                log_fn(f"WARNING: {self.name}: an OpenGL error in this frame "
                       f"({getattr(exc, 'description', exc)!r} in "
                       f"{getattr(getattr(exc, 'baseOperation', None), '__name__', '?')})"
                       f" -- frame skipped; acquisition and recording are "
                       f"not affected")
            if self.failed_in_a_row == self.rebuild_after and self.rebuild:
                log_fn(f"{self.name}: rebuilding the 3D scene after "
                       f"{self.failed_in_a_row} failed frames")
                QtCore.QTimer.singleShot(0, self.rebuild)
            if self.failed_in_a_row >= self.give_up_after:
                self.disabled = True
                log_fn(f"WARNING: {self.name}: switched off after "
                       f"{self.failed_in_a_row} failed frames in a row. "
                       f"Restart the program to get it back -- nothing else "
                       f"depends on it.")
                if self.give_up:
                    QtCore.QTimer.singleShot(0, self.give_up)
        else:
            self.failed_in_a_row = 0
