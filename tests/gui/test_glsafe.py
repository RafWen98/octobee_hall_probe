"""A GL error in a 3D view is absorbed, never raised out of paintGL."""

from OpenGL import error as glerror

from octobee.gui.widgets import glsafe
from tests.helpers import check


def _bad():
    raise glerror.GLError(err=1281, result=None, cArguments=(0.07, 0.08, 0.1, 1.0),
                          baseOperation=None, description=b"invalid value")


def test_a_gl_error_is_absorbed(app):
    print("\nglsafe")
    said, rebuilt, gave_up = [], [], []
    glsafe.log_fn = said.append
    try:
        safe = glsafe.SafePaint("test view", rebuild=lambda: rebuilt.append(1),
                                give_up=lambda: gave_up.append(1),
                                rebuild_after=3, give_up_after=5)
        raised = None
        try:
            for _ in range(4):
                safe.paint(_bad)
        except Exception as exc:          # what used to happen
            raised = exc
        check("a failing frame does not raise", raised is None, repr(raised))
        check("and is reported once, not once a frame",
              sum("WARNING" in m for m in said) == 1, str(said))
        app.processEvents()
        check("the scene is rebuilt after a few failures in a row",
              rebuilt == [1], str(rebuilt))

        safe.paint(lambda: None)
        check("a good frame resets the count", safe.failed_in_a_row == 0)

        for _ in range(5):
            safe.paint(_bad)
        app.processEvents()
        check("a view that keeps failing is switched off, with a warning",
              safe.disabled and gave_up == [1]
              and any("switched off" in m for m in said), str(said[-1:]))
        ran = []
        safe.paint(lambda: ran.append(1))
        check("and is not painted any more", ran == [])

        other = glsafe.SafePaint("other")
        other.paint(lambda: None)
        check("a healthy view paints normally",
              other.failed_total == 0 and not other.disabled)
    finally:
        glsafe.log_fn = print
