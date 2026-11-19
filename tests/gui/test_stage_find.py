"""Find stages with nothing on the bus: say why, never raise.

It called ostage.kinesisis_running() -- a name that does not exist -- so the
one moment the button has something useful to say (no stages found), it threw
AttributeError instead. The window's own quiet auto-connect takes the same
path, so a session started with the stages off raised it at startup.
"""

from octobee.motion import stage as ostage
from tests.gui.test_status import _window
from tests.helpers import check


def test_find_stages_with_none_on_the_bus(app, workdir, monkeypatch):
    print("\nfind stages, none there")
    win = _window(workdir, "find")
    try:
        tab = win.tab_stages
        said = []
        monkeypatch.setattr(win.session, "log", said.append)
        monkeypatch.setattr(ostage, "list_devices", lambda: [])
        for kinesis, expect in ((True, "Kinesis"), (False, "power and USB")):
            monkeypatch.setattr(ostage, "kinesis_is_running", lambda k=kinesis: k)
            said.clear()
            found = tab.on_stage_find(quiet=True)
            check(f"no stages, Kinesis {'running' if kinesis else 'not running'}:"
                  f" it says why", not found and any(expect in m for m in said),
                  str(said))
    finally:
        win.close()
