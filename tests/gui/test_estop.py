"""The emergency stop, the latch, and the watchdog."""

import argparse
import os

from PyQt6 import QtCore, QtWidgets

from octobee.gui import window as gui
from octobee.motion import stage as ostage
from tests.helpers import (
    check,
)



def test_gui_estop(app, workdir):
    """The stop button: reachable, latching, and it releases nothing by itself.

    Every check here is a property that was missing before, and each one has a
    failure that ends with the head somewhere it should not be:

      the button lives outside the tab stack -- the old STOP was on the Stages
        tab, so during a scan watched from the Live tab it was on a page you
        could not see
      it is enabled with no stages connected -- "not connected" is this
        process's belief, and if that belief were reliable there would be
        nothing to stop
      it latches, and pressing it again does not release -- a person reaching
        for a stop in a hurry may well hit it twice
      the latch survives reconnecting -- otherwise Disconnect/Connect is an
        undocumented way to release a stopped machine
    """
    print("\nemergency stop")

    class FakeStage:
        def __init__(self, name):
            self.name, self.serial, self.model = name, "45000000", "LTS300C"
            self.homed, self.invert, self.is_open = True, False, True
            self.travel_mm = self.limit_mm = (0.0, 300.0)
            self.limit_declared = True
            # An envelope stated is an envelope declared: this double has one,
            # so it must answer the question the window asks at connect.
            self.limit_declared = True
            # A stage that recorded its position without trouble, which is the
            # other thing the window asks each axis at connect.
            self.ledger_error = ""
            self.trust_origin = "homed"
            self.stopped = 0

        @property
        def position_trusted(self):
            return not self.stopped

        @property
        def distrust_reason(self):
            return None if self.position_trusted else "stopped immediately"

        @property
        def position_mm(self):
            return 10.0

        @property
        def vel_params(self):
            return (6.0, 10.0)

    class FakeSet:
        def __init__(self, axes):
            self.axes, self.names = axes, list(axes)
            self.interlock = ostage.MotionInterlock()

        def __getitem__(self, k):
            return self.axes[k]

        def __iter__(self):
            return iter(self.axes.values())

        def emergency_stop(self, reason="emergency stop"):
            for st in self.axes.values():
                st.stopped += 1
            self.interlock.trip(reason)
            return []

        def untrusted(self):
            return [(n, st.distrust_reason) for n, st in self.axes.items()
                    if not st.position_trusted]

        def reset_interlock(self):
            return (self.interlock.reset(), self.untrusted())

        def home_sequence(self):
            return list(self.axes)

        def close(self):
            pass

    ns = argparse.Namespace(
        uut=None, demo=True, replay=None, no_connect=True,
        stages=os.path.join(workdir, "stages.json"),
        geometry=os.path.join(workdir, "estop_geom.json"),
        calibration=os.path.join(workdir, "estop_cal.json"),
        machine=os.path.join(workdir, "estop_machine.json"),
        out_dir=os.path.join(workdir, "estopcaps"),
        screenshot=None, screenshot_tab=0, screenshot_warmup=0)
    win = gui.MainWindow(ns)
    try:
        # Reachable from anywhere: it is a toolbar widget, not a tab child.
        tb = win.findChild(QtWidgets.QToolBar)
        widgets = [tb.widgetForAction(a) for a in tb.actions()]
        check("the stop button is on the toolbar, not inside a tab",
              tb.isAncestorOf(win.btn_estop),
              "a stop you have to navigate to is not a stop")
        check("it is the last thing on the toolbar",
              widgets[-2:] == [win.btn_estop, win.btn_estop_reset],
              str([type(w).__name__ for w in widgets[-3:]]))
        check("an expanding spacer pins it to the top right at any width",
              widgets[-3] is not None
              and widgets[-3].sizePolicy().horizontalPolicy()
              == QtWidgets.QSizePolicy.Policy.Expanding)
        check("Escape is bound application-wide",
              win.sc_estop.context()
              == QtCore.Qt.ShortcutContext.ApplicationShortcut,
              "otherwise it stops working the moment the wizard has focus")

        # With nothing connected it must still work, and still latch.
        check("the stop button is live with no stages connected",
              win.session.stages is None and win.btn_estop.isEnabled())
        win.on_estop()
        check("stopping with nothing connected still latches",
              win.motion.reason is not None)
        # isHidden, not isVisible: the window itself is never shown in this
        # test, so isVisible() is False for every child either way and both
        # checks would pass without testing anything.
        check("and the reset button appears only once latched",
              not win.btn_estop_reset.isHidden())
        win.motion._reason = None           # reset without opening the modal
        win._refresh_estop_ui()
        check("clearing the latch hides the reset button",
              win.btn_estop_reset.isHidden())

        stages = FakeSet({"x": FakeStage("x"), "y": FakeStage("y")})
        win.session.stages = stages
        for ax in ("x", "y"):
            win.tab_stages.stage_rows[ax]["present"] = True
        win.tab_stages.sync_controls()
        check("motion controls are live before the stop",
              win.tab_stages.stage_rows["x"]["target"].isEnabled()
              and win.tab_stages.btn_scan_start.isEnabled())
        # The right-hand jog pane is a second set of buttons onto the same
        # three axes. It is gated by the same call, and that is the only
        # reason it is safe: a second panel with its own idea of when a move
        # is allowed would be a second panel that can drive a latched machine.
        pane = win.tab_stages.jog_pane
        check("the right-hand jog pane is live before the stop too",
              pane.rows["x"]["target"].isEnabled())
        check("and it offers nothing for an axis that is not connected",
              not pane.rows["z"]["target"].isEnabled(),
              "z is not in this FakeSet")

        win.on_estop()
        check("the stop reaches every axis",
              all(st.stopped == 1 for st in stages),
              "one axis left running is the whole failure this prevents")
        check("and latches the machine, not just the axis that was moving",
              stages.interlock.tripped is not None)
        check("motion controls go dead while latched",
              not win.tab_stages.stage_rows["x"]["target"].isEnabled()
              and not win.tab_stages.btn_scan_start.isEnabled())
        check("and so does the right-hand jog pane",
              not pane.rows["x"]["target"].isEnabled()
              and not pane.btn_home.isEnabled(),
              "a stop that only disables one of two panels is not a stop")
        check("but its stop button stays live, like the tab's",
              pane.btn_stop.isEnabled())
        try:
            stages.interlock.require_clear("a move")
            ok = False
        except ostage.MotionInterlocked:
            ok = True
        check("and the interlock refuses moves at the point of command", ok,
              "the button cannot reach a thread already past its own check")

        first = win.motion.reason
        win.on_estop()
        check("pressing stop twice does not release the machine",
              win.motion.reason == first and stages.interlock.tripped is not None,
              "a person in a hurry hits it twice")

        # A stopped axis must not accept an absolute move again just because
        # the latch was cleared: an immediate stop can have lost steps.
        check("a stopped axis is no longer trusted",
              not stages["x"].position_trusted)
        was, lost = stages.reset_interlock()
        win.motion._reason = None
        check("resetting the latch clears it", was is not None
              and stages.interlock.tripped is None)
        check("but the axes stay untrusted until they are homed",
              [n for n, _ in lost] == ["x", "y"],
              "clearing a stop is not the same as knowing where the head is")

        # Disconnect/reconnect must not be a back door round a latched stop.
        win.on_estop()
        win.session.stages = None
        win.tab_stages._stage_pending = FakeSet({"x": FakeStage("x")})
        win.tab_stages.on_stage_action_done("connect", "")
        app.processEvents()
        check("reconnecting does not release a latched stop",
              win.session.stages.interlock.tripped is not None,
              "the latch belongs to the rig, not to a StageSet object")
    finally:
        # Teardown, not an API: the latch is deliberately read-only in
        # production and only moves through trigger()/reset().
        win.motion._reason = None
        win.session.stages = None
        win.close()
        app.processEvents()
