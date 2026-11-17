"""What the window says about the rig without being asked: the strip under the
toolbar, the banner, the Log badge, the sensor buttons and the readiness list.

Headless against the synthetic probe, like the rest of the GUI tests.
"""

import argparse
import os
import time

from PyQt6 import QtWidgets

from octobee.gui import preflight
from octobee.gui import window as gui
from octobee.gui.widgets import status as gstatus
from octobee.motion import stage as ostage

from tests.helpers import check, pump


def _window(workdir, name):
    return gui.MainWindow(argparse.Namespace(
        uut=None, demo=True, replay=None, no_connect=True,
        stages=os.path.join(workdir, f"{name}_stages.json"),
        geometry=os.path.join(workdir, f"{name}_geom.json"),
        calibration=os.path.join(workdir, f"{name}_cal.json"),
        machine=os.path.join(workdir, f"{name}_machine.json"),
        out_dir=os.path.join(workdir, f"{name}_caps"),
        screenshot=None, screenshot_tab=0, screenshot_warmup=0))


class _Stage:
    def __init__(self, name):
        self.name, self.lost = name, False

    @property
    def position_trusted(self):
        return not self.lost

    @property
    def distrust_reason(self):
        return "stopped by an emergency stop" if self.lost else None


class _Set:
    """As much of a StageSet as an emergency stop and its reset touch."""

    def __init__(self, names):
        self.axes = {n: _Stage(n) for n in names}
        self.names = list(names)
        self.interlock = ostage.MotionInterlock()

    def __getitem__(self, k):
        return self.axes[k]

    def __iter__(self):
        return iter(self.axes.values())

    def emergency_stop(self, reason="emergency stop"):
        for st in self.axes.values():
            st.lost = True
        self.interlock.trip(reason)
        return []

    def untrusted(self):
        return [(n, st.distrust_reason) for n, st in self.axes.items()
                if st.lost]

    def reset_interlock(self):
        return (self.interlock.reset(), self.untrusted())

    def home_sequence(self):
        return list(self.names)

    def close(self):
        pass


def test_the_strip_says_what_the_rig_is_doing(app, workdir):
    """Connection, sensors, losses, recording and alarms, on every tab."""
    print("\nrun strip")
    win = _window(workdir, "strip")
    try:
        strip = win.strip
        check("the strip is above the tabs, not in one",
              not win.tabs.isAncestorOf(strip) and strip.parent() is not None)
        check("a synthetic source is not passed off as the probe",
              strip.chip_link.state == preflight.WARN
              and "not the probe" in strip.chip_link.text(),
              strip.chip_link.text())
        check("nothing is recording, and it says so",
              strip.chip_rec.text() == "not recording")
        check("no banner while nothing is wrong", strip.banner.isHidden())

        # The demo probe has S16 railed; two health scans exclude it.
        for _ in range(12):
            pump(win, app, 1.0)
            if "S16" in win.session.cal.dead:
                break
        check("the dead sensor is on the strip",
              "15 of 16" in strip.chip_sensors.text()
              and "S16" in strip.chip_sensors.text()
              and strip.chip_sensors.state == preflight.WARN,
              strip.chip_sensors.text())
        check("and red on the Live tab's own button for it",
              win.tab_live.sensor_state(16)[0] in ("dead", "fault"),
              str(win.tab_live.sensor_state(16)))
        check("a healthy sensor is not",
              win.tab_live.sensor_state(1)[0] in ("ok", "noisy"),
              str(win.tab_live.sensor_state(1)))
        check("the sensor buttons still switch traces on and off",
              win.tab_live.chk_sensors[2].isCheckable()
              and win.tab_live.chk_sensors[2].isChecked())
        win.tab_live.chk_sensors[2].setChecked(False)
        check("hiding a trace does not change its health colour",
              win.tab_live.sensor_state(3)[0] in ("ok", "noisy"))
        win.tab_live.chk_sensors[2].setChecked(True)
        strip.sensors_clicked.emit()
        check("clicking the sensors chip opens the Sensors tab",
              win.tabs.currentWidget() is win.table)

        win.tab_export.chk_csv.setChecked(True)
        win.tab_export.chk_raw.setChecked(False)
        win.act_record.setChecked(True)
        pump(win, app, 1.2)
        check("a recording is on the strip, with its clock",
              "REC" in strip.chip_rec.text() and "0:00:0" in strip.chip_rec.text()
              and strip.chip_rec.state == preflight.FAIL,
              strip.chip_rec.text())

        win._alarm("the stream reported an error (test)")
        check("an alarm raises the banner",
              not strip.banner.isHidden()
              and "the stream reported an error" in strip.lbl_banner.text())
        check("and is still on the status bar",
              "ALARM" in win.status.currentMessage())
        strip.btn_banner_dismiss.click()
        check("dismissing it takes the banner down",
              strip.banner.isHidden() and strip.alarm == "")
        # Closing, so stopping does not leave its "saved to" box open behind
        # this test: Qt keeps a box opened that way as the active modal one.
        win._closing = True
        win.act_record.setChecked(False)
        check("stopping the recording clears the chip",
              strip.chip_rec.text() == "not recording")
    finally:
        win._closing = True
        win.close()


def test_getting_control_back_after_an_emergency_stop(app, workdir,
                                                      monkeypatch):
    """Reset has to be on screen, and it has to lead to homing.

    The Reset button was made visible by calling setVisible() on a widget the
    toolbar owns, which a QToolBar ignores: its layout follows the widget's
    action. After a stop there was no Reset anywhere, and the machine stayed
    latched until the program was restarted.
    """
    print("\nemergency stop: the way back")
    win = _window(workdir, "reset")
    try:
        tb = win.findChild(QtWidgets.QToolBar)
        check("the toolbar's Reset is shown through its action",
              tb.widgetForAction(win.act_estop_reset) is win.btn_estop_reset)
        check("there is no Reset before a stop",
              not win.act_estop_reset.isVisible()
              and win.strip.btn_banner_reset.isHidden())

        win.session.stages = _Set(["x", "y", "z"])
        win.on_estop()
        check("after a stop the toolbar's Reset is laid out",
              win.act_estop_reset.isVisible())
        check("the banner says how to get control back",
              not win.strip.banner.isHidden()
              and "home the axes" in win.strip.lbl_banner.text(),
              win.strip.lbl_banner.text())
        check("and has a Reset of its own beside the words",
              not win.strip.btn_banner_reset.isHidden())
        check("which the operator cannot dismiss",
              win.strip.btn_banner_dismiss.isHidden())
        check("the job chip says stopped",
              "STOPPED" in win.strip.chip_job.text()
              and win.strip.chip_job.state == preflight.FAIL)

        # Answer the reset question with "Reset and home those axes".
        asked = {}

        def fake_exec(box):
            asked["text"] = box.informativeText()
            asked["buttons"] = [b.text() for b in box.buttons()]
            return 0

        def fake_clicked(box):
            return next(b for b in box.buttons() if "home" in b.text())

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", fake_exec)
        monkeypatch.setattr(QtWidgets.QMessageBox, "clickedButton",
                            fake_clicked)
        homed = []
        monkeypatch.setattr(win.tab_stages, "_home_axes",
                            lambda names: homed.append(list(names)))
        win.strip.btn_banner_reset.click()
        app.processEvents()
        check("the reset question names what is still refused",
              "x: stopped by an emergency stop" in asked.get("text", "")
              and "Jogging" in asked.get("text", ""), asked.get("text", ""))
        check("and offers to home in the same step",
              "Reset and home those axes" in asked.get("buttons", []),
              str(asked.get("buttons")))
        check("the latch is cleared", not win.motion.latched
              and win.session.stages.interlock.tripped is None)
        check("the lost axes are sent to be homed, in the safe order",
              homed == [["x", "y", "z"]], str(homed))
        check("Reset goes away again",
              not win.act_estop_reset.isVisible()
              and win.strip.banner.isHidden())

        # "Reset and allow motion" must not home anything by itself.
        win.on_estop()
        homed.clear()
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "clickedButton",
            lambda box: next(b for b in box.buttons()
                             if b.text() == "Reset and allow motion"))
        win.motion.reset()
        app.processEvents()
        check("a plain reset homes nothing", not win.motion.latched
              and homed == [], str(homed))
        check("but says where Home all is",
              "press Home all" in win.log_pane.toPlainText())

        # Closing the box, or "Stay stopped", leaves it latched.
        win.on_estop()
        monkeypatch.setattr(QtWidgets.QMessageBox, "clickedButton",
                            lambda box: None)
        win.motion.reset()
        check("declining leaves the machine latched", win.motion.latched)
        win.motion._reason = None
        win._refresh_estop_ui()
        win.session.stages = None
    finally:
        win._closing = True
        win.close()


def test_the_log_tab_counts_what_was_not_read(app, workdir):
    """A warning must show somewhere other than a tab nobody has open."""
    print("\nlog badge")
    win = _window(workdir, "badge")
    try:
        log = win.log_pane
        i = win.tabs.indexOf(log)
        log.mark_seen()
        check("nothing unread, plain label", win.tabs.tabText(i) == "Log")
        win.session.log("an ordinary line")
        check("ordinary lines are not counted", win.tabs.tabText(i) == "Log")
        win.session.log("WARNING: 3 block(s) dropped while recording")
        win.session.log("stages: jog refused — emergency stop is latched")
        check("warnings are counted on the tab",
              win.tabs.tabText(i) == "Log (2)", win.tabs.tabText(i))
        win.session.log("ALARM: recording stopped")
        check("an alarm counts too, and is kept apart",
              win.tabs.tabText(i) == "Log (3)" and log.unseen_alarms == 1)
        check("the text itself is untouched",
              "WARNING: 3 block(s) dropped" in log.toPlainText())
        log.mark_seen()
        check("reading the Log clears the count",
              win.tabs.tabText(i) == "Log" and log.unseen_warnings == 0)
        titles = [win.tabs.tabText(k) for k in range(win.tabs.count())]
        check("Profile and Help are not in the row of working tabs",
              "Profile" not in titles and "Help" not in titles, str(titles))
        check("they are still built, and open from the toolbar",
              win.tab_help is not None and win.tab_profile is not None
              and win.act_help.text() == "Help")
        win.act_help.trigger()
        check("Help opens in a window of its own",
              "Help" in win._side_windows
              and win._side_windows["Help"].isAncestorOf(win.tab_help))
        win._side_windows["Help"].close()
    finally:
        win._closing = True
        win.close()


def test_one_connect_button_and_one_jog_pane(app, workdir):
    """Connect or Disconnect, never both; and the axes on screen once."""
    print("\ntoolbar and side panel")
    win = _window(workdir, "side")
    try:
        check("with a source running only Disconnect is offered",
              win.act_disconnect.isVisible() and not win.act_connect.isVisible())
        win.on_disconnect()
        check("and after it, only Connect",
              win.act_connect.isVisible() and not win.act_disconnect.isVisible())
        check("the strip follows", win.strip.chip_link.text() == "disconnected")

        pane = win.tab_stages.jog_pane
        check("the jog pane starts in the side panel",
              win.side_panel.isAncestorOf(pane)
              and not win.tab_stages.isAncestorOf(pane))
        check("the Stages tab has no jog controls of its own",
              all("target" not in row and "step" not in row
                  for row in win.tab_stages.stage_rows.values()))
        win.act_side.setChecked(False)
        check("hiding the side panel hides the column",
              win.side_panel.isHidden())
        check("and moves the jog pane into the Stages tab",
              win.tab_stages.isAncestorOf(pane) and not pane.isHidden())
        check("without a second Home all and Stop beside it",
              win.tab_stages.btn_stage_homeall.isHidden()
              and win.tab_stages.btn_stage_stop.isHidden())
        win.act_side.setChecked(True)
        check("showing it again puts the pane back",
              win.side_panel.isAncestorOf(pane)
              and not win.side_panel.isHidden()
              and not win.tab_stages.btn_stage_homeall.isHidden())
    finally:
        win._closing = True
        win.close()


def test_the_machine_tab_asks_before_it_starts(app, workdir, monkeypatch):
    """The readiness list is on the Volume map page and Start obeys it."""
    print("\nready to run")
    win = _window(workdir, "ready")
    try:
        tab = win.tab_machine
        titles = [tab.pages.tabText(i) for i in range(tab.pages.count())]
        check("set-up and the volume map are separate pages",
              titles == ["Set-up", "Volume map"], str(titles))
        check("the run buttons are outside the part that scrolls",
              not any(isinstance(p, QtWidgets.QScrollArea)
                      for p in _ancestors(tab.btn_vol_start, tab.pages)))
        got = {c.name: c for c in tab.readiness()}
        check("with no stages it is not ready, and says why",
              got["Stages"].state == preflight.FAIL
              and got["Path"].state == preflight.FAIL)
        tab.refresh_ready()
        check("the list on the page says the same",
              tab.ready.lbl_summary.text().startswith("not ready"),
              tab.ready.lbl_summary.text())

        check("nothing is running, so there is no job to report",
              tab.run_status() is None and win.strip.chip_job.text() == "idle")
        tab._run_progress = (10, 40, "x 0..300")
        tab._run_costs = [60.0 * (k + 1) for k in range(40)]
        tab._run_t0 = time.time() - 600.0
        win.session.volume_runner = object()
        text, frac = tab.run_status()
        check("a map in flight reports its line and the time left",
              text.startswith("volume map: line 10/40")
              and "30 min left" in text and abs(frac - 0.25) < 1e-6, text)
        win._refresh_strip()
        check("and that is what the strip shows, on any tab",
              "line 10/40" in win.strip.chip_job.text()
              and not win.strip.bar_job.isHidden())
        win.session.volume_runner = None
        tab._run_progress = None

        seen = []
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                            lambda *a, **k: seen.append(a[2]))
        tab.map_finished.emit("", "x never started moving")
        check("a map that ends early raises the banner",
              "volume map ended early" in win.strip.alarm, win.strip.alarm)
        check("durations read as a person would say them",
              gstatus.format_duration(7800) == "2 h 10 min"
              and gstatus.format_duration(840) == "14 min"
              and gstatus.format_clock(3725) == "1:02:05")
    finally:
        win._closing = True
        win.close()


def _ancestors(widget, stop):
    out = []
    w = widget.parentWidget()
    while w is not None and w is not stop:
        out.append(w)
        w = w.parentWidget()
    return out
