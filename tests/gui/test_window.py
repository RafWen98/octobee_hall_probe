"""The whole application, driven headless."""

import argparse
import os
import tempfile
import time

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from octobee.acq import carrier as ob
from octobee.calib import convert as ocal
from octobee.gui import window as gui
from octobee.calib import roll as opc
from octobee import record as orec
from octobee import machine as omach
from octobee.calib import geometry as pgeom
from tests.helpers import (
    _simsopt_file,
    check,
    pump,
    read_clkdiv,
    read_csv,
)



def test_autoconnect(app, workdir):
    """The window connects itself when it opens -- and shuts up when it can't.

    Hermetic: ConnectWorker and connect_stages are both stubbed, so this never
    touches a carrier or a USB device. What is under test is the wiring --
    whether an attempt is made, and how a failure is reported -- not the
    connecting itself, which test_app covers against real hardware.
    """
    print("\nautomatic connect")

    attempts = []
    modals = []

    class FakeConnectWorker(QtCore.QThread):
        done = QtCore.pyqtSignal(object, object, str)
        progress = QtCore.pyqtSignal(str)

        def __init__(self, hosts, fs):
            super().__init__()
            attempts.append(hosts)

        def start(self):
            # Deferred, not immediate: a real connect takes seconds, and the
            # guard under test only matters while one is in flight. A stub
            # that finishes inside start() would report every guard as broken.
            QtCore.QTimer.singleShot(
                150, lambda: self.done.emit(None, None,
                                            "no route to host (stubbed)"))

    real_worker = gui.ConnectWorker
    real_critical = QtWidgets.QMessageBox.critical
    gui.ConnectWorker = FakeConnectWorker
    QtWidgets.QMessageBox.critical = staticmethod(
        lambda *a, **k: modals.append(a))

    def build(**over):
        ns = argparse.Namespace(
            uut=None, demo=False, replay=None,
            stages=os.path.join(workdir, "stages.json"),
            geometry=os.path.join(workdir, "probe_geometry.json"),
            calibration=os.path.join(workdir, "calibration.json"),
            machine=os.path.join(workdir, "machine.json"),
            out_dir=os.path.join(workdir, "captures"),
            screenshot=None, screenshot_tab=0, screenshot_warmup=0,
            no_connect=False)
        for k, v in over.items():
            setattr(ns, k, v)
        w = gui.MainWindow(ns)
        # Before any event is processed, so the deferred connect cannot reach
        # the real stages when it fires.
        w.tab_stages.connect_stages = lambda quiet=False: False
        return w

    def pump(seconds=1.0):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.02)

    try:
        win = build()
        pump()
        check("opening the window starts a connect on its own",
              len(attempts) == 1, f"{len(attempts)} attempt(s)")
        check("a failed automatic connect logs instead of raising a dialog",
              not modals and "connect failed" in win.log_pane.toPlainText(),
              f"{len(modals)} modal(s)")
        check("and it says the rest of the window still works",
              "everything that does not need the carriers" in
              win.log_pane.toPlainText())
        check("Connect is left enabled to try again",
              win.act_connect.isEnabled())
        win.close()

        # A second attempt while one is in flight would open a second stream
        # against the same carriers. --no-connect here so the only attempts
        # counted are the ones this test makes.
        attempts.clear()
        win = build(no_connect=True)
        for _ in range(3):
            win.on_connect()
        check("overlapping connects start exactly one worker",
              len(attempts) == 1, f"{len(attempts)} worker(s) for 3 calls")
        pump(0.4)
        check("and the next attempt is allowed once that one has finished",
              (win.on_connect(), len(attempts))[1] == 2,
              f"{len(attempts)} worker(s) after the first failed")
        pump(0.3)
        win.close()

        attempts.clear()
        win = build(no_connect=True)
        pump(0.6)
        check("--no-connect starts disconnected", not attempts
              and "press Connect" in win.log_pane.toPlainText())
        win.close()

        attempts.clear()
        win = build(demo=True)
        pump(0.6)
        check("a demo window does not go looking for carriers",
              not attempts and win.session.source is not None)
        win.close()
    finally:
        gui.ConnectWorker = real_worker
        QtWidgets.QMessageBox.critical = real_critical


def test_close_is_clean(app, workdir):
    """closeEvent must not raise, because Qt will not tell you if it does.

    Qt calls closeEvent from C++ and swallows whatever comes back, so a stale
    attribute reference in there is invisible: the window still disappears, the
    test still passes, and the shutdown work after the failing line -- waiting
    for motion workers, stopping the poll timer, releasing the USB handles --
    silently does not happen. Releasing those handles is the part that matters:
    they are exclusive-open, so a missed close means Kinesis will not start
    again until the process dies.

    So this calls closeEvent directly, where an exception propagates.
    """
    print("\nclean shutdown")
    ns = argparse.Namespace(
        uut=None, demo=True, replay=None, no_connect=True,
        stages=os.path.join(workdir, "stages.json"),
        geometry=os.path.join(workdir, "close_geom.json"),
        calibration=os.path.join(workdir, "close_cal.json"),
        machine=os.path.join(workdir, "close_machine.json"),
        out_dir=os.path.join(workdir, "closecaps"),
        screenshot=None, screenshot_tab=0, screenshot_warmup=0)
    win = gui.MainWindow(ns)
    try:
        # Closed with a recording open, which is the awkward case: stopping it
        # is what normally raises the "saved to" box, and a modal put up
        # during teardown is a window that will not close.
        win.tab_export.chk_csv.setChecked(True)
        win.tab_export.chk_raw.setChecked(False)
        win.act_record.setChecked(True)
        check("a file is open before the close", win.session.csv_rec is not None)
        win.closeEvent(QtGui.QCloseEvent())
        check("closeEvent completes without raising", True)
        check("the recording is closed on the way out",
              win.session.csv_rec is None)
        check("and nothing was left on screen to dismiss",
              win._saved_box is None,
              "the path is still in the Log and on the Data output tab")
    finally:
        win.close()
        app.processEvents()


def test_machine_tab(app, workdir):
    """The Machine tab, driven the way a person drives it."""
    print("\nmachine tab")
    coil_path = _simsopt_file(os.path.join(workdir, "tab_coils.json"))
    cfg_path = os.path.join(workdir, "tab_machine.json")
    omach.MachineConfig(coil_file=coil_path, coil_radius_mm=20.0,
                        energised=["C1", "C2"]).save(cfg_path)

    ns = argparse.Namespace(
        uut=None, demo=True, replay=None, no_connect=True,
        stages=os.path.join(workdir, "stages.json"),
        geometry=os.path.join(workdir, "tab_geom.json"),
        calibration=os.path.join(workdir, "tab_cal.json"),
        machine=cfg_path,
        out_dir=os.path.join(workdir, "tabcaps"),
        screenshot=None, screenshot_tab=0, screenshot_warmup=0)
    win = gui.MainWindow(ns)
    try:
        titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
        check("the window has a Machine tab", "Machine" in titles, str(titles))
        check("the coil file named in the placement is loaded on startup",
              win.session.coils is not None and len(win.session.coils) == 2,
              "nothing loaded" if win.session.coils is None else win.session.coils.note)
        check("every coil is listed", win.tab_machine.tbl_coils.rowCount() == 2,
              f"{win.tab_machine.tbl_coils.rowCount()} rows")

        # Switch a coil off through the table, as a click does.
        item = win.tab_machine.tbl_coils.item(0, 0)
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        check("unticking a coil switches it off",
              win.session.machine.energised == ["C2"], str(win.session.machine.energised))
        check("and the view is told",
              win.tab_machine.machine_view.energised == {"C2"},
              str(win.tab_machine.machine_view.energised))

        # A coil that is off is still solid: putting the probe on top of the
        # coil that was just switched off must still report a collision.
        on_coil = win.session.coils["C1"].points_mm[0]
        for attr, value in zip(("x_mm", "y_mm", "z_mm"), on_coil):
            win.tab_machine.machine_pose_spins[attr].setValue(float(value))
        win.tab_machine.refresh_machine(force=True)
        check("driving the probe onto a switched-off coil still collides",
              "INSIDE" in win.tab_machine.lbl_clearance.text(), win.tab_machine.lbl_clearance.text())

        win.tab_machine.machine_pose_spins["x_mm"].setValue(float(on_coil[0]) + 4000.0)
        win.tab_machine.refresh_machine(force=True)
        check("and moving it away again clears",
              "clear of" in win.tab_machine.lbl_clearance.text(),
              win.tab_machine.lbl_clearance.text())

        # The pose the tab is showing is the pose that gets written down.
        win.tab_machine.on_machine_save()
        saved = omach.MachineConfig.load(cfg_path)
        check("Save writes the pose that is on screen",
              abs(saved.pose.x_mm - (on_coil[0] + 4000.0)) < 1e-6
              and saved.energised == ["C2"],
              f"{saved.pose.x_mm:.1f} mm, {saved.energised}")

        # What a field map would carry, built the way on_scan_start builds it.
        meta = win.session.machine.to_scan_meta(win.session.coils, None)
        check("a map started now would record the machine around it",
              meta["summary"].startswith("1/2 coils energised"),
              meta["summary"])

        _check_machine_gizmo(win)
        _check_machine_ring(win)
        _check_machine_volume(win, app)
    finally:
        win.close()
        app.processEvents()


def _drag(view, frm, to):
    """Press at one widget pixel, move to another, release. As a mouse does."""
    for kind, at, buttons in (
            (QtCore.QEvent.Type.MouseButtonPress, frm,
             QtCore.Qt.MouseButton.NoButton),
            (QtCore.QEvent.Type.MouseMove, to,
             QtCore.Qt.MouseButton.LeftButton),
            (QtCore.QEvent.Type.MouseButtonRelease, to,
             QtCore.Qt.MouseButton.NoButton)):
        point = QtCore.QPointF(float(at[0]), float(at[1]))
        getattr(view, {QtCore.QEvent.Type.MouseButtonPress: "mousePressEvent",
                       QtCore.QEvent.Type.MouseMove: "mouseMoveEvent",
                       QtCore.QEvent.Type.MouseButtonRelease:
                           "mouseReleaseEvent"}[kind])(
            QtGui.QMouseEvent(kind, point, point,
                              QtCore.Qt.MouseButton.LeftButton, buttons,
                              QtCore.Qt.KeyboardModifier.NoModifier))


def _check_machine_ring(win):
    """The ring round the zero point turns the probe about the machine's Z.

    Dragged in the plane the ring lives in rather than in pixels, so the check
    is a real one: pick two points on the ring 40 degrees apart in the world,
    project them, and see whether the pose came out 40 degrees different.
    """
    tab, view = win.tab_machine, win.tab_machine.machine_view
    origin = view._gizmo_origin
    radius = view._ring_mm

    def on_ring(deg):
        a = np.radians(deg)
        world = origin + radius * np.array([np.cos(a), np.sin(a), 0.0])
        return view._to_screen(world[None, :])[0]

    was = win.session.machine.pose.rot_z_deg
    frm, to = on_ring(200.0), on_ring(240.0)
    check("the ring is on screen where the pointer can reach it",
          np.isfinite(frm).all() and np.isfinite(to).all()
          and view._ring_at(QtCore.QPointF(*frm)),
          f"{frm} -> {to}")
    check("and an arrow does not also claim that pixel",
          view._axis_at(QtCore.QPointF(*frm)) is None,
          "the arrow would win the click and slide the probe instead")
    _drag(view, frm, to)
    now = win.session.machine.pose.rot_z_deg
    check("dragging the ring turns the probe by the angle dragged",
          abs(((now - was) - 40.0 + 180.0) % 360.0 - 180.0) < 1.0,
          f"{was:g} -> {now:g} deg")
    check("and only the rotation moved",
          (win.session.machine.pose.x_mm, win.session.machine.pose.y_mm,
           win.session.machine.pose.z_mm) == (
              tab.machine_pose_spins["x_mm"].value(),
              tab.machine_pose_spins["y_mm"].value(),
              tab.machine_pose_spins["z_mm"].value()))
    check("the drawing follows the pose, not the other way round",
          np.allclose(view._placement.rotation(),
                      omach.rotation_matrix(now), atol=1e-12))

    # The point of the rotation: it says how the assembly is MOUNTED relative
    # to the coils, so the probe's own axes have to turn with it. A handle that
    # stayed square to the machine would be drawing an orientation the rig does
    # not have, and the mounting angle would be invisible.
    tab.machine_pose_spins["rot_z_deg"].setValue(0.0)
    square = view._gizmo_dirs.copy()
    check("with nothing mounted at an angle the probe's axes are the machine's",
          np.allclose(square, np.eye(3), atol=1e-12), str(square.round(3)))

    tab.machine_pose_spins["rot_z_deg"].setValue(90.0)
    turned = view._gizmo_dirs.copy()
    check("turning the mounting angle turns the probe's own axes with it",
          np.allclose(turned[:, 0], [0.0, 1.0, 0.0], atol=1e-9)
          and np.allclose(turned[:, 1], [-1.0, 0.0, 0.0], atol=1e-9),
          f"rig x now points {turned[:, 0].round(3)}, rig y {turned[:, 1].round(3)}")
    check("and Z is untouched, because Z is what it turned about",
          np.allclose(turned[:, 2], [0.0, 0.0, 1.0], atol=1e-12),
          str(turned[:, 2].round(3)))
    check("the drawn probe turns with them",
          np.allclose(view._placement.rotation(), turned, atol=1e-12),
          "the handle and the body would be showing different orientations")

    # And dragging one of those arrows now moves along the RIG axis, which on a
    # rig turned 90 deg is machine y. Anything else would mean the arrow and
    # the stage it stands for disagree about which way they go.
    screen = view._gizmo_screen()
    was = np.array([win.session.machine.pose.x_mm, win.session.machine.pose.y_mm,
                    win.session.machine.pose.z_mm])
    step = screen[1] - screen[0]                    # the rig x arrow
    push = 40.0
    want = push / float(np.hypot(*step)) * view._gizmo_len_mm
    _drag(view, 0.5 * (screen[0] + screen[1]),
          0.5 * (screen[0] + screen[1]) + push * step / np.hypot(*step))
    now = np.array([win.session.machine.pose.x_mm, win.session.machine.pose.y_mm,
                    win.session.machine.pose.z_mm])
    check("dragging the rig's x arrow on a rig turned 90 deg moves machine y",
          abs((now - was)[1] - want) < 0.06 and abs((now - was)[0]) < 0.06,
          f"moved {(now - was).round(2)}, wanted {want:.2f} mm along machine y")

    tab.machine_pose_spins["rot_z_deg"].setValue(0.0)
    for attr, value in zip(("x_mm", "y_mm", "z_mm"), was):
        tab.machine_pose_spins[attr].setValue(float(value))


def _check_machine_volume(win, app):
    """Planning a swept volume: what is drawn, and what is refused.

    Driven with coil avoidance ON against a coil set the probe is sitting
    right next to, because the interesting answer is the carved one -- a plan
    that ignores the coils is arithmetic, and a plan that does not is the
    feature.
    """
    tab, view = win.tab_machine, win.tab_machine.machine_view
    session = win.session

    # A box the probe can reach into, with a coil through part of it.
    # The test coil set is two 1 m circles about the origin, so a box reaching
    # out towards x = 1 m has winding through the middle of it. Anywhere else
    # and the avoidance has nothing to do and the check proves nothing.
    for attr, value in (("x_mm", 800.0), ("y_mm", -150.0), ("z_mm", -150.0)):
        tab.machine_pose_spins[attr].setValue(value)
    tab.chk_vol_all.setChecked(False)
    for i, (start, size) in enumerate(((0.0, 300.0),) * 3):
        tab.vol_spins["from"][i].setValue(start)
        tab.vol_spins["size"][i].setValue(size)
    tab.spin_vol_step.setValue(50.0)
    tab.spin_vol_margin.setValue(5.0)

    vol = tab.volume()
    check("the volume the controls describe is the box that was typed",
          np.allclose(vol.lo_mm, [0, 0, 0]) and np.allclose(vol.hi_mm, [300] * 3)
          and vol.step_mm == 50.0, vol.describe())
    check("editing it invalidates any plan made for the old one",
          tab._plan is None and not tab.btn_vol_start.isEnabled(),
          "a stale path would be drawn while the rig ran a different one")
    check("and the box is drawn where the stages can actually go",
          view._volume_item.visible(),
          f"volume item visible: {view._volume_item.visible()}")

    tab.chk_vol_avoid.setChecked(True)
    tab.on_volume_plan()
    deadline = time.time() + 60.0
    while (tab._plan_worker is not None and tab._plan_worker.isRunning()
           and time.time() < deadline):
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()

    check("planning produces lines to sweep", tab._plan is not None
          and len(tab._plan.lines) > 0, tab.lbl_volume.text()[:100])
    check("the coils took some of the volume away",
          tab._reachable is not None and not tab._reachable.all(),
          "nothing was carved -- the probe fits everywhere, so this "
          "configuration does not exercise the avoidance")
    check("the path that survived is drawn, and so is what did not",
          view._path_item.visible() and view._dropped_item.visible(),
          f"path {view._path_item.visible()}, dropped "
          f"{view._dropped_item.visible()}")

    # The promise the label makes has to be the promise the plan keeps.
    if session.probe_cloud is None:
        session.probe_cloud = omach.probe_cloud(session.geom)
    ends = []
    for line in tab._plan.lines:
        for where in (line.start_mm, line.stop_mm):
            ends.append([line.fixed.get(a, 0.0) if a != line.sweep else where
                         for a in ("x", "y", "z")])
    good = omach.clear_of_coils(
        session.machine.pose.flange_path_mm(np.array(ends)),
        session.machine.pose, session.probe_cloud, session.coils,
        session.machine.coil_radius_mm, margin_mm=5.0)
    check("every planned line really does clear the coils by the margin set",
          bool(good.all()), f"{int((~good).sum())} of {len(good)} do not")

    # Flying it must move the drawn probe and leave the stages alone.
    tab.on_volume_preview()
    check("flying the path starts an animation", tab._preview is not None)
    seen = set()
    for _ in range(12):
        tab._preview_step()
        if tab._preview_at:
            seen.add(tuple(round(v, 3) for v in tab._preview_at.values()))
        app.processEvents()
    check("and it walks the probe along the path rather than sitting still",
          len(seen) > 3, f"{len(seen)} distinct positions")
    check("the clearance readout follows the flown probe",
          "clear of" in tab.lbl_clearance.text()
          or "INSIDE" in tab.lbl_clearance.text(), tab.lbl_clearance.text())
    tab._stop_preview()
    check("stopping it hands the drawing back to the real stage reading",
          tab._preview is None and tab._preview_at is None)

    # Nothing to run against: there are no stages in this test, and starting
    # anyway is the failure that would drive an unconnected rig.
    check("starting a map with no stages connected is refused",
          not tab.btn_vol_start.isEnabled() or session.stages is not None,
          "the start button is live with nothing to move")

def _check_machine_gizmo(win):
    """The drag handle on the probe's zero point moves the probe, and only it.

    Driven through the widget's own mouse handlers rather than by calling the
    move directly: what is worth testing is the whole chain -- picking the
    arrow out of the drawing in screen pixels, turning a mouse movement back
    into millimetres, and writing them into the placement box -- and every step
    of that is where a sign or a projection can be wrong.
    """
    tab = win.tab_machine
    view = tab.machine_view
    view.resize(800, 600)
    # Back inside the machine, and framed: the checks above left the probe four
    # metres off to one side, where the handle is behind the camera.
    for attr in ("x_mm", "y_mm", "z_mm"):
        tab.machine_pose_spins[attr].setValue(0.0)
    view.reset_camera()
    tab.refresh_machine(force=True)

    screen = view._gizmo_screen()
    check("the handle is on screen at the probe's zero point",
          screen is not None and np.allclose(
              view._gizmo_origin, win.session.machine.pose.origin_mm(None)),
          "off screen" if screen is None else str(view._gizmo_origin))
    if screen is None:
        return

    before = np.array([win.session.machine.pose.x_mm,
                       win.session.machine.pose.y_mm,
                       win.session.machine.pose.z_mm])
    step = screen[1] - screen[0]                    # the x arrow, in pixels
    push_px = 40.0
    want = push_px / float(np.hypot(*step)) * view._gizmo_len_mm
    _drag(view, 0.5 * (screen[0] + screen[1]),
          0.5 * (screen[0] + screen[1]) + push_px * step / np.hypot(*step))

    after = np.array([win.session.machine.pose.x_mm,
                      win.session.machine.pose.y_mm,
                      win.session.machine.pose.z_mm])
    # 0.06 mm: the spin box the drag writes through keeps one decimal.
    check("dragging the x arrow slides the probe that far along machine x",
          abs((after[0] - before[0]) - want) < 0.06,
          f"moved {after[0] - before[0]:.3f} mm, wanted {want:.3f} mm")
    check("and leaves y and z exactly where they were",
          np.array_equal(after[1:], before[1:]),
          f"{before[1:]} -> {after[1:]}")
    check("the spin box shows what was dragged",
          abs(tab.machine_pose_spins["x_mm"].value() - after[0]) < 1e-9,
          f"{tab.machine_pose_spins['x_mm'].value():.3f} vs {after[0]:.3f}")

    # Empty space still orbits. A handle that swallowed every click would make
    # the view unturnable, which is worse than not having one.
    azimuth = view.opts["azimuth"]
    _drag(view, (5.0, 5.0), (95.0, 25.0))
    check("a drag away from the handle still turns the camera",
          view.opts["azimuth"] != azimuth
          and np.array_equal([win.session.machine.pose.x_mm,
                              win.session.machine.pose.y_mm,
                              win.session.machine.pose.z_mm], after),
          f"azimuth {azimuth:.1f} -> {view.opts['azimuth']:.1f}")

    view.set_gizmo_visible(False)
    check("turning the handle off takes it out of the drawing",
          view._gizmo_screen() is None
          and not view._gizmo_items[0].visible(), "still there")
    view.set_gizmo_visible(True)


def test_app(app, args, workdir):
    kind = ("live hardware" if args.live else
            f"replay {args.replay}" if args.replay else "synthetic probe")
    print(f"\napplication ({kind})")
    ns = argparse.Namespace(
        uut=None, demo=not (args.replay or args.live), replay=args.replay,
        stages=os.path.join(workdir, "stages.json"),
        geometry=os.path.join(workdir, "probe_geometry.json"),
        calibration=os.path.join(workdir, "calibration.json"),
        machine=os.path.join(workdir, "machine.json"),
        # Into the temp dir, not captures/. A test run used to leave ~9 MB of
        # synthetic recordings, reports and health CSVs in the real capture
        # directory, named identically to the bench data beside them and
        # indistinguishable from it afterwards.
        out_dir=os.path.join(workdir, "captures"),
        screenshot=None, screenshot_tab=0, screenshot_warmup=0)
    win = gui.MainWindow(ns)
    clkdiv_before = {}
    if args.live:
        # Everything this tool does to the carriers' clock has to be undone on
        # the way out. A run that leaves them slowed down silently poisons the
        # next one, which is how a "full rate" snapshot once came back at
        # 20 kSPS.
        clkdiv_before = {h: read_clkdiv(h) for h in ob.DEFAULT_UUTS}
        print(f"  carriers found at clkdiv {clkdiv_before}")
        win.on_connect()
        deadline = time.time() + 90
        while win.session.source is None and time.time() < deadline:
            app.processEvents()
            time.sleep(0.05)
    check("source started", win.session.source is not None)
    if win.session.source is None:
        # The failed check above is the result; nothing below it can run.
        return

    pump(win, app, 3.0)
    check("data is flowing", win.session.roll.filled > 100,
          f"{win.session.roll.filled} points buffered")
    check("sensor table populated", win._last_table is not None
          and len(win._last_table) == 16)
    check("channel health computed", win.session.last_health is not None
          and len(win.session.last_health) == 64)
    check("S16 detected as dead", "S16" in win.session.cal.dead,
          f"excluded: {sorted(win.session.cal.dead)}")

    # ---- tare ----
    v = win.session.roll.view()
    before = np.abs(np.median(v, axis=0)).max() if v.shape[0] else float("nan")
    win.tab_calib.start_collect("tare", 0.5)
    pump(win, app, 2.5)
    check("tare completed", win.session.collecting is None)
    check("tare stored a non-trivial zero", np.any(win.session.cal.zero_mt != 0),
          f"max |zero| {np.abs(win.session.cal.zero_mt).max():.4f} mT (was reading "
          f"{before:.4f} mT)")
    pump(win, app, 1.0)

    # zero_mt is defined BEFORE the gain trim, so the zero a tare stores must
    # not depend on what trim happens to be loaded. It used to: _finish_tare
    # reconstructed the uncorrected field as `data + zero_mt`, which only
    # inverts to_mt() when the trim is 1.0 and the matrix is identity, so
    # taring after a magnet pass or a roll solve stored a zero scaled by the
    # trim -- invisibly, because a uniformly wrong zero looks like a zero.
    #
    # Driven directly rather than through the live source: the demo probe's
    # magnet keeps moving, so two tares taken seconds apart legitimately
    # differ and could not tell "scaled by the trim" from "the field moved".
    # The same fixed block through both trims can.
    zeros_at = {}
    fixed = np.full((64, pgeom.N_SENSORS, 4), 2.2)         # volts, all at VCM
    fixed[:, :, 0] += 0.063                                # Bx = VCM + 63 mV
    for trim in (1.0, 2.0):
        win.session.cal.clear_tare()
        win.session.cal.clear_matrix()
        win.session.cal.gain_corr = np.full((pgeom.N_SENSORS, 3), trim)
        win.session.collecting = {"what": "tare", "blocks": [], "n": 0, "need": 1,
                          "peak": None, "baseline": None, "tag": None,
                          "decim": 1}
        win.tab_calib.collect_block(win.session.cal.to_mt(fixed), fixed)
        check(f"tare at trim {trim:g} completed", win.session.collecting is None)
        zeros_at[trim] = win.session.cal.zero_mt.copy()
    d = float(np.abs(zeros_at[2.0] - zeros_at[1.0]).max())
    check("the stored zero does not scale with the gain trim", d < 1e-9,
          f"trim 1.0 -> {zeros_at[1.0][0]}, trim 2.0 -> {zeros_at[2.0][0]}")
    check("and it is the uncorrected field, not the corrected one",
          abs(zeros_at[1.0][0, 0] - 1.0) < 0.01,
          f"63 mV at 63 V/T should tare at 1 mT, got {zeros_at[1.0][0, 0]:.4f}")
    win.session.cal.clear_gain()
    win.session.cal.clear_tare()
    pump(win, app, 1.0)

    # ---- magnet pass ----
    win.tab_calib.btn_magnet.setChecked(True)
    pump(win, app, 4.0)
    win.tab_calib.btn_magnet.setChecked(False)
    app.processEvents()
    check("magnet pass captured peaks", win.session.magnet_peaks is not None)
    if win.session.magnet_peaks is not None:
        live = win.session.cal.live_mask()
        resp = int((win.session.magnet_peaks[live] > 1e-4).sum())
        check("most live sensors responded", resp >= 10, f"{resp}/15 responded")
        rep = ocal.spread_report(win.session.magnet_peaks, live=live)
        check("spread report produced", "raw_spread" in rep,
              f"spread {rep.get('raw_spread', float('nan')):.2f}x")

        # ---- gain trim ----
        before_spread = rep.get("raw_spread")
        win.tab_calib.on_apply_gain()
        trimmed = win.session.magnet_peaks * win.session.cal.gain_corr[:, 0]
        after = trimmed[live].max() / trimmed[live].min()
        check("gain trim narrows the spread", after < 1.0001,
              f"{before_spread:.2f}x -> {after:.4f}x")
        win.tab_calib.on_clear_gain()
        check("gain trim clears", np.allclose(win.session.cal.gain_corr, 1.0))

    # ---- Earth-field roll calibration panel ----
    # The demo source is a moving dipole, not a uniform field being rolled
    # through, so the numbers it produces are meaningless. What is being
    # tested here is the wiring: that a sweep is captured from UNCORRECTED
    # field, that the solve/apply path runs, and that sweeps survive a trip
    # through disk. The physics is graded in test_posecal() against a known
    # truth instead.
    check("roll panel is built",
          all(hasattr(win.tab_calib, a)
              for a in ("btn_solve_roll", "btn_apply_roll", "lbl_sweeps",
                        "spin_bearth", "chk_isotropic", "spin_sweep_s")))
    check("solve is disabled with no sweeps recorded",
          not win.tab_calib.btn_solve_roll.isEnabled())

    win.session.cal.gain_corr = np.full((pgeom.N_SENSORS, 3), 2.0)
    for tag in ("A", "B", "C"):
        win.tab_calib.start_sweep(tag, 0.4)
        pump(win, app, 2.0)
    win.session.cal.clear_gain()
    check("all three sweeps recorded", set(win.session.sweeps) == {"A", "B", "C"},
          win.tab_calib.lbl_sweeps.text())
    check("solve enabled once sweeps exist", win.tab_calib.btn_solve_roll.isEnabled())

    if set(win.session.sweeps) == {"A", "B", "C"}:
        # A sweep must be independent of whatever trim was loaded when it was
        # taken -- that is why it is captured pre-correction. The x2 gain above
        # was live during capture and must not show up in the stored data.
        sw = win.session.sweeps["A"]
        check("sweeps are stored uncorrected",
              np.abs(sw.b_mt).max() < 1e4 and np.isfinite(sw.b_mt).all())
        check("sweeps record the range they were taken at",
              sw.ranges_mt is not None
              and np.allclose(sw.ranges_mt, win.session.cal.ranges_mt))

        win.tab_calib.on_solve_roll()
        check("roll solve produced a solution", win.session.pose_solution is not None,
              "" if win.session.pose_solution is not None
              else " ".join(win.tab_calib.cal_report.toPlainText().split())[:300])
        check("apply is enabled after a solve", win.tab_calib.btn_apply_roll.isEnabled())
        if win.session.pose_solution is not None:
            sol = win.session.pose_solution
            check("solve used all three orientations",
                  sorted(sol.tags) == ["A", "B", "C"], f"{sol.tags}")
            check("report reaches the calibration pane",
                  "gain spread" in win.tab_calib.cal_report.toPlainText())
            win.session.cal.apply_pose_solution(sol)
            check("applying installs the matrix and clears the trim",
                  win.session.cal.has_matrix and np.allclose(win.session.cal.gain_corr, 1.0))
            check("the applied calibration still converts",
                  np.isfinite(win.session.cal.to_mt(
                      np.full((4, pgeom.N_SENSORS, 4), 2.2))).all())
            win.session.cal.clear_matrix()
            win.session.cal.clear_tare()

        with tempfile.TemporaryDirectory() as d:
            ok = True
            for tag, sw in win.session.sweeps.items():
                back = opc.RollSweep.load(sw.save(os.path.join(d, f"rs_{tag}")))
                ok &= (back.tag == tag
                       and back.b_mt.shape == sw.b_mt.shape
                       and np.allclose(back.ranges_mt, sw.ranges_mt))
            check("sweeps round trip through disk", ok)

    win.tab_calib.on_clear_sweeps()
    check("clearing sweeps disables solve",
          not win.session.sweeps and not win.tab_calib.btn_solve_roll.isEnabled())

    # ---- display cannot starve acquisition ----
    win.cmb_view.setCurrentIndex(list(gui.VIEW_RATES).index(10.0))
    start_interval = win.view_timer.interval()
    check("view runs on its own timer, not the acquisition one",
          win.view_timer is not win.timer
          and win.timer.interval() <= start_interval,
          f"acquisition {win.timer.interval()} ms, view {start_interval} ms")
    for _ in range(8):
        win._note_draw_time(500.0)          # pretend every redraw is very slow
    check("a slow redraw backs the view off automatically",
          win.view_timer.interval() > start_interval,
          f"{start_interval} ms -> {win.view_timer.interval()} ms")
    check("the backoff is bounded",
          win.view_timer.interval() <= gui.MAX_VIEW_INTERVAL_MS,
          f"{win.view_timer.interval()} ms")
    # Re-pick the SAME entry, which is what a user does to undo a backoff.
    win.cmb_view.activated.emit(win.cmb_view.currentIndex())
    check("re-picking the same rate clears the backoff",
          win.view_timer.interval() == start_interval,
          f"back to {win.view_timer.interval()} ms")

    win.probe_pane.chk_3d.setChecked(False)
    before = win.session.roll.filled
    pump(win, app, 1.0)
    check("acquisition continues with the 3D head off", win.session.roll.filled >= before)
    win.probe_pane.chk_3d.setChecked(True)

    # The peak bars are the other thing in the right-hand pane that can be
    # given up for space, and for the same reason: the head and the stage
    # controls are what you want tall while driving the rig somewhere.
    win.probe_pane.chk_bars.setChecked(False)
    before = win.session.roll.filled
    pump(win, app, 1.0)
    check("the peak bars can be turned off", win.probe_pane.bars.isHidden())
    check("acquisition continues with the peak bars off",
          win.session.roll.filled >= before)
    win.probe_pane.chk_bars.setChecked(True)
    pump(win, app, 0.5)
    check("and turned back on", not win.probe_pane.bars.isHidden())

    # The jog pane is in the right-hand column, not in the tab stack: moving
    # the head and watching the field answer is one action.
    pane = win.tab_stages.jog_pane
    check("the jog pane is in the right-hand pane, outside the tabs",
          pane.parent() is not None and not win.tabs.isAncestorOf(pane),
          "on the tab strip it would cost a trip away from what you are watching")
    check("with nothing connected it offers no move",
          not pane.rows["x"]["target"].isEnabled()
          and not pane.btn_home.isEnabled())

    win.act_pause.setChecked(True)
    n_before = win.session.roll.filled
    pump(win, app, 1.0)
    check("pausing the view does not pause acquisition",
          win.session.roll.filled >= n_before and win.paused)
    win.act_pause.setChecked(False)

    # ---- recording ----
    win.tab_export.chk_csv.setChecked(True)
    win.tab_export.chk_raw.setChecked(True)
    win.tab_export.chk_tube.setChecked(False)
    check("the Record button carries no dot before it is pressed",
          win.act_record.icon().isNull())
    win.act_record.setChecked(True)
    check("CSV recorder opened", win.session.csv_rec is not None)
    check("raw recorder opened", win.session.raw_rec is not None)
    check("a dot appears on the Record button while capturing",
          not win.act_record.icon().isNull()
          and win.act_record.text() == "Recording")
    csv_path = win.session.csv_rec.path if win.session.csv_rec else None
    raw_path = win.session.raw_rec.path if win.session.raw_rec else None
    pump(win, app, 3.0)
    rows_written = win.session.csv_rec.n_rows if win.session.csv_rec else 0
    win.act_record.setChecked(False)
    check("recording stopped cleanly", win.session.csv_rec is None and win.session.raw_rec is None)
    check("and the dot goes with the file being closed",
          win.act_record.icon().isNull() and win.act_record.text() == "Record")

    # Stopping says where the data went. The Log and the Data output tab both
    # carried the path already and both were being missed -- the one place
    # nobody looks after a capture is the tab they were not on.
    saved = win._saved_box
    check("stopping a recording says where the files were written",
          saved is not None)
    if saved is not None:
        told = saved.text() + "\n" + saved.informativeText()
        check("the box names both files",
              os.path.basename(csv_path or "") in told
              and os.path.basename(raw_path or "") in told, told.replace("\n", " | "))
        check("and the folder they are in",
              os.path.basename(win.session.out_dir) in told)
        check("it does not block the window while it is up",
              not saved.isModal() or saved.windowModality()
              != QtCore.Qt.WindowModality.ApplicationModal,
              "a snapshot and a field map both stop recording first")
        saved.close()
        # Twice: the box is only released when its deleteLater is delivered,
        # and Qt holds activeModalWidget on it until then.
        app.processEvents()
        app.processEvents()
    check("dismissing it leaves nothing modal behind",
          QtWidgets.QApplication.activeModalWidget() is None)
    check("and the window stops holding on to it", win._saved_box is None)

    # ---- read the CSV back and check it ----
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            lines = f.readlines()
        header = [ln for ln in lines if ln.startswith("#")]
        cols = next(ln for ln in lines
                    if not ln.startswith("#")).strip().split(",")
        names, data = read_csv(csv_path)
        col = {n: i for i, n in enumerate(names)}
        check("CSV has a provenance header", len(header) >= 8,
              f"{len(header)} header lines")
        check("CSV has 1 + 16*4 columns", len(cols) == 65, f"{len(cols)} columns")
        check("CSV row count matches the recorder",
              data.shape[0] == rows_written, f"{data.shape[0]} rows")
        dt = np.diff(data[:, col["t_s"]])
        check("CSV timebase matches the output rate",
              abs(np.median(dt) - 1.0 / win.session.out_rate) < 1e-6,
              f"dt {np.median(dt)*1e3:.3f} ms at {win.session.out_rate:g} Hz")
        s1 = data[:, [col["S1_Bx_mT"], col["S1_By_mT"], col["S1_Bz_mT"]]]
        check("CSV |B| column equals the vector norm",
              np.allclose(np.linalg.norm(s1, axis=1), data[:, col["S1_absB_mT"]],
                          atol=1e-4, rtol=1e-3))
        check("CSV values are finite", np.isfinite(data).all())
        check("CSV carries real signal, not zeros",
              np.abs(data[:, col["S1_absB_mT"]]).max() > 1e-6,
              f"max |B| on S1 {np.abs(data[:, col['S1_absB_mT']]).max():.4f} mT")

    # ---- read the raw file back and check it ----
    if raw_path and os.path.exists(raw_path):
        x, meta = orec.load_raw(raw_path)
        check("raw sidecar records the channel names",
              len(meta["channel_names"]) == x.shape[1] == 64)
        check("raw sample count matches the sidecar",
              x.shape[0] == meta["n_samples"], f"{x.shape[0]} samples")
        boxes = orec.raw_to_boxes(x, meta)
        b = win.session.cal.convert(boxes, meta["volts_per_count"])
        check("raw file reconverts to finite field values",
              np.isfinite(b).all() and b.shape[1:] == (16, 3),
              f"shape {b.shape}")

    # ---- exports ----
    win.tab_export.export_summary()
    win.tab_health.export_csv()
    win.tab_export.export_json()
    win.tab_health.analyse()
    app.processEvents()
    txt = win.tab_health.health_text.toPlainText()
    check("diagnostics text produced", "per-sensor verdict" in txt,
          f"{len(txt)} chars")
    check("diagnostics names the VCM channels", "VCM" in txt)

    exports = [line for line in win.tab_export.export_log.toPlainText().splitlines()
               if line.strip()]
    check("three one-shot exports logged", len(exports) >= 3,
          f"{len(exports)} entries")
    for line in exports:
        p = line.split("] ", 1)[-1].split("  (")[0]
        if os.path.exists(p):
            check(f"export exists and is non-empty: {os.path.basename(p)}",
                  os.path.getsize(p) > 0, f"{os.path.getsize(p)} bytes")

    # ---- calibration round trip ----
    cal_path = os.path.join(workdir, "roundtrip.json")
    win.session.cal.zero_mt[0, 0] = 1.2345
    win.session.cal.ranges_mt[5] = 400.0
    win.session.cal.save(cal_path)
    reloaded = ocal.Calibration.load(cal_path)
    check("calibration survives a save/load round trip",
          np.allclose(reloaded.zero_mt, win.session.cal.zero_mt)
          and np.allclose(reloaded.ranges_mt, win.session.cal.ranges_mt)
          and reloaded.dead == win.session.cal.dead)

    # ---- geometry round trip and rebuild ----
    g = pgeom.Geometry(mapping="ring-major")
    g.tube_width_mm = 55.0
    g.save(ns.geometry)
    win.tab_calib.on_reload_geometry()
    check("geometry reload reaches the 3D view",
          abs(win.probe_pane.view3d.geom.tube_width_mm - 55.0) < 1e-9)
    check("geometry reload reaches the sensor table",
          win.table.geom.mapping == "ring-major")
    pump(win, app, 1.0)
    check("still acquiring after a geometry change", win.session.roll.filled > 100)

    # ---- tube frame CSV ----
    win.tab_export.chk_tube.setChecked(True)
    win.tab_export.chk_raw.setChecked(False)
    win.act_record.setChecked(True)
    tube_path = win.session.csv_rec.path if win.session.csv_rec else None
    pump(win, app, 1.5)
    win.act_record.setChecked(False)
    if win._saved_box is not None:
        win._saved_box.close()
        app.processEvents()
        app.processEvents()
    if tube_path and os.path.exists(tube_path):
        with open(tube_path, encoding="utf-8") as f:
            head = "".join(f.readlines()[:12])
        check("tube-frame CSV is labelled as such", "frame: tube" in head)
        tn, td = read_csv(tube_path)
        tc = {n: i for i, n in enumerate(tn)}
        check("tube-frame CSV has data", td.shape[0] > 10, f"{td.shape[0]} rows")
        # Rotation preserves length, so |B| must be identical in either frame.
        s1t = td[:, [tc["S1_Bx_mT"], tc["S1_By_mT"], tc["S1_Bz_mT"]]]
        check("tube-frame |B| still matches its own components",
              np.allclose(np.linalg.norm(s1t, axis=1), td[:, tc["S1_absB_mT"]],
                          atol=1e-4, rtol=1e-3))

    if args.live:
        st = win.session.source.stats()
        check("no stream gaps on the live link", st.get("gaps", 0) == 0,
              f"gaps {st.get('gaps')}, lost {st.get('lost')}")
        # Measure over a window in which the loop is actually being pumped.
        # A count accumulated across the whole run says more about this
        # harness -- which deliberately blocks for most of a second at a time
        # parsing files back -- than about the application, whose real contract
        # is that a running session does not shed data.
        for stream in win.session.source.streamers:
            stream.dropped = 0
        pump(win, app, 3.0)
        dropped = sum(x.dropped for x in win.session.source.streamers)
        check("no blocks dropped while the session is running", dropped == 0,
              f"{dropped} dropped over 3 s of streaming")

        # The snapshot stops the stream, captures at the full 200 kSPS, and
        # hands the port back. It is the only path that takes over the
        # carriers' stream ownership, so it gets exercised for real.
        win.tab_export.spin_snap_s.setValue(1.0)
        win.on_snapshot()
        deadline = time.time() + 120
        while (win._snap_worker is not None and win._snap_worker.isRunning()
               and time.time() < deadline):
            app.processEvents()
            time.sleep(0.05)
        app.processEvents()
        snaps = [line for line in win.tab_export.export_log.toPlainText().splitlines()
                 if ".npz" in line]
        check("snapshot written", bool(snaps), snaps[-1] if snaps else "none")
        if snaps:
            sp = snaps[-1].split("] ", 1)[-1].split("  (")[0]
            cap = ocal.load_capture(sp)
            check("snapshot is a full-rate capture",
                  cap["fs_hz"][0] > 150000, f"{cap['fs_hz'][0]/1e3:g} kSPS")
            check("snapshot holds both boxes", len(cap["ai"]) == 2)
            rows = ocal.channel_health(cap["ai"], cap["vpc"], cap["hosts"])
            check("snapshot reproduces the S16 fault",
                  "S16" in ocal.suggest_dead(rows))

    win.on_disconnect()
    if args.live:
        time.sleep(2.0)
        after = {h: read_clkdiv(h) for h in ob.DEFAULT_UUTS}
        check("carriers left at the clock they were found at",
              after == clkdiv_before, f"{clkdiv_before} -> {after}")
    win.close()
