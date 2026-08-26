"""The guided-magnet wizard."""

import argparse
import os

import numpy as np
from PyQt6 import QtWidgets

from octobee.calib import convert as ocal
from octobee.calib import magnet as omag
from octobee.gui import window as gui
from octobee.gui.dialogs.magnet import MagnetWizard
from octobee.motion import stage as ostage
from octobee.calib import geometry as pgeom
from tests.helpers import (
    _synth_magnet_run,
    check,
)



def test_magnet_wizard_reopens(app, workdir):
    """Closing the wizard must let it be opened again.

    The bug this pins: MagnetWizard.closeEvent() accepted the event without
    calling super(), so QDialog never called reject(), never emitted
    finished(), and the main window kept a reference to a hidden dialog for
    ever. The button then "worked" -- it raised the dead one -- and did nothing
    visible, which from a bug report is indistinguishable from the whole
    program having frozen.
    """
    print("\nguided magnet wizard, reopened")

    class FakeStage:
        def __init__(self, name):
            self.name, self.serial, self.model = name, "45000000", "LTS300C"
            self.homed, self.invert = True, False
            self.travel_mm = (0.0, 300.0)
            self.limit_mm = (0.0, 300.0)
            self.limit_declared = True
            self.limit_declared = True
            self.position_trusted, self.distrust_reason = True, None

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

        def home_sequence(self):
            return list(self.axes)

        def untrusted(self):
            return [(n, st.distrust_reason) for n, st in self.axes.items()
                    if not st.position_trusted]

        def close(self):
            pass

    ns = argparse.Namespace(
        uut=None, demo=True, replay=None, no_connect=True,
        geometry=os.path.join(workdir, "reopen_geom.json"),
        calibration=os.path.join(workdir, "reopen_cal.json"),
        machine=os.path.join(workdir, "reopen_machine.json"),
        out_dir=os.path.join(workdir, "reopencaps"),
        screenshot=None, screenshot_tab=0, screenshot_warmup=0)
    win = gui.MainWindow(ns)
    win.session.stages = FakeSet({"x": FakeStage("x"), "y": FakeStage("y")})
    try:
        for attempt in (1, 2, 3):
            win.tab_calib.on_guided_magnet()
            app.processEvents()
            check(f"the wizard opens (attempt {attempt})",
                  win.tab_calib._magnet_wizard is not None
                  and win.tab_calib._magnet_wizard.isVisible())
            win.tab_calib._magnet_wizard.close()
            app.processEvents()
            check(f"and closing it lets go of the handle (attempt {attempt})",
                  win.tab_calib._magnet_wizard is None,
                  "a stuck handle makes the button silently do nothing")

        # Closing must not leave anything modal behind either -- that would
        # block every other window in the program, not just this one.
        check("nothing modal is left blocking the rest of the window",
              QtWidgets.QApplication.activeModalWidget() is None)
        probe = QtWidgets.QDialog(win)
        probe.show()
        app.processEvents()
        check("other windows still open afterwards", probe.isVisible())
        probe.close()

        # An unfinished run is only in the dialog, so closing has to ask.
        win.tab_calib.on_guided_magnet()
        app.processEvents()
        wiz = win.tab_calib._magnet_wizard
        g = win.session.geom
        for sweep in _synth_magnet_run(
                g, np.ones(16),
                np.array([0.0, 200.0, g.fsv_radius_mm + 25.0])).sweeps[:2]:
            wiz.run.add(sweep)
        asked = []
        real_q = QtWidgets.QMessageBox.question
        QtWidgets.QMessageBox.question = staticmethod(
            lambda *a, **k: asked.append(a[2])
            or QtWidgets.QMessageBox.StandardButton.Cancel)
        try:
            wiz.close()
            app.processEvents()
            check("closing an unfinished run asks before abandoning it",
                  bool(asked) and "2 of 4 poses" in asked[0])
            check("and cancelling keeps the window open", wiz.isVisible())
            # The prompt used to say the poses existed only in the window.
            # They are written as they land now, so saying that would talk an
            # operator out of a close that costs nothing -- and, worse, teach
            # them to distrust the warning that matters.
            asked.clear()
            wiz._save_base = os.path.join(str(workdir), "magcal_partial")
            wiz.close()
            app.processEvents()
            check("and says where the measured poses already are, rather "
                  "than threatening a loss that is not real",
                  bool(asked) and "magcal_partial.npz" in asked[0]
                  and "only in this window" not in asked[0])
        finally:
            QtWidgets.QMessageBox.question = real_q
        wiz._finished = True          # pretend it was applied, so it can go
        wiz.close()
        app.processEvents()
    finally:
        win.close()


def test_magnet_wizard_saves(app, workdir):
    """Finishing the wizard must leave the trim ON DISK, not just in memory.

    The bug this pins: the wizard applied the gain trim to the running
    calibration and never wrote calibration.json, so closing the window threw
    away a twenty-minute measurement without saying anything. Checking the
    in-memory object would have passed happily -- the assertion has to be
    against the file.
    """
    print("\nguided magnet wizard")
    cal_path = os.path.join(workdir, "wizard_cal.json")
    geom_path = os.path.join(workdir, "wizard_geom.json")
    pgeom.Geometry().save(geom_path)
    ocal.Calibration().save(cal_path)

    ns = argparse.Namespace(
        uut=None, demo=True, replay=None, geometry=geom_path,
        calibration=cal_path, machine=os.path.join(workdir, "wiz_machine.json"),
        out_dir=os.path.join(workdir, "wizcaps"),
        screenshot=None, screenshot_tab=0, screenshot_warmup=0,
        no_connect=True)
    win = gui.MainWindow(ns)

    # Every modal in finish() answered without clicking: warning() returns, and
    # exec() leaving clickedButton() as None is "leave the file alone".
    real_warn, real_exec = QtWidgets.QMessageBox.warning, QtWidgets.QMessageBox.exec
    QtWidgets.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QtWidgets.QMessageBox.exec = lambda self: 0
    try:
        wiz = MagnetWizard(win)
        g = win.session.geom
        rng = np.random.default_rng(19)
        gain = rng.normal(1.0, 0.07, 16)
        magnet = np.array([0.0, 200.0, g.fsv_radius_mm + 25.0])
        for sweep in _synth_magnet_run(g, gain, magnet).sweeps:
            wiz.run.add(sweep)

        before = ocal.Calibration.load(cal_path)
        check("the calibration on disk starts untrimmed",
              np.allclose(before.gain_corr, 1.0))

        wiz.chk_apply.setChecked(True)
        wiz.finish()

        after = ocal.Calibration.load(cal_path)
        check("finishing the wizard writes the trim to calibration.json",
              not np.allclose(after.gain_corr, 1.0),
              f"trim now spans {after.gain_corr.min():.3f}.."
              f"{after.gain_corr.max():.3f} in the FILE")
        check("and the trim it wrote is the one the run measured",
              np.allclose(after.gain_corr[:, 0], win.session.cal.gain_corr[:, 0]))
        check("the file records where the trim came from",
              "guided magnet run" in (after.notes or ""), after.notes or "")
        check("and the run itself is on disk beside it",
              any(f.startswith("magcal_") and f.endswith(".npz")
                  for f in os.listdir(ns.out_dir)),
              str(os.listdir(ns.out_dir)))

        # Unticked, it must leave the file alone -- and say the run is safe.
        ocal.Calibration().save(cal_path)
        win.session.cal = ocal.Calibration.load(cal_path)
        wiz2 = MagnetWizard(win)
        for sweep in _synth_magnet_run(g, gain, magnet).sweeps:
            wiz2.run.add(sweep)
        wiz2.chk_apply.setChecked(False)
        wiz2.finish()
        check("unticked, it does not touch the calibration file",
              np.allclose(ocal.Calibration.load(cal_path).gain_corr, 1.0))
        check("and it says the measurement is still recoverable",
              "nothing about it is lost" in win.log_pane.toPlainText())
        wiz.close()
        wiz2.close()
    finally:
        QtWidgets.QMessageBox.warning = real_warn
        QtWidgets.QMessageBox.exec = real_exec
        win.close()


def test_magnet_wizard_standard_run(app, workdir):
    """The wizard opens on the standard, and every pose reaches the disk.

    Two things that were previously only true in someone's head. The Sweep box
    was sized from probe_geometry.json and the standoff, so editing the
    geometry silently changed what "the calibration run" meant and two runs a
    month apart were not the same measurement. And the run was written once,
    by finish(), so anything that ended the program before the fourth pose --
    a crash, a closed window -- threw away the three already driven, about
    half an hour of stage time each.
    """
    print("\nguided magnet wizard, the standard run")

    class FakeStage:
        def __init__(self, name):
            self.name, self.serial, self.model = name, "45000000", "LTS300C"
            self.homed, self.invert = True, False
            self.travel_mm = self.limit_mm = (0.0, 300.0)
            self.limit_declared = True
            self.position_trusted, self.distrust_reason = True, None

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

        def home_sequence(self):
            return list(self.axes)

        def untrusted(self):
            return []

        def close(self):
            pass

    ns = argparse.Namespace(
        uut=None, demo=True, replay=None, no_connect=True,
        geometry=os.path.join(workdir, "std_geom.json"),
        calibration=os.path.join(workdir, "std_cal.json"),
        machine=os.path.join(workdir, "std_machine.json"),
        out_dir=os.path.join(workdir, "stdcaps"),
        screenshot=None, screenshot_tab=0, screenshot_warmup=0)
    win = gui.MainWindow(ns)
    win.session.stages = FakeSet({"x": FakeStage("x"), "y": FakeStage("y"),
                                  "z": FakeStage("z")})
    try:
        # A geometry deliberately unlike the bench's, to prove the standard
        # does not follow it. The derived sweep here would be 230 mm.
        win.session.geom = pgeom.Geometry(plate_pitch_mm=50.0)
        wiz = MagnetWizard(win)
        std = omag.STANDARD_RUN
        got = {
            "sweep_mm": wiz.spin_span.value(),
            "step_mm": wiz.spin_step.value(),
            "seconds_per_point": wiz.spin_secs.value(),
            "standoff_mm": wiz.spin_standoff.value(),
            "cut_half_span_mm": wiz.spin_across_half.value(),
            "cut_step_mm": wiz.spin_across_step.value(),
            "dither_half_span_mm": wiz.spin_dither_half.value(),
            "dither_points": wiz.spin_dither_pts.value(),
        }
        for k, want in std.items():
            check(f"the wizard opens on the standard {k} of {want:g}",
                  abs(got[k] - want) < 1e-9, f"got {got[k]:g}")
        check("and the standard does not move when the geometry does",
              wiz.spin_span.value() == std["sweep_mm"],
              f"a 50 mm plate pitch would derive "
              f"{omag.suggested_sweep(win.session.geom)[0]:g} mm")
        check("which is the 145-point pose the panel promises",
              "= 145 points per pose" in wiz.lbl_points.text(),
              wiz.lbl_points.text())

        # Touching the standoff is what hands the sizing back to the physics
        # rule -- and it must not do so before it is touched.
        wiz.spin_standoff.setValue(40.0)
        check("moving the standoff re-derives the cut it no longer sizes",
              wiz.spin_across_half.value() == 40.0
              and wiz.spin_dither_half.value() == 10.0,
              f"cut {wiz.spin_across_half.value():g}, "
              f"dither {wiz.spin_dither_half.value():g}")

        # ---- and each pose is on disk as it lands
        g = win.session.geom
        run = _synth_magnet_run(g, np.ones(16),
                                np.array([0.0, 200.0, g.fsv_radius_mm + 25.0]))
        bases = set()
        for n, sweep in enumerate(run.sweeps, start=1):
            wiz._pending = sweep
            wiz._finish_pose()
            app.processEvents()
            bases.add(wiz._save_base)
            back = omag.MagnetRun.load(wiz._save_base + ".npz")
            check(f"pose {n} is on disk before pose {n + 1} is driven",
                  len(back) == n, f"loaded {len(back)}")
        check("all four went to ONE pair of files, not four",
              len(bases) == 1, str(bases))
        check("and the log says so as it goes",
              "4 of 4 poses saved to" in win.log_pane.toPlainText())
        wiz._finished = True
        wiz.close()
    finally:
        win.close()
