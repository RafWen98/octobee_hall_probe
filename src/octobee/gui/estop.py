"""
octobee/gui/estop.py -- the latch, and every thread allowed to command motion.

Why this is not part of a tab
-----------------------------
An emergency stop that only works while you are looking at the Stages tab is
not an emergency stop. This is the one thing in the application that has to
reach everything regardless of what is on screen, so it belongs beside the
session rather than inside any of the tabs that use it.

It exists as an object rather than a handful of window methods for a specific
reason, written down in register() below: the guided-magnet wizard owns a
motion thread of its own, and for a while the stop button knew only about the
main window's. Anything that moves the machine registers here, and there is
one list.

The latch is kept here as well as on the StageSet because the StageSet is
created at connect time -- without this, a stop pressed while disconnected
would be forgotten the moment something opened the stages again.
"""

from PyQt6 import QtCore, QtWidgets

from octobee.motion import stage as ostage


def is_running(worker):
    """True if `worker` is a live QThread. Tolerates None and dead objects."""
    try:
        return worker is not None and worker.isRunning()
    except RuntimeError:            # the C++ object is already gone
        return False


class MotionControl(QtCore.QObject):
    """The emergency stop, the graded stop, and the motion-worker registry."""

    changed = QtCore.pyqtSignal()      # latch or busy state moved

    def __init__(self, session, parent=None):
        """`parent` is the widget modal dialogs are shown over."""
        super().__init__(parent)
        self.session = session
        self._parent = parent
        self._workers = []
        self._retired = []
        self._reason = None
        self._alarms = set()

    # ---- the registry -----------------------------------------------------

    def register(self, worker):
        """Track a thread that commands motion, so a stop can reach it.

        The reason this exists rather than a single `self._scan_worker`: the
        guided-magnet wizard owns a ScanWorker of its own, and the stop button
        used to know only about the main window's. Pressing it during a guided
        run stopped the axis mid-move and then the wizard's thread, which had
        never been told anything, saw motion end, took its reading at the
        wrong place and commanded the next pose. The button appeared to do
        nothing except corrupt a point. Anything that moves the machine
        registers here.
        """
        self._workers = [w for w in self._workers
                         if w is not worker and is_running(w)]
        self._workers.append(worker)
        self.changed.emit()

    def retire(self, worker):
        """This worker has finished; stop counting it as busy.

        Called from the done handlers rather than waiting for isRunning() to
        go false, because a worker emits done() from inside run() -- so at the
        moment the handler executes the thread can still report itself as
        running, and the controls it should have re-enabled stay grey until
        something else happens to refresh them.

        The reference is kept, not dropped. A QThread garbage-collected while
        its run() is still unwinding takes the process with it.
        """
        self._workers = [w for w in self._workers if w is not worker]
        self._retired.append(worker)
        del self._retired[:-8]
        self.changed.emit()

    def busy(self):
        """True if any thread is currently allowed to command motion."""
        self._workers = [w for w in self._workers if is_running(w)]
        return bool(self._workers)

    def abort_all(self):
        """Ask every registered worker to stop. Returns how many were running."""
        n = 0
        for w in list(self._workers):
            if is_running(w):
                n += 1
                if hasattr(w, "abort"):
                    w.abort()
        return n

    # ---- the latch --------------------------------------------------------

    @property
    def reason(self):
        """Why motion is latched off, or None."""
        return self._reason

    @property
    def latched(self):
        return self._reason is not None

    def trigger(self, reason):
        """Stop the machine and latch it off. The single stop path.

        Ordered hardware first. Aborting the workers first would spend
        milliseconds in Python while the carriage is still moving, and the
        whole value of an immediate stop over a profiled one is measured in
        exactly those milliseconds. The interlock goes down with the stop, so
        a worker that wakes up in between is refused at the point of command
        rather than racing us.

        Never raises. It is wired to a button, a key and a status watchdog,
        and an exception on any of those paths would leave the machine in
        whatever half-stopped state it got to.
        """
        log = self.session.log
        stages = self.session.stages
        already = self._reason is not None
        if not already:
            self._reason = reason
        errors = []
        if stages is not None:
            try:
                errors = stages.emergency_stop(reason)
            except Exception as exc:
                errors = [f"{type(exc).__name__}: {exc}"]
        stopped = self.abort_all()
        self.changed.emit()
        if already:
            log(f"EMERGENCY STOP (again) — already latched: {self._reason}")
            return
        log(f"*** EMERGENCY STOP — {reason} ***")
        if stages is None:
            log("  the stages are not connected here; nothing to stop over "
                "USB. If something is moving, cut power to the controllers.")
        else:
            log(f"  {', '.join(stages.names)}: immediate stop, all further "
                f"motion refused until reset")
        if stopped:
            log(f"  {stopped} running job(s) aborted")
        for e in errors:
            log(f"  STOP FAILED on {e} — CUT POWER TO THE CONTROLLERS IF "
                f"ANYTHING IS STILL MOVING")
        if errors:
            QtWidgets.QMessageBox.critical(
                self._parent, "Emergency stop did not reach every axis",
                "The stop could not be delivered to:\n\n  "
                + "\n  ".join(errors)
                + "\n\nIf anything is still moving, cut power to the "
                  "controllers now. This is what a hardware emergency stop "
                  "is for; the software one needs the USB link to work.")

    def reset(self):
        """Clear the latch, after saying what is still not true afterwards."""
        if self._reason is None:
            return
        stages = self.session.stages
        lost = stages.untrusted() if stages is not None else []
        box = QtWidgets.QMessageBox(self._parent)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Reset the emergency stop")
        box.setText(f"Latched by: {self._reason}")
        detail = ("Resetting allows motion again. It does not undo anything "
                  "— check that whatever caused the stop is actually dealt "
                  "with, and that the head is where you think it is.")
        if lost:
            detail += ("\n\nThese axes will still refuse absolute moves until "
                       "they are homed, because an immediate stop can lose "
                       "steps:\n\n  "
                       + "\n  ".join(f"{n}: {why}" for n, why in lost))
        box.setInformativeText(detail)
        go = box.addButton("Reset and allow motion",
                           QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Stay stopped",
                      QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is not go:
            return
        was = self._reason
        self._reason = None
        if stages is not None:
            _, lost = stages.reset_interlock()
        self.session.log(f"emergency stop reset (was: {was}) — motion allowed")
        for name, why in lost:
            self.session.log(f"  {name}: absolute moves still refused — {why}")
        self.changed.emit()

    # ---- the graded stop --------------------------------------------------

    def stop(self):
        """End the move in progress, without latching the machine off.

        The graded one: a profiled stop, so nothing loses steps and the
        positions stay trustworthy. This is "that is far enough", where the
        red button is "something is wrong". Both abort the running jobs --
        a stop that leaves the raster thread free to command the next point
        is not a stop either.

        Not routed through the worker queue: a stop that has to wait behind the
        move it is trying to interrupt is not a stop button.
        """
        stopped = self.abort_all()
        if stopped:
            self.session.log(f"stages: {stopped} running job(s) will stop after "
                             f"the point in flight")
        if self.session.stages is None:
            return
        try:
            self.session.stages.stop_all()
            self.session.log("stages: stopped")
        except ostage.StageError as exc:
            self.session.log(f"stages: stop failed — {exc}")

    # ---- the watchdog -----------------------------------------------------

    def watchdog(self, snap):
        """One axis in trouble stops the whole machine.

        Runs on the 200 ms stage poll, which is already reading every axis for
        the table, so this costs nothing extra.

        The convention it implements: on a stacked multi-axis rig the axis
        that reports the fault is not necessarily the one about to do damage.
        A motion error means a commanded move did not happen -- stalled,
        driver fault, obstruction -- and any other axis still executing its
        half of a coordinated move is now going somewhere that was only safe
        while all three agreed. Finding a hard limit is the same argument
        arriving one step later: the soft limits were supposed to stop it and
        did not.

        Latched through the same path as the button, so the machine ends up in
        one state with one reason attached, however it got there.
        """
        name = snap["name"]
        if snap["error"]:
            key = (name, "error")
            if key not in self._alarms:
                self._alarms.add(key)
                self.trigger(f"{name} reported a motion error")
        elif snap["at_hard_limit"] and not snap["moving"]:
            key = (name, "limit")
            if key not in self._alarms:
                self._alarms.add(key)
                # Homing ends on the limit switch by design, and so does any
                # axis parked there afterwards -- so this is a warning, not a
                # stop, unless it happened while something was driving.
                if self.busy():
                    self.trigger(
                        f"{name} reached a hard limit switch during a move")
                else:
                    self.session.log(f"stages: {name} is sitting on a hard "
                                     f"limit switch")
        else:
            self._alarms.discard((name, "error"))
            self._alarms.discard((name, "limit"))
