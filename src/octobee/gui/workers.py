"""Background threads. Anything that would otherwise block the event loop."""

import os
import threading
import time

import numpy as np
from PyQt6 import QtCore

from octobee.acq import carrier as ob
from octobee import live as olive
from octobee import machine as omach
from octobee.motion import encoder as oenc
from octobee.motion import scan as oscan
from octobee.motion import stage as ostage
from octobee.gui.sources import LiveSource

# ==========================================================================
# background workers
# ==========================================================================

class ConnectWorker(QtCore.QThread):
    """
    Bring both carriers up: release the stream, set the clock, read the frame
    layout, and start reading.

    The two boxes are prepared concurrently. Every step is a network round trip
    against an independent box, so doing them one after the other doubled the
    wait for no reason. What is left is dominated by two deliberate settling
    delays (stopping the stream, and letting a new clkdiv take), not by chatter.
    """

    done = QtCore.pyqtSignal(object, object, str)     # source, prev_clkdiv, error
    progress = QtCore.pyqtSignal(str)

    def __init__(self, hosts, target_fs, block_samples=2048):
        super().__init__()
        self.hosts = hosts
        self.target_fs = target_fs
        self.block_samples = block_samples

    def run(self):
        prev, layouts, streamers, errors = {}, {}, {}, {}

        def prepare(h):
            try:
                self.progress.emit(f"{h}: releasing the stream")
                if ob.stop_live_stream(h):
                    self.progress.emit(f"{h}: stopped a running capture")
                if self.target_fs:
                    self.progress.emit(f"{h}: setting "
                                       f"{self.target_fs/1000:g} kSPS")
                    prev[h], actual = olive.set_rate(h, self.target_fs)
                    self.progress.emit(f"{h}: running at {actual/1000:g} kSPS")
                self.progress.emit(f"{h}: reading the frame layout")
                lay = ob.probe_uut(h)
                layouts[h] = lay
                # take_over=False: the stream was already released above, and
                # repeating it here cost another round trip per box.
                # Deeper queue than the default. The reader refills at link
                # speed, roughly five times real time, so the nominal seconds
                # of buffering are worth about a fifth of that against a stall
                # on the main thread. 128 blocks is ~25 MB per box and buys a
                # couple of seconds, which covers any single operation the GUI
                # performs.
                st = ob.Streamer(h, lay, block_samples=self.block_samples,
                                 take_over=False, queue_depth=128)
                st.start()
                streamers[h] = st
                self.progress.emit(f"{h}: waiting for data")
            except Exception as e:
                errors[h] = e

        threads = [threading.Thread(target=prepare, args=(h,), daemon=True)
                   for h in self.hosts]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        try:
            if errors:
                h, e = next(iter(errors.items()))
                raise RuntimeError(f"{h}: {type(e).__name__}: {e}")
            ordered = [streamers[h] for h in self.hosts]
            # A carrier takes a few seconds to start pushing after the socket
            # opens, and the reader only emits a block once BOTH have data --
            # so reporting success on the socket alone leaves the window blank
            # with no explanation.
            deadline = time.time() + 30
            while time.time() < deadline:
                for h in self.hosts:
                    if streamers[h].error:
                        raise RuntimeError(f"{h}: {streamers[h].error}")
                if all(st.bytes_read > 0 for st in ordered):
                    break
                time.sleep(0.1)
            silent = [h for h in self.hosts if streamers[h].bytes_read == 0]
            if silent:
                raise RuntimeError(
                    f"connected but no data arrived from {', '.join(silent)}. "
                    f"Something else most likely owns port 4210 there -- check "
                    f"for a running Phoebus 'Streaming Capture' on that box.")
            source = LiveSource(self.hosts, [layouts[h] for h in self.hosts],
                                ordered)
            self.done.emit(source, prev, "")
        except Exception as e:
            for st in streamers.values():
                st.stop()
            self.done.emit(None, prev, f"{type(e).__name__}: {e}")


class SnapshotWorker(QtCore.QThread):
    """
    Take a lossless capture at the carriers' own full rate.

    Live streaming runs the ADCs slowed down, because the link cannot carry
    200 kSPS continuously. A snapshot is short enough that the box can buffer
    it and deliver every sample, so the clock is put back first -- otherwise
    the "full rate" capture would quietly be the reduced one. The boxes are
    left at the restored clock, which is where they started.
    """

    done = QtCore.pyqtSignal(str, str, float)         # path, error, fs_hz

    def __init__(self, hosts, seconds, path, restore_clkdiv=None):
        super().__init__()
        self.hosts = hosts
        self.seconds = seconds
        self.path = path
        self.restore_clkdiv = dict(restore_clkdiv or {})

    def run(self):
        fs = 0.0
        try:
            if self.restore_clkdiv:
                for h, prev in self.restore_clkdiv.items():
                    olive.restore_rate(h, prev)
                time.sleep(3.0)                       # let the clock settle
            res = ob.capture_all(self.hosts, self.seconds, verbose=False)
            save = {}
            for bi, h in enumerate(self.hosts):
                d, lay = res[h]
                save[f"ai_{bi}"] = d["ai"]
                save[f"sam_cnt_{bi}"] = d["sam_cnt"]
                save[f"usec_cnt_{bi}"] = d["usec_cnt"]
                save[f"temp_raw_{bi}"] = d.get("temp_raw", np.zeros(8))
                save[f"pwr_good_{bi}"] = d.get("pwr_good", 0)
                save[f"fs_hz_{bi}"] = lay.fs_hz
                save[f"vpc_{bi}"] = lay.volts_per_count
                save[f"adc_range_{bi}"] = lay.adc_range
            save["hosts"] = np.array(list(self.hosts))
            fs = float(save["fs_hz_0"])
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            np.savez_compressed(self.path, **save)
            self.done.emit(self.path, "", fs)
        except Exception as e:
            self.done.emit("", f"{type(e).__name__}: {e}", fs)


def _stage_set_for(mapping):
    """Build a StageSet for axis->serial, reading everything stages.json says.

    Not StageSet.from_config: the axis assignment here comes from the table in
    the Stages tab, which is allowed to differ from what is on disk -- that is
    how you try an assignment out before saving it. Everything ELSE still has
    to come from the file, and doing that inline at each call site is how the
    soft limits and the home order got left out of one of them. One place, so
    there is one thing to keep complete.
    """
    frames = ostage.load_axis_frames()
    motion = ostage.load_axis_motion(axes=mapping)
    return ostage.StageSet(
        {name: ostage.Stage(serial, name=name,
                            invert=frames.get(name, {}).get("invert", False),
                            origin_mm=frames.get(name, {}).get("origin_mm"),
                            limit_mm=frames.get(name, {}).get("limit_mm"),
                            vel_mm_s=motion[name][0],
                            accel_mm_s2=motion[name][1])
         for name, serial in mapping.items()},
        home_order=ostage.load_home_order(axes=mapping))


class StageWorker(QtCore.QThread):
    """
    Run one blocking stage operation off the GUI thread.

    Connecting, homing and moving all block for seconds to minutes. Reading
    position and status does not -- those come out of the DLL's own polling
    cache, so the GUI reads them directly on its own timer and only the
    commands come through here.
    """

    done = QtCore.pyqtSignal(str, str)                 # what, error
    progress = QtCore.pyqtSignal(str)

    def __init__(self, what, fn):
        super().__init__()
        self.what = what
        self.fn = fn
        self._abort = False

    def abort(self):
        """Ask a multi-step command to stop between steps.

        Homing three axes is three moves, and stopping the one in flight used
        to leave the loop free to start the next: the operator pressed stop,
        one carriage halted, and the next set off. Commands that take more
        than one move check `aborted` between them.
        """
        self._abort = True

    @property
    def aborted(self):
        return self._abort

    def run(self):
        try:
            self.fn(self.progress.emit)
            self.done.emit(self.what, "")
        except Exception as e:
            self.done.emit(self.what, f"{type(e).__name__}: {e}")


class PlanWorker(QtCore.QThread):
    """Work out which of a volume's grid nodes the probe body can occupy.

    Off the GUI thread because it is seconds, not milliseconds: a 300 mm cube
    on a 5 mm grid is 227,000 positions, and the whole point of drawing the
    reachable region is that somebody adjusts the pose and watches it change.
    A one-second freeze per adjustment would make that unusable.

    Nothing here touches hardware; it is arithmetic against the coil file.
    """

    done = QtCore.pyqtSignal(object, str)             # boolean grid, error
    progress = QtCore.pyqtSignal(int, int)

    def __init__(self, volume, placement, cloud, coils, radius_mm, margin_mm,
                 labels=None):
        super().__init__()
        self.volume = volume
        self.placement = placement
        self.cloud = cloud
        self.coils = coils
        self.radius_mm = float(radius_mm)
        self.margin_mm = float(margin_mm)
        self.labels = labels
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            grid = omach.reachable_grid(
                self.volume.lo_mm, self.volume.step_mm, self.volume.shape,
                self.placement, self.cloud, self.coils, self.radius_mm,
                margin_mm=self.margin_mm, labels=self.labels,
                progress=lambda i, n: self.progress.emit(int(i), int(n)))
            self.done.emit(None if self._abort else grid, "")
        except Exception as e:
            self.done.emit(None, f"{type(e).__name__}: {e}")


class EncoderCalibrationWorker(QtCore.QThread):
    """Find which stream column is which axis, and its counts per millimetre.

    Drives one axis at a time a known distance and reads the encoder counts
    either side of the move. The counts themselves arrive on the acquisition
    tick, so this thread never touches the stream: it asks for a reading
    through `counts_now`, which the tick keeps fresh, and waits long enough
    either side for one to have landed.

    Moving ONE axis at a time is not politeness, it is the whole method -- the
    column that moved is the column that belongs to the axis that moved, and
    two at once makes that unanswerable.
    """

    done = QtCore.pyqtSignal(object, str)             # {axis: spec}, error
    message = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int, int)

    # Long enough for a tick to have delivered a block at any sane output rate,
    # and for the axis to have stopped ringing enough that it is not still
    # counting.
    SETTLE_S = 0.6

    def __init__(self, stages, read_counts, distance_mm=None, axes=None):
        super().__init__()
        self.stages = stages
        self.read_counts = read_counts
        self.distance_mm = float(distance_mm or oenc.CALIBRATION_MM)
        self.axes = list(axes or oenc.AXES)
        self._abort = False

    def abort(self):
        self._abort = True

    def _counts(self):
        time.sleep(self.SETTLE_S)
        return self.read_counts()

    def run(self):
        found, problems = {}, []
        try:
            self.stages.interlock.require_clear("an encoder calibration")
            for i, axis in enumerate(self.axes):
                if self._abort:
                    break
                self.progress.emit(i, len(self.axes))
                if axis not in self.stages.names:
                    continue
                st = self.stages[axis]
                lo, hi = st.limit_mm
                here = st.position_mm
                # Away from whichever limit is closer, so a rig parked at the
                # end of its travel calibrates instead of refusing.
                step = self.distance_mm
                if here + step > hi:
                    step = -self.distance_mm
                if here + step < lo:
                    problems.append(f"{axis}: no room to move "
                                    f"{self.distance_mm:g} mm either way "
                                    f"inside {lo:g}..{hi:g} mm")
                    continue
                before = self._counts()
                st.move_by(step)
                after = self._counts()
                if before is None or after is None:
                    problems.append(
                        f"{axis}: no encoder counts arrived -- only "
                        f"acq1001_695 carries them, and only while the stream "
                        f"is running")
                    st.move_by(-step)
                    continue
                column, scale, note = oenc.fit_scale(before, after, step)
                st.move_by(-step)            # put it back where it was found
                if column is None:
                    problems.append(f"{axis}: {note}")
                    continue
                found[axis] = {"column": column, "counts_per_mm": scale}
                self.message.emit(f"encoders: {axis} -> {note}")
            self.progress.emit(len(self.axes), len(self.axes))
            self.done.emit(found, "; ".join(problems))
        except Exception as e:
            self.done.emit(found, f"{type(e).__name__}: {e}")


class SweepWorker(QtCore.QThread):
    """Drive the stages through a volume sweep.

    Unlike ScanWorker this does NOT own the carriers. A sweep reads the live
    stream that is already running, so the boxes stay on their reduced rate,
    the live plot keeps working, and there is no clock to restore afterwards.
    What it owns is the motion; the field arrives through the acquisition tick
    and is filed against `runner.line_index` from there.
    """

    done = QtCore.pyqtSignal(int, str)                # lines completed, error
    progress = QtCore.pyqtSignal(int, int, str)       # i, n, where
    message = QtCore.pyqtSignal(str)

    def __init__(self, runner):
        super().__init__()
        self.runner = runner
        self._abort = False

    def abort(self):
        """Ask the sweep to stop. The line in flight is cut short, not finished.

        The opposite of ScanWorker's choice, and for the opposite reason: a
        settled point half taken is worthless, where a line half swept is half
        a line of perfectly good data with its own positions against it.
        """
        self._abort = True

    def run(self):
        try:
            self.runner.log_fn = self.message.emit
            n = self.runner.run(
                should_abort=lambda: self._abort,
                on_progress=lambda i, total, line: self.progress.emit(
                    i, total, ", ".join(f"{k}={v:g}"
                                        for k, v in sorted(line.fixed.items()))))
            self.done.emit(n, "")
        except Exception as e:
            self.done.emit(self.runner.lines_done, f"{type(e).__name__}: {e}")


class ScanWorker(QtCore.QThread):
    """
    Drive a field map: move, settle, average, repeat.

    Owns the carriers for the duration. The live stream has to be released and
    the boxes put back on their own clock before the first capture, exactly as
    SnapshotWorker does -- octobee_scan explains why a scan that quietly ran at
    the reduced live rate is worse than one that fails outright: it produces a
    map that looks entirely plausible and is several times noisier than the
    seconds-per-point setting implies.
    """

    done = QtCore.pyqtSignal(object, str)             # FieldMap, error
    progress = QtCore.pyqtSignal(int, int, str, float)  # i, n, where, sem_ut
    # The measurement itself, point by point: where the stage was and the
    # (16, 3) mT it read there. `progress` carries a formatted string and a
    # noise figure, which is all a progress bar needs and nothing a plot can
    # use -- and a scan that only reports its numbers at the end is a scan you
    # watch as a progress bar for eleven minutes and then find out about.
    # Separate signal rather than more arguments on `progress`, so nothing
    # already connected to it has to change.
    point = QtCore.pyqtSignal(dict, object)           # {axis: mm}, (16, 3) mT
    message = QtCore.pyqtSignal(str)

    def __init__(self, hosts, stages, grid, seconds, cal, settle_s,
                 restore_clkdiv=None, extra_meta=None):
        super().__init__()
        self.hosts = hosts
        self.stages = stages
        self.grid = grid
        self.seconds = seconds
        self.cal = cal
        self.settle_s = settle_s
        self.restore_clkdiv = dict(restore_clkdiv or {})
        self.extra_meta = dict(extra_meta or {})
        self._abort = False

    def abort(self):
        """Ask the scan to stop after the point in flight.

        Deliberately not an immediate stop: the capture running right now is
        already paid for, and finishing it costs seconds where discarding it
        loses a point from the map.
        """
        self._abort = True

    def run(self):
        fm = None
        try:
            if self.restore_clkdiv:
                self.message.emit("restoring the carriers' own clock")
                for h, prev in self.restore_clkdiv.items():
                    olive.restore_rate(h, prev)
                time.sleep(3.0)
            for h in self.hosts:
                if ob.stop_live_stream(h):
                    self.message.emit(f"{h}: released a running capture")

            def on_point(i, n, point, row, stats):
                where = " ".join(f"{k}={v:g}" for k, v in point.items())
                self.progress.emit(i, n, where,
                                   float(stats.get("sem_ut", float("nan"))))
                # A copy, because `row` is the scan's own array and this is
                # crossing into the GUI thread, where it will be read at some
                # later moment the scan knows nothing about.
                self.point.emit(dict(point), np.array(row, float))

            fm = oscan.run_scan(
                self.hosts, self.stages, self.grid, self.seconds, self.cal,
                settle_s=self.settle_s,
                progress=on_point,
                should_abort=lambda: self._abort,
                extra_meta=self.extra_meta,
                log=self.message.emit)
            self.done.emit(fm, "")
        except Exception as e:
            self.done.emit(fm, f"{type(e).__name__}: {e}")
