"""Background threads. Anything that would otherwise block the event loop."""

import os
import threading
import time

import numpy as np
from PyQt6 import QtCore

from octobee.acq import carrier as ob
from octobee import live as olive
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


def _is_running(worker):
    """True if this worker thread is alive.

    Tolerates None and a C++ object Qt has already deleted underneath us --
    both are "not running", and this is called from the stop path, which is
    the last place that should be able to raise.
    """
    if worker is None:
        return False
    try:
        return bool(worker.isRunning())
    except RuntimeError:
        return False


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
