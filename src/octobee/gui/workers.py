"""Background threads. Anything that would otherwise block the event loop."""

import contextlib
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

# How many decoded blocks each carrier's reader may hold for the GUI thread.
STREAM_QUEUE_BLOCKS = 1024


class MapSaveWorker(QtCore.QThread):
    """Write a finished volume map without stopping the stream.

    The GUI thread is the one that drains the carriers' queues. Assembling
    and compressing a map of half a million rows takes several seconds, and
    done on that thread it overflowed both queues at the end of every long
    map -- 648 blocks, 6.6 s of stream, on 2026-11-19 -- a hole in the
    recording running beside it. numpy and zlib let go of the interpreter
    lock for the heavy parts, so on a thread of its own the tick carries on.

    `done` carries (written path or None, exception or None, kept), `kept`
    being where the rows went as .csv parts if the .npz could not be written.
    """

    done = QtCore.pyqtSignal(object, object, str)

    def __init__(self, log, path, log_hz, pose, sync_note):
        super().__init__()
        self.log, self.path, self.log_hz = log, path, log_hz
        self.pose, self.sync_note = pose, sync_note

    def run(self):
        log = self.log
        # The parts are a spare copy: whatever stops them, save anyway.
        with contextlib.suppress(Exception):
            log.finish_parts()
        try:
            written = log.save(self.path, self.log_hz, self.pose,
                               sync_note=self.sync_note)
        except Exception as exc:
            # The .csv parts are now the only copy on disk: put the last
            # rows in one too.
            kept = ""
            try:
                log.checkpoint(force=True)
                log.finish_parts()
                if log.parts_written:
                    kept = (f" The rows are also on disk as .csv parts in "
                            f"{os.path.dirname(log.parts_written[-1])}.")
            except Exception as part_exc:
                kept = (f" The last .csv part could not be written either "
                        f"({part_exc}).")
            self.done.emit(None, exc, kept)
            return
        self.done.emit(written, None, "")


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
                # ~10 s of stream per carrier (1024 x 2048 samples at
                # 200 kHz, ~160 kB a block, so ~170 MB at worst and nothing
                # while the GUI keeps up). 128 was 1.3 s: any stall of the GUI
                # thread longer than that -- a 3 s compression at the end of a
                # volume map was one -- dropped blocks from both carriers.
                st = ob.Streamer(h, lay, block_samples=self.block_samples,
                                 take_over=False, queue_depth=STREAM_QUEUE_BLOCKS)
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

    Steps one axis at a time along a span, standing still at each point long
    enough to read the counts and the controller's own position together. The
    counts arrive on the acquisition tick, so this thread never touches the
    stream: it asks for a reading through `counts_now`, which the tick keeps
    fresh.

    Moving ONE axis at a time is not politeness, it is the whole method -- the
    column that moved is the column that belongs to the axis that moved, and
    two at once makes that unanswerable.

    What it does, each of which changed a number on this rig or exists because
    a number could not be trusted:

      * it fits against the position the controller REPORTS at each stop, not
        the position it was asked for. An axis that ends 80 um short of its
        target puts that 80 um straight into the scale otherwise -- which is
        exactly what happened to x, at 0.41%;
      * it takes more than two points, so the fit has a residual;
      * it makes several passes, alternating direction through the points, so
        the scale comes with a spread. One run cannot say whether it would
        come out the same twice, and y did not: 1.1% apart between two runs;
      * it approaches every point from below, so backlash is a constant and
        not a slope, and the readings can be compared with a ruler later;
      * it waits for the counter to stop changing, not for a fixed time;
      * it asks whether one straight line describes the axis at all.

    It still compares the encoder with the controller, and both are on the
    same motor. What neither can tell is whether the stage travels what it
    says it does; EncoderReferenceWorker brings in a ruler for that.
    """

    done = QtCore.pyqtSignal(object, str)             # {axis: spec}, error
    message = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int, int)

    # Before reading, at least this long: the axis has to have stopped ringing.
    SETTLE_S = 0.6

    # Then wait until two consecutive fresh count rows agree to within this
    # many counts (a third of a micrometre at 14,400 counts/mm), giving up
    # after STABLE_TIMEOUT_S and counting the stop as unsettled.
    STABLE_COUNTS = 5.0
    STABLE_TIMEOUT_S = 3.0
    POLL_S = 0.05

    def __init__(self, stages, read_counts, span_mm=None, points=None,
                 axes=None, passes=None):
        super().__init__()
        self.stages = stages
        self.read_counts = read_counts
        self.span_mm = float(span_mm or oenc.CALIBRATION_SPAN_MM)
        self.points = max(2, int(points or oenc.CALIBRATION_POINTS))
        self.passes = max(1, int(passes or oenc.CALIBRATION_PASSES))
        self.axes = list(axes or oenc.AXES)
        self._abort = False
        self._steps_done = 0
        self._unsettled = 0
        self._steps_total = len(self.axes) * self.passes * self.points

    def abort(self):
        self._abort = True

    # ---- reading -----------------------------------------------------
    def _stable_counts(self):
        """A count row from a standstill, or None if no counts are arriving.

        The tick hands over a new array each time, so a row that is a
        different object from the last is a row taken since. Two of those that
        agree mean the counter has stopped. Giving up is safe in the sense that
        a stale row from a standing axis reads the same number -- but it is
        counted, so the run can say how many of its readings were not settled.
        """
        time.sleep(self.SETTLE_S)
        last = self.read_counts()
        if last is None:
            return None
        deadline = time.monotonic() + self.STABLE_TIMEOUT_S
        while True:
            row = self.read_counts()
            if row is None:
                return None
            if row is not last:
                gap = np.max(np.abs(np.asarray(row, dtype=float)
                                    - np.asarray(last, dtype=float)))
                if gap <= self.STABLE_COUNTS:
                    return row
                last = row
            if time.monotonic() >= deadline:
                self._unsettled += 1
                return row
            time.sleep(self.POLL_S)

    def _stand(self, st):
        """(position_mm, counts) at a standstill, or None if no counts came.

        Counts first, then the position, both with nothing moving -- the same
        order the sweep takes its datum in, so the two agree about what a
        simultaneous reading means.
        """
        counts = self._stable_counts()
        if counts is None:
            return None
        return float(st.position_mm), np.asarray(counts, dtype=float)

    # ---- geometry ----------------------------------------------------
    @staticmethod
    def _approach_for(lo, hi):
        return min(oenc.APPROACH_MM, 0.1 * max(hi - lo, 0.0))

    def _span_for(self, here, lo, hi):
        """The longest run of `span_mm` that fits, nearest to where it is.

        Centred on the current position where there is room, slid to fit
        where there is not, and shortened only when the axis genuinely has
        less travel than was asked for. A rig parked at the end of its travel
        calibrates rather than refusing. `lo` is the lowest TARGET, which is
        the soft limit plus the room needed to approach it from below.
        """
        span = max(0.0, min(self.span_mm, hi - lo))
        start = here - span / 2.0
        if start < lo:
            start = lo
        if start + span > hi:
            start = hi - span
        # min(): hi - span + span is not always hi in floating point, and a
        # target one part in 1e16 past the soft limit is refused like any
        # other.
        return start, min(start + span, hi)

    def _visit(self, st, target, lo, approach):
        """Go to `target`, arriving from below whatever side it started on."""
        pre = max(lo, target - approach)
        if pre < target - 1e-9:
            st.move_to(float(pre))
        st.move_to(float(target))

    def _run_pass(self, st, targets, lo, approach):
        """Visit `targets` in the order given, returning ([pos], [counts])."""
        pos, counts = [], []
        for target in targets:
            if self._abort:
                return None
            self._visit(st, float(target), lo, approach)
            stand = self._stand(st)
            self._steps_done += 1
            self.progress.emit(self._steps_done, self._steps_total)
            if stand is None:
                return None
            pos.append(stand[0])
            counts.append(stand[1])
        return pos, counts

    # ---- the run -----------------------------------------------------
    def _skip_progress(self):
        self._steps_done += self.passes * self.points
        self.progress.emit(self._steps_done, self._steps_total)

    def run(self):
        found, problems = {}, []
        try:
            self.stages.interlock.require_clear("an encoder calibration")
            for axis in self.axes:
                if self._abort:
                    break
                if axis not in self.stages.names:
                    continue
                st = self.stages[axis]
                lo, hi = st.limit_mm
                here = st.position_mm
                approach = self._approach_for(lo, hi)
                start, stop = self._span_for(here, lo + approach, hi)
                if stop - start < oenc.MIN_CALIBRATION_SPAN_MM:
                    problems.append(
                        f"{axis}: only {stop - start:g} mm of travel inside "
                        f"{lo:g}..{hi:g} mm, which is too short to fit over")
                    self._skip_progress()
                    continue
                if stop - start < self.span_mm - 1e-6:
                    self.message.emit(
                        f"encoders: {axis} has room for {stop - start:g} mm, "
                        f"not the {self.span_mm:g} mm asked for -- using it")

                targets = np.linspace(start, stop, self.points)
                passes = []
                for k in range(self.passes):
                    # Up the points, then down them, then up: the order of
                    # visiting is not the order of position, so a slow drift
                    # does not line up with a scale. Each is still approached
                    # from below.
                    order = targets if k % 2 == 0 else targets[::-1]
                    got = self._run_pass(st, order, lo, approach)
                    if got is None:
                        break
                    passes.append(got)
                # Back where it was found, whatever happened above -- except
                # after an abort. Whoever pressed stop wanted the axis to
                # stop, and this was a move right back across the span.
                if not self._abort:
                    st.move_to(here)
                if len(passes) < self.passes:
                    if not self._abort:
                        problems.append(
                            f"{axis}: no encoder counts arrived -- only "
                            f"acq1001_695 carries them, and only while the "
                            f"stream is running")
                    continue

                fits = [oenc.fit_axis(p, c) for p, c in passes]
                bad = next((f for f in fits if not f), None)
                if bad is not None:
                    problems.append(f"{axis}: {bad.why}")
                    continue
                if len({f.column for f in fits}) != 1:
                    problems.append(
                        f"{axis}: followed different columns on different "
                        f"passes ({sorted({f.column for f in fits})}), so "
                        f"none of it can be believed")
                    continue

                # The scale is the line through every reading of every pass.
                # Its spread is what the passes say about each other.
                pos = np.concatenate([p for p, _ in passes])
                cnt = np.vstack([c for _, c in passes])
                pooled = oenc.fit_axis(pos, cnt)
                if not pooled:
                    problems.append(f"{axis}: {pooled.why}")
                    continue
                scale = pooled.counts_per_mm
                per_pass = [f.counts_per_mm for f in fits]
                spread = (None if len(per_pass) < 2 else
                          (max(per_pass) - min(per_pass)) / abs(scale) * 1e6)
                bend = oenc.nonlinearity_um(pos, cnt[:, pooled.column])
                repeatable = (spread is None
                              or spread <= oenc.PASS_SPREAD_LIMIT_PPM)
                found[axis] = {
                    "column": int(pooled.column),
                    "counts_per_mm": float(scale),
                    "span_mm": round(pooled.span_mm, 3),
                    "points": self.points,
                    "passes": len(passes),
                    "residual_um": _round_or_none(pooled.residual_um, 2),
                    "worst_um": _round_or_none(pooled.worst_um, 2),
                    "pass_spread_ppm": _round_or_none(spread, 1),
                    "nonlinearity_um": _round_or_none(bend, 2),
                    "repeatable": bool(repeatable),
                }
                for k, f in enumerate(fits):
                    self.message.emit(
                        f"encoders: {axis} pass {k + 1}/{len(fits)} -> "
                        f"{f.describe()}")
                line = (f"encoders: {axis} = {scale:+,.1f} counts/mm"
                        + (f", passes agree to {spread:.0f} ppm"
                           if spread is not None else "")
                        + (f", {bend:.1f} um from straight"
                           if bend is not None else ""))
                self.message.emit(line)
                if not repeatable:
                    self.message.emit(
                        f"WARNING: encoders: {axis}: its passes disagree by "
                        f"{spread:.0f} ppm, more than the "
                        f"{oenc.PASS_SPREAD_LIMIT_PPM:.0f} ppm a measurement "
                        f"is allowed. This is not a repeatable scale.")
            if self._unsettled:
                self.message.emit(
                    f"WARNING: encoders: {self._unsettled} reading(s) were "
                    f"taken before the counter had stopped changing")
            self.progress.emit(self._steps_total, self._steps_total)
            self.done.emit(found, "; ".join(problems))
        except Exception as e:
            self.done.emit(found, f"{type(e).__name__}: {e}")


class EncoderReferenceWorker(EncoderCalibrationWorker):
    """Check one axis against a ruler, dial indicator or tape.

    Everything else compares the encoder with the controller. This asks for a
    length that is neither: it parks the axis at a series of points, each
    approached from below, and at every one asks the operator what the ruler
    reads. The question is blocking -- `ask` is emitted, the GUI shows a box,
    and the answer comes back through supply() -- so the axis stands still for
    as long as the reading takes.

    What it produces is fit_reference's answer: whether the controller's
    millimetre is a millimetre, and how many counts a true millimetre is.
    The zero of the ruler does not matter, and neither does its direction.
    """

    ask = QtCore.pyqtSignal(str, int, int, float)    # axis, i, n, controller mm

    def __init__(self, stages, read_counts, axis="x", span_mm=None,
                 points=None):
        super().__init__(stages, read_counts, span_mm=span_mm,
                         points=points or 7, axes=[axis], passes=1)
        self._reply = None
        self._event = threading.Event()

    def supply(self, value):
        """The ruler reading, in mm, or None to stop."""
        self._reply = value
        self._event.set()

    def abort(self):
        self._abort = True
        self._event.set()

    def run(self):
        axis = self.axes[0]
        try:
            self.stages.interlock.require_clear("a ruler check")
            if axis not in self.stages.names:
                self.done.emit(None, f"{axis}: no such stage")
                return
            st = self.stages[axis]
            lo, hi = st.limit_mm
            here = st.position_mm
            approach = self._approach_for(lo, hi)
            start, stop = self._span_for(here, lo + approach, hi)
            if stop - start < oenc.MIN_CALIBRATION_SPAN_MM:
                self.done.emit(None, f"{axis}: only {stop - start:g} mm of "
                                     f"travel, too short to check")
                return
            if stop - start < self.span_mm - 1e-6:
                self.message.emit(
                    f"ruler: {axis} has room for {stop - start:g} mm, not the "
                    f"{self.span_mm:g} mm asked for -- using it")
            tape, ctrl, counts = [], [], []
            n = self.points
            for i, target in enumerate(np.linspace(start, stop, n)):
                if self._abort:
                    break
                self._visit(st, float(target), lo, approach)
                stand = self._stand(st)
                if stand is None:
                    st.move_to(here)
                    self.done.emit(None, f"{axis}: no encoder counts arrived "
                                         f"-- only acq1001_695 carries them, "
                                         f"and only while the stream is "
                                         f"running")
                    return
                self._reply = None
                self._event.clear()
                # Looked at again after the clear: a stop that landed during
                # the move just made set the event, and clearing it would
                # otherwise throw the stop away and ask for a reading.
                if self._abort:
                    break
                self.ask.emit(axis, i, n, stand[0])
                self._event.wait()
                if self._abort or self._reply is None:
                    break
                tape.append(float(self._reply))
                ctrl.append(stand[0])
                counts.append(stand[1])
                self.progress.emit(i + 1, n)
            # Back where it was found only after a complete run. A stop must
            # not be answered with a move, and neither must a cancelled
            # reading: the question is a modal box, Esc closes it, and Esc is
            # also the emergency stop -- someone reaching for that would
            # otherwise start the axis back across the span.
            if not self._abort and len(tape) == n:
                st.move_to(here)
            if len(tape) < n:
                self.done.emit(None, f"stopped before every reading was "
                                     f"taken; {axis} was left where it is, "
                                     f"{float(st.position_mm):.3f} mm (it "
                                     f"started at {here:.3f})")
                return
            result = oenc.fit_reference(tape, ctrl, np.vstack(counts))
            if "why" in result:
                self.done.emit(None, f"{axis}: {result['why']}")
                return
            result["axis"] = axis
            result["tape_mm"] = tape
            result["controller_mm"] = ctrl
            self.done.emit(result, "")
        except Exception as e:
            self.done.emit(None, f"{type(e).__name__}: {e}")


def _round_or_none(x, places):
    return None if x is None else round(float(x), places)


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
