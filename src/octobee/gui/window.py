#!/usr/bin/env python3
"""
octobee/gui/window.py -- control, live view, calibration and data export for the
16-sensor OCTO-BEE Hall probe, in one window.

    python octobee/gui/window.py                       # talk to the two carriers
    python octobee/gui/window.py --demo                # synthetic probe, no hardware
    python octobee/gui/window.py --replay capture.npz  # play back a saved capture

What it is for
--------------
The probe is 16 three-axis chips on a square tube, split across two carriers
that are configured differently and are not synchronised. Reading it through
Phoebus means one box at a time, PV names typed by hand, and a CSV round trip.
This does the whole job in one place: connect, watch, zero, cross-calibrate,
and write the data out.

The three things it insists on, because each of them has silently corrupted a
measurement on this bench already:

  * every sensor's own VCM is subtracted from that sensor's own axes;
  * amplitudes are compared as |B|, never as a single axis, because the chips
    point in 16 different directions;
  * per-channel health (railed, stuck, noisy) is always on screen, because one
    sensor is physically dead and the analogue path is noisier at one end of
    the concentrator than the other.
"""

import contextlib
import faulthandler
import argparse
import os
import queue
import sys
import threading
import time
import traceback
from collections import deque
from typing import ClassVar

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from octobee import paths
from octobee.acq import carrier as ob
from octobee.calib import convert as ocal
from octobee import help as ohelp
from octobee.calib import magnet as omag
from octobee import live as olive
from octobee import machine as omach
from octobee import profile as oprof
from octobee.calib import roll as opc
from octobee import record as orec
from octobee.motion import scan as oscan
from octobee.motion import stage as ostage
from octobee.calib import geometry as pgeom
from octobee.gui.widgets.machine3d import MachineView3D
from octobee.gui.widgets.probe3d import ProbeView3D, color_for

N_SENSORS = pgeom.N_SENSORS
AXES = ("Bx", "By", "Bz")
OUT_RATES = (100.0, 200.0, 500.0, 1000.0, 2000.0)
# Reducing the ADC clock is NOT free. The SENM3Dx analog low-pass sits at
# 100 kHz (PWM_CTRL bits 5:4, already at its narrowest setting), so sampling
# below 200 kSPS folds noise from 0-100 kHz into 0-fs/2. The density penalty is
# sqrt(100kHz / (fs/2)) -- measured 3.1x at 20 kSPS, matching the predicted 3.16x.
# You trade noise for stream bandwidth; the label states the cost.
STREAM_RATES = {"leave the box alone (recommended)": 0.0,
                "200 kSPS (no aliasing, best noise)": 200000.0,
                "50 kSPS (2.0x noise, aliased)": 50000.0,
                "20 kSPS (3.2x noise, aliased)": 20000.0}

# The per-channel health scan is the most expensive thing in the refresh loop
# and its answer changes only when a connector does, so it runs on its own
# slower clock over a short window. Left on the display clock over the full
# history it starves the reader threads and the carriers' queues overflow.
# Redraw rate for the live view. The acquisition tick is independent of this,
# so turning it down costs you smoothness and nothing else -- no samples, no
# recorded data. 10 Hz is already far beyond what a hand-passed magnet needs.
VIEW_RATES = (2.0, 5.0, 10.0, 20.0)
DEFAULT_VIEW_HZ = 10.0
MAX_VIEW_INTERVAL_MS = 2000        # slowest the automatic backoff will go

HEALTH_PERIOD_S = 2.0
HEALTH_WINDOW_S = 1.0
RAW_HISTORY_S = 5.0

# Antialiasing off, and every plot pen exactly 1 pixel wide. This is not a
# cosmetic preference, it is the difference between a usable application and an
# unusable one. Qt strokes a cosmetic pen of non-integer width through a
# completely different and vastly slower path: measured on a 20 s window of 15
# traces, a single repaint took 39 SECONDS at width 1.6 with antialiasing, and
# 74 ms at width 1. Turning antialiasing off as well brings it to 45 ms. The
# symptom is a live plot that appears to hang the moment real data arrives,
# with the cost invisible to any timing of our own code because it happens
# inside Qt's paint.
PLOT_PEN_WIDTH = 1
# Points handed to each curve, as a multiple of the plot's width in pixels.
# 0.5 gives one min/max pair per ~4 pixels, which still renders a hand-passed
# magnet spike over many bins while costing a third of what 1.0 does. Raise it
# if you need finer structure on screen and can afford the repaint.
PLOT_TARGET_MULT = 0.5
pg.setConfigOptions(antialias=False, background=(18, 20, 26),
                    foreground=(210, 214, 222))


class ProfiledPlot(pg.PlotWidget):
    """
    A PlotWidget that times its own repaint.

    Without this the table has a hole in it: asking a curve to setData is
    cheap, and the expensive part -- Qt actually rasterising 16 or 48
    polylines -- happens later in the event loop, where it would show up only
    as unexplained lag. Timing it here means every row of the profile adds up
    to something, and "none of these is big but the loop still stalls" becomes
    a real conclusion rather than a gap in the measurement.
    """

    def __init__(self, label, profiler=None, **kw):
        super().__init__(**kw)
        self._label = label
        self.profiler = profiler or oprof.Profiler(enabled=False)

    def paintEvent(self, ev):
        with self.profiler.time(self._label):
            super().paintEvent(ev)


def sensor_colors():
    return [pg.mkColor([int(255 * c) for c in color_for(i / (N_SENSORS - 1))[:3]])
            for i in range(N_SENSORS)]


# ==========================================================================
# data sources
# ==========================================================================

class SourceBase:
    """Common interface: read() returns per-box (n, 32) count arrays, equal n."""
    hosts = ()
    vpc = ()
    # Volts at ADC count 0, per box. Zero on every bipolar range; carried so a
    # unipolar range converts correctly rather than reading half a span low.
    volt_offset = ()
    fs_hz = 20000.0
    live = False

    def read(self):
        return None

    def temperatures(self):
        return np.full(N_SENSORS, np.nan)

    def stats(self):
        return {}

    def stop(self):
        pass


class LiveSource(SourceBase):
    """Reads both carriers through octobee.Streamer and keeps them aligned."""

    live = True

    def __init__(self, hosts, layouts, streamers):
        self.hosts = list(hosts)
        self.layouts = list(layouts)
        self.streamers = list(streamers)
        self.vpc = [lay.volts_per_count for lay in layouts]
        self.volt_offset = [lay.volt_offset for lay in layouts]
        self.fs_hz = float(layouts[0].fs_hz)
        self._pending = {h: deque() for h in self.hosts}
        self._n = dict.fromkeys(self.hosts, 0)
        self._temp = {h: np.zeros(8, dtype=np.uint32) for h in self.hosts}
        self.gaps = 0
        self.lost = 0
        self.error = None
        self._discard_startup_backlog()
        self.t0 = time.time()

    def _discard_startup_backlog(self):
        """
        Throw away whatever queued up before anyone was consuming, and zero the
        counters.

        The reader threads start as soon as each carrier is ready, but the two
        come up several seconds apart and the GUI only begins draining once
        both are live. In that gap the first box fills its queue and starts
        shedding blocks -- around 70 of them, every single session. None of it
        matters: nothing is being recorded or displayed yet. But it left
        "dropped blocks" showing a permanent non-zero count, which is precisely
        the number that is supposed to mean "the data you are recording has
        holes in it". A warning that is always on is not a warning.
        """
        for st in self.streamers:
            while True:
                try:
                    st.q.get_nowait()
                except queue.Empty:
                    break
            st.dropped = 0
            st.bytes_read = 0

    def read(self):
        for h, st in zip(self.hosts, self.streamers):
            if st.error and self.error is None:
                self.error = f"{h}: {st.error}"
            for blk in st.get_all():
                if blk is None:
                    continue
                self._pending[h].append(blk["ai"])
                self._n[h] += blk["ai"].shape[0]
                if "temp_raw" in blk:
                    self._temp[h] = blk["temp_raw"]
                gaps, lost = ob.check_continuity(blk["sam_cnt"])
                self.gaps += gaps
                self.lost += lost
        n = min(self._n.values()) if self._n else 0
        if n <= 0:
            return None
        out = []
        for h in self.hosts:
            buf = np.concatenate(self._pending[h], axis=0)
            out.append(buf[:n])
            rest = buf[n:]
            self._pending[h].clear()
            if rest.shape[0]:
                self._pending[h].append(rest)
            self._n[h] = rest.shape[0]
        return out

    def temperatures(self):
        return ocal.temperatures_c([self._temp[h] for h in self.hosts])

    def stats(self):
        total = sum(st.bytes_read for st in self.streamers)
        dt = max(time.time() - self.t0, 1e-3)
        return {"MB/s": total / dt / 1e6,
                "gaps": self.gaps, "lost": self.lost,
                "dropped blocks": sum(st.dropped for st in self.streamers)}

    def stop(self):
        for st in self.streamers:
            st.stop()


class ReplaySource(SourceBase):
    """Plays a saved .npz back at wall-clock speed, looping."""

    def __init__(self, path, speed=1.0):
        cap = ocal.load_capture(path)
        self.hosts = cap["hosts"]
        self.ai = cap["ai"]
        self.vpc = cap["vpc"]
        self.volt_offset = cap.get("volt_offset", [0.0] * len(self.vpc))
        self.fs_hz = float(cap["fs_hz"][0])
        self._temp = cap["temp_raw"]
        self.speed = float(speed)
        self.n = min(a.shape[0] for a in self.ai)
        self.pos = 0
        self.t_last = time.time()
        self.path = path

    def read(self):
        now = time.time()
        want = int((now - self.t_last) * self.fs_hz * self.speed)
        self.t_last = now
        if want <= 0:
            return None
        want = min(want, self.n)
        end = self.pos + want
        if end <= self.n:
            out = [a[self.pos:end] for a in self.ai]
            self.pos = end % self.n
        else:                                     # wrap
            k = end - self.n
            out = [np.concatenate([a[self.pos:self.n], a[:k]]) for a in self.ai]
            self.pos = k
        return out

    def temperatures(self):
        return ocal.temperatures_c(self._temp)

    def stats(self):
        return {"replay": os.path.basename(self.path),
                "position": f"{self.pos/self.fs_hz:.1f}s"}


class DemoSource(SourceBase):
    """
    Synthetic probe, for working on the GUI with no hardware attached.

    A dipole flies along the tube past one face after another, so the 3D view,
    the |B| comparison and the geometry weighting all get exercised. It
    deliberately reproduces two real faults of this probe: S16 railed, and a
    noisy analogue path at the far end of each concentrator.
    """

    def __init__(self, geom, fs_hz=20000.0, b_range_mt=20.0):
        self.geom = geom
        self.hosts = ["demo_694", "demo_695"]
        self.fs_hz = float(fs_hz)
        # Through a real Layout rather than by hand, so the synthetic source
        # converts counts exactly the way the carriers' own frames do.
        lay = ob.Layout(ssb=96, fs_hz=self.fs_hz, adc_range="+/-10V")
        self.vpc = [lay.volts_per_count] * 2
        self.volt_offset = [lay.volt_offset] * 2
        self.v_per_t = ob.RANGE_TO_VPT[b_range_mt]
        self.t = 0.0
        self.t_last = time.time()
        self.vcm_v = 2.20 + np.linspace(0, 0.09, N_SENSORS)
        # Mirrors the measured pattern: quiet at the start of each concentrator,
        # progressively noisier toward the far end.
        self.noise_counts = np.concatenate([np.linspace(0.6, 13, 8),
                                            np.linspace(0.8, 16, 8)])
        self.rng = np.random.default_rng(7)
        self._pos = geom.positions()
        self._rot = geom.rotations()

    def _field_mt(self, t):
        """
        Dipole flying past the probe, seen in each chip's own axes.

        Vectorised over time: `t` may be an array, and the whole (k, 16, 3)
        block comes back in one go. Doing this one instant at a time rebuilt
        the sensor positions and rotation matrices on every sample, which made
        the synthetic source cost far more than the real one and turned the
        profile of --demo into a measurement of itself.
        """
        t = np.atleast_1d(np.asarray(t, float))
        z = (t * 60.0) % (self.geom.tube_length_mm + 120.0) - 60.0
        ang = 2 * np.pi * t / 11.0
        # Fly past the chips, which sit at the arm tips -- not past the tube.
        r = self.geom.fsv_radius_mm + 25.0
        src = np.stack([r * np.cos(ang), r * np.sin(ang), z], axis=1)  # (k,3)
        m = np.array([0.0, 0.0, 6.0e4])                # arbitrary, mT*mm^3
        d = self._pos[None, :, :] - src[:, None, :]    # (k,16,3)
        rr = np.maximum(np.linalg.norm(d, axis=2, keepdims=True), 8.0)
        rhat = d / rr
        b_tube = (3.0 * rhat * (rhat @ m)[:, :, None] - m) / rr ** 3
        return np.einsum("sji,ksj->ksi", self._rot, b_tube)   # tube -> chip

    def read(self):
        now = time.time()
        n = int((now - self.t_last) * self.fs_hz)
        if n <= 0:
            return None
        n = min(n, int(self.fs_hz))
        t = self.t + np.arange(n) / self.fs_hz
        self.t_last, self.t = now, t[-1]

        # The magnet is slow, so evaluate on a coarse grid and interpolate up.
        k = max(2, n // 256)
        tk = t[::k]
        bk = self._field_mt(tk)                                   # (k,16,3)
        b = np.empty((n, N_SENSORS, 3))
        for s in range(N_SENSORS):
            for a in range(3):
                b[:, s, a] = np.interp(t, tk, bk[:, s, a])

        counts = np.empty((n, N_SENSORS, 4))
        for s in range(N_SENSORS):
            vcm_counts = self.vcm_v[s] / self.vpc[0]
            for a in range(3):
                v = b[:, s, a] * 1e-3 * self.v_per_t
                counts[:, s, a] = vcm_counts + v / self.vpc[0]
            counts[:, s, 3] = vcm_counts
            counts[:, s, :] += self.rng.normal(0, self.noise_counts[s], (n, 4))

        counts[:, 15, :] = -32768.0                    # S16 railed, as on the bench
        counts[:, 15, 2] = self.vcm_v[15] / self.vpc[0]   # ...except one channel

        # back to wire order Bz, By, Bx, VCM
        wire = counts[:, :, [2, 1, 0, 3]]
        wire = np.clip(np.round(wire), -32768, 32767).astype(np.int16)
        flat = wire.reshape(n, N_SENSORS * 4)
        return [flat[:, :32], flat[:, 32:]]

    def temperatures(self):
        return 20.0 + np.linspace(-2, 3, N_SENSORS)

    def stats(self):
        return {"mode": "DEMO -- synthetic data, no hardware"}


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


# ==========================================================================
# small widgets
# ==========================================================================

class Rolling:
    """Fixed-length rolling window of (n, 16, 3) millitesla."""

    def __init__(self, npoints):
        self.buf = np.zeros((npoints, N_SENSORS, 3), dtype=np.float32)
        self.n = npoints
        self.filled = 0

    def resize(self, npoints):
        if npoints == self.n:
            return
        new = np.zeros((npoints, N_SENSORS, 3), dtype=np.float32)
        k = min(npoints, self.filled)
        if k:
            new[-k:] = self.buf[-k:]
        self.buf, self.n, self.filled = new, npoints, k

    def clear(self):
        self.filled = 0

    def push(self, block):
        k = block.shape[0]
        if k >= self.n:
            self.buf[:] = block[-self.n:]
            self.filled = self.n
        else:
            self.buf[:-k] = self.buf[k:]
            self.buf[-k:] = block
            self.filled = min(self.n, self.filled + k)

    def view(self):
        return self.buf[self.n - self.filled:]


NO_AXIS = "(none)"

METHOD_FULL = "Full — plane sweep and standoff dither  (recommended)"
METHOD_AXIAL = "Axial only — one sweep per pose  (quick)"
METHOD_CUSTOM = "Custom — set the axes by hand"

# What each method is for, in the words someone choosing between them needs.
# Deliberately includes what each one CANNOT do: the axial run is not a
# degraded version of the full one, it is the right answer when there is only
# one stage, and the numbers are what make that judgeable rather than a matter
# of picking the option with "recommended" after it.
METHOD_NOTES = {
    "full": (
        "Three passes per pose: find the rings, cut across each one, then "
        "dither the standoff. Every sensor is measured at the top of its own "
        "peak and at a measured distance, so a millimetre of arm placement "
        "stops looking like 15 % of gain. On a synthetic probe misplaced by "
        "1 mm on all three axes this recovers gain to 1.8 %, against 9.9 % "
        "for the axial run. Needs three stages, and about five times the "
        "points."),
    "axial": (
        "One sweep along the tube per pose — the original routine. Every "
        "sensor still passes the same fixed magnet, so the trim still needs "
        "no 1/r³ model, but each chip is measured wherever its arm happens to "
        "have put it: 1 mm of misplacement is up to 15 % of trim at a 20 mm "
        "standoff. The right choice when only the tube axis is motorised, or "
        "for a quick check."),
    "custom": (
        "The axis boxes below are yours. One combination is not offered above "
        "and is worth knowing about: a transverse cut with NO standoff dither "
        "is worse than doing neither, because peaking over the plane moves "
        "every chip to its true, nearer approach and amplifies whatever error "
        "is left. The run will refuse a dither without a cut for the "
        "matching reason — the dither's model is only true once the cut has "
        "put the chip under the magnet."),
}

# What each pass is doing, for the header while it runs. The names matter more
# than they look: the three passes take very different lengths of time and an
# operator who cannot tell which one is running cannot tell a slow dither from
# a stalled stage.
PASS_NAMES = {
    "locate": "pass A, finding the rings",
    "cut": "pass B, cutting across each ring",
    "dither": "pass C, measuring the standoff",
}


class MagnetWizard(QtWidgets.QDialog):
    """Guided single-magnet calibration: four sweeps, one per quarter turn.

    The routine is in octobee/calib/magnet.py; this is the part that has to be a
    dialog, because between poses the instrument cannot do anything until a
    person has turned the head and said so. That pause is the whole reason
    this is guided rather than a button: the run is only valid if the magnet
    and the cradle stay put across all four poses, and nothing in software can
    check that.

    Each pose is an ordinary one-axis scan, run through the same ScanWorker as
    a field map -- move, settle, average at the full 200 kSPS, repeat -- so the
    per-point noise argument is identical and there is one mover, not two.
    """

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Guided magnet calibration")
        self.setModal(False)
        self.resize(760, 640)
        self.run = omag.MagnetRun(axis="y")
        self._worker = None
        self._t0 = 0.0
        self._start_mm = None
        self._finished = False
        self._park = {}          # across/normal positions, taken once
        self._setting_method = False   # guards the method -> axis-box writes
        self._pass = None        # 'locate' | 'cut' | 'dither'
        self._pending = None     # the PoseSweep the passes are building up
        self._rings = None       # where pass A found this pose's four rings
        self._released = False   # have the carriers been taken off live yet

        lay = QtWidgets.QVBoxLayout(self)
        self.lbl_step = QtWidgets.QLabel()
        f = self.lbl_step.font()
        f.setPointSize(f.pointSize() + 3)
        f.setBold(True)
        self.lbl_step.setFont(f)
        lay.addWidget(self.lbl_step)

        self.txt = QtWidgets.QTextBrowser()
        self.txt.setMaximumHeight(250)
        lay.addWidget(self.txt)

        names = list(win.stages.names) if win.stages else ["y"]

        # ---- which run this is ---------------------------------------------
        # A named choice rather than "set these two axis boxes and hope",
        # because the passes are not independent options: plane-without-dither
        # is measurably worse than doing neither, and it was previously
        # reachable by setting one combo and not the other. The methods here
        # are the combinations that are actually worth running.
        self.box_method = QtWidgets.QGroupBox("Which run")
        ml = QtWidgets.QVBoxLayout(self.box_method)
        self.cmb_method = QtWidgets.QComboBox()
        for label, tag in ((METHOD_FULL, "full"),
                           (METHOD_AXIAL, "axial"),
                           (METHOD_CUSTOM, "custom")):
            self.cmb_method.addItem(label, tag)
        self.lbl_method = QtWidgets.QLabel()
        self.lbl_method.setWordWrap(True)
        ml.addWidget(self.cmb_method)
        ml.addWidget(self.lbl_method)
        lay.addWidget(self.box_method)

        self.box_setup = QtWidgets.QGroupBox("Sweep")
        gl = QtWidgets.QGridLayout(self.box_setup)

        # ---- pass A: the axial locate
        self.cmb_axis = QtWidgets.QComboBox()
        self.cmb_axis.addItems(names)
        if "y" in names:
            self.cmb_axis.setCurrentText("y")
        self.cmb_axis.setToolTip(
            "The axis the tube lies along. The whole argument for this "
            "routine is that every sensor passes the magnet at the same "
            "approach, which is only true along the tube axis.")
        span, step = omag.suggested_sweep(win.geom)
        self.spin_span = QtWidgets.QDoubleSpinBox()
        self.spin_span.setRange(10.0, 300.0)
        self.spin_span.setValue(span)
        self.spin_span.setSuffix(" mm")
        self.spin_span.setToolTip(
            "How far to drive from where the head is parked now. Long enough "
            "to carry every ring past the magnet and out the other side.")
        self.spin_step = QtWidgets.QDoubleSpinBox()
        self.spin_step.setRange(0.1, 50.0)
        self.spin_step.setValue(step)
        self.spin_step.setSuffix(" mm")
        self.spin_step.setToolTip(
            "Coarse on purpose. This pass only has to find WHERE the four "
            "rings are; the transverse cut below is what measures them.")
        self.spin_secs = QtWidgets.QDoubleSpinBox()
        self.spin_secs.setRange(0.2, 30.0)
        self.spin_secs.setValue(1.0)
        self.spin_secs.setSuffix(" s")
        self.spin_secs.setToolTip(
            "Averaging time per point. The peak needs little, but the "
            "standoff dither reads a CURVATURE out of seven points and that "
            "is where the noise actually costs something -- if the standoff "
            "column comes back mostly '--', raise this before anything else.")
        self.spin_settle = QtWidgets.QDoubleSpinBox()
        self.spin_settle.setRange(0.0, 10.0)
        self.spin_settle.setValue(oscan.DEFAULT_SETTLE_S)
        self.spin_settle.setSuffix(" s")

        # ---- the standoff, which sizes both of the other passes
        self.spin_standoff = QtWidgets.QDoubleSpinBox()
        self.spin_standoff.setRange(3.0, 200.0)
        self.spin_standoff.setValue(20.0)
        self.spin_standoff.setSuffix(" mm")
        self.spin_standoff.setToolTip(
            "Roughly how far the magnet is from the chips. Nothing is "
            "measured with this -- pass C measures the real distance, "
            "per sensor -- but both of the other passes have to be SIZED "
            "from it:\n\n"
            "  - the transverse cut needs a half-span of about one standoff, "
            "because that is the width of the peak it is looking for. Much "
            "less and the cut is flat and the peak cannot be placed; much "
            "more and the extra points buy nothing.\n"
            "  - the dither needs a quarter of it, for the same reason in "
            "reverse.\n\n"
            "Within a factor of two is close enough. Changing it resizes "
            "both.")

        for col, (lbl, wdg) in enumerate((("tube axis", self.cmb_axis),
                                          ("standoff ~", self.spin_standoff),
                                          ("sweep", self.spin_span),
                                          ("step", self.spin_step),
                                          ("per point", self.spin_secs),
                                          ("settle", self.spin_settle))):
            gl.addWidget(QtWidgets.QLabel(lbl), 0, col)
            gl.addWidget(wdg, 1, col)

        # ---- pass B: the transverse cut
        self.cmb_across = QtWidgets.QComboBox()
        self.cmb_across.addItem(NO_AXIS)
        self.cmb_across.addItems([n for n in names if n != "y"])
        if "x" in names:
            self.cmb_across.setCurrentText("x")
        self.cmb_across.setToolTip(
            "The transverse axis -- across the face, the way the arms reach. "
            "Sweeping it as well as the tube axis is what lets each sensor be "
            "measured at ITS OWN peak instead of at a slice through it.\n\n"
            "Set to none and this is the old one-axis routine: still valid, "
            "but a millimetre of arm placement then costs up to 15% of trim "
            "if the magnet is not exactly on the chip line.")
        self.spin_across_half = QtWidgets.QDoubleSpinBox()
        self.spin_across_half.setRange(1.0, 100.0)
        self.spin_across_half.setSuffix(" mm")
        self.spin_across_step = QtWidgets.QDoubleSpinBox()
        self.spin_across_step.setRange(0.05, 20.0)
        self.spin_across_step.setSuffix(" mm")

        # ---- pass C: the standoff dither
        self.cmb_normal = QtWidgets.QComboBox()
        self.cmb_normal.addItem(NO_AXIS)
        self.cmb_normal.addItems([n for n in names if n != "y"])
        if "z" in names:
            self.cmb_normal.setCurrentText("z")
        self.cmb_normal.setToolTip(
            "The standoff axis -- toward the magnet. This one is NOT swept "
            "for a peak, because |B| has no maximum in that direction; it "
            "just keeps rising as the chip gets closer. A few points along it "
            "measure the distance itself, which is the last first-order error "
            "in the trim and the one the plane cannot touch.\n\n"
            "Set to none to skip it. The run is still better than a bare "
            "axial sweep, but each chip's distance stays assumed.")
        self.spin_dither_half = QtWidgets.QDoubleSpinBox()
        self.spin_dither_half.setRange(0.2, 50.0)
        self.spin_dither_half.setSuffix(" mm")
        self.spin_dither_half.setToolTip(
            "Half-span of the dither. Bigger than instinct says: the standoff "
            "comes out of the CURVATURE of the field along this axis, which "
            "is second order, so a +-1 mm dither at a 20 mm standoff buries "
            "it 0.4% under the slope and the fit returns noise.\n\n"
            "The near end of the dither is also the closest the chips get to "
            "the magnet. At a quarter of the standoff that is 2.4x the peak "
            "field -- keep the peak under about a third of the range, or the "
            "dither clips and the fit reads a flat top.")
        self.spin_dither_pts = QtWidgets.QSpinBox()
        self.spin_dither_pts.setRange(3, 21)
        self.spin_dither_pts.setValue(omag.DITHER_POINTS)
        self.spin_dither_pts.setToolTip(
            "Three is the minimum that can separate a slope from a curvature, "
            "and it leaves nothing over to judge the fit by. Seven is where "
            "the bench stopped improving.")

        for col, (lbl, wdg) in enumerate(
                (("transverse axis", self.cmb_across),
                 ("cut half-span", self.spin_across_half),
                 ("cut step", self.spin_across_step),
                 ("standoff axis", self.cmb_normal),
                 ("dither half-span", self.spin_dither_half),
                 ("dither points", self.spin_dither_pts))):
            gl.addWidget(QtWidgets.QLabel(lbl), 2, col)
            gl.addWidget(wdg, 3, col)

        self._resize_passes()
        self.spin_standoff.valueChanged.connect(self._resize_passes)

        self.lbl_points = QtWidgets.QLabel("")
        self.lbl_points.setWordWrap(True)
        gl.addWidget(self.lbl_points, 4, 0, 1, 6)
        for sp in (self.spin_span, self.spin_step, self.spin_secs,
                   self.spin_settle, self.spin_across_half,
                   self.spin_across_step, self.spin_dither_half):
            sp.valueChanged.connect(self._update_estimate)
        self.spin_dither_pts.valueChanged.connect(self._update_estimate)
        lay.addWidget(self.box_setup)

        # Wired after the axis combos exist, and in this order: the method sets
        # the combos, and the combos falling back to Custom is how a hand edit
        # is noticed. Connecting the second one first would make _apply_method's
        # own writes look like hand edits and knock it straight to Custom.
        self.cmb_method.currentIndexChanged.connect(self._apply_method)
        # The tube axis feeds into which axes the method may use, so changing
        # it has to re-derive them -- otherwise picking x as the tube axis
        # leaves the transverse cut still pointed at x.
        self.cmb_axis.currentTextChanged.connect(self._apply_method)
        for cb in (self.cmb_across, self.cmb_normal):
            cb.currentTextChanged.connect(self._method_edited)
        self._method_default(names)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setMinimumHeight(22)
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        self.bar.setFormat("no sweep running")
        lay.addWidget(self.bar)

        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setFont(QtGui.QFont("Consolas", 9))
        self.report.setPlaceholderText(
            "After the first pose, this fills with each sensor's peak, where "
            "along the sweep it happened, and the trim it implies.\n\n"
            "It is rewritten after every pose, so you can see the faces "
            "arrive one at a time — and stop early if something is obviously "
            "wrong rather than paying for all four.")
        lay.addWidget(self.report, 1)

        row = QtWidgets.QHBoxLayout()
        self.chk_apply = QtWidgets.QCheckBox(
            "apply the gain trim and save it to calibration.json")
        self.chk_apply.setChecked(True)
        self.chk_apply.setToolTip(
            "Folds the measured per-sensor response into the calibration's "
            "gain trim -- without any geometry weighting, because every "
            "sensor was measured at the top of its own peak and scaled to a "
            "common, measured standoff.\n\nIt SAVES as well as applies. An hour of "
            "measurement that only exists in memory until someone remembers "
            "to press Save calibration is an hour waiting to be lost.")
        row.addWidget(self.chk_apply)
        row.addStretch(1)
        self.btn_primary = QtWidgets.QPushButton()
        self.btn_primary.clicked.connect(self.on_primary)
        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        row.addWidget(self.btn_primary)
        row.addWidget(self.btn_close)
        lay.addLayout(row)

        self._update_estimate()
        self._refresh()

    # ---- state -----------------------------------------------------------
    @property
    def busy(self):
        return self._worker is not None and self._worker.isRunning()

    # ---- which run ------------------------------------------------------
    def _method(self):
        return self.cmb_method.currentData()

    def _method_default(self, names):
        """Pick the best method this rig can actually run, and say so.

        Full needs three axes. Offering it on a one-stage rig and then
        refusing at Start is the failure this combo exists to remove, so an
        unrunnable method is disabled in the list rather than left as a trap.
        """
        # One rule for "can this rig run the full method", shared with
        # _apply_method: two axes free of the tube axis. Counting stages
        # separately here would let the two disagree the moment the tube axis
        # moved, and the one that decides what actually gets scanned is the
        # other one.
        can_full = self._full_axes() is not None
        item = self.cmb_method.model().item(0)
        item.setEnabled(can_full)
        if not can_full:
            self.cmb_method.setItemText(
                0, METHOD_FULL.replace("(recommended)",
                                       "(needs three stages)"))
        self.cmb_method.setCurrentIndex(0 if can_full else 1)
        self._apply_method()

    def _full_axes(self):
        """(transverse, standoff) for the full run, or None if it cannot run.

        Picked against the tube axis rather than hardcoded, because the tube
        axis is a combo the operator can change. Sweeping x and cutting across
        x is the same axis twice: the grid is keyed by axis name, so the two
        collapse into one and the run silently becomes something other than
        what the panel says -- which is the failure mode this whole routine is
        built to avoid, arriving through the door marked convenience.
        """
        sweep = self.cmb_axis.currentText()
        free = [self.cmb_across.itemText(i)
                for i in range(self.cmb_across.count())]
        free = [n for n in free if n not in (NO_AXIS, sweep)]
        if len(free) < 2:
            return None
        # This rig's convention when it is available -- x across the face, z
        # toward the magnet -- and otherwise just two distinct axes.
        across = "x" if "x" in free else free[0]
        rest = [n for n in free if n != across]
        return (across, "z" if "z" in rest else rest[0])

    def _apply_method(self):
        """Drive the axis boxes from the method, and lock them unless Custom."""
        method = self._method()
        full = self._full_axes()
        # The tube axis can change under a chosen method and leave it with
        # nowhere to put the other two passes. Say so by disabling it, the
        # same way a rig with too few stages does, rather than by quietly
        # running a different scan.
        self.cmb_method.model().item(0).setEnabled(full is not None)
        if method == "full" and full is None:
            self.cmb_method.setCurrentIndex(1)      # falls back to axial
            return
        self.lbl_method.setText(METHOD_NOTES.get(method, ""))
        custom = method == "custom"
        for w in (self.cmb_across, self.cmb_normal):
            w.setEnabled(custom)
        if not custom:
            want = {"full": full, "axial": (NO_AXIS, NO_AXIS)}[method]
            self._setting_method = True
            try:
                for cb, name in zip((self.cmb_across, self.cmb_normal), want):
                    if cb.findText(name) >= 0:
                        cb.setCurrentText(name)
            finally:
                self._setting_method = False
        # Sizing only matters to the passes that exist; hiding it would move
        # the layout about, so it greys out instead.
        for w in (self.spin_across_half, self.spin_across_step):
            w.setEnabled(self._across_name() is not None)
        for w in (self.spin_dither_half, self.spin_dither_pts,
                  self.spin_standoff):
            w.setEnabled(self._normal_name() is not None
                         or self._across_name() is not None)
        self._update_estimate()

    def _method_edited(self):
        """A hand edit to an axis box means this is no longer a named method."""
        if getattr(self, "_setting_method", False):
            return
        if self._method() != "custom":
            self.cmb_method.setCurrentIndex(self.cmb_method.count() - 1)
        else:
            self._apply_method()

    def _across_name(self):
        n = self.cmb_across.currentText()
        return None if n == NO_AXIS else n

    def _normal_name(self):
        n = self.cmb_normal.currentText()
        return None if n == NO_AXIS else n

    def _resize_passes(self):
        """Re-size the cut and the dither from the nominal standoff.

        Both of them have a natural size set by the distance to the magnet and
        no natural size otherwise, so leaving the operator to pick millimetres
        out of the air is how a run comes back with a flat cut and an
        unfittable dither. They stay editable; this only moves them when the
        standoff changes.
        """
        d = self.spin_standoff.value()
        half, step = omag.suggested_plane(d)
        self.spin_across_half.setValue(half)
        self.spin_across_step.setValue(step)
        self.spin_dither_half.setValue(float(omag.suggested_dither(d)[-1]))

    def _positions(self):
        start = self._start_mm
        if start is None:
            start = self.win.stages[self.cmb_axis.currentText()].position_mm
        return oscan.parse_axis_spec(
            f"{start}:{start + self.spin_span.value()}:{self.spin_step.value()}")

    def _across_offsets(self):
        half, step = self.spin_across_half.value(), self.spin_across_step.value()
        return oscan.parse_axis_spec(f"{-half}:{half}:{step}")

    def _dither_offsets(self):
        half = self.spin_dither_half.value()
        return np.linspace(-half, half, int(self.spin_dither_pts.value()))

    def _pass_sizes(self):
        """(locate, cut, dither) point counts for one pose."""
        rings = pgeom.SENSORS_PER_FACE
        # Through parse_axis_spec rather than span/step + 1, so the estimate is
        # the number of points the scan will actually visit: the spec drops a
        # final point that floating point put past the stop, and an estimate
        # that is one out every time trains you to distrust it.
        n_loc = len(oscan.parse_axis_spec(
            f"0:{self.spin_span.value()}:{self.spin_step.value()}"))
        n_cut = rings * len(self._across_offsets()) if self._across_name() else 0
        n_dit = rings * int(self.spin_dither_pts.value()) \
            if self._normal_name() else 0
        return n_loc, n_cut, n_dit

    def _update_estimate(self):
        n_loc, n_cut, n_dit = self._pass_sizes()
        n = n_loc + n_cut + n_dit
        per = self.spin_secs.value() + self.spin_settle.value() + 2.0
        bits = [f"locate {n_loc}"]
        bits.append(f"cut {n_cut}" if n_cut else "no transverse cut")
        bits.append(f"dither {n_dit}" if n_dit else "no standoff dither")
        self.lbl_points.setText(
            f"{' + '.join(bits)} = {n} points per pose, about "
            f"{n * per / 60:.1f} min each — {omag.N_POSES * n * per / 60:.0f} "
            f"min for all {omag.N_POSES} poses")

    def _refresh(self):
        done = len(self.run)
        # Both boxes, not just the sweep one: changing the method between poses
        # would make pose 3 a different measurement from poses 1 and 2, and the
        # run has no way to say so afterwards.
        self.box_method.setEnabled(done == 0 and not self.busy)
        self.box_setup.setEnabled(done == 0 and not self.busy)
        self.btn_primary.setEnabled(not self.busy)
        if self.busy:
            self.lbl_step.setText(
                f"Pose {done + 1} of {omag.N_POSES} — "
                f"{PASS_NAMES.get(self._pass, 'sweeping')}")
            self.btn_primary.setText("Sweeping...")
            return
        if done == 0:
            self.lbl_step.setText("Before you start")
            self.txt.setMarkdown(SETUP_TEXT)
            self.btn_primary.setText("Start pose 1")
        elif done < omag.N_POSES:
            self.lbl_step.setText(f"Turn the head — pose {done + 1} of "
                                  f"{omag.N_POSES}")
            self.txt.setMarkdown(TURN_TEXT.format(n=done + 1))
            self.btn_primary.setText(f"Start pose {done + 1}")
        else:
            self.lbl_step.setText("All four poses recorded")
            self.txt.setMarkdown(DONE_TEXT)
            self.btn_primary.setText("Apply and save")

    # ---- running ---------------------------------------------------------
    def on_primary(self):
        if self.busy:
            return
        if len(self.run) >= omag.N_POSES:
            self.finish()
            return
        self.start_pose()

    def _check_travel(self, name, lo_off, hi_off, park):
        """True if [park+lo_off, park+hi_off] fits in this axis's envelope."""
        st = self.win.stages[name]
        # limit_mm, not travel_mm: this is the check that decides whether a
        # 40-minute unattended run is about to drive the head somewhere, so it
        # has to be against what the axis is allowed to use, not against the
        # length of the leadscrew.
        lo, hi = st.limit_mm
        if hi <= lo:
            return True
        if park + lo_off < lo or park + hi_off > hi:
            envelope = ("travel" if st.limit_mm == st.travel_mm
                        else "allowed range")
            QtWidgets.QMessageBox.warning(
                self, "Guided magnet calibration",
                f"The run needs {name} between {park + lo_off:.1f} and "
                f"{park + hi_off:.1f} mm, which is outside its "
                f"{lo:g}..{hi:g} mm {envelope}. Re-park the head, or shorten "
                f"that pass.")
            return False
        return True

    def start_pose(self):
        win = self.win
        if win.stages is None:
            QtWidgets.QMessageBox.warning(
                self, "Guided magnet calibration",
                "The stages are not connected. This routine drives the head "
                "past the magnet; without motion there is nothing to guide.")
            return
        axis = self.cmb_axis.currentText()
        across, normal = self._across_name(), self._normal_name()
        named = [n for n in (axis, across, normal) if n]
        if len(set(named)) != len(named):
            # The grid is a dict keyed by axis name, so naming one axis twice
            # does not fail -- the second silently replaces the first and the
            # run becomes a different scan from the one on the panel, with a
            # pass missing and nothing in the output to say so. Custom is the
            # way in: the named methods derive their axes and cannot collide.
            QtWidgets.QMessageBox.warning(
                self, "Guided magnet calibration",
                f"The same axis is named twice: tube {axis}, transverse "
                f"{across or 'none'}, standoff {normal or 'none'}.\n\n"
                f"Each pass has to move a different axis. One axis doing two "
                f"jobs does not fail — it quietly drops a pass and returns a "
                f"result that looks like a complete run.")
            return
        if normal and not across:
            # Not a fussy pairing rule -- the dither's model is "move straight
            # toward the magnet", and the transverse cut is the only thing that
            # makes that true. Off to one side by a, with the magnet h away
            # along the dither axis, the fit returns h*r^2/(h^2 - a^2) instead
            # of the distance: on this rig's geometry, 51 mm for a chip that is
            # really 26 mm away. Correcting with that number is worse than not
            # correcting.
            QtWidgets.QMessageBox.warning(
                self, "Guided magnet calibration",
                "The standoff dither needs the transverse cut.\n\n"
                "The dither measures distance by assuming it is moving "
                "straight toward the magnet, and the cut is what puts each "
                "chip under the magnet so that it is. Without it the fit "
                "returns a number that is not the distance — on this "
                "geometry it reads 51 mm for a chip 26 mm away — and "
                "correcting with that is worse than not correcting at all.\n\n"
                "Either pick a transverse axis as well, or set the standoff "
                "axis to none and run the plain axial sweep.")
            return
        if win._estop_reason is not None:
            QtWidgets.QMessageBox.warning(
                self, "Guided magnet calibration",
                f"The emergency stop is latched: {win._estop_reason}.\n\n"
                f"Reset it before driving the head.")
            return
        for name in (axis, across, normal):
            if name and not win.stages[name].position_trusted:
                QtWidgets.QMessageBox.warning(
                    self, "Guided magnet calibration",
                    f"Axis {name}: {win.stages[name].distrust_reason}.\n\n"
                    f"Where it says it is and where it is are unrelated. The "
                    f"peaks would still line up with each other, but nothing "
                    f"else could be compared with them afterwards. Home it "
                    f"first.")
                return

        # The park is taken once, on the first pose, and reused: the whole
        # routine rests on all four poses being measured against the same
        # fixed magnet, so re-reading the stage each time would let a nudged
        # axis redefine the origin halfway through without saying so.
        if self._start_mm is None:
            self._start_mm = win.stages[axis].position_mm
            self._park = {n: win.stages[n].position_mm
                          for n in (across, normal) if n}
        if not self._check_travel(axis, 0.0, self.spin_span.value(),
                                  self._start_mm):
            self._start_mm = None
            return
        if across and not self._check_travel(
                across, -self.spin_across_half.value(),
                self.spin_across_half.value(), self._park[across]):
            return
        if normal and not self._check_travel(
                normal, -self.spin_dither_half.value(),
                self.spin_dither_half.value(), self._park[normal]):
            return

        self._pending = None
        self._rings = None
        self._run_pass("locate")

    def _grid_for(self, kind):
        """The ScanGrid for one pass.

        Every pass names the same axes in the same order, even the ones it does
        not move. That is what lets the passes be concatenated afterwards: a
        FieldMap only records the columns its grid named, so a locate that
        named one axis and a cut that named three would come back as two point
        clouds in different spaces with no way to say where the first one was
        on the axes it left out.
        """
        axis = self.cmb_axis.currentText()
        across, normal = self._across_name(), self._normal_name()
        if kind == "locate":
            along = self._positions()
        else:
            along = np.asarray(self._rings, float)

        axes = {axis: along}
        if across:
            axes[across] = (self._park[across] + self._across_offsets()
                            if kind == "cut" else [self._cut_centre()])
        if normal:
            axes[normal] = (self._park[normal] + self._dither_offsets()
                            if kind == "dither" else [self._park[normal]])
        return oscan.ScanGrid(axes)

    def _cut_centre(self):
        """Where to park the transverse axis for the dither.

        The mean of the four rings' transverse peaks, not each ring's own. A
        ring whose arm sits a millimetre off the others is a millimetre off the
        top of a peak that is flat to second order there, which costs 0.4 % of
        field at a 20 mm standoff -- far below what the dither can resolve, and
        worth trading for a dither that is one grid instead of four.
        """
        park = self._park.get(self._across_name())
        if self._pending is None or not self._pending.is_plane:
            return park
        across = self._pending.peak_across_mm
        loud = np.argsort(self._pending.peaks)[-pgeom.SENSORS_PER_FACE:]
        vals = across[loud]
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if len(vals) else park

    def _run_pass(self, kind):
        win = self.win
        self._pass = kind
        grid = self._grid_for(kind)

        # First pass of the first pose only: the carriers are still on the live
        # stream, and the worker is what puts them back on their own clock.
        # Everything after finds them already released, so passing the saved
        # clkdiv again would restore a rate that is no longer the one in force.
        first = not self._released
        if first:
            self._released = True
            if win.act_record.isChecked():
                win.act_record.setChecked(False)
            if isinstance(win.source, LiveSource):
                win.source.stop()
            win.source = None
            win.lbl_state.setText("guided magnet calibration in progress...")
        restore = win.prev_clkdiv if first else {}
        if first:
            win.prev_clkdiv = {}

        self.bar.setRange(0, len(grid))
        self.bar.setValue(0)
        self.bar.setFormat("%v / %m")
        self._t0 = time.time()
        self._worker = ScanWorker(win.hosts, win.stages, grid,
                                  self.spin_secs.value(), win.cal,
                                  self.spin_settle.value(), restore)
        self._worker.message.connect(win.log.log)
        self._worker.progress.connect(self.on_progress)
        self._worker.done.connect(self.on_pass_done)
        # Without this the main window's stop button does not know this thread
        # exists, and stopping the axes mid-pass would leave it to carry on to
        # the next one -- see MainWindow.register_motion_worker.
        win.register_motion_worker(self._worker)
        win.log.log(f"guided magnet calibration: pose {len(self.run) + 1} of "
                    f"{omag.N_POSES}, {PASS_NAMES[kind]}, {len(grid)} points "
                    f"-- {grid.describe()}")
        self._worker.start()
        self._refresh()

    def on_progress(self, i, n, where, sem_ut):
        self.bar.setValue(i)
        elapsed = time.time() - self._t0
        self.bar.setFormat(f"%v / %m — {elapsed / max(i, 1) * (n - i):.0f} s left")

    def _sweep_from(self, fm, pose):
        return omag.PoseSweep.from_fieldmap(
            pose, fm, axis=self.cmb_axis.currentText(),
            across=self._across_name(), normal=self._normal_name(),
            note=f"pose {pose + 1}")

    def _dithers_from(self, fm):
        """Split the dither pass into one Dither per ring.

        Grouped by which ring each row is nearest rather than by counting rows
        off in blocks: run_scan drops a point that failed and carries on, so
        the blocks are not guaranteed to be the length the grid implies, and
        counting would silently slice one ring's dither across two.
        """
        axis, normal = self.cmb_axis.currentText(), self._normal_name()
        pos = np.asarray(fm.pos_mm, float)
        names = list(fm.axes)
        if normal is None or normal not in names or not len(pos):
            return []
        ai, ni = names.index(axis), names.index(normal)
        rings = np.asarray(self._rings, float)
        which = np.argmin(np.abs(pos[:, ai][:, None] - rings[None, :]), axis=1)
        out = []
        for k in range(len(rings)):
            sel = which == k
            if int(sel.sum()) < 3:
                continue
            z = pos[sel, ni]
            out.append(omag.Dither(z - z.mean(), fm.b_mt[sel],
                                   at_mm=float(np.mean(pos[sel, ai]))))
        return out

    def on_pass_done(self, fm, error):
        worker, self._worker = self._worker, None
        self.win.retire_motion_worker(worker)
        self.bar.setFormat("no sweep running")
        kind, self._pass = self._pass, None
        self.win._sync_stage_controls()
        if self.win._estop_reason is not None:
            # An aborted pass comes back with no error and a partial FieldMap
            # -- that is deliberate, it is how a scan keeps the points it did
            # take. But this routine chains: locate starts cut, cut starts
            # dither. Without this the stop would end one pass and immediately
            # start the next, which is not what anyone pressing it meant, and
            # the pose would be built out of half a sweep.
            self.win.log.log("guided magnet calibration: stopped by the "
                             "emergency stop — this pose is abandoned")
            self._pending = None
            self._rings = None
            self._refresh()
            return
        if error:
            self.win.log.log(f"guided magnet calibration: {error}")
            QtWidgets.QMessageBox.warning(self, "Guided magnet calibration",
                                          error)
            self._refresh()
            return
        if fm is None or not len(fm):
            self._refresh()
            return
        pose = len(self.run)

        if kind == "locate":
            self._pending = self._sweep_from(fm, pose)
            self._rings = omag.ring_positions(self._pending)
            self.win.log.log(
                "guided magnet calibration: rings at "
                + ", ".join(f"{v:.1f}" for v in self._rings) + " mm")
            if self._across_name():
                self._run_pass("cut")
                return
        elif kind == "cut":
            self._pending.merge(self._sweep_from(fm, pose))
        elif kind == "dither":
            self._pending.dithers.extend(self._dithers_from(fm))
            self._finish_pose()
            return

        if self._normal_name():
            self._run_pass("dither")
            return
        self._finish_pose()

    def _finish_pose(self):
        self.run.across = self._across_name()
        self.run.normal = self._normal_name()
        self.run.add(self._pending)
        sw, self._pending = self._pending, None
        peaks = sw.peaks
        loud = int(np.argmax(peaks)) + 1
        extra = ""
        if sw.dithers:
            d = sw.standoff_mm[np.argsort(peaks)[-pgeom.SENSORS_PER_FACE:]]
            d = d[np.isfinite(d)]
            extra = (f", standoff {np.mean(d):.1f} mm" if len(d)
                     else ", standoff not fittable -- try a longer average")
        self.win.log.log(
            f"guided magnet calibration: pose {len(self.run)} done, loudest "
            f"S{loud} at {peaks.max():.3f} mT{extra}")
        self.report.setPlainText(self.run.report(self.win.geom))
        self._refresh()

    # ---- finishing -------------------------------------------------------
    def finish(self):
        win = self.win
        base = os.path.join(win.out_dir,
                            time.strftime("magcal_%Y%m%d_%H%M%S"))
        path = self.run.save(base)
        win.log.log(f"guided magnet calibration: saved {path}")
        # Said before the trim is touched, because it decides what the trim
        # is worth: opposite faces that disagree mean part of it is geometry.
        balance = self.run.face_balance(win.geom)
        for note in balance["notes"]:
            win.log.log(f"guided magnet calibration: {note}")

        if self.chk_apply.isChecked():
            resp, _best = self.run.response()
            _corr, skipped = win.cal.cross_calibrate(resp)
            note = (f"kept their previous trim: {', '.join(skipped)}"
                    if skipped else "every live sensor trimmed")
            passes = ["axial"]
            if any(s.is_plane for s in self.run.sweeps):
                passes.append("plane")
            if any(s.dithers for s in self.run.sweeps):
                passes.append("standoff dither")
            win.cal.notes = (f"gain trim from the guided magnet run of "
                             f"{time.strftime('%Y-%m-%d %H:%M')} "
                             f"({os.path.basename(path)}); passes: "
                             f"{', '.join(passes)}; no geometry weighting -- "
                             f"every sensor was measured at its own peak")
            cal_path = win.cal.save(win.args.calibration)
            win.log.log(f"gain trim applied from the guided run "
                        f"(no geometry weighting needed); {note} -> "
                        f"{cal_path}")
            win._calibration_changed("the gain trim")
            win.refresh_cal_report()
            if balance["notes"]:
                QtWidgets.QMessageBox.warning(
                    self, "Gain trim applied, with a caveat",
                    "The trim is applied and saved, but this run's opposite "
                    "faces disagree:\n\n  - "
                    + "\n  - ".join(balance["notes"])
                    + "\n\nThe within-face numbers are unaffected — those "
                      "four sensors never moved relative to each other. It is "
                      "the face-to-face part of the trim that is carrying "
                      "geometry as well as gain.")
        else:
            win.log.log(f"gain trim NOT applied. The run is on disk at {path} "
                        f"and nothing about it is lost -- it can be applied "
                        f"later without repeating the measurement.")
        notes = self.run.check_geometry(win.geom)
        if notes:
            self.offer_geometry_update(notes)
        win.log.log("guided magnet calibration: the carriers are still off "
                    "the live stream — press Connect to go back to live.")
        self._finished = True
        self.btn_primary.setEnabled(False)
        self.lbl_step.setText("Finished")

    def offer_geometry_update(self, notes):
        """Report what the run disagrees with, and offer to write it down.

        The offer is deliberately explicit about what changes and what does
        not. Slots are a measurement: which sensor passed the magnet first is
        not a matter of opinion. Face NUMBERS are not -- turning the tube in
        its cradle renames all four -- so the grouping is checked and the
        labels are left alone.
        """
        win = self.win
        try:
            slots = self.run.measured_slots()
        except ValueError as exc:
            QtWidgets.QMessageBox.information(
                self, "Geometry",
                "The run disagrees with probe_geometry.json:\n\n  - "
                + "\n  - ".join(notes)
                + f"\n\nIt cannot say what the right answer is, though: "
                  f"{exc}\n\nNothing has been changed.")
            return
        diff = [(sid, win.geom.slot(sid), slot)
                for sid, slot in sorted(slots.items())
                if win.geom.slot(sid) != slot]
        if not diff:
            return
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        box.setWindowTitle("Geometry")
        box.setText("The run disagrees with probe_geometry.json.")
        box.setInformativeText(
            "  - " + "\n  - ".join(notes)
            + "\n\nThe magnet says these sensors sit elsewhere along the "
              "tube:\n\n"
            + "\n".join(f"    S{sid}:  slot {was}  →  slot {now}"
                         for sid, was, now in diff)
            + "\n\nOnly the slots change. Which face index a group of four "
              "carries is a naming choice the magnet cannot make — turning "
              "the tube renames them all — so the face labels are left as "
              "they are.")
        write = box.addButton("Write it to probe_geometry.json",
                              QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Leave the file alone",
                      QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is not write:
            win.log.log("guided magnet calibration: geometry left as it was")
            return
        changed = self.run.apply_to_geometry(win.geom)
        path = win.geom.save(win.args.geometry)
        win.log.log("geometry updated from the magnet run: "
                    + ", ".join(f"S{sid} slot {was}->{now}"
                                for sid, was, now in changed)
                    + f" -> {path}")
        # Re-read it rather than refreshing by hand: Reload geometry already
        # knows every place a position reaches -- the 3D view, the plot, the
        # sensor table, the demo source -- and a second copy of that list here
        # would be one item short the first time somebody adds a fifth.
        win.on_reload_geometry()

    def closeEvent(self, event):
        """Stop cleanly, and let QDialog finish the job.

        The super() call is not decoration. QDialog.closeEvent() is what calls
        reject() and therefore emits finished(), which is the signal the main
        window uses to forget this dialog. Accepting the event and returning
        looks identical on screen and leaves the window holding a reference to
        a hidden dialog for ever -- after which the menu item that opens this
        quietly does nothing, because it raises the dead one instead.
        """
        if self.busy:
            reply = QtWidgets.QMessageBox.question(
                self, "Guided magnet calibration",
                "A sweep is running. Stop it and close?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel)
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._worker.abort()
            self._worker.wait(30000)
        if self.run.sweeps and not self._finished:
            # Poses recorded but never applied: closing here throws away the
            # only copy, since the run is written to disk by finish().
            reply = QtWidgets.QMessageBox.question(
                self, "Guided magnet calibration",
                f"{len(self.run)} pose(s) recorded and not saved.\n\n"
                f"They are only in this window — closing discards them, and "
                f"the sweeps would have to be driven again. Close anyway?",
                QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel)
            if reply != QtWidgets.QMessageBox.StandardButton.Discard:
                event.ignore()
                return
            self.win.log.log(f"guided magnet calibration: {len(self.run)} "
                             f"unsaved pose(s) discarded")
        super().closeEvent(event)


SETUP_TEXT = """
**One magnet, clamped once, and the head driven past it four times.**

Before pose 1:

1. **Clamp the magnet** beside the head's travel, level with one face and a few
   centimetres clear of it. It must not move again until all four poses are
   done — it is the common reference for every sensor.
2. **Park the head** so the magnet is just clear of the first ring of sensors,
   and roughly centred on them on the other two axes. The run starts from
   wherever the head is now.
3. Note the rough **standoff** — how far the magnet is from the chips — into the
   box below. Nothing is measured with it, but it sizes the other two passes.
4. Nothing ferrous may move nearby during a sweep, including you.

Pick the run at the top. **Full** is three passes per pose, and they do
different jobs:

* **A, locate** — a coarse run along the tube. Finds where the four rings are.
* **B, cut** — a fine sweep *across* the face at each ring. Every sensor gets
  measured at the top of its own peak instead of at a slice through it, which
  is what stops a millimetre of arm placement becoming 15 % of fake gain.
* **C, dither** — a few points *toward and away from* the magnet at each ring.
  |B| has no maximum in that direction, so there is no peak to find; what it
  measures is the distance itself, which is the last error the plane can't
  reach.

**Axial only** runs pass A alone — the original routine, and the right choice
when only the tube axis is motorised.

You do not have to park perfectly. Finding each peak is what pass B is for —
that is most of the point of it.
"""

TURN_TEXT = """
**Index the tube one quarter turn** in its cradle, to pose {n}.

* About the tube's **own axis and nothing else**. A pose that also shifts the
  head sideways breaks the equal-approach argument quietly — the numbers still
  look plausible afterwards.
* The exact angle does not matter. Four faces getting their turn does.
* Do not touch the magnet, the cradle position, or the cable dress.

Then start pose {n}.
"""

DONE_TEXT = """
All four faces have had their turn at the magnet. The table below is the
result: each sensor's peak, taken from the pose where its own face was toward
the magnet, at the top of its own transverse peak, and scaled to a common
standoff — so the spread is gain and nothing else.

Read the **standoff** column before the trim. If it is mostly `--`, pass C
could not fit and the distances are assumed rather than measured; raise the
averaging time and run again. If the spread of standoffs is large, that is the
arms, and it is exactly what would otherwise have shown up as gain.

**Apply and save** writes the run to the captures folder, folds the trim into
the calibration if the box is ticked, and reports anything the run disagrees
with in `probe_geometry.json` — it never rewrites the geometry itself.
"""


class LivePlot(QtWidgets.QWidget):
    """Rolling traces. |B| per sensor by default, since that is comparable."""

    MODES = ("|B| per sensor", "all axes (chip frame)", "all axes (tube frame)")
    # What the y axis shows. The pipeline always works in millitesla; these
    # walk that back down the conversion chain so the electrical signal can be
    # inspected directly -- useful when you want to know whether a number is
    # the sensor talking or the conversion assuming.
    UNITS = ("mT", "uT", "mV (chip output)", "ADC counts")

    def __init__(self, geom, profiler=None):
        super().__init__()
        self.profiler = profiler or oprof.Profiler(enabled=False)
        self.target_mult = PLOT_TARGET_MULT
        self.unit_scale = np.ones(N_SENSORS)
        self.unit_name = "mT"
        self.geom = geom
        self.mode = self.MODES[0]
        self.visible = set(range(N_SENSORS))
        self.dead = set()
        self.colors = sensor_colors()

        self.plot = ProfiledPlot("Qt paint (live plot)", profiler)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "time", units="s")
        self.plot.setLabel("left", "B", units="mT")
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.setClipToView(True)
        self.legend = self.plot.addLegend(colCount=4, labelTextSize="7pt")

        self.mag_curves, self.axis_curves = [], []
        for s in range(N_SENSORS):
            c = self.plot.plot(
                pen=pg.mkPen(self.colors[s], width=PLOT_PEN_WIDTH),
                name=f"S{s+1}")
            self._thin(c)
            self.mag_curves.append(c)
            row = []
            for _a, style in enumerate((QtCore.Qt.PenStyle.SolidLine,
                                       QtCore.Qt.PenStyle.DashLine,
                                       QtCore.Qt.PenStyle.DotLine)):
                cc = self.plot.plot(pen=pg.mkPen(
                    self.colors[s], width=PLOT_PEN_WIDTH, style=style))
                self._thin(cc)
                row.append(cc)
            self.axis_curves.append(row)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plot)
        self.set_mode(self.mode)

    @staticmethod
    def _thin(curve):
        """
        Make each curve downsample itself to the screen.

        A 20 s window at 500 Hz is 10 000 points per trace, and there are up to
        48 traces. Setting this on the PlotItem is not enough -- it has to be on
        the curves -- and without it Qt rasterises every one of those points
        into a plot around a thousand pixels wide. Measured, that was the single
        most expensive thing in the whole application, tens of milliseconds per
        repaint, and invisible to any timing of our own code because it happens
        inside Qt's paint. 'peak' keeps the extremes of each bin, so a magnet
        spike still shows at full height.
        """
        curve.setDownsampling(auto=True, method="peak")
        curve.setClipToView(True)

    def set_mode(self, mode):
        self.mode = mode
        self._relabel()
        self._apply_visibility()

    def set_units(self, scale, name):
        """Per-sensor scale factor taking millitesla to the displayed unit."""
        self.unit_scale = np.asarray(scale, float).reshape(N_SENSORS)
        self.unit_name = name
        self._relabel()

    def _relabel(self):
        mag = self.mode == self.MODES[0]
        self.plot.setLabel("left", "|B|" if mag else "B",
                           units=self.unit_name)

    def reset_view(self):
        """Undo any panning or zooming and go back to following the data.

        pyqtgraph switches auto-ranging OFF the instant you drag or scroll a
        plot, and nothing on screen says so: the traces simply stop filling the
        axes, which reads as the signal having changed rather than the view
        having been moved. A scroll over the axis can also leave the mouse
        enabled on one axis only, which is even harder to spot.

        Order matters. ViewBox.autoRange() fits once and disables auto-ranging
        as it goes -- it calls setRange(), which defaults to
        disableAutoRange=True -- so enabling has to come last or the button
        would leave the plot frozen at exactly the moment it was pressed.
        """
        vb = self.plot.getViewBox()
        vb.setMouseEnabled(x=True, y=True)
        vb.enableAutoRange(x=True, y=True)
        vb.updateAutoRange()

    def set_visible_sensors(self, sensors):
        self.visible = set(sensors)
        self._apply_visibility()

    def set_dead(self, dead):
        """
        Hide railed sensors. A stuck channel sits at -32768 counts, which is
        ~195 mT of nonsense -- left on the plot it pins the y axis and every
        real trace collapses onto the zero line.
        """
        dead = {int(d[1:]) - 1 for d in dead if d.startswith("S") and d[1:].isdigit()}
        if dead != self.dead:
            self.dead = dead
            self._apply_visibility()

    def _apply_visibility(self):
        mag = self.mode == self.MODES[0]
        for s in range(N_SENSORS):
            on = s in self.visible and s not in self.dead
            self.mag_curves[s].setVisible(on and mag)
            for c in self.axis_curves[s]:
                c.setVisible(on and not mag)
        for s in range(N_SENSORS):
            if s >= len(self.legend.items):
                continue
            on = s in self.visible and s not in self.dead
            for part in self.legend.items[s]:
                if part is not None:
                    part.setVisible(on)

    @staticmethod
    def _to_screen(t, Y, target):
        """
        Reduce (n, ncurves) to about `target` points while keeping the extremes.

        Each output bin contributes its minimum and its maximum, so a magnet
        spike one sample wide still reaches full height -- which a plain stride
        would throw away. Sharing one x array across all curves keeps this fully
        vectorised, at the cost of placing each pair at the bin edges rather
        than at the exact sample; one bin is about one pixel, so that is not
        visible.

        Doing this ourselves rather than leaving it to the plotting library
        makes the repaint cost depend on the width of the window in pixels
        instead of on the length of the buffer, so a longer time window or a
        higher output rate no longer costs anything to draw.
        """
        n = Y.shape[0]
        if n <= target:
            return t, Y
        bins = max(2, target // 2)
        k = n // bins
        m = bins * k
        Yb = Y[:m].reshape(bins, k, Y.shape[1])
        out = np.empty((bins * 2, Y.shape[1]), dtype=Y.dtype)
        out[0::2] = Yb.min(axis=1)
        out[1::2] = Yb.max(axis=1)
        tb = t[:m:k]
        x = np.repeat(tb, 2)
        x[1::2] = tb + (t[-1] - t[0]) / bins * 0.5
        return x, out

    def update_data(self, b_mt, fs_out):
        n = b_mt.shape[0]
        if n < 2:
            return
        t = (np.arange(n) - n) / fs_out
        shown = self.visible - self.dead
        self.profiler.note("curves drawn", len(shown) * (1 if self.mode ==
                                                         self.MODES[0] else 3))
        self.profiler.note("points buffered", n)
        target = max(256, int(self.plot.width() * self.target_mult))
        k = self.unit_scale
        if self.mode == self.MODES[0]:
            # |k B| == k |B| for positive k, so scaling the magnitude is the
            # same as scaling the components first.
            mag = np.linalg.norm(b_mt, axis=-1) * k[None, :]
            x, y = self._to_screen(t, mag, target)
            for s in shown:
                self.mag_curves[s].setData(x, y[:, s])
        else:
            b = self.geom.to_tube_frame(b_mt) if "tube" in self.mode else b_mt
            b = b * k[None, :, None]
            x, y = self._to_screen(t, b.reshape(n, -1), target)
            for s in shown:
                for a in range(3):
                    self.axis_curves[s][a].setData(x, y[:, s * 3 + a])
        self.profiler.note("points drawn per curve", len(x))


class SensorBars(QtWidgets.QWidget):
    """
    Peak |B| per sensor since the last refresh -- the "are the spikes the same
    height" view, and the reason this window exists.

    Excluded sensors are drawn as a flat red stub rather than at their real
    value: a railed channel reads nearly 200 mT, which would flatten every real
    bar against the axis.
    """

    def __init__(self, profiler=None):
        super().__init__()
        self.plot = ProfiledPlot("Qt paint (peak bars)", profiler)
        self.plot.showGrid(y=True, alpha=0.25)
        self.plot.setLabel("left", "peak |B|", units="mT")
        self.window_s = 0.5
        self.plot.setTitle(f"peak |B| per sensor, last {self.window_s:g} s",
                           size="9pt")
        ax = self.plot.getAxis("bottom")
        ax.setTicks([[(i, f"{i+1}") for i in range(N_SENSORS)]])
        self.plot.setLabel("bottom", "sensor")
        self.colors = sensor_colors()
        self.bars = pg.BarGraphItem(x=np.arange(N_SENSORS), height=np.zeros(N_SENSORS),
                                    width=0.72, brushes=self.colors)
        self.plot.addItem(self.bars)
        self.median_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("w", style=QtCore.Qt.PenStyle.DashLine))
        self.plot.addItem(self.median_line)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plot)
        self.dead = set()

    def update_values(self, mag, dead=()):
        self.dead = set(dead)
        live = np.array([f"S{i+1}" not in self.dead for i in range(N_SENSORS)])
        h = np.array(mag, dtype=float)
        h[~np.isfinite(h)] = 0.0
        vis = h[live]
        med = float(np.median(vis)) if vis.size else 0.0
        stub = max(med * 0.04, (float(vis.max()) if vis.size else 1.0) * 0.02)
        brushes = []
        for i in range(N_SENSORS):
            if not live[i]:
                h[i] = stub                       # keep the axis on the real data
                brushes.append(pg.mkBrush(140, 40, 40))
            else:
                brushes.append(pg.mkBrush(self.colors[i]))
        self.bars.setOpts(height=h, brushes=brushes)
        self.median_line.setValue(med)


class SensorTable(QtWidgets.QTableWidget):
    """Per-sensor state, and where the per-sensor measurement range is set."""

    COLS = ("sensor", "face", "state", "|B| mT", "Bx", "By", "Bz",
            "noise uT", "VCM V", "T degC*", "range", "gain trim")
    # The star on T degC: octobee.temp_c is uncalibrated and each chip's TA
    # output carries its own offset, so the spread between sensors is not a
    # gradient. Read per chip, over time.
    COL_TIPS: ClassVar[dict] = {"T degC*": ob.TEMP_UNCALIBRATED_NOTE}
    STATE_COLORS: ClassVar[dict] = {
        "ok": (40, 120, 60), "noisy": (150, 110, 20),
        "fault": (150, 70, 20), "dead": (140, 30, 30),
        "unknown": (70, 70, 80)}

    range_changed = QtCore.pyqtSignal(int, float)

    def __init__(self, geom):
        super().__init__(N_SENSORS, len(self.COLS))
        self.geom = geom
        self.setHorizontalHeaderLabels(self.COLS)
        for name, tip in self.COL_TIPS.items():
            self.horizontalHeaderItem(self.COLS.index(name)).setToolTip(tip)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

        self._combos = []
        for r in range(N_SENSORS):
            for c in range(len(self.COLS)):
                it = QtWidgets.QTableWidgetItem("")
                it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.setItem(r, c, it)
            self.item(r, 0).setText(f"S{r+1}")
            combo = QtWidgets.QComboBox()
            for v in sorted(ob.RANGE_TO_VPT):
                combo.addItem(f"+/-{v:g} mT", v)
            combo.setCurrentIndex(0)
            combo.currentIndexChanged.connect(
                lambda _i, row=r, cb=combo: self.range_changed.emit(
                    row, float(cb.currentData())))
            self.setCellWidget(r, self.COLS.index("range"), combo)
            self._combos.append(combo)
        self.refresh_geometry(geom)

    def refresh_geometry(self, geom):
        self.geom = geom
        for r in range(N_SENSORS):
            f = pgeom.FACE_NAMES[geom.face(r + 1)]
            self.item(r, 1).setText(f"{f}/{geom.slot(r+1)}")

    def set_ranges(self, ranges):
        for r, v in enumerate(ranges):
            cb = self._combos[r]
            i = cb.findData(float(v))
            if i >= 0 and i != cb.currentIndex():
                cb.blockSignals(True)
                cb.setCurrentIndex(i)
                cb.blockSignals(False)

    def update_rows(self, table):
        for r, row in enumerate(table):
            state = row["state"]
            self.item(r, 2).setText(state)
            self.item(r, 2).setBackground(
                QtGui.QColor(*self.STATE_COLORS.get(state, (70, 70, 80))))
            self.item(r, 2).setToolTip(row.get("note", "") or "")
            vals = [("|B| mT", f"{row['absB_mT']:.3f}"),
                    ("Bx", f"{row['Bx_mT']:.3f}"),
                    ("By", f"{row['By_mT']:.3f}"),
                    ("Bz", f"{row['Bz_mT']:.3f}"),
                    ("noise uT", f"{row['noise_uT_rms']:.0f}"),
                    ("VCM V", f"{row['vcm_v']:.4f}"),
                    ("T degC*", "--" if row["temp_c"] is None
                     else f"{row['temp_c']:.1f}"),
                    ("gain trim", f"{row['gain_trim']:.3f}")]
            for name, txt in vals:
                self.item(r, self.COLS.index(name)).setText(txt)


class LogPane(QtWidgets.QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(4000)
        self.setFont(QtGui.QFont("Consolas", 9))

    def log(self, msg):
        self.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ==========================================================================
# main window
# ==========================================================================

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hosts = list(args.uut) if args.uut else list(ob.DEFAULT_UUTS)
        self.out_dir = getattr(args, "out_dir", None) or "captures"
        # Collected rather than logged: the log pane does not exist yet, and a
        # config that failed to parse is the first thing the user must be told.
        self._config_errors = []
        self.geom = pgeom.Geometry.load_or_default(
            args.geometry, on_error=self._config_errors.append)
        self.cal = ocal.Calibration.load_or_default(
            args.calibration, on_error=self._config_errors.append)
        # "Present and readable", not merely "present" -- a file that exists
        # and failed to parse must not be reported as loaded.
        self.cal_from_file = (os.path.exists(args.calibration)
                              and not self._config_errors)
        # The machine the probe is measuring inside: which coils, which of them
        # are live, and where the head sits among them. Loaded before the UI so
        # the tab opens showing the last session's placement rather than the
        # origin -- a pose is measured with a tape, and re-entering it every
        # morning is how it drifts.
        self.machine = omach.MachineConfig.load_or_default(
            args.machine, on_error=self._config_errors.append)
        self.coils = omach.CoilSet.load_or_none(
            self.machine.coil_file, on_error=self._config_errors.append)
        for lost in self.machine.adopt(self.coils):
            self._config_errors.append(f"{args.machine}: {lost}")
        self._probe_cloud = None
        self._machine_key = None
        self._machine_quiet = False
        self._machine_travel_taken = False
        self.source = None
        self.prev_clkdiv = {}
        self.out_rate = 500.0
        self.window_s = 20.0
        self.roll = Rolling(int(self.out_rate * self.window_s))
        self.raw_hist = None
        self.raw_hist_n = 0
        self.csv_rec = None
        self.raw_rec = None
        self.collecting = None          # 'tare' | 'magnet' | 'sweep'
        self.sweeps = {}                # tag -> opc.RollSweep
        self.pose_solution = None
        self.magnet_peaks = None
        self.last_health = None
        self._last_table = None
        self._last_health_t = 0.0
        self._last_dropped = 0
        self.paused = False
        self._draw_ms = 0.0
        self.prof = oprof.Profiler(enabled=bool(getattr(args, "profile", False)))
        self.lag = oprof.LagMonitor(interval_ms=100)
        self._connect_worker = None
        self._snap_worker = None
        self.stages = None
        self.stage_combos = {}
        self._stage_pending = None
        self._stage_worker = None
        self._magnet_wizard = None
        self._connect_was_automatic = False
        self._connecting = False
        self._scan_worker = None
        self._scan_t0 = 0.0
        # Every thread that can command motion, and the latch that overrides
        # all of them. Set before _build_ui: the toolbar's stop button and the
        # Esc shortcut are live from the moment the window exists, and both
        # read these.
        self._motion_workers = []
        self._retired_workers = []
        self._estop_reason = None
        self._estop_alarms = set()
        self._state_before_estop = None

        self.setWindowTitle("OCTO-BEE Hall probe")
        self.resize(1720, 980)
        self._build_ui()
        self._apply_dark()

        # Three clocks, fastest first: acquisition must never be blocked by
        # drawing, drawing should be smooth but is the user's to trade away,
        # and the diagnostics are slow and expensive.
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(50)
        self.view_timer = QtCore.QTimer(self)
        self.view_timer.timeout.connect(self.on_view_tick)
        self.view_timer.start(int(1000 / DEFAULT_VIEW_HZ))
        self.slow = QtCore.QTimer(self)
        self.slow.timeout.connect(self.on_slow_tick)
        self.slow.start(1000)
        # Asks for exactly 100 ms and records how late it actually gets it.
        # That difference is the honest measure of "is the app blocked".
        self.lag_timer = QtCore.QTimer(self)
        self.lag_timer.timeout.connect(self.lag.tick)
        self.lag_timer.start(100)
        self.prof_timer = QtCore.QTimer(self)
        self.prof_timer.timeout.connect(self.refresh_profile)
        self.prof_timer.start(1000)
        # The stage positions come out of the Kinesis DLL's own polling
        # cache, so this reads memory rather than talking to USB -- it
        # can run fast enough to make jogging feel direct without
        # competing with acquisition.
        self.stage_timer = QtCore.QTimer(self)
        self.stage_timer.timeout.connect(self.refresh_stage_table)
        # Same clock as the stage table, and for the same reason: what the
        # machine view draws is where the stages say the head is, so the two
        # must not be able to disagree on screen.
        self.stage_timer.timeout.connect(self.refresh_machine)
        self.stage_timer.start(200)

        for msg in self._config_errors:
            self.log.log(f"WARNING: {msg}")

        if self.coils is not None:
            self.log.log(f"coil set: {self.coils.note} — "
                         f"{self.machine.coil_file}")
            self.log.log("machine: " + self.machine.energised_summary(self.coils))
            self.machine_view.set_coils(self.coils, self.machine.coil_radius_mm,
                                        self.machine.energised)
            self.machine_view.reset_camera()
        self._refresh_machine_controls()
        self.refresh_machine(force=True)

        if args.demo:
            self._set_source(DemoSource(self.geom), "demo")
        elif args.replay:
            self._set_source(ReplaySource(args.replay), f"replay {args.replay}")
        elif getattr(args, "no_connect", False):
            self.log.log("ready -- press Connect to take over the stream from "
                         "Phoebus and start reading both carriers")
        else:
            # Deferred rather than called here: connecting wants a running
            # event loop to report progress into, and __init__ is finishing
            # inside show(). A few hundred milliseconds also means the window
            # is painted before the first "connecting..." line arrives, so a
            # slow carrier looks like a slow connect rather than a slow start.
            self.log.log("connecting automatically -- run with --no-connect "
                         "to start disconnected")
            QtCore.QTimer.singleShot(300, self._auto_connect)
        self._report_calibration_source()

    def _report_calibration_source(self):
        if self.cal_from_file:
            self.log.log(f"calibration loaded from {self.args.calibration}: "
                         + self.cal.summary().replace("\n", "; "))
            if self.cal.notes:
                first = self.cal.notes.split(". ")[0]
                self.log.log(f"  {first}. (full note in "
                             f"{self.args.calibration})")
        else:
            self.log.log(
                f"no {self.args.calibration} found -- using built-in defaults, "
                f"which put every sensor on +/-20 mT / 63 V/T. That matches "
                f"the probe as audited on 2026-08-19, when all 16 chips were "
                f"harmonised to gain 3000, but nothing here checks it at run "
                f"time. If the gain has been changed since, set the range per "
                f"sensor in the Sensors tab and save the calibration -- a "
                f"wrong range is invisible on screen and simply rescales "
                f"everything.")

    # ---- construction ----------------------------------------------------
    def _build_ui(self):
        self._build_toolbar()
        self.log = LogPane()

        self.view3d = ProbeView3D(self.geom, profiler=self.prof)
        self.machine_view = MachineView3D(self.geom, profiler=self.prof)
        self.bars = SensorBars(profiler=self.prof)
        self.plot = LivePlot(self.geom, profiler=self.prof)
        self.table = SensorTable(self.geom)
        self.table.range_changed.connect(self.on_range_changed)
        self.table.set_ranges(self.cal.ranges_mt)

        left = QtWidgets.QTabWidget()
        left.addTab(self._live_tab(), "Live")
        left.addTab(self.table, "Sensors")
        left.addTab(self._calib_tab(), "Calibration")
        left.addTab(self._health_tab(), "Diagnostics")
        left.addTab(self._stage_tab(), "Stages")
        left.addTab(self._machine_tab(), "Machine")
        left.addTab(self._export_tab(), "Data output")
        left.addTab(self._profile_tab(), "Profile")
        left.addTab(self.log, "Log")
        left.addTab(self._help_tab(), "Help")
        left.setCurrentIndex(0)
        self.tabs = left

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)
        head = QtWidgets.QLabel("Probe head — colour and arrow length are |B|, "
                                "arrows are the tube-frame field direction")
        head.setWordWrap(True)
        head.setStyleSheet("color:#9aa3b2; font-size:11px;")
        rl.addWidget(head)
        rl.addWidget(self.view3d, 3)
        rl.addWidget(self._view3d_controls())
        rl.addWidget(self.bars, 1)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([1000, 720])
        self.setCentralWidget(split)

        self.status = self.statusBar()
        self.lbl_state = QtWidgets.QLabel("disconnected")
        self.lbl_rate = QtWidgets.QLabel("")
        self.lbl_rec = QtWidgets.QLabel("")
        for w in (self.lbl_state, self.lbl_rate, self.lbl_rec):
            self.status.addPermanentWidget(w)

    def _build_toolbar(self):
        tb = QtWidgets.QToolBar("main")
        tb.setIconSize(QtCore.QSize(16, 16))
        tb.setMovable(False)
        self.addToolBar(tb)

        self.act_connect = QtGui.QAction("Connect", self)
        self.act_connect.triggered.connect(self.on_connect)
        tb.addAction(self.act_connect)
        self.act_disconnect = QtGui.QAction("Disconnect", self)
        self.act_disconnect.triggered.connect(self.on_disconnect)
        self.act_disconnect.setEnabled(False)
        tb.addAction(self.act_disconnect)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel(" stream rate "))
        self.cmb_rate = QtWidgets.QComboBox()
        for label, fs in STREAM_RATES.items():
            self.cmb_rate.addItem(label, fs)
        self.cmb_rate.setToolTip(
            "Each carrier produces 19.2 MB/s at 200 kSPS but its stream path "
            "delivers only ~10-15 MB/s (the Ethernet is 1 Gbps and not the "
            "limit -- reading localhost:4210 on the box itself tops out at "
            "15 MB/s), so a sustained full-rate stream falls behind.\n\n"
            "Lowering the clock is a trade, not a free win: the sensor's analog "
            "low-pass is fixed at 100 kHz, so anything below 200 kSPS aliases "
            "and the noise density rises by sqrt(100kHz/(fs/2)) -- 2x at "
            "50 kSPS, 3.2x at 20 kSPS (measured 3.1x).\n\n"
            "For short captures prefer 200 kSPS: the box buffers and delivers "
            "every sample, just slower than real time (a 3 s capture from both "
            "boxes came back with zero lost samples). The original clkdiv is "
            "restored on disconnect.")
        tb.addWidget(self.cmb_rate)

        tb.addWidget(QtWidgets.QLabel("  output rate "))
        self.cmb_out = QtWidgets.QComboBox()
        for r in OUT_RATES:
            self.cmb_out.addItem(f"{r:g} Hz", r)
        self.cmb_out.setCurrentIndex(list(OUT_RATES).index(500.0))
        self.cmb_out.currentIndexChanged.connect(self.on_out_rate)
        tb.addWidget(self.cmb_out)

        tb.addWidget(QtWidgets.QLabel("  refresh "))
        self.cmb_view = QtWidgets.QComboBox()
        for r in VIEW_RATES:
            self.cmb_view.addItem(f"{r:g} Hz", r)
        self.cmb_view.setCurrentIndex(list(VIEW_RATES).index(DEFAULT_VIEW_HZ))
        self.cmb_view.setToolTip(
            "How often the plot, the bars and the 3D head are redrawn. This is "
            "purely cosmetic: acquisition, recording and the calibration all "
            "run on their own clock, so turning it down costs smoothness and "
            "nothing else. Drop it to 2 Hz if the window feels heavy.")
        self.cmb_view.currentIndexChanged.connect(self.on_view_rate)
        # 'activated' as well as 'currentIndexChanged': after an automatic
        # backoff the combo still reads the rate you asked for, so re-picking
        # that same entry is exactly how you would expect to restore it -- and
        # currentIndexChanged stays silent when the index has not moved.
        self.cmb_view.activated.connect(self.on_view_rate)
        tb.addWidget(self.cmb_view)

        self.act_pause = QtGui.QAction("Pause view", self)
        self.act_pause.setCheckable(True)
        self.act_pause.setToolTip(
            "Stop redrawing entirely. Acquisition and recording carry on -- "
            "use this while recording if you want every cycle going to the data.")
        self.act_pause.toggled.connect(self.on_pause)
        tb.addAction(self.act_pause)

        tb.addWidget(QtWidgets.QLabel("  window "))
        self.spin_window = QtWidgets.QDoubleSpinBox()
        self.spin_window.setRange(1.0, 120.0)
        self.spin_window.setValue(self.window_s)
        self.spin_window.setSuffix(" s")
        self.spin_window.valueChanged.connect(self.on_window)
        tb.addWidget(self.spin_window)
        tb.addSeparator()

        self.act_tare = QtGui.QAction("Zero (tare)", self)
        self.act_tare.setToolTip("Take 2 s of ambient data and store it as the "
                                 "zero point of every axis of every sensor.")
        self.act_tare.triggered.connect(lambda: self.start_collect("tare", 2.0))
        tb.addAction(self.act_tare)

        self.act_record = QtGui.QAction("Record", self)
        self.act_record.setCheckable(True)
        self.act_record.toggled.connect(self.on_record)
        tb.addAction(self.act_record)

        self.act_snapshot = QtGui.QAction("Snapshot", self)
        self.act_snapshot.setToolTip("Pause streaming and take a lossless "
                                     "full-rate capture to .npz.")
        self.act_snapshot.triggered.connect(self.on_snapshot)
        tb.addAction(self.act_snapshot)

        self._build_estop(tb)

    def _build_estop(self, tb):
        """The stop button, top right, always there and always live.

        Everything about this is deliberate.

        WHERE. Top right of the toolbar, pushed there by an expanding spacer
        so it stays in the corner at any window width. It is outside the tab
        stack because the tab stack is exactly the problem: the STOP that used
        to be the only one lived on the Stages tab, which meant that while the
        head was traversing and you were watching the live plot -- the normal
        way to run a scan -- the stop button was on a page you could not see.
        A stop you have to navigate to is not a stop.

        ALWAYS ENABLED. It is not greyed out when the stages are disconnected,
        because "the stages are disconnected" is this process's belief, and if
        that belief were reliable there would be nothing to stop. It also
        aborts the scan and the wizard, which can be running when `stages` is
        in any state at all.

        KEY. Escape, application-wide, so it works while the guided-magnet
        window has focus. Esc is what a person hits, and it costs nothing to
        honour it. Note the limit: a MODAL dialog eats its own Esc, so while
        one of the confirmation boxes is up the key goes to the box. Those all
        appear before motion starts rather than during it, and the button is
        still there behind them.

        RESET IS SEPARATE. The stop button never becomes the start button.
        Pressing stop twice must not be a way to release the machine, and a
        person reaching for a stop button in a hurry may well hit it twice.
        """
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                             QtWidgets.QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        self.btn_estop = QtWidgets.QPushButton("■  EMERGENCY STOP")
        self.btn_estop.setStyleSheet(
            "QPushButton { background:#c1121f; color:#fff; font-weight:bold; "
            "font-size:13px; padding:6px 18px; border:2px solid #7a0b12; "
            "border-radius:4px; }"
            "QPushButton:hover { background:#e01e2a; }"
            "QPushButton:pressed { background:#7a0b12; }")
        self.btn_estop.setToolTip(
            "Stop every axis immediately and refuse all further motion until "
            "it is reset (Esc).\n\n"
            "Immediate, not profiled: the deceleration ramp is abandoned, so "
            "steps can be lost and every axis needs re-homing before it will "
            "accept an absolute move again. That is the trade — 1.8 mm of "
            "coasting is what a profiled stop costs at the default profile.\n\n"
            "This is a software stop over USB. It needs this program, the USB "
            "link and the controllers all working. It is not a substitute for "
            "a hardware emergency stop in series with the motor supply.")
        self.btn_estop.clicked.connect(self.on_estop)
        tb.addWidget(self.btn_estop)

        self.btn_estop_reset = QtWidgets.QPushButton("Reset")
        self.btn_estop_reset.setToolTip(
            "Clear the latch and allow motion again. Deliberately a separate "
            "button: releasing a machine is an act, not the absence of one.")
        self.btn_estop_reset.setVisible(False)
        self.btn_estop_reset.clicked.connect(self.on_estop_reset)
        tb.addWidget(self.btn_estop_reset)

        self.sc_estop = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self)
        self.sc_estop.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self.sc_estop.activated.connect(self.on_estop)

    def _view3d_controls(self):
        w = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        self.chk_auto = QtWidgets.QCheckBox("auto scale")
        self.chk_auto.setChecked(True)
        self.chk_auto.toggled.connect(
            lambda v: setattr(self.view3d, "auto_scale", v))
        row.addWidget(self.chk_auto)
        row.addWidget(QtWidgets.QLabel("full scale"))
        self.spin_fs = QtWidgets.QDoubleSpinBox()
        self.spin_fs.setRange(0.001, 4000.0)
        self.spin_fs.setDecimals(3)
        self.spin_fs.setValue(1.0)
        self.spin_fs.setSuffix(" mT")
        self.spin_fs.valueChanged.connect(
            lambda v: setattr(self.view3d, "full_scale_mt", v))
        row.addWidget(self.spin_fs)
        self.chk_3d = QtWidgets.QCheckBox("3D")
        self.chk_3d.setChecked(True)
        self.chk_3d.setToolTip(
            "Draw the probe head. This is the most expensive thing in the "
            "window -- untick it if the display cannot keep up. Nothing else "
            "changes: acquisition, calibration and recording are unaffected.")
        self.chk_3d.toggled.connect(self.on_3d_toggle)
        row.addWidget(self.chk_3d)
        chk_arrows = QtWidgets.QCheckBox("arrows")
        chk_arrows.setChecked(True)
        chk_arrows.toggled.connect(self.view3d.set_arrows_visible)
        row.addWidget(chk_arrows)
        chk_lbl = QtWidgets.QCheckBox("labels")
        chk_lbl.setChecked(True)
        chk_lbl.toggled.connect(self.view3d.set_labels_visible)
        row.addWidget(chk_lbl)
        btn = QtWidgets.QPushButton("reset view")
        btn.clicked.connect(self.view3d.reset_camera)
        row.addWidget(btn)
        row.addStretch(1)
        return w

    def _live_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("show"))
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(LivePlot.MODES)
        self.cmb_mode.currentTextChanged.connect(self.plot.set_mode)
        top.addWidget(self.cmb_mode)
        top.addWidget(QtWidgets.QLabel(" in "))
        self.cmb_units = QtWidgets.QComboBox()
        self.cmb_units.addItems(LivePlot.UNITS)
        self.cmb_units.setToolTip(
            "The pipeline always works in millitesla. These walk that back "
            "down the conversion chain so you can see the electrical signal "
            "itself: 'mV (chip output)' is the chip's own output with its VCM "
            "reference subtracted, and 'ADC counts' is what came off the wire. "
            "Any tare or gain trim in force is still applied.")
        self.cmb_units.currentTextChanged.connect(self.on_units)
        top.addWidget(self.cmb_units)
        top.addSpacing(12)
        self.chk_sensors = []
        for s in range(N_SENSORS):
            cb = QtWidgets.QCheckBox(f"{s+1}")
            cb.setChecked(True)
            cb.toggled.connect(self.on_sensor_toggle)
            self.chk_sensors.append(cb)
            top.addWidget(cb)
        btn_all = QtWidgets.QPushButton("all")
        btn_all.clicked.connect(lambda: self._set_all_sensors(True))
        btn_none = QtWidgets.QPushButton("none")
        btn_none.clicked.connect(lambda: self._set_all_sensors(False))
        top.addWidget(btn_all)
        top.addWidget(btn_none)
        top.addStretch(1)
        btn_reset = QtWidgets.QPushButton("reset view")
        btn_reset.setToolTip(
            "Put the axes back on auto after a drag or a scroll. Zooming a "
            "pyqtgraph plot silently stops it auto-ranging, so the traces "
            "stop filling the axes and it looks like the signal changed "
            "rather than the view.")
        btn_reset.clicked.connect(self.plot.reset_view)
        top.addWidget(btn_reset)
        lay.addLayout(top)
        lay.addWidget(self.plot)
        return w

    def _calib_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        g1 = QtWidgets.QGroupBox("1. Zero point")
        f1 = QtWidgets.QHBoxLayout(g1)
        self.spin_tare_s = QtWidgets.QDoubleSpinBox()
        self.spin_tare_s.setRange(0.2, 30.0)
        self.spin_tare_s.setValue(2.0)
        self.spin_tare_s.setSuffix(" s")
        b_tare = QtWidgets.QPushButton("Take zero from ambient")
        b_tare.clicked.connect(
            lambda: self.start_collect("tare", self.spin_tare_s.value()))
        b_clear = QtWidgets.QPushButton("Clear zero")
        b_clear.clicked.connect(self.on_clear_tare)
        f1.addWidget(QtWidgets.QLabel("average over"))
        f1.addWidget(self.spin_tare_s)
        f1.addWidget(b_tare)
        f1.addWidget(b_clear)
        f1.addStretch(1)
        lay.addWidget(g1)

        g_guided = QtWidgets.QGroupBox(
            "2. Guided magnet calibration — measure and equalise response")
        fg = QtWidgets.QVBoxLayout(g_guided)
        blurb = QtWidgets.QLabel(
            "Clamp one magnet, then drive the HEAD past it under motor "
            "control, a quarter turn at a time. Each pose sweeps the plane the "
            "sensors lie in and dithers the standoff, so every sensor is "
            "measured at the top of its own peak and at a measured distance — "
            "no 1/r³ model, and a millimetre of arm placement no longer looks "
            "like 15 % of gain. The run also reports which sensor is really on "
            "which face.")
        blurb.setWordWrap(True)
        fg.addWidget(blurb)
        self.btn_guided = QtWidgets.QPushButton(
            "Guided magnet calibration (motorised, 4 poses)…")
        self.btn_guided.clicked.connect(self.on_guided_magnet)
        row_g = QtWidgets.QHBoxLayout()
        row_g.addWidget(self.btn_guided)
        b_cleargain2 = QtWidgets.QPushButton("Clear gain trim")
        b_cleargain2.clicked.connect(self.on_clear_gain)
        row_g.addWidget(b_cleargain2)
        row_g.addStretch(1)
        fg.addLayout(row_g)
        lay.addWidget(g_guided)

        # ---- superseded routines, built but not shown -----------------------
        # Both of these are replaced by the guided run above and are hidden by
        # default. They are still CONSTRUCTED, and that is deliberate: the
        # calibration report, the export and the collector all reach into these
        # widgets, so deleting them would mean chasing those references for two
        # routines that are still occasionally worth a look. Hidden, not
        # removed -- and the checkbox says why rather than just offering a
        # toggle.
        self.chk_superseded = QtWidgets.QCheckBox(
            "show the superseded manual routines (hand magnet pass, "
            "Earth-field roll)")
        self.chk_superseded.setChecked(False)
        self.chk_superseded.setToolTip(
            "Both were ways of getting at what the guided run above now "
            "measures directly.\n\n"
            "The hand magnet pass holds a magnet near the probe by hand: every "
            "chip is then at a different distance, so it needs a 1/r^n model "
            "of the geometry to divide that out -- and the geometry is exactly "
            "what is not yet established.\n\n"
            "The Earth-field roll sweep was the way round that: a uniform "
            "field every sensor must read identically. It still does something "
            "the magnet run does not -- it pins chip ORIENTATION in three "
            "dimensions -- but for gain it has been superseded.\n\n"
            "Nothing recorded with either is lost by leaving this unticked.")
        self.chk_superseded.toggled.connect(self.on_show_superseded)
        lay.addWidget(self.chk_superseded)

        g2 = QtWidgets.QGroupBox("Superseded: hand magnet pass")
        f2 = QtWidgets.QGridLayout(g2)
        self.btn_magnet = QtWidgets.QPushButton("Start magnet pass")
        self.btn_magnet.setCheckable(True)
        self.btn_magnet.toggled.connect(self.on_magnet_pass)
        f2.addWidget(self.btn_magnet, 0, 0)
        self.lbl_magnet = QtWidgets.QLabel("no pass recorded")
        f2.addWidget(self.lbl_magnet, 0, 1, 1, 4)

        f2.addWidget(QtWidgets.QLabel("magnet position in the tube frame (mm), "
                                      "for dividing out 1/r^n distance:"), 1, 0, 1, 5)
        self.spin_mx = QtWidgets.QDoubleSpinBox()
        self.spin_my = QtWidgets.QDoubleSpinBox()
        self.spin_mz = QtWidgets.QDoubleSpinBox()
        for sp, v in ((self.spin_mx, 60.0), (self.spin_my, 0.0),
                      (self.spin_mz, 100.0)):
            sp.setRange(-1000.0, 1000.0)
            sp.setValue(v)
            sp.setSuffix(" mm")
        self.spin_exp = QtWidgets.QDoubleSpinBox()
        self.spin_exp.setRange(1.0, 4.0)
        self.spin_exp.setValue(3.0)
        self.spin_exp.setSingleStep(0.1)
        self.spin_exp.setPrefix("1/r^")
        self.chk_geom = QtWidgets.QCheckBox("use geometry weighting")
        self.chk_geom.setChecked(False)
        self.chk_geom.setToolTip(
            "A magnet at a fixed distance from the TUBE is at a different "
            "distance from each CHIP. Tick this and give the magnet position "
            "to divide that out, so what is left is electrical.")
        pos_row = QtWidgets.QWidget()
        pl = QtWidgets.QHBoxLayout(pos_row)
        pl.setContentsMargins(0, 0, 0, 0)
        for lbl, sp in (("x", self.spin_mx), ("y", self.spin_my),
                        ("z", self.spin_mz)):
            pl.addWidget(QtWidgets.QLabel(lbl))
            pl.addWidget(sp)
        pl.addSpacing(12)
        pl.addWidget(self.spin_exp)
        pl.addWidget(self.chk_geom)
        pl.addStretch(1)
        f2.addWidget(pos_row, 2, 0, 1, 5)
        superseded = QtWidgets.QLabel(
            "Superseded by the guided run above, which needs no 1/r^n model "
            "because it measures each sensor at its own peak and at a measured "
            "distance. Kept for comparison — a hand pass is still the quickest "
            "way to see whether all sixteen channels are alive.")
        superseded.setWordWrap(True)
        f2.addWidget(superseded, 4, 0, 1, 5)

        self.btn_apply_gain = QtWidgets.QPushButton("Apply gain trim from this pass")
        self.btn_apply_gain.setEnabled(False)
        self.btn_apply_gain.clicked.connect(self.on_apply_gain)
        b_cleargain = QtWidgets.QPushButton("Clear gain trim")
        b_cleargain.clicked.connect(self.on_clear_gain)
        btn_row = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(btn_row)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.btn_apply_gain)
        bl.addWidget(b_cleargain)
        bl.addStretch(1)
        f2.addWidget(btn_row, 6, 0, 1, 5)
        lay.addWidget(g2)


        g3 = QtWidgets.QGroupBox(
            "Superseded: Earth-field roll calibration")
        f3 = QtWidgets.QGridLayout(g3)
        blurb = QtWidgets.QLabel(
            "The Earth's field is uniform to nanotesla across the whole head, so "
            "every sensor must read the SAME vector. This was the way round the "
            "1/r³ position error of the hand magnet pass; the guided run now "
            "measures that position error instead of dodging it. What this "
            "still does and the magnet run does not is pin chip ORIENTATION in "
            "three dimensions.\n"
            "Roll the tube steadily in its cradle through ≥2 turns per sweep.  "
            "A: as mounted.  B: lifted out and replaced end-for-end (separates "
            "offset from axial response).  C: cradle turned to another azimuth "
            "(pins transverse-vs-axial — the flip alone cannot).")
        blurb.setWordWrap(True)
        f3.addWidget(blurb, 0, 0, 1, 6)

        self.spin_sweep_s = QtWidgets.QDoubleSpinBox()
        self.spin_sweep_s.setRange(5.0, 300.0)
        self.spin_sweep_s.setValue(60.0)
        self.spin_sweep_s.setSuffix(" s")
        f3.addWidget(QtWidgets.QLabel("sweep length"), 1, 0)
        f3.addWidget(self.spin_sweep_s, 1, 1)
        col = 2
        for tag, tip in (("A", "as mounted"),
                         ("B", "tube lifted out and replaced end-for-end"),
                         ("C", "cradle turned to another azimuth")):
            b = QtWidgets.QPushButton(f"Record sweep {tag}")
            b.setToolTip(tip)
            b.clicked.connect(lambda _, t=tag: self.start_sweep(
                t, self.spin_sweep_s.value()))
            f3.addWidget(b, 1, col)
            col += 1

        self.lbl_sweeps = QtWidgets.QLabel("no sweeps recorded")
        f3.addWidget(self.lbl_sweeps, 2, 0, 1, 6)

        self.spin_bearth = QtWidgets.QDoubleSpinBox()
        self.spin_bearth.setRange(20.0, 70.0)
        self.spin_bearth.setValue(opc.DEFAULT_B_EARTH_UT)
        self.spin_bearth.setSuffix(" uT")
        self.spin_bearth.setDecimals(2)
        self.spin_bearth.setToolTip(
            "Total field at your location, from ngdc.noaa.gov/geomag or BGS. "
            "This sets ABSOLUTE scale only — matching, offsets and "
            "orientation are all solved without it.")
        f3.addWidget(QtWidgets.QLabel("|B| here"), 3, 0)
        f3.addWidget(self.spin_bearth, 3, 1)
        self.chk_isotropic = QtWidgets.QCheckBox("assume the average chip is isotropic")
        self.chk_isotropic.setToolTip(
            "Only used when no second azimuth was recorded. Fixes the "
            "transverse-vs-axial gauge by assuming the median chip has equal "
            "sensitivity on all three axes. Fair for a monolithic part, but an "
            "assumption — record sweep C to measure it instead.")
        f3.addWidget(self.chk_isotropic, 3, 2, 1, 3)

        self.btn_solve_roll = QtWidgets.QPushButton("Solve")
        self.btn_solve_roll.setEnabled(False)
        self.btn_solve_roll.clicked.connect(self.on_solve_roll)
        self.btn_apply_roll = QtWidgets.QPushButton("Apply to calibration")
        self.btn_apply_roll.setEnabled(False)
        self.btn_apply_roll.clicked.connect(self.on_apply_roll)
        row = QtWidgets.QWidget()
        rl = QtWidgets.QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.btn_solve_roll)
        rl.addWidget(self.btn_apply_roll)
        for text, slot in (("Clear sweeps", self.on_clear_sweeps),
                           ("Save sweeps", self.on_save_sweeps),
                           ("Load sweeps", self.on_load_sweeps)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            rl.addWidget(b)
        rl.addStretch(1)
        f3.addWidget(row, 4, 0, 1, 6)
        lay.addWidget(g3)

        self._superseded_boxes = (g2, g3)
        for box in self._superseded_boxes:
            box.setVisible(False)

        g4 = QtWidgets.QGroupBox("3. Calibration file and geometry")
        f4 = QtWidgets.QHBoxLayout(g4)
        for text, slot in (("Save calibration", self.on_save_cal),
                           ("Load calibration", self.on_load_cal),
                           ("Edit geometry", self.on_edit_geometry),
                           ("Reload geometry", self.on_reload_geometry)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            f4.addWidget(b)
        self.chk_vcm = QtWidgets.QCheckBox("subtract VCM")
        self.chk_vcm.setChecked(self.cal.subtract_vcm)
        self.chk_vcm.setToolTip(
            "Each chip's own virtual ground. The 16 differ by up to ~90 mV, "
            "which is ~1.4 mT of fake field at the 20 mT range. Leave this on.")
        self.chk_vcm.toggled.connect(self.on_vcm_toggle)
        f4.addWidget(self.chk_vcm)
        f4.addStretch(1)
        lay.addWidget(g4)

        self.cal_report = QtWidgets.QPlainTextEdit()
        self.cal_report.setReadOnly(True)
        self.cal_report.setFont(QtGui.QFont("Consolas", 9))
        lay.addWidget(self.cal_report, 1)
        self.refresh_cal_report()
        return w

    def _help_tab(self):
        """Searchable documentation, indexed from the README at startup.

        Read from disk once, here, rather than at each search: it is 75 kB and
        the file does not change while the window is open. If it did -- someone
        editing the README on the bench machine -- Reload picks it up without
        restarting.
        """
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        top = QtWidgets.QHBoxLayout()
        self.help_search = QtWidgets.QLineEdit()
        self.help_search.setPlaceholderText(
            "Search the documentation — try 'homing', 'roll sweep', "
            "'why is it loud', 'VCM'")
        self.help_search.setClearButtonEnabled(True)
        self.help_search.textChanged.connect(self._help_filter)
        btn_reload = QtWidgets.QPushButton("Reload")
        btn_reload.setToolTip("Re-read README.md, for when it has been edited "
                              "while this window was open.")
        btn_reload.clicked.connect(self._help_reload)
        top.addWidget(self.help_search, 1)
        top.addWidget(btn_reload)
        lay.addLayout(top)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.help_list = QtWidgets.QListWidget()
        self.help_list.currentRowChanged.connect(self._help_show)
        self.help_view = QtWidgets.QTextBrowser()
        self.help_view.setOpenExternalLinks(True)
        self.help_list.setMinimumWidth(300)
        split.addWidget(self.help_list)
        split.addWidget(self.help_view)
        # A truncated heading is a heading you cannot search by eye. These are
        # long and specific on purpose, so the list gets a real share.
        split.setSizes([430, 900])
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        lay.addWidget(split, 1)

        self.help_count = QtWidgets.QLabel("")
        self.help_count.setStyleSheet("color:#9aa3b2;")
        lay.addWidget(self.help_count)

        self._help_reload()
        return w

    def _help_reload(self):
        self.help_topics = ohelp.load_topics()
        self._help_filter(self.help_search.text()
                          if hasattr(self, "help_search") else "")

    def _help_filter(self, query):
        self.help_hits = ohelp.search(self.help_topics, query, limit=200)
        self.help_list.clear()
        for t in self.help_hits:
            item = QtWidgets.QListWidgetItem(
                ("    " if t.level > 2 else "") + t.title)
            item.setToolTip(f"{t.title}  —  {t.source}")
            if t.source == "this window":
                item.setForeground(QtGui.QColor("#8fc7ff"))
            self.help_list.addItem(item)
        n_all = len(self.help_topics)
        self.help_count.setText(
            f"{len(self.help_hits)} of {n_all} topics"
            + (f" matching '{query}'" if query.strip() else
               " — the README, plus the topics about this window in blue"))
        if self.help_hits:
            self.help_list.setCurrentRow(0)
        else:
            self.help_view.setMarkdown(
                f"### Nothing matches '{query}'\n\n"
                f"Search matches whole words in a heading first, then the "
                f"text underneath. Try fewer words, or a term the "
                f"documentation would actually use — 'clkdiv' rather than "
                f"'sample rate setting'.")

    def _help_show(self, row):
        if not (0 <= row < len(self.help_hits)):
            return
        t = self.help_hits[row]
        self.help_view.setMarkdown(f"## {t.title}\n\n*from {t.source}*\n\n"
                                   + t.body)
        self.help_view.verticalScrollBar().setValue(0)

    def show_help_for(self, query):
        """Open the Help tab at a search. For 'explain this' buttons."""
        self.help_search.setText(query)
        tabs = self.help_search.window().findChild(QtWidgets.QTabWidget)
        if tabs is not None:
            for i in range(tabs.count()):
                if tabs.tabText(i) == "Help":
                    tabs.setCurrentIndex(i)
                    break

    def _health_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        top = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Analyse the last few seconds")
        b.clicked.connect(self.on_health)
        top.addWidget(b)
        b2 = QtWidgets.QPushButton("Export channel health CSV")
        b2.clicked.connect(self.on_export_health)
        top.addWidget(b2)
        self.chk_autodead = QtWidgets.QCheckBox(
            "auto-exclude dead sensors from statistics")
        self.chk_autodead.setChecked(True)
        top.addWidget(self.chk_autodead)
        top.addStretch(1)
        lay.addLayout(top)
        self.health_text = QtWidgets.QPlainTextEdit()
        self.health_text.setReadOnly(True)
        self.health_text.setFont(QtGui.QFont("Consolas", 9))
        self.health_text.setPlainText(
            "Run this with no magnet near the probe.\n\n"
            "It reports every one of the 64 raw channels: mean, noise, and "
            "whether it is railed or stuck.\n\n"
            "Read the VCM rows first. VCM carries no field, so noise there is "
            "analogue pickup in the cabling and grounding, not a sensor "
            "problem -- and on this probe it grows steadily along each "
            "concentrator, which is a wiring fault, not a calibration one.")
        lay.addWidget(self.health_text)
        return w

    # ---- stages ----------------------------------------------------------
    def _stage_tab(self):
        """Thorlabs LTS300C control and motorised field mapping.

        Laid out in the order you have to do things: find the stages, tell the
        software which one is which axis, reference them, then scan. Homing sits
        behind a confirmation because it drives the carriage into a limit switch
        at speed and the software has no idea what is bolted to it.
        """
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        g1 = QtWidgets.QGroupBox("1. Stages")
        f1 = QtWidgets.QGridLayout(g1)
        note = QtWidgets.QLabel(
            "The stages are exclusive-open USB devices: if the Thorlabs "
            "Kinesis application is running it owns all of them and none will "
            "appear here. Close Kinesis first.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#9aa3b2; font-size:11px;")
        f1.addWidget(note, 0, 0, 1, 4)

        self.stage_table = QtWidgets.QTableWidget(0, 5)
        self.stage_table.setHorizontalHeaderLabels(
            ["serial", "model", "axis", "position", "state"])
        self.stage_table.verticalHeader().setVisible(False)
        self.stage_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.stage_table.horizontalHeader().setStretchLastSection(True)
        self.stage_table.setMaximumHeight(140)
        f1.addWidget(self.stage_table, 1, 0, 1, 4)

        self.btn_stage_scan = QtWidgets.QPushButton("Find stages")
        # Not inlinable: clicked() would pass its `checked` bool straight into
        # `quiet`, and a button press must never be the silent variant.
        self.btn_stage_scan.clicked.connect(
            lambda: self.on_stage_find())               # noqa: PLW0108
        self.btn_stage_connect = QtWidgets.QPushButton("Connect")
        self.btn_stage_connect.clicked.connect(self.on_stage_connect)
        self.btn_stage_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_stage_disconnect.clicked.connect(self.on_stage_disconnect)
        self.btn_stage_disconnect.setEnabled(False)
        self.btn_stage_savemap = QtWidgets.QPushButton("Save axis map")
        self.btn_stage_savemap.setToolTip(
            "Records which serial number is which axis in stages.json, so the "
            "next session does not have to be told again.")
        self.btn_stage_savemap.clicked.connect(self.on_stage_save_map)
        f1.addWidget(self.btn_stage_scan, 2, 0)
        f1.addWidget(self.btn_stage_connect, 2, 1)
        f1.addWidget(self.btn_stage_disconnect, 2, 2)
        f1.addWidget(self.btn_stage_savemap, 2, 3)
        lay.addWidget(g1)

        g2 = QtWidgets.QGroupBox("2. Manual control")
        f2 = QtWidgets.QGridLayout(g2)
        f2.addWidget(QtWidgets.QLabel("jog step"), 0, 1)
        f2.addWidget(QtWidgets.QLabel("move to"), 0, 4)
        self.stage_rows = {}
        for r, ax in enumerate(("x", "y", "z"), start=1):
            lbl = QtWidgets.QLabel(ax)
            step = QtWidgets.QDoubleSpinBox()
            step.setRange(0.001, 100.0)
            step.setDecimals(3)
            step.setValue(1.0)
            step.setSuffix(" mm")
            minus = QtWidgets.QPushButton("−")
            plus = QtWidgets.QPushButton("+")
            minus.setMaximumWidth(36)
            plus.setMaximumWidth(36)
            target = QtWidgets.QDoubleSpinBox()
            target.setRange(0.0, 300.0)
            target.setDecimals(3)
            target.setSuffix(" mm")
            go = QtWidgets.QPushButton("Go")
            home = QtWidgets.QPushButton("Home")
            home.setToolTip("Drives this axis into its limit switch. Make sure "
                            "the probe and its cabling are clear first.")
            minus.clicked.connect(
                lambda _, a=ax, s=step: self.on_stage_jog(a, -s.value()))
            plus.clicked.connect(
                lambda _, a=ax, s=step: self.on_stage_jog(a, +s.value()))
            go.clicked.connect(
                lambda _, a=ax, t=target: self.on_stage_goto(a, t.value()))
            home.clicked.connect(lambda _, a=ax: self.on_stage_home([a]))
            for c, wdg in enumerate((lbl, step, minus, plus, target, go, home)):
                f2.addWidget(wdg, r, c)
            self.stage_rows[ax] = {"step": step, "target": target,
                                   "present": False,
                                   "widgets": (lbl, step, minus, plus, target,
                                               go, home)}

        # A jog only reaches full speed if it is long enough to ramp up to it,
        # so the same setting is quiet at 2 mm and loud at 20. This is where
        # that ceiling lives, next to the buttons that run into it.
        f2.addWidget(QtWidgets.QLabel("speed"), 4, 0)
        self.spin_stage_vel = QtWidgets.QDoubleSpinBox()
        # The box cannot ask for more than the module will do, so the number on
        # screen is always the number that will be used.
        self.spin_stage_vel.setRange(0.1, ostage.MAX_VEL_MM_S)
        self.spin_stage_vel.setDecimals(2)
        self.spin_stage_vel.setValue(ostage.DEFAULT_VEL_MM_S)
        self.spin_stage_vel.setSuffix(" mm/s")
        self.spin_stage_vel.setToolTip(
            "Top speed any move is allowed to reach. Kinesis ships these "
            "stages at 20 mm/s, which puts anything past about a 5 mm step "
            "into the motor's resonance -- that is the noise. Lower this, not "
            f"the step size. Hard ceiling {ostage.MAX_VEL_MM_S:g} mm/s: every "
            "move re-applies this profile first, so nothing that touched the "
            "controller in between can make a move faster than this.")
        self.spin_stage_acc = QtWidgets.QDoubleSpinBox()
        self.spin_stage_acc.setRange(0.1, 50.0)
        self.spin_stage_acc.setDecimals(2)
        self.spin_stage_acc.setValue(ostage.DEFAULT_ACCEL_MM_S2)
        self.spin_stage_acc.setSuffix(" mm/s²")
        self.spin_stage_acc.setToolTip(
            "How hard it ramps. Also sets what a SHORT move peaks at: a move "
            "too brief to reach the speed cap tops out at sqrt(accel × "
            "distance) instead.")
        self.btn_stage_speed = QtWidgets.QPushButton("Apply")
        self.btn_stage_speed.setToolTip(
            "Send this profile to every connected axis and record it in "
            "stages.json, so later sessions and the field map use it too. "
            "Homing has its own speed and is not affected.")
        self.btn_stage_speed.clicked.connect(self.on_stage_speed)
        f2.addWidget(self.spin_stage_vel, 4, 1)
        f2.addWidget(self.spin_stage_acc, 4, 2)
        f2.addWidget(self.btn_stage_speed, 4, 3)
        self.lbl_stage_peak = QtWidgets.QLabel("")
        self.lbl_stage_peak.setStyleSheet("color:#9aa3b2;")
        f2.addWidget(self.lbl_stage_peak, 4, 4, 1, 3)
        for sp in (self.spin_stage_vel, self.spin_stage_acc):
            sp.valueChanged.connect(self._update_peak_label)
        self._update_peak_label()

        self.btn_stage_homeall = QtWidgets.QPushButton("Home all axes")
        self.btn_stage_homeall.clicked.connect(lambda: self.on_stage_home(None))
        # Not the emergency stop -- that one is the red button in the top
        # right of the toolbar, where it is reachable from every tab. This is
        # the graded version: profiled, so nothing loses steps, and it does
        # not latch. Two buttons because they answer different questions, and
        # a person who wants "that is far enough" should not have to re-home
        # three axes to get it.
        self.btn_stage_stop = QtWidgets.QPushButton("Stop moving")
        self.btn_stage_stop.setStyleSheet(
            "background:#7a1f1f; color:#fff; font-weight:bold;")
        self.btn_stage_stop.setToolTip(
            "End the move in progress and stop any running field map, without "
            "latching the machine off. Profiled, so positions stay "
            "trustworthy and nothing needs re-homing.\n\n"
            "For 'something is wrong', use EMERGENCY STOP in the top right "
            "(or Esc) — that one is immediate and refuses all further motion "
            "until it is reset.")
        self.btn_stage_stop.clicked.connect(self.on_stage_stop)
        f2.addWidget(self.btn_stage_homeall, 5, 0, 1, 3)
        f2.addWidget(self.btn_stage_stop, 5, 5, 1, 2)
        lay.addWidget(g2)

        g3 = QtWidgets.QGroupBox("3. Field map")
        f3 = QtWidgets.QGridLayout(g3)
        blurb = QtWidgets.QLabel(
            "Move, stop, average, repeat. Each point is averaged at the "
            "carriers' own 200 kSPS with the stream released, so a 5 s point "
            "reaches roughly 0.04 µT — the same argument as the pose capture, "
            "applied to position instead of roll.\n"
            "Every row is traversed in the same direction so leadscrew "
            "backlash cannot stamp a comb into the map.")
        blurb.setWordWrap(True)
        f3.addWidget(blurb, 0, 0, 1, 6)
        for c, head in enumerate(("", "start", "stop", "step"), start=0):
            f3.addWidget(QtWidgets.QLabel(head), 1, c)
        self.scan_rows = {}
        for r, ax in enumerate(("x", "y", "z"), start=2):
            chk = QtWidgets.QCheckBox(ax)
            spins = []
            for val, lo, hi in ((0.0, 0.0, 300.0), (10.0, 0.0, 300.0),
                                (1.0, 0.001, 300.0)):
                sp = QtWidgets.QDoubleSpinBox()
                sp.setRange(lo, hi)
                sp.setDecimals(3)
                sp.setValue(val)
                sp.setSuffix(" mm")
                spins.append(sp)
            f3.addWidget(chk, r, 0)
            for c, sp in enumerate(spins, start=1):
                f3.addWidget(sp, r, c)
            chk.toggled.connect(self._update_scan_estimate)
            for sp in spins:
                sp.valueChanged.connect(self._update_scan_estimate)
            self.scan_rows[ax] = {"chk": chk, "spins": spins}

        self.spin_scan_s = QtWidgets.QDoubleSpinBox()
        self.spin_scan_s.setRange(0.5, 120.0)
        self.spin_scan_s.setValue(5.0)
        self.spin_scan_s.setSuffix(" s")
        self.spin_scan_s.valueChanged.connect(self._update_scan_estimate)
        self.spin_scan_settle = QtWidgets.QDoubleSpinBox()
        self.spin_scan_settle.setRange(0.0, 30.0)
        self.spin_scan_settle.setValue(oscan.DEFAULT_SETTLE_S)
        self.spin_scan_settle.setSuffix(" s")
        self.spin_scan_settle.setToolTip(
            "Wait after the stage stops before averaging. The controller says "
            "'stopped' when its profile ends, which is not when a cantilevered "
            "probe stops ringing. Too short does not look like an error — it "
            "looks like a field gradient.")
        self.spin_scan_settle.valueChanged.connect(self._update_scan_estimate)
        f3.addWidget(QtWidgets.QLabel("average per point"), 5, 0, 1, 2)
        f3.addWidget(self.spin_scan_s, 5, 2)
        f3.addWidget(QtWidgets.QLabel("settle"), 5, 3)
        f3.addWidget(self.spin_scan_settle, 5, 4)

        self.lbl_scan_est = QtWidgets.QLabel("")
        self.lbl_scan_est.setStyleSheet("color:#9aa3b2;")
        f3.addWidget(self.lbl_scan_est, 6, 0, 1, 6)

        self.btn_scan_start = QtWidgets.QPushButton("Start field map")
        self.btn_scan_start.clicked.connect(self.on_scan_start)
        self.btn_scan_abort = QtWidgets.QPushButton("Abort")
        self.btn_scan_abort.setEnabled(False)
        self.btn_scan_abort.clicked.connect(self.on_scan_abort)
        self.bar_scan = QtWidgets.QProgressBar()
        self.bar_scan.setTextVisible(True)
        f3.addWidget(self.btn_scan_start, 7, 0, 1, 2)
        f3.addWidget(self.btn_scan_abort, 7, 2)
        f3.addWidget(self.bar_scan, 7, 3, 1, 3)
        lay.addWidget(g3)

        lay.addStretch(1)
        self._update_scan_estimate()
        self._set_stage_controls_enabled(False)
        return w

    def _set_stage_controls_enabled(self, on):
        """Connected or not. Everything finer than that is _sync_stage_controls."""
        for row in self.stage_rows.values():
            row["present"] = on and row["present"]
        self.btn_stage_disconnect.setEnabled(on)
        self.btn_stage_connect.setEnabled(not on)
        self._sync_stage_controls()

    def on_stage_find(self, quiet=False):
        """Fill the stage table from the bus. Returns True if any were found.

        `quiet` is for the automatic connect, which must not throw a modal in
        front of someone who only asked to connect the probe: a rig with no
        stages plugged in is a normal way to use this window.
        """
        try:
            serials = ostage.list_devices()
        except ostage.StageError as exc:
            self.log.log(f"stages: {exc}")
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "Stages", str(exc))
            return False
        if not serials:
            msg = ("No stages on the bus. The Kinesis application is running "
                   "and holds all of them — close it and try again."
                   if ostage.kinesis_is_running() else
                   "No stages on the bus. Check power and USB.")
            self.log.log(f"stages: {msg}")
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "Stages", msg)
            return False

        saved = {v: k for k, v in ostage.load_axis_map().items()}
        self.stage_table.setRowCount(len(serials))
        self.stage_combos = {}
        for r, s in enumerate(serials):
            # The description is decoration. TLI_GetDeviceInfo fails while
            # another process still has the device open, and letting that kill
            # the whole listing would mean a stale handle somewhere else on the
            # machine takes the Stages tab down with it.
            try:
                desc = ostage.device_info(s)["description"]
            except ostage.StageError as exc:
                desc = "—"
                self.log.log(f"stages: {s} description unavailable ({exc})")
            self.stage_table.setItem(r, 0, QtWidgets.QTableWidgetItem(s))
            self.stage_table.setItem(r, 1, QtWidgets.QTableWidgetItem(desc))
            combo = QtWidgets.QComboBox()
            combo.addItems(["—", "x", "y", "z"])
            if s in saved:
                combo.setCurrentText(saved[s])
            self.stage_table.setCellWidget(r, 2, combo)
            self.stage_combos[s] = combo
            self.stage_table.setItem(r, 3, QtWidgets.QTableWidgetItem("—"))
            self.stage_table.setItem(r, 4, QtWidgets.QTableWidgetItem("closed"))
        self.stage_table.resizeColumnsToContents()
        # Sized for the widest thing each column will ever hold, not for what
        # is in it right now: the position and state cells are rewritten five
        # times a second, and re-fitting the columns on every poll makes the
        # whole table twitch.
        for col, width in ((0, 90), (1, 90), (2, 70), (3, 120)):
            self.stage_table.setColumnWidth(col, width)
        self.log.log(f"stages: found {len(serials)} controller(s): "
                     f"{', '.join(serials)}")
        if not saved:
            self.log.log(
                "stages: no axis map yet. Assign x/y/z in the table — the "
                "software cannot work out which stage is which physical "
                "direction, and guessing would silently transpose the "
                "coordinate frame of every map you take. Use the CLI's "
                "'octobee/motion/stage.py identify' to wiggle each one if unsure.")
        return True

    def _axis_map_from_table(self):
        mapping = {}
        for serial, combo in getattr(self, "stage_combos", {}).items():
            name = combo.currentText()
            if name != "—":
                if name in mapping:
                    raise ostage.StageError(
                        f"axis '{name}' is assigned to two stages")
                mapping[name] = serial
        return mapping

    def on_stage_save_map(self):
        try:
            mapping = self._axis_map_from_table()
        except ostage.StageError as exc:
            QtWidgets.QMessageBox.warning(self, "Axis map", str(exc))
            return
        if not mapping:
            QtWidgets.QMessageBox.information(
                self, "Axis map", "Assign at least one axis first.")
            return
        ostage.save_axis_map(mapping)
        self.log.log(f"stages: wrote {ostage.AXIS_CONFIG} — "
                     + ", ".join(f"{k}={v}" for k, v in mapping.items()))

    def on_stage_connect(self):
        if not getattr(self, "stage_combos", None):
            self.on_stage_find()
            if not getattr(self, "stage_combos", None):
                return
        try:
            mapping = self._axis_map_from_table()
        except ostage.StageError as exc:
            QtWidgets.QMessageBox.warning(self, "Axis map", str(exc))
            return
        if not mapping:
            QtWidgets.QMessageBox.information(
                self, "Stages",
                "Assign at least one stage to an axis before connecting.")
            return
        # The mounting comes from stages.json, not from the table: which way a
        # bracket runs is not something the axis dropdown can express, and
        # building the stages without it would silently ignore a reversed axis
        # and mirror every map taken from this window.
        stages = _stage_set_for(mapping)
        self.btn_stage_connect.setEnabled(False)
        self.log.log("stages: opening " + ", ".join(
            f"{k}={v}" for k, v in mapping.items()))

        def work(emit):
            stages.open()
            for st in stages:
                emit(f"{st.name}: {st.model}, travel "
                     f"{st.travel_mm[0]:g}..{st.travel_mm[1]:g} mm"
                     + ("" if st.homed else ", NOT HOMED"))
                if st.invert:
                    emit(f"{st.name}: mounted REVERSED — rig zero is at the "
                         f"far end of its travel, so homing parks it at "
                         f"{st.travel_mm[1]:g} mm, not 0")
                if st.calibration_file() is None:
                    emit(f"{st.name}: no Thorlabs calibration file loaded — "
                         f"on-axis accuracy is ~47 um rather than <5 um")

        self._stage_pending = stages
        self._stage_action("connect", work)

    def on_stage_disconnect(self):
        if self._scan_worker is not None and self._scan_worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, "Stages", "A field map is running. Abort it first.")
            return
        if self.stages is not None:
            self.stages.close()
            self.stages = None
        self._set_stage_controls_enabled(False)
        for r in range(self.stage_table.rowCount()):
            self.stage_table.setItem(r, 3, QtWidgets.QTableWidgetItem("—"))
            self.stage_table.setItem(r, 4, QtWidgets.QTableWidgetItem("closed"))
        self.log.log("stages: closed")

    def _stage_action(self, what, fn):
        """Run one blocking stage command in a worker, one at a time."""
        if self._estop_reason is not None:
            self.log.log(f"stages: {what} refused — emergency stop is latched "
                         f"({self._estop_reason}). Reset it first.")
            return
        if self.motion_busy():
            self.log.log("stages: busy — wait for the current move to finish")
            return
        self._stage_worker = StageWorker(what, fn)
        self._stage_worker.progress.connect(self.log.log)
        self._stage_worker.done.connect(self.on_stage_action_done)
        self.register_motion_worker(self._stage_worker)
        self._stage_worker.start()
        self._sync_stage_controls()

    def on_stage_action_done(self, what, error):
        self.retire_motion_worker(self._stage_worker)
        if error:
            self.log.log(f"stages: {what} failed — {error}")
            if what == "connect":
                self._stage_pending = None
                self.btn_stage_connect.setEnabled(True)
            self._sync_stage_controls()
            if self._estop_reason is not None:
                # The move failed because the machine was stopped, which the
                # operator already knows -- they pressed the button, and the
                # status bar is red. A modal on top of that is noise, and with
                # several axes in flight it is several modals.
                return
            QtWidgets.QMessageBox.warning(self, "Stages", error)
            return
        if what == "connect":
            self.stages = self._stage_pending
            self._stage_pending = None
            if self._estop_reason is not None:
                # Latched while disconnected, or latched and then reconnected
                # to clear it. Neither may release the machine: the interlock
                # belongs to the rig, not to whichever StageSet object happens
                # to be alive.
                self.stages.interlock.trip(self._estop_reason)
                self.log.log("stages: connected, but motion stays latched off "
                             f"— {self._estop_reason}")
            for ax, row in self.stage_rows.items():
                have = ax in self.stages.names
                row["present"] = have
                if have:
                    # The soft limit, not the travel: a spin box that offers a
                    # number the axis is not allowed to go to is an invitation
                    # to find that out by pressing Go.
                    lo, hi = self.stages[ax].limit_mm
                    row["target"].setRange(lo, hi)
            self._set_stage_controls_enabled(True)
            for ax, row in self.scan_rows.items():
                have = ax in self.stages.names
                row["chk"].setEnabled(have)
                if not have:
                    row["chk"].setChecked(False)
                else:
                    lo, hi = self.stages[ax].limit_mm
                    for sp in row["spins"][:2]:     # start and stop, not step
                        sp.setRange(lo, hi)
                for sp in row["spins"]:
                    sp.setEnabled(have)
            vel, acc = next(iter(self.stages)).vel_params
            self.spin_stage_vel.setValue(vel)
            self.spin_stage_acc.setValue(acc)
            self.log.log(f"stages: connected — {vel:.3g} mm/s, "
                         f"{acc:.3g} mm/s² profile")
            self._report_stage_envelope()
            # Deferred: this opens a modal, and doing that from inside a
            # worker's completion signal blocks the thread's own teardown.
            QtCore.QTimer.singleShot(0, self._prompt_home_if_needed)
        else:
            self.log.log(f"stages: {what} done")
        self._sync_stage_controls()

    def _report_stage_envelope(self):
        """Say what will actually stop the head, once, at connect.

        An axis with no soft limit is not obviously different from one that
        has them: both move, both report a position, and the difference only
        shows up the first time a scan range is set a little too wide. Saying
        it at connect is the cheapest place to put that.
        """
        bare, whole = [], []
        for name in self.stages.names:
            st = self.stages[name]
            lo, hi = st.limit_mm
            if st.limit_mm != st.travel_mm:
                self.log.log(f"stages: {name} may use {lo:g}..{hi:g} mm "
                             f"(soft limit inside {st.travel_mm[0]:g}.."
                             f"{st.travel_mm[1]:g} mm of travel)")
            elif st.limit_declared:
                # Declared as the full travel: someone looked and there is
                # nothing in the way. Same movement as an axis nobody has
                # configured, but it is an answer rather than a gap, so it is
                # reported once and does not nag.
                whole.append(name)
            else:
                bare.append(name)
        if whole:
            self.log.log(f"stages: {', '.join(whole)} may use their whole "
                         f"travel — declared in {ostage.AXIS_CONFIG}, not "
                         f"merely unset")
        if bare:
            self.log.log(
                f"stages: {', '.join(bare)} have NO soft limit — the whole "
                f"travel is allowed, and the only thing that will stop the "
                f"head short of the fixture is the limit switch. Set "
                f'"limit_mm" per axis in {ostage.AXIS_CONFIG}.')
        self.log.log("stages: home order "
                     + " → ".join(self.stages.home_sequence()))

    def _update_peak_label(self):
        """Say what the current profile means for the step sizes in the boxes.

        The setting is a ceiling, not a speed: what a jog actually reaches
        depends on how far it goes. Showing both makes the connection between
        "I raised the step size" and "it got loud" visible before it happens.
        """
        vel = self.spin_stage_vel.value()
        acc = self.spin_stage_acc.value()
        parts = []
        for d in (1.0, 5.0, 20.0):
            parts.append(f"{d:g} mm → "
                         f"{ostage.peak_speed_mm_s(d, vel, acc):.1f}")
        self.lbl_stage_peak.setText("peak " + ",  ".join(parts) + " mm/s")

    def on_stage_speed(self):
        """Apply the profile to every open axis, and remember it."""
        if self.stages is None:
            return
        vel = self.spin_stage_vel.value()
        acc = self.spin_stage_acc.value()

        def apply(emit):
            for st in self.stages:
                st.set_vel_params(vel, acc)
            ostage.save_axis_motion(velocity_mm_s=vel, accel_mm_s2=acc)
            emit(f"stages: {vel:g} mm/s, {acc:g} mm/s² on "
                 f"{', '.join(self.stages.names)} — saved to stages.json")

        self._stage_action("set motion profile", apply)

    def on_stage_jog(self, axis, delta):
        if self.stages is None or axis not in self.stages.names:
            return
        st = self.stages[axis]
        self._stage_action(f"jog {axis} {delta:+g} mm",
                           lambda emit: st.move_by(delta))

    def on_stage_goto(self, axis, mm):
        if self.stages is None or axis not in self.stages.names:
            return
        st = self.stages[axis]
        if not st.position_trusted:
            QtWidgets.QMessageBox.warning(
                self, "Position cannot be trusted",
                f"Axis {axis}: {st.distrust_reason}.\n\n"
                f"Its position counter cannot be believed, so an absolute "
                f"move would go somewhere arbitrary. Home it first, or use "
                f"the jog buttons, which are relative and do not need a "
                f"reference.")
            return
        self._stage_action(f"move {axis} to {mm:g} mm",
                           lambda emit: st.move_to(mm))

    def on_stage_home(self, axes):
        if self.stages is None:
            return
        wanted = set(axes or self.stages.names)
        # Ordered even for a subset: home_sequence() is the order that is safe
        # on this rig, and "the two axes you happened to tick" is not.
        names = [n for n in self.stages.home_sequence() if n in wanted]
        if not names:
            return
        one_at_a_time = ("" if len(names) == 1 else
                        f"\n\nThey go one at a time, in this order: "
                        f"{' → '.join(names)}.")
        reply = QtWidgets.QMessageBox.warning(
            self, "Home",
            f"Homing drives {', '.join(names)} into the limit switch at full "
            f"homing speed, across the whole travel — past any soft limit, "
            f"which does not apply to homing.{one_at_a_time}\n\n"
            f"Is the probe head — and its cabling — clear of the entire range "
            f"of movement?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel)
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            self.log.log("stages: homing cancelled")
            return
        self._home_axes(names)

    def _home_axes(self, names):
        """Home these axes, one at a time.

        The caller has already established it is safe. Sequential, where this
        used to start all of them together: with three carriages sweeping
        their full travel at once, whether they foul each other depends on
        which one happens to be slower, and that is not a property anything
        here can check. See StageSet.home_all.
        """
        stages = [self.stages[n] for n in names]

        def work(emit):
            for st in stages:
                # Between axes, not only within one: an emergency stop raises
                # out of home() on its own, but the plain "Stop moving" does
                # not latch, so without this it would halt one carriage and
                # then send the next one off across its whole travel.
                if self._stage_worker.aborted:
                    emit(f"homing stopped before {st.name}")
                    return
                st.home(wait=False)
                st.wait(timeout_s=300.0, what="homing")
                st.trust_after_homing()
                emit(f"{st.name}: homed, at {st.position_mm:.4f} mm")

        self._stage_action(f"home {', '.join(names)}", work)

    def _prompt_home_if_needed(self):
        """Offer to reference the axes, once, when they first come up.

        An unhomed stage still reports a position, and that number is whatever
        was left in the counter -- so the one moment this is worth raising is
        now, before anyone has read a coordinate off the screen and believed
        it. It stays a question rather than an automatic move because homing
        drives the carriage into the limit switch across the whole travel,
        with the probe head and its cable dress mounted; only the person in
        the room can say that is clear.
        """
        if self.stages is None:
            return
        unhomed = [n for n in self.stages.names
                   if not self.stages[n].position_trusted]
        if not unhomed:
            self.log.log("stages: every axis is already referenced")
            return
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Stages are not referenced")
        box.setText(f"Axes {', '.join(unhomed)} have not been homed.")
        box.setInformativeText(
            "Until they are, their position readings are whatever was left in "
            "the counter, absolute moves are refused and a field map has no "
            "origin. Jogging is relative and works either way.\n\n"
            "Homing drives each axis into its limit switch across the whole "
            "travel. Is the probe head — and its cabling — clear of the "
            "entire range of movement?")
        yes = box.addButton("Home all axes now",
                            QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Not yet", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is yes:
            self._home_axes(self.stages.names)
        else:
            self.log.log(f"stages: {', '.join(unhomed)} left unreferenced — "
                         f"home from the Stages tab before mapping")

    # ---- stopping -------------------------------------------------------

    def register_motion_worker(self, worker):
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
        self._motion_workers = [w for w in self._motion_workers
                                if w is not worker and _is_running(w)]
        self._motion_workers.append(worker)

    def retire_motion_worker(self, worker):
        """This worker has finished; stop counting it as busy.

        Called from the done handlers rather than waiting for isRunning() to
        go false, because a worker emits done() from inside run() -- so at the
        moment the handler executes the thread can still report itself as
        running, and the controls it should have re-enabled stay grey until
        something else happens to refresh them.

        The reference is kept, not dropped. A QThread garbage-collected while
        its run() is still unwinding takes the process with it.
        """
        self._motion_workers = [w for w in self._motion_workers if w is not worker]
        self._retired_workers.append(worker)
        del self._retired_workers[:-8]

    def motion_busy(self):
        """True if any thread is currently allowed to command motion."""
        self._motion_workers = [w for w in self._motion_workers if _is_running(w)]
        return bool(self._motion_workers)

    def _abort_motion_workers(self):
        """Ask every registered worker to stop. Returns how many were running."""
        n = 0
        for w in list(self._motion_workers):
            if _is_running(w):
                n += 1
                if hasattr(w, "abort"):
                    w.abort()
        return n

    def on_estop(self, _checked=False):
        """The button, the Esc key and the watchdog all come through here."""
        self.trigger_estop("operator pressed the emergency stop")

    def trigger_estop(self, reason):
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
        # Kept on the window as well as on the StageSet: the set is created at
        # connect, so without this a stop pressed while disconnected would be
        # forgotten by the time something opened the stages again.
        already = self._estop_reason is not None
        if not already:
            self._estop_reason = reason
        errors = []
        if self.stages is not None:
            try:
                errors = self.stages.emergency_stop(reason)
            except Exception as exc:
                errors = [f"{type(exc).__name__}: {exc}"]
        stopped = self._abort_motion_workers()
        self._refresh_estop_ui()
        if already:
            self.log.log(f"EMERGENCY STOP (again) — already latched: "
                         f"{self._estop_reason}")
            return
        self.log.log(f"*** EMERGENCY STOP — {reason} ***")
        if self.stages is None:
            self.log.log("  the stages are not connected here; nothing to "
                         "stop over USB. If something is moving, cut power "
                         "to the controllers.")
        else:
            self.log.log(f"  {', '.join(self.stages.names)}: immediate stop, "
                         f"all further motion refused until reset")
        if stopped:
            self.log.log(f"  {stopped} running job(s) aborted")
        for e in errors:
            self.log.log(f"  STOP FAILED on {e} — CUT POWER TO THE "
                         f"CONTROLLERS IF ANYTHING IS STILL MOVING")
        if errors:
            QtWidgets.QMessageBox.critical(
                self, "Emergency stop did not reach every axis",
                "The stop could not be delivered to:\n\n  "
                + "\n  ".join(errors)
                + "\n\nIf anything is still moving, cut power to the "
                  "controllers now. This is what a hardware emergency stop "
                  "is for; the software one needs the USB link to work.")

    def on_estop_reset(self):
        """Clear the latch, after saying what is still not true afterwards."""
        if self._estop_reason is None:
            return
        lost = self.stages.untrusted() if self.stages is not None else []
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Reset the emergency stop")
        box.setText(f"Latched by: {self._estop_reason}")
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
        box.addButton("Stay stopped", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is not go:
            return
        was = self._estop_reason
        self._estop_reason = None
        if self.stages is not None:
            _, lost = self.stages.reset_interlock()
        self.log.log(f"emergency stop reset (was: {was}) — motion allowed")
        for name, why in lost:
            self.log.log(f"  {name}: absolute moves still refused — {why}")
        self._refresh_estop_ui()

    def _refresh_estop_ui(self):
        """Make the latched state impossible to miss or to mistake."""
        latched = self._estop_reason is not None
        self.btn_estop_reset.setVisible(latched)
        self.btn_estop.setText("■  STOPPED" if latched
                               else "■  EMERGENCY STOP")
        if latched:
            if self._state_before_estop is None:
                self._state_before_estop = self.lbl_state.text()
            self.lbl_state.setText(f"EMERGENCY STOP — {self._estop_reason}")
            self.lbl_state.setStyleSheet(
                "color:#fff; background:#c1121f; font-weight:bold; padding:1px 8px;")
        else:
            self.lbl_state.setStyleSheet("")
            # Put back what the status bar said before, rather than leaving
            # "EMERGENCY STOP" sitting there unstyled -- which reads as a
            # machine that is still stopped.
            self.lbl_state.setText(self._state_before_estop or "disconnected")
            self._state_before_estop = None
        self._sync_stage_controls()

    def _sync_stage_controls(self):
        """Motion controls are live only when this window may command motion.

        Two things gate them, and only one of them used to. The interlock is
        the new one. The other is that something else is already driving: a
        field map or a guided-magnet run owns the axes for its duration, and
        the jog, Go and Home buttons stayed enabled behind it -- so a Home
        during a raster would start a second thread commanding the same axis
        through the same DLL, the scan's wait() would watch the homing move as
        if it were its own, and the map would carry on being written with
        every remaining point taken somewhere other than where it says.
        """
        live = (self.stages is not None and self._estop_reason is None
                and not self.motion_busy())
        for row in self.stage_rows.values():
            for wdg in row["widgets"]:
                wdg.setEnabled(live and row.get("present", True))
        self.btn_stage_homeall.setEnabled(live)
        self.btn_stage_speed.setEnabled(live)
        self.btn_scan_start.setEnabled(live)
        # Stopping stays available in every state that is not "no stages".
        self.btn_stage_stop.setEnabled(self.stages is not None)

    def on_stage_stop(self):
        """End the move in progress, without latching the machine off.

        The graded one: a profiled stop, so nothing loses steps and the
        positions stay trustworthy. This is "that is far enough", where the
        red button is "something is wrong". Both abort the running jobs --
        a stop that leaves the raster thread free to command the next point
        is not a stop either.

        Not routed through the worker queue: a stop that has to wait behind the
        move it is trying to interrupt is not a stop button.
        """
        stopped = self._abort_motion_workers()
        if stopped:
            self.log.log(f"stages: {stopped} running job(s) will stop after "
                         f"the point in flight")
        if self.stages is None:
            return
        try:
            self.stages.stop_all()
            self.log.log("stages: stopped")
        except ostage.StageError as exc:
            self.log.log(f"stages: stop failed — {exc}")

    def _stage_watchdog(self, snap):
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
            if key not in self._estop_alarms:
                self._estop_alarms.add(key)
                self.trigger_estop(f"{name} reported a motion error")
        elif snap["at_hard_limit"] and not snap["moving"]:
            key = (name, "limit")
            if key not in self._estop_alarms:
                self._estop_alarms.add(key)
                # Homing ends on the limit switch by design, and so does any
                # axis parked there afterwards -- so this is a warning, not a
                # stop, unless it happened while something was driving.
                if self.motion_busy():
                    self.trigger_estop(
                        f"{name} reached a hard limit switch during a move")
                else:
                    self.log.log(f"stages: {name} is sitting on a hard limit "
                                 f"switch")
        else:
            self._estop_alarms.discard((name, "error"))
            self._estop_alarms.discard((name, "limit"))

    def refresh_stage_table(self):
        """Position and state per stage, off the DLL's own polling cache."""
        if self.stages is None or not self.stage_table.rowCount():
            return
        by_serial = {st.serial: st for st in self.stages}
        for r in range(self.stage_table.rowCount()):
            item = self.stage_table.item(r, 0)
            st = by_serial.get(item.text() if item else None)
            if st is None or not st.is_open:
                continue
            try:
                snap = st.snapshot()
            except ostage.StageError:
                continue
            self._stage_watchdog(snap)
            # Before opening, all the bus can say is "APT Stepper Motor
            # Controller"; the actual model only arrives with the settings.
            if snap["model"] and self.stage_table.item(r, 1).text() != snap["model"]:
                self.stage_table.setItem(
                    r, 1, QtWidgets.QTableWidgetItem(snap["model"]))
            self.stage_table.setItem(
                r, 3, QtWidgets.QTableWidgetItem(f"{snap['position_mm']:.4f} mm"))
            state = "moving" if snap["moving"] else "idle"
            if not snap["trusted"]:
                # "NOT HOMED" was the old wording and it was too narrow: the
                # counter can be untrustworthy on an axis whose homed bit is
                # still set, which is the case that actually hurts.
                state += (", NOT HOMED" if not snap["homed"]
                          else ", POSITION LOST")
            if snap["error"]:
                state += ", MOTION ERROR"
            if snap["at_hard_limit"]:
                state += ", ON HARD LIMIT"
            if snap["interlocked"]:
                state += ", STOPPED"
            if snap["invert"]:
                # Worth carrying in the always-visible row: the rig number and
                # the number on the controller's own display disagree on a
                # reversed axis, and that is alarming until you know why.
                state += f"  [reversed, device {snap['position_dev_mm']:.3f}]"
            self.stage_table.setItem(r, 4, QtWidgets.QTableWidgetItem(state))

    # ---- field map -------------------------------------------------------
    def _scan_grid(self):
        axes = {}
        for ax, row in self.scan_rows.items():
            if not row["chk"].isChecked():
                continue
            start, stop, step = (sp.value() for sp in row["spins"])
            if stop < start:
                raise ValueError(
                    f"{ax}: stop ({stop:g}) is before start ({start:g}). The "
                    f"scan always runs in the +ve direction so that every "
                    f"point is approached from the same side and backlash "
                    f"cannot bias alternate rows.")
            axes[ax] = oscan.parse_axis_spec(f"{start}:{stop}:{step}")
        if not axes:
            raise ValueError("tick at least one axis to scan")
        return oscan.ScanGrid(axes)

    def _update_scan_estimate(self):
        try:
            grid = self._scan_grid()
        except ValueError as exc:
            self.lbl_scan_est.setText(str(exc))
            return
        n = len(grid)
        # Per point: the average itself, the settle, and the move. The move is
        # a guess -- it depends on step size and velocity -- so this is a floor
        # to plan around, not a promise.
        per = self.spin_scan_s.value() + self.spin_scan_settle.value() + 2.0
        total = n * per
        self.lbl_scan_est.setText(
            f"{n} points — {grid.describe()} — roughly "
            f"{total / 60:.0f} min ({total / 3600:.1f} h) at {per:.1f} s/point")

    def on_scan_start(self):
        if self.stages is None:
            return
        if self.motion_busy():
            return
        if self._estop_reason is not None:
            QtWidgets.QMessageBox.warning(
                self, "Field map",
                f"The emergency stop is latched: {self._estop_reason}.\n\n"
                f"Reset it before starting a map.")
            return
        try:
            grid = self._scan_grid()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Field map", str(exc))
            return
        unhomed = [(n, self.stages[n].distrust_reason) for n in grid.names
                   if not self.stages[n].position_trusted]
        if unhomed:
            QtWidgets.QMessageBox.warning(
                self, "Field map",
                "These axes' position counters cannot be believed, so the map "
                "would have no origin. Home them first.\n\n  "
                + "\n  ".join(f"{n}: {why}" for n, why in unhomed))
            return
        # Checked here as well as inside every move, because a range that is
        # out of bounds should be a refusal before a six-hour job starts, not
        # a point failure two hours in.
        outside = []
        for name, pts in grid.axes.items():
            lo, hi = self.stages[name].limit_mm
            first, last = float(min(pts)), float(max(pts))
            if not (lo <= first and last <= hi):
                outside.append(f"{name}: asks for {first:g}..{last:g} mm, "
                               f"allowed {lo:g}..{hi:g} mm")
        if outside:
            QtWidgets.QMessageBox.warning(
                self, "Field map",
                "The scan range goes outside what these axes are allowed to "
                "use:\n\n  " + "\n  ".join(outside)
                + f"\n\nEither shorten the range, or change \"limit_mm\" in "
                  f"{ostage.AXIS_CONFIG} — having measured that the head "
                  f"really can go there.")
            return

        n = len(grid)
        per = self.spin_scan_s.value() + self.spin_scan_settle.value() + 2.0
        reply = QtWidgets.QMessageBox.question(
            self, "Field map",
            f"{n} points over {grid.describe()}.\n"
            f"Roughly {n * per / 60:.0f} minutes.\n\n"
            f"This takes the carriers off the live stream and puts them back "
            f"on their own 200 kSPS clock for the duration — the live plot "
            f"will stop.\n\nStart?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel)
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        if self.act_record.isChecked():
            self.act_record.setChecked(False)
        if isinstance(self.source, LiveSource):
            self.source.stop()
        self.source = None
        self.lbl_state.setText("field map in progress...")
        self.bar_scan.setRange(0, n)
        self.bar_scan.setValue(0)
        self.btn_scan_abort.setEnabled(True)
        self.act_snapshot.setEnabled(False)

        self._scan_worker = ScanWorker(
            self.hosts, self.stages, grid, self.spin_scan_s.value(),
            self.cal, self.spin_scan_settle.value(), self.prev_clkdiv,
            # Which coils were on, at what current, and where the probe was
            # bolted. None of it can be recovered from the numbers later, and
            # a map without it is a table of vectors at nowhere in particular.
            extra_meta={"machine": self.machine.to_scan_meta(
                self.coils, self._machine_stage_mm(ignore_tracking=True))})
        # As with the snapshot: the worker restores the clock itself, so
        # clearing this now means a failed scan cannot leave a stale value
        # for disconnect to apply on top.
        self.prev_clkdiv = {}
        self._scan_worker.message.connect(self.log.log)
        self._scan_worker.progress.connect(self.on_scan_progress)
        self._scan_worker.done.connect(self.on_scan_done)
        self._scan_t0 = time.time()
        self.register_motion_worker(self._scan_worker)
        self._scan_worker.start()
        self._sync_stage_controls()

    def on_scan_progress(self, i, n, where, sem_ut):
        self.bar_scan.setValue(i)
        elapsed = time.time() - self._scan_t0
        eta = elapsed / max(i, 1) * (n - i)
        self.bar_scan.setFormat(f"%v / %m — {eta / 60:.0f} min left")
        self.lbl_state.setText(f"field map {i}/{n}: {where}")
        if i == 1 or i % 10 == 0 or i == n:
            self.log.log(f"  point {i}/{n} at {where}: noise {sem_ut:.3f} uT")

    def on_scan_abort(self):
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._scan_worker.abort()
            self.btn_scan_abort.setEnabled(False)
            self.log.log("field map: stopping after the point in flight")

    def on_scan_done(self, fm, error):
        self.retire_motion_worker(self._scan_worker)
        self.btn_scan_abort.setEnabled(False)
        self.act_snapshot.setEnabled(True)
        self.bar_scan.setFormat("%v / %m")
        if self._estop_reason is None:
            self.lbl_state.setText("disconnected")
        self._sync_stage_controls()
        if error:
            self.log.log(f"field map failed: {error}")
            QtWidgets.QMessageBox.warning(self, "Field map", error)
        if fm is None or not len(fm):
            return
        path = fm.save(os.path.join(
            self.out_dir, time.strftime("fieldmap_%Y%m%d_%H%M%S")))
        sem = np.array([s.get("sem_ut", np.nan) for s in fm.stats])
        self.log.log(
            f"field map: {len(fm)} of {fm.meta['n_requested']} points, "
            f"noise median {np.nanmedian(sem):.3f} uT / worst "
            f"{np.nanmax(sem):.3f} uT -> {path}")
        self.log.log("field map: the carriers are still off the live stream — "
                     "press Connect to go back to the live view.")

    # ---- the machine around the probe ------------------------------------
    def _machine_tab(self):
        """Where the head is inside the coil set, and which coils are live.

        A field map is a table of vectors at rig millimetres, and rig
        millimetres mean nothing on their own: the same numbers describe a
        useful measurement and a useless one depending on where the probe was
        bolted and what was switched on. This tab is where both are declared,
        and it draws the consequence rather than asking anyone to picture it.

        Laid out in the order the answers arrive: which machine, which coils
        are carrying current, then where the probe sits in it.
        """
        w = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(split)

        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)

        note = QtWidgets.QLabel(
            "The coil file is in machine coordinates; the probe is placed by "
            "saying where its mounting flange sits in them. The stage reading "
            "is added along the rig's own axes, so the drawn head follows a "
            "jog. Nothing here is measured — it is the frame the measurement "
            "is written down in, and it is only as good as the numbers typed "
            "into it.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#9aa3b2; font-size:11px;")
        lay.addWidget(note)

        # ---- 1. the coil set ----
        g1 = QtWidgets.QGroupBox("1. Coil set")
        f1 = QtWidgets.QGridLayout(g1)
        self.ed_coil_file = QtWidgets.QLineEdit(self.machine.coil_file)
        self.ed_coil_file.setPlaceholderText(
            "a simsopt configuration file, e.g. designA_after_scaled.json")
        btn_browse = QtWidgets.QPushButton("Browse…")
        btn_browse.clicked.connect(self.on_machine_browse)
        btn_load = QtWidgets.QPushButton("Load")
        btn_load.clicked.connect(lambda: self.on_machine_load())  # noqa: PLW0108
        f1.addWidget(self.ed_coil_file, 0, 0, 1, 2)
        f1.addWidget(btn_browse, 0, 2)
        f1.addWidget(btn_load, 0, 3)

        self.lbl_coil_summary = QtWidgets.QLabel("no coil set loaded")
        self.lbl_coil_summary.setWordWrap(True)
        self.lbl_coil_summary.setStyleSheet("color:#9aa3b2; font-size:11px;")
        f1.addWidget(self.lbl_coil_summary, 1, 0, 1, 4)

        f1.addWidget(QtWidgets.QLabel("winding radius"), 2, 0)
        self.spin_coil_radius = QtWidgets.QDoubleSpinBox()
        self.spin_coil_radius.setRange(0.5, 500.0)
        self.spin_coil_radius.setDecimals(1)
        self.spin_coil_radius.setSuffix(" mm")
        self.spin_coil_radius.setValue(self.machine.coil_radius_mm)
        self.spin_coil_radius.setToolTip(
            "The circular cross-section swept along each coil centreline — "
            "the volume the probe cannot enter. A simsopt file has no such "
            "number in it: the optimiser works with infinitely thin "
            f"filaments, so this starts at {omach.DEFAULT_COIL_RADIUS_MM:g} mm "
            "and is a guess until the real conductor has been measured. "
            "Clearance is reported to this surface, not to the centreline.")
        self.spin_coil_radius.valueChanged.connect(self.on_machine_radius)
        f1.addWidget(self.spin_coil_radius, 2, 1)
        lay.addWidget(g1)

        # ---- 2. currents ----
        g2 = QtWidgets.QGroupBox("2. Which coils are carrying current")
        f2 = QtWidgets.QGridLayout(g2)
        f2.addWidget(QtWidgets.QLabel("configuration"), 0, 0)
        self.cmb_coil_config = QtWidgets.QComboBox()
        self.cmb_coil_config.setToolTip(
            "A simsopt file usually holds the same coils several times over, "
            "once per current set the optimiser was working with. The "
            "geometry is identical; only the currents differ.")
        self.cmb_coil_config.currentIndexChanged.connect(self.on_machine_config)
        f2.addWidget(self.cmb_coil_config, 0, 1)
        f2.addWidget(QtWidgets.QLabel("× scale"), 0, 2)
        self.spin_coil_scale = QtWidgets.QDoubleSpinBox()
        self.spin_coil_scale.setRange(-1000.0, 1000.0)
        self.spin_coil_scale.setDecimals(5)
        self.spin_coil_scale.setSingleStep(0.01)
        self.spin_coil_scale.setValue(self.machine.current_scale)
        self.spin_coil_scale.setToolTip(
            "Every current in the chosen configuration is multiplied by this. "
            "The file's numbers are the design point in amp-turns; a bench "
            "test at a few hundred amp-turns is that design point scaled down "
            "by a factor of a hundred or more. The ratios between coils stay "
            "as optimised, which is what makes a single number enough.")
        self.spin_coil_scale.valueChanged.connect(self.on_machine_scale)
        f2.addWidget(self.spin_coil_scale, 0, 3)

        self.tbl_coils = QtWidgets.QTableWidget(0, 4)
        self.tbl_coils.setHorizontalHeaderLabels(
            ["coil", "where", "amp-turns", "curve"])
        self.tbl_coils.verticalHeader().setVisible(False)
        self.tbl_coils.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_coils.horizontalHeader().setStretchLastSection(True)
        # Tall enough that a six-coil machine needs no scrolling: the point
        # of the table is to be able to see at a glance what is switched on.
        self.tbl_coils.setMaximumHeight(230)
        self.tbl_coils.itemChanged.connect(self.on_machine_coil_toggled)
        f2.addWidget(self.tbl_coils, 1, 0, 1, 4)

        btn_all_on = QtWidgets.QPushButton("All on")
        btn_all_on.clicked.connect(lambda: self.on_machine_all(True))
        btn_all_off = QtWidgets.QPushButton("All off")
        btn_all_off.clicked.connect(lambda: self.on_machine_all(False))
        hint = QtWidgets.QLabel(
            "A coil that is switched off is still solid copper: clearance is "
            "checked against every coil, energised or not.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9aa3b2; font-size:11px;")
        f2.addWidget(btn_all_on, 2, 0)
        f2.addWidget(btn_all_off, 2, 1)
        f2.addWidget(hint, 2, 2, 1, 2)
        lay.addWidget(g2)

        # ---- 3. placement ----
        g3 = QtWidgets.QGroupBox("3. Where the probe is")
        f3 = QtWidgets.QGridLayout(g3)
        self.machine_pose_spins = {}
        fields = (("x_mm", "x", " mm", -20000.0, 20000.0),
                  ("y_mm", "y", " mm", -20000.0, 20000.0),
                  ("z_mm", "z", " mm", -20000.0, 20000.0),
                  ("yaw_deg", "yaw", " °", -360.0, 360.0),
                  ("pitch_deg", "pitch", " °", -360.0, 360.0),
                  ("roll_deg", "roll", " °", -360.0, 360.0))
        for i, (attr, label, suffix, lo, hi) in enumerate(fields):
            row, col = divmod(i, 3)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(1)
            spin.setSuffix(suffix)
            spin.setValue(getattr(self.machine.pose, attr))
            spin.valueChanged.connect(self.on_machine_pose_edited)
            f3.addWidget(QtWidgets.QLabel(label), row * 2, col)
            f3.addWidget(spin, row * 2 + 1, col)
            self.machine_pose_spins[attr] = spin
        self.machine_pose_spins["x_mm"].setToolTip(
            "The probe's mounting flange — the end of the tube the boards "
            "count up from — in machine millimetres, with every stage at its "
            "zero. Yaw turns the assembly about the machine's Z, which is the "
            "axis of the torus; it is applied last, so it swings the whole "
            "probe round the machine whatever pitch and roll are set to.")

        self.chk_track_stage = QtWidgets.QCheckBox("follow the stages")
        self.chk_track_stage.setChecked(self.machine.track_stage)
        self.chk_track_stage.setToolTip(
            "Add the live stage reading to the pose, so the drawn head moves "
            "as the rig does. Off, the drawing stays where it was put — which "
            "is what you want when planning with the stages disconnected.")
        self.chk_track_stage.toggled.connect(self.on_machine_track)
        btn_zero = QtWidgets.QPushButton("Stage zero is here")
        btn_zero.setToolTip(
            "Take the stages' current reading as the position the pose above "
            "describes. Press it with the rig parked where the flange was "
            "measured, and every later move is drawn relative to that.")
        btn_zero.clicked.connect(self.on_machine_zero_stage)
        f3.addWidget(self.chk_track_stage, 4, 0)
        f3.addWidget(btn_zero, 4, 1, 1, 2)

        self.lbl_machine_stage = QtWidgets.QLabel("stages not connected")
        self.lbl_machine_stage.setWordWrap(True)
        self.lbl_machine_stage.setStyleSheet("color:#9aa3b2; font-size:11px;")
        f3.addWidget(self.lbl_machine_stage, 5, 0, 1, 3)
        lay.addWidget(g3)

        self.lbl_clearance = QtWidgets.QLabel("no coil set loaded")
        self.lbl_clearance.setWordWrap(True)
        self.lbl_clearance.setStyleSheet(
            "font-size:15px; padding:6px; border:1px solid #2a2e3a;")
        lay.addWidget(self.lbl_clearance)

        row = QtWidgets.QHBoxLayout()
        btn_save = QtWidgets.QPushButton("Save placement")
        btn_save.setToolTip(f"Writes all of this to {self.args.machine}, so "
                            f"the next session opens where this one left off.")
        btn_save.clicked.connect(self.on_machine_save)
        btn_fit = QtWidgets.QPushButton("Fit machine")
        # Not inlinable: clicked() would pass its `checked` bool into a
        # method that takes no arguments.
        btn_fit.clicked.connect(
            lambda: self.machine_view.reset_camera())   # noqa: PLW0108
        btn_zoom = QtWidgets.QPushButton("Zoom to probe")
        btn_zoom.clicked.connect(
            lambda: self.machine_view.look_at_probe(self.machine.pose,
                                                    self._machine_stage_mm()))
        for b in (btn_save, btn_fit, btn_zoom):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        chk_reach = QtWidgets.QCheckBox("stage envelope")
        chk_reach.setChecked(True)
        chk_reach.setToolTip(
            "The box the flange can be driven through, from each axis's "
            "allowed travel — the volume this rig can actually reach without "
            "being unbolted and moved.")
        chk_reach.toggled.connect(
            lambda on: self.machine_view.set_reach_visible(on,
                                                           self.machine.pose))
        chk_names = QtWidgets.QCheckBox("coil labels")
        chk_names.setChecked(True)
        chk_names.toggled.connect(self.machine_view.set_labels_visible)
        row2.addWidget(chk_reach)
        row2.addWidget(chk_names)
        row2.addStretch(1)
        lay.addLayout(row2)
        lay.addStretch(1)

        split.addWidget(panel)
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(2, 2, 2, 2)
        legend = QtWidgets.QLabel(
            "amber: current flowing   ·   slate: switched off, still in the "
            "way   ·   green line: closest approach   ·   red: the probe is "
            "inside a winding")
        legend.setWordWrap(True)
        legend.setStyleSheet("color:#9aa3b2; font-size:11px;")
        rl.addWidget(legend)
        rl.addWidget(self.machine_view, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([440, 900])
        return w

    # ---- machine: loading and current ------------------------------------
    def on_machine_browse(self):
        start = os.path.dirname(self.ed_coil_file.text()) or os.getcwd()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Coil configuration", start, "simsopt JSON (*.json);;"
                                               "All files (*)")
        if path:
            self.ed_coil_file.setText(path)
            self.on_machine_load()

    def on_machine_load(self, path=None, quiet=False):
        """Read a coil file and hand it to the view.

        A failure is reported and then dropped: the rest of the window works
        perfectly well without a coil set, and the file most likely to fail is
        one that was moved, which is a thing to fix rather than a reason to
        lose the session.
        """
        path = path or self.ed_coil_file.text().strip()
        if not path:
            return
        problems = []
        coils = omach.CoilSet.load_or_none(path, on_error=problems.append)
        for msg in problems:
            self.log.log(f"WARNING: {msg}")
        if coils is None:
            if not quiet:
                QtWidgets.QMessageBox.warning(
                    self, "Coil set", "\n\n".join(problems)
                    or f"{path} could not be read.")
            return
        self.coils = coils
        self.machine.coil_file = path
        for lost in self.machine.adopt(coils):
            self.log.log(f"coil set: {lost}")
        self.log.log(f"coil set: {coils.note} — {path}")
        self._refresh_machine_controls()
        self.machine_view.set_coils(coils, self.machine.coil_radius_mm,
                                    self.machine.energised)
        self.machine_view.reset_camera()
        self.refresh_machine(force=True)

    def _refresh_machine_controls(self):
        """Push the config into the widgets without echoing back out again."""
        self._machine_quiet = True
        try:
            self.ed_coil_file.setText(self.machine.coil_file)
            self.cmb_coil_config.clear()
            if self.coils is not None:
                self.cmb_coil_config.addItems(self.coils.configurations)
                if self.machine.configuration in self.coils.configurations:
                    self.cmb_coil_config.setCurrentIndex(
                        self.coils.configurations.index(
                            self.machine.configuration))
                self.lbl_coil_summary.setText(self.coils.note)
            else:
                self.lbl_coil_summary.setText("no coil set loaded")
            self._fill_coil_table()
        finally:
            self._machine_quiet = False

    def _fill_coil_table(self):
        rows = list(self.coils) if self.coils is not None else []
        self.tbl_coils.setRowCount(len(rows))
        for r, coil in enumerate(rows):
            name = QtWidgets.QTableWidgetItem(coil.label)
            name.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable
                          | QtCore.Qt.ItemFlag.ItemIsEnabled)
            name.setCheckState(
                QtCore.Qt.CheckState.Checked if self.machine.is_on(coil.label)
                else QtCore.Qt.CheckState.Unchecked)
            name.setData(QtCore.Qt.ItemDataRole.UserRole, coil.label)
            self.tbl_coils.setItem(r, 0, name)
            amps = self.machine.current(coil)
            cells = (coil.where(),
                     f"{amps:+,.0f}" if abs(amps) < 10000 else
                     f"{amps / 1000:+,.2f} k",
                     coil.description)
            for c, text in enumerate(cells, start=1):
                item = QtWidgets.QTableWidgetItem(text)
                if not self.machine.is_on(coil.label):
                    item.setForeground(QtGui.QColor("#6a7182"))
                self.tbl_coils.setItem(r, c, item)
        self.tbl_coils.resizeColumnsToContents()

    def on_machine_config(self, _index):
        if self._machine_quiet or self.coils is None:
            return
        self.machine.configuration = self.cmb_coil_config.currentText()
        self._machine_quiet = True
        try:
            self._fill_coil_table()
        finally:
            self._machine_quiet = False
        self.log.log("coil set: " + self.machine.energised_summary(self.coils))

    def on_machine_scale(self, value):
        if self._machine_quiet:
            return
        self.machine.current_scale = float(value)
        self._machine_quiet = True
        try:
            self._fill_coil_table()
        finally:
            self._machine_quiet = False

    def on_machine_coil_toggled(self, item):
        if self._machine_quiet or item.column() != 0:
            return
        label = item.data(QtCore.Qt.ItemDataRole.UserRole)
        on = item.checkState() == QtCore.Qt.CheckState.Checked
        energised = [c for c in (self.machine.energised or []) if c != label]
        if on:
            energised.append(label)
        # Keep the file's own order, so the list in machine.json reads the same
        # way the table does however the boxes were clicked.
        order = self.coils.labels if self.coils is not None else energised
        self.machine.energised = [c for c in order if c in energised]
        self.machine_view.set_energised(self.machine.energised)
        self._machine_quiet = True
        try:
            self._fill_coil_table()
        finally:
            self._machine_quiet = False
        self.log.log("coil set: " + self.machine.energised_summary(self.coils))

    def on_machine_all(self, on):
        if self.coils is None:
            return
        self.machine.energised = list(self.coils.labels) if on else []
        self.machine_view.set_energised(self.machine.energised)
        self._machine_quiet = True
        try:
            self._fill_coil_table()
        finally:
            self._machine_quiet = False
        self.log.log("coil set: " + self.machine.energised_summary(self.coils))

    def on_machine_radius(self, value):
        if self._machine_quiet:
            return
        self.machine.coil_radius_mm = float(value)
        self.machine_view.set_radius(value)
        self.refresh_machine(force=True)

    # ---- machine: placement ----------------------------------------------
    def on_machine_pose_edited(self, _value=None):
        if self._machine_quiet:
            return
        for attr, spin in self.machine_pose_spins.items():
            setattr(self.machine.pose, attr, float(spin.value()))
        self.refresh_machine(force=True)

    def on_machine_track(self, on):
        self.machine.track_stage = bool(on)
        self.refresh_machine(force=True)

    def on_machine_zero_stage(self):
        stage = self._machine_stage_mm(ignore_tracking=True)
        if not stage:
            QtWidgets.QMessageBox.information(
                self, "Stage zero",
                "The stages are not connected, so there is no reading to "
                "take. Connect them in the Stages tab first.")
            return
        self.machine.pose.stage_zero_mm.update(
            {ax: float(v) for ax, v in stage.items()})
        self.log.log("machine: stage zero taken at "
                     + ", ".join(f"{k}={v:.3f} mm"
                                 for k, v in sorted(stage.items())))
        self.refresh_machine(force=True)

    def on_machine_save(self):
        path = self.machine.save(self.args.machine)
        self.log.log(f"machine: placement written to {path} — "
                     + self.machine.pose.describe())
        self.log.log("machine: " + self.machine.energised_summary(self.coils))

    def _machine_stage_mm(self, ignore_tracking=False):
        """The live stage reading, or None if there is nothing to read.

        Position is taken even from an axis whose counter is not trusted: this
        drawing is a picture, not an interlock, and refusing to draw an unhomed
        rig would hide exactly the situation someone is trying to understand.
        The label says so.
        """
        if self.stages is None:
            return None
        if not (ignore_tracking or self.machine.track_stage):
            return None
        out = {}
        for name in self.stages.names:
            st = self.stages[name]
            if not st.is_open:
                continue
            try:
                out[name] = float(st.position_mm)
            except ostage.StageError:
                continue
        return out or None

    def _machine_adopt_travel(self):
        """Take the stage envelope from the axes themselves, once they exist."""
        if self.stages is None:
            return
        for name in self.stages.names:
            if name not in omach.Placement.AXES:
                continue
            try:
                lo, hi = self.stages[name].limit_mm
            except (ostage.StageError, TypeError, ValueError):
                continue
            self.machine.pose.travel_mm[name] = (float(lo), float(hi))

    def refresh_machine(self, force=False):
        """Redraw the head where it is now, and re-measure the clearance.

        Runs off the stage timer, so it must be cheap when nothing has moved:
        the pose and the stage reading are hashed and the whole thing skipped
        when they are unchanged, which is most ticks.
        """
        if getattr(self, "machine_view", None) is None:
            return
        stage = self._machine_stage_mm()
        pose = self.machine.pose
        key = (pose.x_mm, pose.y_mm, pose.z_mm, pose.yaw_deg, pose.pitch_deg,
               pose.roll_deg, self.machine.coil_radius_mm,
               tuple(sorted(pose.stage_zero_mm.items())),
               None if stage is None else tuple(sorted(
                   (k, round(v, 3)) for k, v in stage.items())),
               id(self.coils))
        if key == self._machine_key and not force:
            return
        self._machine_key = key

        if self.stages is not None and not self._machine_travel_taken:
            self._machine_adopt_travel()
            self._machine_travel_taken = True

        with self.prof.time("machine view"):
            self.machine_view.set_pose(pose, stage)
            if self._probe_cloud is None:
                self._probe_cloud = omach.probe_cloud(self.geom)
            if self.coils is not None and len(self.coils):
                gap = omach.clearance(pose.to_machine(self._probe_cloud, stage),
                                      self.coils, self.machine.coil_radius_mm)
                self.machine_view.set_clearance(gap)
                self.lbl_clearance.setText(gap.text())
                self.lbl_clearance.setStyleSheet(
                    "font-size:15px; padding:6px; border:1px solid "
                    + ("#8a2020; background:#2a1416; color:#ff8a8a;"
                       if gap.collides else "#2a2e3a;"))
            else:
                self.machine_view.set_clearance(None)

        if stage is None:
            self.lbl_machine_stage.setText(
                "not following the stages — the head is drawn at the pose "
                "above" if self.stages is not None
                else "stages not connected — the head is drawn at the pose "
                     "above")
        else:
            where = ", ".join(f"{k}={stage[k]:.3f}" for k in sorted(stage))
            origin = pose.origin_mm(stage)
            untrusted = [n for n in sorted(stage)
                         if not self.stages[n].position_trusted]
            note = ("  (position not trusted on "
                    + ", ".join(untrusted) + " — home first)") if untrusted else ""
            self.lbl_machine_stage.setText(
                f"stage {where} mm  →  flange at ({origin[0]:+.0f}, "
                f"{origin[1]:+.0f}, {origin[2]:+.0f}) mm{note}")

    def _export_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        g1 = QtWidgets.QGroupBox("Continuous recording (the Record button)")
        f1 = QtWidgets.QGridLayout(g1)
        self.chk_csv = QtWidgets.QCheckBox("calibrated CSV (millitesla)")
        self.chk_csv.setChecked(True)
        self.chk_raw = QtWidgets.QCheckBox(
            "raw counts, full stream rate (.bin + .json sidecar)")
        self.chk_tube = QtWidgets.QCheckBox(
            "rotate into the common tube frame")
        self.chk_tube.setToolTip(
            "Chip-frame axes point 16 different ways, so only |B| is "
            "comparable between sensors. Tube frame makes the components "
            "comparable too -- at the cost of depending on the geometry file "
            "being right.")
        f1.addWidget(self.chk_csv, 0, 0)
        f1.addWidget(self.chk_tube, 0, 1)
        f1.addWidget(self.chk_raw, 1, 0, 1, 2)
        self.lbl_recinfo = QtWidgets.QLabel("not recording")
        f1.addWidget(self.lbl_recinfo, 2, 0, 1, 2)
        lay.addWidget(g1)

        g2 = QtWidgets.QGroupBox("One-shot exports")
        f2 = QtWidgets.QHBoxLayout(g2)
        for text, slot in (("Snapshot to .npz (full rate)", self.on_snapshot),
                           ("Sensor summary CSV", self.on_export_summary),
                           ("Full report JSON", self.on_export_json)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            f2.addWidget(b)
        self.spin_snap_s = QtWidgets.QDoubleSpinBox()
        self.spin_snap_s.setRange(0.2, 30.0)
        self.spin_snap_s.setValue(3.0)
        self.spin_snap_s.setSuffix(" s")
        f2.addWidget(QtWidgets.QLabel("snapshot length"))
        f2.addWidget(self.spin_snap_s)
        f2.addStretch(1)
        lay.addWidget(g2)

        self.export_log = QtWidgets.QPlainTextEdit()
        self.export_log.setReadOnly(True)
        self.export_log.setFont(QtGui.QFont("Consolas", 9))
        lay.addWidget(self.export_log, 1)
        return w

    def _apply_dark(self):
        self.setStyleSheet("""
            QWidget { background:#12141a; color:#d5d9e2; }
            QGroupBox { border:1px solid #2a2e3a; border-radius:4px;
                        margin-top:9px; padding-top:8px; }
            QGroupBox::title { subcontrol-origin: margin; left:8px;
                               color:#8f98a8; }
            QTabBar::tab { background:#1a1d26; padding:6px 14px; }
            QTabBar::tab:selected { background:#262b38; }
            QTableWidget { gridline-color:#2a2e3a;
                           alternate-background-color:#171a22; }
            QHeaderView::section { background:#1a1d26; border:0;
                                   padding:4px; color:#8f98a8; }
            QPushButton { background:#232836; border:1px solid #333a4a;
                          padding:5px 10px; border-radius:3px; }
            QPushButton:hover { background:#2c3244; }
            QPushButton:checked { background:#3a5070; }
            QPlainTextEdit { background:#0d0f14; border:1px solid #262b38; }
            QToolBar { background:#171a22; border-bottom:1px solid #262b38;
                       spacing:4px; padding:3px; }
        """)

    # ---- connection ------------------------------------------------------
    def _auto_connect(self):
        """The connect the window makes for itself when it opens.

        Marked so that a failure logs rather than throwing a modal across the
        window: the carriers being off is a normal way to arrive at this
        program -- to look at a saved capture, to edit a calibration -- and a
        dialog to dismiss on every launch would train you to dismiss dialogs.
        """
        if self.source is not None:
            return
        self._connect_was_automatic = True
        self.on_connect()

    def on_connect(self):
        if self.source is not None:
            return
        # A second attempt while the first is still in flight would start a
        # second worker against the same carriers -- and the automatic connect
        # means there is now something in flight that nobody pressed. Tracked
        # with a flag rather than by asking the thread: isRunning() is false
        # for the moment between start() and the thread actually running, and
        # that moment is exactly when a second call arrives.
        if self._connecting:
            return
        fs = float(self.cmb_rate.currentData())
        self.act_connect.setEnabled(False)
        self.lbl_state.setText("connecting...")
        self.log.log(f"connecting to {', '.join(self.hosts)}"
                     + (f", setting {fs/1000:g} kSPS" if fs else ""))
        self._connect_worker = ConnectWorker(self.hosts, fs)
        self._connect_worker.done.connect(self.on_connected)
        self._connect_worker.progress.connect(self.on_connect_progress)
        self._connecting = True
        self._connect_worker.start()
        # The rig is one instrument: the probe reads the field, the stages say
        # where it was read. Connecting one and not the other is a state the
        # operator has to remember to fix, and forgetting it does not announce
        # itself -- it just means the field map button is greyed out for no
        # visible reason. The two run in separate workers, so a bench with no
        # stages plugged in connects the probe exactly as before.
        self.connect_stages(quiet=True)

    def connect_stages(self, quiet=False):
        """Open the stages named in stages.json, without touching the probe.

        Returns False and logs if there is nothing to connect. Never raises
        into the caller: the probe half of the window has to come up whatever
        the motion hardware is doing.
        """
        try:
            return self._connect_stages(quiet)
        except Exception as exc:
            # Deliberately broad. This runs inside Connect, and the probe half
            # of the window must come up whatever the motion hardware, the
            # Kinesis install or a stale USB handle is doing.
            self.log.log(f"stages: not connected — {type(exc).__name__}: {exc}")
            self._stage_pending = None
            self.btn_stage_connect.setEnabled(True)
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "Stages", str(exc))
            return False

    def _connect_stages(self, quiet):
        if self.stages is not None or self._stage_pending is not None:
            return False
        if self._stage_worker is not None and self._stage_worker.isRunning():
            return False
        mapping = ostage.load_axis_map()
        if not mapping:
            self.log.log(
                "stages: no axis map in stages.json, so nothing was connected "
                "automatically. Assign x/y/z on the Stages tab once and every "
                "later session picks it up.")
            return False
        if not self.on_stage_find(quiet=quiet):
            return False
        missing = [ser for ser in mapping.values()
                   if ser not in (self.stage_combos or {})]
        if missing:
            self.log.log(f"stages: {', '.join(missing)} is in stages.json but "
                         f"not on the bus — connect from the Stages tab once "
                         f"it is back")
            return False
        stages = _stage_set_for(mapping)
        self._stage_pending = stages
        self.btn_stage_connect.setEnabled(False)
        self._stage_action("connect", lambda emit: stages.open())
        return True

    def on_connect_progress(self, msg):
        # Connecting involves several seconds of deliberate settling delays.
        # Without this the window sits on "connecting..." long enough to look
        # like it has hung, which is exactly how it was first reported.
        self.lbl_state.setText(f"connecting -- {msg}")
        self.log.log(msg)

    def on_connected(self, source, prev, error):
        self._connecting = False
        self.act_connect.setEnabled(True)
        was_automatic = self._connect_was_automatic
        self._connect_was_automatic = False
        if error:
            self.lbl_state.setText("disconnected")
            self.log.log(f"connect failed: {error}")
            if was_automatic:
                self.log.log(
                    "the automatic connect is the only thing that failed -- "
                    "everything that does not need the carriers still works, "
                    "and Connect will try again")
            else:
                QtWidgets.QMessageBox.critical(
                    self, "Connect failed",
                    f"{error}\n\nIf this says the stream closed immediately, "
                    f"something else owns port 4210 -- usually a Phoebus "
                    f"'Streaming Capture' still running on that box.")
            for h, p in (prev or {}).items():
                with contextlib.suppress(Exception):
                    olive.restore_rate(h, p)
            return
        self.prev_clkdiv = prev or {}
        self._set_source(source, "live")

    def _set_source(self, source, kind):
        self.source = source
        # Measure the running state, not the startup: building the window and
        # bringing up the GL context stalls the loop once, and leaving that in
        # the numbers makes a healthy session look blocked.
        self.prof.reset()
        self.lag.reset()
        # counts-per-volt is read off the box, so the counts scale is only
        # known once a source exists.
        if hasattr(self, "cmb_units"):
            self.on_units()
        self.act_disconnect.setEnabled(True)
        self.act_connect.setEnabled(kind != "live")
        self.lbl_state.setText(f"{kind}: {', '.join(source.hosts)} "
                               f"@ {source.fs_hz/1000:g} kSPS")
        self._reset_buffers()
        self.log.log(f"{kind} source running at {source.fs_hz/1000:g} kSPS, "
                     f"decimating to {self.out_rate:g} Hz for display and CSV")

    def on_disconnect(self):
        if self.act_record.isChecked():
            self.act_record.setChecked(False)
        if self.source is not None:
            self.source.stop()
            self.source = None
        # Symmetry with Connect: one button owns the whole rig. Leaving the
        # stages open would hold the USB devices against the Kinesis app and
        # against the next run of this window, which looks like a cabling
        # fault rather than a stale handle.
        if self.stages is not None:
            self.on_stage_disconnect()
        for h, p in self.prev_clkdiv.items():
            try:
                olive.restore_rate(h, p)
                self.log.log(f"{h}: clkdiv restored to {p}")
            except Exception as e:
                self.log.log(f"{h}: could not restore clkdiv: {e}")
        self.prev_clkdiv = {}
        self.act_disconnect.setEnabled(False)
        self.act_connect.setEnabled(True)
        self.lbl_state.setText("disconnected")

    def _reset_buffers(self):
        self.roll = Rolling(max(4, int(self.out_rate * self.window_s)))
        self.raw_hist = None
        self.raw_hist_n = 0
        self._last_health_t = 0.0
        self._last_dropped = 0

    def _calibration_changed(self, what):
        """
        Drop the buffered history whenever the conversion changes.

        The rolling buffer holds millitesla, not counts, so after a tare or a
        range change the points already in it were computed a different way.
        Left in place they poison exactly the things that read the buffer:
        plot autoscaling, the peak bars, and -- worst -- the baseline of a
        magnet pass, which would then measure the calibration step rather than
        the magnet.

        A running CSV recording is rolled over to a new file for the same
        reason, one level up. CsvRecorder stamps calibration_id and the whole
        conversion state into its header once, at open, so that a file can be
        matched back to the calibration that produced it. Carrying on writing
        into that file after the conversion changed would produce one CSV whose
        rows came from two different calibrations, under a header naming only
        the first -- which is worse than either file alone, because nothing
        about it looks wrong.
        """
        self.roll.clear()
        self.view3d.reset_scale()
        if hasattr(self, "cmb_units"):
            self.on_units()
        if self.collecting is not None and self.collecting["what"] == "magnet":
            self.btn_magnet.setChecked(False)
            self.log.log(f"magnet pass abandoned: {what} changed mid-pass")
        self._roll_over_csv(what)

    def _roll_over_csv(self, what):
        """Close the CSV in progress and open a fresh one with a new header."""
        if self.csv_rec is None:
            return
        old = self.csv_rec
        rows = old.n_rows
        old.close()
        path = orec.default_name("octobee", "csv", self.out_dir)
        self.csv_rec = orec.CsvRecorder(
            path, self.out_rate, self.cal, self.geom,
            tube_frame=self.chk_tube.isChecked(),
            meta={"hosts": ",".join(self.source.hosts) if self.source else "",
                  "stream_rate_hz": self.source.fs_hz if self.source else 0.0,
                  "continues": os.path.basename(old.path),
                  "rolled_over_because": f"{what} changed"})
        self.log.log(
            f"{what} changed while recording: closed {os.path.basename(old.path)} "
            f"at {rows} rows and started {os.path.basename(path)}, so each file "
            f"matches the calibration named in its own header. The raw .bin is "
            f"unaffected -- it holds counts, which no calibration changes.")

    @property
    def decim(self):
        if self.source is None:
            return 1
        return max(1, int(round(self.source.fs_hz / self.out_rate)))

    # ---- the acquisition tick --------------------------------------------
    def on_tick(self):
        if self.source is None:
            return
        with self.prof.time("acquisition tick (total)"):
            with self.prof.time("  socket read + decode"):
                blocks = self.source.read()
            if not blocks or blocks[0].shape[0] == 0:
                if getattr(self.source, "error", None):
                    self.log.log(f"stream error: {self.source.error}")
                    self.source.error = None
                return
            self.prof.note("samples per acquisition tick", blocks[0].shape[0])

            with self.prof.time("  keep raw history"):
                self._keep_raw(blocks)
            if self.raw_rec is not None:
                with self.prof.time("  write raw file"):
                    self.raw_rec.write(blocks)

            with self.prof.time("  counts -> tesla"):
                grouped = ocal.assemble(blocks, self.source.vpc,
                                        self.source.volt_offset)  # (n,16,4) V
                b = self.cal.to_mt(grouped)                        # (n,16,3) mT
            with self.prof.time("  decimate + buffer"):
                bd = ocal.decimate(b, self.decim)
                if bd.shape[0] == 0:
                    return
                self.roll.push(bd.astype(np.float32))

            if self.csv_rec is not None:
                with self.prof.time("  write CSV"):
                    self.csv_rec.write(bd)

            if self.collecting is not None:
                with self.prof.time("  magnet/tare collect"):
                    self._collect_block(b, grouped)

    # Drawing is deliberately NOT done here. This tick has to keep draining the
    # reader queues or the carriers overrun and the recording gets holes in it,
    # and repainting a 3D scene of 16 boards costs far more than the arithmetic
    # above. Everything visual runs on its own timer.

    def on_view_tick(self):
        """Redraw the live view. Its rate is the user's to choose."""
        if self.source is None or self.paused:
            return
        recent = self.roll.view()
        if recent.shape[0] < 2:
            return
        t0 = time.perf_counter()
        with self.prof.time("view tick (total)"):
            if self.chk_3d.isChecked():
                with self.prof.time("  3D head: build vectors"):
                    k = min(8, recent.shape[0])
                    fs = self.view3d.update_fields(recent[-k:].mean(axis=0))
                if (self.chk_auto.isChecked()
                        and abs(fs - self.spin_fs.value()) > 1e-4):
                    self.spin_fs.blockSignals(True)
                    self.spin_fs.setValue(max(fs, 0.001))
                    self.spin_fs.blockSignals(False)
            with self.prof.time("  live plot setData"):
                self.plot.update_data(recent, self.out_rate)
            with self.prof.time("  peak bars"):
                # Peak over a trailing half second, not the instantaneous
                # value: a magnet passed by hand is over well inside one
                # refresh, so sampling one point would miss most passes.
                n = max(2, min(recent.shape[0],
                               int(self.out_rate * self.bars.window_s)))
                self.bars.update_values(
                    np.linalg.norm(recent[-n:], axis=-1).max(axis=0),
                    self.cal.dead)
        self._note_draw_time((time.perf_counter() - t0) * 1000.0)

    def _note_draw_time(self, ms):
        """
        Watch how long a redraw costs and back off if it cannot keep up.

        A machine without working GPU acceleration falls back to software
        OpenGL, where painting the 3D head can take seconds. The symptom is a
        window that seems fine until data starts arriving and then locks solid.
        Rather than leave that to be diagnosed by hand, measure it and slow the
        redraw down -- acquisition and recording are on another clock and are
        not affected either way.
        """
        self._draw_ms = 0.75 * self._draw_ms + 0.25 * ms if self._draw_ms else ms
        interval = self.view_timer.interval()
        if self._draw_ms > 0.7 * interval and interval < MAX_VIEW_INTERVAL_MS:
            new = min(MAX_VIEW_INTERVAL_MS, interval * 2)
            self.view_timer.setInterval(new)
            self.log.log(
                f"a redraw is taking {self._draw_ms:.0f} ms, more than the "
                f"{interval} ms budget -- slowing the view to "
                f"{1000.0/new:.1f} Hz. Acquisition and recording are unaffected. "
                f"If this keeps happening, untick '3D': on a machine without "
                f"GPU acceleration the 3D head is by far the most expensive "
                f"thing on screen.")

    def _keep_raw(self, blocks):
        """Keep the last few seconds of raw counts for the health analysis."""
        cap = int(self.source.fs_hz * RAW_HISTORY_S)
        if self.raw_hist is None:
            self.raw_hist = [deque() for _ in blocks]
        for i, blk in enumerate(blocks):
            self.raw_hist[i].append(blk)
        self.raw_hist_n += blocks[0].shape[0]
        while self.raw_hist_n > cap and len(self.raw_hist[0]) > 1:
            drop = self.raw_hist[0][0].shape[0]
            for d in self.raw_hist:
                d.popleft()
            self.raw_hist_n -= drop

    def _raw_arrays(self, seconds=None):
        """
        Raw counts from the tail of the history, per box.

        `seconds` matters for speed, not just convenience: concatenating the
        whole 5 s history is tens of megabytes of copying, and doing that on
        every refresh starves the reader threads until the box's queues
        overflow. The periodic health check asks for a short window; the
        Diagnostics button asks for everything.
        """
        if not self.raw_hist or not self.raw_hist[0]:
            return None
        if seconds is None:
            return [np.concatenate(list(d), axis=0) for d in self.raw_hist]
        want = max(1, int(self.source.fs_hz * seconds))
        out = []
        for d in self.raw_hist:
            blocks, n = [], 0
            for blk in reversed(d):
                blocks.append(blk)
                n += blk.shape[0]
                if n >= want:
                    break
            if not blocks:
                return None
            out.append(np.concatenate(blocks[::-1], axis=0)[-want:])
        return out

    def on_slow_tick(self):
        if self.source is None:
            return
        recent = self.roll.view()
        if recent.shape[0] < 2:
            return

        # Health first: it decides which sensors are excluded, and everything
        # that follows scales its axes around that decision. It runs on a slower
        # clock than the display, because scanning 64 channels is the most
        # expensive thing in this loop and the answer changes only when a
        # connector does.
        now = time.time()
        if now - self._last_health_t >= HEALTH_PERIOD_S:
            raw = self._raw_arrays(HEALTH_WINDOW_S)
            if raw is not None:
                self._last_health_t = now
                with self.prof.time("  channel health scan"):
                    rows = ocal.channel_health(
                        raw, self.source.vpc, self.source.hosts,
                        self.source.volt_offset)
                self.last_health = rows
                if self.chk_autodead.isChecked():
                    dead = ocal.suggest_dead(rows)
                    if dead != self.cal.dead:
                        self.cal.dead = dead
                        self._mark_dead_checkboxes(dead)
                        if dead:
                            self.log.log(
                                f"excluding {', '.join(sorted(dead))}: "
                                + "; ".join(
                                    f"{k} {v[1]}" for k, v in
                                    ocal.health_verdict(rows).items()
                                    if v[0] == "dead"))
        if self.last_health:
            table = orec.sensor_table(self.cal, recent.astype(np.float64),
                                      self.last_health,
                                      self.source.temperatures(), geom=self.geom)
            self.table.update_rows(table)
            self._last_table = table

        self.plot.set_dead(self.cal.dead)
        self.view3d.set_dead(self.cal.dead)

        # A dropped block is a hole in whatever is being recorded, so say so
        # rather than leaving it as a number in the corner of the status bar.
        dropped = self.source.stats().get("dropped blocks", 0)
        if dropped > self._last_dropped:
            if self.csv_rec is not None or self.raw_rec is not None:
                self.log.log(f"WARNING: {dropped - self._last_dropped} block(s) "
                             f"dropped while recording -- the file has a gap "
                             f"there. Lower the output rate or the stream rate.")
            self._last_dropped = dropped

        st = dict(self.source.stats())
        if self._draw_ms:
            st["draw ms"] = round(self._draw_ms, 1)
        eff = 1000.0 / max(self.view_timer.interval(), 1)
        want = float(self.cmb_view.currentData())
        # Show the rate actually being achieved, marked when the automatic
        # backoff has taken it below what was asked for.
        st["view Hz"] = f"{eff:.1f}" + (" (auto)" if eff < want * 0.95 else "")
        self.lbl_rate.setText("  |  ".join(
            f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
            for k, v in st.items()))
        self._update_rec_label()

    # ---- collection (tare, magnet pass) ----------------------------------
    def start_collect(self, what, seconds):
        if self.source is None:
            self.log.log("not connected")
            return
        self.collecting = {"what": what, "blocks": [], "n": 0,
                           "need": int(seconds * self.source.fs_hz),
                           "peak": None, "baseline": None, "tag": None,
                           "decim": self.decim}
        self.log.log(f"collecting {seconds:g} s for {what}...")

    def _collect_block(self, b, grouped=None):
        """
        Accumulate one acquisition block into whatever collection is running.

        b        (n,16,3) fully calibrated mT -- offset, gain and matrix applied
        grouped  (n,16,4) volts straight off assemble(), nothing applied

        Which of the two a mode wants is not a detail: anything that MEASURES a
        correction has to work from `grouped`, or it fits on top of the
        correction already loaded and the answer depends on where you started.
        Only the magnet pass, which reports a field rather than deriving a
        calibration, legitimately uses `b`.
        """
        c = self.collecting
        if c["what"] == "magnet":
            # Deviation from the field that was there when the pass STARTED.
            # Re-deriving the baseline from the rolling window each block would
            # let the magnet drag the baseline along with it and shrink the peak.
            dev = np.linalg.norm(b - c["baseline"][None, :, :], axis=-1)
            best = dev.max(axis=0)
            c["peak"] = best if c["peak"] is None else np.maximum(c["peak"], best)
            c["n"] += b.shape[0]
            return

        # Both remaining modes measure a calibration, so both take the
        # uncorrected field. Decimate hard while we are at it: for a sweep the
        # information is in the shape of the ellipse, and a couple of hundred
        # hertz resolves a hand roll a thousand times over.
        if grouped is None:
            return
        raw = self.cal.to_mt(grouped, apply_zero=False, apply_gain=False,
                             apply_matrix=False)
        c["blocks"].append(ocal.decimate(raw, max(1, c["decim"])))
        c["n"] += b.shape[0]
        if c["n"] < c["need"]:
            return

        data = np.concatenate(c["blocks"], axis=0)
        what, tag = c["what"], c["tag"]
        self.collecting = None
        if what == "sweep":
            self._finish_sweep(data, tag)
        else:
            self._finish_tare(data)

    def _finish_tare(self, data):
        """
        Set the zero from an UNCORRECTED capture.

        `data` is what to_mt(apply_zero=False, apply_gain=False,
        apply_matrix=False) produced, which is what zero_mt is defined against:
        Calibration applies offset, then gain, then matrix, so the zero is a
        pre-gain quantity.

        This used to reconstruct it instead, as `data + zero_mt`, from the
        fully corrected buffer. That inverts the chain only when the gain trim
        is 1.0 and the matrix is the identity -- so after a magnet-pass trim or
        an applied roll calibration the stored zero came out scaled by the
        trim, silently. Collecting the uncorrected field in the first place
        means there is nothing to invert.
        """
        z = self.cal.tare(data)
        self.log.log(f"zeroed on {data.shape[0]} points; "
                     f"largest offset removed {np.abs(z).max():.4f} mT "
                     f"(S{int(np.argmax(np.abs(z).max(axis=1)))+1})")
        self._calibration_changed("the zero point")
        self.refresh_cal_report()


    # ---- Earth-field roll calibration ------------------------------------
    def start_sweep(self, tag, seconds):
        """Record one hand-rolled sweep in the mounting orientation `tag`."""
        if self.source is None:
            self.log.log("not connected")
            return
        fs = self.source.fs_hz
        self.collecting = {"what": "sweep", "tag": tag, "blocks": [], "n": 0,
                           "need": int(seconds * fs), "peak": None,
                           "baseline": None,
                           # aim for a few hundred Hz, whatever the ADC is at
                           "decim": max(1, int(fs / 200))}
        self.log.log(f"rolling sweep {tag}: roll the tube steadily through at "
                     f"least two full turns over the next {seconds:g} s")

    def _finish_sweep(self, data, tag):
        sw = opc.RollSweep(data, tag=tag,
                           ranges_mt=self.cal.ranges_mt.copy(),
                           temps_c=self._last_temps())
        self.sweeps[tag] = sw
        amp = sw.amplitudes()
        quiet = [f"S{i+1}" for i in range(N_SENSORS)
                 if amp[i] < opc.MIN_SEED_AMPLITUDE_MT
                 and not self.cal.is_dead(i + 1)]
        self.log.log(f"sweep {tag}: {len(sw)} points, median transverse swing "
                     f"{np.median(amp)*1e3:.2f} uT"
                     + (f"; SAW ALMOST NOTHING: {', '.join(quiet)}" if quiet else ""))
        self._refresh_sweep_label()

    def _last_temps(self):
        t = getattr(self, "last_temps_c", None)
        return None if t is None else np.asarray(t, float)

    def _refresh_sweep_label(self):
        if not self.sweeps:
            self.lbl_sweeps.setText("no sweeps recorded")
            self.btn_solve_roll.setEnabled(False)
            return
        bits = [f"{t} ({len(s)} pts)" for t, s in sorted(self.sweeps.items())]
        self.lbl_sweeps.setText("recorded: " + ", ".join(bits))
        self.btn_solve_roll.setEnabled(True)

    def on_clear_sweeps(self):
        self.sweeps.clear()
        self.pose_solution = None
        self.btn_apply_roll.setEnabled(False)
        self._refresh_sweep_label()
        self.log.log("roll sweeps cleared")

    def on_solve_roll(self):
        if not self.sweeps:
            return
        try:
            sol = opc.solve_roll(list(self.sweeps.values()), self.geom,
                                 self.spin_bearth.value(),
                                 dead=sorted(self.cal.dead),
                                 anisotropy=("assume_isotropic"
                                             if self.chk_isotropic.isChecked()
                                             else "solve"))
        except (ValueError, np.linalg.LinAlgError) as e:
            # Deliberately not a modal. Solving is a diagnostic step you may run
            # repeatedly while getting a sweep right, and a dialog you have to
            # dismiss in the middle of a live acquisition is worse than useless.
            self.log.log(f"roll solve failed: {e}")
            self.cal_report.setPlainText(f"Roll solve failed:\n\n{e}")
            self.pose_solution = None
            self.btn_apply_roll.setEnabled(False)
            return
        self.pose_solution = sol
        ids = opc.identify_faces(sol, self.geom)
        text = [sol.report(dead=self.cal.dead), ""]
        text.append("face mapping: " + ("agrees with probe_geometry.json"
                                        if ids["agrees"] else
                                        "DISAGREES at " + ", ".join(ids["mismatch"])))
        text.append(f"ambient uniformity: {opc.ambient_uniformity(sol)*100:.3f} "
                    "% of |B|  (a dirty spot shows up here, not in the residual)")
        self.cal_report.setPlainText("\n".join(text))
        self.btn_apply_roll.setEnabled(True)
        self.log.log("roll solve done; review the report before applying")

    def on_apply_roll(self):
        if self.pose_solution is None:
            return
        sol = self.pose_solution
        warn = []
        if not sol.identified[:, 2].all():
            warn.append("The axial column (chip +Y on every sensor here) was "
                        "NOT identified by the data and will be taken from the "
                        "nominal geometry.")
        if not sol.anisotropy_identified:
            warn.append("Transverse-vs-axial sensitivity was not pinned. "
                        "Inter-sensor matching is still valid; the absolute "
                        "axial ratio is not.")
        if sol.offset_leverage < 0.6:
            warn.append("The axial field barely changed between orientations, "
                        "so offsets and axial response are poorly separated.")
        if warn:
            r = QtWidgets.QMessageBox.question(
                self, "Apply roll calibration",
                "\n\n".join(warn) + "\n\nApply anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No)
            if r != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        try:
            self.cal.apply_pose_solution(sol)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Apply roll calibration", str(e))
            return
        g = sol.gains()
        self.log.log("roll calibration applied: matrix + offsets installed, "
                     f"magnet gain trim cleared; gain spread was "
                     f"{(np.nanmax(g)-np.nanmin(g))*100:.2f} %")
        self.chk_vcm.setChecked(self.cal.subtract_vcm)
        self._calibration_changed("the roll calibration")
        self.refresh_cal_report()

    def on_save_sweeps(self):
        if not self.sweeps:
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Save roll sweeps into", self.out_dir)
        if not d:
            return
        for tag, sw in sorted(self.sweeps.items()):
            path = sw.save(os.path.join(d, f"rollsweep_{tag}"))
            self.log.log(f"wrote {path}")

    def on_load_sweeps(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Load roll sweeps", self.out_dir, "Roll sweeps (*.npz)")
        for f in files:
            try:
                sw = opc.RollSweep.load(f)
            except (OSError, ValueError, KeyError) as e:
                self.log.log(f"could not load {f}: {e}")
                continue
            self.sweeps[sw.tag] = sw
            self.log.log(f"loaded sweep {sw.tag} from {os.path.basename(f)}")
        self._refresh_sweep_label()

    def on_clear_tare(self):
        self.cal.clear_tare()
        self.log.log("zero cleared")
        self._calibration_changed("the zero point")
        self.refresh_cal_report()

    def on_magnet_pass(self, on):
        if on:
            if self.source is None:
                self.btn_magnet.setChecked(False)
                return
            recent = self.roll.view()
            if recent.shape[0] < 2:
                self.btn_magnet.setChecked(False)
                self.lbl_magnet.setText("no data yet -- wait a moment and retry")
                return
            base = np.median(recent, axis=0).astype(np.float64)
            self.collecting = {"what": "magnet", "blocks": [], "n": 0,
                               "need": 0, "peak": None, "baseline": base,
                               "tag": None, "decim": self.decim}
            self.btn_magnet.setText("Stop magnet pass")
            self.lbl_magnet.setText("recording -- pass the magnet along the probe, "
                                    "then press stop")
        else:
            c = self.collecting
            self.collecting = None
            self.btn_magnet.setText("Start magnet pass")
            if not c or c.get("peak") is None:
                self.lbl_magnet.setText("no data captured")
                return
            self.magnet_peaks = np.asarray(c["peak"], float)
            live = self.cal.live_mask()
            rep = ocal.spread_report(self.magnet_peaks, live=live)
            n = rep.get("n_responding", 0)
            spread = rep.get("raw_spread")
            self.lbl_magnet.setText(
                f"{n} sensors responded, peak spread "
                f"{spread:.2f}x" if spread else f"{n} sensors responded")
            self.btn_apply_gain.setEnabled(True)
            self.log.log("magnet pass: peak |B| per sensor = "
                         + ", ".join(f"S{i+1}={v:.3f}"
                                     for i, v in enumerate(self.magnet_peaks)))
            self.refresh_cal_report()

    def on_apply_gain(self):
        if self.magnet_peaks is None:
            return
        w = None
        if self.chk_geom.isChecked():
            pt = (self.spin_mx.value(), self.spin_my.value(), self.spin_mz.value())
            w = self.geom.expected_response(pt, self.spin_exp.value())
        _corr, skipped = self.cal.cross_calibrate(self.magnet_peaks, weights=w)
        note = (f"kept their previous trim (no usable response): "
                f"{', '.join(skipped)}") if skipped else "every live sensor trimmed"
        self.log.log(f"gain trim applied using "
                     f"{'geometry-weighted' if w is not None else 'raw'} peaks; "
                     f"{note}")
        self.lbl_magnet.setText(f"gain trim applied -- {note}")
        self._calibration_changed("the gain trim")
        self.refresh_cal_report()

    def on_guided_magnet(self):
        """Open the guided routine, or explain what it needs first."""
        if self.stages is None:
            QtWidgets.QMessageBox.information(
                self, "Guided magnet calibration",
                "This routine drives the head past a fixed magnet, so it "
                "needs the stages. Press Connect — it brings up the carriers "
                "and the stages together — or connect them from the Stages "
                "tab.\n\nWithout motion, the hand magnet pass above is the "
                "alternative; it needs geometry weighting to be fair, which "
                "this one does not.")
            return
        # Visibility, not the reference, decides whether one is already open.
        # Belt and braces against the failure above: any future path that
        # hides this dialog without finishing it would otherwise wedge the
        # button, and the symptom -- a button that does nothing at all -- is
        # about as hard to diagnose from a bug report as it gets.
        if self._magnet_wizard is not None:
            if self._magnet_wizard.isVisible():
                self._magnet_wizard.raise_()
                self._magnet_wizard.activateWindow()
                return
            self._magnet_wizard.deleteLater()
            self._magnet_wizard = None
        self._magnet_wizard = MagnetWizard(self)
        self._magnet_wizard.finished.connect(
            lambda _: setattr(self, "_magnet_wizard", None))
        self._magnet_wizard.show()

    def on_show_superseded(self, on):
        for box in self._superseded_boxes:
            box.setVisible(bool(on))

    def on_clear_gain(self):
        self.cal.clear_gain()
        self.log.log("gain trim cleared")
        self._calibration_changed("the gain trim")
        self.refresh_cal_report()

    # ---- calibration state -----------------------------------------------
    def on_range_changed(self, row, value):
        self.cal.ranges_mt[row] = value
        self.log.log(f"S{row+1} range set to +/-{value:g} mT "
                     f"({ob.RANGE_TO_VPT[value]:g} V/T)")
        self._calibration_changed(f"the S{row+1} range")
        self.refresh_cal_report()

    def on_vcm_toggle(self, on):
        self.cal.subtract_vcm = bool(on)
        if not on:
            self.log.log("WARNING: VCM subtraction off -- readings now include "
                         "each chip's ~2.2 V virtual ground offset")
        self._calibration_changed("VCM subtraction")
        self.refresh_cal_report()

    def refresh_cal_report(self):
        lines = [self.cal.summary(), ""]
        z = self.cal.zero_mt
        g = self.cal.gain_corr
        lines.append(f"{'sensor':>7} {'range':>9} {'zero Bx':>9} {'zero By':>9} "
                     f"{'zero Bz':>9} {'gain trim':>10}")
        for s in range(N_SENSORS):
            lines.append(f"{'S'+str(s+1):>7} {self.cal.ranges_mt[s]:8.0f}mT "
                         f"{z[s,0]:9.4f} {z[s,1]:9.4f} {z[s,2]:9.4f} "
                         f"{g[s].mean():10.4f}")
        if self.magnet_peaks is not None:
            lines += ["", "last magnet pass, peak |B| per sensor [mT]:"]
            live = self.cal.live_mask()
            for s in range(N_SENSORS):
                tag = "" if live[s] else "   (excluded)"
                lines.append(f"{'S'+str(s+1):>7} {self.magnet_peaks[s]:10.4f}{tag}")
            rep = ocal.spread_report(self.magnet_peaks, live=live)
            if "raw_spread" in rep:
                lines.append(f"\nraw spread across responding sensors: "
                             f"{rep['raw_spread']:.2f}x")
            if self.chk_geom.isChecked():
                pt = (self.spin_mx.value(), self.spin_my.value(),
                      self.spin_mz.value())
                rep2 = ocal.spread_report(self.magnet_peaks, self.geom, pt,
                                          self.spin_exp.value(), live)
                if "corrected_spread" in rep2:
                    lines.append(
                        f"expected spread from 1/r^{self.spin_exp.value():g} "
                        f"geometry alone: {rep2['geometry_spread']:.2f}x")
                    lines.append(
                        f"spread left after removing geometry: "
                        f"{rep2['corrected_spread']:.2f}x  <- this part is "
                        f"electrical (gain register, EEPROM calibration, "
                        f"Hall bias), not mounting")
        self.cal_report.setPlainText("\n".join(lines))

    def on_save_cal(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save calibration", self.args.calibration, "JSON (*.json)")
        if path:
            self.cal.notes = f"saved from octobee_gui {time.strftime('%Y-%m-%d %H:%M')}"
            self.cal.save(path)
            self.log.log(f"calibration written to {path}")

    def on_load_cal(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load calibration", ".", "JSON (*.json)")
        if not path:
            return
        try:
            cal = ocal.Calibration.load(path)
        except (OSError, ValueError, TypeError) as exc:
            # Deliberately not load_or_default: an explicit Load that quietly
            # substituted +/-20 mT defaults would be the worst outcome here.
            # Keep the calibration already in force and say what went wrong.
            self.log.log(f"could not load {path}: {type(exc).__name__}: {exc}")
            QtWidgets.QMessageBox.warning(
                self, "Load calibration",
                f"{path} could not be read:\n\n{type(exc).__name__}: {exc}\n\n"
                f"The calibration already loaded is unchanged.")
            return
        self.cal = cal
        self.table.set_ranges(self.cal.ranges_mt)
        self.chk_vcm.setChecked(self.cal.subtract_vcm)
        self._calibration_changed("the whole calibration")
        self.refresh_cal_report()
        self.log.log(f"calibration loaded from {path}")

    def on_edit_geometry(self):
        path = self.args.geometry
        if not os.path.exists(path):
            self.geom.save(path)
        QtWidgets.QMessageBox.information(
            self, "Probe geometry",
            f"The tube layout lives in:\n\n{os.path.abspath(path)}\n\n"
            f"Edit it to match the real probe -- tube width, chip pitch, and "
            f"which sensor is on which face. The default face assignment has "
            f"NOT been verified on this hardware. Then press "
            f"'Reload geometry'.")

    def on_reload_geometry(self):
        self.geom = pgeom.Geometry.load_or_default(self.args.geometry)
        self.view3d.rebuild(self.geom)
        # The body that has to clear the coils just changed shape.
        self.machine_view.set_geometry(self.geom)
        self._probe_cloud = None
        self.refresh_machine(force=True)
        self.plot.geom = self.geom
        self.table.refresh_geometry(self.geom)
        if isinstance(self.source, DemoSource):
            self.source.geom = self.geom
        self.log.log(f"geometry reloaded from {self.args.geometry}")

    # ---- diagnostics ------------------------------------------------------
    def on_health(self):
        raw = self._raw_arrays()
        if raw is None:
            self.health_text.setPlainText("no data -- connect first")
            return
        rows = ocal.channel_health(raw, self.source.vpc, self.source.hosts,
                                   self.source.volt_offset)
        self.last_health = rows
        verdict = ocal.health_verdict(rows)
        n = raw[0].shape[0]
        out = [f"{n} samples per box at {self.source.fs_hz/1e3:g} kSPS "
               f"({n/self.source.fs_hz:.2f} s), 1 count = "
               f"{self.source.vpc[0]*1e6:.1f} uV", ""]
        out.append(f"{'host':>13} {'ch':>3} {'signal':>9} {'mean [V]':>10} "
                   f"{'noise [counts]':>15} {'p-p':>8}  flag")
        for r in rows:
            flag = ocal.bad_reason(r).upper()
            out.append(f"{r['host']:>13} {r['ch']:3d} "
                       f"{r['sensor']+' '+r['axis']:>9} {r['mean_v']:10.4f} "
                       f"{r['std_counts']:15.2f} {r['p2p_counts']:8.0f}  {flag}")
        out += ["", "per-sensor verdict:"]
        for sid in range(1, N_SENSORS + 1):
            st, note = verdict[sid]
            out.append(f"  S{sid:<3d} {st:<8s} {note}")

        vcm = [r for r in rows if r["is_vcm"]]
        out += ["", "VCM reference channels (these carry NO field, so any noise "
                    "on them is analogue pickup in the cabling and grounding):"]
        for r in vcm:
            out.append(f"  {r['sensor']:>4} {r['std_counts']:7.2f} counts rms "
                       f"({r['std_uv']:8.1f} uV)   VCM = {r['mean_v']:.4f} V")
        vs = np.array([r["mean_v"] for r in vcm])
        if vs.size:
            spread_v = vs.max() - vs.min()
            vpt = self.cal.volts_per_tesla.mean()
            out.append(f"\nVCM spread across the chips: {spread_v*1e3:.1f} mV "
                       f"= {spread_v/vpt*1e3:.2f} mT of apparent field if it "
                       f"were not subtracted.")
        trend = [r["std_counts"] for r in vcm]
        if len(trend) >= 8 and trend[7] > 3 * max(trend[0], 0.1):
            out.append("VCM noise climbs steadily along the first concentrator "
                       "-- that pattern is a cabling/ground problem, not a "
                       "sensor calibration problem.")
        self.health_text.setPlainText("\n".join(out))
        self.tabs.setCurrentIndex(3)

    def on_export_health(self):
        if not self.last_health:
            self.on_health()
        if not self.last_health:
            return
        path = orec.default_name("channel_health", "csv", self.out_dir)
        orec.write_health_csv(path, self.last_health)
        self._exported(path)

    # ---- data output ------------------------------------------------------
    def on_record(self, on):
        if on:
            if self.source is None:
                self.act_record.setChecked(False)
                return
            if self.chk_csv.isChecked():
                p = orec.default_name("octobee", "csv", self.out_dir)
                self.csv_rec = orec.CsvRecorder(
                    p, self.out_rate, self.cal, self.geom,
                    tube_frame=self.chk_tube.isChecked(),
                    meta={"hosts": ",".join(self.source.hosts),
                          "stream_rate_hz": self.source.fs_hz})
                self.log.log(f"recording CSV to {p} at {self.out_rate:g} Hz")
            if self.chk_raw.isChecked():
                p = orec.default_name("octobee", "bin", self.out_dir)
                self.raw_rec = orec.RawRecorder(
                    p, self.source.hosts, self.source.vpc,
                    [self.source.fs_hz] * len(self.source.hosts),
                    cal=self.cal)
                self.log.log(f"recording raw counts to {p}")
            if self.csv_rec is None and self.raw_rec is None:
                self.log.log("nothing selected to record -- see the Data output tab")
                self.act_record.setChecked(False)
        else:
            for rec, kind in ((self.csv_rec, "CSV"), (self.raw_rec, "raw")):
                if rec is not None:
                    p = rec.close()
                    size = os.path.getsize(p) / 1e6 if os.path.exists(p) else 0
                    self._exported(f"{p}  ({size:.2f} MB, {kind})")
            self.csv_rec = None
            self.raw_rec = None
            self.lbl_recinfo.setText("not recording")

    def _update_rec_label(self):
        parts = []
        if self.csv_rec is not None:
            parts.append(f"CSV {self.csv_rec.n_rows} rows "
                         f"{self.csv_rec.size_bytes/1e6:.1f} MB")
        if self.raw_rec is not None:
            parts.append(f"raw {self.raw_rec.n_samples} samples "
                         f"{self.raw_rec.size_bytes/1e6:.1f} MB")
        txt = "  |  ".join(parts) if parts else ""
        self.lbl_rec.setText(("REC  " + txt) if parts else "")
        self.lbl_recinfo.setText(txt or "not recording")

    def on_snapshot(self):
        if not isinstance(self.source, LiveSource):
            QtWidgets.QMessageBox.information(
                self, "Snapshot",
                "A full-rate snapshot needs the live hardware.")
            return
        if self.act_record.isChecked():
            self.act_record.setChecked(False)
        secs = self.spin_snap_s.value()
        path = orec.default_name("snapshot", "npz", self.out_dir)
        self.log.log(f"snapshot: stopping the stream, restoring the carriers' "
                     f"own clock, and capturing {secs:g} s losslessly")
        self.source.stop()
        self.source = None
        self.act_snapshot.setEnabled(False)
        self.lbl_state.setText("snapshot in progress...")
        self._snap_worker = SnapshotWorker(self.hosts, secs, path,
                                           self.prev_clkdiv)
        # The worker puts the clock back itself, so there is nothing left for
        # disconnect to restore. Clearing it here rather than on completion
        # means a failed snapshot cannot leave a stale value behind either.
        self.prev_clkdiv = {}
        self._snap_worker.done.connect(self.on_snapshot_done)
        self._snap_worker.start()

    def on_snapshot_done(self, path, error, fs_hz):
        self.act_snapshot.setEnabled(True)
        if error:
            self.log.log(f"snapshot failed: {error}")
            QtWidgets.QMessageBox.critical(self, "Snapshot failed", error)
        else:
            size = os.path.getsize(path) / 1e6
            self._exported(f"{path}  ({size:.2f} MB raw, {fs_hz/1e3:g} kSPS)")
        self.lbl_state.setText("disconnected -- press Connect to resume")
        self.act_disconnect.setEnabled(False)
        self.act_connect.setEnabled(True)

    def on_export_summary(self):
        table = getattr(self, "_last_table", None)
        if not table:
            self.log.log("no sensor data yet")
            return
        path = orec.default_name("sensor_summary", "csv", self.out_dir)
        orec.write_sensor_csv(path, table)
        self._exported(path)

    def on_export_json(self):
        table = getattr(self, "_last_table", None)
        if not table:
            self.log.log("no sensor data yet")
            return
        live = self.cal.live_mask()
        payload = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hosts": list(self.source.hosts) if self.source else [],
            "stream_rate_hz": self.source.fs_hz if self.source else None,
            "output_rate_hz": self.out_rate,
            "calibration": self.cal.to_dict(),
            "geometry": self.geom.to_dict(),
            "sensors": table,
            "channel_health": self.last_health or [],
        }
        if self.magnet_peaks is not None:
            payload["magnet_pass"] = {
                "peak_absB_mT": self.magnet_peaks,
                "spread": ocal.spread_report(self.magnet_peaks, live=live),
            }
            if self.chk_geom.isChecked():
                pt = [self.spin_mx.value(), self.spin_my.value(),
                      self.spin_mz.value()]
                payload["magnet_pass"]["magnet_point_mm"] = pt
                payload["magnet_pass"]["geometry_corrected"] = ocal.spread_report(
                    self.magnet_peaks, self.geom, pt, self.spin_exp.value(), live)
        path = orec.default_name("octobee_report", "json", self.out_dir)
        orec.write_report_json(path, payload)
        self._exported(path)

    def _exported(self, what):
        self.export_log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {what}")
        self.log.log(f"wrote {what}")

    # ---- misc UI ----------------------------------------------------------
    def _mark_dead_checkboxes(self, dead):
        for s, cb in enumerate(self.chk_sensors):
            bad = f"S{s+1}" in dead
            cb.setStyleSheet("color:#c05050;" if bad else "")
            cb.setToolTip("excluded: railed or stuck channels" if bad else "")

    def on_units(self, *_):
        """
        Build the per-sensor factor that takes millitesla to the chosen unit.

        All 16 chips currently run gain 3000, so the factor is the same for
        every sensor. It stays per-sensor anyway because the range is a
        per-sensor setting and the two halves genuinely have differed -- they
        were 34.65 and 63 V/T until the 2026-08-19 harmonisation. Under a split
        gain one field is not one voltage, and a single global factor would
        quietly misreport half the probe.
        """
        name = self.cmb_units.currentText()
        vpt = self.cal.volts_per_tesla                     # V/T, per sensor
        if name == "mT":
            k = np.ones(N_SENSORS)
        elif name == "uT":
            k = np.full(N_SENSORS, 1e3)
        elif name.startswith("mV"):
            k = vpt                                        # mT * V/T -> mV
        else:                                              # ADC counts
            vpc = np.array([(self.source.vpc[0] if s < 8 else self.source.vpc[-1])
                            if self.source else 20.0 / 65536.0
                            for s in range(N_SENSORS)])
            k = vpt * 1e-3 / vpc                           # mT -> volts -> counts
        self.plot.set_units(k, {"uT": "µT"}.get(name, name))

    def on_sensor_toggle(self, _=None):
        self.plot.set_visible_sensors(
            {i for i, cb in enumerate(self.chk_sensors) if cb.isChecked()})

    def _set_all_sensors(self, on):
        for cb in self.chk_sensors:
            cb.blockSignals(True)
            cb.setChecked(on)
            cb.blockSignals(False)
        self.on_sensor_toggle()

    def _profile_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        top = QtWidgets.QHBoxLayout()
        self.chk_prof = QtWidgets.QCheckBox("measure where the time goes")
        self.chk_prof.setChecked(self.prof.enabled)
        self.chk_prof.setToolTip(
            "Times every stage separately, including the OpenGL paint, and "
            "watches the Qt event loop for stalls. Costs almost nothing, so it "
            "is fine to leave on.")
        self.chk_prof.toggled.connect(self.on_profile_toggle)
        top.addWidget(self.chk_prof)
        b_reset = QtWidgets.QPushButton("Reset")
        b_reset.clicked.connect(self.on_profile_reset)
        top.addWidget(b_reset)
        b_copy = QtWidgets.QPushButton("Copy to clipboard")
        b_copy.clicked.connect(
            lambda: QtWidgets.QApplication.clipboard().setText(
                self.profile_text.toPlainText()))
        top.addWidget(b_copy)
        top.addStretch(1)
        lay.addLayout(top)
        self.profile_text = QtWidgets.QPlainTextEdit()
        self.profile_text.setReadOnly(True)
        self.profile_text.setFont(QtGui.QFont("Consolas", 9))
        lay.addWidget(self.profile_text)
        return w

    def on_profile_toggle(self, on):
        self.prof.enabled = bool(on)
        self.prof.reset()
        self.lag.reset()
        self.log.log("profiling on" if on else "profiling off")
        self.refresh_profile()

    def on_profile_reset(self):
        self.prof.reset()
        self.lag.reset()
        self.refresh_profile()

    def refresh_profile(self):
        if not hasattr(self, "profile_text"):
            return
        if not self.prof.enabled:
            self.profile_text.setPlainText(self.prof.text())
            return
        # Ask the live context what it is, once we have one. A software
        # renderer here is the single most likely explanation for a window
        # that seizes up the moment data starts arriving.
        if "GL renderer" not in self.prof.notes and self.view3d.isVisible():
            info = self.view3d.gl_info()
            for k, v in info.items():
                self.prof.note(k, v)
            if oprof.is_software_renderer(info):
                self.prof.note("VERDICT", "no GPU acceleration -- the 3D head "
                                          "is being drawn on the CPU")
                self.log.log(
                    f"OpenGL is running on a software renderer "
                    f"({info.get('GL renderer')}). Every repaint of the probe "
                    f"head is done on the CPU, which is almost certainly why "
                    f"the window struggles. Untick '3D' to confirm.")
        parts = [self.prof.text(), "",
                 f"event loop lag: mean {self.lag.mean_ms:.1f} ms, "
                 f"worst {self.lag.max_ms:.0f} ms",
                 f"  -> {self.lag.verdict()}"]
        if self.source is not None:
            st = self.source.stats()
            parts += ["", "stream:"]
            for k, v in st.items():
                parts.append(f"  {k:<26} {v}")
            qs = [getattr(x, "q", None) for x in
                  getattr(self.source, "streamers", [])]
            for h, q in zip(getattr(self.source, "hosts", []), qs):
                if q is not None:
                    parts.append(f"  {h + ' reader queue':<26} {q.qsize()} blocks "
                                 f"waiting")
        parts += ["", "how to read this:",
                  "  'GL paint (probe head)' large  -> the 3D view is the cost;"
                  " untick 3D or lower the refresh rate",
                  "  'counts -> tesla' large        -> the data processing is"
                  " the cost; lower the stream rate",
                  "  reader queue growing           -> acquisition is falling"
                  " behind and recordings will have holes",
                  "  event loop lag large but every row small -> something"
                  " outside this list is blocking"]
        self.profile_text.setPlainText("\n".join(parts))

    def on_3d_toggle(self, on):
        self.view3d.setVisible(bool(on))
        self._draw_ms = 0.0
        self.log.log("3D head on" if on else
                     "3D head off -- everything else carries on unchanged")

    def on_view_rate(self):
        hz = float(self.cmb_view.currentData())
        self.view_timer.setInterval(int(1000 / hz))
        self._draw_ms = 0.0
        self.log.log(f"view redraw rate {hz:g} Hz "
                     f"(acquisition and recording are unaffected)")

    def on_pause(self, on):
        self.paused = bool(on)
        self.log.log("view paused -- acquisition and recording continue"
                     if on else "view resumed")

    def on_out_rate(self):
        self.out_rate = float(self.cmb_out.currentData())
        self.roll.resize(max(4, int(self.out_rate * self.window_s)))
        self.log.log(f"output rate {self.out_rate:g} Hz "
                     f"(decimation {self.decim}x from the stream)")

    def on_window(self, v):
        self.window_s = float(v)
        self.roll.resize(max(4, int(self.out_rate * self.window_s)))

    def closeEvent(self, ev):
        if self.prof.enabled:
            print(self.prof.text())
            print(f"\nevent loop lag: mean {self.lag.mean_ms:.1f} ms, "
                  f"worst {self.lag.max_ms:.0f} ms -- {self.lag.verdict()}")
        if self.act_record.isChecked():
            self.act_record.setChecked(False)
        if self.motion_busy():
            # Ask before walking away from a moving machine. Closing used to
            # go straight through to stages.close(), which does not stop
            # anything -- the move is already in the controller and it runs to
            # completion whether or not this window still exists. So the
            # window would vanish and the head would carry on traversing with
            # nothing watching it and no stop button anywhere.
            reply = QtWidgets.QMessageBox.warning(
                self, "Something is still moving",
                "A stage job is still running.\n\n"
                "Closing stops every axis first — the move in progress is "
                "abandoned where it is, so re-home before trusting a "
                "position afterwards.\n\nClose and stop?",
                QtWidgets.QMessageBox.StandardButton.Close
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel)
            if reply != QtWidgets.QMessageBox.StandardButton.Close:
                ev.ignore()
                return
            self.trigger_estop("the window was closed while a job was running")
        for w in [*self._motion_workers, self._stage_worker]:
            if _is_running(w):
                # Bounded: a worker wedged inside a DLL call must not stop the
                # window closing, or the only way out is Task Manager -- which
                # is the one exit that leaves the stages held and moving.
                w.wait(30000)
        self.stage_timer.stop()
        if self.stages is not None:
            # These are exclusive-open USB devices: leaving them held means the
            # Kinesis application will not start until this process dies.
            # close() stops each axis on the way out; see Stage.close.
            self.stages.close()
            self.stages = None
        self.on_disconnect()
        super().closeEvent(ev)


CRASH_LOG = "octobee_crash.log"


class CrashHandler:
    """Write unhandled exceptions down, because otherwise nobody ever sees one.

    Bare PyQt aborts the process when an exception escapes a slot -- but this
    program imports pyqtgraph, which replaces sys.excepthook with its own
    override precisely to stop that. Measured here: the same raising slot exits
    127 without pyqtgraph and 0 with it. So exceptions in slots are already
    survivable; what they are not is VISIBLE. pyqtgraph prints the traceback to
    stderr, and the desktop icon runs pythonw.exe, where stderr goes nowhere at
    all. The program carries on in a state nobody knows about.

    So this is a recorder, not a rescue: the traceback goes to a file, the Log
    tab gets a line, and one dialog says the window may now be inconsistent.
    The previous hook is still called, so pyqtgraph keeps doing whatever it
    does.

    It does NOT catch a native crash -- an access violation, or Qt calling
    qFatal(). Those never reach Python. faulthandler, enabled beside this in
    main(), is what leaves a record of those.
    """

    def __init__(self, path=None, window=None):
        self.path = path or os.path.join(
            paths.repo_root() or os.getcwd(), CRASH_LOG)
        self.window = window
        self.count = 0
        self.previous = None

    def install(self):
        self.previous = sys.excepthook
        sys.excepthook = self
        return self

    def __call__(self, exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        self.count += 1
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # The file first, and in its own try: everything after this can fail
        # (there may be no window, Qt may be half torn down) and the whole
        # point is that the traceback survives regardless.
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(f"\n===== {stamp} =====\n{text}")
        except OSError:
            pass
        with contextlib.suppress(Exception):
            if self.window is not None:
                self.window.log.log(
                    f"INTERNAL ERROR ({exc_type.__name__}: {exc}) -- written "
                    f"to {os.path.basename(self.path)}. The program is still "
                    f"running but whatever was in progress did not finish.")
        # One dialog, not one per occurrence: a fault inside a repainting
        # timer fires several times a second, and a queue of identical boxes is
        # indistinguishable from the freeze this is meant to replace.
        #
        # NON-modal, and show() rather than QMessageBox.critical(). The
        # convenience call is the blocking one: it spins a nested event loop
        # and does not return until somebody clicks. Doing that from inside an
        # excepthook stops the program dead in the middle of whatever was
        # interrupted -- measured here, with no window to parent it to, the
        # call simply never returned. A reporter that freezes the application
        # is worse than no reporter at all.
        if self.count == 1 and self.window is not None:
            with contextlib.suppress(Exception):
                box = QtWidgets.QMessageBox(
                    QtWidgets.QMessageBox.Icon.Critical, "Internal error",
                    f"{exc_type.__name__}: {exc}\n\n"
                    f"The full traceback is in {self.path}\n\n"
                    f"The program is still running, but whatever it was doing "
                    f"when this happened did not finish \u2014 save anything "
                    f"you care about and restart. Further errors will be "
                    f"logged without another dialog.",
                    parent=self.window)
                box.setModal(False)
                box.setAttribute(
                    QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
                # Held on self: a box with nothing referencing it is collected
                # before it is ever painted.
                self._box = box
                box.show()
        # Whatever was installed before us -- pyqtgraph's override, normally --
        # still gets its turn, so nothing it relies on is quietly removed.
        if self.previous is not None and self.previous is not self:
            with contextlib.suppress(Exception):
                self.previous(exc_type, exc, tb)


ICON_NAME = "octobee.ico"


def _apply_app_icon(app):
    """Put the probe-head icon on the window and the taskbar button.

    The AppUserModelID is the non-obvious half. Without it Windows attributes
    the window to pythonw.exe, so the taskbar shows the generic Python icon and
    groups this with every other Python process -- leaving the desktop shortcut
    as the only place the application looks like itself.
    """
    root = paths.repo_root()
    path = os.path.join(root, ICON_NAME) if root else ICON_NAME
    if not os.path.exists(path):
        return
    if sys.platform == "win32":
        try:
            import ctypes  # noqa: PLC0415  (Windows-only, cosmetic)
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "harrer.octobee.hallprobe.1")
        except Exception:
            pass                                      # cosmetic only
    app.setWindowIcon(QtGui.QIcon(path))


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uut", action="append", default=None,
                   help=f"carrier hostname, repeat for both "
                        f"(default: {' '.join(ob.DEFAULT_UUTS)})")
    p.add_argument("--demo", action="store_true",
                   help="synthetic probe, no hardware needed")
    p.add_argument("--replay", help="play back a saved .npz capture")
    p.add_argument("--geometry", default=pgeom.CONFIG_NAME)
    p.add_argument("--calibration", default=ocal.CONFIG_NAME)
    p.add_argument("--machine", default=omach.CONFIG_NAME,
                   help="where the coil set, the energised coils and the "
                        "probe's placement in the machine are remembered "
                        f"(default: {omach.CONFIG_NAME})")
    p.add_argument("--out-dir", default="captures",
                   help="where recordings, snapshots and exports are written "
                        "(default: captures)")
    p.add_argument("--no-connect", action="store_true",
                   help="start disconnected. Without this the window connects "
                        "to the carriers and the stages as soon as it opens")
    p.add_argument("--profile", action="store_true",
                   help="start with the Profile tab measuring where the time "
                        "goes, and print a summary on exit")
    p.add_argument("--screenshot", help="render one frame to this PNG and exit "
                                        "(for headless checks)")
    p.add_argument("--screenshot-tab", type=int, default=0,
                   help="which tab to show in the screenshot")
    p.add_argument("--screenshot-warmup", type=float, default=3.0,
                   help="seconds of data to collect before the screenshot")
    a = p.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    crash = CrashHandler().install()
    # The other half of the same job. A Qt fatal error or an access violation
    # never becomes a Python exception -- the process simply dies, and all the
    # Windows event log records is "Qt6Core.dll, 0xc0000409", which names no
    # Python at all. faulthandler catches the abort signal itself and dumps the
    # Python stack of every thread as it goes, which is the difference between
    # a crash report that says where to look and one that says nothing.
    # Line-buffered and never closed on purpose: it has to be usable from
    # inside a signal handler, at a moment when the process is already dying.
    with contextlib.suppress(OSError):
        _fault_log = open(crash.path, "a", buffering=1, encoding="utf-8")
        _fault_log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                         f"session started =====\n")
        faulthandler.enable(file=_fault_log)
    _apply_app_icon(app)
    app.setApplicationName("OCTO-BEE Hall probe")
    win = MainWindow(a)
    crash.window = win
    win.show()

    if a.screenshot:
        def shoot():
            win.tabs.setCurrentIndex(a.screenshot_tab)
            if win.source is None:
                win.on_connect()
                deadline = time.time() + 60
                while win.source is None and time.time() < deadline:
                    app.processEvents()
                    time.sleep(0.05)
                if win.source is None:
                    print("could not connect", file=sys.stderr)
            t_end = time.time() + a.screenshot_warmup
            while time.time() < t_end:
                win.on_tick()
                app.processEvents()
                time.sleep(0.02)
            win.on_view_tick()
            win.on_slow_tick()
            app.processEvents()
            time.sleep(0.2)
            app.processEvents()
            win.grab().save(a.screenshot)
            print(f"wrote {a.screenshot}")
            # close(), not quit(): closeEvent stops the recorders and puts the
            # boxes' clkdiv back. Quitting straight out would leave both
            # carriers running at 20 kSPS for whoever connects next.
            win.close()
            app.quit()
        QtCore.QTimer.singleShot(1200, shoot)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
