#!/usr/bin/env python3
"""
octobee_gui.py -- control, live view, calibration and data export for the
16-sensor OCTO-BEE Hall probe, in one window.

    python octobee_gui.py                       # talk to the two carriers
    python octobee_gui.py --demo                # synthetic probe, no hardware
    python octobee_gui.py --replay capture.npz  # play back a saved capture

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

import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

import octobee as ob
import octobee_calibration as ocal
import octobee_profile as oprof
import octobee_posecal as opc
import octobee_record as orec
import octobee_scan as oscan
import octobee_stage as ostage
import probe_geometry as pgeom
from probe_view3d import ProbeView3D, color_for

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
        self.vpc = [l.volts_per_count for l in layouts]
        self.fs_hz = float(layouts[0].fs_hz)
        self._pending = {h: deque() for h in self.hosts}
        self._n = {h: 0 for h in self.hosts}
        self._temp = {h: np.zeros(8, dtype=np.uint32) for h in self.hosts}
        self.gaps = 0
        self.lost = 0
        self.bytes0 = 0
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
        import queue
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
                g, l = ob.check_continuity(blk["sam_cnt"])
                self.gaps += g
                self.lost += l
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
        self.vpc = [ob.ADC_RANGES["+/-10V"] / 65536.0] * 2
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
        import threading
        prev, layouts, streamers, errors = {}, {}, {}, {}

        def prepare(h):
            try:
                self.progress.emit(f"{h}: releasing the stream")
                if ob.stop_live_stream(h):
                    self.progress.emit(f"{h}: stopped a running capture")
                if self.target_fs:
                    self.progress.emit(f"{h}: setting "
                                       f"{self.target_fs/1000:g} kSPS")
                    import octobee_live as olive
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
            except Exception as e:                    # noqa: BLE001
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
        except Exception as e:                        # noqa: BLE001
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
                import octobee_live as olive
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
        except Exception as e:                        # noqa: BLE001
            self.done.emit("", f"{type(e).__name__}: {e}", fs)


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

    def run(self):
        try:
            self.fn(self.progress.emit)
            self.done.emit(self.what, "")
        except Exception as e:                        # noqa: BLE001
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
                 restore_clkdiv=None):
        super().__init__()
        self.hosts = hosts
        self.stages = stages
        self.grid = grid
        self.seconds = seconds
        self.cal = cal
        self.settle_s = settle_s
        self.restore_clkdiv = dict(restore_clkdiv or {})
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
                import octobee_live as olive
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
                log=self.message.emit)
            self.done.emit(fm, "")
        except Exception as e:                        # noqa: BLE001
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
            for a, style in enumerate((QtCore.Qt.PenStyle.SolidLine,
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
            "noise uT", "VCM V", "T degC", "range", "gain trim")
    STATE_COLORS = {"ok": (40, 120, 60), "noisy": (150, 110, 20),
                    "fault": (150, 70, 20), "dead": (140, 30, 30),
                    "unknown": (70, 70, 80)}

    range_changed = QtCore.pyqtSignal(int, float)

    def __init__(self, geom):
        super().__init__(N_SENSORS, len(self.COLS))
        self.geom = geom
        self.setHorizontalHeaderLabels(self.COLS)
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
                    ("T degC", "--" if row["temp_c"] is None
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
        self.geom = pgeom.Geometry.load_or_default(args.geometry)
        self.cal = ocal.Calibration.load_or_default(args.calibration)
        self.cal_from_file = os.path.exists(args.calibration)
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
        self.collect_target = 0
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
        self._scan_worker = None
        self._scan_t0 = 0.0

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
        self.stage_timer.start(200)

        if args.demo:
            self._set_source(DemoSource(self.geom), "demo")
        elif args.replay:
            self._set_source(ReplaySource(args.replay), f"replay {args.replay}")
        else:
            self.log.log("ready -- press Connect to take over the stream from "
                         "Phoebus and start reading both carriers")
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
        left.addTab(self._export_tab(), "Data output")
        left.addTab(self._profile_tab(), "Profile")
        left.addTab(self.log, "Log")
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
        for k in STREAM_RATES:
            self.cmb_rate.addItem(k, STREAM_RATES[k])
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

    def _view3d_controls(self):
        w = QtWidgets.QWidget()
        l = QtWidgets.QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        self.chk_auto = QtWidgets.QCheckBox("auto scale")
        self.chk_auto.setChecked(True)
        self.chk_auto.toggled.connect(
            lambda v: setattr(self.view3d, "auto_scale", v))
        l.addWidget(self.chk_auto)
        l.addWidget(QtWidgets.QLabel("full scale"))
        self.spin_fs = QtWidgets.QDoubleSpinBox()
        self.spin_fs.setRange(0.001, 4000.0)
        self.spin_fs.setDecimals(3)
        self.spin_fs.setValue(1.0)
        self.spin_fs.setSuffix(" mT")
        self.spin_fs.valueChanged.connect(
            lambda v: setattr(self.view3d, "full_scale_mt", v))
        l.addWidget(self.spin_fs)
        self.chk_3d = QtWidgets.QCheckBox("3D")
        self.chk_3d.setChecked(True)
        self.chk_3d.setToolTip(
            "Draw the probe head. This is the most expensive thing in the "
            "window -- untick it if the display cannot keep up. Nothing else "
            "changes: acquisition, calibration and recording are unaffected.")
        self.chk_3d.toggled.connect(self.on_3d_toggle)
        l.addWidget(self.chk_3d)
        chk_arrows = QtWidgets.QCheckBox("arrows")
        chk_arrows.setChecked(True)
        chk_arrows.toggled.connect(self.view3d.set_arrows_visible)
        l.addWidget(chk_arrows)
        chk_lbl = QtWidgets.QCheckBox("labels")
        chk_lbl.setChecked(True)
        chk_lbl.toggled.connect(self.view3d.set_labels_visible)
        l.addWidget(chk_lbl)
        btn = QtWidgets.QPushButton("reset view")
        btn.clicked.connect(self.view3d.reset_camera)
        l.addWidget(btn)
        l.addStretch(1)
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

        g2 = QtWidgets.QGroupBox("2. Magnet pass — measure and equalise response")
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
        f2.addWidget(btn_row, 3, 0, 1, 5)
        lay.addWidget(g2)


        g3 = QtWidgets.QGroupBox(
            "3. Earth-field roll calibration — match the sensors to each other")
        f3 = QtWidgets.QGridLayout(g3)
        blurb = QtWidgets.QLabel(
            "The Earth's field is uniform to nanotesla across the whole head, so "
            "every sensor must read the SAME vector. That removes the 1/r³ "
            "position error the magnet pass above cannot escape.\n"
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

        g4 = QtWidgets.QGroupBox("4. Calibration file and geometry")
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
        self.btn_stage_scan.clicked.connect(self.on_stage_find)
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
                                   "widgets": (lbl, step, minus, plus, target,
                                               go, home)}

        self.btn_stage_homeall = QtWidgets.QPushButton("Home all axes")
        self.btn_stage_homeall.clicked.connect(lambda: self.on_stage_home(None))
        self.btn_stage_stop = QtWidgets.QPushButton("STOP")
        self.btn_stage_stop.setStyleSheet(
            "background:#7a1f1f; color:#fff; font-weight:bold;")
        self.btn_stage_stop.clicked.connect(self.on_stage_stop)
        f2.addWidget(self.btn_stage_homeall, 4, 0, 1, 3)
        f2.addWidget(self.btn_stage_stop, 4, 5, 1, 2)
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
        for row in self.stage_rows.values():
            for wdg in row["widgets"]:
                wdg.setEnabled(on)
        self.btn_stage_homeall.setEnabled(on)
        self.btn_stage_stop.setEnabled(on)
        self.btn_scan_start.setEnabled(on)
        self.btn_stage_disconnect.setEnabled(on)
        self.btn_stage_connect.setEnabled(not on)

    def on_stage_find(self):
        try:
            serials = ostage.list_devices()
        except ostage.StageError as exc:
            self.log.log(f"stages: {exc}")
            QtWidgets.QMessageBox.warning(self, "Stages", str(exc))
            return
        if not serials:
            msg = ("No stages on the bus. The Kinesis application is running "
                   "and holds all of them — close it and try again."
                   if ostage.kinesis_is_running() else
                   "No stages on the bus. Check power and USB.")
            self.log.log(f"stages: {msg}")
            QtWidgets.QMessageBox.warning(self, "Stages", msg)
            return

        saved = {v: k for k, v in ostage.load_axis_map().items()}
        self.stage_table.setRowCount(len(serials))
        self.stage_combos = {}
        for r, s in enumerate(serials):
            info = ostage.device_info(s)
            self.stage_table.setItem(r, 0, QtWidgets.QTableWidgetItem(s))
            self.stage_table.setItem(
                r, 1, QtWidgets.QTableWidgetItem(info["description"]))
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
                "'octobee_stage.py identify' to wiggle each one if unsure.")

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
        frames = ostage.load_axis_frames()
        stages = ostage.StageSet(
            {name: ostage.Stage(serial, name=name,
                                invert=frames.get(name, {}).get("invert", False),
                                origin_mm=frames.get(name, {}).get("origin_mm"))
             for name, serial in mapping.items()})
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
        if self._stage_worker is not None and self._stage_worker.isRunning():
            self.log.log("stages: busy — wait for the current move to finish")
            return
        self._stage_worker = StageWorker(what, fn)
        self._stage_worker.progress.connect(self.log.log)
        self._stage_worker.done.connect(self.on_stage_action_done)
        self._stage_worker.start()

    def on_stage_action_done(self, what, error):
        if error:
            self.log.log(f"stages: {what} failed — {error}")
            if what == "connect":
                self._stage_pending = None
                self.btn_stage_connect.setEnabled(True)
            QtWidgets.QMessageBox.warning(self, "Stages", error)
            return
        if what == "connect":
            self.stages = self._stage_pending
            self._stage_pending = None
            self._set_stage_controls_enabled(True)
            for ax, row in self.stage_rows.items():
                have = ax in self.stages.names
                for wdg in row["widgets"]:
                    wdg.setEnabled(have)
                if have:
                    lo, hi = self.stages[ax].travel_mm
                    row["target"].setRange(lo, hi)
            for ax, row in self.scan_rows.items():
                have = ax in self.stages.names
                row["chk"].setEnabled(have)
                if not have:
                    row["chk"].setChecked(False)
                for sp in row["spins"]:
                    sp.setEnabled(have)
            self.log.log("stages: connected")
        else:
            self.log.log(f"stages: {what} done")

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
        if not st.homed:
            QtWidgets.QMessageBox.warning(
                self, "Not homed",
                f"Axis {axis} has not been homed, so its position counter is "
                f"unreferenced and an absolute move would go somewhere "
                f"arbitrary. Home it first, or use the jog buttons, which are "
                f"relative and do not need a reference.")
            return
        self._stage_action(f"move {axis} to {mm:g} mm",
                           lambda emit: st.move_to(mm))

    def on_stage_home(self, axes):
        if self.stages is None:
            return
        names = axes or self.stages.names
        names = [n for n in names if n in self.stages.names]
        if not names:
            return
        reply = QtWidgets.QMessageBox.warning(
            self, "Home",
            f"Homing drives {', '.join(names)} into the limit switch at full "
            f"homing speed, across the whole travel.\n\n"
            f"Is the probe head — and its cabling — clear of the entire range "
            f"of movement?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel)
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            self.log.log("stages: homing cancelled")
            return
        stages = [self.stages[n] for n in names]

        def work(emit):
            for st in stages:
                st.home(wait=False)
            for st in stages:
                st.wait(timeout_s=300.0, what="homing")
                emit(f"{st.name}: homed, at {st.position_mm:.4f} mm")

        self._stage_action(f"home {', '.join(names)}", work)

    def on_stage_stop(self):
        """Stop everything now, including a running scan.

        Not routed through the worker queue: a stop that has to wait behind the
        move it is trying to interrupt is not a stop button.
        """
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._scan_worker.abort()
            self.log.log("stages: field map will stop after this point")
        if self.stages is None:
            return
        try:
            self.stages.stop_all()
            self.log.log("stages: stopped")
        except ostage.StageError as exc:
            self.log.log(f"stages: stop failed — {exc}")

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
            # Before opening, all the bus can say is "APT Stepper Motor
            # Controller"; the actual model only arrives with the settings.
            if snap["model"] and self.stage_table.item(r, 1).text() != snap["model"]:
                self.stage_table.setItem(
                    r, 1, QtWidgets.QTableWidgetItem(snap["model"]))
            self.stage_table.setItem(
                r, 3, QtWidgets.QTableWidgetItem(f"{snap['position_mm']:.4f} mm"))
            state = "moving" if snap["moving"] else "idle"
            if not snap["homed"]:
                state += ", NOT HOMED"
            if snap["error"]:
                state += ", MOTION ERROR"
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
        if self._scan_worker is not None and self._scan_worker.isRunning():
            return
        try:
            grid = self._scan_grid()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Field map", str(exc))
            return
        unhomed = [n for n in grid.names if not self.stages[n].homed]
        if unhomed:
            QtWidgets.QMessageBox.warning(
                self, "Field map",
                f"Axes {', '.join(unhomed)} have not been homed, so their "
                f"position counters are unreferenced and the map would have no "
                f"origin. Home them first.")
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
        self.btn_scan_start.setEnabled(False)
        self.btn_scan_abort.setEnabled(True)
        self.act_snapshot.setEnabled(False)

        self._scan_worker = ScanWorker(
            self.hosts, self.stages, grid, self.spin_scan_s.value(),
            self.cal, self.spin_scan_settle.value(), self.prev_clkdiv)
        # As with the snapshot: the worker restores the clock itself, so
        # clearing this now means a failed scan cannot leave a stale value
        # for disconnect to apply on top.
        self.prev_clkdiv = {}
        self._scan_worker.message.connect(self.log.log)
        self._scan_worker.progress.connect(self.on_scan_progress)
        self._scan_worker.done.connect(self.on_scan_done)
        self._scan_t0 = time.time()
        self._scan_worker.start()

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
        self.btn_scan_start.setEnabled(self.stages is not None)
        self.btn_scan_abort.setEnabled(False)
        self.act_snapshot.setEnabled(True)
        self.bar_scan.setFormat("%v / %m")
        self.lbl_state.setText("disconnected")
        if error:
            self.log.log(f"field map failed: {error}")
            QtWidgets.QMessageBox.warning(self, "Field map", error)
        if fm is None or not len(fm):
            return
        path = fm.save(os.path.join(
            "captures", time.strftime("fieldmap_%Y%m%d_%H%M%S")))
        sem = np.array([s.get("sem_ut", np.nan) for s in fm.stats])
        self.log.log(
            f"field map: {len(fm)} of {fm.meta['n_requested']} points, "
            f"noise median {np.nanmedian(sem):.3f} uT / worst "
            f"{np.nanmax(sem):.3f} uT -> {path}")
        self.log.log("field map: the carriers are still off the live stream — "
                     "press Connect to go back to the live view.")

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
    def on_connect(self):
        if self.source is not None:
            return
        fs = float(self.cmb_rate.currentData())
        self.act_connect.setEnabled(False)
        self.lbl_state.setText("connecting...")
        self.log.log(f"connecting to {', '.join(self.hosts)}"
                     + (f", setting {fs/1000:g} kSPS" if fs else ""))
        self._connect_worker = ConnectWorker(self.hosts, fs)
        self._connect_worker.done.connect(self.on_connected)
        self._connect_worker.progress.connect(self.on_connect_progress)
        self._connect_worker.start()

    def on_connect_progress(self, msg):
        # Connecting involves several seconds of deliberate settling delays.
        # Without this the window sits on "connecting..." long enough to look
        # like it has hung, which is exactly how it was first reported.
        self.lbl_state.setText(f"connecting -- {msg}")
        self.log.log(msg)

    def on_connected(self, source, prev, error):
        self.act_connect.setEnabled(True)
        if error:
            self.lbl_state.setText("disconnected")
            self.log.log(f"connect failed: {error}")
            QtWidgets.QMessageBox.critical(
                self, "Connect failed",
                f"{error}\n\nIf this says the stream closed immediately, "
                f"something else owns port 4210 -- usually a Phoebus "
                f"'Streaming Capture' still running on that box.")
            for h, p in (prev or {}).items():
                try:
                    import octobee_live as olive
                    olive.restore_rate(h, p)
                except Exception:                    # noqa: BLE001
                    pass
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
        for h, p in self.prev_clkdiv.items():
            try:
                import octobee_live as olive
                olive.restore_rate(h, p)
                self.log.log(f"{h}: clkdiv restored to {p}")
            except Exception as e:                   # noqa: BLE001
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
        """
        self.roll.clear()
        self.view3d.reset_scale()
        if hasattr(self, "cmb_units"):
            self.on_units()
        if self.collecting is not None and self.collecting["what"] == "magnet":
            self.btn_magnet.setChecked(False)
            self.log.log(f"magnet pass abandoned: {what} changed mid-pass")

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
                grouped = ocal.assemble(blocks, self.source.vpc)   # (n,16,4) V
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
                    rows = ocal.channel_health(raw, self.source.vpc,
                                               self.source.hosts)
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
                           "peak": None}
        self.log.log(f"collecting {seconds:g} s for {what}...")

    def _collect_block(self, b, grouped=None):
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
        if c["what"] == "sweep":
            # A sweep has to be solved from UNCORRECTED field, or the solver
            # would be fitting on top of whatever trim happens to be loaded and
            # the answer would depend on where you started. Decimate hard while
            # we are at it: the information is in the shape of the ellipse, and
            # a couple of hundred hertz resolves a hand roll a thousand times
            # over.
            if grouped is None:
                return
            raw = self.cal.to_mt(grouped, apply_zero=False, apply_gain=False,
                                 apply_matrix=False)
            c["blocks"].append(ocal.decimate(raw, max(1, c["decim"])))
            c["n"] += b.shape[0]
            if c["n"] >= c["need"]:
                data = np.concatenate(c["blocks"], axis=0)
                tag = c["tag"]
                self.collecting = None
                self._finish_sweep(data, tag)
            return

        c["blocks"].append(ocal.decimate(b, max(1, self.decim)))
        c["n"] += b.shape[0]
        if c["n"] >= c["need"]:
            data = np.concatenate(c["blocks"], axis=0)
            self.collecting = None
            self._finish_tare(data)

    def _finish_tare(self, data):
        # tare must be computed on data with the old zero removed, so add it back
        raw = data + self.cal.zero_mt[None, :, :]
        z = self.cal.tare(raw)
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
            self, "Save roll sweeps into", "captures")
        if not d:
            return
        for tag, sw in sorted(self.sweeps.items()):
            path = sw.save(os.path.join(d, f"rollsweep_{tag}"))
            self.log.log(f"wrote {path}")

    def on_load_sweeps(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Load roll sweeps", "captures", "Roll sweeps (*.npz)")
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
                               "need": 0, "peak": None, "baseline": base}
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
            self.log.log(f"magnet pass: peak |B| per sensor = "
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
        corr, skipped = self.cal.cross_calibrate(self.magnet_peaks, weights=w)
        note = (f"kept their previous trim (no usable response): "
                f"{', '.join(skipped)}") if skipped else "every live sensor trimmed"
        self.log.log(f"gain trim applied using "
                     f"{'geometry-weighted' if w is not None else 'raw'} peaks; "
                     f"{note}")
        self.lbl_magnet.setText(f"gain trim applied -- {note}")
        self._calibration_changed("the gain trim")
        self.refresh_cal_report()

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
        if path:
            self.cal = ocal.Calibration.load(path)
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
        rows = ocal.channel_health(raw, self.source.vpc, self.source.hosts)
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
        path = orec.default_name("channel_health", "csv")
        orec.write_health_csv(path, self.last_health)
        self._exported(path)

    # ---- data output ------------------------------------------------------
    def on_record(self, on):
        if on:
            if self.source is None:
                self.act_record.setChecked(False)
                return
            stamp = time.strftime("%Y%m%d_%H%M%S")
            if self.chk_csv.isChecked():
                p = os.path.join("captures", f"octobee_{stamp}.csv")
                self.csv_rec = orec.CsvRecorder(
                    p, self.out_rate, self.cal, self.geom,
                    tube_frame=self.chk_tube.isChecked(),
                    meta={"hosts": ",".join(self.source.hosts),
                          "stream_rate_hz": self.source.fs_hz})
                self.log.log(f"recording CSV to {p} at {self.out_rate:g} Hz")
            if self.chk_raw.isChecked():
                p = os.path.join("captures", f"octobee_{stamp}.bin")
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
        path = orec.default_name("snapshot", "npz")
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
        path = orec.default_name("sensor_summary", "csv")
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
        path = orec.default_name("octobee_report", "json")
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
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._scan_worker.abort()
            self._scan_worker.wait(30000)
        self.stage_timer.stop()
        if self.stages is not None:
            # These are exclusive-open USB devices: leaving them held means the
            # Kinesis application will not start until this process dies.
            self.stages.close()
            self.stages = None
        self.on_disconnect()
        super().closeEvent(ev)


ICON_NAME = "octobee.ico"


def _apply_app_icon(app):
    """Put the probe-head icon on the window and the taskbar button.

    The AppUserModelID is the non-obvious half. Without it Windows attributes
    the window to pythonw.exe, so the taskbar shows the generic Python icon and
    groups this with every other Python process -- leaving the desktop shortcut
    as the only place the application looks like itself.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ICON_NAME)
    if not os.path.exists(path):
        return
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "harrer.octobee.hallprobe.1")
        except Exception:                             # noqa: BLE001
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
    _apply_app_icon(app)
    app.setApplicationName("OCTO-BEE Hall probe")
    win = MainWindow(a)
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
