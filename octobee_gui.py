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
import octobee_record as orec
import probe_geometry as pgeom
from probe_view3d import ProbeView3D, color_for

N_SENSORS = pgeom.N_SENSORS
AXES = ("Bx", "By", "Bz")
OUT_RATES = (100.0, 200.0, 500.0, 1000.0, 2000.0)
STREAM_RATES = {"20 kSPS (link-safe)": 20000.0,
                "50 kSPS": 50000.0,
                "200 kSPS (full, may drop)": 200000.0,
                "leave the box alone": 0.0}

# The per-channel health scan is the most expensive thing in the refresh loop
# and its answer changes only when a connector does, so it runs on its own
# slower clock over a short window. Left on the display clock over the full
# history it starves the reader threads and the carriers' queues overflow.
HEALTH_PERIOD_S = 2.0
HEALTH_WINDOW_S = 1.0
RAW_HISTORY_S = 5.0

pg.setConfigOptions(antialias=True, background=(18, 20, 26), foreground=(210, 214, 222))


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
        self.t0 = time.time()
        self.error = None

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

    def _field_mt(self, t):
        """Dipole moving along +Z, orbiting the tube, seen in each chip's axes."""
        z = (t * 60.0) % (self.geom.tube_length_mm + 120.0) - 60.0
        ang = 2 * np.pi * t / 11.0
        r = self.geom.tube_width_mm / 2.0 + 35.0
        src = np.array([r * np.cos(ang), r * np.sin(ang), z])
        m = np.array([0.0, 0.0, 1.0]) * 6.0e4          # arbitrary strength, mT*mm^3
        pos = self.geom.positions()
        d = pos - src
        rr = np.linalg.norm(d, axis=1, keepdims=True)
        rr = np.maximum(rr, 8.0)
        rhat = d / rr
        b_tube = (3.0 * rhat * (rhat @ m)[:, None] - m[None, :]) / rr ** 3
        R = self.geom.rotations()
        return np.einsum("sji,sj->si", R, b_tube)      # tube -> chip frame

    def read(self):
        now = time.time()
        n = int((now - self.t_last) * self.fs_hz)
        if n <= 0:
            return None
        n = min(n, int(self.fs_hz))
        t = self.t + np.arange(n) / self.fs_hz
        self.t_last, self.t = now, t[-1]

        # field at ~20 points, interpolated up -- the magnet is slow
        k = max(2, n // 256)
        tk = t[::k]
        bk = np.array([self._field_mt(tt) for tt in tk])          # (k,16,3)
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
                st = ob.Streamer(h, lay, block_samples=self.block_samples,
                                 take_over=False)
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

    def __init__(self, geom):
        super().__init__()
        self.geom = geom
        self.mode = self.MODES[0]
        self.visible = set(range(N_SENSORS))
        self.dead = set()
        self.colors = sensor_colors()

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "time", units="s")
        self.plot.setLabel("left", "B", units="mT")
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.setClipToView(True)
        self.legend = self.plot.addLegend(colCount=4, labelTextSize="7pt")

        self.mag_curves, self.axis_curves = [], []
        for s in range(N_SENSORS):
            c = self.plot.plot(pen=pg.mkPen(self.colors[s], width=1.6),
                               name=f"S{s+1}")
            self.mag_curves.append(c)
            row = []
            for a, style in enumerate((QtCore.Qt.PenStyle.SolidLine,
                                       QtCore.Qt.PenStyle.DashLine,
                                       QtCore.Qt.PenStyle.DotLine)):
                row.append(self.plot.plot(
                    pen=pg.mkPen(self.colors[s], width=1.1, style=style)))
            self.axis_curves.append(row)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plot)
        self.set_mode(self.mode)

    def set_mode(self, mode):
        self.mode = mode
        mag = mode == self.MODES[0]
        self.plot.setLabel("left", "|B|" if mag else "B", units="mT")
        self._apply_visibility()

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

    def update_data(self, b_mt, fs_out):
        n = b_mt.shape[0]
        if n < 2:
            return
        t = (np.arange(n) - n) / fs_out
        shown = self.visible - self.dead
        if self.mode == self.MODES[0]:
            mag = np.linalg.norm(b_mt, axis=-1)
            for s in shown:
                self.mag_curves[s].setData(t, mag[:, s])
        else:
            b = self.geom.to_tube_frame(b_mt) if "tube" in self.mode else b_mt
            for s in shown:
                for a in range(3):
                    self.axis_curves[s][a].setData(t, b[:, s, a])


class SensorBars(QtWidgets.QWidget):
    """
    Peak |B| per sensor since the last refresh -- the "are the spikes the same
    height" view, and the reason this window exists.

    Excluded sensors are drawn as a flat red stub rather than at their real
    value: a railed channel reads nearly 200 mT, which would flatten every real
    bar against the axis.
    """

    def __init__(self):
        super().__init__()
        self.plot = pg.PlotWidget()
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
        self.collecting = None          # 'tare' | 'magnet' -> list of blocks
        self.collect_target = 0
        self.magnet_peaks = None
        self.last_health = None
        self._last_table = None
        self._last_health_t = 0.0
        self._last_dropped = 0
        self._connect_worker = None
        self._snap_worker = None

        self.setWindowTitle("OCTO-BEE Hall probe")
        self.resize(1720, 980)
        self._build_ui()
        self._apply_dark()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(60)
        self.slow = QtCore.QTimer(self)
        self.slow.timeout.connect(self.on_slow_tick)
        self.slow.start(500)

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
                f"which put every sensor on the +/-20 mT range. On this probe "
                f"S1-S8 actually run +/-40 mT, so their readings will be out by "
                f"1.82x until you set the range per sensor in the Sensors tab "
                f"and save the calibration.")

    # ---- construction ----------------------------------------------------
    def _build_ui(self):
        self._build_toolbar()
        self.log = LogPane()

        self.view3d = ProbeView3D(self.geom)
        self.bars = SensorBars()
        self.plot = LivePlot(self.geom)
        self.table = SensorTable(self.geom)
        self.table.range_changed.connect(self.on_range_changed)
        self.table.set_ranges(self.cal.ranges_mt)

        left = QtWidgets.QTabWidget()
        left.addTab(self._live_tab(), "Live")
        left.addTab(self.table, "Sensors")
        left.addTab(self._calib_tab(), "Calibration")
        left.addTab(self._health_tab(), "Diagnostics")
        left.addTab(self._export_tab(), "Data output")
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
            "The link tops out near 9.8 MB/s but each box makes 19.2 MB/s at "
            "200 kSPS, so a sustained full-rate stream cannot keep up. 20 kSPS "
            "streams cleanly and is still ~10000x oversampled for a hand-passed "
            "magnet. The original clkdiv is restored on disconnect.")
        tb.addWidget(self.cmb_rate)

        tb.addWidget(QtWidgets.QLabel("  output rate "))
        self.cmb_out = QtWidgets.QComboBox()
        for r in OUT_RATES:
            self.cmb_out.addItem(f"{r:g} Hz", r)
        self.cmb_out.setCurrentIndex(list(OUT_RATES).index(500.0))
        self.cmb_out.currentIndexChanged.connect(self.on_out_rate)
        tb.addWidget(self.cmb_out)

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

        g3 = QtWidgets.QGroupBox("3. Calibration file and geometry")
        f3 = QtWidgets.QHBoxLayout(g3)
        for text, slot in (("Save calibration", self.on_save_cal),
                           ("Load calibration", self.on_load_cal),
                           ("Edit geometry", self.on_edit_geometry),
                           ("Reload geometry", self.on_reload_geometry)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            f3.addWidget(b)
        self.chk_vcm = QtWidgets.QCheckBox("subtract VCM")
        self.chk_vcm.setChecked(self.cal.subtract_vcm)
        self.chk_vcm.setToolTip(
            "Each chip's own virtual ground. The 16 differ by up to ~90 mV, "
            "which is ~1.4 mT of fake field at the 20 mT range. Leave this on.")
        self.chk_vcm.toggled.connect(self.on_vcm_toggle)
        f3.addWidget(self.chk_vcm)
        f3.addStretch(1)
        lay.addWidget(g3)

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
        blocks = self.source.read()
        if not blocks or blocks[0].shape[0] == 0:
            if getattr(self.source, "error", None):
                self.log.log(f"stream error: {self.source.error}")
                self.source.error = None
            return

        self._keep_raw(blocks)
        if self.raw_rec is not None:
            self.raw_rec.write(blocks)

        grouped = ocal.assemble(blocks, self.source.vpc)          # (n,16,4) volts
        b = self.cal.to_mt(grouped)                                # (n,16,3) mT
        bd = ocal.decimate(b, self.decim)
        if bd.shape[0] == 0:
            return
        self.roll.push(bd.astype(np.float32))

        if self.csv_rec is not None:
            self.csv_rec.write(bd)

        if self.collecting is not None:
            self._collect_block(b)

        recent = self.roll.view()
        if recent.shape[0]:
            k = min(8, recent.shape[0])
            fs = self.view3d.update_fields(recent[-k:].mean(axis=0))
            if self.chk_auto.isChecked():
                self.spin_fs.blockSignals(True)
                self.spin_fs.setValue(max(fs, 0.001))
                self.spin_fs.blockSignals(False)

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
        self.plot.update_data(recent, self.out_rate)
        # Peak over a trailing half second, not the instantaneous value: a magnet
        # passed by hand is over well inside one refresh, so sampling one point
        # would miss most passes.
        k = max(2, min(recent.shape[0], int(self.out_rate * self.bars.window_s)))
        self.bars.update_values(
            np.linalg.norm(recent[-k:], axis=-1).max(axis=0), self.cal.dead)

        # A dropped block is a hole in whatever is being recorded, so say so
        # rather than leaving it as a number in the corner of the status bar.
        dropped = self.source.stats().get("dropped blocks", 0)
        if dropped > self._last_dropped:
            if self.csv_rec is not None or self.raw_rec is not None:
                self.log.log(f"WARNING: {dropped - self._last_dropped} block(s) "
                             f"dropped while recording -- the file has a gap "
                             f"there. Lower the output rate or the stream rate.")
            self._last_dropped = dropped

        st = self.source.stats()
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

    def _collect_block(self, b):
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
                    [self.source.fs_hz] * len(self.source.hosts))
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

    def on_sensor_toggle(self, _=None):
        self.plot.set_visible_sensors(
            {i for i, cb in enumerate(self.chk_sensors) if cb.isChecked()})

    def _set_all_sensors(self, on):
        for cb in self.chk_sensors:
            cb.blockSignals(True)
            cb.setChecked(on)
            cb.blockSignals(False)
        self.on_sensor_toggle()

    def on_out_rate(self):
        self.out_rate = float(self.cmb_out.currentData())
        self.roll.resize(max(4, int(self.out_rate * self.window_s)))
        self.log.log(f"output rate {self.out_rate:g} Hz "
                     f"(decimation {self.decim}x from the stream)")

    def on_window(self, v):
        self.window_s = float(v)
        self.roll.resize(max(4, int(self.out_rate * self.window_s)))

    def closeEvent(self, ev):
        if self.act_record.isChecked():
            self.act_record.setChecked(False)
        self.on_disconnect()
        super().closeEvent(ev)


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
    p.add_argument("--screenshot", help="render one frame to this PNG and exit "
                                        "(for headless checks)")
    p.add_argument("--screenshot-tab", type=int, default=0,
                   help="which tab to show in the screenshot")
    p.add_argument("--screenshot-warmup", type=float, default=3.0,
                   help="seconds of data to collect before the screenshot")
    a = p.parse_args()

    app = QtWidgets.QApplication(sys.argv)
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
