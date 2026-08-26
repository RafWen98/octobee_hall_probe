"""Where samples come from: the carriers, a saved capture, or a synthetic probe."""

import os
import queue
import time
from collections import deque

import numpy as np

from octobee.acq import carrier as ob
from octobee.calib import convert as ocal
from octobee.gui.constants import N_SENSORS

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
    # Quadrature encoder counts for the samples the last read() returned, as
    # (n, columns) raw uint32, or None where the source has no encoders. Left
    # beside read() rather than returned from it because only one carrier has
    # them and every existing caller wants the analogue channels; what matters
    # is that these rows are the SAME rows, trimmed identically, so a position
    # and a field sample with the same index really were converted together.
    enc = None
    enc_columns = 0
    enc_host = None
    enc_sites = ()

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
        # Which box the positions come from. Decided on what the carriers say
        # about themselves -- how many of their quadrature modules are actually
        # counting -- and NOT on whether a box contributes a longword between
        # the analogue channels and the scratchpad. Both of these carriers do:
        # acq1001_694 aggregates a quadrature module in site 2 with phaseA_en
        # off, so it emits a column that never moves, and choosing by "has some
        # encoder longwords" picks it over the 695, which has three that count.
        # That would have logged a constant as the probe's position.
        scored = [(len(getattr(lay, "counting_sites", ())), h, lay)
                  for h, lay in zip(self.hosts, self.layouts)]
        best = max(scored, key=lambda item: item[0], default=(0, None, None))
        self._enc_host = best[1] if best[0] else None
        self.enc_columns = (0 if self._enc_host is None
                            else int(best[2].n_site2_lw))
        self.enc_sites = (() if self._enc_host is None
                          else tuple(sorted(best[2].encoder_columns)))
        self._enc_pending = deque()
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
                if h == self._enc_host and blk.get("enc") is not None:
                    self._enc_pending.append(blk["enc"])
                if "temp_raw" in blk:
                    self._temp[h] = blk["temp_raw"]
                gaps, lost = ob.check_continuity(blk["sam_cnt"])
                self.gaps += gaps
                self.lost += lost
        n = min(self._n.values()) if self._n else 0
        if n <= 0:
            self.enc = None
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
        self.enc = self._take_enc(n)
        return out

    @property
    def enc_host(self):
        """Which carrier the encoder counts come from, or None."""
        return self._enc_host

    def _take_enc(self, n):
        """The encoder rows belonging to the n analogue rows just handed out.

        Trimmed by exactly the same n, from a buffer fed by exactly the same
        blocks, so row i of the counts and row i of the field are the same
        sample. If the two ever disagree in length the counts are dropped
        rather than shifted into line: a position column that is silently off
        by a few samples is worse than one that is absent, because nothing
        downstream can tell.
        """
        if not self._enc_pending:
            return None
        buf = np.concatenate(self._enc_pending, axis=0)
        self._enc_pending.clear()
        if buf.shape[0] < n:
            self.error = self.error or (
                f"{self._enc_host}: {buf.shape[0]} encoder rows against {n} "
                f"analogue rows -- encoder positions dropped for this block")
            return None
        rest = buf[n:]
        if rest.shape[0]:
            self._enc_pending.append(rest)
        return buf[:n]

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
