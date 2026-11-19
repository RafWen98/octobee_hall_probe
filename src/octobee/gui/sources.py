"""Where samples come from: the carriers, a saved capture, or a synthetic probe."""

import os
import queue
import time

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
        # Rows received and not yet handed out, one (n, nchan) array per box,
        # and the same for the encoder counts of the encoder box.
        self._buf = {h: np.empty((0, lay.nchan_ai), dtype=np.int16)
                     for h, lay in zip(self.hosts, self.layouts)}
        self._enc_buf = None
        # Each pending row's own sample counter, cut in step with the rows, so
        # the counter of the first row handed out is known whatever was
        # discarded before it (see block_index).
        self._sam_buf = {h: np.empty(0, dtype=np.uint32) for h in self.hosts}
        # What the rows the last read() returned were: see block_index.
        self.block_index = None
        # Rows handed out so far. Both boxes are paired by ROW INDEX, so this
        # is the one clock the pairing lives on; see _owe().
        self._taken = 0
        # Last sample counter seen per box, so the step between one block and
        # the next is checked too (None until a box's first block).
        self._last_sam = dict.fromkeys(self.hosts)
        # Samples per box that the counter says went by, lost ones included.
        self._span = dict.fromkeys(self.hosts, 0)
        # Rows still to be thrown away from a box when they arrive, as
        # (row index, count); see _owe().
        self._debt = {h: [] for h in self.hosts}
        self._reported = set()
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

    @property
    def failed(self):
        """Hosts whose reader thread has died, as a list (empty while healthy).

        read() reports each error once, so the caller can no longer tell from
        `error` alone that nothing more will ever arrive; this can. One dead
        box is enough to stop the source -- the other half of every row is
        missing -- so a non-empty list means "stop recording and reconnect".
        """
        return [h for h, st in zip(self.hosts, self.streamers) if st.error]

    # A counter step this large is the counter going BACKWARDS (a box that
    # restarted), not a loss of two billion samples. Counted as a gap, but
    # there is nothing sensible to realign by.
    _MAX_LOSS = 1 << 31

    def read(self):
        for h, st in zip(self.hosts, self.streamers):
            # Once. st.error persists after a failure, and the GUI polls this
            # every 50 ms, so re-reporting it was twenty identical log lines a
            # second for as long as the session stayed open.
            if st.error and h not in self._reported and self.error is None:
                self._reported.add(h)
                self.error = f"{h}: {st.error}"
            blocks = [blk for blk in st.get_all() if blk is not None]
            if not blocks:
                continue
            # One block per box per tick, joined here, not one at a time:
            # _accept grows the pending buffer by concatenation, which copies
            # everything already pending. At two or three blocks a tick that
            # is nothing; a GUI that has fallen behind collects a full queue
            # (128) per tick and the copying grows with the SQUARE of that --
            # 45x the cost per sample, enough that a stream which had fallen
            # behind for a few seconds never caught up again and shed ~20%
            # of its blocks for the rest of the run. Joining first copies
            # each sample once. find_gaps checks every step inside the joined
            # counter, so a break between two of these blocks is still seen
            # at the row it happened.
            self._take_block(h, self._join(blocks))
            for blk in blocks:
                if "temp_raw" in blk:
                    self._temp[h] = blk["temp_raw"]
        n = min(len(b) for b in self._buf.values()) if self._buf else 0
        if n <= 0:
            self.enc = None
            self.block_index = None
            return None
        out = []
        for h in self.hosts:
            out.append(self._buf[h][:n])
            self._buf[h] = self._buf[h][n:]
        # For the raw archive: where in each box's own counter this block
        # starts, and the loss totals so far. Rows are contiguous in the file
        # even where a hole was repaired, so this is what lets an offline
        # reader see that a box's counter jumped inside or between blocks.
        self.block_index = {
            "sam_cnt": [int(self._sam_buf[h][0]) for h in self.hosts],
            "gaps": self.gaps, "lost": self.lost,
            "dropped_blocks": sum(st.dropped for st in self.streamers)}
        for h in self.hosts:
            self._sam_buf[h] = self._sam_buf[h][n:]
        self.enc = self._take_enc(n)
        self._taken += n
        return out

    @staticmethod
    def _join(blocks):
        """One block holding the rows of `blocks`, in order (see read)."""
        if len(blocks) == 1:
            return blocks[0]
        out = dict(blocks[-1])
        for key in ("ai", "sam_cnt", "enc"):
            parts = [b.get(key) for b in blocks]
            if all(p is not None for p in parts):
                out[key] = np.concatenate(parts, axis=0)
            else:
                out.pop(key, None)
        return out

    def _take_block(self, h, blk):
        """Accept one block from box h, noticing and repairing any loss.

        The two carriers are paired by row index: row i of box A is shown,
        recorded and converted against row i of box B, and the encoder counts
        (which ride in box 695's frames) against both. That is only true while
        neither box has lost a sample. A block shed by a full queue, or
        samples missing inside one, removes rows from ONE box, and every row
        after that is permanently shifted against the other -- one half of the
        probe paired with a partner half and a position from a different
        instant, with nothing in the file to show it.

        So each box's own sample counter is followed across blocks (the step
        BETWEEN blocks was never checked, which is how a dropped block went
        unseen) and every break is counted in gaps/lost AND repaired: the
        other box is made to discard the same number of samples at the same
        place, so the rows after the hole are paired as they were before it.
        The time axis therefore has a hole of that many samples, which the
        counters report, instead of a silent shift.

        A box's counter is only ever compared with ITS OWN earlier values. The
        two start at unrelated numbers, so nothing is compared across boxes.
        """
        sam = np.asarray(blk["sam_cnt"], dtype=np.uint32)
        if not len(sam):
            return
        ai = blk["ai"]
        enc = blk.get("enc") if h == self._enc_host else None
        pos = 0
        for row, lost in ob.find_gaps(sam, self._last_sam[h]):
            self._accept(h, ai[pos:row], sam[pos:row],
                         None if enc is None else enc[pos:row])
            pos = row
            self.gaps += 1
            if lost >= self._MAX_LOSS:
                continue
            self.lost += lost
            self._span[h] += lost
            for other in self.hosts:
                if other != h:
                    self._owe(other, self._taken + len(self._buf[h]), lost)
        self._accept(h, ai[pos:], sam[pos:], None if enc is None else enc[pos:])
        self._last_sam[h] = int(sam[-1])
        self._span[h] += len(sam)

    def _accept(self, h, ai, sam, enc):
        """Append rows to box h, discarding any it owes (see _owe)."""
        while len(ai):
            have = self._taken + len(self._buf[h])
            due = self._debt[h]
            if due and due[0][0] <= have:
                at, count = due[0]
                k = min(count, len(ai))
                ai, sam = ai[k:], sam[k:]
                enc = None if enc is None else enc[k:]
                if k == count:
                    due.pop(0)
                else:
                    due[0] = (at, count - k)
                continue
            k = len(ai) if not due else min(len(ai), due[0][0] - have)
            self._buf[h] = np.concatenate([self._buf[h], ai[:k]], axis=0)
            self._sam_buf[h] = np.concatenate([self._sam_buf[h], sam[:k]])
            if enc is not None:
                self._enc_buf = (enc[:k] if self._enc_buf is None
                                 else np.concatenate([self._enc_buf, enc[:k]]))
            ai, sam = ai[k:], sam[k:]
            enc = None if enc is None else enc[k:]

    def _owe(self, h, at, count):
        """Box h must give up `count` rows starting at row index `at`.

        `at` is where the other box lost samples, in the shared row numbering.
        Rows of h that are already here are cut out of its pending data (they
        are always still pending: `at` is never behind what has been handed
        out, because the box that lost samples had not yet got that far). Any
        part h has not received yet becomes a debt, paid as those rows
        arrive -- which also covers a count larger than what is pending, so a
        long loss is carried forward rather than clipped.

        For the encoder box the counts are cut with the same rows, so a count
        stays with the sample it was latched in. When the loss was on the
        encoder box itself nothing is cut from its counts: they are missing
        exactly where its samples are, and the OTHER box does the discarding.
        """
        have = self._taken + len(self._buf[h])
        if at < have:
            i = at - self._taken
            k = min(count, have - at)
            buf = self._buf[h]
            self._buf[h] = np.concatenate([buf[:i], buf[i + k:]], axis=0)
            sam = self._sam_buf[h]
            self._sam_buf[h] = np.concatenate([sam[:i], sam[i + k:]])
            if h == self._enc_host and self._enc_buf is not None:
                e = self._enc_buf
                self._enc_buf = np.concatenate([e[:i], e[i + k:]], axis=0)
            count -= k
        if count:
            self._debt[h].append((at, count))
            self._debt[h].sort()

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
        buf, self._enc_buf = self._enc_buf, None
        if buf is None:
            return None
        if buf.shape[0] < n:
            self.error = self.error or (
                f"{self._enc_host}: {buf.shape[0]} encoder rows against {n} "
                f"analogue rows -- encoder positions dropped for this block")
            return None
        if buf.shape[0] > n:
            self._enc_buf = buf[n:]
        return buf[:n]

    def temperatures(self):
        return ocal.temperatures_c([self._temp[h] for h in self.hosts])

    def stats(self):
        total = sum(st.bytes_read for st in self.streamers)
        dt = max(time.time() - self.t0, 1e-3)
        out = {"MB/s": total / dt / 1e6,
               "gaps": self.gaps, "lost": self.lost,
               "dropped blocks": sum(st.dropped for st in self.streamers)}
        if len(self.hosts) > 1:
            # The boxes free-run on their own clocks (200001 Hz against
            # 199999 Hz on this rig), so the number of samples each has
            # produced slowly diverges, ~2 per second. Rows are paired by
            # order, not resampled, so this is how far apart in time the two
            # halves of a row have drifted. It includes whatever each box has
            # buffered, so read its trend, not a single value.
            out["box offset"] = (self._span[self.hosts[0]]
                                 - self._span[self.hosts[1]])
        return out

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
