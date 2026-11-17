#!/usr/bin/env python3
"""
octobee/record.py -- getting data out of the probe and onto disk.

Three routes, because they answer different questions:

  CsvRecorder   calibrated millitesla, decimated to a sane output rate, written
                continuously while you work. This is the one you hand to
                whoever asked for "the data". Opens in anything.

  RawRecorder   the untouched 16-bit counts, streamed to a flat binary file with
                a JSON sidecar. Nothing is lost and nothing is interpreted, so a
                capture taken today survives a later correction to the channel
                map, the VCM handling or the gain registers. Unbounded length --
                it never accumulates in RAM.

  report_*      one-shot exports of the calibration and health tables.

A note on rates: the ADC free-runs at 200 kSPS, but a hand-passed magnet is a
sub-Hz signal and the Ethernet link caps near 9.8 MB/s. Recording calibrated CSV
at 200 kSPS would produce ~90 MB/s of text describing a signal with no content
above a few Hz. The default output rate is 1 kHz, which is already ~1000x
oversampled for the physics, and the raw route exists for when you really do
need every sample.
"""

import argparse
import csv
import hashlib
import json
import os
import time

import numpy as np

from octobee import paths
from octobee.acq import carrier as ob
from octobee.calib import convert as ocal
from octobee.calib import geometry as pg
from octobee.motion import encoder as oenc

AXES = ocal.AXES
N_SENSORS = ocal.N_SENSORS

# The LiveSource.stats() entries that say how much of the stream went missing.
_STREAM_KEYS = ("gaps", "lost", "dropped blocks")


def _stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def default_name(prefix, ext, directory=None):
    """A path in the captures directory that does not exist yet.

    The stamp has one-second resolution, and the recorders open their files
    with "w". Two recordings started in the same second -- a rollover, or
    Record pressed twice -- therefore got the SAME name, and the second one
    silently truncated the first. A suffix is added until the name is free.
    """
    directory = directory or paths.captures_dir()
    os.makedirs(directory, exist_ok=True)
    stem = f"{prefix}_{_stamp()}"
    path = os.path.join(directory, f"{stem}.{ext}")
    n = 1
    while os.path.exists(path) or os.path.exists(path + ".json"):
        n += 1
        path = os.path.join(directory, f"{stem}_{n}.{ext}")
    return path


def calibration_id(cal):
    """
    Twelve hex characters identifying the calibration that produced a file.

    Covers every field that changes a converted number -- ranges, VCM, zero,
    gain trim and the pose matrix -- and nothing that does not, so re-saving
    calibration.json with a new comment does not invalidate old captures.
    """
    h = hashlib.sha256()
    for part in (cal.ranges_mt, cal.zero_mt, cal.gain_corr, cal.matrix,
                 np.array([float(cal.subtract_vcm)])):
        h.update(np.ascontiguousarray(part, dtype=np.float64).tobytes())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------
# calibrated CSV
# --------------------------------------------------------------------------

class CsvRecorder:
    """
    Streaming CSV of calibrated field data.

    Columns: t_s, then per sensor Bx/By/Bz and |B|, all in millitesla. With
    `tube_frame` the per-sensor axes are additionally rotated into the common
    tube frame, which is the only form in which different sensors' components
    can be compared or summed.

    Where the stream carries encoder counts -- acq1001_695 aggregates the three
    quadrature sites into its frames; see octobee/motion/encoder.py -- the
    counts for the same rows are appended, and with them the X/Y/Z travel in
    millimetres for every axis whose scale has been fitted. That is the whole
    point of counting position in the frame rather than polling it: the count
    in a row was latched when that row's samples were converted, so a recorded
    field value and the position it was measured at share a clock by
    construction, with nothing to interpolate -- for the sensors on the SAME
    carrier as the encoders (S9-S16, acq1001_695). S1-S8 are on the other box,
    which free-runs on its own clock and is only paired with this one by
    sample order; see the timebase and encoder_timing lines of the header.

    `enc_datum` is {axis: (counts_at_datum, mm_at_datum)} -- what the encoder
    read at the moment the controller said the axis was at mm_at_datum. A
    quadrature counter is incremental, so this is what makes the millimetres
    absolute. Without one for an axis the column still gets written, named
    `X_rel_mm` rather than `X_mm`: travel measured from wherever the head
    happened to be on the first row. The name is the difference, so a column
    cannot be read as absolute when it is not.
    """

    def __init__(self, path, fs_out, cal, geom=None, tube_frame=False,
                 include_mag=True, meta=None, samples_per_row=None,
                 encoders=None, enc_columns=0, enc_datum=None):
        self.path = path
        self.fs_out = float(fs_out)
        self.samples_per_row = samples_per_row
        self.cal = cal
        self.geom = geom
        self.tube_frame = bool(tube_frame and geom is not None)
        self.include_mag = include_mag
        self.encoders = encoders if encoders else None
        self.enc_datum = dict(enc_datum or {})
        # Every column the stream carries, not only the calibrated ones. A
        # column nobody has fitted a scale to yet is still a real measurement,
        # and a recording made today is the only place it can be recovered
        # from once the run is over.
        self.enc_columns = int(enc_columns or 0)
        if self.encoders is not None:
            self.enc_columns = max(self.enc_columns,
                                   self.encoders.columns_needed())
        self.enc_axes = [
            a for a in oenc.AXES
            if self.encoders is not None and a in self.encoders
            and self.encoders.axes[a]["column"] < self.enc_columns]
        # Latched from the first finite count on a relative axis; see
        # _positions_mm.
        self._origin = {}
        self.n_rows = 0
        # Rows written with no counts against them -- a block that arrived
        # without them, or with the wrong number. Counted rather than
        # discarded: the field data in those rows is good.
        self.n_unpaired = 0
        self.t = 0.0
        self.started = time.time()
        # Stream loss counters as first and last reported; see note_stream.
        self._stream0 = None
        self._stream = None
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, "w", encoding="utf-8", newline="")
        self._write_header(meta or {})

    def _write_header(self, meta):
        frame = "tube" if self.tube_frame else "chip"
        info = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "units": "millitesla", "frame": frame,
                "output_rate_hz": self.fs_out,
                "vcm_subtracted": self.cal.subtract_vcm,
                "ranges_mt": self.cal.ranges_mt.tolist(),
                "tare_applied": bool(np.any(self.cal.zero_mt)),
                "gain_trim_applied": not bool(np.allclose(self.cal.gain_corr, 1.0)),
                "pose_matrix_applied": bool(self.cal.has_matrix),
                # A short digest of everything that affects the numbers below,
                # so a CSV can be matched back to the calibration that made it
                # even after calibration.json has moved on.
                "calibration_id": calibration_id(self.cal),
                "excluded": sorted(self.cal.dead)}
        # What a row actually is, stated rather than left to be inferred.
        #
        # Files written before 2026-08-26 do not carry these two lines, and for
        # those the inference was wrong: the live recorder decimated each
        # arriving block on its own and dropped the remainder, so rows were
        # contiguous in the file and not in time -- t_s ran slow by the
        # discarded fraction, up to 9.9% at 100 Hz. A file with a "timebase"
        # line was written after the carry was added.
        #
        # The carry fixed the remainder and nothing else. "t_s is real elapsed
        # time" was still false after a dropped block or lost samples, because
        # those are simply absent from the rows and t_s steps over them by one
        # row, and it was never exactly true across the two carriers, which
        # free-run on their own clocks (200001 Hz against 199999 Hz). So say
        # what a row is, and where to look for what it is not.
        if self.samples_per_row:
            info["samples_per_row"] = int(self.samples_per_row)
        info["timebase"] = (
            "t_s is the row number over the output rate. No stream sample is "
            "dropped by decimation, but samples LOST in transit (a dropped "
            "block, a break in the sample counter) are not in the file and t_s "
            "does not skip over them, so after any such gap it is not real "
            "elapsed time -- the stream_gaps footer counts them where the "
            "application reported them. The two carriers are not "
            "synchronised: each free-runs on its own clock and they are paired "
            "row for row by sample order, not resampled, so S1-S8 and S9-S16 "
            "in one row drift apart by about 2 samples per second")
        if self.enc_columns:
            info["encoder_columns"] = self.enc_columns
            info["encoder_axes"] = (
                self.encoders.describe() if self.encoders
                else "no encoder axes calibrated -- counts only, no mm")
            info["encoder_datum"] = self._datum_note()
            info["encoder_timing"] = (
                f"counts were latched by the ADC clock of {ob.DEFAULT_UUTS[1]} "
                f"(S9-S16), in the same samples the S9-S16 field on this row "
                f"was averaged from -- no interpolation, no clock offset. "
                f"S1-S8 come from {ob.DEFAULT_UUTS[0]}, a separate carrier "
                f"paired by sample order, so against them the position is only "
                f"as good as that pairing (see timebase)")
        info.update(meta)
        for k, v in info.items():
            self._f.write(f"# {k}: {v}\n")
        if self.tube_frame:
            self._f.write("# axes are tube-frame X/Y/Z: +Z along the tube, "
                          "+X out of face 0\n")
        else:
            self._f.write("# axes are each chip's own Bx/By/Bz -- they point in "
                          "different lab directions per sensor; compare |B|\n")
        cols = ["t_s"]
        for s in range(1, N_SENSORS + 1):
            cols += [f"S{s}_{a}_mT" for a in ("Bx", "By", "Bz")]
            if self.include_mag:
                cols.append(f"S{s}_absB_mT")
        n_field = len(cols)
        cols += [f"enc{c}_counts" for c in range(self.enc_columns)]
        cols += [f"{a.upper()}_mm" if a in self.enc_datum
                 else f"{a.upper()}_rel_mm" for a in self.enc_axes]
        if self.enc_columns:
            self._f.write("# enc*_counts are continuous (unwrapped) quadrature "
                          "counts averaged over the row; position columns are "
                          "derived from them\n")
        self._f.write(",".join(cols) + "\n")
        self._cols = cols
        # Per column, because %.6g is right for millitesla and wrong for
        # counts: a 32-bit counter passes six significant digits within a few
        # millimetres of travel, after which the position column would be
        # quantised by the format it was printed with.
        self._fmt = (["%.6g"] * n_field + ["%.10g"] * self.enc_columns
                     + ["%.9g"] * len(self.enc_axes))

    def _datum_note(self):
        """What the millimetre columns are measured from, in words."""
        if not self.enc_axes:
            return "not applicable -- no axis has a fitted counts/mm scale"
        parts = []
        for a in self.enc_axes:
            d = self.enc_datum.get(a)
            if d is None:
                parts.append(f"{a} none -- {a.upper()}_rel_mm is travel from "
                             f"the first row, not an absolute position")
            else:
                parts.append(f"{a} {d[0]:,.0f} counts = {d[1]:.4f} mm from "
                             f"the controller")
        return "; ".join(parts)

    def note_stream(self, stats):
        """Record the source's loss counters, for the footer written at close.

        `stats` is LiveSource.stats(): its gaps / lost / dropped blocks are
        totals since the source connected, so the first call is kept as the
        baseline and the footer reports what happened while THIS file was
        open. A file that never gets this says nothing about gaps, and its
        header says so.
        """
        got = {k: int(stats.get(k, 0)) for k in _STREAM_KEYS}
        if self._stream0 is None:
            self._stream0 = got
        self._stream = got

    def write(self, b_mt, counts=None, stream=None):
        """Append (n, 16, 3) millitesla already decimated to fs_out.

        `stream` is optionally the source's stats() dict, passed through to
        note_stream.

        `counts` is the (n, columns) block of continuous encoder counts for
        THE SAME rows, decimated by the same factor. A block whose counts do
        not line up row for row is written without them -- as empty position
        columns -- rather than with them shifted, because a position column a
        few rows out is undetectable in the file and a blank one is not.
        """
        b = np.asarray(b_mt, float)
        if b.ndim != 3 or b.shape[0] == 0:
            return 0
        if self.tube_frame:
            b = self.geom.to_tube_frame(b)
        n = b.shape[0]
        t = self.t + np.arange(n) / self.fs_out
        self.t = t[-1] + 1.0 / self.fs_out
        if self.include_mag:
            mag = np.linalg.norm(b, axis=-1)[:, :, None]
            block = np.concatenate([b, mag], axis=2).reshape(n, -1)
        else:
            block = b.reshape(n, -1)
        parts = [t[:, None], block]
        if self.enc_columns:
            c = self._counts_block(counts, n)
            parts.append(c)
            if self.enc_axes:
                parts.append(self._positions_mm(c))
        rows = np.concatenate(parts, axis=1)
        np.savetxt(self._f, rows, delimiter=",", fmt=self._fmt)
        # Flushed every call, as the raw recorder is: without it a killed run
        # loses up to a buffer of rows and leaves the last one cut off
        # mid-line, which a reader then fails on. A call is one acquisition
        # tick (~20 a second) carrying a few dozen rows of a few kilobytes at
        # most, so one flush per call costs nothing measurable.
        self._f.flush()
        if stream is not None:
            self.note_stream(stream)
        self.n_rows += n
        return n

    def _counts_block(self, counts, n):
        """(n, enc_columns) counts, NaN where this block carried none."""
        out = np.full((n, self.enc_columns), np.nan)
        if counts is None:
            self.n_unpaired += n
            return out
        counts = np.asarray(counts, dtype=float)
        if counts.ndim != 2 or counts.shape[0] != n:
            self.n_unpaired += n
            return out
        k = min(self.enc_columns, counts.shape[1])
        out[:, :k] = counts[:, :k]
        return out

    def _positions_mm(self, counts):
        """(n, enc_columns) counts -> (n, len(self.enc_axes)) millimetres.

        An axis with a datum is absolute: the controller's millimetres at the
        datum plus the encoder's own displacement since. An axis without one
        counts from its first finite reading in this file, which is the most
        that can honestly be said about it -- and is still the travel, which
        is what a swept recording is usually about.
        """
        out = np.full((len(counts), len(self.enc_axes)), np.nan)
        for i, axis in enumerate(self.enc_axes):
            col = self.encoders.axes[axis]["column"]
            datum = self.enc_datum.get(axis)
            if datum is None:
                if axis not in self._origin:
                    finite = counts[np.isfinite(counts[:, col]), col]
                    if not len(finite):
                        continue
                    self._origin[axis] = float(finite[0])
                datum = (self._origin[axis], 0.0)
            out[:, i] = float(datum[1]) + self.encoders.displacement_mm(
                counts[:, col], axis, datum[0])
        return out

    @property
    def size_bytes(self):
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def close(self):
        if self._f and not self._f.closed:
            if self._stream is not None:
                d = {k: self._stream[k] - self._stream0[k]
                     for k in _STREAM_KEYS}
                self._f.write(
                    f"# stream_gaps: {d['gaps']} break(s) in a sample counter, "
                    f"{d['lost']} sample(s) lost, {d['dropped blocks']} "
                    f"block(s) dropped while this file was open -- each is a "
                    f"hole in t_s that the rows do not show\n")
            self._f.close()
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------
# raw archival
# --------------------------------------------------------------------------

class RawRecorder:
    """
    Streams raw int16 counts to <path> with a <path>.json sidecar.

    Layout is (n_samples, n_channels) int16, channels being every box's 32
    ACQ423 channels concatenated in host order -- i.e. exactly what came off the
    wire, in wire order, with nothing subtracted. The sidecar records the volts
    per count, sample rate, host order and channel names so it can always be
    read back, including by a future version that has corrected the channel map.

    The .bin itself holds counts only, in the order the two boxes were paired
    live. Two optional companions let it be re-aligned offline, without changing
    the .bin (old files load exactly as before):

      <path>.index.csv  one line per write: the row the block starts at, its
                        length, each box's own sample counter at that row, and
                        the loss totals so far. The counters are what show a
                        box's step across a hole that was repaired by trimming
                        the other box, and the cumulative counts show which
                        block it fell in.
      <path>.enc        the encoder counts for the same rows (uint32, little
                        endian, `enc_columns` per row) -- the position, which
                        the .bin has no room for.
    """

    def __init__(self, path, hosts, vpc, fs_hz, nchan_per_box=32, meta=None,
                 cal=None):
        self.path = path
        self.hosts = list(hosts)
        self.nchan = nchan_per_box * len(hosts)
        self.n_samples = 0
        self._index = None          # opened on the first indexed write
        self._enc = None            # likewise
        self._enc_rows = 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, "wb")
        self.meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "dtype": "int16", "order": "C",
                     "n_channels": self.nchan, "hosts": self.hosts,
                     "volts_per_count": list(map(float, vpc)),
                     "fs_hz": list(map(float, fs_hz)),
                     "nchan_per_box": nchan_per_box,
                     "channel_names": _channel_names(len(hosts), nchan_per_box),
                     "note": "raw ACQ423 counts, wire order, nothing subtracted"}
        if cal is not None:
            # Counts alone are not interpretable: turning them into tesla needs
            # the amplifier gain each chip was running AT THE TIME. That lives
            # in a register, not in the data, and it does get changed -- when
            # the two halves of this probe were harmonised from 1500/3000 to
            # 3000, every capture taken before that silently became 1.82x wrong
            # on half the channels if read back with the new settings. So stamp
            # it into the sidecar.
            self.meta["ranges_mt"] = list(map(float, cal.ranges_mt))
            self.meta["volts_per_tesla"] = list(map(float, cal.volts_per_tesla))
            self.meta["vcm_subtracted_by_reader"] = bool(cal.subtract_vcm)
        self.meta.update(meta or {})
        # Write the sidecar NOW, not only on close. Without n_channels the .bin
        # cannot be reshaped at all, so a recorder killed mid-run -- which is
        # the normal end of an overnight log -- used to leave an archive file
        # that nothing could read. Everything except n_samples is already known.
        self._write_sidecar()

    def _write_sidecar(self):
        with open(self.path + ".json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)

    def write(self, ai_by_box, index=None, enc=None):
        """Append per-box (n, 32) int16 blocks, trimmed to the shorter one.

        `index` is LiveSource.block_index for these rows and `enc` its .enc
        counts; both optional, and both are only ever written alongside the
        data, never into it. A block whose counts do not match its length is
        left out of the .enc file, and its index line says so.
        """
        arrs = [np.asarray(a) for a in ai_by_box]
        if not arrs or min(a.shape[0] for a in arrs) == 0:
            return 0
        n = min(a.shape[0] for a in arrs)
        block = np.concatenate([a[:n] for a in arrs], axis=1).astype("<i2")
        self._f.write(block.tobytes())
        # Flushed every block, because "the recording survives a crash" is the
        # point of this route and Python's buffer does not. At the GUI's block
        # size this is one write() syscall per acquisition tick against the
        # megabytes it carries, so it costs nothing measurable.
        self._f.flush()
        if index is not None:
            self._write_index(n, index, enc)
        self.n_samples += n
        return n

    def _write_index(self, n, index, enc):
        """One line of <path>.index.csv, and the block's counts to <path>.enc."""
        enc_row = ""
        if enc is not None and np.shape(enc)[:1] == (n,) and np.ndim(enc) == 2:
            if self._enc is None:
                self._enc = open(self.path + ".enc", "wb")
                self.meta["enc_columns"] = int(np.shape(enc)[1])
                self.meta["enc_file"] = os.path.basename(self.path) + ".enc"
                self._write_sidecar()
            self._enc.write(np.ascontiguousarray(enc, dtype="<u4").tobytes())
            self._enc.flush()
            enc_row = self._enc_rows
            self._enc_rows += n
        if self._index is None:
            self._index = open(self.path + ".index.csv", "w", encoding="utf-8",
                               newline="")
            self._index.write(",".join(
                ["row", "n"] + [f"sam_cnt_{h}" for h in self.hosts]
                + ["gaps", "lost", "dropped_blocks", "enc_row"]) + "\n")
            self.meta["index_file"] = os.path.basename(self.path) + ".index.csv"
            self._write_sidecar()
        fields = [self.n_samples, n, *index["sam_cnt"], index["gaps"],
                  index["lost"], index["dropped_blocks"], enc_row]
        self._index.write(",".join(str(v) for v in fields) + "\n")
        self._index.flush()

    @property
    def size_bytes(self):
        return self.n_samples * self.nchan * 2

    def close(self):
        if self._f and not self._f.closed:
            self._f.close()
        for extra in (self._index, self._enc):
            if extra is not None and not extra.closed:
                extra.close()
        self.meta["n_samples"] = self.n_samples
        self._write_sidecar()
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _channel_names(n_boxes, nchan_per_box):
    names = []
    for b in range(n_boxes):
        for c in range(1, nchan_per_box + 1):
            s, a = ob.channel_label(c, b)
            names.append(f"{s}_{a}")
    return names


def load_raw(path, mmap=False):
    """
    Read a RawRecorder file back. Returns (array (n, nchan), meta dict).

    `mmap` maps the file instead of reading it. The writer is explicitly
    unbounded, so a long log can be larger than RAM; mapping lets you slice a
    few seconds out of an overnight capture without loading the rest.

    A file whose recorder was killed has no n_samples in its sidecar. The row
    count comes from the file size either way, so that costs nothing.
    """
    with open(path + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    nchan = int(meta["n_channels"])
    x = (np.memmap(path, dtype="<i2", mode="r") if mmap
         else np.fromfile(path, dtype="<i2"))
    n = x.size // nchan
    return x[:n * nchan].reshape(n, nchan), meta


def load_raw_index(path):
    """The per-block index of a RawRecorder file as {column: array}, or None.

    None for a file written without one -- every file before the index
    existed, and any recording made through a caller that did not pass it.
    """
    try:
        with open(path + ".index.csv", encoding="utf-8") as f:
            names = f.readline().strip().split(",")
            rows = [ln.strip().split(",") for ln in f if ln.strip()]
    except OSError:
        return None
    out = {}
    for i, name in enumerate(names):
        # enc_row is blank for a block that carried no counts.
        out[name] = np.array([float(r[i]) if r[i] != "" else np.nan
                              for r in rows if len(r) == len(names)])
    return out


def load_raw_enc(path, meta):
    """The encoder counts stored beside a RawRecorder file, (n, columns) or None."""
    if not meta.get("enc_file"):
        return None
    cols = int(meta["enc_columns"])
    x = np.fromfile(os.path.join(os.path.dirname(path), meta["enc_file"]),
                    dtype="<u4")
    return x[:x.size // cols * cols].reshape(-1, cols)


def raw_to_boxes(x, meta):
    """(n, nchan) from load_raw -> list of per-box (n, 32) arrays."""
    k = meta["nchan_per_box"]
    return [x[:, i * k:(i + 1) * k] for i in range(len(meta["hosts"]))]


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------

def _write_rows(path, keys, rows):
    """
    Write `rows` as CSV with `keys` as the header.

    Through csv.DictWriter, not a manual ",".join, because these tables carry
    free text. health_verdict() produces notes like "Bx railed, By stuck" and
    "VCM noise 9.0 counts -- analogue pickup, not the sensor", and an unquoted
    comma in those splits the row into one more field than the header has. The
    misalignment lands precisely on the faulty sensors -- the rows anyone
    reading the file cares most about.
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore",
                           restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in keys})
    return path


def write_health_csv(path, rows):
    """Per-channel diagnostics from octobee_calibration.channel_health()."""
    keys = ["host", "box", "ch", "sensor", "axis", "mean_counts", "mean_v",
            "std_counts", "std_uv", "min", "max", "p2p_counts", "railed",
            "stuck", "out_of_range", "bad", "is_vcm"]
    return _write_rows(path, keys, rows)


def write_sensor_csv(path, table):
    """
    Per-sensor calibration summary. `table` is a list of dicts; whatever keys
    the first row has become the columns.
    """
    if not table:
        return None
    return _write_rows(path, list(table[0].keys()), table)


def write_report_json(path, payload):
    def _plain(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        raise TypeError(type(o))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_plain)
    return path


def sensor_table(cal, b_mt, health_rows, temps=None, peaks=None, geom=None):
    """
    Build the per-sensor summary used by both the GUI table and the CSV export.
    b_mt is (n, 16, 3) calibrated millitesla.
    """
    verdict = ocal.health_verdict(health_rows)
    noise = ocal.noise_mt(b_mt)
    level = np.median(b_mt, axis=0)
    mag = np.linalg.norm(level, axis=-1)
    vcm_v = {}
    vcm_noise = {}
    for r in health_rows:
        if r["is_vcm"]:
            vcm_v[r["sensor_id"]] = r["mean_v"]
            vcm_noise[r["sensor_id"]] = r["std_counts"]

    table = []
    for s in range(N_SENSORS):
        sid = s + 1
        state, note = verdict.get(sid, ("unknown", ""))
        row = {"sensor": f"S{sid}",
               "state": state,
               "face": geom.face(sid) if geom else None,
               "slot": geom.slot(sid) if geom else None,
               "range_mt": cal.ranges_mt[s],
               "vcm_v": round(vcm_v.get(sid, float("nan")), 5),
               "vcm_noise_counts": round(vcm_noise.get(sid, float("nan")), 2),
               # Uncalibrated and per-chip-offset: see octobee.temp_c. NaN
               # means the box carried no SPAD temperature at all, and has to
               # stay empty rather than become the word "nan" in a CSV.
               "temp_c": (None if temps is None or not np.isfinite(temps[s])
                          else round(float(temps[s]), 1)),
               "Bx_mT": round(float(level[s, 0]), 4),
               "By_mT": round(float(level[s, 1]), 4),
               "Bz_mT": round(float(level[s, 2]), 4),
               "absB_mT": round(float(mag[s]), 4),
               "noise_uT_rms": round(float(noise[s].max() * 1e3), 1),
               "zero_Bx_mT": round(float(cal.zero_mt[s, 0]), 4),
               "zero_By_mT": round(float(cal.zero_mt[s, 1]), 4),
               "zero_Bz_mT": round(float(cal.zero_mt[s, 2]), 4),
               "gain_trim": round(float(cal.gain_corr[s].mean()), 4),
               "note": note}
        if peaks is not None:
            row["peak_absB_mT"] = round(float(peaks["mag_pk"][s]), 4)
        table.append(row)
    return table


def main(argv=None):
    p = argparse.ArgumentParser(description="export a saved capture")
    p.add_argument("capture", help=".npz from octobee/acq/carrier.py capture")
    p.add_argument("-o", "--out", default=None, help="output CSV")
    p.add_argument("--rate", type=float, default=1000.0, help="output rate Hz")
    p.add_argument("--tube-frame", action="store_true")
    p.add_argument("--range", type=float, default=20.0,
                   choices=sorted(ob.RANGE_TO_VPT))
    a = p.parse_args(argv)

    cap = ocal.load_capture(a.capture)
    cal = ocal.Calibration(ranges_mt=np.full(N_SENSORS, a.range))
    geom = pg.Geometry.load_or_default()
    b = cal.convert(cap["ai"], cap["vpc"],
                    offset_by_box=cap.get("volt_offset"))
    dec = max(1, int(round(cap["fs_hz"][0] / a.rate)))
    b = ocal.decimate(b, dec)
    out = a.out or default_name("export", "csv")
    with CsvRecorder(out, cap["fs_hz"][0] / dec, cal, geom, a.tube_frame,
                     meta={"source": a.capture},
                     samples_per_row=dec) as rec:
        rec.write(b)
    print(f"wrote {out}: {b.shape[0]} rows at {cap['fs_hz'][0]/dec:g} Hz, "
          f"{os.path.getsize(out)/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
