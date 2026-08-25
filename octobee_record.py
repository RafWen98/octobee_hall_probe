#!/usr/bin/env python3
"""
octobee_record.py -- getting data out of the probe and onto disk.

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

import octobee as ob
import octobee_calibration as ocal
import probe_geometry as pg

AXES = ocal.AXES
N_SENSORS = ocal.N_SENSORS


def _stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def default_name(prefix, ext, directory="captures"):
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"{prefix}_{_stamp()}.{ext}")


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
    """

    def __init__(self, path, fs_out, cal, geom=None, tube_frame=False,
                 include_mag=True, meta=None):
        self.path = path
        self.fs_out = float(fs_out)
        self.cal = cal
        self.geom = geom
        self.tube_frame = bool(tube_frame and geom is not None)
        self.include_mag = include_mag
        self.n_rows = 0
        self.t = 0.0
        self.started = time.time()
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
        self._f.write(",".join(cols) + "\n")
        self._cols = cols

    def write(self, b_mt):
        """Append (n, 16, 3) millitesla already decimated to fs_out."""
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
        rows = np.column_stack([t, block])
        np.savetxt(self._f, rows, delimiter=",", fmt="%.6g")
        self.n_rows += n
        return n

    @property
    def size_bytes(self):
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def close(self):
        if self._f and not self._f.closed:
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
    """

    def __init__(self, path, hosts, vpc, fs_hz, nchan_per_box=32, meta=None,
                 cal=None):
        self.path = path
        self.hosts = list(hosts)
        self.nchan = nchan_per_box * len(hosts)
        self.n_samples = 0
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

    def write(self, ai_by_box):
        """Append per-box (n, 32) int16 blocks, trimmed to the shorter one."""
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
        self.n_samples += n
        return n

    @property
    def size_bytes(self):
        return self.n_samples * self.nchan * 2

    def close(self):
        if self._f and not self._f.closed:
            self._f.close()
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
    p.add_argument("capture", help=".npz from octobee.py capture")
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
                     meta={"source": a.capture}) as rec:
        rec.write(b)
    print(f"wrote {out}: {b.shape[0]} rows at {cap['fs_hz'][0]/dec:g} Hz, "
          f"{os.path.getsize(out)/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
