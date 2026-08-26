"""A recording's clock must be real time, not a count of rows that survived.

The bug these pin: on_tick decimated each arriving block on its own, and
decimate() keeps whole groups and drops the remainder. Blocks do not arrive in
multiples of the factor, so the tail of every block was measured, converted,
and thrown away -- 1.96% of a recording at the default 500 Hz, 9.9% at the
100 Hz you would choose for an overnight run. The CSV then stamped the rows it
did get at exactly 1/fs_out apart, so the loss did not show as a gap. It made
the time column run slow, by 47 minutes over an eight-hour night.
"""

import numpy as np

from octobee.calib import convert as ocal
from octobee.calib import geometry as pgeom
from octobee import record as orec
from tests.helpers import (
    check,
)


def _blocks(rng, n_ticks, fs_in, tick_hz, jitter=40):
    """Sample counts as they really arrive: whatever landed in one tick."""
    for _ in range(n_ticks):
        yield max(1, int(rng.normal(fs_in / tick_hz, jitter)))


def test_decimator_keeps_every_sample():
    """Nothing measured may be silently discarded."""
    print("\nstream decimation")
    rng = np.random.default_rng(5)

    for factor in (10, 40, 200):
        d = ocal.Decimator(factor)
        for n in _blocks(rng, 400, 20000.0, 20.0):
            d.push(np.zeros((n, 16, 3), np.float32))
        accounted = d.rows_out * factor + d.pending
        check(f"decim {factor}: every sample is in a row or in the carry",
              accounted == d.samples_in,
              f"{d.samples_in - accounted} lost of {d.samples_in}")
        check(f"decim {factor}: carry is smaller than one row",
              d.pending < factor, f"{d.pending} held")

    # the values must match doing it in one pass over the whole stream
    rng = np.random.default_rng(9)
    whole = np.arange(4000, dtype=np.float32).reshape(-1, 1, 1)
    d = ocal.Decimator(7)
    out, i = [], 0
    for n in (13, 400, 1, 999, 87, 2500):
        out.append(d.push(whole[i:i + n]))
        i += n
    streamed = np.concatenate([o for o in out if len(o)], axis=0)
    at_once = ocal.decimate(whole[:i], 7)
    check("block by block gives the same numbers as one pass",
          np.allclose(streamed, at_once[:len(streamed)]),
          f"{streamed.shape} vs {at_once.shape}")

    d = ocal.Decimator(1)
    check("factor 1 passes blocks straight through",
          d.push(np.zeros((5, 16, 3), np.float32)).shape[0] == 5)


def test_csv_timebase_is_real_elapsed_time(workdir):
    """An hour of recording must say it was an hour."""
    print("\nCSV timebase")
    fs_in, fs_out, tick_hz = 20000.0, 100.0, 20.0     # 100 Hz: the worst case
    factor = int(round(fs_in / fs_out))
    rng = np.random.default_rng(5)

    cal = ocal.Calibration()
    geom = pgeom.Geometry.load_or_default()
    path = f"{workdir}/timebase.csv"
    rec = orec.CsvRecorder(path, fs_out, cal, geom, samples_per_row=factor)
    dec = ocal.Decimator(factor)

    samples_in = 0
    for n in _blocks(rng, int(10 * 60 * tick_hz), fs_in, tick_hz):
        samples_in += n
        bd = dec.push(np.zeros((n, 16, 3), np.float32))
        if bd.shape[0]:
            rec.write(bd)
    rec.close()

    true_s = samples_in / fs_in
    csv_s = rec.n_rows / fs_out
    drift = abs(csv_s - true_s)
    # one row is the most the clock can be out: the carry is by definition
    # less than a full row, and it is written as soon as it completes.
    check("the CSV clock matches real elapsed time",
          drift <= 1.0 / fs_out + 1e-9,
          f"{true_s:.2f} s real, {csv_s:.2f} s in the file, "
          f"drift {drift*1000:.1f} ms over {true_s/60:.0f} min")
    check("no sample was dropped",
          rec.n_rows * factor + dec.pending == samples_in,
          f"{samples_in - rec.n_rows * factor - dec.pending} lost")


def test_csv_header_states_its_timebase(workdir):
    """A file must say what a row is, rather than leaving it to be inferred.

    Files written before this cannot promise a real timebase, and the only
    honest way to tell them apart later is a field they do not have.
    """
    print("\nCSV timebase header")
    cal = ocal.Calibration()
    path = f"{workdir}/header.csv"
    rec = orec.CsvRecorder(path, 500.0, cal, samples_per_row=40)
    rec.write(np.zeros((3, 16, 3), np.float32))
    rec.close()
    head = [ln for ln in open(path, encoding="utf-8") if ln.startswith("#")]
    text = "".join(head)
    check("the header names the samples per row", "samples_per_row: 40" in text,
          text[:200])
    check("the header states the timebase is contiguous",
          "timebase: contiguous" in text)
