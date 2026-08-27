"""A live recording must carry the position the field was measured at.

The Record button wrote calibrated millitesla and a time column and nothing
else, so a recording made while the head was travelling said what the field
was and not where. The counts were already arriving in the same frames -- the
sweep has used them for a while -- they simply never reached the CSV.

What these pin is the two ways that can go quietly wrong: counts landing
against the wrong rows, and a relative position column being read as an
absolute one.
"""

import argparse
import os

import numpy as np

from octobee.calib import convert as ocal
from octobee.gui import session as osess
from octobee.motion import encoder as oenc
from octobee import record as orec
from tests.helpers import (
    check,
)

# x forward, y wired backwards, z uncalibrated -- all three states at once.
ENCODERS = oenc.EncoderMap({"x": {"column": 0, "counts_per_mm": 2000.0},
                            "y": {"column": 1, "counts_per_mm": -2000.0}})
COLUMNS = 3


def _read(path):
    """(header text, column names, rows as dicts of str)."""
    head, cols, rows = [], None, []
    for ln in open(path, encoding="utf-8"):
        if ln.startswith("#"):
            head.append(ln)
        elif cols is None:
            cols = ln.strip().split(",")
        else:
            rows.append(dict(zip(cols, ln.strip().split(","))))
    return "".join(head), cols, rows


def _field(n):
    return np.zeros((n, 16, 3), np.float32)


def test_counts_and_millimetres_reach_the_csv(workdir):
    """The travel, on the same rows as the field it was measured with."""
    print("\nCSV encoder columns")
    path = f"{workdir}/enc.csv"
    rec = orec.CsvRecorder(path, 100.0, ocal.Calibration(),
                           encoders=ENCODERS, enc_columns=COLUMNS,
                           enc_datum={"x": (10_000.0, 5.0)})
    # 10 mm of x at 2000 counts/mm, 1 mm of y backwards, z counting on its own.
    counts = np.column_stack([10_000.0 + np.arange(5) * 5_000.0,
                              50_000.0 - np.arange(5) * 500.0,
                              np.full(5, 1_234_567_890.0)])
    rec.write(_field(5), counts)
    rec.close()
    text, cols, rows = _read(path)

    check("every stream column is written, calibrated or not",
          [c for c in cols if c.endswith("_counts")]
          == ["enc0_counts", "enc1_counts", "enc2_counts"],
          str([c for c in cols if c.endswith("_counts")]))
    check("an axis with a datum gets an absolute millimetre column",
          "X_mm" in cols and "X_rel_mm" not in cols, str(cols[-4:]))
    check("an axis without one is named as relative, not as a position",
          "Y_rel_mm" in cols and "Y_mm" not in cols, str(cols[-4:]))
    check("an uncalibrated column gets no millimetres invented for it",
          "Z_mm" not in cols and "Z_rel_mm" not in cols, str(cols[-4:]))

    check("the datum puts x where the controller said it was",
          float(rows[0]["X_mm"]) == 5.0, rows[0]["X_mm"])
    check("and the counts carry it from there",
          float(rows[4]["X_mm"]) == 15.0,
          f'{rows[4]["X_mm"]} after 10 mm of counts')
    check("a negative scale runs the axis the way it is wired",
          float(rows[4]["Y_rel_mm"]) == 1.0, rows[4]["Y_rel_mm"])
    check("a relative axis starts from its own first row",
          float(rows[0]["Y_rel_mm"]) == 0.0, rows[0]["Y_rel_mm"])

    # %.6g would print this as 1.23457e+09 -- a 500 count error, a quarter of
    # a millimetre, produced by the format rather than by the hardware.
    check("counts are not quantised by the number format",
          float(rows[0]["enc2_counts"]) == 1_234_567_890.0,
          rows[0]["enc2_counts"])
    check("the header says what the millimetres are measured from",
          "encoder_datum:" in text and "10,000 counts = 5.0000 mm" in text)


def test_counts_that_do_not_line_up_are_left_out(workdir):
    """Better a blank position column than one shifted by a few rows.

    A shifted one is undetectable in the file: it reads as a real position
    and puts every field value a little way from where it was taken.
    """
    print("\nCSV encoder alignment")
    path = f"{workdir}/gap.csv"
    rec = orec.CsvRecorder(path, 100.0, ocal.Calibration(),
                           encoders=ENCODERS, enc_columns=COLUMNS)
    good = np.column_stack([np.arange(3) * 2_000.0, np.zeros(3),
                            np.zeros(3)])
    rec.write(_field(3), good)
    rec.write(_field(3), None)                    # a source with no encoders
    rec.write(_field(3), good[:2])                # short by a row
    rec.write(_field(3), good + 100_000.0)
    rec.close()
    _, _, rows = _read(path)

    check("the rows are all there regardless -- the field is good",
          len(rows) == 12, str(len(rows)))
    check("rows with no counts have empty positions, not guessed ones",
          all(r["X_rel_mm"] == "nan" for r in rows[3:9]),
          str([r["X_rel_mm"] for r in rows[3:9]]))
    check("a block short by one row is refused whole, not aligned to the top",
          all(r["enc0_counts"] == "nan" for r in rows[6:9]),
          str([r["enc0_counts"] for r in rows[6:9]]))
    check("counting resumes where the counter actually is",
          float(rows[9]["X_rel_mm"]) == 50.0, rows[9]["X_rel_mm"])
    check("and the file knows how many rows went out unpaired",
          rec.n_unpaired == 6, str(rec.n_unpaired))


def test_a_stream_with_no_encoders_is_unchanged(workdir):
    """acq1001_694 alone, a replay, the demo source: no columns, no header."""
    print("\nCSV without encoders")
    path = f"{workdir}/plain.csv"
    rec = orec.CsvRecorder(path, 100.0, ocal.Calibration())
    rec.write(_field(2))
    rec.close()
    text, cols, rows = _read(path)
    check("no encoder columns appear", not any("enc" in c or c.endswith("_mm")
                                               for c in cols))
    check("and the header does not claim any", "encoder_columns" not in text)
    check("the field columns are exactly what they always were",
          len(cols) == 1 + 16 * 4 and len(rows) == 2, str(len(cols)))


def test_rows_stay_in_step_across_a_stream_of_blocks(workdir):
    """The two decimators must produce the same rows, block after block.

    The field and the counts are decimated separately -- one is millitesla by
    then and the other must not be -- so "same factor, same samples in" is the
    whole of the guarantee that row n of the CSV holds the counts latched with
    row n's field.
    """
    print("\nCSV encoder decimation")
    factor = 40
    rng = np.random.default_rng(3)
    path = f"{workdir}/stream.csv"
    rec = orec.CsvRecorder(path, 500.0, ocal.Calibration(),
                           encoders=ENCODERS, enc_columns=COLUMNS,
                           samples_per_row=factor)
    field_dec, count_dec = ocal.Decimator(factor), ocal.Decimator(factor)

    n_in = 0
    for _ in range(200):
        n = int(rng.integers(700, 1300))          # blocks are not multiples
        # x advances one count per sample, so the row means are known.
        c = np.column_stack([n_in + np.arange(n, dtype=float),
                             np.zeros(n), np.zeros(n)])
        n_in += n
        bd = field_dec.push(_field(n))
        ed = count_dec.push(c)
        check_len = bd.shape[0] == ed.shape[0]
        if not check_len:
            check("the two decimators produced the same number of rows", False,
                  f"{bd.shape[0]} field vs {ed.shape[0]} count rows")
            break
        if bd.shape[0]:
            rec.write(bd, ed)
    rec.close()
    _, _, rows = _read(path)

    check("no row was written without its counts",
          rec.n_unpaired == 0, str(rec.n_unpaired))
    check("every row that came out of the stream is in the file",
          len(rows) == rec.n_rows == n_in // factor, str(len(rows)))
    # Row k averages samples [k*factor, (k+1)*factor), so its mean count is
    # k*factor + (factor-1)/2. If a block ever slipped, this walks off.
    expected = np.arange(len(rows)) * factor + (factor - 1) / 2.0
    got = np.array([float(r["enc0_counts"]) for r in rows])
    check("and every one carries the counts latched with its own samples",
          np.allclose(got, expected),
          f"first mismatch at row {int(np.argmax(got != expected))}"
          if not np.allclose(got, expected) else "")


def test_a_fitted_scale_survives_a_restart(workdir, monkeypatch):
    """A counts/mm measured yesterday must still be there this morning.

    EncoderMap.save() has always written into stages.json and nothing has ever
    read it back, so every session started with no scale and every position
    quietly fell back to the controllers -- on a rig whose encoders had been
    calibrated, with nothing on screen to say the calibration was not in use.
    Without this the recorded position columns would be counts and never
    millimetres from the second run onwards.
    """
    print("\nencoder scale persistence")
    monkeypatch.setattr(osess.paths, "config", lambda name: f"{workdir}/{name}")
    monkeypatch.setattr(oenc.paths, "config", lambda name: f"{workdir}/{name}")

    path = ENCODERS.save(note="measured by the test")
    check("the scale is written beside the rest of the axis facts",
          os.path.basename(path) == osess.AXIS_CONFIG, path)

    back = oenc.EncoderMap.load()
    check("and it reads back as the same thing",
          back.to_dict() == ENCODERS.to_dict(), back.describe())

    args = argparse.Namespace(
        uut=None, demo=True, replay=None, no_connect=True,
        calibration=f"{workdir}/cal.json", geometry=f"{workdir}/geom.json",
        machine=f"{workdir}/machine.json", out_dir=f"{workdir}/captures")
    s = osess.Session(args)
    check("a fresh session starts with the scale already loaded",
          s.encoders.to_dict() == ENCODERS.to_dict(),
          s.encoders.describe())
    check("and does not report it as a configuration error",
          not any("encoder" in m for m in s.config_errors),
          "; ".join(s.config_errors))


def test_an_unreadable_stages_file_does_not_stop_the_session(workdir,
                                                             monkeypatch):
    """Half a config is a reason to say so, not a reason not to start."""
    print("\nencoder scale, unreadable")
    monkeypatch.setattr(osess.paths, "config", lambda name: f"{workdir}/{name}")
    monkeypatch.setattr(oenc.paths, "config", lambda name: f"{workdir}/{name}")
    with open(f"{workdir}/{osess.AXIS_CONFIG}", "w", encoding="utf-8") as f:
        f.write("{ this is not json")

    args = argparse.Namespace(
        uut=None, demo=True, replay=None, no_connect=True,
        calibration=f"{workdir}/cal.json", geometry=f"{workdir}/geom.json",
        machine=f"{workdir}/machine.json", out_dir=f"{workdir}/captures")
    s = osess.Session(args)
    check("the session still opens", not s.encoders)
    check("and says why the positions will come from the controllers",
          any("encoder" in m for m in s.config_errors),
          "; ".join(s.config_errors))
