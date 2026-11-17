"""What gets written to disk, and what survives a kill."""

import csv
import os
import tempfile

import numpy as np

from octobee import record as orec
from octobee.calib import convert as ocal
from tests.helpers import (
    check,
)



def test_csv_quoting():
    """Free text in a table must not split the row."""
    print("\nCSV quoting")
    with tempfile.TemporaryDirectory() as d:
        # health_verdict() really does produce notes like this one.
        table = [{"sensor": "S1", "state": "fault",
                  "note": "Bx railed, By stuck"},
                 {"sensor": "S2", "state": "noisy",
                  "note": 'VCM noise 9.0 counts -- pickup, not the sensor'},
                 {"sensor": "S3", "state": "ok", "note": None}]
        p = orec.write_sensor_csv(os.path.join(d, "s.csv"), table)
        with open(p, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        check("every row has exactly as many fields as the header",
              all(len(r) == len(rows[0]) for r in rows),
              f"header {len(rows[0])}, rows {[len(r) for r in rows[1:]]}")
        check("the comma survives inside the field, not as a new column",
              rows[1][2] == "Bx railed, By stuck", rows[1][2])
        check("a None renders as empty rather than the word None",
              rows[3][2] == "")


def test_raw_survives_a_kill():
    """A recorder that never gets close()d must still leave a readable file."""
    print("\nraw archive durability")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "raw.bin")
        rec = orec.RawRecorder(p, ["a", "b"], [3.0517578125e-4] * 2,
                               [200000.0] * 2)
        check("the sidecar is written at open, not at close",
              os.path.exists(p + ".json"))
        rec.write([np.full((32, 32), 7, dtype=np.int16)] * 2)
        # Deliberately no close(): this is the overnight-log-gets-killed case.
        x, meta = orec.load_raw(p)
        check("an unclosed recording still reads back",
              x.shape == (32, 64) and int(x[0, 0]) == 7, str(x.shape))
        check("and the sidecar already knows the channel layout",
              meta["n_channels"] == 64
              and len(meta["channel_names"]) == 64)
        rec.close()
        x2, meta2 = orec.load_raw(p)
        check("closing adds the final sample count",
              meta2["n_samples"] == 32 and x2.shape == x.shape)


def test_default_names_do_not_collide(monkeypatch):
    """Two recordings in one second must not share a file."""
    print("'+chr(92)+'nunique default names")
    monkeypatch.setattr(orec, "_stamp", lambda: "20260101_000000")  # one second
    with tempfile.TemporaryDirectory() as d:
        names = []
        for _ in range(3):
            p = orec.default_name("octobee", "csv", d)
            with open(p, "w") as f:               # what a recorder does
                f.write("x")
            names.append(p)
        check("three names in the same second are three different paths",
              len(set(names)) == 3, str([os.path.basename(n) for n in names]))
        check("the first keeps the plain name",
              os.path.basename(names[0]) == "octobee_20260101_000000.csv")


def test_csv_rows_reach_disk_without_a_close():
    """A killed run must not lose the rows it had written, or cut one in half."""
    print("\nCSV durability")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "live.csv")
        rec = orec.CsvRecorder(p, 100.0, ocal.Calibration())
        rec.write(np.zeros((5, 16, 3), np.float32))
        # Deliberately not closed: read it the way a post-mortem would.
        with open(p, encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if not ln.startswith("#")]
        ncol = len(lines[0].split(","))
        check("all five rows are already on disk, each whole",
              len(lines) == 6 and all(len(ln.split(",")) == ncol for ln in lines),
              f"{len(lines)} lines")
        rec.close()


def test_csv_header_is_honest_about_the_boxes():
    """The encoder counts share samples with ONE box, and a loss breaks t_s."""
    print("\nCSV header claims")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "enc.csv")
        rec = orec.CsvRecorder(p, 100.0, ocal.Calibration(), enc_columns=3)
        rec.note_stream({"gaps": 4, "lost": 40, "dropped blocks": 1})
        rec.write(np.zeros((2, 16, 3), np.float32), np.zeros((2, 3)),
                  stream={"gaps": 6, "lost": 4040, "dropped blocks": 3})
        rec.close()
        text = open(p, encoding="utf-8").read()
        timing = next(ln for ln in text.splitlines()
                      if ln.startswith("# encoder_timing"))
        check("encoder_timing names the box the counts share samples with",
              "S9-S16" in timing and "acq1001_695" in timing, timing)
        check("and says S1-S8 are on a different, merely paired, carrier",
              "S1-S8" in timing and "paired by sample order" in timing, timing)
        check("the timebase no longer claims to be real elapsed time",
              "t_s is real elapsed time" not in text
              and "free-runs on its own clock" in text)
        foot = [ln for ln in text.splitlines() if ln.startswith("# stream_gaps")]
        check("the footer reports what was lost while the file was open",
              len(foot) == 1 and "2 break(s)" in foot[0]
              and "4000 sample(s)" in foot[0] and "2 block(s)" in foot[0],
              str(foot))


def test_raw_index_lets_an_archive_be_realigned():
    """The .bin stays as it was; a companion says where each box's counter was."""
    print("\nraw archive index")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "raw.bin")
        rec = orec.RawRecorder(p, ["a", "b"], [3.0517578125e-4] * 2,
                               [200000.0] * 2)
        blk = [np.full((10, 32), 7, dtype=np.int16)] * 2
        enc = np.arange(30, dtype=np.uint32).reshape(10, 3)
        rec.write(blk, index={"sam_cnt": [100, 9000], "gaps": 0, "lost": 0,
                              "dropped_blocks": 0}, enc=enc)
        # the second block follows a repaired hole: box a's counter jumped
        rec.write(blk, index={"sam_cnt": [150, 9010], "gaps": 1, "lost": 40,
                              "dropped_blocks": 1}, enc=enc)
        rec.write(blk)          # a caller that passes nothing still works
        rec.close()
        x, meta = orec.load_raw(p)
        check("the .bin format is unchanged", x.shape == (30, 64)
              and meta["n_samples"] == 30, str(x.shape))
        idx = orec.load_raw_index(p)
        check("the index has a line per indexed write, at the right rows",
              idx is not None and list(idx["row"]) == [0, 10]
              and list(idx["n"]) == [10, 10], str(idx))
        check("with each box's own counter, so the jump is visible",
              list(idx["sam_cnt_a"]) == [100, 150]
              and list(idx["sam_cnt_b"]) == [9000, 9010])
        check("and the loss totals so far",
              list(idx["gaps"]) == [0, 1] and list(idx["lost"]) == [0, 40]
              and list(idx["dropped_blocks"]) == [0, 1])
        e = orec.load_raw_enc(p, meta)
        check("the encoder counts are archived row for row",
              e is not None and e.shape == (20, 3)
              and np.array_equal(e[:10], enc) and np.array_equal(e[10:], enc))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "old.bin")
        rec = orec.RawRecorder(p, ["a"], [1.0], [1.0])
        rec.write([np.zeros((4, 32), np.int16)])
        rec.close()
        check("a recording made without any of it has no index and still loads",
              orec.load_raw_index(p) is None
              and orec.load_raw_enc(p, orec.load_raw(p)[1]) is None
              and orec.load_raw(p)[0].shape == (4, 32))
