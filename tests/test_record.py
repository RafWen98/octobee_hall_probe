"""What gets written to disk, and what survives a kill."""

import csv
import os
import tempfile

import numpy as np

from octobee import record as orec
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
