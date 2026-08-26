"""Decoding a frame, including the buffer lengths a real stream produces."""

import numpy as np

from octobee.acq import carrier as ob
from tests.helpers import (
    check,
)


def _layout(ssb):
    return ob.Layout(fs_hz=200000.0, adc_range="+/-10V", ssb=ssb,
                     nchan_ai=32, spad_lw=7)


def test_decode_round_trip():
    """Every field must come back exactly, on both carriers' frame shapes.

    694 sends 96 B frames with one site-2 word; 695 sends 104 B with three
    encoder words. A mis-sliced frame is not a crash, it is wrong numbers that
    still look like data.
    """
    print("\nframe decode")
    for name, ssb in (("acq1001_694", 96), ("acq1001_695", 104)):
        lay = _layout(ssb)
        n = 37
        frame = np.zeros((n, lay.n_i16), dtype="<i2")
        rng = np.random.default_rng(19)
        ai = rng.integers(-30000, 30000, size=(n, lay.nchan_ai)).astype("<i2")
        frame[:, :lay.nchan_ai] = ai
        u32 = frame.view("<u4").reshape(n, lay.n_lw)
        enc = rng.integers(0, 2**31, size=(n, lay.spad0_lw - lay.site2_lw),
                           dtype=np.uint32)
        u32[:, lay.site2_lw:lay.spad0_lw] = enc
        sam = np.arange(1000, 1000 + n, dtype=np.uint32)
        usec = np.arange(n, dtype=np.uint32) * 5
        u32[:, lay.spad0_lw] = sam
        u32[:, lay.spad0_lw + 1] = usec
        temps = np.array([100, 200, 300, 400, 500, 600, 700, 800],
                         dtype=np.uint32)
        u32[0, lay.spad0_lw + 2:lay.spad0_lw + 6] = [
            temps[2 * k] | (temps[2 * k + 1] << 16) for k in range(4)]
        u32[0, lay.spad0_lw + 6] = 0xDEADBEEF

        got = ob.decode(frame.tobytes(), lay)
        check(f"{name}: analogue channels round-trip",
              np.array_equal(got["ai"], ai))
        check(f"{name}: sample counter round-trips",
              np.array_equal(got["sam_cnt"], sam))
        check(f"{name}: encoder words round-trip", np.array_equal(got["enc"], enc))
        check(f"{name}: temperatures unpack in order",
              np.array_equal(got["temp_raw"], temps))
        check(f"{name}: PWR_GOOD round-trips", got["pwr_good"] == 0xDEADBEEF)


def test_decode_survives_any_buffer_length():
    """A truncated stream must lose the tail, not the whole capture.

    capture_one accumulates whatever recv() returns and decodes the total, and
    TCP promises nothing about alignment. An odd byte count -- a box rebooting
    mid-word, a link dropping -- used to raise "buffer size must be a multiple
    of element size", throwing away every sample that HAD been collected and
    telling the operator nothing about why.
    """
    print("\ndecode with a ragged buffer")
    lay = _layout(96)
    n = 10
    frame = np.zeros((n, lay.n_i16), dtype="<i2")
    ai = np.arange(n * lay.nchan_ai, dtype="<i2").reshape(n, lay.nchan_ai)
    frame[:, :lay.nchan_ai] = ai
    whole = frame.tobytes()

    for label, buf, want in (
            ("exactly N whole frames", whole, n),
            ("N frames + an even partial frame", whole + b"\x00" * 50, n),
            ("N frames + one stray byte", whole + b"\x01", n),
            ("N frames + three stray bytes", whole + b"\x01\x02\x03", n),
            ("half a frame, even length", b"\x00" * 40, None),
            ("a single stray byte", b"\x01", None),
            ("nothing at all", b"", None)):
        got = ob.decode(buf, lay)
        if want is None:
            check(f"{label} -> None", got is None,
                  "expected None" if got is None else f"got {got['ai'].shape}")
        else:
            check(f"{label} -> {want} whole samples",
                  got is not None and got["ai"].shape[0] == want
                  and np.array_equal(got["ai"], ai),
                  "decoded nothing" if got is None else str(got["ai"].shape))
