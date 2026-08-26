"""Decoding a frame, including the buffer lengths a real stream produces."""

import numpy as np

import queue

from octobee.acq import carrier as ob
from octobee.gui.sources import LiveSource
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


def test_which_carrier_has_the_encoders():
    """A longword between the channels and the scratchpad is not an encoder.

    This is the shape the rig really has, read off the boxes: BOTH carriers
    aggregate a quadrature module in site 2, and only acq1001_695 has its
    counters switched on. acq1001_694 therefore emits a site-2 longword that
    never changes.

    So "the carrier with some encoder longwords" identifies the 694 -- it is
    first in DEFAULT_UUTS -- and a swept map would have taken the probe's
    position from a constant. Nothing downstream could have noticed: a column
    that does not move looks exactly like an axis that did not move.
    """
    print("\nwhich carrier the positions come from")

    # acq1001_694: sites 1,2 -- one quadrature module, phaseA_en off.
    lay694 = ob.Layout(ssb=96, nchan_ai=32, spad_lw=7, fs_hz=200000.0,
                       adc_range="+/-10V", sites=(1, 2), counting_sites=())
    # acq1001_695: sites 1,2,5,6 -- three modules, all counting.
    lay695 = ob.Layout(ssb=104, nchan_ai=32, spad_lw=7, fs_hz=200000.0,
                       adc_range="+/-10V", sites=(1, 2, 5, 6),
                       counting_sites=(2, 5, 6))

    check("both carriers carry a site-2 longword, so the byte count cannot "
          "tell them apart",
          lay694.n_site2_lw == 1 and lay695.n_site2_lw == 3,
          f"694 {lay694.n_site2_lw} LW, 695 {lay695.n_site2_lw} LW")
    check("but only the one whose counters are running claims any columns",
          lay694.encoder_columns == {}
          and lay695.encoder_columns == {2: 0, 5: 1, 6: 2},
          f"694 {lay694.encoder_columns}, 695 {lay695.encoder_columns}")
    check("and the columns are in site order, which is the order they arrive",
          list(lay695.encoder_columns.values()) == [0, 1, 2],
          str(lay695.encoder_columns))

    # The choice the live source makes, in DEFAULT_UUTS order -- 694 first,
    # which is the order that made this wrong.
    class _Stub:
        """Just enough of a Streamer for LiveSource to construct."""

        def __init__(self):
            self.q = queue.Queue()
            self.error, self.dropped, self.bytes_read = None, 0, 0

        def get_all(self):
            return []

    hosts = ["acq1001_694", "acq1001_695"]
    src = LiveSource.__new__(LiveSource)
    src.hosts, src.layouts = hosts, [lay694, lay695]
    LiveSource.__init__(src, hosts, [lay694, lay695], [_Stub(), _Stub()])
    check("the live source takes its positions from the carrier that is "
          "actually counting",
          src.enc_host == "acq1001_695", str(src.enc_host))
    check("and knows which sites they are, for the operator to check against "
          "rc.user",
          src.enc_sites == (2, 5, 6) and src.enc_columns == 3,
          f"{src.enc_sites}, {src.enc_columns} columns")

    # And a rig with nothing counting anywhere must say so rather than pick one.
    none = LiveSource.__new__(LiveSource)
    LiveSource.__init__(none, hosts, [lay694, lay694], [_Stub(), _Stub()])
    check("with no counters running anywhere, no host is chosen",
          none.enc_host is None and none.enc_columns == 0,
          str(none.enc_host))


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
