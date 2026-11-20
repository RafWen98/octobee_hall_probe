"""Decoding a frame, including the buffer lengths a real stream produces."""

import numpy as np

import os
import queue
import socket
import threading
import time

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


def test_find_gaps_sees_the_step_between_blocks():
    """check_continuity is per block; a shed block left no trace."""
    print("\ncounter gaps across blocks")
    a = np.arange(100, 200, dtype=np.uint32)
    b = np.arange(300, 400, dtype=np.uint32)          # block 200..299 missing
    check("within a block alone the loss is invisible (the old behaviour)",
          ob.check_continuity(b) == (0, 0))
    check("with the previous block's last counter it is found at row 0",
          ob.find_gaps(b, prev=int(a[-1])) == [(0, 100)],
          str(ob.find_gaps(b, prev=int(a[-1]))))
    inside = np.concatenate([a[:50], a[60:]])
    check("a loss inside a block is placed at the row after it",
          ob.find_gaps(inside) == [(50, 10)], str(ob.find_gaps(inside)))
    wrap = (np.arange(10, dtype=np.int64) + 2 ** 32 - 4).astype(np.uint32)
    check("a 32-bit wrap is a step of one, not a gap",
          ob.find_gaps(wrap) == [] and ob.find_gaps(
              np.array([0, 1], np.uint32), prev=2 ** 32 - 1) == [])


def test_a_decode_failure_is_reported_not_swallowed(monkeypatch):
    """Only OSError was caught; anything else killed the reader silently."""
    print("\nStreamer survives to report")

    class _Sock:
        def __init__(self, nbytes):
            self.nbytes = nbytes

        def settimeout(self, t):
            pass

        def recv(self, n):
            return b"\x00" * self.nbytes

        def close(self):
            pass

    lay = _layout(96)
    st = ob.Streamer("fake", layout=lay, block_samples=4, take_over=False)
    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: _Sock(st.block_bytes))

    def boom(buf, layout):
        raise ValueError("frame shape nobody planned for")

    monkeypatch.setattr(ob, "decode", boom)
    st._run()
    check("the reader records what killed it",
          isinstance(st.error, ValueError) and "nobody planned" in str(st.error),
          repr(st.error))


def test_a_scratchpad_too_short_is_refused(monkeypatch):
    print("\nprobe_uut scratchpad check")

    class _Uut:
        def __init__(self, host):
            self.host = host

        def value(self, knob, site=0):
            return {"ssb": "96", "data32": "0", "spad": "0,0,0"}.get(knob, "")

        def close(self):
            pass

    monkeypatch.setattr(ob, "Uut", _Uut)
    try:
        ob.probe_uut("acq1001_xyz")
        msg = None
    except RuntimeError as e:
        msg = str(e)
    check("a box with the scratchpad off is refused with a reason",
          msg is not None and "scratchpad" in msg and "SamCnt" in msg, str(msg))


def test_uut_cmd_does_not_read_a_stale_reply():
    """A late banner must not be taken as the answer to the next command."""
    print("\nUut.cmd drains stale input")
    client, server = socket.socketpair()
    try:
        u = ob.Uut("fake")
        u._sockets[0] = client
        server.sendall(b"WELCOME late banner\n")
        time.sleep(0.05)                       # let it land in the buffer

        def answer():
            server.settimeout(2)
            server.recv(64)
            server.sendall(b"SIG:CLK:FREQ 200000\n")

        t = threading.Thread(target=answer)
        t.start()
        reply = u.cmd("SIG:CLK:FREQ")
        t.join()
        check("the reply is the answer, not the banner",
              reply == "SIG:CLK:FREQ 200000", repr(reply))
    finally:
        client.close()
        server.close()


def test_capture_default_goes_to_the_captures_directory(tmp_path, monkeypatch):
    print("\ncapture default path")
    monkeypatch.setenv("OCTOBEE_CAPTURES_DIR", str(tmp_path / "caps"))
    lay = _layout(96)
    d = {"ai": np.zeros((2, 32), np.int16), "sam_cnt": np.arange(2, dtype=np.uint32),
         "usec_cnt": np.arange(2, dtype=np.uint32)}
    monkeypatch.setattr(ob, "capture_all",
                        lambda hosts, seconds: dict.fromkeys(hosts, (d, lay)))
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    ob._cmd_capture(("a", "b"), 1.0, None)
    check("with no -o the file lands in the captures directory",
          os.path.exists(tmp_path / "caps" / "octobee_capture.npz"))
    check("and not in the working directory", not os.listdir(cwd))


def _fake_network(monkeypatch, boxes, local=("192.168.1.100",)):
    """Stand in for the bench LAN: `boxes` maps address -> SERIAL.

    Returns a list that records every subnet scanned, so a test can tell
    whether the cheap path was taken.
    """
    scans = []
    monkeypatch.setattr(ob, "_located", {})
    monkeypatch.setattr(ob, "_scanned", {})
    monkeypatch.setattr(ob, "_read_serial", lambda ip, timeout=1.0: boxes.get(ip))
    monkeypatch.setattr(ob, "_local_ips", lambda: list(local))

    def scan(prefix):
        scans.append(prefix)
        return sorted(ip for ip in boxes if ip.startswith(prefix + "."))

    monkeypatch.setattr(ob, "_scan_subnet", scan)
    return scans


def test_a_carrier_where_the_table_says_is_used_without_a_scan(monkeypatch):
    print("\naddress(): boxes where expected")
    scans = _fake_network(monkeypatch, {"192.168.1.84": "CE4010694",
                                        "192.168.1.83": "CE4010695"})
    got = [ob.address(h) for h in ob.DEFAULT_UUTS]
    check("both names map to their table addresses",
          got == ["192.168.1.84", "192.168.1.83"], repr(got))
    check("no subnet was scanned", scans == [], repr(scans))


def test_a_carrier_that_moved_is_found_by_its_serial(monkeypatch, capsys):
    print("\naddress(): 694 moved")
    scans = _fake_network(monkeypatch, {"192.168.1.91": "CE4010694",
                                        "192.168.1.83": "CE4010695"})
    got = ob.address("acq1001_694")
    out = capsys.readouterr().out
    check("694 is found at its new address", got == "192.168.1.91", repr(got))
    check("the move is reported, not silent", "found CE4010694 at 192.168.1.91" in out, out)
    ob.address("acq1001_694")
    ob.address("acq1001_695")
    check("the subnet is scanned once, then remembered", scans == ["192.168.1"],
          repr(scans))


def test_two_carriers_that_swapped_addresses_are_not_swapped(monkeypatch):
    """The failure that would be silent: sensors 1-8 and 9-16 trading places."""
    print("\naddress(): swapped boxes")
    _fake_network(monkeypatch, {"192.168.1.84": "CE4010695",
                                "192.168.1.83": "CE4010694"})
    check("694 follows its serial", ob.address("acq1001_694") == "192.168.1.83")
    check("695 follows its serial", ob.address("acq1001_695") == "192.168.1.84")


def test_a_missing_carrier_names_what_the_scan_did_find(monkeypatch):
    print("\naddress(): 694 absent")
    _fake_network(monkeypatch, {"192.168.1.83": "CE4010695"})
    try:
        ob.address("acq1001_694")
        err = None
    except ConnectionError as e:
        err = str(e)
    check("a ConnectionError is raised", err is not None)
    check("it lists the boxes that were there",
          err is not None and "CE4010695 at 192.168.1.83" in err, err)


def test_a_pc_off_the_carrier_subnet_is_told_how_to_fix_it(monkeypatch):
    """What actually happened on 2026-11-19: Ethernet sat on 169.254.x."""
    print("\naddress(): PC not on 192.168.1.x")
    scans = _fake_network(monkeypatch, {}, local=("10.70.91.186", "169.254.139.50"))
    try:
        ob.address("acq1001_695")
        err = None
    except ConnectionError as e:
        err = str(e)
    check("nothing is scanned from the wrong network", scans == [], repr(scans))
    check("the error says the PC needs a 192.168.1.x address and how",
          err is not None and "192.168.1.x" in err and "netsh" in err, err)


def test_a_host_not_in_the_table_is_used_as_given(monkeypatch):
    print("\naddress(): unknown names pass through")
    scans = _fake_network(monkeypatch, {})
    check("an IP passes through", ob.address("10.0.0.5") == "10.0.0.5")
    check("an unknown hostname passes through", ob.address("fake") == "fake")
    check("neither triggers a scan", scans == [], repr(scans))
