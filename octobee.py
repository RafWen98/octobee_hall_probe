#!/usr/bin/env python3
"""
octobee.py -- core library for the Hampton3D / OCTO-BEE Hall probe system.

Hardware chain (as found on the bench, 2026-08-19)
--------------------------------------------------
    16 x SENIS SENM3Dx v2  (3D Hall ASIC, analog out)
      -> OCTO-BEE-CONC / OCTO-BEE-CELF  (analog concentrator, 8 sensors each)
      -> ACQ423ELF  (32 ch, 16-bit, site 1)
      -> ACQ1001S carrier, raw stream on TCP port 4210

    Two carriers, one LAN cable each, 8 sensors per carrier:
        acq1001_694  192.168.1.82   ACQ423 s/n E42310297   -> sensors  1..8
        acq1001_695  192.168.1.83   ACQ423 s/n E42310298   -> sensors  9..16

    Which carrier holds which physical half of the probe is an ASSUMPTION.
    Confirm it (and the axis order) with onbox_sensor_audit.py / a magnet pass.

Channel map (OCTO-BEE User Guide rev 6, sec 3.7)
------------------------------------------------
    Each SENIS eval kit occupies 4 consecutive ACQ423 channels *on its own box*:
        ch 4k+1 = Bz     ch 4k+2 = By     ch 4k+3 = Bx     ch 4k+4 = VCM
    so kit 1 -> ch 1-4, kit 2 -> ch 5-8, ... kit 8 -> ch 29-32.

    VCM is the sensor's own virtual-ground reference (~2.2 V). It is NOT a field
    channel -- it is the zero point that must be subtracted from that same
    sensor's Bx/By/Bz before any field is quoted. Every chip has its own VCM,
    and on this system they differ by up to ~90 mV between chips.

Stream frame layout
-------------------
    Read back from each box at run time, because the two boxes DIFFER:

    acq1001_694: ssb=96  (24 LW)  sites=1,2      -> 1 site-2 word
    acq1001_695: ssb=104 (26 LW)  sites=1,2,5,6  -> 3 encoder words (ENC X/Y/Z)

        int16 word  0..31      : ACQ423 channels 1..32 (little endian, packed)
        uint32 LW  16..n       : site-2 / encoder words (0 when none fitted)
        uint32 LW  n+0         : SamCnt   -- sample counter, +1 per sample
        uint32 LW  n+1         : uSecCnt  -- 1 MHz counter, +5 per sample @200kSPS
        uint32 LW  n+2..n+5    : TCH1/2, TCH3/4, TCH5/6, TCH7/8 ASIC temperatures
        uint32 LW  n+6         : PWR_GOOD -- power monitor, latched at boot only
"""

import socket
import time

import numpy as np

# The two carriers. Order here defines the global sensor numbering.
DEFAULT_UUTS = ("acq1001_694", "acq1001_695")

PORT_STREAM = 4210          # raw aggregator stream
PORT_SITE0 = 4220           # motherboard command port; site N -> 4220 + N

NCHAN_AI = 32               # ACQ423ELF channels per box
SENSORS_PER_BOX = NCHAN_AI // 4
AXES = ("Z", "Y", "X", "VCM")

# ACQ423ELF full scale, from the 'GAIN:ALL' knob. Volts per count = span/65536.
ADC_RANGES = {
    "+/-10V": 20.0,
    "+/-5V": 10.0,
    "0-10V": 10.0,
    "0-5V": 5.0,
}

# SENM3Dx v2 datasheet Table 7 / Table 21: amplifier gain -> (range_mT, V_per_T)
GAIN_TO_RANGE = {
    3000: (20.0, 63.0),
    1500: (40.0, 34.65),
    150: (400.0, 3.78),
    15: (4000.0, 0.378),
}
RANGE_TO_VPT = {r: v for r, v in GAIN_TO_RANGE.values()}

# Temperature sensor, datasheet Table 11 (analog TA read by the CELF's ADS7128)
TEMP_V_AT_0C = 1.65
TEMP_V_PER_C = 0.01030
TEMP_ADC_VREF = 5.0
TEMP_ADC_BITS = 12


# --------------------------------------------------------------------------
# channel naming
# --------------------------------------------------------------------------

def channel_label(ch, box_index=0):
    """1-based per-box channel -> ('S3', 'Bx'). box_index 0 -> S1..S8, 1 -> S9..S16."""
    sensor = box_index * SENSORS_PER_BOX + (ch - 1) // 4 + 1
    axis = AXES[(ch - 1) % 4]
    return f"S{sensor}", ("VCM" if axis == "VCM" else f"B{axis.lower()}")


def channel_labels(box_index=0):
    return [f"{s} {a}" for s, a in
            (channel_label(c, box_index) for c in range(1, NCHAN_AI + 1))]


def all_labels(n_boxes=2):
    out = []
    for b in range(n_boxes):
        out += channel_labels(b)
    return out


def is_vcm(ch0):
    """0-based per-box channel index -> True if it is a VCM reference channel."""
    return ch0 % 4 == 3


def vcm_channel_for(ch0):
    """0-based field channel -> 0-based index of its sensor's VCM channel."""
    return (ch0 // 4) * 4 + 3


# --------------------------------------------------------------------------
# UUT command interface
# --------------------------------------------------------------------------

class Uut:
    """
    Plain-text key=value command interface to an ACQ400 carrier.

    On reply timing: the command port answers in about 2 ms and terminates
    every reply with a newline -- including the empty reply a write gives back.
    It never closes the connection, so reading "until the socket runs dry" costs
    a full socket timeout on every single command. That is invisible in a script
    that reads two knobs and crippling in one that reads fourteen: connecting to
    both carriers took 85 s, essentially all of it spent waiting for timeouts
    that were never going to carry data.

    So: poll for the reply to start, then drain until the line goes quiet.
    A knob read costs a few milliseconds instead of 1.3 s.
    """

    REPLY_TIMEOUT = 2.0      # longest wait for a reply to begin
    REPLY_GAP = 0.06         # quiet period that marks the end of one
    BANNER_WAIT = 0.15       # some ports greet, most do not

    def __init__(self, host, timeout=5.0):
        self.host = host
        self.timeout = timeout
        self._sockets = {}

    def _sock(self, site):
        if site not in self._sockets:
            s = socket.create_connection((self.host, PORT_SITE0 + site),
                                         timeout=self.timeout)
            s.settimeout(self.BANNER_WAIT)
            try:
                s.recv(4096)          # drain banner if there is one
            except OSError:
                pass
            self._sockets[site] = s
        return self._sockets[site]

    def cmd(self, knob, site=0, settle=0.0):
        s = self._sock(site)
        s.sendall((knob + "\n").encode())
        s.settimeout(self.REPLY_GAP)
        buf = b""
        deadline = time.time() + self.REPLY_TIMEOUT
        while time.time() < deadline:
            try:
                d = s.recv(16384)
            except TimeoutError:
                if buf:
                    break                    # reply finished
                continue                     # still waiting for it to start
            except OSError:
                break
            if not d:
                break                        # peer closed
            buf += d
            deadline = time.time() + self.REPLY_GAP
        if settle:
            time.sleep(settle)
        return buf.decode(errors="replace").strip()

    def value(self, knob, site=0):
        """Return just the value part of a 'KNOB value' reply."""
        r = self.cmd(knob, site)
        if r.startswith(knob):
            r = r[len(knob):]
        return r.strip()

    def close(self):
        for s in self._sockets.values():
            try:
                s.close()
            except OSError:
                pass
        self._sockets.clear()


class Layout:
    """Stream frame geometry for one carrier, read back from the box."""

    def __init__(self, ssb, nchan_ai=NCHAN_AI, spad_lw=7, fs_hz=200000.0,
                 adc_range="+/-10V"):
        self.ssb = ssb
        self.nchan_ai = nchan_ai
        self.spad_lw = spad_lw
        self.fs_hz = fs_hz
        self.adc_range = adc_range
        self.n_i16 = ssb // 2
        self.n_lw = ssb // 4
        self.ai_lw = nchan_ai // 2            # ACQ423 packed: 2 channels per LWORD
        self.spad0_lw = self.n_lw - spad_lw   # first SPAD longword
        self.site2_lw = self.ai_lw
        self.n_site2_lw = self.spad0_lw - self.ai_lw

    @property
    def volts_per_count(self):
        return ADC_RANGES[self.adc_range] / 65536.0

    def __repr__(self):
        return (f"Layout(ssb={self.ssb}B, {self.nchan_ai}ch, "
                f"{self.n_site2_lw} enc LW, {self.spad_lw} SPAD LW, "
                f"fs={self.fs_hz/1e3:g}kHz, {self.adc_range})")


def probe_uut(host):
    """Read live frame geometry, sample rate and ADC range off one box."""
    u = Uut(host)
    try:
        ssb = int(u.value("ssb"))
        if int(u.value("data32")):
            raise RuntimeError(f"{host}: data32=1, this decoder assumes packed 16-bit")
        spad = u.value("spad").split(",")          # e.g. '1,7,0'
        spad_lw = int(spad[1]) if len(spad) > 1 and spad[0] != "0" else 0
        fs = float(u.value("SIG:CLK_S1:FREQ").split()[-1])
        nchan_ai = int(u.value("NCHAN", site=1))
        rng = u.value("GAIN:ALL", site=1).strip()
        return Layout(ssb, nchan_ai, spad_lw, fs, rng)
    finally:
        u.close()


def stop_live_stream(host, wait=2.0):
    """
    Release port 4210.

    Phoebus' 'Streaming Capture' start button runs a CONTINUOUS stream that owns
    the data mover; while it runs a fresh connection to 4210 gets an immediate
    EOF. This is the handoff described in OCTO-BEE guide sec 3.5.
    Returns True if a running stream was actually stopped.
    """
    u = Uut(host)
    try:
        if "RUN" in u.cmd("CONTINUOUS:STATE"):
            u.cmd("CONTINUOUS 0")
            time.sleep(wait)
            return True
        return False
    finally:
        u.close()


# --------------------------------------------------------------------------
# frame decoding
# --------------------------------------------------------------------------

def decode(buf, layout):
    """
    bytes -> dict of arrays. Trailing partial frames are discarded.
    Returns None if the buffer holds less than one whole frame.
    """
    i16 = np.frombuffer(buf, dtype="<i2")
    n = len(i16) // layout.n_i16
    if n == 0:
        return None
    i16 = i16[:n * layout.n_i16].reshape(n, layout.n_i16)
    u32 = i16.view("<u4").reshape(n, layout.n_lw)

    out = {
        "ai": i16[:, :layout.nchan_ai],
        "enc": u32[:, layout.site2_lw:layout.spad0_lw],
        "sam_cnt": u32[:, layout.spad0_lw],
        "usec_cnt": u32[:, layout.spad0_lw + 1],
    }
    # SPAD: [SamCnt, uSecCnt, TCH1/2, TCH3/4, TCH5/6, TCH7/8, PWR_GOOD]
    if layout.spad_lw >= 7:
        t = u32[0, layout.spad0_lw + 2:layout.spad0_lw + 6]
        out["temp_raw"] = np.array(
            [v for w in t for v in (w & 0xFFFF, (w >> 16) & 0xFFFF)], dtype=np.uint32)
        out["pwr_good"] = int(u32[0, layout.spad0_lw + 6])
    return out


def temp_c(raw16):
    """ASIC temperature code (16-bit word, upper 12 bits valid) -> degC."""
    code12 = np.asarray(raw16, dtype=np.float64).astype(np.int64) >> 4
    volts = code12 * TEMP_ADC_VREF / (1 << TEMP_ADC_BITS)
    return (volts - TEMP_V_AT_0C) / TEMP_V_PER_C


def check_continuity(sam_cnt):
    """(n_gaps, n_samples_lost) from the SPAD sample counter."""
    d = np.diff(sam_cnt.astype(np.int64))
    bad = d != 1
    return int(bad.sum()), int((d[bad] - 1).sum()) if bad.any() else 0


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------

def capture_one(host, seconds=2.0, layout=None, take_over=True, verbose=True):
    """Grab `seconds` of raw stream from one box and decode it."""
    if layout is None:
        layout = probe_uut(host)
    if take_over and stop_live_stream(host) and verbose:
        print(f"{host}: stopped the Phoebus CONTINUOUS stream to free port 4210")

    nbytes = int(seconds * layout.fs_hz) * layout.ssb
    s = socket.create_connection((host, PORT_STREAM), timeout=10)
    s.settimeout(10)
    buf = bytearray()
    t0 = time.time()
    try:
        while len(buf) < nbytes and time.time() - t0 < seconds + 15:
            d = s.recv(1 << 18)
            if not d:
                break
            buf += d
    finally:
        s.close()
    if not buf:
        raise RuntimeError(
            f"no data on {host}:{PORT_STREAM}. Something else owns the stream -- "
            f"stop the Phoebus 'Streaming Capture' on that box and retry.")
    if verbose:
        print(f"{host}: {len(buf)/1e6:.2f} MB, {len(buf)//layout.ssb} samples, "
              f"{time.time()-t0:.1f} s")
    return decode(bytes(buf), layout), layout


def capture_all(hosts=DEFAULT_UUTS, seconds=2.0, take_over=True, verbose=True):
    """
    Capture from every box concurrently.

    The carriers are NOT hardware-synchronised (each free-runs on its own
    200 kHz clock), so the two halves are only aligned to about the start of
    the capture. Fine for magnet-pass amplitude work; not for phase.
    """
    import threading
    results = {}

    def grab(h):
        try:
            results[h] = capture_one(h, seconds, None, take_over, verbose)
        except Exception as e:                       # noqa: BLE001 - reported below
            results[h] = e

    threads = [threading.Thread(target=grab, args=(h,)) for h in hosts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for h in hosts:
        if isinstance(results.get(h), Exception):
            raise results[h]
    return results


class Streamer:
    """Background reader for one box; decoded blocks arrive via get_all()."""

    def __init__(self, host, layout=None, block_samples=4096, take_over=True,
                 queue_depth=64):
        import queue
        import threading
        self.host = host
        self.layout = layout or probe_uut(host)
        self.block_bytes = block_samples * self.layout.ssb
        self.take_over = take_over
        self.q = queue.Queue(maxsize=queue_depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.error = None
        self.bytes_read = 0
        self.dropped = 0

    def start(self):
        if self.take_over:
            stop_live_stream(self.host)
        self._thread.start()

    def _run(self):
        import queue
        try:
            s = socket.create_connection((self.host, PORT_STREAM), timeout=10)
            s.settimeout(5)
        except OSError as e:
            self.error = e
            return
        buf = bytearray()
        try:
            while not self._stop.is_set():
                d = s.recv(1 << 18)
                if not d:
                    self.error = RuntimeError("stream closed by UUT")
                    break
                buf += d
                self.bytes_read += len(d)
                while len(buf) >= self.block_bytes:
                    blk = bytes(buf[:self.block_bytes])
                    del buf[:self.block_bytes]
                    try:
                        self.q.put_nowait(decode(blk, self.layout))
                    except queue.Full:
                        self.dropped += 1            # GUI is behind; drop the block
        except OSError as e:
            if not self._stop.is_set():
                self.error = e
        finally:
            s.close()

    def get_all(self):
        import queue
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _cmd_info(hosts):
    for bi, h in enumerate(hosts):
        lay = probe_uut(h)
        u = Uut(h)
        s0 = bi * SENSORS_PER_BOX + 1
        print(f"=== {h}  (sensors S{s0}..S{s0+SENSORS_PER_BOX-1})")
        print(f"    model     : {u.value('MODEL', 1)}   serial {u.value('SERIAL', 1)}")
        print(f"    sites     : {u.value('SITELIST')}")
        print(f"    layout    : {lay}")
        print(f"    ADC range : {lay.adc_range}  ({lay.volts_per_count*1e6:.1f} uV/count)")
        print(f"    stream    : {u.cmd('CONTINUOUS:STATE')}")
        u.close()
    print("\nchannel map per box (OCTO-BEE guide sec 3.7):")
    print("    ch 4k+1 = Bz   ch 4k+2 = By   ch 4k+3 = Bx   ch 4k+4 = VCM")
    for bi, h in enumerate(hosts):
        print(f"  {h}:")
        for row in range(SENSORS_PER_BOX):
            cells = []
            for c in range(row * 4 + 1, row * 4 + 5):
                s, a = channel_label(c, bi)
                cells.append(f"ch{c:02d}={s:>3s} {a:3s}")
            print("    " + "   ".join(cells))


def _cmd_capture(hosts, seconds, out):
    res = capture_all(hosts, seconds)
    save = {}
    for bi, h in enumerate(hosts):
        d, lay = res[h]
        gaps, lost = check_continuity(d["sam_cnt"])
        print(f"{h}: {d['ai'].shape[0]} samples x 32 ch, {gaps} gap(s), {lost} lost")
        save[f"ai_{bi}"] = d["ai"]
        save[f"sam_cnt_{bi}"] = d["sam_cnt"]
        save[f"usec_cnt_{bi}"] = d["usec_cnt"]
        save[f"temp_raw_{bi}"] = d.get("temp_raw", np.zeros(8))
        save[f"pwr_good_{bi}"] = d.get("pwr_good", 0)
        save[f"fs_hz_{bi}"] = lay.fs_hz
        save[f"vpc_{bi}"] = lay.volts_per_count
        save[f"adc_range_{bi}"] = lay.adc_range
    save["hosts"] = np.array(list(hosts))
    np.savez_compressed(out, **save)
    print(f"saved {out}")


def alias_penalty(fs_hz, lpf_hz=100000.0):
    """
    Noise-density penalty from sampling below Nyquist for the sensor's analog
    low-pass. The SENM3Dx LPF is fixed at 100 kHz (PWM_CTRL bits 5:4, already at
    its narrowest), so anything under 200 kSPS folds 0-100 kHz into 0-fs/2.
    Measured 3.1x at 20 kSPS against the 3.16x this predicts.
    """
    nyq = fs_hz / 2.0
    return 1.0 if nyq >= lpf_hz else (lpf_hz / nyq) ** 0.5


def _cmd_rate(hosts, fs_hz):
    """Report, and optionally set, the ADC sample rate on every box."""
    if fs_hz:
        for h in hosts:
            u = Uut(h)
            try:
                clk_mb = float(u.value("SIG:CLK_MB:FREQ").split()[-1])
                u.cmd(f"clkdiv {max(1, int(round(clk_mb / fs_hz)))}", site=1)
            finally:
                u.close()
        time.sleep(4)
    for h in hosts:
        u = Uut(h)
        try:
            fs = float(u.value("SIG:CLK_S1:FREQ").split()[-1])
            ssb = int(u.value("ssb"))
            div = u.value("clkdiv", site=1)
            pen = alias_penalty(fs)
            note = "" if pen <= 1.05 else "  <- aliased, sensor LPF is 100 kHz"
            print(f"{h}: {fs:>9.0f} Hz  clkdiv {div:>5}  "
                  f"{fs*ssb/1e6:5.2f} MB/s  noise x{pen:.2f}{note}")
        finally:
            u.close()
    print("both carriers sustain full 200 kSPS indefinitely (39.1 MB/s combined,")
    print("zero lost samples over 150 s). Lower the rate only if your HOST cannot")
    print("keep up -- it costs noise, it does not buy reliability.")


def _cmd_restore(hosts, fs_hz):
    """
    Put the site-1 ADC clock back. octobee_live.py restores it on a clean exit,
    but a killed process (or a lost connection) leaves the box at the reduced
    rate -- this is the undo.
    """
    for h in hosts:
        u = Uut(h)
        try:
            clk_mb = float(u.value("SIG:CLK_MB:FREQ").split()[-1])
            div = max(1, int(round(clk_mb / fs_hz)))
            u.cmd(f"clkdiv {div}", site=1)
        finally:
            u.close()
    time.sleep(4)
    for h in hosts:
        u = Uut(h)
        try:
            print(f"{h}: clkdiv {u.value('clkdiv', 1)}, "
                  f"{u.value('SIG:CLK_S1:FREQ').split()[-1]} Hz")
        finally:
            u.close()


def main():
    import argparse
    p = argparse.ArgumentParser(description="OCTO-BEE stream utilities")
    p.add_argument("--uut", action="append", default=None,
                   help="carrier hostname; repeat for both boxes "
                        f"(default: {' '.join(DEFAULT_UUTS)})")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="print live configuration of every box")
    r = sub.add_parser("restore", help="put the ADC clock back to its normal rate "
                                       "(use if a live session was killed)")
    r.add_argument("--fs", type=float, default=200000.0)
    rt = sub.add_parser("rate", help="report, or set, the ADC sample rate. "
                                     "Lowering it aliases -- the cost is printed.")
    rt.add_argument("--fs", type=float, default=0.0,
                    help="sample rate in Hz; omit to just report")
    c = sub.add_parser("capture", help="capture raw data from every box to an .npz")
    c.add_argument("--seconds", type=float, default=2.0)
    c.add_argument("-o", "--out", default="octobee_capture.npz")
    a = p.parse_args()
    hosts = tuple(a.uut) if a.uut else DEFAULT_UUTS

    if a.cmd == "info":
        _cmd_info(hosts)
    elif a.cmd == "restore":
        _cmd_restore(hosts, a.fs)
    elif a.cmd == "rate":
        _cmd_rate(hosts, a.fs)
    else:
        _cmd_capture(hosts, a.seconds, a.out)


if __name__ == "__main__":
    main()
