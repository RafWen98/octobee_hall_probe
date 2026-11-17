"""LiveSource: keeping two free-running boxes paired through a loss.

The carriers are paired by row index. A block shed by a full queue (or samples
missing inside one) removes rows from ONE box, and without repair every later
row is shifted against the other box and against the encoder counts. These
tests drive LiveSource with synthetic blocks -- no sockets -- in which every
sample carries the instant it was taken at, so "row i of both boxes is the same
instant" can be asserted directly.
"""

import queue

import numpy as np

from octobee.acq import carrier as ob
from octobee.gui.sources import LiveSource
from tests.helpers import (
    check,
)

BLOCK = 100
HOSTS = ["acq1001_694", "acq1001_695"]


class _Stream:
    """Just enough of a Streamer: blocks pushed by the test, error settable."""

    def __init__(self):
        self.q = queue.Queue()
        self.error, self.dropped, self.bytes_read = None, 0, 0
        self.blocks = []

    def get_all(self):
        out, self.blocks = self.blocks, []
        return out


def _layouts():
    lay694 = ob.Layout(ssb=96, spad_lw=7, fs_hz=200000.0, adc_range="+/-10V",
                       sites=(1, 2), counting_sites=())
    lay695 = ob.Layout(ssb=104, spad_lw=7, fs_hz=200000.0, adc_range="+/-10V",
                       sites=(1, 2, 5, 6), counting_sites=(2, 5, 6))
    return [lay694, lay695]


def _source():
    streams = [_Stream(), _Stream()]
    return LiveSource(HOSTS, _layouts(), streams), streams


def _block(box, times, base):
    """A decoded block whose samples are the instants `times`.

    Channel 0 of the analogue data and every encoder column hold the instant,
    and the sample counter is base + instant. The two boxes get unrelated
    bases, as the real ones do.
    """
    t = np.asarray(times, dtype=np.int64)
    ai = np.zeros((len(t), 32), dtype=np.int16)
    ai[:, 0] = t
    blk = {"ai": ai, "sam_cnt": ((t + base) & 0xFFFFFFFF).astype(np.uint32),
           "usec_cnt": np.zeros(len(t), dtype=np.uint32)}
    blk["enc"] = np.repeat(t[:, None], 3, axis=1).astype(np.uint32)
    return blk


def _run(drop=(), cut=(), bases=(1000, 5_000_000), n_blocks=10, interleave=True):
    """Feed n_blocks per box, minus whole blocks `drop` and samples `cut`.

    drop: {(box, block_index)}; cut: {(box, first_instant, n)} samples removed
    from inside a block. Returns (source, rows_a, rows_b, enc) as read back.
    """
    src, streams = _source()
    seen = [[], [], []]

    def pull():
        out = src.read()
        if out is not None:
            seen[0].append(out[0][:, 0].astype(int))
            seen[1].append(out[1][:, 0].astype(int))
            seen[2].append(None if src.enc is None else src.enc[:, 0].astype(int))

    for b in range(n_blocks):
        for box in (0, 1):
            if (box, b) in drop:
                continue
            times = np.arange(b * BLOCK, (b + 1) * BLOCK)
            for cbox, first, n in cut:
                if cbox == box:
                    times = times[~((times >= first) & (times < first + n))]
            streams[box].blocks.append(_block(box, times, bases[box]))
            if interleave:
                pull()
    pull()
    a = np.concatenate(seen[0]) if seen[0] else np.array([], int)
    c = np.concatenate(seen[1]) if seen[1] else np.array([], int)
    enc = [e for e in seen[2] if e is not None]
    return src, a, c, (np.concatenate(enc) if enc else np.array([], int))


def _paired(a, b, enc):
    """Row i of both boxes, and of the counts, is the same instant."""
    return len(a) == len(b) == len(enc) and np.array_equal(a, b) \
        and np.array_equal(a, enc)


def test_no_loss_is_untouched():
    print("\nLiveSource, a clean stream")
    src, a, b, enc = _run()
    check("every row comes out, paired, with its counts",
          len(a) == 10 * BLOCK and _paired(a, b, enc), f"{len(a)} rows")
    check("and nothing is counted as lost",
          src.gaps == 0 and src.lost == 0, f"{src.gaps} gaps, {src.lost} lost")


def test_dropped_block_on_each_box():
    """A block shed on box A, then on box B, must leave the pairing intact."""
    print("\nLiveSource, a dropped block")
    for box, name in ((0, "the non-encoder box (694)"),
                      (1, "the encoder box (695)")):
        src, a, b, enc = _run(drop={(box, 3)})
        check(f"loss on {name}: the lost samples are counted, between blocks",
              src.gaps == 1 and src.lost == BLOCK,
              f"{src.gaps} gaps, {src.lost} lost")
        check(f"loss on {name}: rows after the hole are still paired",
              _paired(a, b, enc), f"{len(a)} rows, first mismatch "
              f"{int(np.argmax(a != b)) if len(a) == len(b) else '?'}")
        check(f"loss on {name}: the time axis has a hole where the block was",
              not np.any((a >= 3 * BLOCK) & (a < 4 * BLOCK))
              and len(a) == 9 * BLOCK, f"{len(a)} rows")
        check(f"loss on {name}: drift is reported in stats()",
              "box offset" in src.stats(), str(src.stats()))


def test_loss_inside_a_block():
    print("\nLiveSource, samples missing inside a block")
    for box in (0, 1):
        src, a, b, enc = _run(cut={(box, 453, 7)})
        check(f"box {box}: the in-block loss is counted",
              src.gaps == 1 and src.lost == 7,
              f"{src.gaps} gaps, {src.lost} lost")
        check(f"box {box}: rows are paired after it",
              _paired(a, b, enc) and len(a) == 10 * BLOCK - 7,
              f"{len(a)} rows")


def test_loss_larger_than_what_is_pending():
    """Box A's data all arrives first, so box B owes rows it has not got yet."""
    print("\nLiveSource, a debt carried forward")
    src, a, b, enc = _run(drop={(0, 2), (0, 3), (0, 6)}, interleave=False)
    check("three dropped blocks are counted",
          src.lost == 3 * BLOCK and src.gaps == 2,
          f"{src.gaps} gaps, {src.lost} lost")
    check("the box that got everything first is trimmed as the other arrives",
          _paired(a, b, enc) and len(a) == 7 * BLOCK, f"{len(a)} rows")
    # One-sided arrival the other way round: the encoder box is far ahead.
    src, streams = _source()
    for blk_i in range(6):
        if blk_i not in (1, 2):
            streams[1].blocks.append(_block(
                1, np.arange(blk_i * BLOCK, (blk_i + 1) * BLOCK), 77))
    src.read()
    for blk_i in range(6):
        streams[0].blocks.append(_block(
            0, np.arange(blk_i * BLOCK, (blk_i + 1) * BLOCK), 99))
    out = src.read()
    check("a loss on the encoder box while the other lags still pairs",
          out is not None
          and np.array_equal(out[0][:, 0], out[1][:, 0])
          and np.array_equal(out[0][:, 0].astype(int), src.enc[:, 0]),
          "" if out is None else f"{len(out[0])} rows")


def test_both_boxes_lose_different_blocks():
    print("\nLiveSource, losses on both boxes")
    src, a, b, enc = _run(drop={(0, 2), (1, 5)})
    check("both are counted", src.gaps == 2 and src.lost == 2 * BLOCK,
          f"{src.gaps} gaps, {src.lost} lost")
    check("and the survivors are still the same instants on both boxes",
          _paired(a, b, enc) and len(a) == 8 * BLOCK, f"{len(a)} rows")


def test_counter_wrap_is_not_a_loss():
    print("\nLiveSource, 32-bit counter wrap")
    bases = (2 ** 32 - 250, 2 ** 32 - 777)
    src, a, b, enc = _run(bases=bases)
    check("a wrap on each box is not counted as lost",
          src.gaps == 0 and src.lost == 0, f"{src.gaps} gaps, {src.lost} lost")
    src, a, b, enc = _run(bases=bases, drop={(0, 3)})
    check("a real loss across the wrap is counted exactly",
          src.lost == BLOCK and _paired(a, b, enc),
          f"{src.lost} lost, {len(a)} rows")


def test_a_failed_reader_is_reported_once():
    print("\nLiveSource, a dead reader")
    src, streams = _source()
    check("healthy: nothing failed", src.failed == [], str(src.failed))
    streams[1].error = RuntimeError("boom")
    lines = []
    for _ in range(5):
        src.read()
        if src.error:
            lines.append(src.error)
            src.error = None        # what the GUI does every tick
    check("the error reaches the caller exactly once",
          len(lines) == 1 and "boom" in lines[0] and HOSTS[1] in lines[0],
          str(lines))
    check("but the source still says it is dead, and which box",
          src.failed == [HOSTS[1]], str(src.failed))


def test_block_index_follows_each_boxes_own_counter():
    """What the raw archive records: where each box's counter is at a block."""
    print("\nLiveSource block index")
    src, streams = _source()
    check("nothing read yet, nothing to index", src.block_index is None)
    for box, base in ((0, 1000), (1, 5_000_000)):
        streams[box].blocks.append(_block(box, np.arange(0, BLOCK), base))
    src.read()
    check("the first block starts at each box's own counter",
          src.block_index["sam_cnt"] == [1000, 5_000_000], str(src.block_index))
    # Box 0 loses its next block; box 1 does not. Box 1's rows are trimmed to
    # match, so the block that follows starts further on in BOTH counters.
    streams[0].blocks.append(_block(0, np.arange(2 * BLOCK, 3 * BLOCK), 1000))
    streams[1].blocks.append(_block(1, np.arange(BLOCK, 2 * BLOCK), 5_000_000))
    streams[1].blocks.append(_block(1, np.arange(2 * BLOCK, 3 * BLOCK), 5_000_000))
    src.read()
    check("a repaired hole shows as a jump in both counters, and is totalled",
          src.block_index["sam_cnt"] == [1000 + 2 * BLOCK, 5_000_000 + 2 * BLOCK]
          and src.block_index["lost"] == BLOCK and src.block_index["gaps"] == 1,
          str(src.block_index))
