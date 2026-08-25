#!/usr/bin/env python3
"""
octobee_live.py -- live plot of every Hall probe channel, both carriers, one window.

Replaces the Phoebus Data Browser workflow: no PV names to type, no manual
start/end times, no CSV export round-trip. It reads the raw stream straight off
port 4210 on both boxes and draws all 16 sensors at once.

    python octobee_live.py                    # stacked, VCM-subtracted (default)
    python octobee_live.py --mode overlay     # all traces on one pair of axes
    python octobee_live.py --mode grid        # one subplot per sensor
    python octobee_live.py --show-vcm         # include the 16 VCM references
    python octobee_live.py --raw              # raw counts, no VCM subtraction
    python octobee_live.py --fs 0             # do not touch the box's sample rate

Sample rate
-----------
Leave it at 200 kSPS. Measured 2026-08-19: both carriers stream full rate
simultaneously and indefinitely with ZERO lost samples -- 39.1 MB/s combined over
the 1 Gbps link, pipeline lag flat at 2.6 s / 3.1 s over a 150 s run (constant
latency, not a growing backlog).

An earlier version of this tool dropped the clock to 20 kSPS by default, on the
strength of a 5-second throughput test that read 9.8 MB/s. That test was simply
too short: the stream ramps over the first ~30 s (16.5 -> 18.9 MB/s), so brief
measurements badly understate the steady-state rate.

Lowering the clock is now opt-in via --fs, and it is a real cost: the sensor's
analog low-pass is fixed at 100 kHz, so any rate below 200 kSPS aliases and the
noise density rises by sqrt(100kHz/(fs/2)) -- 2.0x at 50 kSPS, 3.2x at 20 kSPS.
Only do it if your host cannot keep up with rendering.

Geometry note
-------------
The 16 sensors sit on a square tube, 4 per face, pointing outwards. Each chip
therefore has a different orientation, so its Bx/By/Bz mean different lab-frame
directions from its neighbours'. Comparing a single axis across sensors is
meaningless -- use the |B| trace (--mode grid shows it per sensor), which is
rotation-invariant.
"""

import argparse
import sys
import time

import numpy as np

import octobee as ob


# --------------------------------------------------------------------------
# board-side oversampling
# --------------------------------------------------------------------------

def set_rate(host, fs_hz, settle=3.0):
    """
    Drop the site-1 ADC clock to `fs_hz` via clkdiv. Returns the previous clkdiv
    so it can be put back. The motherboard clock is read from the box rather than
    assumed, so this works if someone has retuned it.
    """
    u = ob.Uut(host)
    try:
        prev = u.value("clkdiv", site=1)
        clk_mb = float(u.value("SIG:CLK_MB:FREQ").split()[-1])
        div = max(1, int(round(clk_mb / fs_hz)))
        u.cmd(f"clkdiv {div}", site=1)
        time.sleep(settle)
        actual = float(u.value("SIG:CLK_S1:FREQ").split()[-1])
        return prev, actual
    finally:
        u.close()


def restore_rate(host, prev):
    """
    Put `host` back on the clkdiv set_rate() returned. Returns True if it did.

    `prev` is the raw string the box replied with, and a comms hiccup can make
    that empty. Returning quietly then leaves the carrier at the reduced rate
    with nothing said -- the exact state `octobee.py restore` exists to undo --
    so the caller is told instead of guessing.
    """
    if not prev:
        return False
    u = ob.Uut(host)
    try:
        u.cmd(f"clkdiv {prev}", site=1)
        return True
    finally:
        u.close()


# --------------------------------------------------------------------------
# rolling buffer
# --------------------------------------------------------------------------

class Rolling:
    """Fixed-length rolling window of decimated, VCM-corrected channel data."""

    def __init__(self, nchan, npoints):
        self.buf = np.full((npoints, nchan), np.nan)
        self.n = npoints

    def push(self, block):
        k = block.shape[0]
        if k >= self.n:
            self.buf[:] = block[-self.n:]
        else:
            self.buf[:-k] = self.buf[k:]
            self.buf[-k:] = block


def prepare(ai, vpc, subtract_vcm, show_vcm, decim):
    """
    Raw (nsamples, 32) counts -> (ndecim, nchan_out) volts, plus the column list.
    Decimation is by block averaging, which also suppresses noise.
    """
    n = (ai.shape[0] // decim) * decim
    if n == 0:
        return None
    x = ai[:n].astype(np.float64).reshape(n // decim, decim, ai.shape[1]).mean(axis=1)
    cols = []
    out = []
    for s in range(ob.SENSORS_PER_BOX):
        vcm = x[:, s * 4 + 3]
        for a, nm in enumerate(("Bz", "By", "Bx")):
            out.append((x[:, s * 4 + a] - vcm) if subtract_vcm else x[:, s * 4 + a])
            cols.append((s, nm))
        if show_vcm:
            out.append(vcm)
            cols.append((s, "VCM"))
    return np.column_stack(out) * vpc, cols


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------

AXIS_STYLE = {"Bz": "-", "By": "--", "Bx": ":", "VCM": "-."}


def build_figure(mode, labels, nsens, window_s, unit, stack_step):
    import matplotlib.pyplot as plt  # noqa: PLC0415  (see main())

    cmap = plt.get_cmap("turbo")
    colors = [cmap(i / max(nsens - 1, 1)) for i in range(nsens)]

    if mode == "grid":
        ncol = 4
        nrow = int(np.ceil(nsens / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(16, 9), sharex=True)
        axs = np.atleast_1d(axs).ravel()
        lines = []
        for gs, (sensor_idx, chans) in enumerate(labels):
            ax = axs[gs]
            ax.set_title(f"S{sensor_idx+1}", fontsize=9)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
            for nm in chans:
                ln, = ax.plot([], [], lw=0.9, ls=AXIS_STYLE[nm],
                              color=colors[sensor_idx], label=nm)
                lines.append(ln)
            mag, = ax.plot([], [], lw=1.4, color="k", alpha=0.7, label="|B|")
            lines.append(mag)
            if gs == 0:
                ax.legend(fontsize=6, ncol=4, loc="upper right")
        for ax in axs[nsens:]:
            ax.set_visible(False)
        fig.supxlabel("time [s]")
        fig.supylabel(unit)
        return fig, axs, lines, colors

    fig, ax = plt.subplots(figsize=(16, 9))
    lines = []
    for sensor_idx, nm in labels:
        ln, = ax.plot([], [], lw=0.9, ls=AXIS_STYLE[nm], color=colors[sensor_idx],
                      label=f"S{sensor_idx+1} {nm}")
        lines.append(ln)
    ax.set_xlabel("time [s]  (0 = now)")
    ax.set_ylabel(unit if mode == "overlay" else "channel")
    ax.grid(alpha=0.3)
    ax.set_xlim(-window_s, 0)
    if mode == "overlay":
        ax.legend(fontsize=6, ncol=8, loc="upper right")
    return fig, [ax], lines, colors


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uut", action="append", default=None,
                   help=f"carrier hostname, repeat for both (default: "
                        f"{' '.join(ob.DEFAULT_UUTS)})")
    p.add_argument("--mode", choices=("stack", "overlay", "grid"), default="stack")
    p.add_argument("--window", type=float, default=10.0, help="seconds shown")
    p.add_argument("--decim", type=int, default=0,
                   help="extra host-side block averaging (0 = auto for ~2 kHz)")
    p.add_argument("--fs", type=float, default=0.0,
                   help="ADC sample rate to run at while plotting, Hz. "
                        "Default 0 = leave the box alone (200 kSPS streams fine). "
                        "Lowering it aliases: 2.0x noise at 50k, 3.2x at 20k. "
                        "The original clock is restored on exit.")
    p.add_argument("--raw", action="store_true",
                   help="plot raw counts without subtracting each sensor's VCM")
    p.add_argument("--show-vcm", action="store_true",
                   help="also plot the 16 VCM reference channels")
    p.add_argument("--range", dest="b_range", type=float, default=None,
                   choices=sorted(ob.RANGE_TO_VPT), metavar="mT",
                   help="convert to mT using this sensor range (20/40/400/4000)")
    p.add_argument("--stack-step", type=float, default=0.0,
                   help="volts (or mT) between stacked traces; 0 = auto")
    a = p.parse_args()

    hosts = tuple(a.uut) if a.uut else ob.DEFAULT_UUTS
    subtract_vcm = not a.raw

    # ---- board setup -----------------------------------------------------
    # Inside the try, not before it. These calls change hardware state on one
    # box at a time, so a failure on the second carrier used to leave the first
    # running at the reduced rate with nothing in the session saying so -- and
    # a quietly aliased carrier looks exactly like a working one.
    prev_div = {}
    streamers = []
    try:
        if a.fs:
            for h in hosts:
                prev_div[h], actual = set_rate(h, a.fs)
                print(f"{h}: clkdiv {prev_div[h]} -> {actual:.0f} Hz "
                      f"(original restored on exit)")

        for h in hosts:
            st = ob.Streamer(h, block_samples=1024, queue_depth=256)
            st.start()
            streamers.append(st)
            print(f"{h}: streaming, {st.layout}")

        fs_eff = streamers[0].layout.fs_hz
        decim = a.decim or max(1, int(round(fs_eff / 2000)))
        plot_fs = fs_eff / decim
        npoints = max(200, int(a.window * plot_fs))
        print(f"effective rate {fs_eff:.0f} Hz, host decim {decim} "
              f"-> {plot_fs:.0f} points/s, {npoints} points in view")

        vpt = ob.RANGE_TO_VPT[a.b_range] if a.b_range else None
        unit = f"B [mT] (+/-{a.b_range:g} mT range)" if vpt else (
            "counts" if a.raw else "V (VCM subtracted)")
        scale = (1e3 / vpt) if vpt else 1.0

        # ---- channel bookkeeping ----------------------------------------
        per_box_axes = ["Bz", "By", "Bx"] + (["VCM"] if a.show_vcm else [])
        flat_labels, grid_labels = [], []
        for bi in range(len(hosts)):
            for s in range(ob.SENSORS_PER_BOX):
                gs = bi * ob.SENSORS_PER_BOX + s
                grid_labels.append((gs, per_box_axes))
                for nm in per_box_axes:
                    flat_labels.append((gs, nm))
        nsens = len(hosts) * ob.SENSORS_PER_BOX

        # matplotlib stays function-local on purpose: octobee_gui imports this
        # module only for set_rate/restore_rate, and it should not pay a
        # multi-second matplotlib import to reach two socket helpers.
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib.animation import FuncAnimation  # noqa: PLC0415

        labels = grid_labels if a.mode == "grid" else flat_labels
        fig, axs, lines, colors = build_figure(
            a.mode, labels, nsens, a.window, unit, a.stack_step)
        fig.canvas.manager.set_window_title(
            "OCTO-BEE live -- " + ", ".join(hosts))

        nch_box = len(per_box_axes) * ob.SENSORS_PER_BOX
        rolls = [Rolling(nch_box, npoints) for _ in hosts]
        t = np.linspace(-a.window, 0, npoints)
        state = {"step": a.stack_step, "t0": time.time(), "frames": 0}

        def update(_):
            for st, roll in zip(streamers, rolls):
                if st.error:
                    print(f"\n{st.host}: {st.error}", file=sys.stderr)
                    continue
                for blk in st.get_all():
                    if blk is None:
                        continue
                    prep = prepare(blk["ai"], st.layout.volts_per_count,
                                   subtract_vcm, a.show_vcm, decim)
                    if prep is not None:
                        roll.push(prep[0] * scale)

            allv = np.concatenate([r.buf for r in rolls], axis=1)
            if np.all(np.isnan(allv)):
                return lines

            # The buffers start life all-NaN and a railed channel stays that
            # way, so nanmedian/nansum legitimately see empty slices here.
            # This has to be a `with`: building the context manager and not
            # entering it, which is what used to happen, silences nothing.
            with np.errstate(invalid="ignore"):
                if a.mode == "grid":
                    k = 0
                    for gs, (sensor_idx, chans) in enumerate(grid_labels):
                        bi, s = divmod(sensor_idx, ob.SENSORS_PER_BOX)
                        col0 = s * len(per_box_axes)
                        seg = rolls[bi].buf[:, col0:col0 + len(per_box_axes)]
                        if np.all(np.isnan(seg)):
                            continue
                        base = np.nanmedian(seg, axis=0)
                        for j in range(len(chans)):
                            lines[k].set_data(t, seg[:, j] - base[j])
                            k += 1
                        field = seg[:, :3] - base[:3]
                        lines[k].set_data(
                            t, np.sqrt(np.nansum(field ** 2, axis=1)))
                        k += 1
                        ax = axs[gs]
                        ax.set_xlim(-a.window, 0)
                        ax.relim()
                        ax.autoscale_view(scaley=True)
                    return lines

                base = np.nanmedian(allv, axis=0)

            ax = axs[0]
            centred = allv - base
            if a.mode == "overlay":
                for i, ln in enumerate(lines):
                    ln.set_data(t, centred[:, i])
                ax.set_xlim(-a.window, 0)
                ax.relim()
                ax.autoscale_view(scaley=True)
            else:                                    # stack
                if not state["step"]:
                    spread = np.nanmax(np.abs(centred))
                    state["step"] = max(spread * 2.5, 1e-9)
                step = state["step"]
                for i, ln in enumerate(lines):
                    ln.set_data(t, centred[:, i] + i * step)
                ax.set_ylim(-step, len(lines) * step)
                ax.set_yticks([i * step for i in range(len(lines))])
                ax.set_yticklabels([f"S{s+1} {nm}" for s, nm in flat_labels],
                                   fontsize=6)
                for tick, (s, _) in zip(ax.get_yticklabels(), flat_labels):
                    tick.set_color(colors[s])

            state["frames"] += 1
            rate = sum(s.bytes_read for s in streamers) / 1e6 / (
                time.time() - state["t0"])
            drops = sum(s.dropped for s in streamers)
            ax.set_title(f"{nsens} sensors, {len(lines)} traces  |  "
                         f"{rate:.1f} MB/s  |  {drops} block(s) dropped  |  "
                         f"{unit}", fontsize=10)
            return lines

        anim = FuncAnimation(fig, update, interval=100,
                             blit=False, cache_frame_data=False)
        fig._octobee_anim = anim          # keep a reference alive
        print("\nclose the window to stop.")
        plt.show()

    finally:
        for st in streamers:
            st.stop()
        for h, prev in prev_div.items():
            try:
                if restore_rate(h, prev):
                    print(f"{h}: clkdiv restored to {prev}")
                else:
                    print(f"{h}: NO clkdiv recorded, so the rate was NOT put "
                          f"back -- run 'python octobee.py restore'",
                          file=sys.stderr)
            except OSError as exc:
                # One unreachable box must not stop the other being restored.
                print(f"{h}: could not restore clkdiv ({exc}) -- run "
                      f"'python octobee.py restore'", file=sys.stderr)


if __name__ == "__main__":
    main()
