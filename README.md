# OCTO-BEE Hall probe — setup, live view, and calibration

Tooling that replaces the Phoebus Data Browser round-trip for the 16-sensor
3D Hall probe, plus what the hardware actually reports today.

Everything here was verified against the live hardware on 2026-08-19.

---

## 1. What the system actually is

You have **two** carriers, not one. That is why you were only ever seeing 32
channels: Phoebus was pointed at one box.

| carrier | IP | ACQ423 s/n | sensors | stream frame |
|---|---|---|---|---|
| `acq1001_694` | 192.168.1.82 | E42310297 | S1–S8 | 96 B (24 LW), 1 site-2 word (**not** an encoder) |
| `acq1001_695` | 192.168.1.83 | E42310298 | S9–S16 | 104 B (26 LW), 3 encoder words (ENC_X/Y/Z) |

Both: ACQ423ELF, 32 ch, 16-bit packed, ±10 V, 200 kSPS, SPAD = 7 longwords.
The two boxes have **different frame layouts**, so anything that decodes the raw
stream has to read the geometry off the box rather than hard-code it. The tools
here do that.

The asymmetry is not cosmetic: only 695 has quadrature-encoder logic in its
FPGA, so only 695 can read an encoder at all. See *The encoders* in section 9.

The carriers are **not** synchronised — each free-runs on its own oscillator
(measured 200001 Hz vs 199999 Hz). The two halves of the probe drift relative to
each other by about 1 sample per 100 k. Fine for amplitude work, not for phase.

### Channel map

Per box, each SENIS eval kit takes 4 consecutive ACQ423 channels:

```
ch 4k+1 = Bz    ch 4k+2 = By    ch 4k+3 = Bx    ch 4k+4 = VCM
```

so kit 1 → ch 1–4, kit 2 → ch 5–8, … kit 8 → ch 29–32, on each box.

**VCM is not a field channel.** It is that chip's own virtual-ground reference
(~2.2 V), and it must be subtracted from that same chip's Bx/By/Bz before any
field number is quoted. The 16 chips' VCMs differ by up to ~90 mV.

This map comes from the OCTO-BEE guide §3.7. It is documentation, not
measurement — confirm it on your hardware with `octobee idmap` (§4).

---

## 2. Quick start

Install once, from the checkout:

```bash
pip install -e ".[gui,dev]"
```

That puts two commands on PATH: `octobee` for the instruments, and
`octobee-gui` for the desktop application. `octobee` on its own lists the
subcommands. Both work from any directory — configuration is located, not
looked for in the working directory.

```bash
octobee probe info                  # live config + channel map, both boxes
octobee live                  # LIVE PLOT, all 16 sensors, one window
octobee report --seconds 5       # health + calibration report
```

`octobee live` modes:

| flag | view |
|---|---|
| *(default)* `--mode stack` | 48 field traces stacked, labelled `S1 Bz` … `S16 Bx` |
| `--mode overlay` | all traces on one pair of axes — use this to compare spike heights |
| `--mode grid` | one subplot per sensor, with a black \|B\| trace |
| `--show-vcm` | adds the 16 VCM references |
| `--range 20` | y-axis in mT instead of volts |

### Sample rate while streaming live

**Both carriers sustain full 200 kSPS indefinitely.** Measured 2026-08-19:
39.1 MB/s combined, **zero lost samples** over a 150 s run, with the pipeline lag
flat at 2.6 s / 3.1 s — constant latency, not a growing backlog. The 1 Gbps link
runs at about 31 % utilisation.

An earlier version of this file claimed a 9.8 MB/s ceiling. That came from a
5-second throughput test and was wrong: the stream **ramps over the first ~30 s**
(16.5 → 18.9 MB/s), so short measurements badly understate the steady rate.
`octobee live` drops the ADC clock to 20 kSPS while it runs (`--fs`) and puts
the original clock back on exit; measured broadband noise is unchanged, and a
hand-passed magnet is a sub-Hz signal anyway.

Short offline captures at the full 200 kSPS are unaffected — the box buffers and
delivers every sample, just slower than real time. A 2 s capture from both boxes
came back with **zero** lost samples.

If a live session is killed rather than closed, the reduced clock is left in
place. Undo it with:

```bash
octobee probe restore        # puts both boxes back to 200 kSPS
octobee probe info           # confirm
```

### Phoebus interaction

Phoebus' *Streaming Capture* button starts a `CONTINUOUS` stream that owns the
data mover; while it runs, port 4210 gives an immediate EOF. All the tools here
stop it automatically. To go back to Phoebus, just press its start button again.

---

## The rest of the documentation

Each of these was a chapter of this file. They are indexed by the
application's Help tab, so anything written here is searchable from
inside the window.

- **[The hardware, and what it is telling us](docs/hardware.md)** — the carriers, the gain chain, the noise you should expect
- **[Calibration procedure](docs/calibration.md)** — zero, magnet pass, roll sweep, and the file they produce
- **[Every file, and what it is for](docs/files.md)** — which module does what, and where configuration lives
- **[The application](docs/gui.md)** — a tour of the window, tab by tab
- **[How sensitive is it really?](docs/sensitivity.md)** — what the noise floor actually is, and what sets it
- **[Long runs and the data rate](docs/logging.md)** — continuous logging, file sizes, and what the box can sustain
- **[The stages, and motorised field maps](docs/stages.md)** — axis maps, homing, the motion profile, and rastering a volume
- **[Measuring inside the coil set](docs/machine.md)** — placing the probe among the windings, and clearance

Reference material that is not ours:
[vendor datasheets and manuals](docs/vendor/), and
[bench notes](docs/notes/).
