# Every file, and what it is for

### Where the repository keeps things

```
src/octobee/     the application, installed with `pip install -e ".[gui,dev]"`
config/          calibration.json, probe_geometry.json, stages.json, machine.json
captures/        recordings, exports and field maps (not in version control)
onbox/           scripts that run on the carriers, not here
docs/            this documentation, the screenshots, and the vendor PDFs
```

Configuration is **located, not just named**. Every tool resolves
`calibration.json` and its neighbours through `octobee/paths.py`, in this
order:

1. `$OCTOBEE_CONFIG_DIR`, if set — the way to keep a second, separate set
2. `config/` in this checkout — what the bench uses
3. the platform's per-user config directory, for an installed copy with no
   checkout (`%APPDATA%\octobee` on Windows)

Captures follow the same pattern under `$OCTOBEE_CAPTURES_DIR`.

This matters more than it sounds. These files were once opened by bare
filename, so what got loaded depended on the working directory the process
happened to start in — and a tool that cannot find `calibration.json` does not
fail, it falls back to built-in ±20 mT defaults and produces numbers that look
entirely reasonable and are uniformly wrong.

### What each file is

| file | runs on | purpose |
|---|---|---|
| `octobee probe` | PC | core library: UUT knobs, frame decode, capture. `info` / `capture` / `restore` CLI |
| `octobee live` | PC | live plot of all 16 sensors, both boxes, one window |
| `octobee report` | PC | health + calibration report, optional PNG |
| `octobee idmap` | PC | proves the channel↔sensor map (SPI sweep or magnet pass) |
| `onbox/sensor_audit.py` | **carrier** | SPI register audit of all 8 chips; `--id-sweep`, `--set-gain` |
| `onbox/gain_config.py` | **carrier** | reads/sets the amplifier gain **permanently** in EEPROM, with a backup and a restore path — see §3 |
| `onbox/set_sample_rate.sh` | **carrier** | report or set the ADC rate, with the aliasing cost stated |
| `onbox/fix_carrier_keys.sh` | **carrier** | repair `authorized_keys` and install the workstation key |
| `octobee/gui/window.py` | PC | the application: live view, 3D probe head, calibration, exports |
| `octobee geometry` | PC | chip positions and per-chip rotation matrices on the tube |
| `octobee/gui/widgets/probe3d.py` | PC | the 3D probe-head widget |
| `octobee calibrate` | PC | counts → tesla, zero, gain trim, pose matrix, channel health |
| `octobee roll` | PC | Earth-field roll calibration: solves per-sensor response, offsets and orientation from hand-rolled sweeps |
| `octobee poses` | PC | records a roll sweep as indexed 90° poses, one full-rate capture each — 11.5× quieter than a hand roll |
| `octobee record` | PC | CSV / raw / report writers |
| `octobee stage` | PC | Thorlabs LTS300C control over the Kinesis C API — no Kinesis app, no pythonnet |
| `octobee scan` | PC | motorised field map: move, settle, average at full rate, repeat |
| `octobee machine` | PC | reads a simsopt coil set, sweeps it into a keep-out volume, and places the probe in it |
| `octobee/gui/widgets/machine3d.py` | PC | the 3D machine widget: coils, stage envelope, and the probe among them |
| `octobee/profile.py` | PC | span timing, event-loop lag and GL renderer detection behind `--profile` |
| `octobee_launch.pyw` | PC | what the desktop icon runs: no console, but startup failures still reported in a native dialog |
| `octobee.ico` | PC | application icon |
| `tests/` | PC | end-to-end verification, offline or against the hardware. 41 tests, 458 checks; the quality gate |
| `pyproject.toml` | PC | packaging and the ruff lint configuration, with a reason beside every rule that is switched off |
| `.github/workflows/checks.yml` | CI | runs `ruff check` and `pytest` on every push |

The command-line tools need only `numpy` and `matplotlib`. The GUI additionally
needs `PyQt6`, `pyqtgraph` and `PyOpenGL` — `pip install -r requirements.txt`.
Nothing else in either case: no HAPI install, no Phoebus, no EPICS.

The stage tools need no Python package at all beyond `numpy` — they call the
Kinesis C API through `ctypes`. They do need Kinesis *installed*, for the DLLs.

---

---

*Part of the [OCTO-BEE documentation](../README.md).*
