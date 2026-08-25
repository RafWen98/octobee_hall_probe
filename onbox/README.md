# onbox — code that runs on the carrier, not on the PC

Everything in this directory runs on an **ACQ1001 carrier** (`acq1001_694`,
`acq1001_695`), not on the workstation that runs the OCTO-BEE application. That
is the whole reason it is a separate directory: different machine, different
Python, different lifecycle, and a vendor library (`acq400_hapi`, the SPI
bindings) that is not installed here and cannot be.

Nothing in `src/octobee/` imports any of it, and nothing here imports the
application.

## What is here

| path | what it does |
| --- | --- |
| `sensor_audit.py` | SPI register audit of all 8 chips on a box; `--id-sweep`, `--set-gain` |
| `gain_config.py` | reads and sets the amplifier gain **permanently in EEPROM**, with a backup and a restore path |
| `set_sample_rate.sh` | report or set the ADC sample rate, with the aliasing cost stated |
| `fix_carrier_keys.sh` | install an SSH public key on a carrier that has lost it |
| `rc.user/` | captured copies of each box's `/mnt/local/rc.user` boot script |
| `eeprom/` | EEPROM gain backups taken before the first permanent gain change |

## How to run any of it

These are copied to the carrier and run there by their bare name:

```sh
scp onbox/sensor_audit.py root@acq1001_694:/tmp/
ssh root@acq1001_694 'python3 /tmp/sensor_audit.py'
ssh root@acq1001_695 'python3 /tmp/sensor_audit.py'   # repeat per box
```

## Why ruff only half-checks this directory

`sensor_audit.py` and `gain_config.py` are linted for style and obvious errors,
but they cannot be *import*-checked from the workstation, because the vendor
modules they import only exist on the carrier.

The part of them that matters most is checked anyway. Both files duplicate
octobee's gain tables, and a drift between the two would silently rescale every
field the probe reports. `selftest.test_gain_tables()` reads the tables straight
out of these files and compares them against octobee's own `GAIN_TO_RANGE`.

`rc.user/` is excluded from linting entirely — it is captured vendor boot
scripts, not code this project maintains.
