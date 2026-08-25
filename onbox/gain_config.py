#!/usr/bin/env python3
"""
gain_config.py -- RUNS ON THE ACQ1001 CARRIER, not on the PC.

Reads, changes and permanently stores the SENM3Dx amplifier gain for all 8
sensors on this carrier, with a full EEPROM backup first and a restore path.

Why the gain is stored where it is
----------------------------------
The gain is a 2-bit field in registers G_CTRL_X/Z/Y (0x0E/0x0F/0x10):

    00 -> gain 3000 (+/-20 mT)      10 -> gain 150  (+/-400 mT)
    01 -> gain 1500 (+/-40 mT)      11 -> gain 15   (+/-4000 mT)

At power-up those registers are loaded from the EEPROM byte EGain_sel at
EEPROM address 0x10E (datasheet Table 29):

    bits 1:0 = X gain sel    bits 3:2 = Z gain sel    bits 5:4 = Y gain sel

so writing the registers alone lasts until the next power cycle; writing
EGain_sel makes it permanent. After any EEPROM change the key (0x1FE) and
checksum (0x1FF) must be rewritten with activate_EEPROM_config(), otherwise the
chip decides the EEPROM is invalid and silently falls back to datasheet
defaults -- losing the factory calibration.

Crucially, the EEPROM holds a SEPARATE factory calibration data set per gain
(datasheet section 10, Table 31):

    gain 3000 -> "Gain 0 data set", EEPROM 0x140-0x14F
    gain 1500 -> "Gain 1 data set", EEPROM 0x150-0x15F
    ... and so on

Each set holds that gain's own DAC_*, SENS_*, OFFSET_* and TC trims. The 2-bit
gain value selects which set is loaded. So changing gain does NOT throw the
calibration away -- the chip loads the trims the factory measured for the new
gain. That is why this is a safe change to make.

Usage
-----
    python3 gain_config.py show
    python3 gain_config.py backup --out /tmp/eeprom_backup.json
    python3 gain_config.py set --gain 3000       # registers + EEPROM
    python3 gain_config.py verify --gain 3000
    python3 gain_config.py restore --in /tmp/eeprom_backup.json
"""

import argparse
import json
import sys
import time

for _p in ("/usr/local/senm3dx", "/usr/local/CARE"):
    if _p not in sys.path:
        sys.path.append(_p)

import ThreeDHALLInterface as tdhi          # noqa: E402

BUS = 1
BOARD = 0x20            # OCTO-BEE "Site 2" -> selector 0x20 + (position-1)
NSENS = 8

EEPROM_SIZE = 256       # 0x100..0x1FF, addressed as offsets 0x00..0xFF
ADDR_EGAIN_SEL = 0x0E   # EEPROM 0x10E
ADDR_KEY = 0xFE         # EEPROM 0x1FE
ADDR_CKSUM = 0xFF       # EEPROM 0x1FF
KEY_VALID = 0xA5

# These duplicate octobee.GAIN_TO_RANGE on the PC side, and they have to:
# this script runs ON THE CARRIER, where the rest of the codebase is not
# installed. selftest.py's test_gain_tables() parses both copies and fails if
# they ever disagree -- the drift these tables can cause is a silent 10x on
# every field number, which is not something to leave to memory.
GAIN_SEL = {3000: 0, 1500: 1, 150: 2, 15: 3}          # datasheet Table 22
SEL_GAIN = {v: k for k, v in GAIN_SEL.items()}
GAIN_RANGE_MT = {3000: 20, 1500: 40, 150: 400, 15: 4000}
GAIN_VPT = {3000: 63.0, 1500: 34.65, 150: 3.78, 15: 0.378}

# per-gain calibration data set base address, as an EEPROM offset
DATASET_BASE = {0: 0x40, 1: 0x50, 2: 0x60, 3: 0x70}


def egain_byte(gain):
    """Build EGain_sel with the same gain on X, Z and Y."""
    s = GAIN_SEL[gain]
    return s | (s << 2) | (s << 4)


def decode_egain(byte):
    """EGain_sel -> (gain_x, gain_z, gain_y)."""
    return (SEL_GAIN.get(byte & 3), SEL_GAIN.get((byte >> 2) & 3),
            SEL_GAIN.get((byte >> 4) & 3))


def open_sensor(pos, verify=False):
    """pos is 1..8. Returns an interface, or None if the chip does not answer."""
    try:
        dx = tdhi.ThreeDHALLInterface()
        dx.openSPI(BUS, BOARD + pos - 1)
        dx.configureSPI("976kHz")
        if verify and not dx.verify_spi_interface(20):
            return None
        dx.read_reg_val(reg_address=0x3F)             # cheap liveness probe
        return dx
    except Exception:
        return None


def open_all():
    devs = {}
    for pos in range(1, NSENS + 1):
        devs[pos] = open_sensor(pos)
        if devs[pos] is None:
            print(f"  !! sensor {pos} does not answer on SPI", file=sys.stderr)
    return devs


def cmd_show(devs):
    print(f"{'pos':>4} {'G_CTRL X/Z/Y':>14} {'gain':>6} {'range':>9} "
          f"{'V/T':>7} {'EGain_sel':>10} {'EEPROM says':>16} {'key':>6} {'valid':>6}")
    for pos in range(1, NSENS + 1):
        d = devs.get(pos)
        if d is None:
            print(f"{pos:>4} {'--':>14}")
            continue
        g = [d.read_reg_val(reg_address=a) for a in (0x0E, 0x0F, 0x10)]
        gain = d.get_gain()
        eg = d.read_EEPROM_reg_val(reg_address=ADDR_EGAIN_SEL)
        key = d.read_EEPROM_key_byte_reg()
        ok = d.verify_EEPROM_data_for_config()
        ex, ez, ey = decode_egain(eg)
        print(f"{pos:>4} {g!s:>14} {gain:>6} "
              f"{'+/-'+str(GAIN_RANGE_MT.get(gain,'?'))+'mT':>9} "
              f"{GAIN_VPT.get(gain, float('nan')):>7.2f} "
              f"{'0x%02x' % eg:>10} {f'{ex}/{ez}/{ey}':>16} "
              f"{'0x%02x' % key:>6} {ok!s:>6}")


def cmd_backup(devs, path):
    """Dump every chip's whole EEPROM to JSON. Do this before any write."""
    out = {}
    for pos in range(1, NSENS + 1):
        d = devs.get(pos)
        if d is None:
            continue
        t0 = time.time()
        data = list(d.read_EEPROM_data(start_addr=0x00, num_bytes=EEPROM_SIZE))
        out[str(pos)] = data
        print(f"  sensor {pos}: {len(data)} bytes in {time.time()-t0:.1f}s "
              f"(EGain_sel=0x{data[ADDR_EGAIN_SEL]:02x}, "
              f"key=0x{data[ADDR_KEY]:02x}, cksum=0x{data[ADDR_CKSUM]:02x})")
        sys.stdout.flush()
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {path} ({len(out)} sensors)")


def cmd_set(devs, gain, skip_eeprom=False):
    want_sel = egain_byte(gain)
    print(f"target gain {gain} -> +/-{GAIN_RANGE_MT[gain]} mT, "
          f"{GAIN_VPT[gain]} V/T, EGain_sel=0x{want_sel:02x}")
    for pos in range(1, NSENS + 1):
        d = devs.get(pos)
        if d is None:
            print(f"  sensor {pos}: absent, skipped")
            continue
        before = d.get_gain()
        d.set_amplifier_gain(data=gain)               # registers, all 3 channels
        time.sleep(0.15)
        after = d.get_gain()
        line = f"  sensor {pos}: gain {before} -> {after}"

        if not skip_eeprom:
            have = d.read_EEPROM_reg_val(reg_address=ADDR_EGAIN_SEL)
            if have != want_sel:
                d.write_EEPROM_reg(reg_address=ADDR_EGAIN_SEL,
                                   data_to_write=want_sel)
                time.sleep(0.05)
                got = d.read_EEPROM_reg_val(reg_address=ADDR_EGAIN_SEL)
                line += f", EGain_sel 0x{have:02x} -> 0x{got:02x}"
                if got != want_sel:
                    line += "  !! WRITE DID NOT STICK"
            else:
                line += f", EGain_sel already 0x{have:02x}"
            # rewrite key + checksum so the chip still trusts its EEPROM
            d.activate_EEPROM_config()
            time.sleep(0.05)
            line += f", EEPROM valid={d.verify_EEPROM_data_for_config()}"
        print(line)
        sys.stdout.flush()


def cmd_verify(devs, gain):
    want_sel = egain_byte(gain)
    bad = []
    print(f"{'pos':>4} {'gain':>6} {'EGain_sel':>10} {'key':>6} {'valid':>6} "
          f"{'DAC Z/X/Y':>14} {'SENS Z/X/Y':>14}  verdict")
    for pos in range(1, NSENS + 1):
        d = devs.get(pos)
        if d is None:
            print(f"{pos:>4}  absent")
            bad.append(pos)
            continue
        g = d.get_gain()
        eg = d.read_EEPROM_reg_val(reg_address=ADDR_EGAIN_SEL)
        key = d.read_EEPROM_key_byte_reg()
        ok = d.verify_EEPROM_data_for_config()
        # DAC/SENS now in the registers should be the new gain's data set
        dac = [d.read_reg_val(reg_address=a) for a in (0x12, 0x11, 0x13)]
        sens = [d.read_reg_val(reg_address=a) for a in (0x15, 0x14, 0x16)]
        good = (g == gain and eg == want_sel and key == KEY_VALID and ok)
        if not good:
            bad.append(pos)
        print(f"{pos:>4} {g:>6} {'0x%02x' % eg:>10} {'0x%02x' % key:>6} "
              f"{ok!s:>6} {dac!s:>14} {sens!s:>14}  "
              f"{'OK' if good else 'MISMATCH'}")
    print()
    if bad:
        print(f"!! not correctly set: sensors {bad}")
    else:
        print(f"all {NSENS} sensors at gain {gain} (+/-{GAIN_RANGE_MT[gain]} mT, "
              f"{GAIN_VPT[gain]} V/T), stored in EEPROM and marked valid.")
        print("This now survives a power cycle.")
    return not bad


def cmd_restore(devs, path):
    """Put every EEPROM byte back exactly as the backup found it."""
    with open(path) as f:
        saved = json.load(f)
    for pos in range(1, NSENS + 1):
        d = devs.get(pos)
        ref = saved.get(str(pos))
        if d is None or ref is None:
            continue
        cur = list(d.read_EEPROM_data(start_addr=0x00, num_bytes=EEPROM_SIZE))
        diff = [i for i in range(EEPROM_SIZE) if cur[i] != ref[i]]
        for i in diff:
            d.write_EEPROM_reg(reg_address=i, data_to_write=ref[i])
            time.sleep(0.02)
        print(f"  sensor {pos}: restored {len(diff)} byte(s) {diff[:8]}"
              f"{'...' if len(diff) > 8 else ''}")
        sys.stdout.flush()
    print("restore complete -- power cycle the crate to reload the registers")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    b = sub.add_parser("backup")
    b.add_argument("--out", default="/tmp/eeprom_backup.json")
    s = sub.add_parser("set")
    s.add_argument("--gain", type=int, required=True, choices=sorted(GAIN_SEL))
    s.add_argument("--registers-only", action="store_true",
                   help="do not touch the EEPROM (change lost at power cycle)")
    v = sub.add_parser("verify")
    v.add_argument("--gain", type=int, required=True, choices=sorted(GAIN_SEL))
    r = sub.add_parser("restore")
    r.add_argument("--in", dest="inp", required=True)
    a = p.parse_args()

    devs = open_all()
    if a.cmd == "show":
        cmd_show(devs)
    elif a.cmd == "backup":
        cmd_backup(devs, a.out)
    elif a.cmd == "set":
        cmd_set(devs, a.gain, a.registers_only)
    elif a.cmd == "verify":
        sys.exit(0 if cmd_verify(devs, a.gain) else 1)
    elif a.cmd == "restore":
        cmd_restore(devs, a.inp)


if __name__ == "__main__":
    main()
