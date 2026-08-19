#!/usr/bin/env python3
"""
onbox_sensor_audit.py -- RUNS ON THE ACQ1001 CARRIER, not on the PC.

Reads every SENM3Dx register that affects sensitivity, offset and calibration,
for all 8 sensors on this carrier, and prints them as one table so that any
chip which is set up differently from its neighbours is obvious at a glance.

This is the only way to answer "are the sensors properly calibrated?" -- gain,
bias current and EEPROM calibration status live inside the ASIC and are reachable
only over SPI, from the carrier itself.

Copy it over and run it:
    scp onbox_sensor_audit.py root@acq1001_694:/tmp/
    ssh root@acq1001_694 'python3 /tmp/onbox_sensor_audit.py'
    ssh root@acq1001_695 'python3 /tmp/onbox_sensor_audit.py'   # repeat per box

Options:
    --id-sweep [SECONDS]   walk sensor 1..8, muting each one's Bx/By/Bz in turn,
                           so a host-side capture can prove which ACQ423 channels
                           belong to which sensor. Pair with octobee_idmap.py.
    --set-gain GAIN        write the same gain (15/150/1500/3000) to all 8 chips.
                           Register-only: lost at power cycle unless you also
                           commit it to EEPROM. USE DELIBERATELY.
"""

import argparse
import sys
import time

sys.path.append("/usr/local/CARE")

import ThreeDHALLInterface as tdhi          # noqa: E402

BUS = 1
BOARD = 0x20            # OCTO-BEE is "Site 2" -> selector 0x20 + (position-1)
NSENS = 8

# SENM3Dx v2 datasheet, Table 14
REGS = [
    ("PWM_CTRL", 0x08), ("CHANNEL_CTRL", 0x09), ("OSC_trim", 0x0A),
    ("THRES_X", 0x0B), ("THRES_Z", 0x0C), ("THRES_Y", 0x0D),
    ("G_CTRL_X", 0x0E), ("G_CTRL_Z", 0x0F), ("G_CTRL_Y", 0x10),
    ("DAC_X", 0x11), ("DAC_Z", 0x12), ("DAC_Y", 0x13),
    ("SENS_X", 0x14), ("SENS_Z", 0x15), ("SENS_Y", 0x16),
    ("SENS_TC_X", 0x17), ("SENS_TC_Z", 0x18), ("SENS_TC_Y", 0x19),
    ("OFFSET_X", 0x1A), ("OFFSET_Z", 0x1B), ("OFFSET_Y", 0x1C),
    ("OFFSET_TC_X", 0x1D), ("OFFSET_TC_Z", 0x1E), ("OFFSET_TC_Y", 0x1F),
]

GAIN_BITS = {0: 3000, 1: 1500, 2: 150, 3: 15}       # G_CTRL bits 1:0, Table 22
GAIN_RANGE_MT = {3000: 20, 1500: 40, 150: 400, 15: 4000}


def open_sensor(pos):
    """pos is 1..8. Returns an interface, or None if the chip does not answer."""
    dx = tdhi.ThreeDHALLInterface()
    dx.openSPI(BUS, BOARD + pos - 1)
    dx.configureSPI("976kHz")
    if not dx.verify_spi_interface(100):
        return None
    return dx


def decode_gain(g_ctrl):
    """G_CTRL register byte -> nominal gain, honouring the 2-bit/4-bit select."""
    if g_ctrl & 0x08:                                # bit 3: 4-bit selection
        pre = 1 if (g_ctrl >> 4) & 1 else 10
        main = {3: 2.5, 1: 5, 0: 10}.get((g_ctrl >> 5) & 3, float("nan"))
        rec = 3 if (g_ctrl >> 7) & 1 else 30
        return pre * main * rec
    return GAIN_BITS[g_ctrl & 0x03]


def audit():
    print(f"{'':14s}" + "".join(f"{'S'+str(p):>8s}" for p in range(1, NSENS + 1)))
    devs = {}
    for pos in range(1, NSENS + 1):
        try:
            devs[pos] = open_sensor(pos)
        except Exception as e:                       # noqa: BLE001
            print(f"  sensor {pos}: openSPI failed: {e}", file=sys.stderr)
            devs[pos] = None

    def row(name, fn, fmt="{:>8}"):
        cells = []
        for pos in range(1, NSENS + 1):
            dx = devs[pos]
            if dx is None:
                cells.append(f"{'--':>8}")
                continue
            try:
                cells.append(fmt.format(fn(dx)))
            except Exception:                        # noqa: BLE001
                cells.append(f"{'ERR':>8}")
        print(f"{name:14s}" + "".join(cells))

    print("-" * (14 + 8 * NSENS))
    row("SPI ok", lambda d: "yes")
    row("STATUS", lambda d: f"0x{d.read_reg_val(reg_address=0x3F):02x}")
    row("EE key", lambda d: f"0x{d.read_EEPROM_key_byte_reg():02x}")
    row("EE cksum", lambda d: f"0x{d.read_EEPROM_check_sum_reg():02x}")
    row("EE valid", lambda d: "YES" if d.verify_EEPROM_data_for_config() else "NO")
    row("temp [C]", lambda d: f"{d.get_temperature():.1f}")
    print("-" * (14 + 8 * NSENS))
    row("gain (api)", lambda d: d.get_gain())
    row("range [mT]", lambda d: GAIN_RANGE_MT.get(d.get_gain(), "?"))
    print("-" * (14 + 8 * NSENS))
    for name, addr in REGS:
        row(name, lambda d, a=addr: d.read_reg_val(reg_address=a))
    print("-" * (14 + 8 * NSENS))
    row("gain X", lambda d: decode_gain(d.read_reg_val(reg_address=0x0E)))
    row("gain Z", lambda d: decode_gain(d.read_reg_val(reg_address=0x0F)))
    row("gain Y", lambda d: decode_gain(d.read_reg_val(reg_address=0x10)))

    # ---- verdict --------------------------------------------------------
    print()
    live = [p for p, d in devs.items() if d is not None]
    if len(live) < NSENS:
        print(f"!! sensors not answering on SPI: "
              f"{[p for p in range(1, NSENS+1) if p not in live]}")

    def spread(label, fn):
        vals = {}
        for p in live:
            try:
                vals[p] = fn(devs[p])
            except Exception:                        # noqa: BLE001
                pass
        uniq = set(vals.values())
        if len(uniq) > 1:
            print(f"!! {label} DIFFERS between chips: " +
                  ", ".join(f"S{p}={v}" for p, v in sorted(vals.items())))
        elif uniq:
            print(f"   {label} identical on all responding chips: {uniq.pop()}")

    spread("amplifier gain", lambda d: d.get_gain())
    spread("EEPROM calibration valid", lambda d: d.verify_EEPROM_data_for_config())
    for nm, a in (("DAC_X (Bx bias)", 0x11), ("DAC_Z (Bz bias)", 0x12),
                  ("DAC_Y (By bias)", 0x13)):
        spread(nm, lambda d, aa=a: d.read_reg_val(reg_address=aa))
    for nm, a in (("SENS_X", 0x14), ("SENS_Z", 0x15), ("SENS_Y", 0x16)):
        spread(nm + " (fine gain)", lambda d, aa=a: d.read_reg_val(reg_address=aa))

    print("\nAny line starting '!!' is a reason the same magnet can give different")
    print("amplitudes on different sensors. Gain and DAC_* set sensitivity directly;")
    print("EEPROM invalid means the factory calibration is NOT loaded on that chip.")
    return devs


def id_sweep(devs, dwell):
    """
    Mute each sensor's three field channels in turn so a host capture can prove
    the channel map. Sensors are swept in position order 1..8, dwell seconds each.
    """
    print(f"\n--- ID sweep: {dwell:.1f} s per sensor, positions 1..{NSENS}")
    print("Start octobee_idmap.py on the PC now, then leave this running.")
    sys.stdout.flush()
    time.sleep(3.0)
    for pos in range(1, NSENS + 1):
        dx = devs.get(pos)
        if dx is None:
            print(f"  sensor {pos}: not present, skipping")
            sys.stdout.flush()
            time.sleep(dwell)
            continue
        print(f"  sensor {pos}: channels OFF")
        sys.stdout.flush()
        dx.change_channel_state(state=["off", "off", "off", "on"],
                                channel=["x", "y", "z", "t"])
        time.sleep(dwell)
        dx.change_channel_state(state=["on", "on", "on", "on"],
                                channel=["x", "y", "z", "t"])
        print(f"  sensor {pos}: channels back ON")
        sys.stdout.flush()
        time.sleep(dwell)
    print("--- sweep done, all channels restored")


def set_gain_all(devs, gain):
    print(f"\n--- writing gain {gain} to every responding chip")
    for pos in range(1, NSENS + 1):
        dx = devs.get(pos)
        if dx is None:
            continue
        dx.set_amplifier_gain(data=gain)
        time.sleep(0.1)
        print(f"  sensor {pos}: gain now {dx.get_gain()}")
    print("Registers only -- this is lost at power cycle. To make it permanent")
    print("write the same values to EEPROM and call activate_EEPROM_config().")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--id-sweep", nargs="?", type=float, const=4.0, default=None,
                   metavar="SECONDS", help="run the channel identification sweep")
    p.add_argument("--set-gain", type=int, default=None,
                   choices=(15, 150, 1500, 3000),
                   help="write this gain to all 8 chips (deliberate action)")
    a = p.parse_args()

    devs = audit()
    if a.set_gain is not None:
        set_gain_all(devs, a.set_gain)
        print("\nre-auditing after the write:")
        devs = audit()
    if a.id_sweep is not None:
        id_sweep(devs, a.id_sweep)


if __name__ == "__main__":
    main()
