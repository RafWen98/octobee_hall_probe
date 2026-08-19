#!/usr/bin/python

# for use with OCTOBEE

import sys
import time
import ThreeDHALLInterface as tdhi

if __name__ == "__main__":

    # always use SPI1 bus.
    # OCTOBEE is always "Site 2", so the site selector is 0x20
    BUS = 1
    BOARD = 0x20
    SPI_CLOCK = "976kHz"

    # device selection argument handling
    try:
        devnum = int(sys.argv[1])
    except (IndexError, ValueError):
        devnum = -1

    while devnum < 0 or devnum > 7:
        try:
            devnum = int(input("Invalid value. Enter an integer between 0 and 7: "))
        except ValueError:
            print("That is not an integer, try again.")
            devnum = -1
    print(f"Selected device {devnum}")

    dx = tdhi.ThreeDHALLInterface()
    dx.openSPI(BUS, BOARD+devnum)
    dx.configureSPI(SPI_CLOCK)
    rc = dx.verify_spi_interface(100)
    try:
        gain_setting = sys.argv[2]
    except IndexError:
        gain_setting = input("input a gain value (15, 150, 1500, 3000)")
    gain_setting = int(gain_setting)
    if rc:
        print(f"gain is currently {dx.get_gain()}")
        try:
            print(f"gain to be set to {gain_setting}")
            dx.set_amplifier_gain(gain_setting)
        except (KeyError, IndexError) as e:
            print(f"error: {e}")
            new_input = input("Not given a valid gain setting. Enter 15, 150, 1500 or 3000")
            gain_setting = int(new_input)
            time.sleep(1)
        print(f"gain is now {dx.get_gain()}")
    else:
        print(f'WARNING: device {devnum} not found')

