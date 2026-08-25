"""
OCTO-BEE -- control, live view, calibration and field mapping for the
16-sensor Hall probe.

The package is laid out along the path a measurement actually takes:

    acq/      talking to the ACQ1001 carriers: stream, decode, channel maps
    calib/    turning counts into field, and keeping the numbers that do it
    motion/   the Thorlabs stages, and moving the probe through a volume
    machine   the coil set the probe is measuring inside
    record    getting data onto disk
    gui/      everything Qt, and nothing but

Nothing below gui/ imports Qt, which is what makes the command-line tools and
the tests runnable without a display.
"""

__version__ = "0.1.0"
