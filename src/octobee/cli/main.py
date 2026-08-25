"""
octobee/cli/main.py -- one command, with the instruments under it.

Every tool here used to be a file you ran by path:

    python octobee_cal.py --plot
    python octobee_stage.py moveto x 120

which meant every one of them needed you to know where the checkout was, and
none of them could be run from anywhere else. They are subcommands now:

    octobee report --plot
    octobee stage moveto x 120

`octobee` with no arguments lists them.

Why this dispatches rather than reimplements
--------------------------------------------
Each module keeps its own main() and its own argument parser. They are not
generic -- `stage` speaks in axes and millimetres, `probe` in carriers and
clock dividers, `roll` in bearings and sweeps -- and a single flat parser over
all of them would have to invent a shared vocabulary that does not exist. So
this sets sys.argv and hands over, which also means every module stays runnable
on its own with `python -m octobee.motion.stage` when that is easier.
"""

import importlib
import sys

# subcommand -> (module, one line for the listing)
COMMANDS = {
    "probe":     ("octobee.acq.carrier",
                  "the carriers: info, capture, sample rate, restore"),
    "idmap":     ("octobee.acq.idmap",
                  "prove which ADC channel is which physical sensor"),
    "report":    ("octobee.report",
                  "calibration and health report for all 16 sensors"),
    "live":      ("octobee.live",
                  "live plot of every channel, both carriers, one window"),
    "calibrate": ("octobee.calib.convert",
                  "inspect and edit the calibration itself"),
    "geometry":  ("octobee.calib.geometry",
                  "where each sensor sits, and which way its chip faces"),
    "roll":      ("octobee.calib.roll",
                  "Earth-field roll calibration from recorded sweeps"),
    "poses":     ("octobee.calib.poses",
                  "record a roll sweep as indexed 90-degree poses"),
    "stage":     ("octobee.motion.stage",
                  "drive the Thorlabs stages: list, home, move, stop"),
    "scan":      ("octobee.motion.scan",
                  "map a field over a volume: move, settle, average, repeat"),
    "machine":   ("octobee.machine",
                  "the coil set the probe is measuring inside"),
    "record":    ("octobee.record",
                  "inspect files this application has written"),
}

USAGE = """usage: octobee <command> [options]

  {commands}

  octobee <command> --help   for that command's own options
  octobee-gui                the desktop application

Configuration is read from $OCTOBEE_CONFIG_DIR, or config/ in the checkout.
"""


def _usage():
    width = max(len(c) for c in COMMANDS)
    rows = "\n  ".join(f"{name:<{width}}  {COMMANDS[name][1]}"
                       for name in sorted(COMMANDS))
    return USAGE.format(commands=rows)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0

    name = argv[0]
    if name not in COMMANDS:
        print(f"octobee: no such command: {name}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    module = importlib.import_module(COMMANDS[name][0])
    # The subcommand parses for itself, and its usage line should read the way
    # it was invoked rather than naming a file nobody typed.
    sys.argv = [f"octobee {name}", *argv[1:]]
    # Python 3.14's argparse works out `prog` from __main__.__spec__ when it
    # can, so a console-script or `-m` launch prints "python.exe -m
    # octobee.cli.main" over the top of the name above. Clearing the spec puts
    # it back on sys.argv[0], which is the line the user actually typed.
    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        main_mod.__spec__ = None
    rc = module.main()
    return 0 if rc is None else rc


if __name__ == "__main__":
    sys.exit(main())
