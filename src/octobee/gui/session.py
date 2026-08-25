"""
octobee/gui/session.py -- the state more than one tab needs to see.

Why this exists
---------------
MainWindow held 285 attributes, and every tab reached into all of them. That
is what made adding a control to the Stages tab an edit to the same object
that owns the field map, the calibration and the acquisition tick: there was
no such thing as a local change, so every change carried the whole
application's risk.

Splitting the window into tabs only helps if there is somewhere for the shared
state to go. Without it, eight tab modules that each hold a reference to the
window are just the same god class spread over eight files, with the coupling
made harder to see rather than easier.

So this is that somewhere, and it is deliberately small. Measured across the
window's twelve method groups, 73 attributes were touched by exactly one group
-- those belong to a tab and moved with it. What is left here is the state that
genuinely crosses tab boundaries:

    the instrument   source, hosts, out_rate, prev_clkdiv
    the numbers      geom, cal, machine, coils
    the hardware     stages
    the run          roll, collecting, csv_rec, raw_rec, out_dir
    the results      last_health, magnet_peaks, sweeps, pose_solution
    the tooling      prof, lag, log

Widgets are NOT in here. A tab that wants another tab's widget to change says
so with a signal; reaching across to set someone else's label is the coupling
this is meant to remove, not relocate.

Configuration loading lives here too, because "what did we load, and did it
work" is the first thing several tabs ask and it needs one answer.
"""

import os

from octobee.acq import carrier as ob
from octobee.calib import convert as ocal
from octobee.calib import geometry as pgeom
from octobee import machine as omach
from octobee import paths
from octobee import profile as oprof


class Session:
    """One run of the application: what is connected, and what it knows."""

    def __init__(self, args):
        self.args = args
        self.hosts = list(args.uut) if args.uut else list(ob.DEFAULT_UUTS)
        self.out_dir = getattr(args, "out_dir", None) or paths.captures_dir()

        # Collected rather than logged: the log pane does not exist yet, and a
        # config that failed to parse is the first thing the user must be told.
        self.config_errors = []
        err = self.config_errors.append

        self.geom = pgeom.Geometry.load_or_default(args.geometry, on_error=err)
        self.cal = ocal.Calibration.load_or_default(args.calibration,
                                                    on_error=err)
        # "Present and readable", not merely "present" -- a file that exists
        # and failed to parse must not be reported as loaded.
        self.cal_from_file = (os.path.exists(args.calibration)
                              and not self.config_errors)

        # The machine the probe is measuring inside: which coils, which of them
        # are live, and where the head sits among them. Loaded before the UI so
        # the tab opens showing the last session's placement rather than the
        # origin -- a pose is measured with a tape, and re-entering it every
        # morning is how it drifts.
        self.machine = omach.MachineConfig.load_or_default(args.machine,
                                                           on_error=err)
        self.coils = omach.CoilSet.load_or_none(self.machine.coil_file,
                                                on_error=err)
        for lost in self.machine.adopt(self.coils):
            err(f"{args.machine}: {lost}")

        # ---- the instrument ------------------------------------------------
        self.source = None
        self.prev_clkdiv = {}
        self.out_rate = 500.0

        # ---- the run -------------------------------------------------------
        self.roll = None                # the live ring buffer, set by the shell
        self.collecting = None          # 'tare' | 'magnet' | 'sweep'
        self.csv_rec = None
        self.raw_rec = None

        # ---- results other tabs read ---------------------------------------
        self.sweeps = {}                # tag -> roll sweep
        self.pose_solution = None
        self.magnet_peaks = None
        self.last_health = None
        self.probe_cloud = None

        # ---- hardware ------------------------------------------------------
        self.stages = None

        # ---- tooling -------------------------------------------------------
        self.prof = oprof.Profiler(enabled=bool(getattr(args, "profile", False)))
        self.lag = oprof.LagMonitor(interval_ms=100)

        self._log_pane = None

    # ---- logging ----------------------------------------------------------
    #
    # A callable rather than the widget itself, so that everything which needs
    # to say what it did can do so without also being able to reach into the
    # Log tab and clear it.

    def attach_log(self, pane):
        """Point logging at the Log tab, once the UI exists."""
        self._log_pane = pane
        return pane

    def log(self, msg):
        if self._log_pane is not None:
            self._log_pane.log(msg)
