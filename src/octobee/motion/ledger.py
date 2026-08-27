"""
octobee/motion/ledger.py -- what each axis was doing when octobee last looked.

Why this exists
---------------
position_trusted is the most important flag in motion/stage.py, and until this
module existed it did not survive a restart. Trust is granted by a homing cycle
this software watched complete, and withdrawn by everything that could have
cost steps: an immediate stop, an emergency stop, a stall, a motion error, a
move that ended against a hard limit switch. All of that lived in memory only.
Close the window and open it again and Stage.open() re-derived trust from the
controller's HOMED bit alone -- which is exactly what position_trusted's own
docstring says must never be done. That bit records that a homing cycle
finished once, and it stays set through every one of the events above.

So the last thing known about each axis is written here, and open() reads it
back before deciding anything:

    trusted false   the recorded reason is replayed. An emergency stop at
                    eleven last night is still an emergency stop at nine this
                    morning, and the operator gets told which it was.
    trusted true    the recorded count is compared against the count now. If
                    the carriage moved while octobee was not watching -- driven
                    from Kinesis, or the process killed in the middle of a move
                    -- the recorded number no longer describes the axis, and
                    trust is refused rather than inherited.

Keyed by serial number, not by axis name, because trust is a fact about the
physical controller and its carriage. Rename x to y in stages.json, or swap two
controllers over, and each one still carries its own history.

It sits beside the configuration and is not any of it: stages.json describes a
rig and belongs in the repository, where this describes one machine at one
moment and is meaningless on another bench. It is in .gitignore for that
reason, and losing it costs one homing cycle rather than a setting.

What it cannot see
------------------
An LTS300C counts steps and has no position feedback. De-energise the driver,
push the carriage by hand, and the count does not change -- so a match here
proves the CONTROLLER was not commanded anywhere, not that the carriage did not
move. Nothing in software can close that gap; only a homing cycle can. It is
the reason this module reports a reason rather than a verdict, and the reason
the GUI still offers to home an axis it has just decided to trust.

Failing safe
------------
Every unreadable, missing, corrupt or stale case returns "not trusted" with a
sentence saying which. The cost of a false distrust is one homing cycle; the
cost of a false trust is an absolute move computed from a number that does not
describe the machine.
"""

import json
import os
import tempfile
import threading
import time

from octobee import paths

__all__ = [
    "MIN_WRITE_INTERVAL_S",
    "STATE_FILE",
    "TRUST_TOLERANCE_MM",
    "forget",
    "recall",
    "remember",
    "state_path",
    "verdict",
]

STATE_FILE = "stage_state.json"

# How far the counter may differ from the recorded one and still be believed.
#
# It should be exactly zero: a powered controller that is commanded nowhere
# reports the same count it did an hour ago. A micron of slack is here so that
# a controller which reports one device unit of jitter (2.4 nm on an LTS300C)
# cannot cost a five-minute homing cycle, and it is three orders of magnitude
# below the +/-47 um the stage is accurate to in the first place -- so nothing
# that matters to a field map can hide underneath it.
TRUST_TOLERANCE_MM = 0.001

# The shortest gap between two ledger writes that merely record a completed
# move. Stage.wait() records after every one of them and a raster is thousands,
# so this is what keeps a scan loop from carrying a file write per point. It
# bounds staleness, not correctness: a count a second out of date fails safe
# into a distrusted axis at the next open(). Anything that CHANGES the trust
# ignores it entirely.
MIN_WRITE_INTERVAL_S = 1.0

# One process, three axes, and StageSet.move_to starts all of them before
# waiting for any: the write below is read-modify-write over a shared file, so
# without this the axis that finishes second reads the file the axis that
# finished first had not written yet, and drops its entry.
_LOCK = threading.Lock()


def state_path(path=None):
    """Where the ledger lives, unless the caller says otherwise."""
    return path or paths.config(STATE_FILE)


def _read(path):
    """The whole file as a dict of serial -> entry. Missing is empty."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    stages = doc.get("stages")
    return stages if isinstance(stages, dict) else {}


def recall(serial, path=None):
    """What was last recorded about this serial, or None.

    A file that has been corrupted -- half-written by a machine that lost power
    mid-save, or edited by hand into invalid JSON -- reads as "nothing known"
    rather than raising. The caller's response to None is to distrust the axis,
    which is the right answer for a damaged ledger too, and a traceback out of
    open() would take the whole connection down over a file whose only job is
    to make the machine more careful.
    """
    try:
        return _read(state_path(path)).get(str(serial))
    except (OSError, ValueError):
        return None


def remember(serial, *, axis=None, count_du=None, position_dev_mm=None,
             trusted=False, reason="", path=None):
    """Record the state of one axis, leaving the other axes' entries alone.

    Written through a temporary file and os.replace, which is atomic on both
    platforms this runs on. A half-written ledger is worse than no ledger: it
    would be read back as a plausible count for an axis that is somewhere else.
    """
    target = state_path(path)
    entry = {
        "axis": axis,
        "count_du": None if count_du is None else int(count_du),
        "position_dev_mm": (None if position_dev_mm is None
                            else round(float(position_dev_mm), 6)),
        "trusted": bool(trusted),
        "reason": str(reason or ""),
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with _LOCK:
        try:
            doc = {"stages": _read(target)}
        except (OSError, ValueError):
            # Unreadable, so it is being replaced wholesale. The alternative is
            # to refuse to write, which leaves the damaged file in place and
            # every axis permanently unable to record anything.
            doc = {"stages": {}}
        doc["stages"][str(serial)] = entry
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(target)),
                                   prefix=".stage_state-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, indent=2)
                fh.write("\n")
            os.replace(tmp, target)
        except BaseException:
            # Otherwise a failing write leaves one orphan per attempt in the
            # config directory, looking like debris nobody owns.
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
    return entry


def forget(serial=None, path=None):
    """Drop one axis's record, or the whole ledger.

    Only for tests and for a bench that has genuinely changed hardware. It is
    not an operator remedy for a distrusted axis -- the remedy for that is to
    home it, which is the one action that actually establishes the thing the
    ledger is recording.
    """
    target = state_path(path)
    with _LOCK:
        try:
            stages = _read(target)
        except (OSError, ValueError):
            stages = {}
        if serial is None:
            stages = {}
        else:
            stages.pop(str(serial), None)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump({"stages": stages}, fh, indent=2)
            fh.write("\n")


def verdict(entry, count_du, du_per_mm, tolerance_mm=TRUST_TOLERANCE_MM):
    """(trusted, why not) for a recorded entry against the count right now.

    Pure, and separate from Stage, so the rule that decides whether an absolute
    move is allowed after a restart can be checked without a stage attached --
    the same reason Stage._resolve_limit is a staticmethod.

    `count_du` is the raw device count, in the controller's own frame. Device
    units rather than rig millimetres deliberately: the rig frame is a fact
    about the bracket, held in stages.json, and someone editing origin_mm or
    invert between two sessions must not be able to make an axis look as if it
    had moved -- or, worse, make one that did look as if it had not.
    """
    if not entry:
        return False, ("octobee has no record of this axis, so nothing here "
                       "has ever watched it home")
    when = str(entry.get("at") or "").strip()
    stamp = f" (recorded {when})" if when else ""
    if not entry.get("trusted"):
        was = str(entry.get("reason") or "").strip()
        return False, ((was + stamp) if was else
                       "the last session left this axis untrusted" + stamp)
    was_du = entry.get("count_du")
    if was_du is None or du_per_mm <= 0:
        return False, ("the recorded position cannot be read back, so there "
                       "is nothing to compare the counter against" + stamp)
    try:
        moved_mm = (int(count_du) - int(was_du)) / float(du_per_mm)
    except (TypeError, ValueError):
        return False, ("the recorded position cannot be read back, so there "
                       "is nothing to compare the counter against" + stamp)
    if abs(moved_mm) > tolerance_mm:
        return False, (f"the counter has moved {moved_mm:+.3f} mm since "
                       f"octobee last saw this axis{stamp}, so something drove "
                       f"it while nothing was watching")
    return True, ""
