"""
octobee/gui/preflight.py -- is the rig ready to be left alone with a volume map?

Why this exists
---------------
A volume map runs for hours with nobody in the room, and every one that has
gone wrong so far went wrong for a reason that was already true when Start was
pressed: two sensors dead since the morning, a stream that had been dropping
rows, a plan made for a pose that had since been nudged, encoders that were
never calibrated. Each of those was on screen somewhere -- on a tab that was
not the one being looked at.

So the questions are asked in one place, and the answers are three-valued:

    ok      nothing to do
    warn    the map will run, and will be worse for it; say so and carry on
    fail    the map must not start

The distinction is deliberate. S16 was dead for months on this probe and maps
were still worth taking, so a dead sensor warns; an axis whose position cannot
be believed sends the head somewhere other than the plan says, so it fails.

No Qt in here: it reads the session and returns a list, so it can be tested
without a window and shown by more than one widget.
"""

import shutil
from collections import namedtuple

OK, WARN, FAIL = "ok", "warn", "fail"

Check = namedtuple("Check", "name state detail")

# What one logged row costs on disk, roughly: 16 sensors x 3 axes as float32,
# the position in two frames, a line index and a time. The saved file is
# compressed and comes out smaller; this is the ceiling, which is the right
# number to hold free space against.
BYTES_PER_ROW = 280
# Below this much free space something else on the disk is about to fail
# whether or not the map fits.
LOW_DISK_BYTES = 2e9


def sensor_summary(session):
    """(state, text) for the probe's sixteen sensors, as the health scan sees them.

    Shared by the checklist and the status strip, so the two cannot disagree
    about how many sensors are alive.
    """
    if session.source is None:
        return WARN, "no stream"
    if session.last_health is None:
        return WARN, "not scanned yet"
    out = sorted({str(d).split(".")[0] for d in session.cal.dead},
                 key=lambda s: int(s[1:]) if s[1:].isdigit() else 0)
    if not out:
        return OK, "16 of 16 healthy"
    return WARN, (f"{16 - len(out)} of 16 healthy — "
                  f"{', '.join(out)} excluded")


def stream_losses(source):
    """(gaps, samples lost, blocks dropped) since this source was opened."""
    try:
        st = source.stats() if source is not None else {}
    except Exception:
        st = {}
    return (int(st.get("gaps", 0) or 0), int(st.get("lost", 0) or 0),
            int(st.get("dropped blocks", 0) or 0))


def _stream(session):
    src = session.source
    if src is None:
        return Check("Stream", FAIL, "not connected — press Connect")
    if getattr(src, "failed", False):
        return Check("Stream", FAIL, "the source has stopped delivering — "
                                     "Disconnect and Connect again")
    if not getattr(src, "live", False):
        return Check("Stream", WARN, "demo or replay data, not the probe")
    return Check("Stream", OK, f"live at {src.fs_hz / 1000:g} kSPS")


def _losses(session):
    if session.source is None:
        return Check("Stream losses", WARN, "no stream to judge")
    gaps, lost, dropped = stream_losses(session.source)
    if not (gaps or lost or dropped):
        return Check("Stream losses", OK, "none since connecting")
    bits = []
    if gaps:
        bits.append(f"{gaps} gap(s), {lost} sample(s) lost")
    if dropped:
        bits.append(f"{dropped} block(s) dropped")
    return Check("Stream losses", WARN,
                 "; ".join(bits) + " since connecting — lower the stream "
                 "rate or reconnect before a long run")


def _stages(session):
    if session.stages is None:
        return [Check("Stages", FAIL, "not connected")]
    out = [Check("Stages", OK, ", ".join(session.stages.names) + " connected")]
    try:
        lost = session.stages.untrusted()
    except Exception as exc:
        lost = [("?", f"{type(exc).__name__}: {exc}")]
    if lost:
        out.append(Check(
            "Axes referenced", FAIL,
            ", ".join(n for n, _ in lost) + " cannot be believed about where "
            "they are — Home all"))
    else:
        out.append(Check("Axes referenced", OK, "every axis is homed"))
    return out


def _encoders(session):
    enc = session.encoders
    on_clock = [a for a in ("x", "y", "z") if a in enc]
    cols = int(getattr(session.source, "enc_columns", 0) or 0)
    if session.source is not None and not cols:
        return Check("Encoders", WARN, "this stream carries no encoder "
                                       "counts — positions will be polled "
                                       "over USB")
    if len(on_clock) == 3:
        return Check("Encoders", OK, "x, y, z on the sample clock")
    polled = [a for a in ("x", "y", "z") if a not in on_clock]
    return Check("Encoders", WARN,
                 f"{', '.join(polled)} not calibrated — positions on "
                 f"{'that axis' if len(polled) == 1 else 'those axes'} will "
                 f"be polled over USB")


def _plan(plan, plan_current):
    if plan is None or not plan.lines:
        return Check("Path", FAIL, "not planned — press Work out the path")
    if not plan_current:
        return Check("Path", FAIL, "planned for a pose, coil set or clearance "
                                   "that has since changed — plan again")
    return Check("Path", OK, f"{len(plan.lines)} lines, about "
                             f"{plan.duration_s() / 3600.0:.1f} h")


def _way_in(plan, head_mm):
    if head_mm is None:
        return Check("Way in", WARN, "the head's position is not known")
    try:
        _via, warning = plan.entry(head_mm)
    except ValueError as exc:
        return Check("Way in", FAIL, str(exc))
    if warning:
        return Check("Way in", WARN, warning)
    return Check("Way in", OK, "the move to the first line is clear")


def _disk(session, plan):
    try:
        free = shutil.disk_usage(session.out_dir).free
    except OSError:
        try:
            free = shutil.disk_usage(".").free
        except OSError:
            return Check("Disk space", WARN, "could not be read")
    need = (plan.rows() * BYTES_PER_ROW) if plan is not None else 0
    text = f"{free / 1e9:.1f} GB free"
    if need:
        text += f", this map needs up to {need / 1e9:.2f} GB"
    if need and free < 1.5 * need:
        return Check("Disk space", FAIL, text)
    if free < LOW_DISK_BYTES:
        return Check("Disk space", WARN, text)
    return Check("Disk space", OK, text)


def run_checks(session, motion=None, plan=None, plan_current=True,
               head_mm=None):
    """Every question worth asking before an unattended map, in reading order.

    `plan` is the SweepPlan in hand or None, `plan_current` whether it was
    made for the pose and coils as they are now, and `head_mm` where the
    stages say the head is ({axis: mm}) or None.
    """
    checks = [_stream(session), _losses(session)]
    state, text = sensor_summary(session)
    checks.append(Check("Sensors", state, text))
    checks.extend(_stages(session))
    if motion is not None:
        checks.append(
            Check("Emergency stop", FAIL,
                  f"latched ({motion.reason}) — press Reset, then home")
            if motion.latched else Check("Emergency stop", OK, "clear"))
    busy = None
    if session.volume_runner is not None:
        busy = "a volume map is already running"
    elif motion is not None and motion.busy():
        busy = "a stage job is running"
    elif session.collecting is not None:
        busy = "a tare or magnet pass is collecting"
    checks.append(Check("Machine idle", FAIL, busy) if busy
                  else Check("Machine idle", OK, "nothing else is running"))
    checks.append(_encoders(session))
    checks.append(_plan(plan, plan_current))
    if plan is not None and plan.lines and plan_current:
        checks.append(_way_in(plan, head_mm))
    checks.append(_disk(session, plan))
    return checks


def worst(checks):
    """The most serious state in a list of checks."""
    states = {c.state for c in checks}
    return FAIL if FAIL in states else WARN if WARN in states else OK


def summary(checks):
    """One line: what is stopping a run, or what it will be worse for."""
    fails = [c.name.lower() for c in checks if c.state == FAIL]
    warns = [c.name.lower() for c in checks if c.state == WARN]
    if fails:
        return "not ready: " + ", ".join(fails)
    if warns:
        return "ready, with warnings: " + ", ".join(warns)
    return "ready to run"
