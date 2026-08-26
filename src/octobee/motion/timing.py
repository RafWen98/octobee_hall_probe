"""
octobee/motion/timing.py -- how long a move takes, and how fast it really goes.

Pure arithmetic, deliberately kept apart from the Kinesis binding next door.
Nothing here imports ctypes or touches a DLL, which means it can be checked on
any machine -- including the CI runner, where Kinesis is not installed and
cannot be, and where every test that needs the binding is skipped.

That mattered enough to be worth a module of its own. The trapezoidal-move
maths is what decides whether a jog is quiet, whether a scan's per-point
deadline is generous or spurious, and how long an overnight field map claims it
will take, and none of those need a stage attached to be wrong.
"""

# Motion profile applied when a stage opens, unless stages.json says otherwise.
# Kinesis's own 20 mm/s / 20 mm/s^2 puts every move longer than about 5 mm into
# the motor's resonance band -- see the note at the top of this file.
#
# 6 mm/s is a BENCH RESULT, not a calculation. The first estimate was 8, on the
# reasoning that a 5 mm jog already peaked at 10 mm/s and was quiet; that was
# wrong, and wrong in an instructive way. A jog only touches its peak for an
# instant, where a long absolute move sits at the cap for tens of seconds, and
# it is the sustained tone that is objectionable. Brief peaks are a poor guide
# to what a traverse will sound like. 8 was still audible on this rig, 6 is not.
#
# It is not free: a full 300 mm traverse goes from about 16 s at the shipped
# settings to about 51 s. The acceleration is for the ramps rather than the top
# speed, and is left where it is -- lowering it further would only make long
# moves slower without changing the tone they settle at.
DEFAULT_VEL_MM_S = 6.0
DEFAULT_ACCEL_MM_S2 = 10.0

# A hard ceiling on every commanded move, whatever a config file, a spin box or
# a command line asks for. The default above is what the rig normally runs at;
# this is the line it may not cross even when someone deliberately raises it.
#
# It exists because a velocity setting is not a promise. The controller keeps
# its own stored settings and ISC_LoadSettings puts them back every time a
# stage is opened -- so anything that opens a device outside this module (the
# Kinesis application, a crashed process, a power cycle) leaves it at the
# shipped 20 mm/s, and the next absolute move runs at that. The stages are
# therefore given their profile again immediately before every move rather
# than once when they are opened: one extra DLL call against a move that takes
# seconds, in exchange for "the speed on screen is the speed it will use".
MAX_VEL_MM_S = 10.0


def clamp_velocity(mm_s):
    """Whatever was asked for, capped at MAX_VEL_MM_S."""
    return min(float(mm_s), MAX_VEL_MM_S)


def ramp_distance_mm(vel_mm_s, accel_mm_s2):
    """How far a trapezoidal move spends getting up to speed, millimetres.

    The reason a swept measurement wants to know: the first and last v^2/2a of
    a move are not at the sweep velocity, so a line that starts exactly at the
    edge of the region being mapped measures its own acceleration. Starting
    this much earlier puts the ramp outside.
    """
    v, a = float(vel_mm_s), float(accel_mm_s2)
    if v <= 0 or a <= 0:
        return 0.0
    return v * v / (2.0 * a)

def peak_speed_mm_s(distance_mm, vel_mm_s, accel_mm_s2):
    """How fast a trapezoidal move of `distance_mm` actually gets, mm/s.

    A move short enough to spend all of itself ramping never reaches the
    velocity cap: it peaks at sqrt(a*d) and turns round. That is the whole
    explanation for a rig that is quiet at 2 mm and howls at 20 -- the setting
    did not change, the distance did.
    """
    d = abs(float(distance_mm))
    if accel_mm_s2 <= 0:
        return float(vel_mm_s)
    return min(float(vel_mm_s), (accel_mm_s2 * d) ** 0.5)


def move_time_s(distance_mm, vel_mm_s, accel_mm_s2):
    """How long a trapezoidal move of this length actually takes, seconds.

    Triangular if it is too short to reach the cap, trapezoidal if not -- the
    same split as peak_speed_mm_s, seen from the other side.
    """
    d, v, a = abs(float(distance_mm)), float(vel_mm_s), float(accel_mm_s2)
    if d <= 0 or v <= 0 or a <= 0:
        return 0.0
    if d <= v * v / a:                       # never reaches the cap
        return 2.0 * (d / a) ** 0.5
    return d / v + v / a


def move_timeout_s(distance_mm, vel_mm_s, accel_mm_s2):
    """A generous deadline for a move of this length, seconds.

    A fixed 180 s was wrong at both ends. At 0.1 mm/s -- which the speed box
    allows -- a 300 mm traverse takes 50 minutes, so the deadline fired on a
    move that was proceeding perfectly and stopped it. That mattered more once
    a timeout started marking the position untrustworthy: a spurious one now
    costs a re-home, and mid-raster it ends the map.

    Three times the calculated time plus 5 s. Wide enough that only a genuine
    stall reaches it, narrow enough that a stalled axis is not left grinding
    for an hour.
    """
    return max(30.0, 3.0 * move_time_s(distance_mm, vel_mm_s, accel_mm_s2) + 5.0)
