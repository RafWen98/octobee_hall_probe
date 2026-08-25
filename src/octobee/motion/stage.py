#!/usr/bin/env python3
"""
octobee/motion/stage.py -- drive the Thorlabs LTS300C translation stages directly,
without the Kinesis application.

Why ctypes and not pythonnet
----------------------------
Kinesis ships two APIs side by side: a .NET one (`*CLI.dll`, what the Kinesis
GUI itself uses) and a flat C one (`Thorlabs.MotionControl.*.dll` with matching
`.h` headers in the install folder). The obvious route is pythonnet, and it is
the wrong one here: this machine runs Python 3.14, and pythonnet has no wheel
that far forward. The C API needs no bridge at all -- ctypes talks to it
directly, on any Python, with no new dependency. The signatures below were
transcribed from the shipped header, not guessed.

Only ONE process may hold these devices
---------------------------------------
The stages are FTDI/APT USB devices and open exclusively. If the Kinesis
application is running it owns all three, and TLI_BuildDeviceList() here
returns an empty list -- which looks exactly like "no stages plugged in".
That failure is common enough that list_devices() reports it explicitly rather
than letting you chase a phantom cabling fault. Close Kinesis first.

What the position readout actually means
----------------------------------------
The LTS300C has no linear scale on the carriage. The device reports
stepsPerRev=200, gearBoxRatio=1.0, pitch=1.0 mm against 409600 device units
per mm, i.e. it counts 2048 microsteps per full step of a 1 mm-pitch
leadscrew and tells you where it counted itself to. So:

    repeatability (returning to a point)   ~ micrometres, excellent
    absolute accuracy (where the point is) ~ 47 um uncalibrated

Thorlabs publish per-serial-number calibration files that map out each
individual leadscrew's error and take that 47 um to <+/-5 um. As of
2026-08-20 none are installed on this machine for 45502844 / 45502854 /
45538374. If you are mapping a field with any real gradient, get them -- a
position error d shows up in your data as a field error |grad B| * d, and
against a 1 mT/mm gradient 47 um is 47 uT, which is 2000x the 0.02 uT noise
floor that octobee/calib/poses.py works so hard to reach. Once you have the
files, install them with set_calibration_file() and they apply inside the
controller from then on.

Positions are meaningless until homed
-------------------------------------
A stage that has not been homed still reports a position, and that number is
whatever happened to be in the counter. `Stage.homed` is the only thing that
makes a reading trustworthy, and nothing in this module homes a stage for you:
homing drives the carriage into a limit switch at speed, and with a probe head
and its cable dress mounted that is a collision, not a formality. Homing is
always an explicit call.

How fast it moves, and why that is audible
------------------------------------------
Kinesis ships the LTS300C at 20 mm/s with 20 mm/s^2 of acceleration, and on
this rig that is loud. A trapezoidal move only reaches the velocity cap if it
is long enough to get there, so the peak speed of a move of d mm is

    v_peak = min(v_max, sqrt(a * d))

At the shipped numbers that is 10 mm/s for a 5 mm jog but the full 20 mm/s for
anything past 20 mm -- which is why small steps are quiet and larger ones
howl. It is the motor's resonance, not a fault, and the cure is to keep the
peak below where it starts rather than to jog in smaller steps. DEFAULT_VEL_MM_S
and DEFAULT_ACCEL_MM_S2 below are that cap, `speed` on the CLI changes it, and
stages.json makes it stick. Homing has its own velocity (2 mm/s here) and is
not affected by any of it.

Above that setting there is a hard ceiling, MAX_VEL_MM_S, that nothing in this
module will move faster than -- and because the controller's own stored
settings come back every time anything opens the device, each stage is given
its profile again immediately before every move rather than trusting the one
applied when it was opened.

Which way is up is not the stage's business
-------------------------------------------
A stage homes to its own limit switch, calls that 0 and counts up to 300.
Which physical direction that runs in depends on how the bracket was bolted
on, and the controller has no idea. On this rig z is mounted reversed, so its
home is the TOP of the working volume: left alone, every map would come out
mirrored along z and look entirely plausible. See Stage for the frame that
fixes it, declared once in stages.json rather than as a minus sign scattered
over the call sites.

EMERGENCY STOP, and what this one is not
----------------------------------------
`StageSet.emergency_stop()` is a CONTROLLED STOP REQUEST, not a safety
function. It stops the axes immediately, latches every further move off, and
marks the positions untrustworthy -- and it does all of that over USB, from
this process, through the Kinesis DLL. Every one of those has to be working.
The failures a mushroom head exists for are the ones where they are not: the
PC wedged, the USB dropped, the controller's firmware confused, this process
killed. In those the motors keep whatever they were last told to do.

    If this rig can hurt someone or destroy something it cannot afford to
    lose, the E-stop that matters is a hardware one in series with the
    controllers' supply. EN ISO 13850 category 0. Nothing below replaces it.

What is below is worth having anyway, because most of what goes wrong on a
bench rig is not a runaway -- it is a raster driving the head into a fixture
that was moved since the map was set up, and for that a button that stops all
three axes and refuses to start again is exactly the right instrument. The
three properties that make it one:

    it latches      stopping the axis that is moving does nothing about the
                    thread that is about to command the next move
    it distrusts    an immediate stop abandons the deceleration ramp, so the
                    count no longer matches the carriage. Absolute moves are
                    refused until the axis is homed again -- see
                    position_trusted, which is the single most important thing
                    in this file
    it is machine-wide, not per axis

Soft limits
-----------
The controller's travel limits describe the leadscrew. Everything this rig can
actually collide with -- fixture, magnet clamp, cable dress -- is INSIDE that
travel, so travel limits protect nothing. "limit_mm" in stages.json is the
working envelope, per axis, in rig millimetres, and it is what every move is
checked against. Measure it once with the head that is actually fitted.

Usage
-----
    python octobee/motion/stage.py list                    # what is on the USB bus
    python octobee/motion/stage.py status                  # per-axis position + flags
    python octobee/motion/stage.py home --axis x           # explicit, one axis
    python octobee/motion/stage.py moveto --x 100 --y 50   # absolute, mm
    python octobee/motion/stage.py stop                    # profiled stop, all axes
    python octobee/motion/stage.py estop                   # EMERGENCY: stop now
    python octobee/motion/stage.py speed                   # motion profile per axis
    python octobee/motion/stage.py speed --vel 8 --save    # slow it down, permanently

    python octobee/motion/stage.py map --assign x=45502844 --assign z=45502854 \\
                               --assign y=45538374 --invert z

Axis names and mounting come from stages.json (see AXIS_CONFIG); without it the
stages are addressed by serial number and assumed to be mounted forwards.
"""

import argparse
import contextlib
import ctypes as C
import json
import os
import subprocess
import sys
import time

from octobee import paths

AXIS_CONFIG = "stages.json"

# The install path is fixed by the Kinesis installer. Overridable because a
# 32-bit install lands in Program Files (x86) and some sites relocate it.
KINESIS_DIR = os.environ.get(
    "KINESIS_DIR", r"C:\Program Files\Thorlabs\Kinesis")

# Integrated Stepper Controllers: the LTS/MLJ/K10CR family. Serial prefix 45
# is LongTravelStage, which is what the LTS300C reports.
ISC_DLL = "Thorlabs.MotionControl.IntegratedStepperMotors.dll"
ISC_PREFIX = "45"

# ISC_GetRealValueFromDeviceUnit / ISC_GetDeviceUnitFromRealValue selector.
UNIT_DISTANCE = 0
UNIT_VELOCITY = 1
UNIT_ACCELERATION = 2

# Fallback scaling for the LTS300C, used only if the DLL's own unit conversion
# fails (it needs ISC_LoadSettings to have succeeded first). Measured against
# the device: axis max 122880000 du == 300.0 mm.
DEFAULT_DU_PER_MM = 409600.0

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

# Status word, from the header's GetStatusBits documentation. Only the bits
# this module acts on or reports are named.
STATUS_BITS = (
    (0x00000001, "cw_hard_limit"),
    (0x00000002, "ccw_hard_limit"),
    (0x00000004, "cw_soft_limit"),
    (0x00000008, "ccw_soft_limit"),
    (0x00000010, "moving_cw"),
    (0x00000020, "moving_ccw"),
    (0x00000040, "jogging_cw"),
    (0x00000080, "jogging_ccw"),
    (0x00000100, "motor_connected"),
    (0x00000200, "homing"),
    (0x00000400, "homed"),
    (0x00001000, "tracking"),
    (0x00002000, "settled"),
    (0x00004000, "motion_error"),
    (0x01000000, "current_limit"),
    (0x80000000, "enabled"),
)
MOVING_MASK = 0x000002F0    # any of moving/jogging cw+ccw, or homing
HOMED_BIT = 0x00000400
MOTION_ERROR_BIT = 0x00004000
ENABLED_BIT = 0x80000000
HARD_LIMIT_MASK = 0x00000003    # either physical end-of-travel switch


class StageError(RuntimeError):
    """Any failure talking to a stage, including a non-zero API return."""


class MotionInterlocked(StageError):
    """A move was refused because the interlock is latched.

    Its own type so a caller can tell "the machine is stopped and someone has
    to say so" apart from "that move was out of range". A scan treats the
    first as fatal and the second as a point it can skip.
    """


class MotionInterlock:
    """One latch shared by every axis of a machine.

    An emergency stop that only stops the axis that is moving is not an
    emergency stop. The thing that pressed it does not know what else is in
    flight -- another worker thread mid-raster, a queued jog, a wizard about
    to command the next pose -- and each of those will happily start moving
    again the instant the current move ends, which on this rig is a fraction
    of a second later. So the stop latches: every axis holds a reference to
    the same interlock, and while it is tripped every commanded move is
    refused at the point of command, whoever asks and from whichever thread.

    Clearing it is deliberately a separate, explicit act. That is the whole
    convention -- the machine does not un-stop itself because the fault
    cleared, it un-stops when a person says the reason is dealt with.
    """

    def __init__(self):
        self._reason = None

    @property
    def tripped(self):
        """The reason string if latched, None if clear."""
        return self._reason

    def trip(self, reason):
        """Latch. The FIRST reason wins -- it is the one that explains why.

        A stop cascades: the operator hits the button, that trips the axis,
        the axis error trips the set. Keeping the last reason would replace
        "z hit the hard limit" with "operator stop", which is the account of
        what happened that is least use afterwards.
        """
        if self._reason is None:
            self._reason = str(reason)
        return self._reason

    def reset(self):
        """Clear the latch. Returns what it was, for the log."""
        was, self._reason = self._reason, None
        return was

    def require_clear(self, what):
        if self._reason is not None:
            raise MotionInterlocked(
                f"motion is latched off ({self._reason}) -- {what} refused. "
                f"Clear the emergency stop before commanding any move.")


# ---------------------------------------------------------------------------
# ctypes binding
# ---------------------------------------------------------------------------

class TLI_DeviceInfo(C.Structure):
    _fields_ = [
        ("typeID", C.c_uint32),
        ("description", C.c_char * 65),
        ("serialNo", C.c_char * 16),
        ("PID", C.c_uint32),
        ("isKnownType", C.c_bool),
        ("motorType", C.c_int),
        ("isPiezoDevice", C.c_bool),
        ("isLaser", C.c_bool),
        ("isCustomType", C.c_bool),
        ("isRack", C.c_bool),
        ("maxChannels", C.c_short),
    ]


class TLI_HardwareInformation(C.Structure):
    _fields_ = [
        ("serialNumber", C.c_uint32),
        ("modelNumber", C.c_char * 8),
        ("type", C.c_ushort),
        ("firmwareVersion", C.c_uint32),
        ("notes", C.c_char * 48),
        ("deviceDependantData", C.c_ubyte * 12),
        ("hardwareVersion", C.c_ushort),
        ("modificationState", C.c_ushort),
        ("numChannels", C.c_short),
    ]


_DLL = None


def _bind(dll):
    """Attach argtypes/restypes. Without these ctypes assumes int returns and
    silently truncates the DWORD status word and the 64-bit pointers."""
    sig = {
        "TLI_BuildDeviceList": ([], C.c_short),
        "TLI_GetDeviceListSize": ([], C.c_short),
        "TLI_GetDeviceListExt": ([C.c_char_p, C.c_uint32], C.c_short),
        "TLI_GetDeviceInfo": ([C.c_char_p, C.POINTER(TLI_DeviceInfo)], C.c_short),
        "ISC_Open": ([C.c_char_p], C.c_short),
        "ISC_Close": ([C.c_char_p], None),
        "ISC_LoadSettings": ([C.c_char_p], C.c_bool),
        "ISC_StartPolling": ([C.c_char_p, C.c_int], C.c_bool),
        "ISC_StopPolling": ([C.c_char_p], None),
        "ISC_ClearMessageQueue": ([C.c_char_p], None),
        "ISC_EnableChannel": ([C.c_char_p], C.c_short),
        "ISC_DisableChannel": ([C.c_char_p], C.c_short),
        "ISC_GetPosition": ([C.c_char_p], C.c_int),
        "ISC_RequestPosition": ([C.c_char_p], C.c_short),
        "ISC_GetStatusBits": ([C.c_char_p], C.c_uint32),
        "ISC_RequestStatusBits": ([C.c_char_p], C.c_short),
        "ISC_Home": ([C.c_char_p], C.c_short),
        "ISC_MoveToPosition": ([C.c_char_p, C.c_int], C.c_short),
        "ISC_MoveRelative": ([C.c_char_p, C.c_int], C.c_short),
        # Declared, but nothing here calls them -- the GUI's jog buttons are
        # relative moves. If you ever do use MoveJog, note that it runs off
        # ISC_SetJogVelParams, a SEPARATE velocity table that SetVelParams does
        # not touch and that MAX_VEL_MM_S therefore does not cap. Cap it there
        # too, or a jog will quietly run at the shipped 20 mm/s.
        "ISC_MoveJog": ([C.c_char_p, C.c_int], C.c_short),
        "ISC_SetJogStepSize": ([C.c_char_p, C.c_uint], C.c_short),
        "ISC_SetJogMode": ([C.c_char_p, C.c_int, C.c_int], C.c_short),
        "ISC_StopProfiled": ([C.c_char_p], C.c_short),
        "ISC_StopImmediate": ([C.c_char_p], C.c_short),
        "ISC_CanMoveWithoutHomingFirst": ([C.c_char_p], C.c_bool),
        "ISC_GetHardwareInfoBlock":
            ([C.c_char_p, C.POINTER(TLI_HardwareInformation)], C.c_short),
        "ISC_GetFirmwareVersion": ([C.c_char_p], C.c_uint32),
        "ISC_GetMotorTravelLimits":
            ([C.c_char_p, C.POINTER(C.c_double), C.POINTER(C.c_double)], C.c_short),
        "ISC_GetMotorParamsExt":
            ([C.c_char_p, C.POINTER(C.c_double), C.POINTER(C.c_double),
              C.POINTER(C.c_double)], C.c_short),
        "ISC_GetRealValueFromDeviceUnit":
            ([C.c_char_p, C.c_int, C.POINTER(C.c_double), C.c_int], C.c_short),
        "ISC_GetDeviceUnitFromRealValue":
            ([C.c_char_p, C.c_double, C.POINTER(C.c_int), C.c_int], C.c_short),
        "ISC_GetVelParams":
            ([C.c_char_p, C.POINTER(C.c_int), C.POINTER(C.c_int)], C.c_short),
        "ISC_SetVelParams": ([C.c_char_p, C.c_int, C.c_int], C.c_short),
        "ISC_GetBacklash": ([C.c_char_p], C.c_long),
        "ISC_SetBacklash": ([C.c_char_p, C.c_long], C.c_short),
        "ISC_GetCalibrationFile": ([C.c_char_p, C.c_char_p, C.c_short], C.c_bool),
        "ISC_SetCalibrationFile": ([C.c_char_p, C.c_char_p, C.c_bool], None),
    }
    missing = []
    for name, (argtypes, restype) in sig.items():
        try:
            fn = getattr(dll, name)
        except AttributeError:
            # An older or differently-built Kinesis is a supportable situation;
            # a raw AttributeError from deep in a binding helper is not. Name
            # every entry point that is absent, in one message.
            missing.append(name)
            continue
        fn.argtypes = argtypes
        fn.restype = restype
    if missing:
        raise StageError(
            f"{ISC_DLL} in {KINESIS_DIR} does not export "
            f"{', '.join(missing)}. That is an older or differently-built "
            f"Kinesis than this module was written against -- update it, or "
            f"point KINESIS_DIR at the install that has these.")


def dll():
    """Load the Kinesis C API once, or explain precisely why it will not."""
    global _DLL
    if _DLL is not None:
        return _DLL
    if not os.path.isdir(KINESIS_DIR):
        raise StageError(
            f"Kinesis is not installed at {KINESIS_DIR}. Set KINESIS_DIR if it "
            f"lives elsewhere.")
    path = os.path.join(KINESIS_DIR, ISC_DLL)
    if not os.path.exists(path):
        raise StageError(f"{ISC_DLL} missing from {KINESIS_DIR}")
    # The DLL pulls in DeviceManager.dll and the FTDI layer from the same
    # folder. Without this they resolve against the process cwd and fail.
    os.add_dll_directory(KINESIS_DIR)
    try:
        d = C.CDLL(path)     # the header declares __cdecl throughout
    except OSError as exc:
        raise StageError(f"cannot load {path}: {exc}") from exc
    _bind(d)
    _DLL = d
    return d


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


def _check(rc, what):
    if rc != 0:
        raise StageError(f"{what} failed (Kinesis error {rc})")


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def list_devices():
    """Serial numbers of every integrated-stepper controller on the bus.

    An empty list almost always means the Kinesis application is running and
    holding the devices, not that they are unplugged -- these are exclusive-open
    FTDI devices. kinesis_is_running() distinguishes the two.
    """
    d = dll()
    _check(d.TLI_BuildDeviceList(), "TLI_BuildDeviceList")
    buf = C.create_string_buffer(512)
    # Checked, like the call above it. An unchecked failure here hands back an
    # empty buffer, which reads as "no stages on the bus" -- the same symptom
    # as the Kinesis-is-running case this module works hard to tell apart.
    _check(d.TLI_GetDeviceListExt(buf, 512), "TLI_GetDeviceListExt")
    return [s for s in buf.value.decode(errors="replace").split(",") if s]


def kinesis_is_running():
    """True if the Kinesis GUI is up, which makes every stage unopenable.

    Only ever used to make an error message more helpful, so any failure to
    ask -- no tasklist, not Windows, a timeout -- is answered with "do not
    know", spelled False.
    """
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq Thorlabs.MotionControl.Kinesis.exe"],
            capture_output=True, text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "Thorlabs.MotionControl.Kinesis.exe" in out


def device_info(serial):
    d = dll()
    info = TLI_DeviceInfo()
    # This one call inverts the convention the rest of the API uses. From the
    # shipped header: "<returns> 1 if successful, 0 if not." Everything else
    # here returns 0 for success, so checking it with _check() reports a
    # perfectly good call as "Kinesis error 1" -- and on this machine it does
    # so while having filled the struct correctly (typeID 45, "APT Stepper
    # Motor Controller"). That took out `octobee/motion/stage.py list` entirely.
    if d.TLI_GetDeviceInfo(serial.encode(), C.byref(info)) == 0:
        raise StageError(f"TLI_GetDeviceInfo({serial}) found no such device "
                         f"-- the device list may be stale; rebuild it with "
                         f"list_devices()")
    return {
        "serial": serial,
        "type_id": info.typeID,
        "description": info.description.decode(errors="replace").strip(),
        "is_isc": serial.startswith(ISC_PREFIX),
    }


# ---------------------------------------------------------------------------
# one axis
# ---------------------------------------------------------------------------

class Stage:
    """One LTS300C. Use as a context manager, or call open()/close().

    Every position in the public API is millimetres. Device units never leave
    this class.

    Rig coordinates vs device coordinates
    -------------------------------------
    A stage knows only its own leadscrew: it homes to a limit switch, calls
    that 0, and counts up to 300 from there. Which physical direction that runs
    in is a property of how the bracket was bolted on, and the controller has
    no idea. Mount a stage upside down and its 0 is the TOP of your working
    volume, so the origin of every map you take sits at a different corner of
    the cube than you think it does -- and a map with an inverted axis looks
    completely plausible, which is what makes this worth handling in one place
    rather than with a minus sign wherever it happens to come up.

    So each axis carries a small frame:

        invert      the stage runs opposite to the rig axis
        origin_mm   the DEVICE position that rig zero sits at. Defaults to the
                    far end of travel when inverted and 0 when not, which is
                    the "reverse-mounted stage" case; set it explicitly to put
                    rig zero on a fixture datum instead of a hard limit.

    device = origin + sign * rig,  sign = -1 if invert else +1

    Everything public -- position_mm, travel_mm, move_to, move_by, snapshot --
    is in RIG millimetres. position_dev_mm is the raw device number, kept for
    diagnostics and because a limit-switch fault is easier to read there.
    """

    def __init__(self, serial, name=None, poll_ms=100, invert=False,
                 origin_mm=None, vel_mm_s=DEFAULT_VEL_MM_S,
                 accel_mm_s2=DEFAULT_ACCEL_MM_S2, limit_mm=None,
                 interlock=None):
        self.serial = str(serial)
        self._sb = self.serial.encode()
        self.name = name or self.serial
        self.poll_ms = poll_ms
        self.is_open = False
        self.model = ""
        self.travel_mm = (0.0, 0.0)          # rig frame, the whole leadscrew
        self.travel_dev_mm = (0.0, 0.0)      # what the stage itself reports
        self.limit_mm = (0.0, 0.0)           # rig frame, what we may use
        self.invert = bool(invert)
        # None means "resolve from travel once the stage tells us what it is".
        self._origin_cfg = origin_mm
        self.origin_mm = 0.0 if origin_mm is None else float(origin_mm)
        self._limit_cfg = None if limit_mm is None else (
            float(limit_mm[0]), float(limit_mm[1]))
        self.interlock = interlock if interlock is not None else MotionInterlock()
        # Position trust is NOT the controller's homed bit -- see position_trusted.
        self._trusted = False
        self._distrust_reason = "not homed since this stage was opened"
        # None for either half means "leave whatever the controller has", which
        # is the only way to look at a stage without changing how it moves.
        self._vel_cfg = None if vel_mm_s is None else float(vel_mm_s)
        self._accel_cfg = None if accel_mm_s2 is None else float(accel_mm_s2)
        self._du_per_mm = DEFAULT_DU_PER_MM

    # ---- lifecycle ----

    def open(self):
        if self.is_open:
            return self
        d = dll()
        rc = d.ISC_Open(self._sb)
        if rc != 0:
            extra = (" -- the Kinesis application is running and holds this "
                     "device; close it first" if kinesis_is_running() else "")
            raise StageError(f"cannot open stage {self.serial} "
                             f"(Kinesis error {rc}){extra}")
        self.is_open = True
        try:
            # LoadSettings is what teaches the DLL this serial's stage type.
            # Travel limits and unit conversion are wrong without it.
            if not d.ISC_LoadSettings(self._sb):
                raise StageError(
                    f"ISC_LoadSettings failed for {self.serial}: the DLL does "
                    f"not know this stage's type, so travel limits and mm "
                    f"conversion cannot be trusted")
            d.ISC_StartPolling(self._sb, self.poll_ms)
            d.ISC_ClearMessageQueue(self._sb)
            # Polling is asynchronous; the first status word is not valid
            # until a poll cycle has actually completed.
            time.sleep(self.poll_ms / 1000.0 * 3)
            self._read_static()
            # A stage that is still powered from an earlier session keeps both
            # its homed bit and a count that really does match the carriage,
            # so opening it does not by itself make the position untrustworthy.
            # What would is a fault it is sitting in right now.
            bits = self.status
            if bits & MOTION_ERROR_BIT:
                self.distrust("the controller is reporting a motion error")
            elif bits & HOMED_BIT:
                self._trusted = True
            # After LoadSettings, which is what put the shipped 20/20 there.
            if self._vel_cfg is not None or self._accel_cfg is not None:
                self.set_vel_params(self._vel_cfg, self._accel_cfg)
        except Exception:
            self.close()
            raise
        return self

    def close(self):
        """Stop the axis if it is moving, then release the device.

        Closing does NOT stop a stage. The move is already in the controller
        and the controller runs it whether or not anything is still listening
        -- so a window closed mid-traverse used to leave an axis driving to a
        target with nobody watching, and the only remaining brake was the
        limit switch. One profiled stop on the way out costs a fraction of a
        second and is the difference between "shut down" and "walked away
        from a moving machine".
        """
        if not self.is_open:
            return
        d = dll()
        try:
            if self.moving:
                d.ISC_StopProfiled(self._sb)
                deadline = time.monotonic() + 5.0
                while self.moving and time.monotonic() < deadline:
                    time.sleep(0.02)
        except StageError:
            # Best effort: a stage that cannot be read cannot be stopped
            # either, and failing here would skip the close below and leak an
            # exclusive-open USB device for the lifetime of the process.
            pass
        try:
            d.ISC_StopPolling(self._sb)
        finally:
            d.ISC_Close(self._sb)
            self.is_open = False

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False

    def _read_static(self):
        d = dll()
        hw = TLI_HardwareInformation()
        d.ISC_GetHardwareInfoBlock(self._sb, C.byref(hw))
        self.model = hw.modelNumber.decode(errors="replace").strip()
        lo, hi = C.c_double(), C.c_double()
        if d.ISC_GetMotorTravelLimits(self._sb, C.byref(lo), C.byref(hi)) == 0:
            self.travel_dev_mm = (lo.value, hi.value)
        # Derive du/mm from the device rather than trusting the constant, so a
        # different stage model on the same prefix cannot silently mis-scale.
        one_mm = C.c_int()
        if d.ISC_GetDeviceUnitFromRealValue(
                self._sb, 1.0, C.byref(one_mm), UNIT_DISTANCE) == 0 and one_mm.value:
            self._du_per_mm = float(one_mm.value)
        self._resolve_frame()

    def _resolve_frame(self):
        """Fix the rig origin and rig-frame travel now the travel is known.

        Deferred to here because the sensible default for a reverse-mounted
        stage is "rig zero is at the far end of travel", and the far end is
        something the stage has to tell us.
        """
        lo, hi = self.travel_dev_mm
        if self._origin_cfg is None:
            self.origin_mm = hi if self.invert else lo
        else:
            self.origin_mm = float(self._origin_cfg)
        # + 0.0 normalises the -0.0 that falls out of negating zero, which is
        # numerically identical but reads as a fault in a travel limit.
        ends = sorted((self._dev_to_rig(lo) + 0.0, self._dev_to_rig(hi) + 0.0))
        self.travel_mm = (ends[0], ends[1])
        self.limit_mm = self._resolve_limit(self.travel_mm, self._limit_cfg)

    @staticmethod
    def _resolve_limit(travel, cfg):
        """The working envelope: the configured soft limit, inside the travel.

        The stage's own limits describe the leadscrew and nothing else. What
        the machine may actually use is smaller, because a probe head, a
        cantilever bracket and a cable dress occupy some of it -- and the
        controller has no way to know that. Everything mechanical this rig can
        hit is inside the travel, so travel limits alone protect nothing.

        Clamped to the travel rather than trusted: a soft limit is a
        restriction, and a config file that asks for a wider one than the
        stage has is a typo, not a permission.

        Split out from _resolve_frame so it can be checked without a stage.
        """
        lo, hi = travel
        if cfg is None:
            return (lo, hi)
        want_lo, want_hi = sorted(cfg)
        if hi > lo:
            want_lo, want_hi = max(want_lo, lo), min(want_hi, hi)
        if not want_hi > want_lo:
            raise StageError(
                f"soft limit {cfg[0]:g}..{cfg[1]:g} mm leaves no travel "
                f"inside {lo:g}..{hi:g} mm")
        return (want_lo, want_hi)

    @property
    def limit_declared(self):
        """True if stages.json states this axis's envelope, whatever it says.

        Distinct from "the envelope is smaller than the travel", and the
        difference is the whole point. An axis with no limit_mm has never been
        considered; an axis whose limit_mm is the full travel has been measured
        and found unobstructed. Both allow exactly the same movement, and only
        one of them is a decision -- so only one of them should still be
        nagging the operator at every connect.
        """
        return self._limit_cfg is not None

    # ---- position trust ----

    @property
    def position_trusted(self):
        """True if this axis's counter can be believed as an ABSOLUTE position.

        The controller's homed bit is not this, and treating it as if it were
        is how a stage drives somewhere confidently wrong. The bit says "a
        homing cycle completed at some point"; it stays set afterwards no
        matter what happens to the count. Steps get lost -- an immediate stop
        abandons the deceleration ramp, a stall, a crash into something that
        is not the limit switch, a driver fault -- and every one of those
        leaves the bit set and the number wrong. The stage then reports a
        position it believes, an absolute move computes a distance from it,
        and the head goes exactly as far wrong as the count drifted.

        So trust is tracked here as well: set by a homing cycle that this
        module watched complete, and cleared by anything that could have cost
        steps. Absolute moves require BOTH.

        Never raises, including on a closed stage. It is read from the stop
        path and from the dialog that offers to reset one, and a property that
        throws there would turn "the stage went away" into a traceback in the
        middle of stopping the machine. A closed stage is simply not trusted.
        """
        return bool(self._trusted and self.is_open and self.homed)

    @property
    def distrust_reason(self):
        if not self.is_open:
            return "the stage is not open"
        if not self._trusted:
            return self._distrust_reason
        if not self.homed:
            return "the controller reports this axis as not homed"
        return None

    def distrust(self, reason):
        """Mark the position counter unreliable. Absolute moves stop working."""
        self._trusted = False
        self._distrust_reason = str(reason)

    def _require_open(self):
        if not self.is_open:
            raise StageError(f"stage {self.name} is not open")

    # ---- units ----

    def _to_du(self, mm):
        return int(round(mm * self._du_per_mm))

    def _du(self, real, unit_type):
        """Real-world value -> device units, for velocity and acceleration.

        Distance has a clean scale factor; velocity and acceleration do not --
        they carry the controller's own sample rate in them, so the conversion
        has to come from the DLL rather than from arithmetic here.
        """
        out = C.c_int()
        rc = dll().ISC_GetDeviceUnitFromRealValue(
            self._sb, float(real), C.byref(out), unit_type)
        if rc != 0:
            raise StageError(
                f"{self.name}: cannot convert {real:g} to device units "
                f"(Kinesis error {rc})")
        return out.value

    def _to_mm(self, du):
        return du / self._du_per_mm

    # ---- rig frame <-> device frame ----

    @property
    def _sign(self):
        return -1.0 if self.invert else 1.0

    def _rig_to_dev(self, mm):
        return self.origin_mm + self._sign * mm

    def _dev_to_rig(self, mm):
        return self._sign * (mm - self.origin_mm)

    def frame_note(self):
        """One line describing the axis frame, for logs and provenance."""
        if not self.invert and self.origin_mm == 0.0:
            return "as mounted"
        return (f"{'reversed' if self.invert else 'forward'}, rig 0 at device "
                f"{self.origin_mm:g} mm")

    # ---- state ----

    @property
    def status(self):
        self._require_open()
        return int(dll().ISC_GetStatusBits(self._sb))

    def status_flags(self, bits=None):
        b = self.status if bits is None else bits
        return [name for bit, name in STATUS_BITS if b & bit]

    @property
    def position_dev_mm(self):
        """Raw device position: what the controller counted, in its own frame."""
        self._require_open()
        return self._to_mm(dll().ISC_GetPosition(self._sb))

    @property
    def position_mm(self):
        """Position in RIG millimetres. Only meaningful if homed."""
        return self._dev_to_rig(self.position_dev_mm)

    @property
    def homed(self):
        return bool(self.status & HOMED_BIT)

    @property
    def moving(self):
        return bool(self.status & MOVING_MASK)

    @property
    def enabled(self):
        return bool(self.status & ENABLED_BIT)

    @property
    def vel_params(self):
        """(max velocity mm/s, acceleration mm/s^2) as the controller has them."""
        self._require_open()
        acc, vel = C.c_int(), C.c_int()
        _check(dll().ISC_GetVelParams(self._sb, C.byref(acc), C.byref(vel)),
               f"read motion profile of {self.name}")
        return (self._real(vel.value, UNIT_VELOCITY),
                self._real(acc.value, UNIT_ACCELERATION))

    @property
    def velocity_mm_s(self):
        return self.vel_params[0]

    @property
    def accel_mm_s2(self):
        return self.vel_params[1]

    @staticmethod
    def resolve_profile(vel_mm_s, accel_mm_s2, current):
        """(requested, current) -> the (velocity, acceleration) to send.

        Split out from set_vel_params so the rule -- fill in from the
        controller, then clamp -- can be checked without a stage attached.
        """
        cur_v, cur_a = current
        v = cur_v if vel_mm_s is None else float(vel_mm_s)
        a = cur_a if accel_mm_s2 is None else float(accel_mm_s2)
        if not (v > 0 and a > 0):
            raise StageError(
                f"velocity and acceleration must both be positive, got "
                f"{v:g} mm/s and {a:g} mm/s^2")
        return clamp_velocity(v), a

    def set_vel_params(self, vel_mm_s=None, accel_mm_s2=None):
        """Cap how fast and how hard this axis moves. Returns what was set.

        Both halves go to the controller in one call, so whichever is left None
        is read back and re-sent unchanged rather than defaulted to something.
        The profile lives in the controller and applies to every move it is
        given, including ones started by the GUI or by a scan.
        """
        self._require_open()
        # Only ask the controller what it has when we need it to fill a gap.
        # On the before-every-move path both halves are known, and the read
        # comes out of the DLL's polling cache anyway -- so it would be a
        # round trip whose answer is discarded.
        current = ((None, None) if vel_mm_s is not None and accel_mm_s2 is not None
                   else self.vel_params)
        try:
            v, a = self.resolve_profile(vel_mm_s, accel_mm_s2, current)
        except StageError as exc:
            raise StageError(f"{self.name}: {exc}") from None
        _check(dll().ISC_SetVelParams(self._sb, self._du(a, UNIT_ACCELERATION),
                                      self._du(v, UNIT_VELOCITY)),
               f"set motion profile on {self.name}")
        # Remember what this stage is meant to run at, so the re-assert before
        # each move has something to assert.
        self._vel_cfg, self._accel_cfg = v, a
        return (v, a)

    def enforce_profile(self):
        """Put this stage's profile back on the controller before it moves.

        Cheap and unconditional: sending the value is one call, where checking
        first would need ISC_RequestVelParams and a poll cycle to come back,
        and would still race with whatever changed it.
        """
        if self._vel_cfg is None and self._accel_cfg is None:
            return          # opened deliberately without touching how it moves
        self.set_vel_params(self._vel_cfg, self._accel_cfg)

    def peak_speed_mm_s(self, distance_mm):
        """How fast a move of this length actually gets on THIS axis, mm/s."""
        return peak_speed_mm_s(distance_mm, *self.vel_params)

    def _real(self, du, unit_type):
        out = C.c_double()
        rc = dll().ISC_GetRealValueFromDeviceUnit(
            self._sb, du, C.byref(out), unit_type)
        return out.value if rc == 0 else float("nan")

    def snapshot(self):
        """Everything the GUI needs for one status row, in one poll."""
        bits = self.status
        return {
            "name": self.name,
            "serial": self.serial,
            "model": self.model,
            "position_mm": self.position_mm,
            "position_dev_mm": self.position_dev_mm,
            "invert": self.invert,
            "frame": self.frame_note(),
            "homed": bool(bits & HOMED_BIT),
            "moving": bool(bits & MOVING_MASK),
            "enabled": bool(bits & ENABLED_BIT),
            "error": bool(bits & MOTION_ERROR_BIT),
            "at_hard_limit": bool(bits & HARD_LIMIT_MASK),
            "trusted": self.position_trusted,
            "distrust_reason": self.distrust_reason,
            "travel_mm": self.travel_mm,
            "limit_mm": self.limit_mm,
            "limit_declared": self.limit_declared,
            "interlocked": self.interlock.tripped,
            "status": bits,
            "flags": self.status_flags(bits),
        }

    # ---- motion ----

    def home(self, timeout_s=180.0, wait=True):
        """Reference the axis against its limit switch.

        This MOVES, at homing velocity, all the way to the end of travel. The
        caller is responsible for having decided that is safe -- nothing here
        knows what is bolted to the carriage.

        Homing always seeks the stage's own limit switch, which is device zero.
        On a reverse-mounted axis that is the FAR end of the rig axis, so the
        stage finishes at rig maximum rather than rig zero. That is correct,
        and it is why the rig origin is a frame setting rather than something
        homing could establish on its own.
        """
        self._require_open()
        self.interlock.require_clear(f"homing {self.name}")
        # A homing cycle deliberately drives into a hard stop, so the count is
        # meaningless from the moment it starts until the moment it finishes.
        # Distrust first: if this is interrupted -- stopped, timed out, faulted
        # -- it must not leave the previous trust standing.
        self.distrust(f"{self.name} is homing")
        d = dll()
        d.ISC_ClearMessageQueue(self._sb)
        _check(d.ISC_Home(self._sb), f"home {self.name}")
        if wait:
            self.wait(timeout_s=timeout_s, what="homing")
            self.trust_after_homing()

    def trust_after_homing(self):
        """Believe the counter again, but only if the cycle really finished.

        Split out because home(wait=False) hands the waiting to the caller --
        StageSet.home_all and the GUI both do -- and the trust has to be
        granted where the wait completed, not where the move was started.
        """
        if self.homed:
            self._trusted = True
            self._distrust_reason = ""
        else:
            self.distrust(f"{self.name}'s homing cycle did not complete")
        return self.position_trusted

    def _timeout_for(self, distance_mm, timeout_s):
        """None means "work it out from how far it has to go".

        Off the cached profile rather than the controller's: enforce_profile()
        has just sent exactly these numbers, so asking for them back is a DLL
        round trip whose answer is already known. Falls back to a read only for
        a stage opened deliberately without a profile.
        """
        if timeout_s is not None:
            return float(timeout_s)
        vel, acc = self._vel_cfg, self._accel_cfg
        if vel is None or acc is None:
            try:
                vel, acc = self.vel_params
            except StageError:
                return 180.0
        return move_timeout_s(distance_mm, vel, acc)

    def move_to(self, mm, timeout_s=None, wait=True):
        """Absolute move to RIG millimetres, against the soft limits."""
        self._require_open()
        self.interlock.require_clear(f"move {self.name} to {mm:g} mm")
        lo, hi = self.limit_mm
        if hi > lo and not (lo <= mm <= hi):
            extra = ("" if self.limit_mm == self.travel_mm else
                     f" (soft limit, inside the {self.travel_mm[0]:g}.."
                     f"{self.travel_mm[1]:g} mm travel)")
            raise StageError(
                f"{self.name}: {mm:g} mm is outside "
                f"{lo:g}..{hi:g} mm{extra}")
        if not self.position_trusted:
            raise StageError(
                f"{self.name}: {self.distrust_reason} -- its position counter "
                f"cannot be believed, so an absolute move would go somewhere "
                f"arbitrary. Home it first.")
        dev = self._rig_to_dev(mm)
        d = dll()
        self.enforce_profile()
        d.ISC_ClearMessageQueue(self._sb)
        here = self.position_mm
        _check(d.ISC_MoveToPosition(self._sb, self._to_du(dev)),
               f"move {self.name} to {mm:g} mm")
        if wait:
            self.wait(timeout_s=self._timeout_for(mm - here, timeout_s),
                      what=f"move to {mm:g} mm")

    def move_by(self, delta_mm, timeout_s=None, wait=True):
        """Relative move, millimetres.

        Deliberately does NOT require homing. A relative move needs no absolute
        reference to be correct, and this is the only way to nudge an axis
        before it has been homed -- which is exactly what you need in order to
        work out which stage is which axis, and to walk the probe clear of an
        obstruction so that homing becomes safe in the first place.

        The travel-limit check is therefore against the counter's own idea of
        where it is, which on an unhomed axis is a guess. The controller's
        limit switches remain the real protection.
        """
        self._require_open()
        self.interlock.require_clear(f"jog {self.name} by {delta_mm:+g} mm")
        lo, hi = self.limit_mm
        target = self.position_mm + delta_mm
        if hi > lo and not (lo <= target <= hi) and self.position_trusted:
            raise StageError(
                f"{self.name}: {target:g} mm is outside "
                f"{lo:g}..{hi:g} mm")
        d = dll()
        self.enforce_profile()
        d.ISC_ClearMessageQueue(self._sb)
        # delta_mm is a RIG distance; on a reverse-mounted axis the device has
        # to travel the other way to produce it.
        _check(d.ISC_MoveRelative(self._sb, self._to_du(self._sign * delta_mm)),
               f"move {self.name} by {delta_mm:g} mm")
        if wait:
            self.wait(timeout_s=self._timeout_for(delta_mm, timeout_s),
                      what=f"move by {delta_mm:g} mm")

    def stop(self, immediate=False):
        """Profiled stop by default; immediate loses steps and thus position.

        `immediate` abandons the deceleration ramp, which is the point of it
        and also its cost: the motor is told to stop from full speed, the load
        keeps going, and the count no longer matches the carriage. So an
        immediate stop marks the position untrusted -- that is not a side
        effect to work around, it is the honest state afterwards.
        """
        self._require_open()
        d = dll()
        if immediate:
            self.distrust(f"{self.name} was stopped immediately, which "
                          f"abandons the deceleration ramp and can lose steps")
        rc = (d.ISC_StopImmediate(self._sb) if immediate
              else d.ISC_StopProfiled(self._sb))
        _check(rc, f"stop {self.name}")

    def emergency_stop(self, reason="emergency stop"):
        """Stop NOW, latch the interlock, and never raise on the way.

        The three things that separate this from stop(immediate=True):

        1. It latches. Stopping the axis that is moving does nothing about the
           thread that is about to command the next move, and on this rig that
           is a fraction of a second away.
        2. It cannot fail. An exception here would skip the remaining axes,
           and the whole reason for a machine-wide stop is that the axis that
           threw is not necessarily the one about to hit something.
        3. It is immediate, not profiled. At 6 mm/s and 10 mm/s^2 a profiled
           stop still travels about 1.8 mm; if 1.8 mm did not matter, nobody
           would be pressing the button.

        Returns "" on success or the error text, for the caller to log.
        """
        self.interlock.trip(reason)
        if not self.is_open:
            return ""
        try:
            self.distrust(f"{self.name} was stopped by an emergency stop "
                          f"({reason}), which can lose steps")
            _check(dll().ISC_StopImmediate(self._sb), f"stop {self.name}")
            return ""
        except Exception as exc:
            # Deliberately broad -- see point 2 above. Whatever went wrong
            # here, the remaining axes still have to be told to stop.
            return f"{self.name}: {type(exc).__name__}: {exc}"

    def wait(self, timeout_s=180.0, what="move", settle_s=0.0):
        """Block until motion stops.

        Polls the status word rather than draining the message queue: the queue
        needs every message consumed in order and one missed event hangs
        forever, whereas the status word is level-triggered and self-correcting.
        """
        self._require_open()
        deadline = time.monotonic() + timeout_s
        # The controller does not raise the moving bit instantly, so a poll
        # taken too early sees "not moving" and returns before the axis has
        # started. Give it one poll interval to get going.
        time.sleep(max(self.poll_ms, 50) / 1000.0 * 2)
        while True:
            bits = self.status
            if bits & MOTION_ERROR_BIT:
                self.distrust(f"{self.name} reported a motion error during "
                              f"{what}")
                self.interlock.trip(f"{self.name}: motion error during {what}")
                raise StageError(f"{self.name}: motion error during {what}")
            if not bits & MOVING_MASK:
                break
            if time.monotonic() > deadline:
                self.stop()
                self.distrust(f"{self.name} timed out during {what}, so it may "
                              f"have stalled rather than arrived")
                raise StageError(
                    f"{self.name}: {what} did not finish within {timeout_s:g} s")
            time.sleep(0.02)
        # Motion ending is not the same as the move having been carried out.
        # An emergency stop ends it too, and the caller -- a raster loop, a
        # homing sequence, a wizard pass -- is otherwise about to read a
        # position and command the next move as if nothing had happened. This
        # is the check that turns a stop into a stop.
        self.interlock.require_clear(f"{self.name}: {what}")
        # Homing is excluded because homing ends ON the limit switch, which
        # is the whole point of it. But an ordinary move that finishes against
        # a hard stop did not arrive, it collided, and the count is a guess
        # from there on.
        if bits & HARD_LIMIT_MASK and what != "homing":
            self.distrust(f"{self.name} ended {what} against a hard limit "
                          f"switch rather than at its target")
            raise StageError(
                f"{self.name}: {what} ran into a hard limit switch. The "
                f"soft limits did not stop it in time -- check "
                f"{AXIS_CONFIG} before moving this axis again.")
        if settle_s:
            time.sleep(settle_s)

    # ---- calibration file ----

    def calibration_file(self):
        self._require_open()
        buf = C.create_string_buffer(260)
        ok = dll().ISC_GetCalibrationFile(self._sb, buf, 260)
        # The call reports success with an empty buffer when no file is
        # loaded, so "" and None both mean "uncalibrated" -- collapse them,
        # or every caller has to remember to test truthiness rather than
        # `is None` and one of them will get it wrong.
        name = buf.value.decode(errors="replace").strip() if ok else ""
        return name or None

    def set_calibration_file(self, path, enabled=True):
        """Apply Thorlabs' per-serial leadscrew error map.

        Takes this stage's absolute on-axis accuracy from ~47 um to <+/-5 um.
        The file is specific to THIS serial number -- applying another stage's
        file makes accuracy worse, not better.
        """
        self._require_open()
        if enabled and not os.path.exists(path):
            raise StageError(f"calibration file not found: {path}")
        dll().ISC_SetCalibrationFile(self._sb, str(path).encode(), bool(enabled))


# ---------------------------------------------------------------------------
# the set of axes
# ---------------------------------------------------------------------------

class StageSet:
    """Named axes ('x', 'y', 'z') over however many stages are present.

    Nothing here assumes three. Two axes is a valid map; the GUI just offers
    fewer columns.
    """

    def __init__(self, axes, interlock=None, home_order=()):
        self.axes = dict(axes)          # name -> Stage
        # One latch for the whole machine, handed to every axis. A per-axis
        # interlock would stop the axis that faulted and leave the other two
        # running, which on a stacked rig is the arrangement most likely to
        # turn one fault into a collision.
        self.interlock = interlock if interlock is not None else MotionInterlock()
        for st in self.axes.values():
            st.interlock = self.interlock
        self.home_order = tuple(home_order)

    # ---- construction ----

    @classmethod
    def from_config(cls, path=None, serials=None):
        """Build from stages.json, falling back to bus order.

        stages.json looks like:

            {"axes":  {"x": "45502844", "y": "45538374", "z": "45502854"},
             "frame": {"z": {"invert": true}}}

        Without it the axes are named by serial number, because guessing which
        physical direction a serial corresponds to would be worse than not
        naming them: a wrong guess produces a coordinate frame that is silently
        transposed, and a transposed field map looks entirely plausible.

        "frame" carries the mounting, axis by axis -- see Stage. An axis whose
        bracket runs it backwards is declared once here rather than corrected
        at each call site.
        """
        path = path or paths.config(AXIS_CONFIG)
        found = list(serials) if serials is not None else list_devices()
        if not found:
            hint = (" -- the Kinesis application is running and holds all "
                    "devices; close it first" if kinesis_is_running()
                    else " -- check power and USB")
            raise StageError(f"no Thorlabs stages found{hint}")
        mapping = load_axis_map(path)
        if mapping:
            missing = [s for s in mapping.values() if s not in found]
            if missing:
                raise StageError(
                    f"{path} maps axes to stages that are not on the bus: "
                    f"{', '.join(missing)} (present: {', '.join(found)})")
            frames = load_axis_frames(path)
            motion = load_axis_motion(path, mapping)
            axes = {}
            for name, serial in mapping.items():
                fr = frames.get(name, {})
                vel, acc = motion[name]
                axes[name] = Stage(serial, name=name,
                                   invert=fr.get("invert", False),
                                   origin_mm=fr.get("origin_mm"),
                                   limit_mm=fr.get("limit_mm"),
                                   vel_mm_s=vel, accel_mm_s2=acc)
            return cls(axes, home_order=load_home_order(path, mapping))
        axes = {s: Stage(s, name=s) for s in found}
        return cls(axes)

    # ---- lifecycle ----

    def open(self):
        opened = []
        try:
            for st in self.axes.values():
                st.open()
                opened.append(st)
        except Exception:
            for st in opened:
                st.close()
            raise
        return self

    def close(self):
        for st in self.axes.values():
            st.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False

    def __getitem__(self, name):
        return self.axes[name]

    def __iter__(self):
        return iter(self.axes.values())

    @property
    def names(self):
        return list(self.axes)

    # ---- collective motion ----

    def snapshot(self):
        return {name: st.snapshot() for name, st in self.axes.items()}

    @property
    def homed(self):
        return all(st.homed for st in self.axes.values())

    @property
    def trusted(self):
        """True only if every axis's position counter can be believed."""
        return all(st.position_trusted for st in self.axes.values())

    def untrusted(self):
        """[(axis, why)] for every axis whose position cannot be believed."""
        return [(n, st.distrust_reason) for n, st in self.axes.items()
                if not st.position_trusted]

    @property
    def moving(self):
        return any(st.moving for st in self.axes.values())

    def position(self):
        return {name: st.position_mm for name, st in self.axes.items()}

    def home_sequence(self):
        """The axes in the order they should be homed, safest first.

        Order matters on a stacked rig and simultaneity hides that. Homing
        drives the full travel into a hard stop, so if the head has to be
        retracted before the other axes can sweep without fouling something,
        that retraction has to HAPPEN FIRST -- not at the same time, which is
        a race whose outcome depends on which axis is slower today.

        `home_order` in stages.json declares it. Axes it does not name follow,
        in map order, so adding a fourth axis cannot silently drop it.
        """
        named = [n for n in self.home_order if n in self.axes]
        return named + [n for n in self.axes if n not in named]

    def home_all(self, timeout_s=180.0, progress=None):
        """Home every axis, one at a time, in home_sequence() order.

        Sequential, where this used to start all three together. Three axes
        homing at once is three carriages sweeping their whole travel with no
        ordering between them -- fine on a rig where nothing can foul anything,
        which is a property of the fixture and the cable dress on the day, not
        of the software. It costs a minute; the failure it avoids costs a probe
        head. Declare home_order in stages.json to control the sequence.
        """
        self.interlock.require_clear("homing")
        for name in self.home_sequence():
            st = self.axes[name]
            st.home(wait=False)
            st.wait(timeout_s=timeout_s, what="homing")
            st.trust_after_homing()
            if progress:
                progress(name, st.position_mm)

    def move_to(self, timeout_s=None, settle_s=0.0, **coords):
        """move_to(x=10, y=20) -- start all named axes, then wait for all.

        Simultaneous rather than sequential: for a raster this roughly halves
        the dead time between captures, and the axes cannot foul each other
        (they are stacked, not sharing a workspace).
        """
        unknown = set(coords) - set(self.axes)
        if unknown:
            raise StageError(f"no such axis: {', '.join(sorted(unknown))}")
        moving = []
        try:
            for name, mm in coords.items():
                if mm is None:
                    continue
                self.axes[name].move_to(mm, wait=False)
                moving.append(self.axes[name])
        except Exception:
            # Whatever refused the second axis -- out of limits, a latched
            # interlock, a DLL error -- the first one is already travelling,
            # and nothing above here is going to wait for it. A raster catches
            # this exception, records a failed point and commands the NEXT
            # move, so without this the machine would be part way through a
            # move nobody is tracking while a new one is issued on top.
            for st in moving:
                with contextlib.suppress(Exception):
                    st.stop()
            raise
        for st in moving:
            st.wait(timeout_s=timeout_s, what="move")
        if settle_s:
            # One settle for the whole group, after the last axis stops. The
            # stages ring mechanically after a move and the probe is on the end
            # of that ring, so this is what makes a capture stationary.
            time.sleep(settle_s)

    def stop_all(self, immediate=False):
        errors = []
        for st in self.axes.values():
            try:
                st.stop(immediate=immediate)
            except StageError as exc:
                errors.append(str(exc))
        if errors:
            raise StageError("; ".join(errors))

    # ---- emergency stop ----

    def emergency_stop(self, reason="emergency stop"):
        """Stop every axis immediately and latch the machine off.

        Never raises. This is called from a button handler, from a status
        watchdog and from a worker's failure path, and in every one of those
        an exception part way through the axes would leave the rest running --
        which is precisely the outcome the call exists to prevent. Errors come
        back as a list of strings for the caller to log.

        Not a substitute for a hardware emergency stop. This asks the
        controllers to stop over USB; it depends on this process running, the
        USB link being up and the controllers being responsive, none of which
        is true in the failures a real E-stop exists for. A category-0 stop is
        a mushroom head in series with the motor supply, and nothing in
        software replaces it -- see EMERGENCY STOP in the module docstring.
        """
        errors = [st.emergency_stop(reason) for st in self.axes.values()]
        # Trip even with no axes open, so the latch is set for whatever opens
        # next rather than only for what happened to be connected.
        self.interlock.trip(reason)
        return [e for e in errors if e]

    def reset_interlock(self):
        """Clear the latch. Returns (was_tripped_reason, [(axis, why)]).

        The second half is the axes whose position is no longer trustworthy
        after whatever tripped it -- an immediate stop can lose steps, so
        clearing the latch does NOT restore absolute moves. Those come back
        when the axis is homed, and the caller is expected to say so.
        """
        return (self.interlock.reset(), self.untrusted())


# ---------------------------------------------------------------------------
# axis map persistence
# ---------------------------------------------------------------------------

def load_axis_map(path=None):
    path = path or paths.config(AXIS_CONFIG)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return {str(k): str(v) for k, v in doc.get("axes", {}).items()}


def load_axis_frames(path=None):
    """axis -> {"invert": bool, "origin_mm": float|None, "limit_mm": pair|None}.

    Kept beside the axis map rather than derived from it because which way a
    stage runs is a fact about the BRACKET, not about the stage: swap two
    controllers over and the serials move, the mounting does not. "limit_mm"
    is the same kind of fact -- the working envelope belongs to the fixture,
    not to the leadscrew.
    """
    path = path or paths.config(AXIS_CONFIG)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    out = {}
    for name, raw_spec in (doc.get("frame") or {}).items():
        # shorthand: {"z": true}
        spec = {"invert": raw_spec} if isinstance(raw_spec, bool) else raw_spec
        origin = spec.get("origin_mm")
        limit = spec.get("limit_mm")
        if limit is not None:
            if len(limit) != 2:
                raise StageError(
                    f"{path}: frame.{name}.limit_mm must be [low, high], "
                    f"got {limit!r}")
            limit = (float(limit[0]), float(limit[1]))
        out[str(name)] = {
            "invert": bool(spec.get("invert", False)),
            "origin_mm": None if origin is None else float(origin),
            "limit_mm": limit,
        }
    return out


def load_home_order(path=None, axes=()):
    """The order home_all() should reference the axes in, safest first.

    Declared rather than inferred: which axis has to retract before the others
    may sweep is a property of the fixture on the bench, and no amount of
    reading the config can work it out. An axis named here that is not on the
    machine is ignored rather than an error -- the safe order for a three-axis
    rig should stay valid when you unplug one to work on it.
    """
    path = path or paths.config(AXIS_CONFIG)
    if not os.path.exists(path):
        return ()
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    order = doc.get("home_order") or ()
    known = set(axes) if axes else None
    return tuple(str(n) for n in order if known is None or n in known)


def load_axis_motion(path=None, axes=()):
    """axis -> (velocity_mm_s, accel_mm_s2), the profile each axis opens with.

    stages.json carries it as

        "motion": {"velocity_mm_s": 8.0, "accel_mm_s2": 10.0,
                   "axes": {"z": {"velocity_mm_s": 5.0}}}

    The block applies to every axis, and "axes" overrides one of them -- z
    lifts the head against gravity and is the one most likely to want its own
    number. Anything left unsaid falls back to the module defaults, so a rig
    with no stages.json still opens quiet rather than at Kinesis's 20 mm/s.
    """
    path = path or paths.config(AXIS_CONFIG)
    doc = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    block = doc.get("motion") or {}
    per_axis = block.get("axes") or {}

    def _pick(spec, key, fallback):
        val = spec.get(key)
        return fallback if val is None else float(val)

    base_v = _pick(block, "velocity_mm_s", DEFAULT_VEL_MM_S)
    base_a = _pick(block, "accel_mm_s2", DEFAULT_ACCEL_MM_S2)
    out = {}
    for name in axes:
        spec = per_axis.get(name) or {}
        # Clamped on the way in, so a stages.json written before the ceiling
        # existed -- or edited by hand -- cannot reintroduce the shipped
        # 20 mm/s through the back door.
        out[str(name)] = (clamp_velocity(_pick(spec, "velocity_mm_s", base_v)),
                          _pick(spec, "accel_mm_s2", base_a))
    return out


def save_axis_motion(path=None, velocity_mm_s=None, accel_mm_s2=None,
                     axis=None):
    """Write the motion profile into stages.json, leaving the rest of it alone.

    With no axis this sets the block that applies to everything; with one it
    writes an override for that axis only.
    """
    path = path or paths.config(AXIS_CONFIG)
    doc = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    block = doc.setdefault("motion", {})
    target = block if axis is None else block.setdefault("axes", {}).setdefault(
        str(axis), {})
    if velocity_mm_s is not None:
        target["velocity_mm_s"] = clamp_velocity(velocity_mm_s)
    if accel_mm_s2 is not None:
        target["accel_mm_s2"] = float(accel_mm_s2)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    return path


def save_axis_map(mapping, path=None, frames=None):
    path = path or paths.config(AXIS_CONFIG)
    doc = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    doc["axes"] = {str(k): str(v) for k, v in mapping.items()}
    if frames is not None:
        existing = doc.get("frame") or {}
        out = {}
        for name, spec in frames.items():
            # Merged onto what is already there, not written over it. The
            # caller of this is the GUI's "Save axis map", which knows about
            # inversion and nothing about soft limits -- and a save that
            # quietly deleted the working envelope would remove the only thing
            # keeping the head out of the fixture, at the moment someone was
            # doing routine housekeeping.
            prev = existing.get(str(name))
            entry = dict(prev) if isinstance(prev, dict) else {}
            entry["invert"] = bool(spec.get("invert", False))
            if spec.get("origin_mm") is not None:
                entry["origin_mm"] = float(spec["origin_mm"])
            if spec.get("limit_mm") is not None:
                entry["limit_mm"] = [float(spec["limit_mm"][0]),
                                     float(spec["limit_mm"][1])]
            out[str(name)] = entry
        doc["frame"] = out
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_list(a):
    serials = list_devices()
    if not serials:
        if kinesis_is_running():
            print("No stages visible -- the Kinesis application is running and "
                  "holds all of them.\nClose Kinesis and try again.")
        else:
            print("No stages visible. Check power and USB.")
        return 1
    mapping = {v: k for k, v in load_axis_map(a.config).items()}
    frames = load_axis_frames(a.config)
    print(f"{len(serials)} integrated-stepper controller(s):\n")
    for s in serials:
        info = device_info(s)
        axis = mapping.get(s)
        fr = frames.get(axis or "", {})
        label = f"  axis '{axis}'" if axis else "  (unmapped)"
        if fr.get("invert"):
            label += " REVERSED"
        try:
            # Built with the frame, so a mapped axis reports the same number
            # here as it does everywhere else. Listing device coordinates on
            # one screen and rig coordinates on another is how a reversed
            # axis gets "fixed" twice.
            # vel/accel None: listing must not change how a stage moves.
            with Stage(s, name=axis or s, invert=fr.get("invert", False),
                       origin_mm=fr.get("origin_mm"),
                       vel_mm_s=None, accel_mm_s2=None) as st:
                snap = st.snapshot()
                pos = (f"{snap['position_mm']:8.3f} mm" if snap["homed"]
                       else f"{snap['position_mm']:8.3f} mm (NOT HOMED)")
                print(f"{s}  {st.model:<8}{label}")
                print(f"    travel {st.travel_mm[0]:g}..{st.travel_mm[1]:g} mm"
                      f"   position {pos}")
                vel, acc = st.vel_params
                print(f"    flags: {', '.join(snap['flags']) or 'none'}")
                print(f"    motion: {vel:g} mm/s, {acc:g} mm/s^2")
                cal = st.calibration_file()
                # Built outside the f-string: a line break inside a
                # replacement field is Python 3.12+ syntax, and this project
                # supports 3.10.
                cal_note = cal or ("NONE -- on-axis accuracy ~47 um instead "
                                   "of <5 um")
                print(f"    calibration file: {cal_note}")
        except StageError as exc:
            print(f"{s}  {info['description']}{label}\n    {exc}")
    return 0


def _cmd_status(a):
    with StageSet.from_config(a.config) as ss:
        for name, snap in ss.snapshot().items():
            state = "moving" if snap["moving"] else "idle"
            if not snap["homed"]:
                state += ", NOT HOMED"
            if snap["error"]:
                state += ", MOTION ERROR"
            if snap["at_hard_limit"]:
                state += ", ON A HARD LIMIT"
            frame = ("" if snap["frame"] == "as mounted"
                     else f"  ({snap['frame']}, device "
                          f"{snap['position_dev_mm']:.4f} mm)")
            vel, acc = ss[name].vel_params
            print(f"{name:>4}  {snap['position_mm']:9.4f} mm  "
                  f"[{snap['serial']} {snap['model']}]  {state}{frame}")
            print(f"      {vel:g} mm/s, {acc:g} mm/s^2  -- a 5 mm step peaks "
                  f"at {ss[name].peak_speed_mm_s(5):.1f} mm/s, "
                  f"20 mm at {ss[name].peak_speed_mm_s(20):.1f}")
            lo, hi = snap["limit_mm"]
            envelope = f"      may use {lo:g}..{hi:g} mm"
            if snap["limit_mm"] != snap["travel_mm"]:
                envelope += (f"  (soft limit inside a "
                             f"{snap['travel_mm'][0]:g}.."
                             f"{snap['travel_mm'][1]:g} mm travel)")
            elif snap["limit_declared"]:
                envelope += ("  -- the whole travel, declared: this axis has "
                             "been checked and has nothing in its way")
            else:
                envelope += ("  -- THE WHOLE TRAVEL, and no envelope has been "
                             "declared for this axis, so nothing but the limit "
                             "switches stops it short of the fixture")
            print(envelope)
            if not snap["trusted"]:
                print(f"      absolute moves REFUSED: {snap['distrust_reason']}")
        print(f"\nhome order: {' -> '.join(ss.home_sequence())}")
        if ss.interlock.tripped:
            print(f"MOTION IS LATCHED OFF: {ss.interlock.tripped}")
    return 0


def _cmd_home(a):
    with StageSet.from_config(a.config) as ss:
        wanted = set(a.axis or ss.names)
        # In the declared safe order even when the caller named a subset, so
        # `home --axis x --axis z` cannot become the one ordering that fouls.
        order = [n for n in ss.home_sequence() if n in wanted]
        names = ", ".join(order)
        if not a.yes:
            print(f"Homing drives {names} into the limit switch at full "
                  f"homing speed, one axis at a time in that order.\nMake "
                  f"sure the probe head and its cabling are clear of the "
                  f"whole travel.")
            if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("aborted")
                return 1
        for name in order:
            st = ss[name]
            st.home(wait=False)
            st.wait(timeout_s=a.timeout, what="homing")
            st.trust_after_homing()
            print(f"{st.name}: homed, at {st.position_mm:.4f} mm")
    return 0


def _cmd_moveto(a):
    coords = {k: v for k, v in (("x", a.x), ("y", a.y), ("z", a.z))
              if v is not None}
    if not coords:
        print("nothing to do: give at least one of --x/--y/--z")
        return 1
    with StageSet.from_config(a.config) as ss:
        ss.move_to(settle_s=a.settle, **coords)
        print("  ".join(f"{n}={p:.4f}" for n, p in ss.position().items()))
    return 0


def _cmd_moveby(a):
    coords = {k: v for k, v in (("x", a.x), ("y", a.y), ("z", a.z))
              if v is not None}
    if not coords:
        print("nothing to do: give at least one of --x/--y/--z")
        return 1
    with StageSet.from_config(a.config) as ss:
        moving = []
        for name, delta in coords.items():
            ss[name].move_by(delta, wait=False)
            moving.append(ss[name])
        for st in moving:
            st.wait(what="relative move")
        print("  ".join(f"{n}={p:.4f}" for n, p in ss.position().items()))
    return 0


def _cmd_identify(a):
    """Wiggle one stage so you can see which physical axis it is.

    Relative, small, and returned to where it started, so it works on an
    unhomed stage and leaves the counter as it found it.
    """
    serials = list_devices()
    targets = [a.serial] if a.serial else serials
    for s in targets:
        if s not in serials:
            print(f"stage {s} is not on the bus")
            return 1
        with Stage(s) as st:
            print(f"{s} ({st.model}): moving +/-{a.distance:g} mm -- watch which "
                  f"axis moves")
            st.move_by(a.distance)
            st.move_by(-a.distance)
            print(f"  back at {st.position_mm:.4f} mm")
        if (len(targets) > 1 and not a.yes
                and input("next stage? [Y/n] ").strip().lower() in ("n", "no")):
            break
    return 0


def _cmd_stop(a):
    with StageSet.from_config(a.config) as ss:
        ss.stop_all(immediate=a.immediate)
        print("stopped")
    return 0


def _cmd_estop(a):
    """Stop everything now, from a second terminal.

    The GUI's button is the one to reach for while the GUI is up. This is for
    when it is not -- a scan started from the command line, a wedged window,
    a script that got away. It opens the devices, stops them immediately and
    exits.

    What it does NOT do is latch: the interlock lives in a process, and this
    process is about to end. Whatever is still running keeps its own idea of
    whether it may move, so stop that too. The axes come back untrusted
    either way and want re-homing before any absolute move.
    """
    errors = []
    try:
        with StageSet.from_config(a.config) as ss:
            errors = ss.emergency_stop("emergency stop from the command line")
            print(f"EMERGENCY STOP: {', '.join(ss.names)} stopped immediately")
    except StageError as exc:
        print(f"EMERGENCY STOP FAILED TO REACH THE STAGES: {exc}",
              file=sys.stderr)
        print("If anything is moving, cut power to the controllers.",
              file=sys.stderr)
        return 2
    for e in errors:
        print(f"  {e}", file=sys.stderr)
    print("Steps may have been lost -- re-home every axis before any "
          "absolute move.")
    print("This did not latch anything off: stop whatever commanded the "
          "move as well.")
    return 1 if errors else 0


def _cmd_speed(a):
    """Show or change the motion profile -- the cure for a howling axis."""
    with StageSet.from_config(a.config) as ss:
        names = a.axis or ss.names
        unknown = [n for n in names if n not in ss.names]
        if unknown:
            print(f"no such axis: {', '.join(unknown)} "
                  f"(have {', '.join(ss.names)})")
            return 1
        changing = a.vel is not None or a.accel is not None
        if a.vel is not None and a.vel > MAX_VEL_MM_S:
            print(f"{a.vel:g} mm/s is above the {MAX_VEL_MM_S:g} mm/s ceiling "
                  f"in this module -- using {MAX_VEL_MM_S:g}")
            a.vel = MAX_VEL_MM_S
        for name in names:
            st = ss[name]
            if changing:
                st.set_vel_params(a.vel, a.accel)
            vel, acc = st.vel_params
            print(f"{name:>4}  {vel:g} mm/s, {acc:g} mm/s^2")
            # The table, not the setting, is what answers "why is 10 mm loud".
            peaks = "  ".join(f"{d:g} mm -> {st.peak_speed_mm_s(d):.1f}"
                              for d in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0))
            print(f"      peak speed by step size, mm/s:  {peaks}")
        if changing and a.save:
            for name in (names if a.axis else (None,)):
                save_axis_motion(a.config, a.vel, a.accel, axis=name)
            where = f"axis {', '.join(names)}" if a.axis else "all axes"
            print()
            print(f"wrote {a.config} ({where}) -- this is now what every "
                  f"session opens with")
        elif changing:
            print()
            print("not saved: this lasts until the controller is power "
                  "cycled. Add --save to make it the default.")
    return 0


def _cmd_map(a):
    serials = list_devices()
    mapping = dict(zip(a.axes.split(","), serials)) if a.axes else {}
    if a.assign:
        mapping = {}
        for item in a.assign:
            name, _, serial = item.partition("=")
            if serial not in serials:
                print(f"stage {serial} is not on the bus "
                      f"(present: {', '.join(serials)})")
                return 1
            mapping[name] = serial
    if not mapping:
        print("give --assign x=45502844 --assign y=... to set the axis map")
        return 1

    # Start from what is already recorded so that setting the map does not
    # quietly drop the mounting, and vice versa.
    frames = load_axis_frames(a.config)
    for name in mapping:
        frames.setdefault(name, {"invert": False, "origin_mm": None})
    for name in (a.invert or []):
        if name not in mapping:
            print(f"--invert {name}: no such axis in the map")
            return 1
        frames[name]["invert"] = True
    for name in (a.forward or []):
        if name not in mapping:
            print(f"--forward {name}: no such axis in the map")
            return 1
        frames[name]["invert"] = False
    for item in (a.origin or []):
        name, _, val = item.partition("=")
        if name not in mapping:
            print(f"--origin {name}: no such axis in the map")
            return 1
        frames[name]["origin_mm"] = None if val in ("", "auto") else float(val)
    frames = {k: v for k, v in frames.items() if k in mapping}

    save_axis_map(mapping, a.config, frames=frames)
    print(f"wrote {a.config}")
    for name, serial in mapping.items():
        fr = frames[name]
        how = "REVERSED" if fr["invert"] else "forward"
        origin = ("auto" if fr["origin_mm"] is None
                  else f"{fr['origin_mm']:g} mm")
        print(f"  {name} = {serial}   {how}, rig zero at device {origin}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=paths.config(AXIS_CONFIG),
                   help=f"axis -> serial map (default {AXIS_CONFIG})")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="every controller on the USB bus").set_defaults(
        func=_cmd_list)
    sub.add_parser("status", help="position and flags per mapped axis"
                   ).set_defaults(func=_cmd_status)

    ph = sub.add_parser("home", help="reference an axis against its limit switch")
    ph.add_argument("--axis", action="append",
                    help="axis name; repeatable. Default: all")
    ph.add_argument("--timeout", type=float, default=180.0)
    ph.add_argument("--yes", action="store_true",
                    help="skip the collision-clearance confirmation")
    ph.set_defaults(func=_cmd_home)

    pm = sub.add_parser("moveto", help="absolute move, mm")
    for ax in "xyz":
        pm.add_argument(f"--{ax}", type=float)
    pm.add_argument("--settle", type=float, default=0.0,
                    help="seconds to wait after the last axis stops")
    pm.set_defaults(func=_cmd_moveto)

    pr = sub.add_parser("moveby", help="relative move, mm (works unhomed)")
    for ax in "xyz":
        pr.add_argument(f"--{ax}", type=float)
    pr.set_defaults(func=_cmd_moveby)

    pi = sub.add_parser("identify",
                        help="wiggle a stage to see which axis it is")
    pi.add_argument("--serial", help="one stage; default walks all of them")
    pi.add_argument("--distance", type=float, default=5.0,
                    help="mm to move out and back (default 5)")
    pi.add_argument("--yes", action="store_true", help="do not pause between stages")
    pi.set_defaults(func=_cmd_identify)

    ps = sub.add_parser("stop", help="stop all axes")
    ps.add_argument("--immediate", action="store_true",
                    help="abrupt stop; loses steps, so re-home afterwards")
    ps.set_defaults(func=_cmd_stop)

    sub.add_parser(
        "estop",
        help="EMERGENCY STOP: stop every axis immediately, from another "
             "terminal").set_defaults(func=_cmd_estop)

    pv = sub.add_parser("speed", help="show or set the motion profile")
    pv.add_argument("--vel", type=float, metavar="MM_S",
                    help=f"maximum velocity, mm/s (default "
                         f"{DEFAULT_VEL_MM_S:g}, hard ceiling "
                         f"{MAX_VEL_MM_S:g})")
    pv.add_argument("--accel", type=float, metavar="MM_S2",
                    help=f"acceleration, mm/s^2 (default "
                         f"{DEFAULT_ACCEL_MM_S2:g})")
    pv.add_argument("--axis", action="append",
                    help="axis name; repeatable. Default: all")
    pv.add_argument("--save", action="store_true",
                    help="write it to the config so every session starts here")
    pv.set_defaults(func=_cmd_speed)

    pp = sub.add_parser("map", help="record which stage is which axis")
    pp.add_argument("--assign", action="append", metavar="AXIS=SERIAL")
    pp.add_argument("--axes", help="comma-separated names, in bus order")
    pp.add_argument("--invert", action="append", metavar="AXIS",
                    help="this axis is mounted backwards: its limit switch is "
                         "at the far end of the rig axis, so rig zero sits at "
                         "the top of its travel. Repeatable.")
    pp.add_argument("--forward", action="append", metavar="AXIS",
                    help="clear --invert for this axis")
    pp.add_argument("--origin", action="append", metavar="AXIS=MM",
                    help="device position that rig zero sits at; 'auto' picks "
                         "the far end when inverted and 0 when not")
    pp.set_defaults(func=_cmd_map)

    a = p.parse_args(argv)
    try:
        return a.func(a)
    except StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
