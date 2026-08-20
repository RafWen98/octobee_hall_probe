#!/usr/bin/env python3
"""
octobee_stage.py -- drive the Thorlabs LTS300C translation stages directly,
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
floor that octobee_posecap.py works so hard to reach. Once you have the
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

Which way is up is not the stage's business
-------------------------------------------
A stage homes to its own limit switch, calls that 0 and counts up to 300.
Which physical direction that runs in depends on how the bracket was bolted
on, and the controller has no idea. On this rig z is mounted reversed, so its
home is the TOP of the working volume: left alone, every map would come out
mirrored along z and look entirely plausible. See Stage for the frame that
fixes it, declared once in stages.json rather than as a minus sign scattered
over the call sites.

Usage
-----
    python octobee_stage.py list                    # what is on the USB bus
    python octobee_stage.py status                  # per-axis position + flags
    python octobee_stage.py home --axis x           # explicit, one axis
    python octobee_stage.py moveto --x 100 --y 50   # absolute, mm
    python octobee_stage.py stop                    # profiled stop, all axes

    python octobee_stage.py map --assign x=45502844 --assign z=45502854 \\
                               --assign y=45538374 --invert z

Axis names and mounting come from stages.json (see AXIS_CONFIG); without it the
stages are addressed by serial number and assumed to be mounted forwards.
"""

import argparse
import ctypes as C
import json
import os
import sys
import time

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


class StageError(RuntimeError):
    """Any failure talking to a stage, including a non-zero API return."""


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
    for name, (argtypes, restype) in sig.items():
        fn = getattr(dll, name)
        fn.argtypes = argtypes
        fn.restype = restype


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
    d.TLI_GetDeviceListExt(buf, 512)
    return [s for s in buf.value.decode(errors="replace").split(",") if s]


def kinesis_is_running():
    """True if the Kinesis GUI is up, which makes every stage unopenable."""
    try:
        import subprocess
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq Thorlabs.MotionControl.Kinesis.exe"],
            capture_output=True, text=True, timeout=10).stdout
        return "Thorlabs.MotionControl.Kinesis.exe" in out
    except Exception:
        return False


def device_info(serial):
    d = dll()
    info = TLI_DeviceInfo()
    d.TLI_GetDeviceInfo(serial.encode(), C.byref(info))
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
                 origin_mm=None):
        self.serial = str(serial)
        self._sb = self.serial.encode()
        self.name = name or self.serial
        self.poll_ms = poll_ms
        self.is_open = False
        self.model = ""
        self.travel_mm = (0.0, 0.0)          # rig frame
        self.travel_dev_mm = (0.0, 0.0)      # what the stage itself reports
        self.invert = bool(invert)
        # None means "resolve from travel once the stage tells us what it is".
        self._origin_cfg = origin_mm
        self.origin_mm = 0.0 if origin_mm is None else float(origin_mm)
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
        except Exception:
            self.close()
            raise
        return self

    def close(self):
        if not self.is_open:
            return
        d = dll()
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

    def _require_open(self):
        if not self.is_open:
            raise StageError(f"stage {self.name} is not open")

    # ---- units ----

    def _to_du(self, mm):
        return int(round(mm * self._du_per_mm))

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
    def velocity_mm_s(self):
        self._require_open()
        acc, vel = C.c_int(), C.c_int()
        dll().ISC_GetVelParams(self._sb, C.byref(acc), C.byref(vel))
        return self._real(vel.value, UNIT_VELOCITY)

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
            "travel_mm": self.travel_mm,
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
        d = dll()
        d.ISC_ClearMessageQueue(self._sb)
        _check(d.ISC_Home(self._sb), f"home {self.name}")
        if wait:
            self.wait(timeout_s=timeout_s, what="homing")

    def move_to(self, mm, timeout_s=180.0, wait=True):
        """Absolute move to RIG millimetres, against the axis travel limits."""
        self._require_open()
        lo, hi = self.travel_mm
        if hi > lo and not (lo <= mm <= hi):
            raise StageError(
                f"{self.name}: {mm:g} mm is outside travel {lo:g}..{hi:g} mm")
        if not self.homed:
            raise StageError(
                f"{self.name} has not been homed -- its position counter is "
                f"unreferenced, so an absolute move would go somewhere "
                f"arbitrary. Home it first.")
        dev = self._rig_to_dev(mm)
        d = dll()
        d.ISC_ClearMessageQueue(self._sb)
        _check(d.ISC_MoveToPosition(self._sb, self._to_du(dev)),
               f"move {self.name} to {mm:g} mm")
        if wait:
            self.wait(timeout_s=timeout_s, what=f"move to {mm:g} mm")

    def move_by(self, delta_mm, timeout_s=180.0, wait=True):
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
        lo, hi = self.travel_mm
        target = self.position_mm + delta_mm
        if hi > lo and not (lo <= target <= hi) and self.homed:
            raise StageError(
                f"{self.name}: {target:g} mm is outside travel "
                f"{lo:g}..{hi:g} mm")
        d = dll()
        d.ISC_ClearMessageQueue(self._sb)
        # delta_mm is a RIG distance; on a reverse-mounted axis the device has
        # to travel the other way to produce it.
        _check(d.ISC_MoveRelative(self._sb, self._to_du(self._sign * delta_mm)),
               f"move {self.name} by {delta_mm:g} mm")
        if wait:
            self.wait(timeout_s=timeout_s, what=f"move by {delta_mm:g} mm")

    def stop(self, immediate=False):
        """Profiled stop by default; immediate loses steps and thus position."""
        self._require_open()
        d = dll()
        rc = (d.ISC_StopImmediate(self._sb) if immediate
              else d.ISC_StopProfiled(self._sb))
        _check(rc, f"stop {self.name}")

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
                raise StageError(f"{self.name}: motion error during {what}")
            if not bits & MOVING_MASK:
                break
            if time.monotonic() > deadline:
                self.stop()
                raise StageError(
                    f"{self.name}: {what} did not finish within {timeout_s:g} s")
            time.sleep(0.02)
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

    def __init__(self, axes):
        self.axes = dict(axes)          # name -> Stage

    # ---- construction ----

    @classmethod
    def from_config(cls, path=AXIS_CONFIG, serials=None):
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
            axes = {}
            for name, serial in mapping.items():
                fr = frames.get(name, {})
                axes[name] = Stage(serial, name=name,
                                   invert=fr.get("invert", False),
                                   origin_mm=fr.get("origin_mm"))
        else:
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
    def moving(self):
        return any(st.moving for st in self.axes.values())

    def position(self):
        return {name: st.position_mm for name, st in self.axes.items()}

    def home_all(self, timeout_s=180.0):
        """Home every axis at once, then wait for all of them.

        Started together rather than in sequence because homing is slow and the
        axes are mechanically independent -- but see the GUI's confirmation
        step: three axes homing simultaneously is three chances to collide.
        """
        for st in self.axes.values():
            st.home(wait=False)
        for st in self.axes.values():
            st.wait(timeout_s=timeout_s, what="homing")

    def move_to(self, timeout_s=180.0, settle_s=0.0, **coords):
        """move_to(x=10, y=20) -- start all named axes, then wait for all.

        Simultaneous rather than sequential: for a raster this roughly halves
        the dead time between captures, and the axes cannot foul each other
        (they are stacked, not sharing a workspace).
        """
        unknown = set(coords) - set(self.axes)
        if unknown:
            raise StageError(f"no such axis: {', '.join(sorted(unknown))}")
        moving = []
        for name, mm in coords.items():
            if mm is None:
                continue
            self.axes[name].move_to(mm, wait=False)
            moving.append(self.axes[name])
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


# ---------------------------------------------------------------------------
# axis map persistence
# ---------------------------------------------------------------------------

def load_axis_map(path=AXIS_CONFIG):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        doc = json.load(fh)
    return {str(k): str(v) for k, v in doc.get("axes", {}).items()}


def load_axis_frames(path=AXIS_CONFIG):
    """axis -> {"invert": bool, "origin_mm": float|None}.

    Kept beside the axis map rather than derived from it because which way a
    stage runs is a fact about the BRACKET, not about the stage: swap two
    controllers over and the serials move, the mounting does not.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        doc = json.load(fh)
    out = {}
    for name, spec in (doc.get("frame") or {}).items():
        if isinstance(spec, bool):          # shorthand: {"z": true}
            spec = {"invert": spec}
        origin = spec.get("origin_mm")
        out[str(name)] = {
            "invert": bool(spec.get("invert", False)),
            "origin_mm": None if origin is None else float(origin),
        }
    return out


def save_axis_map(mapping, path=AXIS_CONFIG, frames=None):
    doc = {}
    if os.path.exists(path):
        with open(path) as fh:
            doc = json.load(fh)
    doc["axes"] = {str(k): str(v) for k, v in mapping.items()}
    if frames is not None:
        out = {}
        for name, spec in frames.items():
            entry = {"invert": bool(spec.get("invert", False))}
            if spec.get("origin_mm") is not None:
                entry["origin_mm"] = float(spec["origin_mm"])
            out[str(name)] = entry
        doc["frame"] = out
    with open(path, "w") as fh:
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
            with Stage(s, name=axis or s, invert=fr.get("invert", False),
                       origin_mm=fr.get("origin_mm")) as st:
                snap = st.snapshot()
                pos = (f"{snap['position_mm']:8.3f} mm" if snap["homed"]
                       else f"{snap['position_mm']:8.3f} mm (NOT HOMED)")
                print(f"{s}  {st.model:<8}{label}")
                print(f"    travel {st.travel_mm[0]:g}..{st.travel_mm[1]:g} mm"
                      f"   position {pos}")
                print(f"    flags: {', '.join(snap['flags']) or 'none'}")
                cal = st.calibration_file()
                print(f"    calibration file: {cal or 'NONE -- on-axis accuracy '
                                                     '~47 um instead of <5 um'}")
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
            frame = ("" if snap["frame"] == "as mounted"
                     else f"  ({snap['frame']}, device "
                          f"{snap['position_dev_mm']:.4f} mm)")
            print(f"{name:>4}  {snap['position_mm']:9.4f} mm  "
                  f"[{snap['serial']} {snap['model']}]  {state}{frame}")
    return 0


def _cmd_home(a):
    with StageSet.from_config(a.config) as ss:
        targets = [ss[n] for n in (a.axis or ss.names)]
        names = ", ".join(st.name for st in targets)
        if not a.yes:
            print(f"Homing drives {names} into the limit switch at full "
                  f"homing speed.\nMake sure the probe head and its cabling "
                  f"are clear of the whole travel.")
            if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("aborted")
                return 1
        for st in targets:
            st.home(wait=False)
        for st in targets:
            st.wait(timeout_s=a.timeout, what="homing")
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
        if len(targets) > 1 and not a.yes:
            if input("next stage? [Y/n] ").strip().lower() in ("n", "no"):
                break
    return 0


def _cmd_stop(a):
    with StageSet.from_config(a.config) as ss:
        ss.stop_all(immediate=a.immediate)
        print("stopped")
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
    p.add_argument("--config", default=AXIS_CONFIG,
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
