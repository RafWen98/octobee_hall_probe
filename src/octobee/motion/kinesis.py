"""
octobee/motion/kinesis.py -- the Thorlabs Kinesis C API, bound with ctypes.

Everything that knows the DLL exists lives here: where it is installed, its
structures, the argtypes that stop a DWORD status word truncating silently,
the lazy handle, and the three discovery calls. `Stage` and `StageSet` next
door are written against this and never load anything themselves.

The separation is not tidiness. This is the one module in the project that
cannot run anywhere but a Windows bench with Kinesis installed -- so it is
also the one module worth being able to point at when something needs to run
without it. A simulated backend, if this rig ever wants one, replaces this
file and nothing else.

Why ctypes and not pythonnet: see the note at the top of stage.py.
"""

import ctypes as C
import os
import subprocess

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
HARD_LIMIT_MASK = 0x00000003    # either physical end-of-travel switch


class StageError(RuntimeError):
    """Any failure talking to a stage, including a non-zero API return."""


def check_rc(rc, what):
    """Raise on a non-zero Kinesis return code.

    Imported by stage.py under its old private name, which is what every call
    site there still uses.
    """
    if rc != 0:
        raise StageError(f"{what} failed (Kinesis error {rc})")


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
    check_rc(d.TLI_BuildDeviceList(), "TLI_BuildDeviceList")
    buf = C.create_string_buffer(512)
    # Checked, like the call above it. An unchecked failure here hands back an
    # empty buffer, which reads as "no stages on the bus" -- the same symptom
    # as the Kinesis-is-running case this module works hard to tell apart.
    check_rc(d.TLI_GetDeviceListExt(buf, 512), "TLI_GetDeviceListExt")
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
    # here returns 0 for success, so checking it with check_rc() reports a
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
