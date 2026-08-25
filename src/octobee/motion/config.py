"""
octobee/motion/config.py -- reading and writing stages.json.

Which serial number is on which axis, which way each one runs, what its working
envelope is, what order they may be homed in, and what motion profile they open
with. All of it is a fact about the RIG rather than about any stage, which is
why it is persisted next to the axis map rather than derived from the hardware:
swap two controllers over and the serials move, the mounting does not.
"""

import json
import os

from octobee import paths
from octobee.motion.kinesis import StageError
from octobee.motion.timing import (
    DEFAULT_ACCEL_MM_S2,
    DEFAULT_VEL_MM_S,
    clamp_velocity,
)

AXIS_CONFIG = "stages.json"

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
