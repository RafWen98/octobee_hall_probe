"""The Thorlabs stages -- binding, framing and safety."""

import inspect
import json
import os


from octobee.motion import stage as ostage
from tests.helpers import (
    _framed,
    check,
    skip,
)



def test_stage_binding():
    """The ctypes binding, as far as it goes with no stage attached."""
    try:
        d = ostage.dll()
    except ostage.StageError as exc:
        # Not a failure. Kinesis is Windows-only proprietary software that a CI
        # runner does not have and cannot be given, and everything below this
        # point tests the binding AGAINST that DLL -- there is nothing left to
        # check without it. What matters is that the skip is visible, so a
        # bench run that lost its Kinesis install does not look like a pass.
        skip("the Kinesis C API loads", str(exc))
        return
    check("the Kinesis C API loads", d is not None)
    check("the binding declares argtypes for every call it makes",
          all(getattr(d, n).argtypes is not None
              for n in ("ISC_Open", "ISC_MoveToPosition", "ISC_GetStatusBits",
                        "ISC_MoveRelative", "TLI_GetDeviceListExt")),
          "an unbound DWORD return truncates the status word silently")
    check("the status word decodes the homed bit",
          "homed" in ostage.Stage("45000000").status_flags(
              ostage.HOMED_BIT | ostage.ENABLED_BIT))
    check("the status word decodes motion errors",
          "motion_error" in ostage.Stage("45000000").status_flags(
              ostage.MOTION_ERROR_BIT))

    st = ostage.Stage("45000000")
    check("device units round trip through millimetres",
          abs(st._to_mm(st._to_du(137.25)) - 137.25) < 1e-6)
    check("the LTS300C default scaling matches the measured 409600 du/mm",
          st._to_du(300.0) == 122880000, f"{st._to_du(300.0)}")

    for name, fn in (("position", lambda: st.position_mm),
                     ("status", lambda: st.status),
                     ("a move", lambda: st.move_to(10.0))):
        try:
            fn()
            ok = False
        except ostage.StageError:
            ok = True
        check(f"{name} on an unopened stage raises rather than lying", ok)


def test_stage_frame():
    """A reverse-mounted axis must be handled in exactly one place.

    The failure this guards against is not a crash. It is a field map that
    comes out mirrored along one axis and looks entirely plausible.
    """
    fwd = _framed("x", False)
    check("an as-mounted axis is the identity",
          fwd.travel_mm == (0.0, 300.0) and fwd._rig_to_dev(75.0) == 75.0,
          f"travel {fwd.travel_mm}")
    check("an as-mounted axis says so",
          fwd.frame_note() == "as mounted", fwd.frame_note())

    rev = _framed("z", True)
    check("a reversed axis puts rig zero at the far end of travel",
          rev.origin_mm == 300.0, f"origin {rev.origin_mm}")
    check("a reversed axis still spans the same rig travel",
          rev.travel_mm == (0.0, 300.0), f"travel {rev.travel_mm}")
    check("a reversed axis has no negative zero in its travel",
          not any(str(v).startswith("-0") for v in rev.travel_mm),
          f"travel {rev.travel_mm}")

    # The cube the user actually cares about: rig zero is the BOTTOM, and the
    # limit switch the stage homes to is the top.
    check("rig zero maps to the stage's far end",
          rev._rig_to_dev(0.0) == 300.0)
    check("rig maximum maps to the stage's home",
          rev._rig_to_dev(300.0) == 0.0)
    check("the midpoint is its own mirror image",
          rev._rig_to_dev(150.0) == 150.0)
    for rig in (0.0, 12.5, 150.0, 299.0):
        check(f"rig {rig:g} mm round trips through the device frame",
              abs(rev._dev_to_rig(rev._rig_to_dev(rig)) - rig) < 1e-9)

    check("a reversed axis reverses relative moves too",
          rev._sign == -1.0 and fwd._sign == 1.0,
          "a rig +5 mm must drive the device -5 mm, or jogging goes backwards")

    # An explicit origin is how you put rig zero on a fixture datum rather
    # than on a hard limit.
    off = _framed("z", True, origin=250.0)
    check("an explicit origin overrides the automatic one",
          off.origin_mm == 250.0 and off._rig_to_dev(0.0) == 250.0)
    check("an explicit origin shifts the rig travel with it",
          off.travel_mm == (-50.0, 250.0), f"travel {off.travel_mm}")


def test_stage_frame_persistence(workdir):
    path = os.path.join(workdir, "stages_frame.json")
    ostage.save_axis_map(
        {"x": "45502844", "z": "45502854", "y": "45538374"}, path,
        frames={"x": {"invert": False},
                "z": {"invert": True},
                "y": {"invert": False, "origin_mm": 12.0}})
    frames = ostage.load_axis_frames(path)
    check("the axis frame round trips", frames["z"]["invert"] is True,
          str(frames))
    check("an explicit origin round trips", frames["y"]["origin_mm"] == 12.0,
          str(frames))
    check("an unset origin round trips as None",
          frames["z"]["origin_mm"] is None, str(frames))
    check("the axis map survives alongside the frame",
          ostage.load_axis_map(path)["z"] == "45502854")

    # Writing the map without frames must not wipe the mounting -- the GUI's
    # "Save axis map" button does exactly that.
    ostage.save_axis_map({"x": "45502844", "z": "45502854"}, path)
    check("saving the map alone preserves the mounting",
          ostage.load_axis_frames(path)["z"]["invert"] is True)

    # Soft limits live in the same block, and the same button rewrites it --
    # so the same argument applies with more at stake. A save that dropped
    # limit_mm would remove the only thing keeping the head out of the
    # fixture, during routine housekeeping, silently.
    ostage.save_axis_map({"z": "45502854"}, path,
                         frames={"z": {"invert": True,
                                       "limit_mm": (20.0, 250.0)}})
    check("a soft limit round trips",
          ostage.load_axis_frames(path)["z"]["limit_mm"] == (20.0, 250.0),
          str(ostage.load_axis_frames(path)))
    ostage.save_axis_map({"z": "45502854"}, path,
                         frames={"z": {"invert": True}})
    check("re-saving the frame without limits preserves the soft limit",
          ostage.load_axis_frames(path)["z"]["limit_mm"] == (20.0, 250.0))
    ostage.save_axis_map({"z": "45502854"}, path)
    check("saving the map alone preserves the soft limit",
          ostage.load_axis_frames(path)["z"]["limit_mm"] == (20.0, 250.0))
    check("an axis with no soft limit reports None",
          ostage.load_axis_frames(path).get("x", {}).get("limit_mm") is None)

    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["frame"]["z"]["limit_mm"] = [20.0]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    try:
        ostage.load_axis_frames(path)
        ok = False
    except ostage.StageError:
        ok = True
    check("a malformed soft limit is refused rather than half-applied", ok,
          "a limit_mm that silently became 'no limit' is the worst outcome")
    doc["frame"]["z"]["limit_mm"] = [20.0, 250.0]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)

    # A bare boolean is the shorthand a human is most likely to hand-edit in.
    with open(path) as fh:
        doc = json.load(fh)
    doc["frame"] = {"z": True}
    with open(path, "w") as fh:
        json.dump(doc, fh)
    check("a bare true is accepted as shorthand for invert",
          ostage.load_axis_frames(path)["z"]["invert"] is True)

    check("a missing frame block is empty, not an error",
          ostage.load_axis_frames(os.path.join(workdir, "nope.json")) == {})


def test_stage_safety():
    """The emergency-stop conventions, without a stage attached.

    All of this is the logic that decides whether a move is allowed to be
    commanded, which is the half that can be checked on a bench with nothing
    plugged in -- and the half where a mistake is a collision rather than an
    exception.
    """
    print("\nmotion interlock and position trust")

    il = ostage.MotionInterlock()
    check("a fresh interlock is clear", il.tripped is None)
    il.trip("z hit the hard limit")
    il.trip("operator pressed the emergency stop")
    check("the FIRST reason to trip is the one kept",
          il.tripped == "z hit the hard limit",
          "the cascade would otherwise overwrite the cause with the symptom")
    try:
        il.require_clear("move x to 10 mm")
        ok = False
    except ostage.MotionInterlocked:
        ok = True
    check("a latched interlock refuses a move", ok)
    check("MotionInterlocked is a StageError",
          issubclass(ostage.MotionInterlocked, ostage.StageError),
          "callers that catch StageError must not miss a stop")
    check("resetting reports what it cleared",
          il.reset() == "z hit the hard limit" and il.tripped is None)

    # One latch for the machine, not one per axis: the axis that faults is not
    # necessarily the one about to hit something.
    ss = ostage.StageSet({"x": ostage.Stage("1", name="x"),
                          "z": ostage.Stage("2", name="z")})
    check("every axis in a set shares one interlock",
          ss["x"].interlock is ss["z"].interlock is ss.interlock)
    ss.emergency_stop("test")
    check("an emergency stop with nothing open still latches",
          ss.interlock.tripped == "test",
          "or a stop pressed while disconnected is forgotten")
    check("and it does not raise on the way", True)

    # Position trust. The whole point is that it is NOT the homed bit.
    st = _framed("x", False)
    check("an unopened, unhomed axis is not trusted", not st.position_trusted)
    st._trusted = True
    check("trust still requires the stage to be open",
          not st.position_trusted,
          "the controller's homed bit cannot even be read on a closed stage")
    check("and neither property raises on a closed stage",
          st.distrust_reason == "the stage is not open",
          "these are read from the stop path, which must not throw")
    st.distrust("stopped immediately")
    st.is_open = True                       # nothing to talk to, just the flag
    check("distrust says why", st.distrust_reason == "stopped immediately")
    st.is_open = False

    # Soft limits: a restriction, never a permission.
    check("no soft limit means the whole travel",
          ostage.Stage._resolve_limit((0.0, 300.0), None) == (0.0, 300.0))
    check("a soft limit narrows the travel",
          ostage.Stage._resolve_limit((0.0, 300.0), (20.0, 280.0))
          == (20.0, 280.0))
    check("a soft limit given the wrong way round is sorted, not refused",
          ostage.Stage._resolve_limit((0.0, 300.0), (280.0, 20.0))
          == (20.0, 280.0))
    check("a soft limit WIDER than the travel is clamped, not obeyed",
          ostage.Stage._resolve_limit((0.0, 300.0), (-50.0, 900.0))
          == (0.0, 300.0),
          "a config that asks for more travel than exists is a typo")
    try:
        ostage.Stage._resolve_limit((0.0, 300.0), (400.0, 500.0))
        ok = False
    except ostage.StageError:
        ok = True
    check("a soft limit entirely outside the travel is refused", ok)

    lim = _framed("z", True)
    lim._limit_cfg = (30.0, 200.0)
    lim._resolve_frame()
    check("the envelope survives the reversed-axis frame",
          lim.limit_mm == (30.0, 200.0) and lim.travel_mm == (0.0, 300.0),
          f"limit {lim.limit_mm}, travel {lim.travel_mm}")

    # "Declared as the whole travel" and "never configured" allow exactly the
    # same movement and are not the same thing: one is a measurement, the
    # other is a gap. Only the gap should still be warning about itself.
    bare = _framed("x", False)
    check("an axis with no limit_mm says so", not bare.limit_declared)
    told = _framed("x", False)
    told._limit_cfg = (0.0, 300.0)
    told._resolve_frame()
    check("an axis declared as its whole travel is still declared",
          told.limit_declared and told.limit_mm == told.travel_mm,
          f"limit {told.limit_mm}, travel {told.travel_mm}")


def test_stage_home_order(workdir):
    """Which axis retracts first is a declared fact, not an emergent one."""
    print("\nhoming order")
    path = os.path.join(workdir, "order.json")
    ostage.save_axis_map({"x": "1", "y": "2", "z": "3"}, path)
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["home_order"] = ["z", "y"]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)

    check("the home order round trips",
          ostage.load_home_order(path) == ("z", "y"))
    check("an axis that is not on the machine is ignored, not an error",
          ostage.load_home_order(path, axes=("x", "z")) == ("z",),
          "unplugging an axis must not invalidate the safe order")
    check("a missing file means no declared order",
          ostage.load_home_order(os.path.join(workdir, "nope.json")) == ())

    ss = ostage.StageSet({n: ostage.Stage(n, name=n) for n in ("x", "y", "z")},
                         home_order=("z", "y"))
    check("undeclared axes follow the declared ones rather than vanishing",
          ss.home_sequence() == ["z", "y", "x"], str(ss.home_sequence()))
    check("with nothing declared the order is the map order",
          ostage.StageSet({n: ostage.Stage(n, name=n)
                           for n in ("x", "y")}).home_sequence() == ["x", "y"])


def test_stage_axis_map(workdir):
    path = os.path.join(workdir, "stages.json")
    ostage.save_axis_map({"x": "45502844", "z": "45538374"}, path)
    back = ostage.load_axis_map(path)
    check("the axis map round trips",
          back == {"x": "45502844", "z": "45538374"}, str(back))
    check("a missing axis map is empty, not an error",
          ostage.load_axis_map(os.path.join(workdir, "nope.json")) == {})

    # A map naming a stage that is not present must fail loudly: the axis
    # names ARE the coordinate frame, and quietly dropping one produces a map
    # whose axes mean something other than what the file says.
    try:
        ostage.StageSet.from_config(path, serials=["45502844"])
        ok = False
    except ostage.StageError:
        ok = True
    check("an axis map naming an absent stage is refused", ok)

    ss = ostage.StageSet.from_config(path, serials=["45502844", "45538374"])
    check("a valid axis map builds the named axes",
          sorted(ss.names) == ["x", "z"], str(ss.names))
    try:
        ss.move_to(y=1.0)
        ok = False
    except ostage.StageError:
        ok = True
    check("moving an axis the set does not have is refused", ok)


def test_stage_motion(workdir):
    """The motion profile: the maths, the config, and what reaches a Stage."""
    print("\nstage motion profile")
    # A move only reaches the cap if it is long enough to ramp up to it. This
    # is the whole reason a bigger jog step sounds different from a small one.
    check("a short move never reaches the velocity cap",
          abs(ostage.peak_speed_mm_s(1.0, 20.0, 20.0) - 4.472) < 1e-3,
          "1 mm at 20/20 peaks at 4.5 mm/s, not 20")
    check("a long move is capped by velocity, not distance",
          ostage.peak_speed_mm_s(300.0, 8.0, 10.0) == 8.0)
    check("the shipped profile is what makes 5 mm the audible threshold",
          abs(ostage.peak_speed_mm_s(5.0, 20.0, 20.0) - 10.0) < 1e-9
          and ostage.peak_speed_mm_s(20.0, 20.0, 20.0) == 20.0,
          "5 mm -> 10 mm/s but 20 mm -> the full 20")
    check("no step size can outrun the configured cap",
          max(ostage.peak_speed_mm_s(d, ostage.DEFAULT_VEL_MM_S,
                                     ostage.DEFAULT_ACCEL_MM_S2)
              for d in (1, 5, 20, 100, 300)) <= ostage.DEFAULT_VEL_MM_S)

    # ---- the ceiling ----
    # A velocity setting is not a promise: anything that opens the device
    # outside this module leaves the controller at its shipped 20 mm/s. So the
    # cap has to hold at every door into the module, not just the front one.
    check("the default profile is under the ceiling",
          ostage.DEFAULT_VEL_MM_S <= ostage.MAX_VEL_MM_S)
    check("a velocity above the ceiling is clamped",
          ostage.clamp_velocity(50.0) == ostage.MAX_VEL_MM_S
          and ostage.clamp_velocity(2.0) == 2.0)
    check("resolving a profile clamps what it sends to the controller",
          ostage.Stage.resolve_profile(20.0, 20.0, (8.0, 10.0))
          == (ostage.MAX_VEL_MM_S, 20.0),
          "20 mm/s asked for, ceiling sent")
    check("resolving with None keeps what the controller already had",
          ostage.Stage.resolve_profile(None, None, (5.0, 7.0)) == (5.0, 7.0))
    for bad in ((0.0, 5.0), (5.0, 0.0), (-1.0, 5.0)):
        try:
            ostage.Stage.resolve_profile(bad[0], bad[1], (8.0, 10.0))
            ok = False
        except ostage.StageError:
            ok = True
        check(f"a profile of {bad} is refused", ok)
    # Every mover must re-apply the profile, or the ceiling is only true until
    # something else touches the controller.
    src = inspect.getsource(ostage.Stage)
    for mover in ("move_to", "move_by"):
        body = src.split(f"def {mover}(")[1].split("\n    def ")[0]
        check(f"{mover} re-applies the profile before it moves",
              "enforce_profile()" in body)

    # A move's deadline has to scale with the move. A fixed one fires on a
    # long slow traverse that is going perfectly -- and since a timeout now
    # stops the axis and marks its position lost, a spurious one costs a
    # re-home and, mid-raster, the rest of the map.
    check("a short move's time is set by acceleration, not the speed cap",
          abs(ostage.move_time_s(1.0, 6.0, 10.0) - 0.632) < 1e-3,
          f"{ostage.move_time_s(1.0, 6.0, 10.0):.3f} s for 1 mm at 6/10")
    check("a long move's time is set by the speed cap",
          abs(ostage.move_time_s(300.0, 6.0, 10.0) - 50.6) < 0.05,
          f"{ostage.move_time_s(300.0, 6.0, 10.0):.1f} s for the full traverse")
    check("the deadline for a full traverse leaves room but is not open-ended",
          150.0 < ostage.move_timeout_s(300.0, 6.0, 10.0) < 200.0,
          f"{ostage.move_timeout_s(300.0, 6.0, 10.0):.0f} s")
    check("a crawl gets a deadline it can actually meet",
          ostage.move_timeout_s(300.0, 0.1, 10.0) > 3000.0,
          f"{ostage.move_timeout_s(300.0, 0.1, 10.0):.0f} s at 0.1 mm/s -- "
          f"the old fixed 180 s stopped this move at 18 mm")
    check("a tiny move still gets a floor, not milliseconds",
          ostage.move_timeout_s(0.001, 6.0, 10.0) == 30.0)

    path = os.path.join(workdir, "motion.json")
    ostage.save_axis_map({"x": "45502844", "z": "45538374"}, path)
    got = ostage.load_axis_motion(path, ["x", "z"])
    check("a config with no motion block still opens quiet",
          got["x"] == (ostage.DEFAULT_VEL_MM_S, ostage.DEFAULT_ACCEL_MM_S2),
          f"{got['x']} rather than Kinesis's 20/20")
    ostage.save_axis_motion(path, velocity_mm_s=6.0, accel_mm_s2=7.0)
    ostage.save_axis_motion(path, velocity_mm_s=3.0, axis="z")
    got = ostage.load_axis_motion(path, ["x", "z"])
    check("the profile round trips and one axis can override it",
          got == {"x": (6.0, 7.0), "z": (3.0, 7.0)}, str(got))

    # A file written before the ceiling existed, or edited by hand, must not be
    # able to reintroduce the shipped 20 mm/s through the back door.
    hand_edited = os.path.join(workdir, "loud.json")
    with open(hand_edited, "w") as fh:
        json.dump({"axes": {"x": "45502844"},
                   "motion": {"velocity_mm_s": 20.0, "accel_mm_s2": 20.0}}, fh)
    loud = ostage.load_axis_motion(hand_edited, ["x"])
    check("a config asking for 20 mm/s is clamped on the way in",
          loud["x"][0] == ostage.MAX_VEL_MM_S, str(loud["x"]))
    ostage.save_axis_motion(hand_edited, velocity_mm_s=99.0)
    with open(hand_edited) as fh:
        check("and saving one is clamped on the way out",
              json.load(fh)["motion"]["velocity_mm_s"] == ostage.MAX_VEL_MM_S)
    check("saving the profile leaves the axis map alone",
          ostage.load_axis_map(path) == {"x": "45502844", "z": "45538374"})

    # It is no use in the file if it does not reach the hardware, and the one
    # link that cannot be seen from outside is Stage carrying it to open().
    ss = ostage.StageSet.from_config(path, serials=["45502844", "45538374"])
    check("each stage is built with its own profile",
          (ss["x"]._vel_cfg, ss["x"]._accel_cfg) == (6.0, 7.0)
          and (ss["z"]._vel_cfg, ss["z"]._accel_cfg) == (3.0, 7.0))
    check("a stage can still be opened without changing how it moves",
          ostage.Stage("45502844", vel_mm_s=None, accel_mm_s2=None)._vel_cfg
          is None)
