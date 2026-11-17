"""The readiness checks asked before an unattended volume map.

No window: preflight reads a session and returns a list, so it is driven here
with objects that have exactly the attributes it reads.
"""

from types import SimpleNamespace

from octobee.gui import preflight
from octobee.gui.preflight import FAIL, OK, WARN

from tests.helpers import check


class _Source:
    live = True
    failed = False
    fs_hz = 20000.0
    enc_columns = 3

    def __init__(self, **stats):
        self._stats = stats

    def stats(self):
        return dict(self._stats)


class _Stages:
    names = ("x", "y", "z")

    def __init__(self, lost=()):
        self._lost = list(lost)

    def untrusted(self):
        return list(self._lost)


class _Plan:
    def __init__(self, lines=40, hours=2.0, rows=1_000_000, way_in=""):
        self.lines = list(range(lines))
        self._hours, self._rows, self._way_in = hours, rows, way_in

    def duration_s(self):
        return self._hours * 3600.0

    def rows(self):
        return self._rows

    def entry(self, position_mm):
        if self._way_in == "blocked":
            raise ValueError("there is no clear way to the first line")
        return [], self._way_in


class _Motion:
    def __init__(self, reason=None, running=False):
        self.reason, self._running = reason, running

    @property
    def latched(self):
        return self.reason is not None

    def busy(self):
        return self._running


def _session(tmp, **over):
    s = SimpleNamespace(
        source=_Source(), last_health=[{"sensor_id": 1}],
        cal=SimpleNamespace(dead=set()), stages=_Stages(),
        encoders={"x": 1, "y": 1, "z": 1}, volume_runner=None,
        collecting=None, out_dir=tmp)
    for k, v in over.items():
        setattr(s, k, v)
    return s


def _by_name(checks):
    return {c.name: c for c in checks}


def test_a_ready_rig_is_all_green(workdir):
    """Everything in order gives no warning and no refusal."""
    print("\npreflight: a ready rig")
    got = preflight.run_checks(_session(workdir), _Motion(), plan=_Plan(),
                               head_mm={"x": 1.0, "y": 1.0, "z": 1.0})
    check("every check passes", preflight.worst(got) == OK,
          str([(c.name, c.state, c.detail) for c in got if c.state != OK]))
    check("and it says so in one line",
          preflight.summary(got) == "ready to run", preflight.summary(got))
    check("the plan's size is on the Path line",
          "40 lines" in _by_name(got)["Path"].detail)


def test_what_stops_a_run(workdir):
    """Each of these must refuse Start, and name itself."""
    print("\npreflight: refusals")
    cases = {
        "Stream": {"session": {"source": None}},
        "Stages": {"session": {"stages": None}},
        "Axes referenced": {"session": {"stages": _Stages(
            [("y", "stopped by an emergency stop")])}},
        "Emergency stop": {"motion": _Motion("operator pressed it")},
        "Machine idle": {"motion": _Motion(running=True)},
        "Path": {"plan": None},
        "Way in": {"plan": _Plan(way_in="blocked")},
        "Disk space": {"plan": _Plan(rows=10**13)},
    }
    for name, case in cases.items():
        got = _by_name(preflight.run_checks(
            _session(workdir, **case.get("session", {})),
            case.get("motion", _Motion()),
            plan=case["plan"] if "plan" in case else _Plan(),
            head_mm={"x": 1.0, "y": 1.0, "z": 1.0}))
        check(f"{name} refuses", name in got and got[name].state == FAIL,
              str(got.get(name)))
    stale = _by_name(preflight.run_checks(
        _session(workdir), _Motion(), plan=_Plan(), plan_current=False))
    check("a plan made for another pose refuses",
          stale["Path"].state == FAIL and "plan again" in stale["Path"].detail)
    check("and its way in is not judged at all", "Way in" not in stale)
    collecting = _by_name(preflight.run_checks(
        _session(workdir, collecting="tare"), _Motion(), plan=_Plan()))
    check("a tare still collecting refuses",
          collecting["Machine idle"].state == FAIL)


def test_what_only_warns(workdir):
    """A map worth taking with a caveat must not be refused."""
    print("\npreflight: warnings")
    dead = _by_name(preflight.run_checks(
        _session(workdir, cal=SimpleNamespace(dead={"S16", "S9.Bx", "S9"})),
        _Motion(), plan=_Plan(), head_mm={"x": 1, "y": 1, "z": 1}))
    check("dead sensors warn and are named once each",
          dead["Sensors"].state == WARN
          and "14 of 16" in dead["Sensors"].detail
          and "S9, S16" in dead["Sensors"].detail, dead["Sensors"].detail)
    lossy = _by_name(preflight.run_checks(
        _session(workdir, source=_Source(gaps=2, lost=5)), _Motion(),
        plan=_Plan()))
    check("stream losses warn, with the count",
          lossy["Stream losses"].state == WARN
          and "5 sample" in lossy["Stream losses"].detail)
    enc = _by_name(preflight.run_checks(
        _session(workdir, encoders={"x": 1}), _Motion(), plan=_Plan()))
    check("uncalibrated encoders warn and name the axes",
          enc["Encoders"].state == WARN and "y, z" in enc["Encoders"].detail)
    outside = _by_name(preflight.run_checks(
        _session(workdir), _Motion(),
        plan=_Plan(way_in="the head starts outside the planned volume"),
        head_mm={"x": 1, "y": 1, "z": 1}))
    check("a head parked outside the box warns", outside["Way in"].state == WARN)
    got = preflight.run_checks(
        _session(workdir, cal=SimpleNamespace(dead={"S16"})), _Motion(),
        plan=_Plan(), head_mm={"x": 1, "y": 1, "z": 1})
    check("warnings alone leave it ready",
          preflight.worst(got) == WARN
          and preflight.summary(got).startswith("ready, with warnings"),
          preflight.summary(got))
    unscanned = preflight.sensor_summary(_session(workdir, last_health=None))
    check("sensors not scanned yet are not called healthy",
          unscanned[0] == WARN)
