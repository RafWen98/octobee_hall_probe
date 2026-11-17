"""The on-box scripts' gain tables, which must not drift."""

import ast
import contextlib
import importlib.util
import os
import shutil
import subprocess
import sys
import types


from octobee import paths
from octobee.acq import carrier as ob
from tests.helpers import (
    check,
)



def test_gain_tables():
    """The on-box scripts duplicate octobee's gain tables. Catch any drift."""
    print("\ngain table consistency")
    # The on-box scripts are in the checkout, not next to this test.
    here = paths.repo_root()
    want_range = {g: r for g, (r, _) in ob.GAIN_TO_RANGE.items()}
    want_vpt = {g: v for g, (_, v) in ob.GAIN_TO_RANGE.items()}
    for name, expect in (("onbox/gain_config.py",
                          {"GAIN_RANGE_MT": want_range, "GAIN_VPT": want_vpt}),
                         ("onbox/sensor_audit.py",
                          {"GAIN_RANGE_MT": want_range})):
        src = open(os.path.join(here, name), encoding="utf-8").read()
        ns = {}
        for line in src.splitlines():
            for key in expect:
                if line.startswith(f"{key} = "):
                    ns[key] = ast.literal_eval(line.split("=", 1)[1].split("#")[0])
        for key, want in expect.items():
            got = ns.get(key)
            check(f"{name}: {key} matches octobee.GAIN_TO_RANGE",
                  got is not None and {int(k): float(v) for k, v in got.items()}
                  == {int(k): float(v) for k, v in want.items()},
                  f"{got} vs {want}")


def _run_rate_script(tmp_path, args, clock_reply):
    """Run set_sample_rate.sh against stand-ins for the vendor tools.

    Returns (exit status, stderr, whether set.site was ever called). The script
    is copied with LF endings first: a checkout with CRLF working-copy endings
    is not something the carrier's sh will run.
    """
    here = paths.repo_root()
    text = open(os.path.join(here, "onbox", "set_sample_rate.sh"),
                encoding="utf-8", newline="").read().replace("\r\n", "\n")
    script = tmp_path / "set_sample_rate.sh"
    script.write_bytes(text.encode())
    get = tmp_path / "get.site"
    get.write_bytes(("#!/bin/sh\necho '" + clock_reply + "'\n").encode())
    called = tmp_path / "set.called"
    setter = tmp_path / "set.site"
    setter.write_bytes(("#!/bin/sh\ntouch '" + called.as_posix() + "'\n").encode())
    env = dict(os.environ, GET_SITE=get.as_posix(), SET_SITE=setter.as_posix())
    r = subprocess.run(["sh", script.as_posix(), *args], capture_output=True,
                       text=True, env=env, timeout=30, check=False,
                       stdin=subprocess.DEVNULL)
    return r.returncode, r.stderr, called.exists()


def test_sample_rate_script_refuses_to_guess(tmp_path):
    """No clock reading, or a rate of 0, must not set the maximum rate."""
    print("\nset_sample_rate.sh guards")
    sh = shutil.which("sh")
    if sh is None:
        print("  (no sh on this machine; the guards are not exercised)")
        return
    rc, err, called = _run_rate_script(tmp_path, ["50000"], "")
    check("an empty SIG:CLK_MB:FREQ is an error and sets nothing",
          rc != 0 and not called and "SIG:CLK_MB:FREQ" in err,
          f"exit {rc}, set.site called {called}: {err.strip()[:100]}")
    rc, err, called = _run_rate_script(tmp_path, ["50000"], "ERROR found")
    check("so is a reply that is not a number",
          rc != 0 and not called, f"exit {rc}, set.site called {called}")
    rc, err, called = _run_rate_script(tmp_path, ["0"], "SIG:CLK_MB:FREQ 1e+08")
    check("a requested rate of 0 is refused",
          rc != 0 and not called, f"exit {rc}, set.site called {called}")
    rc, err, called = _run_rate_script(tmp_path, ["50000"],
                                       "SIG:CLK_MB:FREQ 100000000")
    check("a good clock and rate still sets the divider",
          called, f"exit {rc}: {err.strip()[:100]}")


def _load_onbox(monkeypatch, name):
    """Import an on-box script with the vendor SPI library stubbed out."""
    monkeypatch.setitem(sys.modules, "ThreeDHALLInterface",
                        types.ModuleType("ThreeDHALLInterface"))
    spec = importlib.util.spec_from_file_location(
        name + "_under_test",
        os.path.join(paths.repo_root(), "onbox", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_gain_config(monkeypatch):
    return _load_onbox(monkeypatch, "gain_config")


class _Chip:
    """A chip that takes whatever gain it is given."""

    def __init__(self):
        self.gain = 1500

    def get_gain(self):
        return self.gain

    def set_amplifier_gain(self, data):
        self.gain = data


def test_gain_set_fails_when_a_chip_is_skipped(monkeypatch):
    print("\ngain_config set exit status")
    gc = _load_gain_config(monkeypatch)
    monkeypatch.setattr(gc.time, "sleep", lambda s: None)
    every = {pos: _Chip() for pos in range(1, gc.NSENS + 1)}
    check("all eight chips answering is success",
          gc.cmd_set(every, 3000, skip_eeprom=True) is True)
    missing = dict(every)
    del missing[5]
    check("an absent chip makes it fail",
          gc.cmd_set(missing, 3000, skip_eeprom=True) is False)
    none = dict.fromkeys(range(1, gc.NSENS + 1))
    check("so does no chip answering at all",
          gc.cmd_set(none, 3000, skip_eeprom=True) is False)


def test_id_sweep_restores_the_sensor_on_any_exit(monkeypatch):
    """Ctrl-C during the muted dwell must not leave the axes switched off."""
    print("\nsensor_audit id_sweep restore")
    sa = _load_onbox(monkeypatch, "sensor_audit")
    states = []

    class _Dev:
        def change_channel_state(self, state, channel):
            states.append(state[0])

    ticks = {"n": 0}

    def sleep(seconds):
        ticks["n"] += 1
        if ticks["n"] == 2:             # 1: the 3 s lead-in, 2: first dwell
            raise KeyboardInterrupt

    monkeypatch.setattr(sa.time, "sleep", sleep)
    with contextlib.suppress(KeyboardInterrupt):
        sa.id_sweep({1: _Dev()}, 0.1)
    check("interrupted while muted, the last thing written is ON",
          states == ["off", "on"], str(states))
