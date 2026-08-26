"""Where configuration is looked for, and that the answer never depends on cwd.

The whole point of octobee/paths.py is that a tool run from the wrong
directory must not quietly fall back to built-in defaults and produce
plausible, uniformly wrong numbers. An override taken verbatim reintroduced
exactly that: a relative $OCTOBEE_CONFIG_DIR is relative to wherever the
process happened to start.
"""

import importlib
import os

from octobee import paths
from tests.helpers import (
    check,
)


def _with_env(**env):
    """Reload paths with these variables set, and put the environment back."""
    old = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(paths)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(paths)


def test_config_dir_is_always_absolute():
    print("\nconfig location")
    for value in ("myconfig", "./cfg", "~/octobee-config"):
        gen = _with_env(OCTOBEE_CONFIG_DIR=value)
        next(gen)
        try:
            p = paths.config("calibration.json")
            check(f"$OCTOBEE_CONFIG_DIR={value!r} resolves to an absolute path",
                  os.path.isabs(p), p)
            check(f"$OCTOBEE_CONFIG_DIR={value!r} expands ~",
                  "~" not in p, p)
        finally:
            for _ in gen:
                pass

    gen = _with_env(OCTOBEE_CAPTURES_DIR="shots")
    next(gen)
    try:
        check("$OCTOBEE_CAPTURES_DIR resolves too",
              os.path.isabs(paths.captures_dir()), paths.captures_dir())
    finally:
        for _ in gen:
            pass


def test_empty_override_is_ignored():
    """An empty variable means unset, not "the current directory"."""
    print("\nempty override")
    gen = _with_env(OCTOBEE_CONFIG_DIR="")
    next(gen)
    try:
        check("an empty $OCTOBEE_CONFIG_DIR falls through to the checkout",
              str(paths.config_dir()).endswith("config"), str(paths.config_dir()))
    finally:
        for _ in gen:
            pass


def test_checkout_is_found_and_absolute():
    print("\ncheckout layout")
    check("running from a checkout, config_dir is absolute",
          os.path.isabs(str(paths.config_dir())), str(paths.config_dir()))
    check("captures_dir is absolute too",
          os.path.isabs(paths.captures_dir()), paths.captures_dir())
    check("the shipped icon is where asset() says",
          os.path.exists(paths.asset("octobee.ico")), paths.asset("octobee.ico"))
