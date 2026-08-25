"""
octobee/paths.py -- the one place that answers "where does that file live".

Why this exists
---------------
Every configuration file used to be named, not located. `Calibration.load()`
defaulted to the string "calibration.json" and handed it straight to open(),
so the file it found depended entirely on where the process happened to be
started. Run any command-line tool from anywhere but the checkout and it
silently loaded built-in defaults instead -- and `--save` then wrote a second
calibration.json wherever you were standing.

Silently is the problem. The built-in defaults put every sensor on +/-20 mT,
which is a plausible-looking calibration, so the failure produces numbers that
look entirely reasonable and are uniformly wrong. For an instrument whose only
output is measurements, that is the worst failure mode available.

The desktop launcher already knew this and worked around it with os.chdir(),
which fixed the icon and nothing else -- not a terminal, not a cron job, not a
script that happened to os.chdir() itself.

So: nothing in this package computes a path from __file__, and nothing opens a
bare filename. Everything comes through here.

Where things end up
-------------------
    configuration   $OCTOBEE_CONFIG_DIR, else <checkout>/config, else the
                    platform's per-user config directory
    captures        $OCTOBEE_CAPTURES_DIR, else <checkout>/captures, else
                    ./captures under the working directory
    assets          inside the package, always (the taskbar icon)

Running from a checkout -- which is how the bench uses this -- behaves exactly
as it did before, finding the same files. What changes is that it now finds
them from ANY working directory, and that an installed copy on a machine with
no checkout has somewhere sensible of its own.
"""

import os
import pathlib
import sys

__all__ = [
    "asset", "captures_dir", "config", "config_dir", "in_checkout",
    "repo_root",
]

CONFIG_DIR_ENV = "OCTOBEE_CONFIG_DIR"
CAPTURES_DIR_ENV = "OCTOBEE_CAPTURES_DIR"

CONFIG_SUBDIR = "config"
CAPTURES_SUBDIR = "captures"
ASSET_SUBDIR = "data"

# src/octobee/paths.py -> src/octobee -> src -> the checkout
_PACKAGE = pathlib.Path(__file__).resolve().parent
_CANDIDATE = _PACKAGE.parent.parent


def repo_root():
    """
    The checkout this package was installed from, or None.

    An editable install (`pip install -e .`, which is how this is meant to be
    used) leaves the package inside the checkout, so walking up from __file__
    finds it. A wheel installed into site-packages does not, and this returns
    None rather than a plausible-looking wrong directory -- every caller has a
    fallback, because on someone else's machine there is no checkout to find.

    pyproject.toml is the marker. Anything else that might sit at the top of a
    source tree could equally sit somewhere else.
    """
    if (_CANDIDATE / "pyproject.toml").is_file():
        return _CANDIDATE
    return None


def in_checkout():
    """True when running from an editable install of a source checkout."""
    return repo_root() is not None


def _user_config_dir():
    """The platform's per-user configuration directory for this application.

    Written out rather than taking a dependency on platformdirs: it is eight
    lines, it is only ever reached by an installed copy with no checkout, and
    a dependency that exists to save eight lines is a dependency that will
    eventually break a bench install for no reason.
    """
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or (pathlib.Path.home() / "AppData"
                                             / "Roaming")
    elif sys.platform == "darwin":
        base = pathlib.Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or (pathlib.Path.home()
                                                     / ".config")
    return pathlib.Path(base) / "octobee"


def config_dir():
    """The directory holding calibration.json and its neighbours."""
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return pathlib.Path(override)
    root = repo_root()
    if root is not None:
        return root / CONFIG_SUBDIR
    return _user_config_dir()


def config(name):
    """Absolute path to one configuration file, whether or not it exists.

    Returned as a str: every caller hands it to open() or to argparse, and the
    surrounding code is written in os.path rather than pathlib.
    """
    return str(config_dir() / name)


def captures_dir():
    """Where recordings, exports and field maps are written.

    Falls back to ./captures rather than a per-user directory when there is no
    checkout: captured data is the output of a run, and a tool that produces
    data should put it where the run happened, not somewhere central.
    """
    override = os.environ.get(CAPTURES_DIR_ENV)
    if override:
        return str(pathlib.Path(override))
    root = repo_root()
    if root is not None:
        return str(root / CAPTURES_SUBDIR)
    return CAPTURES_SUBDIR


def asset(name):
    """Absolute path to a file shipped inside the package (the icon)."""
    return str(_PACKAGE / ASSET_SUBDIR / name)
