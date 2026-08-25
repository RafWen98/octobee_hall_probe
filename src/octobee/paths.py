"""
octobee/paths.py -- the one place that answers "where does that file live".

Before the package existed, every module found its files by asking where it
itself was, or by opening a bare filename and letting the working directory
decide. Both stopped being true the moment the code moved into src/: a module
at src/octobee/help.py that looks for README.md "next to itself" is looking
inside the package, not in the checkout.

This module is the answer to that, and it is deliberately the ONLY answer --
nothing else in the package computes a path from __file__.

At this stage it knows about the checkout only. Locating the working
configuration (calibration.json and friends), which is a behaviour change
rather than a move, comes next.
"""

import pathlib

__all__ = ["in_checkout", "repo_root"]

# src/octobee/paths.py -> src/octobee -> src -> the checkout
_PACKAGE = pathlib.Path(__file__).resolve().parent
_CANDIDATE = _PACKAGE.parent.parent


def repo_root():
    """
    The checkout this package was installed from, or None.

    An editable install (`pip install -e .`, which is how this is meant to be
    used) leaves the package inside the checkout, so walking up from __file__
    finds it. A regular wheel installed into site-packages does not, and this
    returns None rather than a plausible-looking wrong directory -- callers are
    expected to have a fallback, because on someone else's machine there is no
    checkout to find.

    pyproject.toml is the marker. Anything else that might sit at the top of a
    source tree could equally sit somewhere else.
    """
    if (_CANDIDATE / "pyproject.toml").is_file():
        return _CANDIDATE
    return None


def in_checkout():
    """True when running from an editable install of a source checkout."""
    return repo_root() is not None
