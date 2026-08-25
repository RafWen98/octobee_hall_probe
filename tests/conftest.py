"""
Fixtures for the OCTO-BEE suite.

Two of these existed before pytest did, as parameters the hand-written runner
passed in by hand: `workdir` and `app`. They are fixtures now, which is what
they always were.

`workdir` is the one that changed meaning. The old runner opened a single
temporary directory and handed the same one to eighteen tests in a row, so
files written by the machine tests were visible to the GUI tests and the order
they ran in was part of the contract without anyone deciding that. tmp_path
gives each test its own.
"""

import argparse
import os

import pytest
from PyQt6 import QtWidgets

from tests import helpers


def pytest_addoption(parser):
    parser.addoption(
        "--replay", default=None, metavar="NPZ",
        help="use a real saved capture instead of the synthetic probe")
    parser.addoption(
        "--live", action="store_true",
        help="connect to the real carriers and test against them "
             "(changes their clkdiv while it runs, and restores it)")


@pytest.fixture(scope="session")
def app(request):
    """One QApplication for the whole run.

    Offscreen unless --live: the suite drives real windows, and on a headless
    runner -- or on a bench machine where you would rather it did not steal
    focus for two minutes -- there is nothing to draw onto.
    """
    if not request.config.getoption("--live"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def workdir(tmp_path):
    """A directory of this test's own, as a str.

    str rather than Path because the code under test is written in os.path and
    hands these straight to open().
    """
    return str(tmp_path)


@pytest.fixture
def args(request):
    """The two run-wide options, in the shape the application expects."""
    return argparse.Namespace(replay=request.config.getoption("--replay"),
                              live=request.config.getoption("--live"))


@pytest.fixture(autouse=True)
def _collect_checks():
    """Turn a test's failed check()s into one pytest failure, at the end.

    check() records rather than raises, which is worth keeping: these tests
    assert twenty or thirty things about one run of a calibration, and aborting
    at the first means fixing them one per CI round instead of seeing the shape
    of what broke.
    """
    helpers.begin()
    yield
    failed = helpers.failures()
    if failed:
        pytest.fail("failed checks:\n  - " + "\n  - ".join(failed),
                    pytrace=False)


def pytest_terminal_summary(terminalreporter):
    """Report the assertion count, not just the test count.

    458 checks over 41 tests: the old runner printed the first number and it is
    the more meaningful one for an instrument, so it survives the move.
    """
    total, skipped = helpers.totals()
    terminalreporter.write_line(f"{total} checks made")
    if skipped:
        terminalreporter.write_line(
            f"{len(skipped)} skipped -- this run checked less than a full one:")
        for name in skipped:
            terminalreporter.write_line(f"  - {name}")
