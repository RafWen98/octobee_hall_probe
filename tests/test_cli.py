"""The command-line front door."""

import importlib
import os

import numpy as np

from octobee import live as olive
from octobee import report as orep
from octobee.acq import carrier as ob
from octobee.acq import idmap as oid
from octobee.calib import convert as ocal
from octobee.cli import main as cli
from tests.helpers import (
    check,
)



def test_cli_commands():
    """Every subcommand the front door advertises must actually be there.

    The dispatcher names modules as strings, so a module that moves or a main()
    that is renamed breaks a command that nothing else exercises -- and breaks
    it at the moment someone types it, which is the worst time to find out.
    """
    print("\ncommand line")
    check("the usage listing names every command",
          all(name in cli._usage() for name in cli.COMMANDS),
          str(sorted(cli.COMMANDS)))
    missing = []
    for name, (modname, _) in sorted(cli.COMMANDS.items()):
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:
            missing.append(f"{name}: {modname} will not import ({exc})")
            continue
        if not callable(getattr(mod, "main", None)):
            missing.append(f"{name}: {modname} has no main()")
    check(f"all {len(cli.COMMANDS)} subcommands resolve to a main()",
          not missing, "; ".join(missing))
    check("an unknown command is refused, not run",
          cli.main(["definitely-not-a-command"]) == 2)


def test_record_is_described_as_the_exporter_it_is():
    """`octobee record` exports a .npz to CSV; it does not inspect anything."""
    print("\nrecord's description")
    blurb = cli.COMMANDS["record"][1]
    check("the listing says export, and names the input and the output",
          "export" in blurb and ".npz" in blurb and "CSV" in blurb
          and "inspect" not in blurb, blurb)


def test_live_decimation_keeps_every_sample():
    """Blocks are not multiples of decim; the remainder must carry over."""
    print("\nlive plot decimation")
    decim, rng = 10, np.random.default_rng(2)
    sizes = [int(v) for v in rng.integers(1, 40, 200)]
    ai = np.zeros((sum(sizes), 32), np.int16)
    ai[:, 0] = np.arange(len(ai)) % 1000
    old, new, dec, at = 0, 0, ocal.Decimator(decim), 0
    for n in sizes:
        blk = ai[at:at + n]
        at += n
        p_old = olive.prepare(blk, 1.0, False, False, decim)
        old += 0 if p_old is None else p_old[0].shape[0]
        p_new = olive.prepare(blk, 1.0, False, False, decim, dec)
        new += 0 if p_new is None else p_new[0].shape[0]
    total = sum(sizes)
    check("per-block averaging loses samples (the old behaviour)",
          old < total // decim, f"{old} rows against {total // decim}")
    check("with a carried remainder every complete group becomes a row",
          new == total // decim, f"{new} rows against {total // decim}")


def test_raw_axis_says_volts():
    print("\nlive plot axis label")
    check("--raw without a range is volts, not counts",
          "counts" not in olive.axis_unit(None, True)
          and "V" in olive.axis_unit(None, True), olive.axis_unit(None, True))
    check("a range still reads in mT",
          "mT" in olive.axis_unit(20.0, True))


def _box(host, vcm_v, adc_range="+/-10V", n=500):
    span, offset = ob.ADC_RANGES[adc_range]
    vpc = span / 65536.0
    ai = np.zeros((n, 32), np.int16)
    ai[:, 3::4] = int(round((vcm_v - offset) / vpc))
    ai[:, 3::4] += (np.arange(n) % 5 - 2).astype(np.int16)[:, None]
    return {"host": host, "ai": ai, "temp_raw": np.full(8, 2200, np.uint32),
            "pwr_good": 1, "fs_hz": 200000.0, "vpc": vpc,
            "volt_offset": offset, "adc_range": adc_range,
            "sam_cnt": np.arange(n, dtype=np.uint32)}


def test_vcm_spread_uses_each_boxes_own_scale(capsys):
    """The uT figure came from whichever box the loop happened to end on."""
    print("\nreport VCM spread")
    boxes = [_box("a", 2.2), _box("b", 2.3)]
    orep.report(boxes, [20.0, 400.0], False, None)
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("VCM spread"))
    vpt = [ob.RANGE_TO_VPT[20.0], ob.RANGE_TO_VPT[400.0]]
    want = abs(2.3 / vpt[1] - 2.2 / vpt[0]) * 1e6
    stale = 0.1 / vpt[1] * 1e6               # the last box's scale on everything
    shown = float(line.split("mV = ")[1].split(" uT")[0])
    check("each chip is converted with its own box's V/T",
          abs(shown - want) < 0.02 * want and abs(shown - stale) > 0.1 * stale,
          f"{shown:.0f} uT shown, {want:.0f} expected, {stale:.0f} if stale")


def test_output_defaults_live_in_the_captures_directory(tmp_path, monkeypatch):
    print("\nreport and idmap output paths")
    monkeypatch.setenv("OCTOBEE_CAPTURES_DIR", str(tmp_path / "caps"))
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    seen = {}
    monkeypatch.setattr(orep, "load_npz", lambda path: [_box("a", 2.2)] * 2)
    monkeypatch.setattr(orep, "report",
                        lambda boxes, rng, plot, path: seen.update(path=path))
    monkeypatch.setattr("sys.argv", ["report", "--load", "x.npz", "--range", "20"])
    orep.main()
    check("report --plot-path defaults to the captures directory",
          os.path.dirname(seen["path"]) == str(tmp_path / "caps")
          and os.path.basename(seen["path"]) == "octobee_cal.png", seen["path"])
    got = oid.resolve_save_path("idmap.npz")
    check("idmap --save with a bare name goes there too",
          got == os.path.join(str(tmp_path / "caps"), "idmap.npz"), got)
    check("and a name with a directory is taken as given",
          oid.resolve_save_path(os.path.join("sub", "x.npz"))
          == os.path.join("sub", "x.npz"))
