"""The on-box scripts' gain tables, which must not drift."""

import ast
import os


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
