"""The encoder scale comes from its parts; a check keeps only the wiring.

3,600 lines x 4 edges, coupled 1:1 to a motor that turns once per mm, is
14,400 counts/mm on every axis. Every fit ever saved on this rig came out below
that by a different amount each time -- dropped edges, saved as the scale.
"""

import json
import os

from PyQt6 import QtWidgets

from octobee.motion import encoder as oenc
from tests.gui.test_status import _window
from tests.helpers import check


def test_the_scale_is_the_parts():
    print("\nnominal scale")
    check("3,600 lines x 4 edges, 1:1, 1 mm lead is 14,400 counts/mm",
          oenc.nominal_counts_per_mm() == 14_400.0,
          f"{oenc.nominal_counts_per_mm()}")
    check("a part changed in stages.json changes it",
          oenc.nominal_counts_per_mm({"lines_per_rev": 1000}) == 4_000.0)
    emap = oenc.EncoderMap({"x": {"column": 0, "counts_per_mm": 14_332.09},
                            "z": {"column": 2, "counts_per_mm": -14_385.5}},
                           nominal={})
    check("a fitted number left in a file only supplies the sign",
          emap.axes["x"]["counts_per_mm"] == 14_400.0
          and emap.axes["z"]["counts_per_mm"] == -14_400.0, emap.describe())
    check("fewer counts than nominal reads as negative ppm",
          round(oenc.ppm_from_nominal(14_332.09, 14_400.0)) == -4716)


def test_nominal_survives_a_save_and_a_load(workdir):
    print("\nnominal round trip")
    path = os.path.join(workdir, "nominal_stages.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"axes": {"x": "1"}}, fh)
    oenc.EncoderMap({"y": {"column": 1, "counts_per_mm": 1.0}},
                    nominal={}).save(path, note="t")
    doc = json.load(open(path, encoding="utf-8"))
    check("the parts are written beside the wiring",
          doc["encoders"]["nominal"]["lines_per_rev"] == 3600
          and doc["axes"] == {"x": "1"}, str(doc["encoders"]))
    back = oenc.EncoderMap.load(path)
    check("and read back as the nominal scale",
          back.nominal is not None
          and back.axes["y"]["counts_per_mm"] == 14_400.0, back.describe())
    # Edited by hand to a fitted number: the parts still win.
    doc["encoders"]["axes"]["y"]["counts_per_mm"] = 14_235.7
    json.dump(doc, open(path, "w", encoding="utf-8"))
    check("a hand-edited scale cannot creep back in",
          oenc.EncoderMap.load(path).axes["y"]["counts_per_mm"] == 14_400.0)


def test_a_check_keeps_the_wiring_and_flags_lost_counts(app, workdir,
                                                        monkeypatch):
    print("\nCheck encoders, with x dropping counts")
    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warned.append(a[2])))
    win = _window(workdir, "nominalcheck")
    try:
        said = []
        real = win.session.log
        win.session.log = lambda m: (said.append(m), real(m))
        found = {
            "x": {"column": 0, "counts_per_mm": 14_332.1, "repeatable": True,
                  "span_mm": 100.0, "points": 11, "residual_um": 15.1,
                  "pass_spread_ppm": 302.0},
            "y": {"column": 1, "counts_per_mm": 14_399.5, "repeatable": True},
            "z": {"column": 2, "counts_per_mm": -14_400.4, "repeatable": True},
        }
        win.tab_machine.on_volume_calibrated(found, "")
        enc = win.session.encoders
        check("every axis is saved at the nominal scale, sign kept",
              [enc.axes[a]["counts_per_mm"] for a in "xyz"]
              == [14_400.0, 14_400.0, -14_400.0], enc.describe())
        back = oenc.EncoderMap.load(win.session.stages_path)
        check("and that is what stages.json now holds",
              back.to_dict() == enc.to_dict() and back.nominal is not None)
        note = json.load(open(win.session.stages_path,
                              encoding="utf-8"))["encoders"]["note"]
        check("the note records how far each axis was from nominal",
              "-4715 ppm" in note or "-4716 ppm" in note, note)
        check("x, 0.47% short, is called lost counts",
              any("WARNING" in m and "x came out" in m and "dropped edges" in m
                  for m in said), "\n".join(said[-6:]))
        check("y and z, within tolerance, are not",
              not any("WARNING" in m and ("y came out" in m
                                          or "z came out" in m)
                      for m in said))
        check("and the operator is told, once",
              len(warned) == 1 and "x: -471" in warned[0],
              str(warned))
    finally:
        win.close()


def test_a_file_of_fitted_scales_is_pointed_out(app, workdir):
    print("\nan old stages.json with fitted scales")
    path = os.path.join(workdir, "fitted_stages.json")
    oenc.EncoderMap({"x": {"column": 0, "counts_per_mm": 14_332.09},
                     "y": {"column": 1, "counts_per_mm": 14_392.76}}
                    ).save(path, note="old")
    win = _window(workdir, "fitted")
    try:
        said = []
        win.session.log = said.append
        win.session.stages_path = path
        win.session.encoders = oenc.EncoderMap.load(path)
        win.tab_machine.announce()
        check("it says the file holds fitted scales, and which is worst",
              any("fitted scales" in m and "x is -4716 ppm" in m
                  for m in said), "\n".join(said))
    finally:
        win.close()
