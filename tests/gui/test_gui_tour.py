"""What a walk through the whole window turned up, kept from coming back.

Each of these was found by driving the real window in demo mode and pressing
everything; none of them raised in the existing tests.
"""

import numpy as np

from tests.gui.test_status import _window
from tests.helpers import check, pump


def test_a_demo_session_reconnects_to_the_demo(app, workdir):
    """Disconnect then Connect in a demo session used to dial the carriers."""
    print("\nreconnecting a demo session")
    win = _window(workdir, "redemo")
    try:
        pump(win, app, 0.3)
        check("the demo is running", win.session.source is not None)
        win.on_disconnect()
        check("disconnect drops it", win.session.source is None)
        win.on_connect()
        check("connect brings the demo straight back, not the hardware",
              win.session.source is not None and not win._connecting
              and type(win.session.source).__name__ == "DemoSource",
              type(win.session.source).__name__)
    finally:
        win.close()


def test_a_stream_without_counts_records_no_position_columns(app, workdir):
    """The demo carries no encoder columns: no empty position columns either."""
    print("\nrecording a stream with no encoder counts")
    win = _window(workdir, "noenc")
    try:
        said = []
        real = win.session.log
        win.session.log = lambda m: (said.append(m), real(m))
        from octobee.motion import encoder as oenc
        win.session.encoders = oenc.EncoderMap(
            {a: {"column": i, "counts_per_mm": 14390.0}
             for i, a in enumerate("xyz")})
        rec = win._new_csv_recorder(workdir + "/noenc.csv", {})
        check("the recorder gets no encoder columns", rec.enc_columns == 0
              and not rec.enc_axes, f"{rec.enc_columns} {rec.enc_axes}")
        rec.close()
    finally:
        win.close()


def test_clear_zero_during_a_zero_wins(app, workdir):
    print("\nClear zero while a zero is collecting")
    win = _window(workdir, "clearzero")
    try:
        pump(win, app, 0.5)
        tab = win.tab_calib
        tab.start_collect("tare", 0.5)
        check("a zero is collecting", win.session.collecting is not None)
        tab.on_clear_tare()
        check("Clear zero abandons it", win.session.collecting is None)
        zero_before = np.array(win.session.cal.zero_mt, copy=True)
        pump(win, app, 1.0)
        check("and it does not land afterwards",
              np.array_equal(zero_before, win.session.cal.zero_mt))
    finally:
        win.close()


def test_the_zero_report_names_a_working_sensor(app, workdir):
    """A dead sensor (the demo's S16, railed) headlined every zero."""
    print("\nthe zero report")
    win = _window(workdir, "zeroreport")
    try:
        said = []
        real = win.session.log
        win.session.log = lambda m: (said.append(m), real(m))
        for _ in range(12):
            pump(win, app, 1.0)
            if "S16" in win.session.cal.dead:
                break
        check("S16 is excluded in the demo", "S16" in win.session.cal.dead)
        win.tab_calib.start_collect("tare", 0.5)
        pump(win, app, 1.5)
        line = next((m for m in said if m.startswith("zeroed on")), "")
        check("the largest offset is reported over the working sensors",
              line and "(S16)" not in line and "S16 excluded" in line, line)
    finally:
        win.close()


def test_the_volume_estimate_uses_the_output_rate(app, workdir):
    print("\nthe volume estimate's row rate")
    win = _window(workdir, "estimate")
    try:
        m = win.tab_machine
        win.cmb_out.setCurrentIndex(win.cmb_out.findData(500.0))
        m.on_volume_edited()
        check("at 500 Hz it says 500 Hz", "at 500 Hz" in m.lbl_volume.text(),
              m.lbl_volume.text()[-80:])
        win.cmb_out.setCurrentIndex(win.cmb_out.findData(100.0))
        check("and follows the toolbar to 100 Hz",
              "at 100 Hz" in m.lbl_volume.text(), m.lbl_volume.text()[-80:])
    finally:
        win.close()


def test_the_encoder_line_matches_the_stream(app, workdir):
    print("\nthe encoder line with a stream that has no counts")
    win = _window(workdir, "encline")
    try:
        from octobee.motion import encoder as oenc
        win.session.encoders = oenc.EncoderMap(
            {a: {"column": i, "counts_per_mm": 14390.0}
             for i, a in enumerate("xyz")})
        win.tab_machine.refresh_encoders()
        text = win.tab_machine.lbl_encoders.text()
        check("scales saved, demo stream: it does not claim the sample clock",
              "sample clock" not in text and "no encoder counts" in text, text)
    finally:
        win.close()
