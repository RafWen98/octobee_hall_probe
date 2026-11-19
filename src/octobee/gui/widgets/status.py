"""
octobee/gui/widgets/status.py -- what the rig is doing, where it cannot be missed.

Three small widgets, all of them answers to the same complaint: the thing that
mattered was on a tab nobody was looking at.

    RunStrip    one line across the top of the window, on every tab: is it
                connected, are the sensors alive, is a file open, how far
                through the map is it, has the stream lost anything -- and,
                when something has gone wrong, a banner that stays until it
                is dismissed
    CheckList   the readiness checks from octobee/gui/preflight.py, one
                coloured line each
    Chip        the coloured pill the strip is made of

None of them decides anything. They are told what to show.
"""

from PyQt6 import QtCore, QtWidgets

from octobee.gui import preflight

IDLE = "idle"

# One place for the four colours, so "amber" means the same thing in the strip,
# in the checklist and on the Live tab's sensor buttons.
COLOURS = {
    preflight.OK: "#3fb37f",
    preflight.WARN: "#e8c547",
    preflight.FAIL: "#e5383b",
    IDLE: "#5b6372",
}


def format_duration(seconds):
    """'2 h 10 min', '14 min', '40 s' -- as a person would say it."""
    s = max(0, round(float(seconds)))
    if s >= 3600:
        return f"{s // 3600} h {s % 3600 // 60:02d} min"
    if s >= 60:
        return f"{s // 60} min"
    return f"{s} s"


def format_clock(seconds):
    """Elapsed time as h:mm:ss."""
    s = max(0, int(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


class Chip(QtWidgets.QLabel):
    """A pill of text with a coloured edge. Clickable where that means something."""

    clicked = QtCore.pyqtSignal()

    def __init__(self, text="", state=IDLE, parent=None):
        super().__init__(text, parent)
        self.state = None
        self.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.set(text, state)

    # A chip is one short line. Anything longer is cut, with the whole of it
    # in the tooltip: a runaway text here (a dict of data arrays, once) widens
    # the window's layout past what Qt and the GL views can draw, and the
    # session crashes in native code with nothing pointing back to the label.
    MAX_CHARS = 160

    def set(self, text, state=IDLE, tip=None):
        text = str(text)
        first = text.split("\n", 1)[0]
        if len(first) > self.MAX_CHARS or first != text:
            if tip is None:
                tip = text[:2000]
            text = first[:self.MAX_CHARS - 1] + "…" if len(first) > self.MAX_CHARS \
                else first + " …"
        if text != self.text():
            self.setText(text)
        if tip is not None:
            self.setToolTip(tip)
        if state == self.state:
            return
        self.state = state
        colour = COLOURS.get(state, COLOURS[IDLE])
        solid = state == preflight.FAIL
        self.setStyleSheet(
            f"QLabel {{ border:1px solid {colour}; border-left:5px solid "
            f"{colour}; border-radius:3px; padding:2px 8px; font-size:12px; "
            f"background:{'#3a1418' if solid else '#171a22'}; "
            f"color:{'#ffd0d0' if solid else '#d5d9e2'}; }}")

    def mousePressEvent(self, ev):
        self.clicked.emit()
        super().mousePressEvent(ev)


class RunStrip(QtWidgets.QWidget):
    """The line under the toolbar that says what the rig is doing.

    Why it is not the status bar: the status bar is at the bottom, in small
    grey text, and is where one-off messages go. This is state -- the same
    five questions, always in the same place, coloured when the answer is not
    the one you want.
    """

    sensors_clicked = QtCore.pyqtSignal()
    log_clicked = QtCore.pyqtSignal()
    job_clicked = QtCore.pyqtSignal()
    reset_clicked = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 2)
        outer.setSpacing(4)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        self.chip_link = Chip("disconnected")
        self.chip_sensors = Chip("sensors —")
        self.chip_sensors.setToolTip("Click for the Sensors tab.")
        self.chip_sensors.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.chip_sensors.clicked.connect(self.sensors_clicked)
        self.chip_stream = Chip("stream —")
        self.chip_stream.setToolTip(
            "Samples the stream has lost since it was connected. Anything "
            "other than zero is a hole in whatever was being recorded.")
        self.chip_rec = Chip("not recording")
        self.chip_job = Chip("idle")
        self.chip_job.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.chip_job.clicked.connect(self.job_clicked)
        self.bar_job = QtWidgets.QProgressBar()
        self.bar_job.setRange(0, 1000)
        self.bar_job.setTextVisible(False)
        self.bar_job.setFixedHeight(8)
        self.bar_job.setMinimumWidth(140)
        self.bar_job.setVisible(False)
        for w in (self.chip_link, self.chip_sensors, self.chip_stream,
                  self.chip_rec, self.chip_job):
            row.addWidget(w)
        row.addWidget(self.bar_job, 1)
        row.addStretch(1)
        outer.addLayout(row)

        # The banner. Two things can be in it at once -- a latched emergency
        # stop, which only Reset removes, and an alarm, which a person
        # dismisses once they have read it.
        self.banner = QtWidgets.QFrame()
        self.banner.setStyleSheet(
            "QFrame { background:#5a1218; border:1px solid #c1121f; "
            "border-radius:3px; } QLabel { background:transparent; "
            "color:#ffe3e3; font-size:13px; border:0; }"
            "QPushButton { background:#7a1f1f; border:1px solid #c1121f; "
            "color:#fff; padding:3px 10px; }")
        b = QtWidgets.QHBoxLayout(self.banner)
        b.setContentsMargins(8, 4, 6, 4)
        self.lbl_banner = QtWidgets.QLabel("")
        self.lbl_banner.setWordWrap(True)
        self.lbl_banner.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        b.addWidget(self.lbl_banner, 1)
        # Beside the sentence that says to press it. The toolbar has one too;
        # this is the one that is where the reader's eye already is.
        self.btn_banner_reset = QtWidgets.QPushButton("Reset…")
        self.btn_banner_reset.setStyleSheet(
            "QPushButton { background:#e8c547; color:#12141a; "
            "font-weight:bold; border:1px solid #9a7d12; padding:3px 12px; }")
        self.btn_banner_reset.setToolTip(
            "Clear the emergency stop and allow motion again. It asks first.")
        self.btn_banner_reset.clicked.connect(self.reset_clicked)
        b.addWidget(self.btn_banner_reset)
        self.btn_banner_log = QtWidgets.QPushButton("Show log")
        self.btn_banner_log.clicked.connect(self.log_clicked)
        b.addWidget(self.btn_banner_log)
        self.btn_banner_dismiss = QtWidgets.QPushButton("Dismiss")
        self.btn_banner_dismiss.setToolTip(
            "Hide the alarm. It stays in the Log.")
        self.btn_banner_dismiss.clicked.connect(self.clear_alarm)
        b.addWidget(self.btn_banner_dismiss)
        outer.addWidget(self.banner)
        self._alarm = ""
        self._notice = ""
        self._offer_reset = False
        self._refresh_banner()

    # ---- the chips --------------------------------------------------------

    def set_connection(self, text, state):
        self.chip_link.set(text, state)

    def set_sensors(self, state, text, tip=None):
        self.chip_sensors.set(f"sensors: {text}", state, tip)

    def set_stream(self, state, text):
        self.chip_stream.set(text, state)

    def set_recording(self, text):
        """`text` is what is being written, or "" when nothing is."""
        if text:
            self.chip_rec.set(f"● REC  {text}", preflight.FAIL)
        else:
            self.chip_rec.set("not recording", IDLE)

    def set_job(self, text, fraction=None, state=preflight.WARN):
        """The long-running job, or None/"" when the machine is idle."""
        if not text:
            self.chip_job.set("idle", IDLE)
            self.bar_job.setVisible(False)
            return
        self.chip_job.set(text, state)
        self.bar_job.setVisible(fraction is not None)
        if fraction is not None:
            self.bar_job.setValue(int(1000 * min(max(fraction, 0.0), 1.0)))

    # ---- the banner -------------------------------------------------------

    @property
    def alarm(self):
        return self._alarm

    def show_alarm(self, text):
        """Raise the banner. Stays until dismissed or replaced."""
        self._alarm = str(text)
        self._refresh_banner()

    def clear_alarm(self):
        self._alarm = ""
        self._refresh_banner()

    def set_notice(self, text, reset=False):
        """A line that is not the operator's to dismiss, or "" to remove it.

        `reset` puts the emergency-stop Reset button beside it.
        """
        self._notice = str(text or "")
        self._offer_reset = bool(reset) and bool(self._notice)
        self._refresh_banner()

    def _refresh_banner(self):
        lines = [t for t in (self._notice, self._alarm) if t]
        self.lbl_banner.setText("\n".join(lines))
        self.btn_banner_dismiss.setVisible(bool(self._alarm))
        self.btn_banner_reset.setVisible(self._offer_reset)
        self.banner.setVisible(bool(lines))


class CheckList(QtWidgets.QWidget):
    """The readiness checks, one line each: a coloured dot, a name, what is so."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._grid = QtWidgets.QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(8)
        self._grid.setVerticalSpacing(2)
        self._grid.setColumnStretch(2, 1)
        self._rows = []
        self.lbl_summary = QtWidgets.QLabel("")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setStyleSheet("font-weight:bold; padding-top:4px;")
        self.checks = []

    def set_checks(self, checks):
        self.checks = list(checks)
        while len(self._rows) < len(self.checks):
            r = len(self._rows)
            dot = QtWidgets.QLabel("●")
            name = QtWidgets.QLabel("")
            detail = QtWidgets.QLabel("")
            detail.setWordWrap(True)
            detail.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            detail.setStyleSheet("color:#9aa3b2; font-size:11px;")
            for c, w in enumerate((dot, name, detail)):
                self._grid.addWidget(
                    w, r, c, QtCore.Qt.AlignmentFlag.AlignTop)
            self._rows.append((dot, name, detail))
        self._grid.addWidget(self.lbl_summary, len(self._rows), 0, 1, 3)
        for i, (dot, name, detail) in enumerate(self._rows):
            shown = i < len(self.checks)
            for w in (dot, name, detail):
                w.setVisible(shown)
            if not shown:
                continue
            chk = self.checks[i]
            dot.setStyleSheet(f"color:{COLOURS[chk.state]};")
            name.setText(chk.name)
            detail.setText(chk.detail)
        state = preflight.worst(self.checks)
        self.lbl_summary.setText(preflight.summary(self.checks))
        self.lbl_summary.setStyleSheet(
            f"font-weight:bold; padding-top:4px; color:{COLOURS[state]};")
