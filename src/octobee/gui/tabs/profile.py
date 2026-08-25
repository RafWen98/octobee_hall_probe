"""
octobee/gui/tabs/profile.py -- where the frame time goes, when the window
feels slow.

"The app goes sluggish once data arrives" has several quite different causes
and they need different fixes, so this tab measures rather than guesses: every
processing stage separately, the OpenGL paint, and the Qt event loop's own
lateness. The last one matters most -- if every row here is small and the loop
is still late, the cost is something not in this list.
"""

from PyQt6 import QtGui, QtWidgets

from octobee import profile as oprof


class ProfileTab(QtWidgets.QWidget):
    """The profiling report, refreshed on a slow timer."""

    def __init__(self, session, gl_info=None, parent=None):
        """
        `gl_info` is a callable returning the live OpenGL context's info dict,
        or None when there is no 3D view on screen to ask.

        It is injected rather than reached for. Whether the probe head is being
        drawn on a GPU or on the CPU is the single most useful thing this tab
        reports, and it belongs to the Live tab's widget -- but this tab has no
        business knowing that a Live tab exists.
        """
        super().__init__(parent)
        self.session = session
        self._gl_info = gl_info

        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()

        self.chk_prof = QtWidgets.QCheckBox("measure where the time goes")
        self.chk_prof.setChecked(self.session.prof.enabled)
        self.chk_prof.setToolTip(
            "Times every stage separately, including the OpenGL paint, and "
            "watches the Qt event loop for stalls. Costs almost nothing, so it "
            "is fine to leave on.")
        self.chk_prof.toggled.connect(self.on_toggle)
        top.addWidget(self.chk_prof)

        b_reset = QtWidgets.QPushButton("Reset")
        b_reset.clicked.connect(self.on_reset)
        top.addWidget(b_reset)

        b_copy = QtWidgets.QPushButton("Copy to clipboard")
        b_copy.clicked.connect(
            lambda: QtWidgets.QApplication.clipboard().setText(
                self.profile_text.toPlainText()))
        top.addWidget(b_copy)
        top.addStretch(1)
        lay.addLayout(top)

        self.profile_text = QtWidgets.QPlainTextEdit()
        self.profile_text.setReadOnly(True)
        self.profile_text.setFont(QtGui.QFont("Consolas", 9))
        lay.addWidget(self.profile_text)

    # ---- handlers ---------------------------------------------------------

    def on_toggle(self, on):
        self.session.prof.enabled = bool(on)
        self.session.prof.reset()
        self.session.lag.reset()
        self.session.log("profiling on" if on else "profiling off")
        self.refresh()

    def on_reset(self):
        self.session.prof.reset()
        self.session.lag.reset()
        self.refresh()

    # ---- the report -------------------------------------------------------

    def refresh(self):
        prof, lag = self.session.prof, self.session.lag
        if not prof.enabled:
            self.profile_text.setPlainText(prof.text())
            return

        # Ask the live context what it is, once we have one. A software
        # renderer here is the single most likely explanation for a window
        # that seizes up the moment data starts arriving.
        if "GL renderer" not in prof.notes:
            info = self._gl_info() if self._gl_info else None
            if info:
                for k, v in info.items():
                    prof.note(k, v)
                if oprof.is_software_renderer(info):
                    prof.note("VERDICT", "no GPU acceleration -- the 3D head "
                                         "is being drawn on the CPU")
                    self.session.log(
                        f"OpenGL is running on a software renderer "
                        f"({info.get('GL renderer')}). Every repaint of the "
                        f"probe head is done on the CPU, which is almost "
                        f"certainly why the window struggles. Untick '3D' to "
                        f"confirm.")

        parts = [prof.text(), "",
                 f"event loop lag: mean {lag.mean_ms:.1f} ms, "
                 f"worst {lag.max_ms:.0f} ms",
                 f"  -> {lag.verdict()}"]

        source = self.session.source
        if source is not None:
            parts += ["", "stream:"]
            for k, v in source.stats().items():
                parts.append(f"  {k:<26} {v}")
            qs = [getattr(x, "q", None)
                  for x in getattr(source, "streamers", [])]
            for h, q in zip(getattr(source, "hosts", []), qs):
                if q is not None:
                    parts.append(f"  {h + ' reader queue':<26} {q.qsize()} "
                                 f"blocks waiting")

        parts += ["", "how to read this:",
                  "  'GL paint (probe head)' large  -> the 3D view is the cost;"
                  " untick 3D or lower the refresh rate",
                  "  'counts -> tesla' large        -> the data processing is"
                  " the cost; lower the stream rate",
                  "  reader queue growing           -> acquisition is falling"
                  " behind and recordings will have holes",
                  "  event loop lag large but every row small -> something"
                  " outside this list is blocking"]
        self.profile_text.setPlainText("\n".join(parts))
