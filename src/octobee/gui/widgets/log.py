"""The log pane: everything the window has done this session."""

import time

from PyQt6 import QtCore, QtGui, QtWidgets

# What makes a line worth more than a glance, most serious first. Matched
# against the text every caller already writes, so nothing had to learn a new
# way of logging to be noticed: a line is an alarm because it says ALARM.
SERIOUS = ("ALARM", "EMERGENCY STOP", "STOP FAILED")
WORRYING = ("WARNING", "failed", "refused", "stream error")


def severity(msg):
    """2 for an alarm, 1 for a warning, 0 for the rest."""
    if any(k in msg for k in SERIOUS):
        return 2
    if any(k in msg for k in WORRYING):
        return 1
    return 0


class _Colours(QtGui.QSyntaxHighlighter):
    """Colour the lines that matter, without touching the text.

    A highlighter rather than appendHtml: the pane's plain text is what gets
    searched, copied and asserted on, and HTML would collapse the indentation
    the multi-line messages rely on.
    """

    def __init__(self, document):
        super().__init__(document)
        self._formats = {}
        for level, colour in ((1, "#e8c547"), (2, "#ff7b7b")):
            fmt = QtGui.QTextCharFormat()
            fmt.setForeground(QtGui.QColor(colour))
            if level == 2:
                fmt.setFontWeight(QtGui.QFont.Weight.Bold)
            self._formats[level] = fmt

    def highlightBlock(self, text):
        level = severity(text)
        if level:
            self.setFormat(0, len(text), self._formats[level])


class LogPane(QtWidgets.QPlainTextEdit):
    # (warnings, alarms) logged since the pane was last looked at. The window
    # puts it on the Log tab's label: a warning used to go to a tab that was
    # not on screen, and say nothing anywhere else.
    attention = QtCore.pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(4000)
        self.setFont(QtGui.QFont("Consolas", 9))
        self._colours = _Colours(self.document())
        self.unseen_warnings = 0
        self.unseen_alarms = 0

    def log(self, msg):
        self.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {msg}")
        level = severity(msg)
        if level and not self.isVisible():
            if level == 2:
                self.unseen_alarms += 1
            else:
                self.unseen_warnings += 1
            self.attention.emit(self.unseen_warnings, self.unseen_alarms)

    def mark_seen(self):
        """The pane is on screen: nothing in it is unseen any more."""
        if self.unseen_warnings or self.unseen_alarms:
            self.unseen_warnings = self.unseen_alarms = 0
            self.attention.emit(0, 0)

    def showEvent(self, ev):
        super().showEvent(ev)
        self.mark_seen()
