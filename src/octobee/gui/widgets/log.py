"""The log pane: everything the window has done this session."""

import time

from PyQt6 import QtGui, QtWidgets


class LogPane(QtWidgets.QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(4000)
        self.setFont(QtGui.QFont("Consolas", 9))

    def log(self, msg):
        self.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {msg}")
