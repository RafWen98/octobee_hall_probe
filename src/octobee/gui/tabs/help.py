"""
octobee/gui/tabs/help.py -- searchable documentation, indexed from the README.

The index itself is octobee/help.py, which has no Qt in it and is tested on its
own. This is only the window onto it: a search box, a list of headings and the
text underneath.
"""

from PyQt6 import QtCore, QtGui, QtWidgets

from octobee import help as ohelp


class HelpTab(QtWidgets.QWidget):
    """Search across the README's sections and the topics about this window."""

    def __init__(self, session, parent=None):
        """
        Read from disk once, here, rather than at each search: it is 75 kB and
        the file does not change while the window is open. If it did -- someone
        editing the README on the bench machine -- Reload picks it up without
        restarting.
        """
        super().__init__(parent)
        self.session = session
        self.topics = []
        self.hits = []

        lay = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        self.help_search = QtWidgets.QLineEdit()
        self.help_search.setPlaceholderText(
            "Search the documentation — try 'homing', 'roll sweep', "
            "'why is it loud', 'VCM'")
        self.help_search.setClearButtonEnabled(True)
        self.help_search.textChanged.connect(self.filter)
        btn_reload = QtWidgets.QPushButton("Reload")
        btn_reload.setToolTip("Re-read README.md, for when it has been edited "
                              "while this window was open.")
        btn_reload.clicked.connect(self.reload)
        top.addWidget(self.help_search, 1)
        top.addWidget(btn_reload)
        lay.addLayout(top)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.help_list = QtWidgets.QListWidget()
        self.help_list.currentRowChanged.connect(self.show_row)
        self.help_view = QtWidgets.QTextBrowser()
        self.help_view.setOpenExternalLinks(True)
        self.help_list.setMinimumWidth(300)
        split.addWidget(self.help_list)
        split.addWidget(self.help_view)
        # A truncated heading is a heading you cannot search by eye. These are
        # long and specific on purpose, so the list gets a real share.
        split.setSizes([430, 900])
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        lay.addWidget(split, 1)

        self.help_count = QtWidgets.QLabel("")
        self.help_count.setStyleSheet("color:#9aa3b2;")
        lay.addWidget(self.help_count)

        self.reload()

    # ---- index ------------------------------------------------------------

    def reload(self):
        self.topics = ohelp.load_topics()
        self.filter(self.help_search.text())

    def filter(self, query):
        self.hits = ohelp.search(self.topics, query, limit=200)
        self.help_list.clear()
        for t in self.hits:
            item = QtWidgets.QListWidgetItem(
                ("    " if t.level > 2 else "") + t.title)
            item.setToolTip(f"{t.title}  —  {t.source}")
            if t.source == "this window":
                item.setForeground(QtGui.QColor("#8fc7ff"))
            self.help_list.addItem(item)
        self.help_count.setText(
            f"{len(self.hits)} of {len(self.topics)} topics"
            + (f" matching '{query}'" if query.strip() else
               " — the README, plus the topics about this window in blue"))
        if self.hits:
            self.help_list.setCurrentRow(0)
        else:
            self.help_view.setMarkdown(
                f"### Nothing matches '{query}'\n\n"
                f"Search matches whole words in a heading first, then the "
                f"text underneath. Try fewer words, or a term the "
                f"documentation would actually use — 'clkdiv' rather than "
                f"'sample rate setting'.")

    def show_row(self, row):
        if not (0 <= row < len(self.hits)):
            return
        t = self.hits[row]
        self.help_view.setMarkdown(f"## {t.title}\n\n*from {t.source}*\n\n"
                                   + t.body)
        self.help_view.verticalScrollBar().setValue(0)

    def show_for(self, query):
        """Open this tab at a search. For 'explain this' buttons elsewhere."""
        self.help_search.setText(query)
