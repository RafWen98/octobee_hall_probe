"""
One module per tab.

Each is a QWidget that builds its own controls, owns them, and reads what it
needs from the Session it is given. A tab never reaches into a sibling: what it
wants from the rest of the window arrives as a constructor argument, and what
it wants the rest of the window to do it says with a signal.
"""
