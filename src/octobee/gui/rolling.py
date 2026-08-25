"""The ring buffer behind the live plot."""


import numpy as np
from octobee.gui.constants import N_SENSORS


# ==========================================================================
# small widgets
# ==========================================================================

class Rolling:
    """Fixed-length rolling window of (n, 16, 3) millitesla."""

    def __init__(self, npoints):
        self.buf = np.zeros((npoints, N_SENSORS, 3), dtype=np.float32)
        self.n = npoints
        self.filled = 0

    def resize(self, npoints):
        if npoints == self.n:
            return
        new = np.zeros((npoints, N_SENSORS, 3), dtype=np.float32)
        k = min(npoints, self.filled)
        if k:
            new[-k:] = self.buf[-k:]
        self.buf, self.n, self.filled = new, npoints, k

    def clear(self):
        self.filled = 0

    def push(self, block):
        k = block.shape[0]
        if k >= self.n:
            self.buf[:] = block[-self.n:]
            self.filled = self.n
        else:
            self.buf[:-k] = self.buf[k:]
            self.buf[-k:] = block
            self.filled = min(self.n, self.filled + k)

    def view(self):
        return self.buf[self.n - self.filled:]
