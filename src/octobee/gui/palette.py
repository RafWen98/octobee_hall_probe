"""One colour per sensor, matched to the 3D view so the two agree."""


import pyqtgraph as pg

from octobee.gui.widgets.probe3d import color_for
from octobee.gui.constants import N_SENSORS

def sensor_colors():
    return [pg.mkColor([int(255 * c) for c in color_for(i / (N_SENSORS - 1))[:3]])
            for i in range(N_SENSORS)]
