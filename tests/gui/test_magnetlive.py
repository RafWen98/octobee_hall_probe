"""The live plot the guided magnet run draws while it is measuring."""

import numpy as np

from octobee.gui.widgets.magnetlive import LOUD, MagnetPassPlot
from tests.helpers import check


def _sweep(n=60, loud=(0, 1, 2, 3), peak_at=(10.0, 25.0, 40.0, 55.0)):
    """A pose's worth of points: four sensors peaking in turn, twelve flat."""
    xs = np.linspace(0.0, 60.0, n)
    rows = []
    for x in xs:
        b = np.zeros((16, 3))
        b[:, 0] = 0.05                       # the twelve that are facing away
        for s, c in zip(loud, peak_at):
            b[s, 0] = 3.0 / (1.0 + ((x - c) / 6.0) ** 2)
        rows.append(b)
    return xs, rows


def test_magnet_live_plot(app):
    """What it draws has to be the measurement, and has to stay legible."""
    print("\nguided magnet run, the live plot")
    w = MagnetPassPlot()
    w.resize(800, 400)
    w.begin("pose 1 of 4 - pass A", "y")
    xs, rows = _sweep()
    for x, row in zip(xs, rows):
        w.add(x, row)

    x0, y0 = w.curves[0].getData()
    check("every point reaches the curves", len(x0) == len(xs))
    check("and what is drawn is |B|, not one axis of it",
          np.allclose(y0, [np.linalg.norm(r[0]) for r in rows]),
          f"first point {y0[0]:.4f} against "
          f"{np.linalg.norm(rows[0][0]):.4f} mT")

    # The four sensors whose face is turned to the magnet are the ones worth
    # looking at; the other twelve are metres away in 1/r^3 terms and would
    # otherwise be twelve flat lines competing for the eye.
    check(f"the {LOUD} answering sensors are the ones named",
          w._named == {0, 1, 2, 3}, f"named {sorted(w._named)}")
    loud_w = w.curves[0].opts["pen"].widthF()
    quiet_w = w.curves[8].opts["pen"].widthF()
    check("they are drawn bright and the other twelve faint",
          loud_w > quiet_w and w.curves[8].opts["pen"].color().alpha() < 255,
          f"pen {loud_w:g} against {quiet_w:g}, alpha "
          f"{w.curves[8].opts['pen'].color().alpha()}")

    # The regression this pins: the full-scale marker used to take part in the
    # auto-range, so a 20 mT range on a run peaking at 3 mT pulled the y axis
    # up to 20 and squashed every trace into the bottom seventh of the plot.
    # The marker that exists to make one failure visible hid everything else.
    # Asserted on childrenBounds rather than viewRange: the view only
    # auto-ranges when it is painted, and offscreen it never is -- so
    # viewRange() here is the untouched default (0, 1), which "passes" this
    # check no matter what the marker does. childrenBounds is what the auto
    # range is computed FROM, and is where ignoreBounds takes effect.
    vb = w.plot.getViewBox()
    w.set_full_scale(20.0)
    app.processEvents()
    top = vb.childrenBounds()[1][1]
    check("a full scale far above the data does not flatten the plot",
          top < 6.0, f"the auto range would reach {top:.2f} mT against a "
                     f"3 mT peak and a 20 mT full scale")

    w.set_full_scale(2.0)
    app.processEvents()
    vb.autoRange()
    check("and one the data crosses is in view where it can be read",
          vb.viewRange()[1][1] > 2.0,
          f"y axis reaches {vb.viewRange()[1][1]:.2f} mT")

    # Ring markers are positions along the TUBE. Passes B and C plot against x
    # and z, so the wizard passes an empty list there -- a line in the right
    # units at the wrong place is worse than no line.
    w.set_rings([10.0, 25.0, 40.0, 55.0])
    check("ring markers are drawn where they are asked for",
          len(w._rings) == 4 and
          [round(r.value(), 1) for r in w._rings] == [10.0, 25.0, 40.0, 55.0])
    w.set_rings([])
    check("and clear when the next pass is not sweeping the tube axis",
          not w._rings)

    # A new pass starts empty, including the legend -- a different face
    # answers in every pose and stale names are worse than none.
    w.begin("pose 1 of 4 - pass B", "x")
    drawn = w.curves[0].getData()[0]
    check("a new pass clears the traces",
          not w._x and (drawn is None or len(drawn) == 0))
    check("and the names that went with them", not w._named)

    # The scan drops a point that failed and carries on, so this is handed
    # whatever run_scan produced rather than a guaranteed shape.
    w.add(1.0, np.zeros((3, 3)))
    check("a malformed row is ignored rather than raising", not w._x)

    # Which sensors are loud is read off the running MAXIMUM, not the latest
    # point: a sensor should stay bright once its peak has gone by instead of
    # flicking back to faint the moment the magnet moves off it.
    w.begin("pose 2 of 4 - pass A", "y")
    for x, row in zip(xs, rows):
        w.add(x, row)
    tail = np.linalg.norm(rows[-1], axis=1)
    check("emphasis follows the running peak, not the last point",
          w._named == {0, 1, 2, 3}
          and int(np.argmax(tail)) in w._named
          and len(w._named) == LOUD,
          f"last point is loudest on S{int(np.argmax(tail)) + 1}, "
          f"named {sorted(s + 1 for s in w._named)}")
    w.close()
