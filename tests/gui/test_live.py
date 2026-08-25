"""The live plot."""


import numpy as np

from octobee.gui.widgets.plot import LivePlot
from octobee.calib import geometry as pgeom
from tests.helpers import (
    check,
)



def test_live_plot_reset(app, workdir):
    """Reset view has to leave the plot AUTO-ranging, not merely re-fitted.

    The trap: ViewBox.autoRange() fits the data and switches auto-ranging off
    as it does it, so the obvious implementation looks right for one frame and
    then freezes. The assertion is therefore on the ViewBox's auto-range state,
    not on the axis limits.
    """
    print("\nlive plot reset view")
    plot = LivePlot(pgeom.Geometry())
    vb = plot.plot.getViewBox()

    rng = np.random.default_rng(4)
    plot.update_data(rng.normal(0, 1, (400, 16, 3)).astype(np.float32), 500.0)
    app.processEvents()
    check("a fresh plot auto-ranges", all(vb.state["autoRange"]),
          str(vb.state["autoRange"]))

    # What a drag and a scroll do, which is what the button has to undo.
    vb.setRange(xRange=(-0.5, -0.4), yRange=(100.0, 200.0))
    vb.setMouseEnabled(x=False, y=True)
    check("zooming turns auto-ranging off, silently",
          not any(vb.state["autoRange"]), str(vb.state["autoRange"]))

    plot.reset_view()
    app.processEvents()
    check("reset view puts BOTH axes back on auto", all(vb.state["autoRange"]),
          str(vb.state["autoRange"]))
    check("and re-enables the mouse on both axes",
          all(vb.state["mouseEnabled"]), str(vb.state["mouseEnabled"]))

    # The flag being set is not the same as the flag WORKING: a view left
    # disabled ignores updateAutoRange() entirely, so recomputing and seeing
    # the range follow ten-times-larger data is what proves the button did
    # something. The explicit call stands in for the repaint that the live
    # window does ten times a second and that a never-shown widget never gets.
    before = vb.viewRange()[1][1]
    plot.update_data(rng.normal(0, 40, (400, 16, 3)).astype(np.float32), 500.0)
    vb.updateAutoRange()
    app.processEvents()
    check("the view follows the data afterwards, not frozen where it was",
          vb.viewRange()[1][1] > before * 2,
          f"y top {before:.2f} -> {vb.viewRange()[1][1]:.2f}")
    plot.deleteLater()
