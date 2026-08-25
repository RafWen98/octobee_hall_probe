"""Where each sensor sits, and which way its chip faces."""


import numpy as np

from octobee.calib import geometry as pgeom
from tests.helpers import (
    check,
)



def test_geometry():
    print("\ngeometry")
    g = pgeom.Geometry()
    R = g.rotations()
    check("rotations are orthonormal",
          all(np.allclose(R[i].T @ R[i], np.eye(3)) for i in range(16)))
    check("rotations are right-handed",
          all(np.linalg.det(R[i]) > 0.99 for i in range(16)))
    # The boards lie flat on the faces and reach out tangentially, so a chip
    # sits barely off its own face but a long way round the tube.
    check("chips sit just clear of the face they are bolted to",
          abs(g.fsv_height_mm - (g.board_thickness_mm + g.fsv_above_board_mm)) < 1e-9,
          f"{g.fsv_height_mm:g} mm off the face, reaching "
          f"{g.fsv_radius_mm:.1f} mm from the axis")
    arm_dirs = np.array([g.arm_dir(i) for i in range(1, 17)])
    check("arms run tangentially, not radially",
          np.allclose(np.einsum("ij,ij->i", arm_dirs, g.normals()), 0.0, atol=1e-9)
          and np.allclose(arm_dirs @ pgeom.TUBE_AXIS, 0.0, atol=1e-9),
          "perpendicular to both the face normal and the tube axis")
    # A chip reading purely on its own +X must map along its arm.
    b = np.zeros((16, 3))
    b[:, 0] = 1.0
    check("chip +X maps along the arm", np.allclose(g.to_tube_frame(b), arm_dirs))
    b = np.zeros((16, 3))
    b[:, 2] = 1.0
    check("chip +Z maps out of the board, along the face normal",
          np.allclose(g.to_tube_frame(b), g.normals()))
    b = np.zeros((16, 3))
    b[:, 1] = 1.0
    check("chip +Y maps along the tube",
          np.allclose(np.abs(g.to_tube_frame(b) @ pgeom.TUBE_AXIS), 1.0))
    # |B| must survive the rotation, that being the whole point of using it.
    rng = np.random.default_rng(1)
    v = rng.normal(size=(50, 16, 3))
    check("|B| is invariant under the rotation",
          np.allclose(np.linalg.norm(v, axis=-1),
                      np.linalg.norm(g.to_tube_frame(v), axis=-1)))
    # The bench mounting: drawing-only, but a mirrored one would silently put
    # the probe's tip at the wrong end of the rig in every screenshot.
    check("the mount is a proper rotation",
          np.allclose(pgeom.MOUNT_ROT.T @ pgeom.MOUNT_ROT, np.eye(3))
          and np.linalg.det(pgeom.MOUNT_ROT) > 0.99)
    check("mounted horizontally: the tube axis lies along world +Y",
          np.allclose(pgeom.to_world(pgeom.TUBE_AXIS), [0, 1, 0]))
    pw = pgeom.to_world(g.positions())
    check("the tip end points forward, S16 with it",
          pw[15][1] > pw[12][1] and np.argmax(pw[:, 1]) % 4 == 3,
          f"S16 at y = {pw[15][1]:.0f} mm, S13 at y = {pw[12][1]:.0f} mm")
    check("|B| survives the mount rotation too",
          np.allclose(np.linalg.norm(v, axis=-1),
                      np.linalg.norm(pgeom.to_world(v), axis=-1)))

    pt = (g.fsv_radius_mm + 20.0, 0.0, 60.0)
    w = g.expected_response(pt)
    check("geometry weighting favours the nearest chip",
          int(np.argmax(w)) + 1 == g.nearest_sensor(pt),
          f"nearest is S{g.nearest_sensor(pt)}, spread {w.max()/w.min():.0f}x")
