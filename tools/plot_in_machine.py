#!/usr/bin/env python3
"""
tools/plot_in_machine.py -- the measured box, drawn where it sits next to the coil.

    python tools/plot_in_machine.py captures/volume_20260924_164240.npz

Everything is drawn in RIG millimetres -- stage coordinates -- because that is
the frame the map was measured in and the only one both objects are known in.

How the coil got placed, and how much of it is measured
------------------------------------------------------
Three things are measured on the bench, and one is fitted.

MEASURED
  * the mount post sits at rig (279, 178, -354). That follows from the tape of
    1 Sept -- with the stages Home (rig 0, 0, 300) the probe flange was
    (-279, -178, +654) mm from the post -- and nothing else.
  * the bracket is 127 mm (5 inch) tall, post top to the lowest copper.
  * the coil is 940 mm tall and 711 mm across, 16 turns, run at 250 A.

FITTED
  * the rotation about the vertical, theta = 322 deg. Solved by asking which
    orientation best reproduces the 3,432 measured surface values.

The bracket height is worth dwelling on, because it is the check that makes the
rest credible: BEFORE it was measured, fitting it to the field alone returned
125 mm. The tape then said 5 inch = 127 mm. A 2 mm agreement on a mechanical
dimension recovered from magnetic data is not a coincidence, and it is why the
placement here is worth more than the earlier guesses.

What is still approximate
-------------------------
The fit reaches a correlation of 0.957 and an rms misfit of 6 %, and wants
5,322 amp-turns where the coil is 16 turns at 250 A = 4,000. Those two numbers
are one fact: the modelled winding is not shaped quite like the wound one. The
real coil is a twisted, non-planar saddle whose 940 x 711 mm bounding box does
not correspond to the model's principal axes -- rolling the model upright to
match those dimensions makes the fit WORSE (0.88), which is the evidence for it.

So: the separation and the geometry are right to within about 6 %. Do not take
a clearance off this figure.

The faces are read from the map you give it -- the same box and face logic as
plot_faces.py, so any volume works -- and coloured with |B| averaged over the
working sensors (--exclude S9,S10,S11; the sidecar's excluded sensors and any
reading beyond their range are dropped unless --keep-all). The figure says how
many sensors were averaged. The fit constants above were solved against ONE map
(24 September); drawing a different map with them places the same coil, not a
re-fitted one.

The distance printed is from the winding's CENTRELINE to the box surface. The
model has no winding radius and the real coil is a twisted saddle, so a
clearance between copper and box is not something this can give.
"""

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.transforms import Affine2D

from octobee import machine as omach
from octobee import paths

# Reading the faces out of the map, and choosing which sensors to average, is
# the same job plot_faces.py does, so it is done once, there. It sits next to
# this file; put this directory first so the import works however the script
# was started.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_faces as pf

AXES = ("x", "y", "z")

HOME_MM = np.array([0.0, 0.0, 300.0])       # where the stages were for the tape
TAPE_MM = np.array([-279.0, -178.0, 654.0])  # flange, from the mount post
POST_MM = HOME_MM - TAPE_MM                  # the post, in rig mm
BRACKET_MM = 127.0                           # measured, post top to lowest copper

# The fitted orientation, as a quaternion (w, x, y, z).
#
# A FULL rotation, not a rotation about the vertical. These are stellarator
# modular coils: non-planar by design, which is the whole point of them. A
# non-planar curve has no plane to stand upright, so its orientation needs all
# three degrees of freedom -- fitting only the rotation about z is a planar
# habit applied to a shape that has no plane, and it costs real accuracy:
# z-only reaches 0.957, the full rotation 0.983.
QUAT = np.array([-0.13402842, -0.47053942, 0.87213863, 0.00180383])
FIT_CORR = 0.983
FIT_RMS_PCT = 4.6
FIT_AMPTURNS = 5029


def _rot(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def coil_in_rig(coil_pts):
    """Put the model winding where the bench says it is, in rig millimetres."""
    X = np.asarray(coil_pts, float) @ _rot(QUAT).T
    lowest = X[X[:, 2].argmin()].copy()
    return X - lowest + np.array([0.0, 0.0, BRACKET_MM]) + POST_MM


def axono(P, azim, elev):
    a, e = np.radians(azim), np.radians(elev)
    P = np.atleast_2d(np.asarray(P, float))
    X = -P[:, 0] * np.sin(a) + P[:, 1] * np.cos(a)
    Y = -(P[:, 0] * np.cos(a) + P[:, 1] * np.sin(a)) * np.sin(e) + P[:, 2] * np.cos(e)
    return np.column_stack([X, Y])


def view_vec(azim, elev):
    a, e = np.radians(azim), np.radians(elev)
    return np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])


def load_faces(map_path, step_mm=10.0, exclude=(), keep_all=False):
    """The six faces of the map you were given, ready to draw.

    Returns ({face id: face dict with u, v, axis, at, ue, ve, grid}, lo, hi,
    SensorChoice), derived from `map_path` and its sidecar by plot_faces. It
    used to read a hand-made CSV export of one old map from beside the file,
    and so drew that map's field under whatever name was passed.
    """
    return pf.load_faces(map_path, step_mm, None, exclude, keep_all)


def draw(faces, lo, hi, coil, path, title, sub):
    allv = np.concatenate([f["grid"].ravel() for f in faces.values()])
    allv = allv[np.isfinite(allv)]
    norm = Normalize(vmin=allv.min(), vmax=allv.max())
    cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(13.6, 11.4), dpi=200)
    fig.patch.set_facecolor("white")

    # The third view looks straight down the coil's axis. Elevation 3 deg
    # rather than 0: at exactly zero the box's top face is edge-on and its
    # parallelogram collapses to a line, which makes the affine transform
    # singular. Three degrees keeps it a sliver and keeps the maths honest.
    # Two oblique views to place things, and the two AXIAL views, which are
    # what the geometry is actually judged on: along +y shows the coil's loop
    # face-on, along +x shows how far it wanders out of any one plane -- the
    # whole character of a modular coil, and invisible in every other view.
    VIEWS = [(48, 20, "from the probe side"),
             (150, 26, "from behind the coil"),
             (270, 3, "along +y — face-on to the coil"),
             (0, 3, "along +x — the coil's twist, and the standoff in y")]
    for i, (az, el, label) in enumerate(VIEWS):
        col, row = i % 2, i // 2
        ax = fig.add_axes([0.025 + 0.455 * col, 0.545 - 0.425 * row, 0.44, 0.345])
        ax.set_aspect("equal")
        ax.axis("off")
        v = view_vec(az, el)

        prims = []
        C = np.vstack([coil, coil[:1]])
        c2 = axono(C, az, el)
        cmid = 0.5 * (C[:-1] + C[1:])
        cd = cmid @ v
        # `k`, not `i`: `i` is the view index of the enclosing loop, and
        # shadowing it here works only because col/row are read before this
        # runs. One more use of `i` below and the panels would silently land
        # in the wrong cells.
        for k in range(len(C) - 1):
            prims.append((cd[k], "coil", (c2[k:k + 2, 0], c2[k:k + 2, 1])))

        for f in faces.values():
            iu, iv = AXES.index(f["u"]), AXES.index(f["v"])
            ifx = 3 - iu - iv
            nrm = np.zeros(3)
            nrm[ifx] = 1.0 if f["end"] == 1 else -1.0
            if nrm @ v <= 1e-6:
                continue
            corner = np.zeros(3)
            corner[ifx], corner[iu], corner[iv] = f["at"], lo[iu], lo[iv]
            Pu = corner.copy()
            Pu[iu] = hi[iu]
            Pv = corner.copy()
            Pv[iv] = hi[iv]
            quad = np.vstack([corner, Pu, Pu + (Pv - corner), Pv])
            q2 = axono(quad, az, el)
            prims.append((quad.mean(axis=0) @ v, "face",
                          (f["grid"], q2[0], q2[1] - q2[0], q2[3] - q2[0])))

        # Far to near. matplotlib draws by zorder, not by the order artists
        # were added, so the depth order has to BE the zorder: each primitive
        # takes its rank in the sort (its outline or inner stroke half a step
        # above it). Fixed zorders per kind -- coil 1, faces 2 -- made the coil
        # draw behind the box in every view, even those where it is in front.
        # The depth of a primitive is its centre's, so this is a painter's
        # approximation: right for a coil that lies wholly on one side of the
        # box, as this one does, and not exact for a segment that passes
        # across a face's plane.
        prims.sort(key=lambda t: t[0])
        pts = []
        for rank, (_, kind, pay) in enumerate(prims):
            z = 10.0 + rank
            if kind == "coil":
                ax.plot(pay[0], pay[1], color="#1a1a1a", lw=7.5,
                        solid_capstyle="round", zorder=z)
                ax.plot(pay[0], pay[1], color="#4a4a4a", lw=4.0,
                        solid_capstyle="round", zorder=z + 0.5)
                pts.append(np.column_stack(pay))
            else:
                grid, o2, e1, e2 = pay
                im = ax.imshow(grid, origin="lower", cmap=cmap, norm=norm,
                               extent=[0, 1, 0, 1], interpolation="nearest",
                               zorder=z)
                im.set_transform(Affine2D.from_values(
                    e1[0], e1[1], e2[0], e2[1], o2[0], o2[1]) + ax.transData)
                q = np.vstack([o2, o2 + e1, o2 + e1 + e2, o2 + e2])
                ax.plot(*np.vstack([q, q[:1]]).T, color="#111", lw=1.1,
                        zorder=z + 0.5)
                pts.append(q)

        P2 = np.vstack(pts)
        pad = 0.05 * max(np.ptp(P2[:, 0]), np.ptp(P2[:, 1]))
        ax.set_xlim(P2[:, 0].min() - pad, P2[:, 0].max() + pad)
        ax.set_ylim(P2[:, 1].min() - pad, P2[:, 1].max() + pad)

        s0 = axono(np.zeros((1, 3)), az, el)[0]
        L = np.linalg.norm(axono(np.array([[200.0, 0, 0]]), az, el)[0] - s0)
        x0, y0 = P2[:, 0].min() + pad, P2[:, 1].min()
        ax.plot([x0, x0 + L], [y0, y0], color="#111", lw=2.4)
        ax.text(x0 + L / 2, y0 - pad * 0.4, "200 mm", fontsize=9.5,
                ha="center", va="top", color="#333")

        org = np.array([0.90, 0.13])
        base = axono(np.zeros((1, 3)), az, el)[0]
        for k, nm in enumerate("xyz"):
            d = np.zeros(3)
            d[k] = 1.0
            vec = axono(d[None, :], az, el)[0] - base
            n2 = np.linalg.norm(vec)
            if n2 < 1e-3:
                # This axis points at the camera -- it has no length on screen.
                # Mark it as a dot rather than dividing by zero for an arrow.
                ax.annotate("⊙ " + nm, xy=org + np.array([0.0, 0.075]),
                            xycoords="axes fraction",
                            fontsize=10, color="#c0392b", ha="center",
                            va="center", zorder=6)
                continue
            vec = vec / n2 * 0.068
            ax.annotate("", xy=org + vec, xytext=org, xycoords="axes fraction",
                        textcoords="axes fraction", zorder=6,
                        arrowprops={"arrowstyle": "-|>", "color": "#c0392b",
                                    "lw": 1.3, "shrinkA": 0, "shrinkB": 0})
            ax.annotate(nm, xy=org + vec * 1.4, xycoords="axes fraction",
                        fontsize=10, color="#c0392b", ha="center", va="center",
                        zorder=6)
        ax.set_title(label, fontsize=11.5, color="#333", pad=8)

    cax = fig.add_axes([0.945, 0.34, 0.012, 0.34])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("|B| measured on the surface  (mT)", fontsize=10.5, labelpad=11)
    cb.ax.tick_params(labelsize=9)

    fig.text(0.025, 0.978, title, fontsize=17, fontweight="bold", color="#111")
    fig.text(0.025, 0.957, sub, fontsize=10.5, color="#555")
    fig.text(0.025, 0.068,
             "MEASURED: the mount post, from the tape of 1 Sept; the 127 mm "
             "bracket; 16 turns at 250 A; and the field on the box.\n"
             "FITTED: the coil's orientation \u2014 all three axes, since a "
             "modular coil is non-planar and has no plane to stand upright.",
             fontsize=8.5, color="#555", linespacing=1.7)
    fig.text(0.025, 0.014,
             f"Correlation {FIT_CORR:.3f}, rms {FIT_RMS_PCT:.1f} %, implying "
             f"{FIT_AMPTURNS:,} amp-turns against the 4,000 actually run. Before "
             f"the bracket was measured, fitting it to the field alone returned "
             f"125 mm; the tape then said 127.\n"
             f"That 2 mm agreement is why the placement is worth trusting for "
             f"scale and separation \u2014 but the winding SHAPE is the model's, not "
             f"the wound one's, so do not take a clearance off this.",
             fontsize=8.5, color="#a0522d", linespacing=1.7)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("map")
    p.add_argument("--coil", default=None,
                   help="simsopt coil file (default: the one "
                        f"{omach.CONFIG_NAME} names, as the GUI uses)")
    p.add_argument("--label", default="C1")
    p.add_argument("--step", type=float, default=10.0)
    p.add_argument("--out", default=None)
    pf.add_sensor_args(p)
    a = p.parse_args(argv)
    exclude = pf.exclude_from_args(p, a)

    try:
        faces, lo, hi, choice = load_faces(a.map, a.step, exclude, a.keep_all)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if len(faces) != 6:
        print(f"expected six faces, found {len(faces)}", file=sys.stderr)
        return 2

    coil_file = a.coil or omach.MachineConfig.load_or_default(
        paths.config(omach.CONFIG_NAME), on_error=print).coil_file
    if not coil_file:
        print(f"no --coil given and none named in {omach.CONFIG_NAME}",
              file=sys.stderr)
        return 2
    cs = omach.CoilSet.load(coil_file)
    hits = [c for c in cs if c.label == a.label]
    if not hits:
        print(f"no coil labelled {a.label!r} in {coil_file}; it has "
              f"{', '.join(c.label for c in cs)}", file=sys.stderr)
        return 2
    coil = coil_in_rig(np.asarray(hits[0].points_mm, float))

    # closest approach from the winding's centreline vertices to the box
    # SURFACE, not its centre. Vertices, not the segments between them, and
    # no winding radius: a centreline distance, labelled as one below.
    d = np.maximum(np.maximum(lo - coil, coil - hi), 0.0)
    gap = float(np.linalg.norm(d, axis=1).min())

    stem = os.path.splitext(os.path.basename(a.map))[0]
    out = a.out or os.path.join(paths.captures_dir(), "plots",
                                stem + "_in_machine.png")
    sub = (f"{hi[0]-lo[0]:.0f} x {hi[1]-lo[1]:.0f} x {hi[2]-lo[2]:.0f} mm "
           f"measured volume and coil {a.label}, both in rig millimetres - "
           f"winding centreline {gap:.0f} mm from the box surface - "
           f"{choice.text} - {stem}")
    path = draw(faces, lo, hi, coil, out,
                "What we measured, and where the coil is", sub)
    print(f"wrote {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
    print(f"mount post at rig ({POST_MM[0]:.0f}, {POST_MM[1]:.0f}, {POST_MM[2]:.0f}) mm")
    print(f"coil spans rig x {coil[:,0].min():.0f}..{coil[:,0].max():.0f}  "
          f"y {coil[:,1].min():.0f}..{coil[:,1].max():.0f}  "
          f"z {coil[:,2].min():.0f}..{coil[:,2].max():.0f}")
    print(f"nearest winding CENTRELINE to the measured surface: {gap:.0f} mm "
          f"(no winding radius subtracted)")
    print(f"colour: {choice.text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
