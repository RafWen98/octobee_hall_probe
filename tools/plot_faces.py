#!/usr/bin/env python3
"""
tools/plot_faces.py -- the six faces of a volume map, as a shareable picture.

Two figures, from one map:

  <out>_net.png    the box UNFOLDED, faces laid out so shared edges touch
  <out>_cube.png   the same six faces wrapped back onto a 3D box

The net is the one to send someone. A 2x3 grid of panels is easier to draw and
tells you nothing about how the faces meet; unfolded, a feature that runs across
an edge stays continuous across the join, and a face that is mirrored or
mis-registered announces itself as a discontinuity at the seam. That is the whole
reason to prefer it.

One colour scale is shared by all six faces and drawn once. Per-face scales would
make each panel prettier and every comparison between them meaningless.

    python tools/plot_faces.py captures/volume_20260924_164240.npz
    python tools/plot_faces.py <map.npz> --sensor 1      # one chip, not the mean
    python tools/plot_faces.py <map.npz> --step 10 --out captures/plots/mymap

The map is opened read-only. Nothing here writes to it.
"""

import argparse
import json
import os
import sys

import numpy as np

# Agg: this is a file-writing tool and may run with no display attached.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.colors import Normalize               # noqa: E402

from octobee import paths                         # noqa: E402

AXES = ("x", "y", "z")

# Where each face sits in the unfolded net, as (column, row) on a 4x3 lattice,
# with the flags needed to make the shared edges actually line up.
#
#              +--------+
#              |  z270  |
#     +--------+--------+--------+--------+
#     |   x0   |   y0   |  x250  |  y250  |
#     +--------+--------+--------+--------+
#              |   z70  |
#              +--------+
#
# `flip_u` mirrors the horizontal axis. Two faces need it: x0 is folded out to
# the LEFT, so its y runs away from the viewer rather than toward them, and y250
# is the far wall seen from behind, so its x is reversed relative to y0. Getting
# either wrong produces a net that looks plausible and is mirrored -- exactly the
# failure this layout exists to expose, so it is worth stating rather than
# discovering.
NET = {
    "z270": dict(col=1, row=0, u="x", v="y", flip_u=False, flip_v=False,
                 title="z = 270", role="ceiling"),
    "x0":   dict(col=0, row=1, u="y", v="z", flip_u=True,  flip_v=False,
                 title="x = 0",   role="left end"),
    "y0":   dict(col=1, row=1, u="x", v="z", flip_u=False, flip_v=False,
                 title="y = 0",   role="front wall"),
    "x250": dict(col=2, row=1, u="y", v="z", flip_u=False, flip_v=False,
                 title="x = 250", role="right end"),
    "y250": dict(col=3, row=1, u="x", v="z", flip_u=True,  flip_v=False,
                 title="y = 250", role="back wall"),
    "z70":  dict(col=1, row=2, u="x", v="y", flip_u=False, flip_v=True,
                 title="z = 70",  role="floor"),
}


def line_faces(path, lo, hi):
    """Which face(s) each swept line lies on, from the sidecar's own record.

    Every line records the TWO axes it held fixed, e.g. {"y": 0, "z": 120}. A
    line belongs to a face for each fixed axis that sits at an end of the box --
    so most lines belong to one, and a line along a box EDGE, with both fixed
    axes at an extreme, belongs to two. Assigning such a line to one face and
    not the other leaves a missing row along every seam.

    Do NOT infer this from the positions. On a shell map both non-swept axes are
    fixed, so "the axis that varies least" is decided by noise, and half the
    lines of each face get filed under a neighbouring one -- which looks like a
    face that is mostly empty with a few stripes across it.

    Returns {face_id: [line index, ...]}, or None if there is no sidecar to
    read, in which case the caller falls back to geometry.
    """
    side = os.path.splitext(path)[0] + ".json"
    if not os.path.exists(side):
        return None
    with open(side, encoding="utf-8") as fh:
        lines = json.load(fh).get("lines")
    if not lines:
        return None

    faces = {}
    for i, ln in enumerate(lines):
        for ax, at in (ln.get("fixed") or {}).items():
            if ax not in AXES:
                continue
            k = AXES.index(ax)
            for end in (lo[k], hi[k]):
                if abs(float(at) - end) < step_tol(lo[k], hi[k]):
                    faces.setdefault(f"{ax}{end:.0f}", []).append(i)
                    break
    return faces


def step_tol(lo_v, hi_v):
    """How close to an end counts as being on it. A hair, in millimetres."""
    return 1.0


def load_faces(path, step_mm, sensor=None):
    """Read a volume map and bin each face onto the grid.

    Returns {face_id: (u_edges, v_edges, grid)} plus the box corners. `sensor`
    is 1-16 for one chip, or None for the mean over all sixteen -- the mean
    smears the field over the ~160 mm the chips are strung along, which is
    honest but blunt, so both are offered.
    """
    with np.load(path) as z:                       # read-only; never written
        ins = z["in_span"]
        pos = z["pos_mm"][ins]
        b = z["b_mt"][ins]
        li = z["line_index"][ins]

    mag = np.linalg.norm(b, axis=2)                # (N, 16)
    val = mag.mean(axis=1) if sensor is None else mag[:, sensor - 1]

    lo, hi = pos.min(axis=0), pos.max(axis=0)
    # Snap to the grid: the ends of a line carry a millimetre of run-up, and a
    # box whose corner is at 69.98 rather than 70 puts every face half a cell out.
    lo = np.round(lo / step_mm) * step_mm
    hi = np.round(hi / step_mm) * step_mm

    faces = line_faces(path, lo, hi)
    if faces is None:
        print("  note: no sidecar -- falling back to inferring faces from "
              "positions, which is unreliable on a shell map", file=sys.stderr)
        faces = {}
        for L in np.unique(li):
            p = pos[li == L]
            i = int(np.argmin(p.max(axis=0) - p.min(axis=0)))
            at = p[:, i].mean()
            for end in (lo[i], hi[i]):
                if abs(at - end) < 2.0:
                    faces.setdefault(f"{AXES[i]}{end:.0f}", []).append(int(L))
                    break

    out = {}
    for fid, lines in faces.items():
        spec = NET.get(fid)
        if spec is None:
            print(f"  note: face {fid} is not in the net layout, skipping",
                  file=sys.stderr)
            continue
        sel = np.isin(li, lines)
        p, v = pos[sel], val[sel]
        iu, iv = AXES.index(spec["u"]), AXES.index(spec["v"])
        ue = np.arange(lo[iu], hi[iu] + step_mm * 0.5, step_mm)
        ve = np.arange(lo[iv], hi[iv] + step_mm * 0.5, step_mm)
        ui = np.clip(np.round((p[:, iu] - ue[0]) / step_mm).astype(int), 0, len(ue) - 1)
        vi = np.clip(np.round((p[:, iv] - ve[0]) / step_mm).astype(int), 0, len(ve) - 1)
        acc = np.zeros((len(ve), len(ue)))
        cnt = np.zeros((len(ve), len(ue)))
        np.add.at(acc, (vi, ui), v)
        np.add.at(cnt, (vi, ui), 1)
        grid = np.divide(acc, cnt, out=np.full_like(acc, np.nan), where=cnt > 0)
        out[fid] = (ue, ve, grid)
    return out, lo, hi


def draw_net(faces, lo, hi, step_mm, title, sub, cbar_label, path):
    """The unfolded box, one shared colour scale."""
    allv = np.concatenate([g.ravel() for _, _, g in faces.values()])
    allv = allv[np.isfinite(allv)]
    norm = Normalize(vmin=allv.min(), vmax=allv.max())
    cmap = plt.get_cmap("viridis")

    span = hi - lo                                 # x, y, z extents in mm
    # Column widths and row heights follow the real geometry, so the net is to
    # scale rather than six equal boxes.
    colw = [span[1], span[0], span[1], span[0]]    # x0 | y0 | x250 | y250
    rowh = [span[1], span[2], span[1]]             # z270 / walls / z70

    fig_w, fig_h = 15.5, 9.2
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor("white")

    L, B, W, H = 0.055, 0.055, 0.80, 0.80
    xo = np.concatenate([[0], np.cumsum(colw)]) / sum(colw)
    yo = np.concatenate([[0], np.cumsum(rowh)]) / sum(rowh)
    GAP = 0.004

    for fid, spec in NET.items():
        if fid not in faces:
            continue
        ue, ve, grid = faces[fid]
        c, r = spec["col"], spec["row"]
        ax = fig.add_axes([
            L + W * (xo[c] + GAP),
            B + H * (1.0 - yo[r + 1] + GAP),
            W * (xo[c + 1] - xo[c] - 2 * GAP),
            H * (yo[r + 1] - yo[r] - 2 * GAP),
        ])
        g = grid
        if spec["flip_u"]:
            g = g[:, ::-1]
        if spec["flip_v"]:
            g = g[::-1, :]
        # A cell is CENTRED on its grid coordinate, so it spans +/- half a step.
        # Drawing it from the coordinate instead shifts the whole face half a
        # cell and makes the box read 10 mm wider than it is.
        h = step_mm / 2.0
        ax.imshow(g, origin="lower", cmap=cmap, norm=norm, aspect="auto",
                  interpolation="nearest",
                  extent=[ue[0] - h, ue[-1] + h, ve[0] - h, ve[-1] + h])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#333"); s.set_linewidth(0.9)

        fin = grid[np.isfinite(grid)]
        # Title above the frame, range under it -- stacked, because both
        # centred on the same line is how the first version of this collided.
        ax.set_title(f"{spec['title']}  ·  {spec['role']}\n"
                     f"{fin.min():.2f}–{fin.max():.2f} mT",
                     fontsize=11, fontweight="bold", pad=6, color="#111",
                     linespacing=1.55)
        u_lab = spec["u"] + (" ←" if spec["flip_u"] else " →")
        ax.text(0.5, -0.03, f"{u_lab}   {ue[0]:.0f}–{ue[-1]:.0f} mm",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, color="#777")
        ax.text(-0.022, 0.5, f"{spec['v']}   {ve[0]:.0f}–{ve[-1]:.0f}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color="#777", rotation=90)

    cax = fig.add_axes([L + W + 0.035, B + 0.10, 0.016, H * 0.55])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(cbar_label, fontsize=11, labelpad=11)
    cb.ax.tick_params(labelsize=9.5)
    cb.outline.set_linewidth(0.8)

    fig.text(0.055, 0.965, title, fontsize=17, fontweight="700", color="#111")
    fig.text(0.055, 0.933, sub, fontsize=11, color="#555")
    fig.text(0.055, 0.016,
             "Box unfolded: the four walls in a row, ceiling above and floor "
             "below the front wall. Shared edges touch, so a feature crossing "
             "a seam stays continuous. One colour scale across all six.",
             fontsize=9, color="#777")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, facecolor="white", bbox_inches=None)
    plt.close(fig)
    return path


def _axono(P, azim_deg, elev_deg):
    """Project 3-D points to 2-D. A camera, written out, not a 3-D axes.

    matplotlib's 3-D backend sorts whole polygons by depth and has no real
    occlusion, so a box built from thousands of cell quads z-fights and shows
    its far faces through its near ones. Projecting by hand and drawing only
    the faces that face the camera makes occlusion exact by construction, and
    keeps every cell crisp because each face stays a single image.
    """
    a = np.radians(azim_deg)
    e = np.radians(elev_deg)
    P = np.atleast_2d(P)
    X = -P[:, 0] * np.sin(a) + P[:, 1] * np.cos(a)
    Y = -(P[:, 0] * np.cos(a) + P[:, 1] * np.sin(a)) * np.sin(e) + P[:, 2] * np.cos(e)
    return np.column_stack([X, Y])


def _view_dir(azim_deg, elev_deg):
    """Unit vector from the box toward the camera."""
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    return np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])


def draw_cube(faces, lo, hi, title, sub, cbar_label, path, step_mm=10.0):
    """The six faces on a box -- only the three facing the camera, per view."""
    from matplotlib.transforms import Affine2D

    allv = np.concatenate([g.ravel() for _, _, g in faces.values()])
    allv = allv[np.isfinite(allv)]
    norm = Normalize(vmin=allv.min(), vmax=allv.max())
    cmap = plt.get_cmap("viridis")
    h = step_mm / 2.0

    fig = plt.figure(figsize=(14.5, 7.0), dpi=200)
    fig.patch.set_facecolor("white")

    VIEWS = [(38, 24, "toward the x = 0 / y = 250 corner"),
             (218, 24, "the opposite corner")]

    for col, (az, el, label) in enumerate(VIEWS):
        ax = fig.add_axes([0.02 + 0.44 * col, 0.08, 0.42, 0.76])
        ax.set_aspect("equal")
        ax.axis("off")
        v = _view_dir(az, el)
        pts2 = []

        for fid, spec in NET.items():
            if fid not in faces:
                continue
            ue, ve, grid = faces[fid]
            iu, iv = AXES.index(spec["u"]), AXES.index(spec["v"])
            ifx = 3 - iu - iv
            at = float(fid[1:])
            # Outward normal: +1 at the high end of its axis, -1 at the low.
            sgn = 1.0 if abs(at - hi[ifx]) < 1e-6 else -1.0
            nrm = np.zeros(3)
            nrm[ifx] = sgn
            if nrm @ v <= 1e-6:          # facing away -- do not draw it at all
                continue

            # The face as a parallelogram: an origin corner and two edges.
            # Span the box exactly. Extending each face half a cell past the
            # corners (which is where its outermost cell centres really sit)
            # makes neighbouring faces overlap by 5 mm, and every shared edge
            # then draws as a doubled line. Stretching the grid over the true
            # bounds instead is a sub-cell distortion and invisible.
            O = np.zeros(3); O[ifx] = at; O[iu] = lo[iu];  O[iv] = lo[iv]
            Pu = O.copy(); Pu[iu] = hi[iu]
            Pv = O.copy(); Pv[iv] = hi[iv]
            o2, u2, v2 = _axono(np.vstack([O, Pu, Pv]), az, el)
            e1, e2 = u2 - o2, v2 - o2
            tr = Affine2D.from_values(e1[0], e1[1], e2[0], e2[1], o2[0], o2[1])

            im = ax.imshow(grid, origin="lower", cmap=cmap, norm=norm,
                           extent=[0, 1, 0, 1], interpolation="nearest",
                           zorder=2)
            im.set_transform(tr + ax.transData)
            # imshow with a transform does not grow the data limits itself.
            quad = np.vstack([o2, o2 + e1, o2 + e1 + e2, o2 + e2])
            pts2.append(quad)
            ax.plot(*np.vstack([quad, quad[:1]]).T, color="#222", lw=1.0,
                    zorder=3)

        P = np.vstack(pts2)
        pad = 0.06 * max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
        ax.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad)
        ax.set_ylim(P[:, 1].min() - pad, P[:, 1].max() + pad)

        # Axis tripod in the corner of the PANEL, not on the box. Drawn over
        # the data it competes with the very gradient it is there to orient.
        org = np.array([0.10, 0.12])
        base = _axono(np.zeros((1, 3)), az, el)[0]
        for k, name in enumerate("xyz"):
            d = np.zeros(3); d[k] = 1.0
            vec = _axono(d[None, :], az, el)[0] - base
            vec = vec / np.linalg.norm(vec) * 0.075
            ax.annotate("", xy=org + vec, xytext=org, xycoords="axes fraction",
                        textcoords="axes fraction", zorder=6,
                        arrowprops=dict(arrowstyle="-|>", color="#c0392b",
                                        lw=1.3, shrinkA=0, shrinkB=0))
            ax.annotate(name, xy=org + vec * 1.35, xycoords="axes fraction",
                        fontsize=10, color="#c0392b", ha="center",
                        va="center", zorder=6)
        ax.set_title(label, fontsize=11.5, color="#333", pad=10)

    cax = fig.add_axes([0.905, 0.22, 0.014, 0.48])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(cbar_label, fontsize=10.5, labelpad=10)
    cb.ax.tick_params(labelsize=9)

    fig.text(0.02, 0.945, title, fontsize=16, fontweight="bold", color="#111")
    fig.text(0.02, 0.905, sub, fontsize=10.5, color="#555")
    fig.text(0.02, 0.022,
             "Axonometric projection, drawn in 2-D. Each view shows only the "
             "three faces turned toward the camera, so nothing is hidden behind "
             "anything and no cell is drawn twice. Between them the two views "
             "cover all six faces.",
             fontsize=8.5, color="#777")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("map", help="a volume_*.npz written by a volume sweep")
    p.add_argument("--step", type=float, default=10.0,
                   help="grid bin, mm (default 10, the sweep's line spacing)")
    p.add_argument("--sensor", type=int, default=None, metavar="N",
                   help="plot sensor N (1-16) instead of the 16-sensor mean")
    p.add_argument("--out", default=None,
                   help="output basename (default captures/plots/<map name>)")
    p.add_argument("--title", default=None)
    a = p.parse_args(argv)

    if a.sensor is not None and not 1 <= a.sensor <= 16:
        p.error("--sensor must be 1..16")

    faces, lo, hi = load_faces(a.map, a.step, a.sensor)
    if not faces:
        print("no faces found -- is this a shell (outer-surface) map?",
              file=sys.stderr)
        return 2

    stem = os.path.splitext(os.path.basename(a.map))[0]
    out = a.out or os.path.join(paths.captures_dir(), "plots", stem)
    what = "16-sensor mean" if a.sensor is None else f"sensor S{a.sensor}"
    title = a.title or "Hall probe surface map"
    sub = (f"{hi[0]-lo[0]:.0f} × {hi[1]-lo[1]:.0f} × {hi[2]-lo[2]:.0f} mm box "
           f"from rig ({lo[0]:.0f}, {lo[1]:.0f}, {lo[2]:.0f}) · "
           f"{a.step:.0f} mm grid · {what} · {stem}")
    cbar = f"|B|  (mT){'' if a.sensor is None else f'  — S{a.sensor}'}"

    n = draw_net(faces, lo, hi, a.step, title, sub, cbar, out + "_net.png")
    c = draw_cube(faces, lo, hi, title, sub, cbar, out + "_cube.png",
                  step_mm=a.step)
    for f in (n, c):
        print(f"wrote {f}  ({os.path.getsize(f)/1e6:.1f} MB)")
    print(f"faces plotted: {', '.join(sorted(faces))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
