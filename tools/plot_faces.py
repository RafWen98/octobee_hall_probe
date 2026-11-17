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
    python tools/plot_faces.py <map.npz> --exclude S9,S10,S11
    python tools/plot_faces.py <map.npz> --step 10 --out captures/plots/mymap

Which box, and which sensors
----------------------------
The box is read from the sidecar's own lines -- the extremes of every axis over
the lines that make up the box -- so any volume unfolds, at any origin, and the
panel titles carry the real coordinates. Interior x-z planes ("part": "slice")
are not faces and are left out; only the planes at the box's extremes count.
Without a sidecar the box falls back to the positions' extent, snapped to
--step, which is a guess and says so.

The colour is |B| averaged over the sensors that are working. A dead or railed
chip pulls a 16-sensor mean to a number that describes nothing, so the mean
leaves out the sensors named by --exclude, any the sidecar records as excluded,
and any whose readings fall outside the range they were set to (a warning names
them). --keep-all turns the automatic part off. The subtitle always says how
many sensors were averaged and which were left out.

The map is opened read-only. Nothing here writes to it.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

# Agg: this is a file-writing tool and may run with no display attached.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.transforms import Affine2D

from octobee import paths

AXES = ("x", "y", "z")
N_SENSORS = 16

# The recorder's default full scale, used when a sidecar does not say.
DEFAULT_RANGE_MT = 20.0
# A reading this far past full scale is not a field. The margin leaves room for
# the gain trim, which can nudge a genuine near-full-scale value a little over.
RANGE_MARGIN = 1.05
# A sensor is called implausible when more than this share of its rows have an
# axis beyond the range. A real field excursion is brief and local; a fault is
# there on every row, so a tenth is far below a fault and far above a transient.
IMPLAUSIBLE_FRACTION = 0.10


# --------------------------------------------------------------------------
# Which sensors to average
# --------------------------------------------------------------------------

def parse_sensors(text):
    """'S9,s10, 11' -> {9, 10, 11}. ValueError names the token that is wrong."""
    out = set()
    for raw in str(text or "").split(","):
        tok = raw.strip()
        if not tok:
            continue
        m = re.fullmatch(r"[Ss]?(\d+)", tok)
        if m is None or not 1 <= int(m.group(1)) <= N_SENSORS:
            raise ValueError(f"bad sensor {tok!r}: expected S1..S{N_SENSORS} "
                             f"or 1..{N_SENSORS}")
        out.add(int(m.group(1)))
    return out


def recorded_dead(side):
    """Sensors a sidecar records as excluded or dead -> set of 1-16.

    The recorder writes "excluded" into a CSV header (octobee.record) from the
    calibration's `dead` set; a volume sidecar does not carry it today, so this
    usually finds nothing -- but a map that does record it should be believed.
    Entries are "S9" or a single channel, "S11.Bx". A channel entry drops the
    whole sensor: the mean is over |B|, which one bad axis spoils.
    """
    found = set()
    scopes = [side] + [side.get(k) for k in ("calibration", "meta")
                       if isinstance(side.get(k), dict)]
    for scope in scopes:
        for key in ("excluded", "dead"):
            for item in scope.get(key) or ():
                m = re.fullmatch(r"[Ss]?(\d+)(?:\..*)?", str(item).strip())
                if m and 1 <= int(m.group(1)) <= N_SENSORS:
                    found.add(int(m.group(1)))
    return found


def recorded_ranges(side):
    """Per-sensor full scale in mT: the sidecar's, else the 20 mT default."""
    for scope in (side, side.get("calibration") or {}):
        r = scope.get("ranges_mt") if isinstance(scope, dict) else None
        if r is not None and len(r) == N_SENSORS:
            return np.asarray(r, float)
    return np.full(N_SENSORS, DEFAULT_RANGE_MT)


def implausible_sensors(b, ranges_mt):
    """{sensor 1-16: share of rows with an axis beyond its range}, over the bad.

    `b` is (rows, 16, 3) in mT. Looks at a stride of the rows: a fault shows on
    all of them, and the full 600 000-row array is not needed to see that.
    """
    b = np.asarray(b)
    b = b[::max(1, len(b) // 20000)]
    over = np.abs(b) > RANGE_MARGIN * np.asarray(ranges_mt)[None, :, None]
    share = over.any(axis=2).mean(axis=0)
    return {s + 1: float(share[s]) for s in range(N_SENSORS)
            if share[s] > IMPLAUSIBLE_FRACTION}


class SensorChoice:
    """Which of the 16 sensors a mean is taken over, and why the rest are out."""

    def __init__(self, reasons):
        # {sensor 1-16: why it is excluded}
        self.reasons = dict(sorted(reasons.items()))
        self.alive = np.array([s + 1 not in self.reasons
                               for s in range(N_SENSORS)])
        self.excluded = sorted(self.reasons)

    @property
    def n(self):
        return int(self.alive.sum())

    @property
    def text(self):
        """'13-sensor mean, S9 S10 S11 excluded' -- never '16' when it is not."""
        out = f"{self.n}-sensor mean"
        if self.excluded:
            out += ", " + " ".join(f"S{s}" for s in self.excluded) + " excluded"
        return out


def _warn(msg):
    print(msg, file=sys.stderr)


def choose_sensors(b, side, exclude=(), keep_all=False, log=_warn):
    """Decide which sensors to average. See the module docstring for the rules.

    `exclude` always applies. Unless `keep_all`, sensors the sidecar records as
    excluded, and sensors reading outside their range, are dropped too; a
    detected fault is announced by name either way, so keeping one is a choice
    made with the warning in view.
    """
    reasons = {int(s): "--exclude" for s in exclude}
    for s in sorted(recorded_dead(side)):
        if keep_all:
            log(f"  note: the sidecar records S{s} as excluded; --keep-all "
                f"averages it anyway")
        else:
            reasons.setdefault(s, "recorded as excluded in the sidecar")
    bad = implausible_sensors(b, recorded_ranges(side))
    for s, share in bad.items():
        rng = recorded_ranges(side)[s - 1]
        log(f"  WARNING: S{s} reads beyond its +/-{rng:g} mT range on "
            f"{100 * share:.0f} % of rows -- a faulty channel, not a field."
            + ("  Kept because of --keep-all." if keep_all
               else "  Excluded; --keep-all keeps it."))
        if not keep_all:
            reasons.setdefault(s, "implausible |B|")
    if len(reasons) >= N_SENSORS:
        raise ValueError("every sensor is excluded -- nothing left to average")
    return SensorChoice(reasons)


def magnitude(b, choice, sensor=None):
    """|B| per row in mT: one sensor, or the mean over the chosen ones."""
    if sensor is not None:
        return np.linalg.norm(b[:, sensor - 1], axis=1)
    return np.linalg.norm(b[:, choice.alive], axis=2).mean(axis=1)


def add_sensor_args(p):
    """The --exclude / --keep-all pair, identical in every tool."""
    p.add_argument("--exclude", default="", metavar="S9,S10",
                   help="sensors to leave out of the mean, e.g. S9,S10,S11 "
                        "(case-insensitive, numbers 1-16)")
    p.add_argument("--keep-all", action="store_true",
                   help="do not auto-exclude sensors the sidecar records as "
                        "dead or that read beyond their range (--exclude "
                        "still applies)")


def exclude_from_args(p, a):
    """--exclude as a set, with a clean usage error on a bad one."""
    try:
        return parse_sensors(a.exclude)
    except ValueError as e:
        p.error(str(e))
    return set()


# --------------------------------------------------------------------------
# Reading a map
# --------------------------------------------------------------------------

def read_sidecar(path):
    """The map's .json sidecar as a dict, or {} when there is none."""
    side = os.path.splitext(path)[0] + ".json"
    if not os.path.exists(side):
        return {}
    with open(side, encoding="utf-8") as fh:
        return json.load(fh)


def box_from_lines(lines):
    """(lo, hi) of the box from its lines, or None if they do not give one.

    Every box line records its swept span and the values it held the other two
    axes at, so the extremes over them are the box -- exact, whatever the
    origin. Slice lines are skipped: they sit inside it.
    """
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for ln in lines or ():
        if ln.get("part") == "slice":
            continue
        k = AXES.index(ln["sweep"])
        for v in (ln["start_mm"], ln["stop_mm"]):
            lo[k], hi[k] = min(lo[k], float(v)), max(hi[k], float(v))
        for ax, at in (ln.get("fixed") or {}).items():
            if ax in AXES:
                k = AXES.index(ax)
                lo[k], hi[k] = min(lo[k], float(at)), max(hi[k], float(at))
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        return None
    return lo, hi


def face_lines(lines, lo, hi, tol=1.0):
    """Which face(s) each swept line lies on, from the sidecar's own record.

    Returns {(axis index, 0 for the low end / 1 for the high end): [line
    index, ...]}.

    Every line records the TWO axes it held fixed, e.g. {"y": 0, "z": 120}. A
    line belongs to a face for each fixed axis that sits at an end of the box --
    so most lines belong to one, and a line along a box EDGE, with both fixed
    axes at an extreme, belongs to two. Assigning such a line to one face and
    not the other leaves a missing row along every seam.

    Do NOT infer this from the positions. On a shell map both non-swept axes are
    fixed, so "the axis that varies least" is decided by noise, and half the
    lines of each face get filed under a neighbouring one -- which looks like a
    face that is mostly empty with a few stripes across it.

    Slice lines are skipped: an interior x-z plane is not a face.
    """
    faces = {}
    for i, ln in enumerate(lines):
        if ln.get("part") == "slice":
            continue
        for ax, at in (ln.get("fixed") or {}).items():
            if ax not in AXES:
                continue
            k = AXES.index(ax)
            if hi[k] - lo[k] <= 0:
                continue                 # a flat axis has no two ends
            for end, v in enumerate((lo[k], hi[k])):
                if abs(float(at) - v) < tol:
                    faces.setdefault((k, end), []).append(i)
                    break
    return faces


def geometry_faces(pos, li, lo, hi):
    """The no-sidecar fallback: infer each line's face from its positions."""
    faces = {}
    for line in np.unique(li):
        p = pos[li == line]
        i = int(np.argmin(p.max(axis=0) - p.min(axis=0)))
        at = p[:, i].mean()
        for end, v in enumerate((lo[i], hi[i])):
            if abs(at - v) < 2.0:
                faces.setdefault((i, end), []).append(int(line))
                break
    return faces


def read_map(path):
    """(pos, b, line_index, sidecar) for the rows inside their line's span."""
    with np.load(path) as z:                       # read-only; never written
        ins = z["in_span"]
        pos = z["pos_mm"][ins]
        b = z["b_mt"][ins]
        li = z["line_index"][ins]
    return pos, b, li, read_sidecar(path)


# --------------------------------------------------------------------------
# The net
# --------------------------------------------------------------------------

# Where each face sits in the unfolded net, as (column, row) on a 4x3 lattice,
# with the flags needed to make the shared edges actually line up. A face is
# named by its ROLE -- which end of which axis -- and not by its coordinate, so
# the same layout serves a box anywhere: the coordinate only goes in the title.
#
#              +--------+
#              |  z hi  |
#     +--------+--------+--------+--------+
#     |  x lo  |  y lo  |  x hi  |  y hi  |
#     +--------+--------+--------+--------+
#              |  z lo  |
#              +--------+
#
# `flip_u` mirrors the horizontal axis. Two faces need it: the x-low face is
# folded out to the LEFT, so its y runs away from the viewer rather than toward
# them, and the y-high face is the far wall seen from behind, so its x is
# reversed relative to y low. Getting either wrong produces a net that looks
# plausible and is mirrored -- exactly the failure this layout exists to
# expose, so it is worth stating rather than discovering.
#
# Keyed by (axis index, end), end 0 = low and 1 = high.
ROLES = {
    (2, 1): {"col": 1, "row": 0, "u": "x", "v": "y", "flip_u": False,
             "flip_v": False, "label": "ceiling"},
    (0, 0): {"col": 0, "row": 1, "u": "y", "v": "z", "flip_u": True,
             "flip_v": False, "label": "left end"},
    (1, 0): {"col": 1, "row": 1, "u": "x", "v": "z", "flip_u": False,
             "flip_v": False, "label": "front wall"},
    (0, 1): {"col": 2, "row": 1, "u": "y", "v": "z", "flip_u": False,
             "flip_v": False, "label": "right end"},
    (1, 1): {"col": 3, "row": 1, "u": "x", "v": "z", "flip_u": True,
             "flip_v": False, "label": "back wall"},
    (2, 0): {"col": 1, "row": 2, "u": "x", "v": "y", "flip_u": False,
             "flip_v": True, "label": "floor"},
}


def face_spec(k, end, lo, hi):
    """One face's net placement, with its id and title from the real box."""
    at = float((lo, hi)[end][k]) + 0.0           # + 0.0: no "-0" in a title
    ax = AXES[k]
    return {**ROLES[(k, end)], "axis": ax, "end": end, "at": at,
            "id": f"{ax}{at:g}", "title": f"{ax} = {at:g}"}


def bin_face(p, v, spec, lo, hi, step_mm):
    """Average the samples `v` at positions `p` onto one face's grid.

    Cells count from the box's own low corner, so an origin that is not a
    multiple of the step still lands each sample in its own cell.
    """
    iu, iv = AXES.index(spec["u"]), AXES.index(spec["v"])
    ue = lo[iu] + step_mm * np.arange(int(round((hi[iu] - lo[iu]) / step_mm)) + 1)
    ve = lo[iv] + step_mm * np.arange(int(round((hi[iv] - lo[iv]) / step_mm)) + 1)
    ui = np.clip(np.round((p[:, iu] - ue[0]) / step_mm).astype(int), 0, len(ue) - 1)
    vi = np.clip(np.round((p[:, iv] - ve[0]) / step_mm).astype(int), 0, len(ve) - 1)
    acc = np.zeros((len(ve), len(ue)))
    cnt = np.zeros((len(ve), len(ue)))
    np.add.at(acc, (vi, ui), v)
    np.add.at(cnt, (vi, ui), 1)
    grid = np.divide(acc, cnt, out=np.full_like(acc, np.nan), where=cnt > 0)
    return {**spec, "ue": ue, "ve": ve, "grid": grid}


def load_faces(path, step_mm, sensor=None, exclude=(), keep_all=False,
               log=_warn):
    """Read a volume map and bin each face onto the grid.

    Returns ({face id: face dict}, lo, hi, choice). A face dict holds its net
    placement (see `face_spec`) plus `ue`, `ve` (cell centres along its two
    in-plane axes) and `grid`. `choice` is the SensorChoice the mean was
    taken over. `sensor` is 1-16 for one chip, or None for the mean over the
    working ones -- the mean smears the field over the ~160 mm the chips are
    strung along, which is honest but blunt, so both are offered.
    """
    pos, b, li, side = read_map(path)
    choice = choose_sensors(b, side, exclude, keep_all, log)
    lines = side.get("lines")
    box = box_from_lines(lines)
    if box is not None:
        lo, hi = box
        fl = face_lines(lines, lo, hi)
    else:
        log("  note: no usable sidecar -- falling back to inferring the box "
            "and its faces from positions, which is unreliable on a shell map")
        # Snap to the grid: the ends of a line carry a millimetre of run-up,
        # and a box whose corner is at 69.98 rather than 70 puts every face
        # half a cell out.
        lo = np.round(pos.min(axis=0) / step_mm) * step_mm
        hi = np.round(pos.max(axis=0) / step_mm) * step_mm
        fl = geometry_faces(pos, li, lo, hi)

    out = {}
    for (k, end), idx in sorted(fl.items()):
        spec = face_spec(k, end, lo, hi)
        sel = np.isin(li, idx)
        v = magnitude(b[sel], choice, sensor)
        out[spec["id"]] = bin_face(pos[sel], v, spec, lo, hi, step_mm)
    return out, lo, hi, choice


def draw_net(faces, lo, hi, step_mm, title, sub, cbar_label, path):
    """The unfolded box, one shared colour scale."""
    allv = np.concatenate([f["grid"].ravel() for f in faces.values()])
    allv = allv[np.isfinite(allv)]
    norm = Normalize(vmin=allv.min(), vmax=allv.max())
    cmap = plt.get_cmap("viridis")

    span = hi - lo                                 # x, y, z extents in mm
    # Column widths and row heights follow the real geometry, so the net is to
    # scale rather than six equal boxes.
    colw = [span[1], span[0], span[1], span[0]]    # x lo | y lo | x hi | y hi
    rowh = [span[1], span[2], span[1]]             # z270 / walls / z70

    fig_w, fig_h = 15.5, 9.2
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor("white")

    L, B, W, H = 0.055, 0.055, 0.80, 0.80
    xo = np.concatenate([[0], np.cumsum(colw)]) / sum(colw)
    yo = np.concatenate([[0], np.cumsum(rowh)]) / sum(rowh)
    GAP = 0.004

    for spec in faces.values():
        ue, ve, grid = spec["ue"], spec["ve"], spec["grid"]
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
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#333")
            s.set_linewidth(0.9)

        fin = grid[np.isfinite(grid)]
        # Title above the frame, range under it -- stacked, because both
        # centred on the same line is how the first version of this collided.
        ax.set_title(f"{spec['title']}  ·  {spec['label']}\n"
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
    allv = np.concatenate([f["grid"].ravel() for f in faces.values()])
    allv = allv[np.isfinite(allv)]
    norm = Normalize(vmin=allv.min(), vmax=allv.max())
    cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(14.5, 7.0), dpi=200)
    fig.patch.set_facecolor("white")

    VIEWS = [(38, 24, f"toward the x = {lo[0]:g} / y = {hi[1]:g} corner"),
             (218, 24, "the opposite corner")]

    for col, (az, el, label) in enumerate(VIEWS):
        ax = fig.add_axes([0.02 + 0.44 * col, 0.08, 0.42, 0.76])
        ax.set_aspect("equal")
        ax.axis("off")
        v = _view_dir(az, el)
        pts2 = []

        for spec in faces.values():
            grid = spec["grid"]
            iu, iv = AXES.index(spec["u"]), AXES.index(spec["v"])
            ifx = 3 - iu - iv
            at = spec["at"]
            # Outward normal: +1 at the high end of its axis, -1 at the low.
            sgn = 1.0 if spec["end"] == 1 else -1.0
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
            corner = np.zeros(3)
            corner[ifx], corner[iu], corner[iv] = at, lo[iu], lo[iv]
            Pu = corner.copy()
            Pu[iu] = hi[iu]
            Pv = corner.copy()
            Pv[iv] = hi[iv]
            o2, u2, v2 = _axono(np.vstack([corner, Pu, Pv]), az, el)
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
            d = np.zeros(3)
            d[k] = 1.0
            vec = _axono(d[None, :], az, el)[0] - base
            vec = vec / np.linalg.norm(vec) * 0.075
            ax.annotate("", xy=org + vec, xytext=org, xycoords="axes fraction",
                        textcoords="axes fraction", zorder=6,
                        arrowprops={"arrowstyle": "-|>", "color": "#c0392b",
                                    "lw": 1.3, "shrinkA": 0, "shrinkB": 0})
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
                   help="plot sensor N (1-16) instead of the mean over the working ones")
    p.add_argument("--out", default=None,
                   help="output basename (default captures/plots/<map name>)")
    p.add_argument("--title", default=None)
    add_sensor_args(p)
    a = p.parse_args(argv)

    if a.sensor is not None and not 1 <= a.sensor <= 16:
        p.error("--sensor must be 1..16")
    exclude = exclude_from_args(p, a)

    try:
        faces, lo, hi, choice = load_faces(a.map, a.step, a.sensor, exclude,
                                           a.keep_all)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not faces:
        print("no faces found -- is this a shell (outer-surface) map?",
              file=sys.stderr)
        return 2

    stem = os.path.splitext(os.path.basename(a.map))[0]
    out = a.out or os.path.join(paths.captures_dir(), "plots", stem)
    if a.sensor is None:
        what = choice.text
    else:
        what = f"sensor S{a.sensor}"
        if a.sensor in choice.reasons:
            print(f"  WARNING: S{a.sensor} is flagged "
                  f"({choice.reasons[a.sensor]}); plotting it anyway",
                  file=sys.stderr)
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
    print(f"colour: {what}")
    if len(faces) < 6:
        print(f"  note: only {len(faces)} of 6 faces have lines in this map",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
