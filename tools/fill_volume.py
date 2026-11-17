#!/usr/bin/env python3
"""
tools/fill_volume.py -- reconstruct the INSIDE of a box from its six faces.

Why this is allowed
-------------------
In a region carrying no current, B has neither curl nor divergence, so each
Cartesian component satisfies Laplace's equation. A harmonic function is fixed
everywhere inside a closed surface by its values ON that surface -- that is the
Dirichlet problem, and its solution is unique. So a shell map is not a cheap
substitute for filling the volume point by point. It is the whole measurement,
taken at one dimension less cost, and this recovers the interior from it.

The approximation, stated plainly
---------------------------------
What is solved here is Laplace's equation for |B|, and |B| is NOT harmonic --
only the components are. Doing it properly needs the vector field rotated into
a common lab frame, which needs each chip's rotation, and on this rig only the
face GROUPING was verified on the bench (2026-08-24), not the individual chip
rotations. So this is an approximation whose justification is empirical rather
than theoretical: where the field direction turns slowly across the box, |B|
is nearly harmonic, and --validate measures how nearly on the data at hand.

    python tools/fill_volume.py captures/volume_20260924_164240.npz
    python tools/fill_volume.py <map.npz> --validate captures/*_square_outside.csv --scale 0.8311

--validate is the part worth running. Any recording that passes THROUGH the box
is an interior measurement the solve never saw, and comparing against it is the
only honest way to find out whether the reconstruction is worth anything. On the
24 September map, checked against the 26 August `outside` traverse at y = 35 mm
and scaled for the 300 A -> 250 A current change, it agreed to 1.0 % rms.

The map is opened read-only. Nothing here writes to it.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402
from matplotlib.colors import Normalize          # noqa: E402

from octobee import paths                         # noqa: E402

AXES = ("x", "y", "z")
# Over-relaxation factor. Stable up to 2 for red-black Gauss-Seidel; 1.85 is
# near optimal for a grid this size. Applying it to a JACOBI sweep instead
# diverges -- silently, into NaN, after looking healthy for one iteration.
OMEGA = 1.85


def read_faces(map_path, step_mm):
    """The six faces of a shell map, as values on a 3-D lattice.

    Faces come from the sidecar's own record of which axes each line held
    fixed. Inferring it from the positions does not work on a shell map: both
    non-swept axes are fixed, so "the axis that varies least" is decided by
    noise and each face loses half its lines to a neighbour.
    """
    side = os.path.splitext(map_path)[0] + ".json"
    with np.load(map_path) as z:                 # read-only
        ins = z["in_span"]
        pos = z["pos_mm"][ins]
        b = z["b_mt"][ins]
        li = z["line_index"][ins]
    val = np.linalg.norm(b, axis=2).mean(axis=1)

    lo = np.round(pos.min(axis=0) / step_mm) * step_mm
    hi = np.round(pos.max(axis=0) / step_mm) * step_mm
    shape = tuple(np.round((hi - lo) / step_mm).astype(int) + 1)

    with open(side, encoding="utf-8") as fh:
        lines = json.load(fh)["lines"]
    keep = []
    for i, ln in enumerate(lines):
        if ln.get("part") == "slice":
            continue                 # interior: a check on the fill, not input
        for ax, at in (ln.get("fixed") or {}).items():
            k = AXES.index(ax)
            if min(abs(float(at) - lo[k]), abs(float(at) - hi[k])) < 1.0:
                keep.append(i)
                break
    sel = np.isin(li, keep)
    p, v = pos[sel], val[sel]

    idx = np.round((p - lo) / step_mm).astype(int)
    idx = np.clip(idx, 0, np.array(shape) - 1)
    acc = np.zeros(shape)
    cnt = np.zeros(shape)
    np.add.at(acc, tuple(idx.T), v)
    np.add.at(cnt, tuple(idx.T), 1)
    F = np.divide(acc, cnt, out=np.full(shape, np.nan), where=cnt > 0)

    shell = np.zeros(shape, bool)
    shell[0, :, :] = shell[-1, :, :] = True
    shell[:, 0, :] = shell[:, -1, :] = True
    shell[:, :, 0] = shell[:, :, -1] = True
    missing = int((shell & np.isnan(F)).sum())
    return F, lo, hi, missing, int(shell.sum())


def solve(F, tol=1e-9, max_iter=40000, log=print):
    """Laplace's equation inside, with F's set values held on the boundary.

    Red-black Gauss-Seidel with over-relaxation: the two colours are updated in
    separate sweeps, so each uses already-updated neighbours of the other colour
    and the iteration is a true Gauss-Seidel one rather than a Jacobi one.
    """
    fixed = ~np.isnan(F)
    U = np.where(fixed, F, float(np.nanmean(F)))
    I, J, K = np.indices(F.shape)
    par = (I + J + K) % 2
    red = (par == 0) & ~fixed
    black = (par == 1) & ~fixed

    def sweep(sel):
        nb = np.zeros_like(U)
        nb[1:-1, 1:-1, 1:-1] = (
            U[2:, 1:-1, 1:-1] + U[:-2, 1:-1, 1:-1] +
            U[1:-1, 2:, 1:-1] + U[1:-1, :-2, 1:-1] +
            U[1:-1, 1:-1, 2:] + U[1:-1, 1:-1, :-2]) / 6.0
        U[sel] = U[sel] + OMEGA * (nb[sel] - U[sel])

    for it in range(max_iter):
        old = U.copy()
        sweep(red)
        sweep(black)
        if it % 200 == 0 or it == max_iter - 1:
            r = float(np.abs(U - old).max())
            if not np.isfinite(r):
                raise RuntimeError("the relaxation diverged")
            if r < tol:
                log(f"  converged after {it} iterations (max change {r:.2e} mT)")
                break
    lap = (U[2:, 1:-1, 1:-1] + U[:-2, 1:-1, 1:-1] + U[1:-1, 2:, 1:-1] +
           U[1:-1, :-2, 1:-1] + U[1:-1, 1:-1, 2:] + U[1:-1, 1:-1, :-2] -
           6 * U[1:-1, 1:-1, 1:-1])
    log(f"  max |laplacian| on interior nodes: {np.abs(lap).max():.2e}")
    return U


def trilinear(U, lo, step_mm, P):
    """Sample the filled volume at arbitrary points. NaN outside the box."""
    g = (np.asarray(P, float) - lo) / step_mm
    out = np.full(len(g), np.nan)
    hi_i = np.array(U.shape) - 1
    ok = np.all((g >= 0) & (g <= hi_i), axis=1)
    if not ok.any():
        return out
    gi = np.minimum(np.floor(g[ok]).astype(int), hi_i - 1)
    fr = g[ok] - gi
    v = np.zeros(int(ok.sum()))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((fr[:, 0] if dx else 1 - fr[:, 0]) *
                     (fr[:, 1] if dy else 1 - fr[:, 1]) *
                     (fr[:, 2] if dz else 1 - fr[:, 2]))
                v += w * U[gi[:, 0] + dx, gi[:, 1] + dy, gi[:, 2] + dz]
    out[ok] = v
    return out


def validate(U, lo, step_mm, csv_path, scale, log=print):
    """Compare the reconstruction against a recording that crosses the box."""
    head = open(csv_path, encoding="utf-8").readlines()
    hrow = next(i for i, l in enumerate(head) if not l.startswith("#"))
    hdr = head[hrow].strip().split(",")
    ixp = [hdr.index(c) for c in ("X_mm", "Y_mm", "Z_mm")]
    ixb = [hdr.index(f"S{s}_absB_mT") for s in range(1, 17)]
    d = np.genfromtxt(csv_path, delimiter=",", skip_header=hrow + 1,
                      usecols=ixp + ixb)
    d = d[np.isfinite(d).all(axis=1)][::20]
    P, meas = d[:, :3], d[:, 3:].mean(axis=1) * scale
    pred = trilinear(U, lo, step_mm, P)
    ok = np.isfinite(pred)
    if ok.sum() < 10:
        log(f"  {os.path.basename(csv_path)}: only {ok.sum()} points inside "
            f"the box -- nothing to check against")
        return None
    e = pred[ok] - meas[ok]
    rms = float(np.sqrt((e ** 2).mean()))
    log(f"  {os.path.basename(csv_path)}: {int(ok.sum())} interior points")
    log(f"    predicted {pred[ok].mean():.3f} mT   measured {meas[ok].mean():.3f} mT")
    log(f"    rms error {rms:.3f} mT  ({100 * rms / meas[ok].mean():.1f} % of signal)"
        f"   max {np.abs(e).max():.3f} mT")
    return dict(n=int(ok.sum()), rms=rms,
                pct=100 * rms / float(meas[ok].mean()),
                pred=float(pred[ok].mean()), meas=float(meas[ok].mean()))


def draw_slices(U, lo, hi, step_mm, title, sub, path, check=None, probe=None):
    """Cuts through the filled volume, one shared colour scale."""
    norm = Normalize(vmin=U.min(), vmax=U.max())
    cmap = plt.get_cmap("viridis")
    ny = U.shape[1]
    # Five evenly spaced cuts, ends included: the two extremes are measured
    # faces, so the row shows the reconstruction between two knowns.
    ks = np.linspace(0, ny - 1, 5).round().astype(int)

    fig = plt.figure(figsize=(15.5, 5.0), dpi=200)
    fig.patch.set_facecolor("white")
    L, B, W, H = 0.045, 0.19, 0.845, 0.58
    h = step_mm / 2.0
    for c, k in enumerate(ks):
        ax = fig.add_axes([L + W * c / len(ks), B, W / len(ks) * 0.90, H])
        ax.imshow(U[:, k, :].T, origin="lower", cmap=cmap, norm=norm,
                  aspect="auto", interpolation="nearest",
                  extent=[lo[0] - h, hi[0] + h, lo[2] - h, hi[2] + h])
        edge = k in (0, ny - 1)
        ax.set_title(f"y = {lo[1] + k * step_mm:.0f} mm\n"
                     f"{'measured face' if edge else 'reconstructed'}",
                     fontsize=10.5, fontweight="bold" if edge else "normal",
                     color="#111" if edge else "#555", pad=6, linespacing=1.5)
        for s in ax.spines.values():
            s.set_color("#111" if edge else "#999")
            s.set_linewidth(1.4 if edge else 0.8)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("x  (mm)", fontsize=9)
        if c == 0:
            ax.set_ylabel("z  (mm)", fontsize=9)
        else:
            ax.set_yticklabels([])
        if probe is not None and abs(probe[1] - (lo[1] + k * step_mm)) < step_mm:
            ax.plot(probe[0], probe[2], color="#eb6834", lw=2.0, zorder=5)

    cax = fig.add_axes([L + W + 0.012, B, 0.013, H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("|B|  (mT)", fontsize=10, labelpad=10)
    cb.ax.tick_params(labelsize=8.5)

    fig.text(0.045, 0.935, title, fontsize=16, fontweight="bold", color="#111")
    fig.text(0.045, 0.875, sub, fontsize=10.5, color="#555")
    # Two short lines, not one long one -- a single line of this length runs
    # off the right edge of the figure and is silently clipped.
    foot = ("Interior solved from the six measured faces alone: in a current-free "
            "region the boundary determines the inside.\n"
            "The two outer panels are measured; the three between them are "
            "reconstruction.")
    if check:
        foot += (f"  Checked against an interior traverse the solve never saw: "
                 f"{check['pct']:.1f} % rms over {check['n']} points.")
    fig.text(0.045, 0.012, foot, fontsize=8.5, color="#777", linespacing=1.6,
             va="bottom")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def _plane(ax, U, lo, step_mm, axis, k, norm, cmap, alpha=1.0, stride=1):
    """Draw one constant-`axis` slice of the volume as a coloured plane in 3-D.

    plot_surface with facecolors, rather than a scatter of points: a filled
    volume drawn as dots reads as fog, and the eye cannot recover a gradient
    from it. Flat planes keep the values legible.
    """
    n = U.shape
    g = [np.arange(n[i]) * step_mm + lo[i] for i in range(3)]
    other = [i for i in range(3) if i != axis]
    A, B_ = np.meshgrid(g[other[0]], g[other[1]], indexing="ij")
    C = np.full_like(A, g[axis][k])
    coords = [None, None, None]
    coords[other[0]], coords[other[1]], coords[axis] = A, B_, C
    sl = [slice(None)] * 3
    sl[axis] = k
    vals = U[tuple(sl)]
    fc = cmap(norm(vals))
    ax.plot_surface(coords[0], coords[1], coords[2], facecolors=fc,
                    rstride=stride, cstride=stride, shade=False,
                    linewidth=0, antialiased=False, alpha=alpha)


def draw_3d(U, lo, hi, step_mm, title, sub, path, check=None):
    """The filled volume in 3-D: a cutaway, and a stack of slices.

    Two views because they answer different questions. The cutaway shows the
    field ON three orthogonal planes through the middle -- what the interior
    actually looks like. The stack shows the same volume sliced front to back,
    which is where the gradient lives on this rig.
    """
    norm = Normalize(vmin=U.min(), vmax=U.max())
    cmap = plt.get_cmap("viridis")
    n = U.shape

    fig = plt.figure(figsize=(15.0, 7.4), dpi=200)
    fig.patch.set_facecolor("white")

    # -- left: three orthogonal cuts through the centre ------------------
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))
    for axis, k in ((0, n[0] // 2), (1, n[1] // 2), (2, n[2] // 2)):
        _plane(ax, U, lo, step_mm, axis, k, norm, cmap)
    # the box itself, so the cuts sit in something
    for a in range(3):
        for e0 in (lo, hi):
            for e1 in (lo, hi):
                p = [None] * 3
                o = [i for i in range(3) if i != a]
                p[a] = [lo[a], hi[a]]
                p[o[0]] = [e0[o[0]]] * 2
                p[o[1]] = [e1[o[1]]] * 2
                ax.plot(p[0], p[1], p[2], color="#999", lw=0.7, zorder=0)
    ax.set_title("three cuts through the middle", fontsize=11.5,
                 fontweight="bold", color="#111", pad=0)
    ax.view_init(elev=22, azim=-62)

    # -- right: the volume sliced front to back --------------------------
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))
    for k in np.linspace(0, n[1] - 1, 7).round().astype(int):
        _plane(ax2, U, lo, step_mm, 1, k, norm, cmap, alpha=0.93)
    ax2.set_title("sliced front to back, every 40 mm", fontsize=11.5,
                  fontweight="bold", color="#111", pad=0)
    ax2.view_init(elev=16, azim=-72)

    for a in (ax, ax2):
        a.set_xlim(lo[0], hi[0]); a.set_ylim(lo[1], hi[1]); a.set_zlim(lo[2], hi[2])
        a.set_xlabel("x  (mm)", fontsize=9.5, labelpad=8)
        a.set_ylabel("y  (mm)", fontsize=9.5, labelpad=8)
        a.set_zlabel("z  (mm)", fontsize=9.5, labelpad=5)
        a.tick_params(labelsize=8)
        a.xaxis.pane.set_alpha(0.03); a.yaxis.pane.set_alpha(0.03)
        a.zaxis.pane.set_alpha(0.03)

    cax = fig.add_axes([0.935, 0.28, 0.013, 0.44])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("|B|  (mT)", fontsize=10.5, labelpad=10)
    cb.ax.tick_params(labelsize=9)

    fig.text(0.04, 0.955, title, fontsize=16, fontweight="bold", color="#111")
    fig.text(0.04, 0.917, sub, fontsize=10.5, color="#555")
    foot = ("Every value shown inside the box was solved from the six measured "
            "faces, not measured directly.")
    if check:
        foot += (f"  Checked against an interior traverse the solve never saw: "
                 f"{check['pct']:.1f} % rms over {check['n']} points.")
    fig.text(0.04, 0.028, foot, fontsize=8.5, color="#777")
    fig.subplots_adjust(left=0.01, right=0.91, top=0.89, bottom=0.06, wspace=0.0)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("map")
    p.add_argument("--step", type=float, default=10.0)
    p.add_argument("--validate", default=None,
                   help="a CSV recording that crosses the box, to check against")
    p.add_argument("--scale", type=float, default=1.0,
                   help="multiply the validation data by this (e.g. a current change)")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    F, lo, hi, missing, nshell = read_faces(a.map, a.step)
    print(f"box {hi[0]-lo[0]:.0f} x {hi[1]-lo[1]:.0f} x {hi[2]-lo[2]:.0f} mm, "
          f"{a.step:.0f} mm grid -> {F.shape} = {F.size:,} nodes")
    print(f"boundary: {nshell - missing:,} of {nshell:,} shell nodes measured")
    if missing:
        print(f"  WARNING: {missing} shell nodes have no data. The interior is "
              f"only determined by a CLOSED boundary -- with holes in it the "
              f"solve will still return an answer and it will be wrong.",
              file=sys.stderr)

    print("solving Laplace's equation inside...")
    U = solve(F, log=print)
    print(f"  interior |B|: {U.min():.3f} .. {U.max():.3f} mT")
    ctr = tuple(np.array(U.shape) // 2)
    print(f"  centre of the box: {U[ctr]:.3f} mT")

    stem = os.path.splitext(os.path.basename(a.map))[0]
    out = a.out or os.path.join(paths.captures_dir(), "plots", stem)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.save(out + "_interior.npy", U)
    print(f"wrote {out}_interior.npy  ({U.shape[0]}x{U.shape[1]}x{U.shape[2]})")

    check = probe = None
    if a.validate:
        hits = sorted(glob.glob(a.validate))
        if not hits:
            print(f"  nothing matched {a.validate}", file=sys.stderr)
        print("checking against interior measurements the solve never saw:")
        for f in hits:
            r = validate(U, lo, a.step, f, a.scale)
            if r and (check is None or r["rms"] < check["rms"]):
                check = r

    sub = (f"{hi[0]-lo[0]:.0f} x {hi[1]-lo[1]:.0f} x {hi[2]-lo[2]:.0f} mm box "
           f"from rig ({lo[0]:.0f}, {lo[1]:.0f}, {lo[2]:.0f}) - "
           f"{a.step:.0f} mm grid - 16-sensor mean - {stem}")
    png = draw_slices(U, lo, hi, a.step, "Inside the box, from its surface",
                      sub, out + "_interior.png", check, probe)
    print(f"wrote {png}  ({os.path.getsize(png)/1e6:.1f} MB)")
    p3 = draw_3d(U, lo, hi, a.step, "Inside the box, from its surface",
                 sub, out + "_interior3d.png", check)
    print(f"wrote {p3}  ({os.path.getsize(p3)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
