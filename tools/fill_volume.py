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
    python tools/fill_volume.py <map.npz> --exclude S9,S10,S11

What goes in, and what is refused
---------------------------------
The box is read from the sidecar's lines (any origin, any size). Its faces are
the planes at the extremes; the shell is every grid node on them. Interior x-z
planes ("part": "slice") are interior MEASUREMENTS and never part of the
shell. By default they are held out of the solve and the result is compared
with them afterwards -- an independent check that costs nothing extra to take.
--slices use fixes them as data instead, which sharpens the interior but leaves
nothing to check it against.

The colour is |B| averaged over the working sensors: --exclude names ones to
drop, the sidecar's own record of excluded sensors and any sensor reading
beyond its range are dropped automatically (--keep-all stops that), and the
figures say how many sensors were averaged.

A shell node with no data is NOT an unknown to solve for: its neighbour sum is
zero outside the box, so left free it relaxes toward 0 mT and drags the
interior down with it. Missing shell nodes are filled from their measured
neighbours along the surface and counted; if more than --max-missing of the
shell is missing the solve is refused (--force overrides).

If the relaxation does not converge within --max-iter the result is reported
as unconverged on stderr and in the figure footer and the exit code is 1
(--force: reported but 0).

--validate is the part worth running. Any recording that passes THROUGH the box
is an interior measurement the solve never saw, and comparing against it is the
only honest way to find out whether the reconstruction is worth anything. Every
recording's rms is printed; the figure shows the worst and the median (all of
them when there are three or fewer), never just the best.

On the 24 September map the one matching recording is the 26 August `outside`
traverse at y = 35 mm, scaled by 0.8311 for the 300 A -> 250 A current change:
585 interior points, 1.0 % rms (0.060 mT) of the 16-sensor mean. That is a
single traverse, not a survey of the box -- the number says the reconstruction
is good along that line, and only there.

The map is opened read-only. Nothing here writes to it.
"""

import argparse
import glob
import os
import sys
from types import SimpleNamespace

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from octobee import paths

# The sensor selection and the box/face logic are shared with plot_faces.py,
# which sits next to this file. Put this directory first so the import works
# however the script was started.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_faces as pf

AXES = ("x", "y", "z")
# Over-relaxation factor. Stable up to 2 for red-black Gauss-Seidel; 1.85 is
# near optimal for a grid this size. Applying it to a JACOBI sweep instead
# diverges -- silently, into NaN, after looking healthy for one iteration.
OMEGA = 1.85
# The share of shell nodes that may be missing before the answer is refused.
# A few holes filled from their neighbours barely move the interior; a whole
# face-sized hole is a different measurement.
MAX_MISSING = 0.02


def _warn(msg):
    print(msg, file=sys.stderr)


def shell_mask(shape):
    """True on the outermost layer of nodes of a grid of this shape."""
    shell = np.zeros(shape, bool)
    shell[0, :, :] = shell[-1, :, :] = True
    shell[:, 0, :] = shell[:, -1, :] = True
    shell[:, :, 0] = shell[:, :, -1] = True
    return shell


def _node_values(pos, v, lo, step_mm, shape):
    """Average samples `v` at `pos` onto the lattice, NaN where none landed."""
    idx = np.clip(np.round((pos - lo) / step_mm).astype(int), 0,
                  np.array(shape) - 1)
    acc = np.zeros(shape)
    cnt = np.zeros(shape)
    np.add.at(acc, tuple(idx.T), v)
    np.add.at(cnt, tuple(idx.T), 1)
    return np.divide(acc, cnt, out=np.full(shape, np.nan), where=cnt > 0)


def read_faces(map_path, step_mm, exclude=(), keep_all=False, log=_warn):
    """The shell of a map as values on a 3-D lattice, plus any interior planes.

    Returns a namespace: `F` (shell values, NaN elsewhere), `S` (values at the
    interior nodes the slice planes measured, NaN elsewhere), `lo`, `hi`,
    `missing` and `nshell` (shell nodes with no data, and shell nodes in all),
    and `choice` (the SensorChoice the mean was taken over).

    Faces come from the sidecar's own record of which axes each line held
    fixed, and the box from the extremes of the same lines. Inferring either
    from the positions does not work on a shell map: both non-swept axes are
    fixed, so "the axis that varies least" is decided by noise and each face
    loses half its lines to a neighbour. Lines marked "slice" are interior
    planes: they are binned separately and never touch the shell.
    """
    pos, b, li, side = pf.read_map(map_path)
    lines = side.get("lines")
    box = pf.box_from_lines(lines)
    if box is None:
        raise ValueError("this map has no sidecar lines -- the box and its "
                         "faces cannot be told from positions on a shell map")
    lo, hi = box
    shape = tuple(np.round((hi - lo) / step_mm).astype(int) + 1)
    if min(shape) < 3:
        raise ValueError(f"a {shape} grid has no interior to fill")
    choice = pf.choose_sensors(b, side, exclude, keep_all, log)

    face_idx = sorted({i for v in pf.face_lines(lines, lo, hi).values()
                       for i in v})
    slice_idx = [i for i, ln in enumerate(lines) if ln.get("part") == "slice"]

    sel = np.isin(li, face_idx)
    F = _node_values(pos[sel], pf.magnitude(b[sel], choice), lo, step_mm, shape)
    shell = shell_mask(shape)
    # Every face line lies on the shell by construction; anything else would be
    # a line the sidecar called a face and the lattice disagrees with.
    F[~shell] = np.nan

    S = np.full(shape, np.nan)
    sel = np.isin(li, slice_idx)
    if sel.any():
        S = _node_values(pos[sel], pf.magnitude(b[sel], choice), lo, step_mm,
                         shape)
        S[shell] = np.nan                # a plane's rim is a face's job
    missing = int((shell & np.isnan(F)).sum())
    return SimpleNamespace(F=F, S=S, lo=lo, hi=hi, missing=missing,
                           nshell=int(shell.sum()), choice=choice)


def fill_missing_shell(F):
    """Fill shell nodes with no data from measured neighbours ON the shell.

    A shell node must carry a value: the solver holds it fixed, and a node
    left free there is solved as if the field outside the box were zero. The
    value comes from the mean of the already-known 6-neighbours that are
    themselves on the shell, so an interior node never feeds the boundary it is
    meant to be determined by. A hole wider than one node is filled inward
    from its rim, a layer per pass. Returns how many nodes were filled; F is
    changed in place. Raises ValueError if no measured node is reachable.
    """
    shell = shell_mask(F.shape)
    todo = shell & np.isnan(F)
    total = int(todo.sum())
    while todo.any():
        known = shell & ~np.isnan(F)
        vals = np.where(known, F, 0.0)
        kn = known.astype(float)
        acc = np.zeros(F.shape)
        cnt = np.zeros(F.shape)
        for ax in range(3):
            for d in (1, -1):
                src = [slice(None)] * 3
                dst = [slice(None)] * 3
                if d == 1:
                    src[ax], dst[ax] = slice(None, -1), slice(1, None)
                else:
                    src[ax], dst[ax] = slice(1, None), slice(None, -1)
                acc[tuple(dst)] += vals[tuple(src)]
                cnt[tuple(dst)] += kn[tuple(src)]
        now = todo & (cnt > 0)
        if not now.any():
            raise ValueError("no measured shell node to fill the gaps from")
        F[now] = acc[now] / cnt[now]
        todo = shell & np.isnan(F)
    return total


def solve(F, tol=1e-9, max_iter=40000, log=print):
    """Laplace's equation inside, with F's set values held fixed.

    Red-black Gauss-Seidel with over-relaxation: the two colours are updated in
    separate sweeps, so each uses already-updated neighbours of the other colour
    and the iteration is a true Gauss-Seidel one rather than a Jacobi one.

    Every shell node must be set. Returns (U, info) with info holding
    `converged`, `iterations` and `residual` (the largest change in the last
    sweep, mT) -- running out of iterations is a result, not a success.
    """
    fixed = ~np.isnan(F)
    if not fixed[shell_mask(F.shape)].all():
        raise ValueError("a shell node is unset; fill_missing_shell first -- "
                         "left free it would relax toward 0 mT")
    U = np.where(fixed, F, float(np.nanmean(F)))
    ii, jj, kk = np.indices(F.shape)
    par = (ii + jj + kk) % 2
    red = (par == 0) & ~fixed
    black = (par == 1) & ~fixed

    def sweep(sel):
        nb = np.zeros_like(U)
        nb[1:-1, 1:-1, 1:-1] = (
            U[2:, 1:-1, 1:-1] + U[:-2, 1:-1, 1:-1] +
            U[1:-1, 2:, 1:-1] + U[1:-1, :-2, 1:-1] +
            U[1:-1, 1:-1, 2:] + U[1:-1, 1:-1, :-2]) / 6.0
        U[sel] = U[sel] + OMEGA * (nb[sel] - U[sel])

    converged = False
    r = float("nan")
    it = 0
    for it in range(max_iter):
        old = U.copy()
        sweep(red)
        sweep(black)
        if it % 200 == 0 or it == max_iter - 1:
            r = float(np.abs(U - old).max())
            if not np.isfinite(r):
                raise RuntimeError("the relaxation diverged")
            if r < tol:
                converged = True
                log(f"  converged after {it} iterations (max change {r:.2e} mT)")
                break
    if not converged:
        log(f"  NOT CONVERGED after {max_iter} iterations "
            f"(max change {r:.2e} mT, wanted < {tol:.0e})")
    lap = (U[2:, 1:-1, 1:-1] + U[:-2, 1:-1, 1:-1] + U[1:-1, 2:, 1:-1] +
           U[1:-1, :-2, 1:-1] + U[1:-1, 1:-1, 2:] + U[1:-1, 1:-1, :-2] -
           6 * U[1:-1, 1:-1, 1:-1])
    free = ~fixed[1:-1, 1:-1, 1:-1]
    log(f"  max |laplacian| on free interior nodes: {np.abs(lap[free]).max():.2e}")
    return U, {"converged": converged, "iterations": it, "residual": r}


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


def validate(U, lo, step_mm, csv_path, scale, alive=None, log=print):
    """Compare the reconstruction against a recording that crosses the box.

    `alive` is the per-sensor mask the map's mean was taken with (see
    SensorChoice). The recording must be averaged over the same sensors, or
    the comparison would be between two different quantities.
    """
    with open(csv_path, encoding="utf-8") as fh:
        head = fh.readlines()
    hrow = next(i for i, ln in enumerate(head) if not ln.startswith("#"))
    hdr = head[hrow].strip().split(",")
    ixp = [hdr.index(c) for c in ("X_mm", "Y_mm", "Z_mm")]
    use = range(1, 17) if alive is None else [s + 1 for s in range(16) if alive[s]]
    ixb = [hdr.index(f"S{s}_absB_mT") for s in use]
    d = np.genfromtxt(csv_path, delimiter=",", skip_header=hrow + 1,
                      usecols=ixp + ixb)
    d = d[np.isfinite(d).all(axis=1)][::20]
    P, meas = d[:, :3], d[:, 3:].mean(axis=1) * scale
    pred = trilinear(U, lo, step_mm, P)
    ok = np.isfinite(pred)
    name = os.path.basename(csv_path)
    if ok.sum() < 10:
        log(f"  {name}: only {ok.sum()} points inside "
            f"the box -- nothing to check against")
        return None
    e = pred[ok] - meas[ok]
    rms = float(np.sqrt((e ** 2).mean()))
    log(f"  {name}: {int(ok.sum())} interior points")
    log(f"    predicted {pred[ok].mean():.3f} mT   measured {meas[ok].mean():.3f} mT")
    log(f"    rms error {rms:.3f} mT  ({100 * rms / meas[ok].mean():.1f} % of signal)"
        f"   max {np.abs(e).max():.3f} mT")
    return {"name": name, "n": int(ok.sum()), "rms": rms,
            "pct": 100 * rms / float(meas[ok].mean()),
            "pred": float(pred[ok].mean()), "meas": float(meas[ok].mean())}


def slice_check(U, S):
    """The solve against the interior-plane nodes it was NOT given, or None."""
    ok = np.isfinite(S)
    if ok.sum() < 10:
        return None
    e = U[ok] - S[ok]
    rms = float(np.sqrt((e ** 2).mean()))
    return {"name": "interior planes", "n": int(ok.sum()), "rms": rms,
            "pct": 100 * rms / float(S[ok].mean())}


def check_note(checks):
    """One footer sentence for a set of checks: ALL when few, else worst + median.

    Reporting only the best of several recordings is selecting the answer, so
    the summary leads with the worst and gives the median beside it.
    """
    if not checks:
        return ""
    head = "Checked against interior measurements the solve never saw: "
    if len(checks) <= 3:
        return head + "; ".join(
            f"{c['name']} {c['pct']:.1f} % rms over {c['n']} points"
            for c in checks) + "."
    ranked = sorted(checks, key=lambda c: c["pct"])
    worst = ranked[-1]
    med = float(np.median([c["pct"] for c in checks]))
    return (head + f"{len(checks)} recordings, worst {worst['pct']:.1f} % rms "
            f"({worst['name']}), median {med:.1f} %.")


def draw_slices(U, lo, hi, step_mm, title, sub, path, notes=(), probe=None):
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
    for note in notes:
        foot += "\n" + note
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


def draw_3d(U, lo, hi, step_mm, title, sub, path, notes=()):
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
    ax2.set_title(f"sliced front to back, every {(hi[1] - lo[1]) / 6:.0f} mm",
                  fontsize=11.5,
                  fontweight="bold", color="#111", pad=0)
    ax2.view_init(elev=16, azim=-72)

    for a in (ax, ax2):
        a.set_xlim(lo[0], hi[0])
        a.set_ylim(lo[1], hi[1])
        a.set_zlim(lo[2], hi[2])
        a.set_xlabel("x  (mm)", fontsize=9.5, labelpad=8)
        a.set_ylabel("y  (mm)", fontsize=9.5, labelpad=8)
        a.set_zlabel("z  (mm)", fontsize=9.5, labelpad=5)
        a.tick_params(labelsize=8)
        a.xaxis.pane.set_alpha(0.03)
        a.yaxis.pane.set_alpha(0.03)
        a.zaxis.pane.set_alpha(0.03)

    cax = fig.add_axes([0.935, 0.28, 0.013, 0.44])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("|B|  (mT)", fontsize=10.5, labelpad=10)
    cb.ax.tick_params(labelsize=9)

    fig.text(0.04, 0.955, title, fontsize=16, fontweight="bold", color="#111")
    fig.text(0.04, 0.917, sub, fontsize=10.5, color="#555")
    foot = ("Every value shown inside the box was solved from the six measured "
            "faces, not measured directly.")
    for note in notes:
        foot += "\n" + note
    fig.text(0.04, 0.012, foot, fontsize=8.5, color="#777", linespacing=1.6,
             va="bottom")
    fig.subplots_adjust(left=0.01, right=0.91, top=0.89, bottom=0.10, wspace=0.0)

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
                   help="CSV recording(s) that cross the box, to check against "
                        "(a glob)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="multiply the validation data by this (e.g. a current change)")
    p.add_argument("--slices", choices=("check", "use"), default="check",
                   help="interior planes in the map: hold out and check the "
                        "solve against them (default), or fix them as data")
    p.add_argument("--max-missing", type=float, default=MAX_MISSING,
                   help="refuse to solve if more than this share of the shell "
                        f"has no data (default {MAX_MISSING})")
    p.add_argument("--max-iter", type=int, default=40000)
    p.add_argument("--force", action="store_true",
                   help="solve despite a mostly-missing shell, and exit 0 "
                        "despite non-convergence")
    p.add_argument("--out", default=None)
    pf.add_sensor_args(p)
    a = p.parse_args(argv)
    exclude = pf.exclude_from_args(p, a)

    try:
        vol = read_faces(a.map, a.step, exclude, a.keep_all)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    F, S, lo, hi = vol.F, vol.S, vol.lo, vol.hi
    missing, nshell = vol.missing, vol.nshell
    notes = []
    print(f"box {hi[0]-lo[0]:.0f} x {hi[1]-lo[1]:.0f} x {hi[2]-lo[2]:.0f} mm, "
          f"{a.step:.0f} mm grid -> {F.shape} = {F.size:,} nodes")
    print(f"sensors: {vol.choice.text}")
    print(f"boundary: {nshell - missing:,} of {nshell:,} shell nodes measured")
    if missing:
        frac = missing / nshell
        msg = (f"{missing} of {nshell} shell nodes ({100 * frac:.1f} %) have no "
               f"data. The interior is only determined by a CLOSED boundary.")
        if frac > a.max_missing and not a.force:
            print(f"  REFUSED: {msg} That is more than --max-missing "
                  f"({100 * a.max_missing:.1f} %); --force solves anyway.",
                  file=sys.stderr)
            return 2
        try:
            fill_missing_shell(F)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"  WARNING: {msg} Filled from measured neighbours on the "
              f"surface -- {missing} nodes are interpolated, not measured.",
              file=sys.stderr)
        notes.append(f"{missing} of {nshell} shell nodes had no data and were "
                     f"filled from their neighbours on the surface.")

    nslice = int(np.isfinite(S).sum())
    if nslice:
        print(f"interior planes: {nslice:,} measured interior nodes "
              f"({'fixed as data' if a.slices == 'use' else 'held out to check the solve'})")
        if a.slices == "use":
            F = np.where(np.isfinite(S), S, F)
            notes.append(f"{nslice} interior-plane nodes were used as data, so "
                         f"they are not an independent check.")

    print("solving Laplace's equation inside...")
    U, info = solve(F, max_iter=a.max_iter, log=print)
    print(f"  interior |B|: {U.min():.3f} .. {U.max():.3f} mT")
    ctr = tuple(np.array(U.shape) // 2)
    print(f"  centre of the box: {U[ctr]:.3f} mT")
    if not info["converged"]:
        print(f"  ERROR: the relaxation did not converge in {a.max_iter} "
              f"iterations (residual {info['residual']:.2e} mT); the interior "
              f"is unreliable." + ("" if a.force else "  --force to accept."),
              file=sys.stderr)
        notes.append(f"NOT CONVERGED after {a.max_iter} iterations (residual "
                     f"{info['residual']:.1e} mT) -- do not trust the interior.")

    stem = os.path.splitext(os.path.basename(a.map))[0]
    out = a.out or os.path.join(paths.captures_dir(), "plots", stem)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.save(out + "_interior.npy", U)
    print(f"wrote {out}_interior.npy  ({U.shape[0]}x{U.shape[1]}x{U.shape[2]})")

    checks = []
    if nslice and a.slices == "check":
        c = slice_check(U, S)
        if c:
            checks.append(c)
            print(f"interior planes: {c['pct']:.1f} % rms over {c['n']} nodes "
                  f"(rms {c['rms']:.3f} mT)")
    if a.validate:
        hits = sorted(glob.glob(a.validate))
        if not hits:
            print(f"  nothing matched {a.validate}", file=sys.stderr)
        print("checking against interior measurements the solve never saw:")
        results = [validate(U, lo, a.step, f, a.scale, vol.choice.alive)
                   for f in hits]
        results = [r for r in results if r]
        if results:
            print("  rms of every recording:")
            for r in results:
                print(f"    {r['pct']:6.1f} %  {r['rms']:.3f} mT  "
                      f"{r['n']:5d} pts  {r['name']}")
        checks += results
    if checks:
        notes.append(check_note(checks))
        print(notes[-1])

    sub = (f"{hi[0]-lo[0]:.0f} x {hi[1]-lo[1]:.0f} x {hi[2]-lo[2]:.0f} mm box "
           f"from rig ({lo[0]:.0f}, {lo[1]:.0f}, {lo[2]:.0f}) - "
           f"{a.step:.0f} mm grid - {vol.choice.text} - {stem}")
    png = draw_slices(U, lo, hi, a.step, "Inside the box, from its surface",
                      sub, out + "_interior.png", notes)
    print(f"wrote {png}  ({os.path.getsize(png)/1e6:.1f} MB)")
    p3 = draw_3d(U, lo, hi, a.step, "Inside the box, from its surface",
                 sub, out + "_interior3d.png", notes)
    print(f"wrote {p3}  ({os.path.getsize(p3)/1e6:.1f} MB)")
    return 0 if info["converged"] or a.force else 1


if __name__ == "__main__":
    sys.exit(main())
