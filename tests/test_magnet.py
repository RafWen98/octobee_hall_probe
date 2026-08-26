"""The guided single-magnet calibration."""

import os

import numpy as np

from octobee.calib import poses as opcap
from octobee.calib import magnet as omag
from octobee.calib import geometry as pgeom
from tests.helpers import (
    _synth_magnet_passes,
    _synth_magnet_run,
    _trim_error_pct,
    check,
)



def test_magnet_routine():
    """The guided run must recover gain WITHOUT knowing the geometry."""
    print("\nguided magnet calibration")
    g = pgeom.Geometry()
    rng = np.random.default_rng(11)
    gain = rng.normal(1.0, 0.08, 16)
    magnet = np.array([0.0, 200.0, g.fsv_radius_mm + 25.0])
    run = _synth_magnet_run(g, gain, magnet)

    check("a run is complete at four poses", run.complete and len(run) == 4)
    resp, best = run.response()
    # The whole claim: each sensor is loudest in the pose that turned its own
    # face to the magnet, which is what makes the 16 peaks comparable.
    faces = np.array([g.face(i + 1) for i in range(16)])
    same_face_same_pose = all(
        len({int(best[i]) for i in range(16) if faces[i] == f}) == 1
        for f in range(pgeom.N_FACES))
    check("each face answers in exactly one pose", same_face_same_pose,
          f"poses per sensor: {best.tolist()}")

    trim = run.trim()
    recovered = trim * gain
    err = float(np.abs(recovered / np.median(recovered) - 1).max())
    check("the trim recovers the injected gain with no geometry model",
          err < 0.02, f"worst residual {100 * err:.2f} % on a 4 mm sample grid")

    # A fixed magnet at one place cannot do this: the raw peaks of a single
    # pose span the 1/r^3 spread, which is what the extra poses remove.
    one_pose = run.sweeps[0].peaks
    check("one pose alone would have been badly unfair",
          one_pose.max() / one_pose.min() > 5 * (resp.max() / resp.min()),
          f"single pose spreads {one_pose.max() / one_pose.min():.0f}x, "
          f"four poses {resp.max() / resp.min():.2f}x")

    # ---- what the run says about where the sensors are ----
    # The sweep meets the tip first, because the tube's +Z maps to +y and the
    # head is driven the +ve way -- so the far sensor reaches a fixed magnet at
    # a SMALLER stage position. Read off MOUNT_ROT, not written down, so a
    # remounted probe reverses it automatically.
    slots = run.measured_slots()
    check("a correct file measures back exactly as it was written",
          all(slots[sid] == g.slot(sid) for sid in range(1, 17)),
          "the run reproduces the layout it was generated from")

    # Now the case that matters: a file that is wrong the way this rig's was,
    # with one face's sensors numbered backwards along the tube.
    # Built from the same layout the synthetic run was generated against --
    # NOT from the file on disk, which is this rig's real measured geometry and
    # would make the comparison meaningless.
    scrambled = pgeom.Geometry()
    for spec in scrambled.sensors:
        if spec["face"] == 0:
            spec["slot"] = pgeom.SENSORS_PER_FACE - 1 - spec["slot"]
    check("a reversed face is caught", bool(run.check_geometry(scrambled)),
          "; ".join(run.check_geometry(scrambled))[:90])
    changed = run.apply_to_geometry(scrambled)
    check("applying the run fixes exactly the reversed face",
          sorted(sid for sid, _w, _n in changed)
          == [sid for sid in range(1, 17) if g.face(sid) == 0],
          str([(f"S{s}", w, n) for s, w, n in changed]))
    check("and the file then agrees with the run that corrected it",
          run.check_geometry(scrambled) == [])
    check("a corrected file is marked as measured, not as a generated layout",
          scrambled.mapping == pgeom.MEASURED)
    try:
        pgeom._default_sensors(pgeom.MEASURED)
        ok = False
    except ValueError:
        ok = True
    check("and cannot be regenerated back into a guess", ok)

    groups = run.slot_groups()
    check("the sensors pass the magnet in rings of four",
          [len(ids) for _at, ids in groups] == [pgeom.SENSORS_PER_FACE] * 4,
          " | ".join("+".join(f"S{s}" for s in ids) for _at, ids in groups))
    check("a correct geometry file draws no complaint",
          run.check_geometry(g) == [], "; ".join(run.check_geometry(g)))

    # The failure this routine is most likely to suffer in the room.
    bad = _synth_magnet_run(g, gain, magnet, shift_pose=2)
    bad_trim = bad.trim() * gain
    check("a pose that also shifted sideways changes the answer",
          float(np.abs(bad_trim / np.median(bad_trim) - 1).max()) > 0.05,
          "so the wizard's insistence on turning about the axis is not "
          "decoration")

    # ---- the setup fault that looks exactly like gain ----
    bal = run.face_balance(g)
    check("a concentric head shows opposite faces agreeing",
          not bal["notes"],
          "  ".join(f"{pgeom.FACE_NAMES[f]}/{pgeom.FACE_NAMES[o]} {r:.3f}x"
                    for f, o, r, _e in bal["pairs"]))

    # 2 mm off-centre against a magnet ~25 mm away: a few per cent of distance,
    # which 1/r^3 turns into a ten per cent "gain" error on two faces.
    ecc = _synth_magnet_run(g, gain, magnet, eccentric_mm=2.0)
    eb = ecc.face_balance(g)
    check("an off-centre head is caught by the opposite-face check",
          bool(eb["notes"]),
          "  ".join(f"{pgeom.FACE_NAMES[f]}/{pgeom.FACE_NAMES[o]} {r:.3f}x"
                    for f, o, r, _e in eb["pairs"]))
    check("and the report stops calling the spread gain",
          "NOT gain alone" in ecc.report(g))
    # Both pairs move, not just the one the offset points along: the chips sit
    # ~14 mm off the rotation axis, so a sideways offset changes the in-plane
    # separation to first order too. What holds is that the pair the offset
    # points ALONG is the worse of the two.
    by_pair = {f: abs(r - 1.0) for f, _o, r, _e in eb["pairs"]}
    check("the pair the offset points along is the worst affected",
          max(by_pair, key=by_pair.get) == 0,
          "  ".join(f"faces {f}/{f + 2}: {v:.3f}" for f, v in by_pair.items()))
    # It is separable BECAUSE it is antisymmetric: the product of two opposite
    # faces survives the offset even though each of them does not.
    prod = {f: eb["means"][f] * eb["means"][o] for f, o, _r, _e in eb["pairs"]}
    clean = {f: bal["means"][f] * bal["means"][o] for f, o, _r, _e in bal["pairs"]}
    check("the product of opposite faces survives the offset",
          all(abs(prod[f] / clean[f] - 1.0) < 0.02 for f in prod),
          "  ".join(f"{prod[f] / clean[f]:.3f}" for f in prod))

    sideways = omag.MagnetRun(run.sweeps, axis="x")
    try:
        sideways.measured_slots()
        ok = False
    except ValueError:
        ok = True
    check("sweeping across the tube instead of along it is refused", ok,
          "the equal-approach argument does not hold, and neither does the "
          "slot order")

    # A run abandoned early still reports what it established.
    part = omag.MagnetRun(run.sweeps[:2])
    check("two poses still report, and say what is missing",
          not part.complete and "2 of 4 poses" in part.report(g)
          and any("poses recorded" in n for n in part.check_geometry(g)))


def test_magnet_plane():
    """The plane and the dither must beat a bare axial sweep on bent arms.

    The axial-only run is not a straw man -- it is what this routine did until
    the passes were added, and on perfectly built arms it is still right. What
    it cannot do is tell a chip that is a millimetre closer to the magnet from
    a chip with 15 % more gain, and that is what is checked here.
    """
    print("\nguided magnet calibration -- plane and standoff passes")
    g = pgeom.Geometry()
    rng = np.random.default_rng(23)
    gain = rng.normal(1.0, 0.05, 16)
    magnet = np.array([0.0, 200.0, g.fsv_radius_mm + 25.0])

    # --- the fit, on its own, against a known distance
    z = omag.suggested_dither(30.0)
    d, n, _rms, _sigma = omag.fit_falloff(z, 100.0 / np.abs(30.0 - z) ** 3.0)
    check("the dither fit recovers a known standoff", abs(d - 30.0) < 0.2,
          f"got {d:.3f} mm, want 30")
    check("and the exponent that produced it", abs(n - 3.0) < 0.05,
          f"got 1/r^{n:.3f}, want 3")
    check("a dither with no contrast is refused rather than guessed",
          not np.isfinite(omag.fit_falloff(z, np.ones_like(z))[0]))
    check("three points claim no precision, having none to spare",
          not np.isfinite(omag.fit_falloff(
              z[:3], 100.0 / np.abs(30.0 - z[:3]) ** 3.0)[3]))

    # --- a perfectly built probe: the passes must not make it worse
    perfect_axial = _synth_magnet_passes(g, gain, magnet, (0, 0, 0),
                                         plane=False, dither=False)
    perfect_full = _synth_magnet_passes(g, gain, magnet, (0, 0, 0))
    e_axial = _trim_error_pct(perfect_axial, gain)
    e_full = _trim_error_pct(perfect_full, gain)
    check("on straight arms the extra passes cost nothing",
          e_full < max(2.0 * e_axial, 1.0),
          f"axial {e_axial:.2f} % -> all three {e_full:.2f} %")

    # --- a real probe: arms a millimetre out, one direction at a time, so
    # each pass is checked against the error it actually claims to remove
    for label, jit in (("transverse", (1.0, 0.0, 0.0)),
                       ("standoff", (0.0, 0.0, 1.0)),
                       ("all three axes", (1.0, 1.0, 1.0))):
        e_axial = _trim_error_pct(
            _synth_magnet_passes(g, gain, magnet, jit, plane=False,
                                 dither=False), gain)
        e_full = _trim_error_pct(
            _synth_magnet_passes(g, gain, magnet, jit), gain)
        check(f"the passes recover gain through {label} misplacement",
              e_full < 0.4 * e_axial,
              f"{e_axial:.2f} % of fake gain -> {e_full:.2f} %")

    jitter = (1.0, 1.0, 1.0)
    bent_axial = _synth_magnet_passes(g, gain, magnet, jitter,
                                      plane=False, dither=False)
    bent_plane = _synth_magnet_passes(g, gain, magnet, jitter, dither=False)
    bent_full = _synth_magnet_passes(g, gain, magnet, jitter)
    e_axial = _trim_error_pct(bent_axial, gain)
    e_plane = _trim_error_pct(bent_plane, gain)
    e_full = _trim_error_pct(bent_full, gain)
    check("a bare axial sweep turns bent arms into fake gain", e_axial > 3.0,
          f"{e_axial:.2f} % of trim is placement, not gain")

    # The counter-intuitive one, pinned so nobody "optimises" pass C away:
    # peaking over the plane moves every chip to its true closest approach,
    # which is nearer, and that amplifies whatever first-order error is left.
    # The cut is only a win once the dither is there to collect it.
    check("the plane WITHOUT the dither is worse, not better",
          e_plane > e_axial,
          f"{e_axial:.2f} % -> {e_plane:.2f} % -- the cut moves the chips "
          f"closer, so what is left hurts more")
    check("and the two together are what actually pay",
          e_full < 0.4 * e_axial,
          f"{e_axial:.2f} % -> {e_full:.2f} % with all three passes")
    check("the cut is what makes the dither's model true",
          np.nanmedian(bent_full.standoffs())
          < 0.7 * np.nanmedian(_synth_magnet_passes(
              g, gain, magnet, jitter, plane=False).standoffs()),
          f"standoff reads {np.nanmedian(bent_full.standoffs()):.1f} mm under "
          f"the magnet, {np.nanmedian(_synth_magnet_passes(g, gain, magnet, jitter, plane=False).standoffs()):.1f} mm off to one side")

    # --- the measurements the passes expose, not just the trim they feed
    across = bent_full.across_positions()
    check("the plane reports where the arms actually are",
          across is not None and np.isfinite(across).all()
          and float(np.ptp(across)) > 0.5,
          f"transverse peaks span {float(np.ptp(across)):.2f} mm")
    stand = bent_full.standoffs()
    n_fit = int(np.isfinite(stand).sum())
    check("the dither measures a standoff for every sensor", n_fit == 16,
          f"{n_fit} of 16 fitted, median {np.nanmedian(stand):.1f} mm")
    fall = bent_full.falloffs()
    check("and agrees the field falls like a dipole",
          2.5 < float(np.nanmedian(fall)) < 3.5,
          f"measured 1/r^{np.nanmedian(fall):.2f}")

    # --- degradation: a run without pass C must not silently invent one
    corr = bent_plane.standoff_correction()
    check("with no dither the standoff correction is exactly the identity",
          np.allclose(corr, 1.0), f"max deviation {np.max(np.abs(corr - 1)):.2e}")
    check("a one-axis run still analyses after the passes were added",
          np.isfinite(bent_axial.trim()).all()
          and bent_axial.across_positions() is None)


def test_magnet_plane_persistence(workdir):
    g = pgeom.Geometry()
    rng = np.random.default_rng(31)
    run = _synth_magnet_passes(g, rng.normal(1.0, 0.05, 16),
                               np.array([0.0, 200.0, g.fsv_radius_mm + 25.0]),
                               (1.0, 1.0, 1.0))
    path = run.save(os.path.join(workdir, "magcal_plane"))
    back = omag.MagnetRun.load(path)
    check("a plane run round trips through disk",
          len(back) == len(run)
          and np.allclose(back.trim(), run.trim())
          and back.across == "x" and back.normal == "z")
    check("and its dithers survive, standoffs and all",
          all(len(s.dithers) == pgeom.SENSORS_PER_FACE for s in back.sweeps)
          and np.allclose(back.standoffs(), run.standoffs(), equal_nan=True))


def test_magnet_persistence(workdir):
    g = pgeom.Geometry()
    rng = np.random.default_rng(5)
    run = _synth_magnet_run(g, rng.normal(1.0, 0.05, 16),
                            np.array([0.0, 200.0, g.fsv_radius_mm + 25.0]),
                            span_mm=120.0, step_mm=10.0)
    path = run.save(os.path.join(workdir, "magcal_test"))
    back = omag.MagnetRun.load(path)
    check("a guided run round trips through disk",
          len(back) == len(run)
          and np.allclose(back.trim(), run.trim())
          and back.axis == run.axis)


def test_survey_pairing():
    """The opposed pose is i + n/2, which only equals i + 2 at four poses."""
    print("\nlocation survey pairing")
    rng = np.random.default_rng(11)
    offsets = rng.normal(0, 0.4, (16, 3))          # dominate the field, as on the bench
    bt_true = 0.047                                # mT transverse

    def poses(n):
        """n poses of a rigid probe rolled about the tube axis."""
        out = []
        for k in range(n):
            phi = 2 * np.pi * k / n
            field = np.array([bt_true * np.cos(phi), bt_true * np.sin(phi), 0.0])
            out.append(offsets + field[None, :])
        return out

    for n in (4, 6, 8):
        d13, _d24, _ratio, worst = opcap.survey_consistency(poses(n),
                                                          np.ones(16, bool))
        # Opposed poses differ by exactly 2*bt whatever n is, so the recovered
        # transverse field must come back at bt_true -- not bt_true*sin(pi/n).
        check(f"{n} poses recover the true transverse field",
              abs(float(np.median(d13)) - bt_true * 1e3) < 1e-6,
              f"got {np.median(d13):.4f} uT, want {bt_true * 1e3:.4f}")
        check(f"{n} poses agree with themselves", worst < 1e-6,
              f"worst self-disagreement {worst:.3g} %")

    try:
        opcap.survey_consistency(poses(3), np.ones(16, bool))
        ok = False
    except ValueError:
        ok = True
    check("an odd pose count is refused rather than mispaired", ok)


def test_magnet_incremental_save(workdir):
    """Saving repeatedly to one base must never destroy the last good copy.

    The wizard writes after every pose now, so save() is routinely called with
    an earlier version of itself already under that name. A write that dies
    part way -- a full disk, a killed process -- would otherwise take the
    poses already on disk with it, which is the exact loss that saving early
    was added to prevent. The temp-file rename is what stops that, and the
    only way to check it is to make a write fail.
    """
    print("\nguided magnet calibration -- saving as the run goes")
    g = pgeom.Geometry()
    rng = np.random.default_rng(37)
    full = _synth_magnet_run(g, rng.normal(1.0, 0.05, 16),
                             np.array([0.0, 200.0, g.fsv_radius_mm + 25.0]))
    base = os.path.join(workdir, "magcal_growing")

    part = omag.MagnetRun(axis=full.axis)
    for n, sweep in enumerate(full.sweeps, start=1):
        part.add(sweep)
        part.save(base)
        check(f"a {n}-pose run reads back with {n} pose(s)",
              len(omag.MagnetRun.load(base + ".npz")) == n)

    def explode(*_a, **_k):
        raise OSError("no space left on device")

    real, omag.np.savez_compressed = omag.np.savez_compressed, explode
    try:
        omag.MagnetRun(full.sweeps[:1], axis=full.axis).save(base)
        check("a failed write is reported, not swallowed", False)
    except OSError:
        check("a failed write is reported, not swallowed", True)
    finally:
        omag.np.savez_compressed = real

    check("and the four poses already saved are still there",
          len(omag.MagnetRun.load(base + ".npz")) == omag.N_POSES)
    check("with no half-written temporary left behind",
          not os.path.exists(base + ".npz.part")
          and not os.path.exists(base + ".json.part"),
          str(sorted(os.listdir(workdir))))


def test_dither_quality():
    """Pass C must be caught when it measures nothing.

    This is the check that was missing on 2026-08-25. A run entered at a 20 mm
    standoff against a real one nearer 50 gave a dither a tenth of the
    distance instead of a quarter; every sensor returned a finite distance,
    the residuals were small, and the correction multiplied the gain trim's
    spread from 1.30x to 5.89x. Nothing about it looked wrong.

    The failure needs NOISE to reproduce -- a noiseless fit recovers the right
    distance even from a badly sized dither, and it is noise sliding d and n
    along their shared valley that produces the confident wrong answer. So
    these runs carry 3 uT/point, which is about what a real average at one
    point leaves.
    """
    print("\nguided magnet calibration -- is pass C worth believing")
    g = pgeom.Geometry()
    rng = np.random.default_rng(23)
    gain = rng.normal(1.0, 0.05, 16)
    jitter = (1.0, 1.0, 1.0)

    def run_at(real_mm, sized_mm, noise_ut=3.0, seed=4):
        magnet = np.array([0.0, 200.0, g.fsv_radius_mm + real_mm])
        return _synth_magnet_passes(g, gain, magnet, jitter,
                                    standoff_mm=sized_mm, noise_ut=noise_ut,
                                    seed=seed)

    # ---- a dither sized for the distance it is actually at
    good = run_at(25.0, 25.0)
    q = good.dither_quality(expect_mm=25.0)
    check("a correctly sized dither is believed", q["usable"], str(q["notes"]))
    check("and it measures the distance it was set at",
          abs(q["median_mm"] - 25.0) < 0.3 * 25.0,
          f"fitted {q['median_mm']:.1f} mm against 25 entered")
    check("and agrees the field falls like a dipole",
          abs(q["median_n"] - 3.0) <= omag.DITHER_N_TOLERANCE,
          f"1/r^{q['median_n']:.2f}")
    check("its sixteen distances do not correlate with its sixteen exponents",
          abs(q["corr_d_n"]) < omag.DITHER_DEGENERACY_R,
          f"corr(d,n) = {q['corr_d_n']:+.3f}")

    # The gate must not be trigger-happy: dropping a GOOD pass C costs real
    # accuracy, which is the whole reason it is not simply always dropped.
    e_full = _trim_error_pct(good, gain)
    prod = good.trim(use_dither=False) * gain
    e_plane = 100.0 * float(np.std(prod / np.median(prod)))
    check("throwing away a good pass C would cost accuracy",
          e_full < 0.5 * e_plane,
          f"A+B+C {e_full:.2f} % against A+B {e_plane:.2f} %")

    # ---- the 2026-08-25 failure: sized for 20, driven against 50 and 75
    for real in (50.0, 75.0):
        bad = run_at(real, 20.0)
        q = bad.dither_quality(expect_mm=20.0)
        check(f"a dither sized for 20 mm at a real {real:.0f} mm is refused",
              not q["usable"], str(q["notes"])[:100])
        check("  and the correlation is what catches it",
              abs(q["corr_d_n"]) > omag.DITHER_DEGENERACY_R,
              f"corr(d,n) = {q['corr_d_n']:+.3f}, against the +0.994 the real "
              f"run of 2026-08-25 returned")
        check("  and the fitted distance disagrees with what was entered",
              any("entered on the panel" in n for n in q["notes"]),
              str(q["notes"])[:100])

    # ---- dropping pass C is a real fallback, not a no-op
    with_c, _ = good.response(use_dither=True)
    without_c, _ = good.response(use_dither=False)
    check("dropping pass C returns a different, plane-only answer",
          not np.allclose(with_c, without_c),
          f"max change {100 * np.max(np.abs(with_c / without_c - 1)):.1f} %")
    check("and the plane-only answer is the bare peaks",
          np.allclose(without_c,
                      good.peak_table()[good.best_pose(), np.arange(16)]))

    # And the belt to the gate's braces, worth pinning because it is what
    # stopped the synthetic bench ever reproducing the 2026-08-25 damage:
    # standoff_correction() already shrinks a displacement toward zero by how
    # much of it survives the fit's own noise, so a dither this degenerate
    # collapses to the identity on its own. The gate exists for the case where
    # it does not -- a fit that is confidently wrong rather than obviously
    # noisy, which is what the real run returned.
    bad = run_at(50.0, 20.0)
    check("a degenerate dither is also shrunk toward doing nothing",
          np.max(np.abs(bad.standoff_correction() - 1.0)) < 0.05,
          f"largest correction {100 * np.max(np.abs(bad.standoff_correction() - 1)):.2f} %")

    # ---- a run with no pass C at all must not claim one
    none = _synth_magnet_passes(g, gain,
                                np.array([0.0, 200.0, g.fsv_radius_mm + 25.0]),
                                jitter, dither=False)
    q = none.dither_quality()
    check("a run with no dither says there is nothing to apply",
          not q["usable"] and q["n_fitted"] == 0, str(q["notes"])[:100])


def test_magnet_peak_vectors():
    """The three components must be kept, and must say how the chips are turned."""
    print("\nguided magnet calibration -- the peak vectors")
    g = pgeom.Geometry()
    rng = np.random.default_rng(5)
    gain = rng.normal(1.0, 0.06, 16)
    magnet = np.array([0.0, 200.0, g.fsv_radius_mm + 25.0])
    # A finer step than the other tests use: they only need each ring PLACED,
    # and a 4 mm grid does that. This one reads a DIRECTION off the peak, and
    # a peak sampled 2 mm off its top tips the vector by a degree or two of
    # pure sampling error -- which would sit on top of every orientation
    # number here and be indistinguishable from a chip being turned.
    run = _synth_magnet_run(g, gain, magnet, step_mm=1.0)

    v = run.peak_vectors()
    check("every sensor returns a peak vector", np.isfinite(v).all(),
          f"{int(np.isfinite(v).all(axis=1).sum())} of 16")
    check("and its magnitude is the peak the trim was built on",
          np.allclose(np.linalg.norm(v, axis=1),
                      run.peak_table()[run.best_pose(), np.arange(16)]))

    # The run is synthesised THROUGH the same rotations the check reads back,
    # so a correct geometry file must come back as sixteen chips agreeing.
    o = run.orientation_check(g)
    check("on a probe built to the file, the sixteen chips agree",
          o["median_deg"] < 1.0,
          f"median {o['median_deg']:.3f} deg, worst {o['max_deg']:.3f} deg")
    check("and no chip is called out", not o["notes"], str(o["notes"])[:120])

    # Turn one chip 30 degrees about its own board normal and the check must
    # see it -- and must see it WITHIN its own face, where no indexing error
    # can be blamed for it.
    bent = pgeom.Geometry()
    bent.sensors[6]["chip_rot_deg"] = 30.0
    o2 = run.orientation_check(bent)
    check("a chip turned 30 deg from the file is caught",
          o2["within_pose_deg"][6] > omag.ORIENT_SENSOR_WARN_DEG,
          f"S7 reads {o2['within_pose_deg'][6]:.1f} deg from its own face")
    check("and it is named", any("S7 " in n for n in o2["notes"]),
          str(o2["notes"])[:160])
    others = np.delete(o2["within_pose_deg"], 6)
    check("without dragging the other fifteen with it",
          np.nanmax(others) < omag.ORIENT_SENSOR_WARN_DEG,
          f"worst of the rest {np.nanmax(others):.2f} deg")

    # The measured placement is the thing the run used to print and throw away.
    p = run.placement()
    check("placement carries one row per sensor and three columns",
          p.shape == (16, 3))
    check("and its axial column is where each sensor peaked",
          np.allclose(p[:, 0], run.peak_positions()))


def test_magnet_range_check():
    """Clipping must be caught before it becomes a trim."""
    print("\nguided magnet calibration -- running out of range")
    g = pgeom.Geometry()
    gain = np.ones(16)
    magnet = np.array([0.0, 200.0, g.fsv_radius_mm + 25.0])
    run = _synth_magnet_passes(g, gain, magnet, (0.0, 0.0, 0.0))
    peak = float(np.max(np.abs(run.peak_vectors())))

    roomy = run.range_check(np.full(16, 20.0 * peak))
    check("a run with headroom is not complained about",
          roomy["usable"] and not roomy["notes"], str(roomy["notes"])[:120])

    tight = run.range_check(np.full(16, peak * 0.5))
    check("a channel over full scale is caught", not tight["usable"],
          str(tight["notes"])[:120])
    check("and the sensors that did it are named", len(tight["over"]) > 0,
          f"over: {tight['over']}")
    check("and it says how much further away to move the magnet",
          tight["back_off_factor"] > 1.0,
          f"{tight['back_off_factor']:.2f}x further")

    # Per AXIS, not on |B|: a sensor can be inside range on magnitude while
    # one of its three outputs is on the rail, and that is the case that
    # produces a plausible-looking |B| out of a clipped channel.
    one_axis = float(np.max(np.abs(run.peak_vectors()[:, 0])))
    mags = float(np.max(np.linalg.norm(run.peak_vectors(), axis=1)))
    fs = np.full(16, one_axis * 0.9)
    check("clipping is judged per axis, so |B| under full scale is no defence",
          not run.range_check(fs)["usable"] and mags > one_axis,
          f"worst axis {one_axis:.3f} mT against |B| {mags:.3f} mT at "
          f"+/-{fs[0]:.3f} mT full scale")

    # The dither goes CLOSER than the peak, so it is usually what trips this.
    near = run.range_check(np.full(16, peak * 1.5))
    check("the dither's near end counts toward the range, not just the peak",
          near["fraction"].max() > 1.0 / 1.5,
          f"the worst point is "
          f"{near['fraction'].max() / (1 / 1.5):.2f}x the peak field")
