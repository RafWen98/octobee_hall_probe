# Rig v2: boom, keyed nose, chains, rig LAN

Parametric CAD for the reworked rig. `build_rig.py` writes:

- `out/stl/*.stl`: every printed part, in print orientation, in mm
- `out/rig_v2.glb`: the coloured assembly used by the 3D viewer page
- `out/parts.json`: part sizes and masses, chain definitions, sensor FSVs

```bash
python -m venv cadenv
cadenv/Scripts/pip install manifold3d trimesh numpy
cadenv/Scripts/python hardware/rig_v2/build_rig.py
```

The nose takes its board seats from `octobee.calib.geometry` and `config/probe_geometry.json`.
The build fails if any FSV lands off `Geometry.position()`, so a printed nose needs no
software change. The build also checks that no two eval boards collide, that every drag
chain closes and keeps a positive run over the full 0–300 mm cube, and that the carriers and CONC frame on
the boom clear the Z stage, the spring drums, the tower and both chains over that cube.

The Y stage lies on its side (sled in the y-z plane); a printed carriage plate carries the boom.
Both ACQ1001 carriers stand on a printed shelf on the boom and both CONCs share one frame, so no ribbon crosses a moving axis.
That puts the Z stage at about 7 kg against its 4 kg vertical rating. `spring_cap` holds two
constant-force springs that carry all but `Z_NET_DOWN_KG` of it. `parts.json` has the mass
budget and the spring force.

Parameters marked **MEASURE** at the top of the script came from photos rather than drawings.
Check them before printing:

- `OCTO_*`: the OCTO-BEE board (CONC and CELF are the same PCB) and its holes, from photos.
  Print `octo_gauge.stl` first and lay it on a CONC.
- `CARRIAGE_ROWS_Y`: the LTS300C carriage hole rows
- `CAR_*`, `CARRIER_KG`: carrier size in its chassis, and its mass (sets the springs)
- `STAGE_END_*`: the Z stage end the spring cap slips over
- `psu_cradle()`: the supply-brick size

Use only non-magnetic fasteners on the boom (aluminium or brass T-nuts, A4 stainless or brass
screws). The nose uses brass inserts and nylon screws.
