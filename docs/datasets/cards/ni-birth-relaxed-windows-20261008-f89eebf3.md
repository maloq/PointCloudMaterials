# Ni birth windows: full-cell GPU FIRE relaxation

[All datasets](../README.md) · [Browsable card](ni-birth-relaxed-windows-20261008-f89eebf3.html) · [Full metadata](../records/ni-birth-relaxed-windows-20261008-f89eebf3.json)

Native 0.1 ps windows from -6 to +6 ps around catalogue first_time_ps of established PTM liquid-origin roots. Full periodic fixed-box FIRE; overlapping source frames deduplicated.

- ID: `ni-birth-relaxed-windows-20261008`
- Materials: Ni
- Classification: **planned**; role: relaxed_configurations
- Location: `/store/PERSO/vmorozov/simulations/ni-birth-relaxed-windows-20261008`
- Present on this machine: True
- Potentials: Mendelev et al. 2012 Ni EAM/FS
- Complete binary records with arrays present: 6571; these are not independent-source counts.
- Stored frames: 6571; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 10.539 GiB
- Missing metadata: temperature_K, ensemble

## Notes and relationships

```json
{
  "title": "Ni birth windows: full-cell GPU FIRE relaxation",
  "materials": [
    "Ni"
  ],
  "potential_ids": [
    "ni-mendelev2012-eam"
  ],
  "role": "relaxed_configurations",
  "classification": "planned",
  "description": "Native 0.1 ps windows from -6 to +6 ps around catalogue first_time_ps of established PTM liquid-origin roots. Full periodic fixed-box FIRE; overlapping source frames deduplicated.",
  "ancestry": "Paired views of original Ni source frames; original ancestry and roles retained; no new independent melts.",
  "precision": "Float64 native minimizer/states, float32 FastNi forces, float64 scalar energy. Verified float16 archive plus shared native restart for full-precision neighborhoods.",
  "evidence": [
    "/store/PERSO/vmorozov/experiments/ni_relaxation/birth-windows-20261008/technical/plan.json"
  ],
  "plan_identity": "53c649624ce569db7c024d105483415d467035575900e4f0cb5d5a885db3c436",
  "births": 70,
  "sources": 33,
  "unique_cells": 6571,
  "requested_event_frames": 8470,
  "missing_event_frames": 0,
  "existing_exact_recipe_cells": 0,
  "roles": {
    "train": 11,
    "selection": 12,
    "test": 10
  }
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| atom_count | [32000] |
| timestep_ps | [0.001] |
| protocol | ["Full periodic cell, fixed box, generating potential; infinity-norm force convergence, no isolated-patch relaxation.", "fixed_periodic_cell_norm_inf_v1"] |
| coordinate_convention | ["positions are wrapped Cartesian coordinates in angstrom relative to box_low in [0, box_high-box_low) before storage quantization; decode to float32 and wrap again"] |
| frame_count | [1] |
| storage_dtype | ["float16"] |

## Evidence

All 19714 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ni-birth-relaxed-windows-20261008-f89eebf3.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T00:11:05.385980+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
