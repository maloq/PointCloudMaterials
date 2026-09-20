# Al: expanded full-cell relaxed configurations

[All datasets](../README.md) · [Browsable card](al_relaxed_cells_expanded_20260917-d6275785.html) · [Full metadata](../records/al_relaxed_cells_expanded_20260917-d6275785.json)

Active fixed-box FIRE target campaign; completed cells, convergence and failure records are preserved.

- ID: `al_relaxed_cells_expanded_20260917`
- Materials: Al
- Classification: **building**; role: relaxed_configurations
- Location: `/store/PERSO/vmorozov/relaxed_tda/al-expanded-20260917`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 780; these are not independent-source counts.
- Stored frames: 780; duplicate-group records: 12
- Allocated storage, excluding registered nested datasets: 1.877 GiB
- Missing metadata: temperature_K, ensemble

## Notes and relationships

```json
{
  "title": "Al: expanded full-cell relaxed configurations",
  "materials": [
    "Al"
  ],
  "role": "relaxed_configurations",
  "classification": "building",
  "description": "Active fixed-box FIRE target campaign; completed cells, convergence and failure records are preserved.",
  "evidence": [
    "docs/relaxed_tda_targets.md",
    "configs/simulation/relaxed_tda_al.json"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| atom_count | [70304] |
| timestep_ps | [0.001] |
| protocol | ["Full periodic cell, fixed box, generating potential; infinity-norm force convergence, no isolated-patch relaxation."] |
| coordinate_convention | ["positions are wrapped Cartesian coordinates in angstrom relative to box_low in [0, box_high-box_low) before storage quantization; decode to float32 and wrap again"] |
| frame_count | [1] |
| storage_dtype | ["float16"] |

## Evidence

All 2340 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al_relaxed_cells_expanded_20260917-d6275785.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-20T01:13:02.416719+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
