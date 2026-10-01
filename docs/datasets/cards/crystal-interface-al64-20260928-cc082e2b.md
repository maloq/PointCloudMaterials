# CIV-MACE128 crystal-interface targets and uniform held-out interiors

[All datasets](../README.md) · [Browsable card](crystal-interface-al64-20260928-cc082e2b.html) · [Full metadata](../records/crystal-interface-al64-20260928-cc082e2b.json)

Derived labels on the original fixed Al64 geometry; crystal-side interface adjacent to non-crystal components >=64 atoms on the periodic 3.6-A graph. Original source/ancestry roles retained, with 16 outcome-blind uniform centers per held-out frame for interior evaluation. No new simulations. Training parent was a mixed-material snapshot encoder.

- ID: `crystal-interface-al64-20260928`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/crystal-interface/al64-covering-20260928`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 1.787 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "CIV-MACE128 crystal-interface targets and uniform held-out interiors",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Derived labels on the original fixed Al64 geometry; crystal-side interface adjacent to non-crystal components >=64 atoms on the periodic 3.6-A graph. Original source/ancestry roles retained, with 16 outcome-blind uniform centers per held-out frame for interior evaluation. No new simulations. Training parent was a mixed-material snapshot encoder.",
  "provenance": "Existing al64-covering-20260927 coordinates and past-confirmed PTM lineages; producer src/research/crystal_vector/interface.py",
  "ancestry": "Inherited frozen al64-v1 independent-melt train/selection/calibration/test roles; added centers are correlated observations, not new sources.",
  "evidence": [
    "${storage:cache}/crystal-interface/al64-covering-20260928/plan.json",
    "${storage:cache}/crystal-interface/al64-covering-20260928/manifest.json",
    "experiments/crystal_interface_20260928/README.md"
  ],
  "limitations": [
    "Atom-layer edge convention; large disordered internal pockets and grain boundaries may contribute.",
    "Empty interface is censored, not zero; direction undefined on the layer and at ties.",
    "Fixed at-risk benchmark lacks interior rows; read uniform held-out phase results separately."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["joint_snapshot_interface_vector_v1"] |
| lineage | ["independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_135035943", "independent_melt_13937749", "independent_melt_139885636", "independent_melt_142641279", "independent_melt_144370470", "independent_melt_146234076", "independent_melt_147978527", "independent_melt_151871197", "independent_melt_15458297"] … (150 values; see JSON) |
| frame_count | [801] |
| atom_count | [70304] |
| timestep_fs | [3.0] |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |

## Evidence

All 152 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/crystal-interface-al64-20260928-cc082e2b.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-01T19:03:53.141383+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
