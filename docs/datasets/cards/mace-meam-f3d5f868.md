# Al · earlier relaxed-MEAM training views

[All datasets](../README.md) · [Browsable card](mace-meam-f3d5f868.html) · [Full metadata](../records/mace-meam-f3d5f868.json)

Derived inputs or targets. Source lineages and target protocol remain part of the dataset definition.

- ID: `mace-meam`
- Materials: Al
- Classification: **derived**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/mace-meam`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.167 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Al \u00b7 earlier relaxed-MEAM training views",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "derived",
  "description": "Derived inputs or targets. Source lineages and target protocol remain part of the dataset definition.",
  "evidence": [
    "${dataset:mace-meam}/manifest.json",
    "output/mace_al_denoising_20260910/fire/data/manifest.json"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| radius_A | [9.192189] |
| frame_offsets_ps | [[-3.0, -2.25, -1.5, -0.75, 0.0]] |
| seed | [20260910] |
| split | ["test", "train", "val"] |
| temperature_K | [400.0, 450.0, 510.0] |

## Evidence

All 2 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/mace-meam-f3d5f868.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T12:07:34.550029+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
