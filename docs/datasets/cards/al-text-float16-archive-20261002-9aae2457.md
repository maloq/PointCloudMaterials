# Al stopped text-dump float16 archives, October 2

[All datasets](../README.md) · [Browsable card](al-text-float16-archive-20261002-9aae2457.html) · [Full metadata](../records/al-text-float16-archive-20261002-9aae2457.json)

Verified float16 replacements for Al text dumps from SCRATCH; scientific completion and ancestry remain those of each original source. Exact source paths, material evidence and output paths are recorded in the conversion plan. This is not an independent simulation collection.

- ID: `al-text-float16-archive-20261002`
- Materials: Al
- Classification: **review_required**; role: storage_conversion
- Location: `/store/PERSO/vmorozov/trajectory-archives/al-float16-20261002`
- Present on this machine: True
- Potentials: Unknown / not applicable
- Complete binary records with arrays present: 184; these are not independent-source counts.
- Stored frames: 124891; duplicate-group records: 162
- Allocated storage, excluding registered nested datasets: 68.360 GiB
- Missing metadata: generating potential identity, temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "title": "Al stopped text-dump float16 archives, October 2",
  "materials": [
    "Al"
  ],
  "role": "storage_conversion",
  "classification": "review_required",
  "description": "Verified float16 replacements for Al text dumps from SCRATCH; scientific completion and ancestry remain those of each original source. Exact source paths, material evidence and output paths are recorded in the conversion plan. This is not an independent simulation collection.",
  "potential_ids": [],
  "limitations": [
    "Potential and ancestry are inherited per source; consult input_metadata.json or original campaign recipe.",
    "Incomplete terminal frames are retained as exact binary byte fragments, not complete observations."
  ],
  "evidence": [
    "docs/storage/al_float16_20261002/plan.json",
    "docs/storage/al_float16_20261002/summary.json"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| frame_count | [1, 11, 1510, 1996, 21, 23581, 24448, 2882, 4402, 5785, 60001, 81] |
| storage_dtype | ["float16"] |
| atom_count | [108, 256, 70304] |
| coordinate_convention | ["wrapped float32 consumer coordinates relative to box_low; decode float16 to float32 and wrap again"] |

## Evidence

All 368 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-text-float16-archive-20261002-9aae2457.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T00:11:05.385980+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
