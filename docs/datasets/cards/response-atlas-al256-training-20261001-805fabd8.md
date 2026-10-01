# Al256 atomistic response-training query collection

[All datasets](../README.md) · [Browsable card](response-atlas-al256-training-20261001-805fabd8.html) · [Full metadata](../records/response-atlas-al256-training-20261001-805fabd8.json)

Fixed MACE-MPA-0 medium, 450K BAOAB, 20/100fs; 56 complete periodic cells; 32/8/16 train/selection/test configurations; fresh value and response streams; float64 numerical restart states, no trajectory export.

- ID: `response-atlas-al256-training-20261001`
- Materials: Al
- Classification: **diagnostic**; role: response_training_mechanism
- Location: `/store/PERSO/vmorozov/simulations/response-atlas-training-20261001`
- Present on this machine: True
- Potentials: MACE-MPA-0 medium checkpoint
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.001 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "title": "Al256 atomistic response-training query collection",
  "materials": [
    "Al"
  ],
  "role": "response_training_mechanism",
  "classification": "diagnostic",
  "potential_ids": [
    "mace-mpa-0-medium"
  ],
  "description": "Fixed MACE-MPA-0 medium, 450K BAOAB, 20/100fs; 56 complete periodic cells; 32/8/16 train/selection/test configurations; fresh value and response streams; float64 numerical restart states, no trajectory export.",
  "ancestry": "All geometries share ideal FCC256 prototype. First16 previously examined numerical-development geometries are training only. Held-out configurations are fresh independent displacement draws, not independent melts.",
  "evidence": [
    "${storage:repo}/configs/response_atlas/atomistic_training_20261001.json",
    "${storage:training_storage}/response_atlas/atomistic-training-20261001/technical/prediction-context.json"
  ],
  "limitations": [
    "Separate synthetic mechanism study, not Al64 window benchmark or MEAM shooting.",
    "Short-time smooth feature responses do not establish picosecond crystallization prediction.",
    "Branches and partial failures publish to STORE before scratch cleanup."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["al256_atomistic_response_training_v1"] |

## Evidence

All 1 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/response-atlas-al256-training-20261001-805fabd8.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-01T19:03:53.141383+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
