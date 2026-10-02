# Al256 response-training default: float32 batched cuEquivariance (planned)

[All datasets](../README.md) · [Browsable card](response-atlas-al256-training-fast-20261002-7a1eafa3.html) · [Full metadata](../records/response-atlas-al256-training-fast-20261002-7a1eafa3.json)

Planned default for the fixed MACE-MPA-0 450K BAOAB 20/100fs mechanism case. Float32, batch4 GPU graphs, eight response-value reuses and one training execution audit; test streams disjoint. Separate seed namespace 70000000. No new training or full collection launched by the default change.

- ID: `response-atlas-al256-training-fast-20261002`
- Materials: Al
- Classification: **diagnostic**; role: response_training_mechanism
- Location: `/store/PERSO/vmorozov/simulations/response-atlas-training-fast-20261002`
- Present on this machine: True
- Potentials: MACE-MPA-0 medium checkpoint
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.001 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "title": "Al256 response-training default: float32 batched cuEquivariance (planned)",
  "materials": [
    "Al"
  ],
  "role": "response_training_mechanism",
  "classification": "diagnostic",
  "potential_ids": [
    "mace-mpa-0-medium"
  ],
  "description": "Planned default for the fixed MACE-MPA-0 450K BAOAB 20/100fs mechanism case. Float32, batch4 GPU graphs, eight response-value reuses and one training execution audit; test streams disjoint. Separate seed namespace 70000000. No new training or full collection launched by the default change.",
  "ancestry": "All geometries share ideal FCC256 prototype. First16 previously examined numerical-development geometries are training only. Held-out configurations are fresh independent displacement draws, not independent melts. Same 56 generating geometries as the October1 mechanism cohort; not new independent parents. Fresh stochastic seed namespace.",
  "evidence": [
    "${storage:repo}/configs/simulation/response_atlas_training_fast_20261002.json",
    "${storage:repo}/configs/simulation/response_atlas_fast_default.json",
    "${storage:repo}/docs/response_atlas.md"
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
| protocol | ["al256_atomistic_response_training_float32_v2"] |

## Evidence

All 1 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/response-atlas-al256-training-fast-20261002-7a1eafa3.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-02T12:18:52.028981+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
