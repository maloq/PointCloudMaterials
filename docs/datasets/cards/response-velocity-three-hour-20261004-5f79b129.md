# Three-hour initial-velocity Al MEAM response pilot

[All datasets](../README.md) · [Browsable card](response-velocity-three-hour-20261004-5f79b129.html) · [Full metadata](../records/response-velocity-three-hour-20261004-5f79b129.json)

Twenty selected original unrelaxed local80 observations,12/4/4 source roles,2 fixed velocity fields each,100/500/1000fs targets from full periodic MEAM cells. Two shots per bank. Earlier matching batches reused with original producer provenance.

- ID: `response-velocity-three-hour-20261004`
- Materials: Al
- Classification: **research**; role: conditional_response_training
- Location: `/store/PERSO/vmorozov/simulations/response-velocity-20261004-three-hour-v1`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.014 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "title": "Three-hour initial-velocity Al MEAM response pilot",
  "materials": [
    "Al"
  ],
  "role": "conditional_response_training",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Twenty selected original unrelaxed local80 observations,12/4/4 source roles,2 fixed velocity fields each,100/500/1000fs targets from full periodic MEAM cells. Two shots per bank. Earlier matching batches reused with original producer provenance.",
  "ancestry": "Subset of the same135 original ancestors, original Al64 source roles unchanged; replicas do not add ancestry. Static transfer uses unchanged all64 rows.",
  "evidence": [
    "${storage:repo}/configs/simulation/response_velocity_three_hour_20261004.json",
    "${storage:repo}/docs/simulations/response_velocity_20261004.md"
  ],
  "limitations": [
    "Sampled initial velocities, not native MD velocities.",
    "Small feasibility pilot:12 train and4 held-out test ancestors,one training seed; not a conclusive model ranking.",
    "No nuclei-birth prediction fit.",
    "User requested no further MEAM checks; all workers have a shared3h deadline and no automatic requeue."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 40 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/response-velocity-three-hour-20261004-5f79b129.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
