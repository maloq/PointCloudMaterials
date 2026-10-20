# Initial-velocity conditioned Al MEAM response pilot

[All datasets](../README.md) · [Browsable card](response-velocity-20261004-e32f5018.html) · [Full metadata](../records/response-velocity-20261004-e32f5018.json)

Stopped at user request on 4 October 2026 before encoder training; completed full-cell fixed-velocity response batches retained. Original135 geometries,90/15/30 roles.

- ID: `response-velocity-20261004`
- Materials: Al
- Classification: **research**; role: conditional_response_training
- Location: `/store/PERSO/vmorozov/simulations/response-velocity-20261004-periodic-v4`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.003 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "title": "Initial-velocity conditioned Al MEAM response pilot",
  "materials": [
    "Al"
  ],
  "role": "conditional_response_training",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Stopped at user request on 4 October 2026 before encoder training; completed full-cell fixed-velocity response batches retained. Original135 geometries,90/15/30 roles.",
  "ancestry": "Original90/15/30 fixed-Al64 source roles; same135 geometries as local80 MACE pilot; velocity/noise replicas add no independent ancestors. Fixed identity e148b7ec215ba5e6d86fc57d21dac266bbd501f1e91320968266b5dbaeb8f44d.",
  "evidence": [
    "${storage:repo}/configs/simulation/response_velocity_20261004.json",
    "${storage:repo}/docs/simulations/response_velocity_20261004.md"
  ],
  "limitations": [
    "Initial velocities sampled, not native parent velocities.",
    "No birth prediction or history model in this pilot.",
    "Remaining MEAM checks skipped at explicit user request on 2026-10-04; collection and training completion are recorded by linked receipts."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 5 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/response-velocity-20261004-e32f5018.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
