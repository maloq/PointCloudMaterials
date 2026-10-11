# Fixed initial-velocity CUDA MEAM response comparison up to0.5ps

[All datasets](../README.md) · [Browsable card](response-velocity-cuda500-20261004-40af896d.html) · [Full metadata](../records/response-velocity-cuda500-20261004-40af896d.json)

135 original unrelaxed local80 queries; four sampled fixed velocities;100/500fs feature means and four analytic response directions. Full70304-atom periodic teacher using compiled CUDA MEAM, float32 force arithmetic and float64 states/observables. Separate newly collected bank.

- ID: `response-velocity-cuda500-20261004`
- Materials: Al
- Classification: **research**; role: conditional_response_training
- Location: `/store/PERSO/vmorozov/simulations/response-velocity-cuda-meam-500fs-20261004-v1`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.003 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "title": "Fixed initial-velocity CUDA MEAM response comparison up to0.5ps",
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "conditional_response_training",
  "classification": "research",
  "description": "135 original unrelaxed local80 queries; four sampled fixed velocities;100/500fs feature means and four analytic response directions. Full70304-atom periodic teacher using compiled CUDA MEAM, float32 force arithmetic and float64 states/observables. Separate newly collected bank.",
  "ancestry": "Original90/15/30 fixed Al64 ancestor roles, same135 geometries. Velocity/noise replicas add no independent ancestors. Fixed release e148b7ec215ba5e6d86fc57d21dac266bbd501f1e91320968266b5dbaeb8f44d.",
  "evidence": [
    "${storage:repo}/configs/simulation/response_velocity_cuda500_20261004.json",
    "${storage:repo}/docs/simulations/response_velocity_20261004.md"
  ],
  "limitations": [
    "Initial velocities are sampled, not native parent MD velocities.",
    "No histories or nuclei-birth prediction fit.",
    "Existing CUDA MEAM validation reused at explicit user request; collection, fits and evaluation have separate completion receipts."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 7 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/response-velocity-cuda500-20261004-40af896d.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T00:11:05.385980+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
