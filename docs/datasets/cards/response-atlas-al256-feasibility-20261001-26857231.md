# Al256 response-oracle feasibility query bundles

[All datasets](../README.md) · [Browsable card](response-atlas-al256-feasibility-20261001-26857231.html) · [Full metadata](../records/response-atlas-al256-feasibility-20261001-26857231.json)

Full-cell periodic perturbed-FCC pilot, fixed MACE-MPA-0, 450 K BAOAB. Float64 query/restart states and compact values/responses; no trajectory export.

- ID: `response-atlas-al256-feasibility-20261001`
- Materials: Al
- Classification: **diagnostic**; role: numerical_feasibility
- Location: `/store/PERSO/vmorozov/simulations/response-atlas-20261001`
- Present on this machine: True
- Potentials: MACE-MPA-0 medium checkpoint
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.010 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "title": "Al256 response-oracle feasibility query bundles",
  "materials": [
    "Al"
  ],
  "role": "numerical_feasibility",
  "classification": "diagnostic",
  "potential_ids": [
    "mace-mpa-0-medium"
  ],
  "description": "Full-cell periodic perturbed-FCC pilot, fixed MACE-MPA-0, 450 K BAOAB. Float64 query/restart states and compact values/responses; no trajectory export.",
  "ancestry": "All 16 controlled configurations share the ideal FCC prototype; not independent liquid melts.",
  "evidence": [
    "${storage:repo}/configs/response_atlas/feasibility_20261001.json"
  ],
  "limitations": [
    "Development-only numerical pilot; no physical generalization or crystallization-rate claim.",
    "New simulation work is staged on SCRATCH and copied to STORE per query."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 16 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/response-atlas-al256-feasibility-20261001-26857231.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-02T12:18:52.028981+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
