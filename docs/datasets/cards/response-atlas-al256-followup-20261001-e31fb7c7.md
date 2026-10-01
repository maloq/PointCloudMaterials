# Al256 response precision follow-up

[All datasets](../README.md) · [Browsable card](response-atlas-al256-followup-20261001-e31fb7c7.html) · [Full metadata](../records/response-atlas-al256-followup-20261001-e31fb7c7.json)

Fixed MACE-MPA-0; 32 AD branches and 16 fresh CRN pairs per direction; 20/100 fs. Float64 query/restart states.

- ID: `response-atlas-al256-followup-20261001`
- Materials: Al
- Classification: **diagnostic**; role: numerical_feasibility
- Location: `/store/PERSO/vmorozov/simulations/response-atlas-followup-20261001`
- Present on this machine: True
- Potentials: MACE-MPA-0 medium checkpoint
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.014 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "title": "Al256 response precision follow-up",
  "materials": [
    "Al"
  ],
  "role": "numerical_feasibility",
  "classification": "diagnostic",
  "potential_ids": [
    "mace-mpa-0-medium"
  ],
  "description": "Fixed MACE-MPA-0; 32 AD branches and 16 fresh CRN pairs per direction; 20/100 fs. Float64 query/restart states.",
  "ancestry": "Four existing development states 0,1,8,12 from the shared FCC prototype; fresh stochastic branches, not new independent parents.",
  "evidence": [
    "${storage:repo}/configs/simulation/response_atlas_followup_20261001.json"
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

All 4 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/response-atlas-al256-followup-20261001-e31fb7c7.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-01T19:03:53.141383+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
