# Expanded unrelaxed local80 future-response geometry

[All datasets](../README.md) · [Browsable card](response-geometry-unrelaxed-20261003-f185b1ec.html) · [Full metadata](../records/response-geometry-unrelaxed-20261003-f185b1ec.json)

720 unrelaxed train parents from90 fixed train ancestors; original15 selection/30 test queries. Same20/100fs MACE paths provide legacy changes, richer changes and richer future endpoints+changes, with directional derivatives. No new independent melt ancestry.

- ID: `response-geometry-unrelaxed-20261003`
- Materials: Al
- Classification: **research**; role: local_response_training
- Location: `/store/PERSO/vmorozov/simulations/response-geometry-unrelaxed-20261003`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM; MACE-MPA-0 medium checkpoint
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.964 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "title": "Expanded unrelaxed local80 future-response geometry",
  "materials": [
    "Al"
  ],
  "role": "local_response_training",
  "classification": "research",
  "potential_ids": [
    "mace-mpa-0-medium",
    "al-lee2003-meam"
  ],
  "description": "720 unrelaxed train parents from90 fixed train ancestors; original15 selection/30 test queries. Same20/100fs MACE paths provide legacy changes, richer changes and richer future endpoints+changes, with directional derivatives. No new independent melt ancestry.",
  "ancestry": "Fixed Al64 e148b7ec215ba5e6d86fc57d21dac266bbd501f1e91320968266b5dbaeb8f44d; original roles and90 parent inputs preserved; seven additional rank-selected centers/frames per train source.",
  "evidence": [
    "${storage:repo}/configs/simulation/response_geometry_20261003.json",
    "${storage:repo}/docs/simulations/response_geometry_20261003.md"
  ],
  "limitations": [
    "Teacher MACE differs from parent MEAM physics.",
    "24A moving environments require new-target36A convergence gate; not full-cell validation.",
    "Prepared/queued collection; completeness determined by per-parent archive receipts.",
    "Reused original held-out ancestors; exploratory comparison."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["response_reuse_collection_v1"] |

## Evidence

All 765 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/response-geometry-unrelaxed-20261003-f185b1ec.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
