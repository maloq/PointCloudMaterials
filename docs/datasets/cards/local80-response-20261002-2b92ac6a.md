# Local80 response training from fixed Al64 MD parents

[All datasets](../README.md) · [Browsable card](local80-response-20261002-2b92ac6a.html) · [Full metadata](../records/local80-response-20261002-2b92ac6a.json)

New MACE-MPA-0 moving-environment 20/100fs response queries from original MEAM parent positions. Inherited90/15/30 non-calibration source roles; one predetermined center each. Local80 geometry-only students. Gated nested environments18/24/30 versus36A.

- ID: `local80-response-20261002`
- Materials: Al
- Classification: **research**; role: local_response_training
- Location: `/store/PERSO/vmorozov/simulations/local80-response-20261002`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM; MACE-MPA-0 medium checkpoint
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.084 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "title": "Local80 response training from fixed Al64 MD parents",
  "materials": [
    "Al"
  ],
  "role": "local_response_training",
  "classification": "research",
  "potential_ids": [
    "mace-mpa-0-medium",
    "al-lee2003-meam"
  ],
  "description": "New MACE-MPA-0 moving-environment 20/100fs response queries from original MEAM parent positions. Inherited90/15/30 non-calibration source roles; one predetermined center each. Local80 geometry-only students. Gated nested environments18/24/30 versus36A.",
  "ancestry": "Fixed Al64 identity e148b7ec215ba5e6d86fc57d21dac266bbd501f1e91320968266b5dbaeb8f44d; original independent melt ancestry retained; no source resplit.",
  "evidence": [
    "${storage:repo}/configs/simulation/local_response_20261002.json",
    "${storage:repo}/docs/simulations/local_response_20261002.md"
  ],
  "limitations": [
    "MLIP teacher differs from parent MEAM potential.",
    "Nested-environment agreement is not exact full-cell validation.",
    "Separate response-query pilot, not all64 crystallization-window evaluation.",
    "Parent float16 quantization retained; new response integration/perturbations float32."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["response_reuse_collection_v1"] |

## Evidence

All 5 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/local80-response-20261002-2b92ac6a.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-02T12:18:52.028981+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
