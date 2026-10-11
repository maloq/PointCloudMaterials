# Force-only local80 Al response teacher comparison

[All datasets](../README.md) · [Browsable card](response-force-engines-20261002-b8c9952f.html) · [Full metadata](../records/response-force-engines-20261002-b8c9952f.json)

Train-only four-parent, two-seed, 18/24/36A moving-open-environment BAOAB20/100fs comparison. New UNEP/MEAM float64 CPU finite differences; original MACE float32 GPU AD references retained. Saves future-feature/response tensors and exact shared noise, not trajectory positions.

- ID: `response-force-engines-20261002`
- Materials: Al
- Classification: **research**; role: response_engine_benchmark
- Location: `/store/PERSO/vmorozov/simulations/response-force-engines-20261002`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM; MACE-MPA-0 medium checkpoint; UNEP-v1 main (2024 publication model, with ZBL)
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.276 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "title": "Force-only local80 Al response teacher comparison",
  "materials": [
    "Al"
  ],
  "role": "response_engine_benchmark",
  "classification": "research",
  "potential_ids": [
    "unep-v1-main-2024",
    "al-lee2003-meam",
    "mace-mpa-0-medium"
  ],
  "description": "Train-only four-parent, two-seed, 18/24/36A moving-open-environment BAOAB20/100fs comparison. New UNEP/MEAM float64 CPU finite differences; original MACE float32 GPU AD references retained. Saves future-feature/response tensors and exact shared noise, not trajectory positions.",
  "ancestry": "Parents0,1,2,45 from local80-response-20261002; fixed Al64 e148b7ec215ba5e6d86fc57d21dac266bbd501f1e91320968266b5dbaeb8f44d. Training ancestors only; no new independent MD roots.",
  "evidence": [
    "${storage:repo}/configs/simulation/response_force_engines_20261002.json",
    "${storage:repo}/docs/simulations/response_force_engines_20261002.md"
  ],
  "limitations": [
    "New potentials define different future laws; neither is MACE-equivalent.",
    "Nested open-environment checks do not establish full-periodic equivalence.",
    "CPU versus cached GPU timings use different resources; no inference of end-to-end encoder training speedup."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 0 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/response-force-engines-20261002-b8c9952f.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T00:11:05.385980+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
