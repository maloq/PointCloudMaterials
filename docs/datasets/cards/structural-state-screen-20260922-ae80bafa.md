# Fixed structural-state matched observed/relaxed MACE screen

[All datasets](../README.md) · [Browsable card](structural-state-screen-20260922-ae80bafa.html) · [Full metadata](../records/structural-state-screen-20260922-ae80bafa.json)

Verified reuse of complete paired 8-Angstrom observed/relaxed neighborhoods: 45 independent roots, 2880 pairs, 25 fitting/5 tuning/15 reused development roots. Cached graphs and fixed geometry targets; independently cached original-MD order and sustained-onset labels were evaluation-only in v1/v2. The separate v3 distance/future study uses current and9ps future order from fitting roots for encoder supervision; onset remains evaluation-only. No new simulation.

- ID: `structural-state-screen-20260922`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/structural-state/screen-20260922`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.083 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Fixed structural-state matched observed/relaxed MACE screen",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Verified reuse of complete paired 8-Angstrom observed/relaxed neighborhoods: 45 independent roots, 2880 pairs, 25 fitting/5 tuning/15 reused development roots. Cached graphs and fixed geometry targets; independently cached original-MD order and sustained-onset labels were evaluation-only in v1/v2. The separate v3 distance/future study uses current and9ps future order from fitting roots for encoder supervision; onset remains evaluation-only. No new simulation.",
  "evidence": [
    "configs/structural_state/screen_20260922.json",
    "experiments/structural_state_20260922/README.md",
    "${storage:cache}/structural-state/screen-20260922/manifest.json",
    "configs/structural_state/future_metric_campaign_20260923.json",
    "experiments/structural_state_future_20260923/README.md"
  ],
  "limitations": [
    "Archived float16 full-cell coordinates; no weak-noise experiment. Full-cell relaxation contains context outside the local input. One-seed small mechanism cohort; only18 positive12ps development event windows. Historical cohorts are not an untouched final test."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| radius | [8.0] |
| radius_A | [8.0] |

## Evidence

All 1 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/structural-state-screen-20260922-ae80bafa.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
