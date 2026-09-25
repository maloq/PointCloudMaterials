# BCR paired full-radius observed/relaxed Al structural audit

[All datasets](../README.md) · [Browsable card](bcr-relaxed-audit-20260922-092baf53.html) · [Full metadata](../records/bcr-relaxed-audit-20260922-092baf53.json)

Derived from existing Lee2003 MEAM trajectories and fixed-box FIRE relaxed full cells; no new simulation. Seeded 45 roots at 400/450/500/510/520 K, 4 frames and 16 paired tracked centers per root. Complete 8 A patches independently extracted in each domain; 25 fitting/5 tuning/15 development roots, no historical test/calibration sources.

- ID: `bcr-relaxed-audit-20260922`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/bcr/relaxed-audit-20260922`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.009 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "BCR paired full-radius observed/relaxed Al structural audit",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Derived from existing Lee2003 MEAM trajectories and fixed-box FIRE relaxed full cells; no new simulation. Seeded 45 roots at 400/450/500/510/520 K, 4 frames and 16 paired tracked centers per root. Complete 8 A patches independently extracted in each domain; 25 fitting/5 tuning/15 development roots, no historical test/calibration sources.",
  "evidence": [
    "configs/bcr/followup_20260922/study.json",
    "output/bcr/conditioning-audit-20260922/technical/relaxed-selection.json",
    "experiments/bcr_followup_20260922/README.md"
  ],
  "limitations": [
    "Planned derived cache; completion is recorded in its manifest and per-cell receipts. Archived full-box float16 positions, float32 boxes and exact identities; used only for structural probes, not weak-noise denoising. Neighbor membership can differ across domains. Full-cell quenches contain external context."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| split | ["development", "fit", "tune"] |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |
| radius_A | [8.0] |

## Evidence

All 181 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/bcr-relaxed-audit-20260922-092baf53.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
