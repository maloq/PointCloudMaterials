# Liquid known/null signals and paired archived relaxed observations

[All datasets](../README.md) · [Browsable card](liquid-controls-relaxed-20260928-aa7d8e5a.html) · [Full metadata](../records/liquid-controls-relaxed-20260928-aa7d8e5a.json)

Seven generated-label controls on unchanged raw observations; exact archive intersection with paired raw/relaxed inputs and original/relaxed cluster distances. Synthetic labels are not physical distances.

- ID: `liquid-controls-relaxed-20260928`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/liquid-predictability/controls-relaxed-20260928`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 2.912 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Liquid known/null signals and paired archived relaxed observations",
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "training_cache",
  "classification": "research",
  "description": "Seven generated-label controls on unchanged raw observations; exact archive intersection with paired raw/relaxed inputs and original/relaxed cluster distances. Synthetic labels are not physical distances.",
  "provenance": "control_data.py, existing verified full-cell FIRE quenches; no MD/minimization launched",
  "ancestry": "Original fixed Al64 independent-melt source roles, unchanged; availability/common-phase conditioning exported.",
  "evidence": [
    "configs/liquid_predictability/controls_relaxed_20260928.json",
    "docs/liquid_controls.md"
  ],
  "limitations": [
    "Full-cell relaxation incorporates external context.",
    "Archived cold coordinates are float16; extraction is float32.",
    "New labels use instantaneous PTM clusters, not sustained MD establishment.",
    "One seed and one synthetic label realization."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| ptm_rmsd_cutoff | [0.1] |

## Evidence

All 1133 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/liquid-controls-relaxed-20260928-aa7d8e5a.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-28T23:47:41.345301+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
