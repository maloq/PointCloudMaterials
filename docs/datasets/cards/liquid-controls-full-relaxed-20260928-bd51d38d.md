# Full-coverage paired liquid geometry and labels

[All datasets](../README.md) · [Browsable card](liquid-controls-full-relaxed-20260928-bd51d38d.html) · [Full metadata](../records/liquid-controls-full-relaxed-20260928-bd51d38d.json)

All original eligible source/frame cells; reuse 976 existing and produce 2130 missing full periodic fixed-box FIRE quenches. Original MD and instantaneous relaxed labels are separately recorded.

- ID: `liquid-controls-full-relaxed-20260928`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/liquid-predictability/controls-full-relaxed-20260928`
- Present on this machine: False
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.000 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Full-coverage paired liquid geometry and labels",
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "training_cache",
  "classification": "research",
  "description": "All original eligible source/frame cells; reuse 976 existing and produce 2130 missing full periodic fixed-box FIRE quenches. Original MD and instantaneous relaxed labels are separately recorded.",
  "provenance": "control_relaxation.py and control_data.py; same original MD sources, new minimizations only",
  "ancestry": "Original fixed Al64 independent-melt source roles, unchanged; availability/common-phase conditioning exported.",
  "evidence": [
    "configs/simulation/liquid_full_relaxation_20260928.json",
    "docs/simulations/liquid_full_relaxation_20260928.md"
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

## Evidence

All 0 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/liquid-controls-full-relaxed-20260928-bd51d38d.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-01T19:03:53.141383+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
