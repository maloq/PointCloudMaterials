# Full frozen liquid-cohort Al FIRE cells

[All datasets](../README.md) · [Browsable card](liquid-full-relaxation-20260928-592298fa.html) · [Full metadata](../records/liquid-full-relaxation-20260928-592298fa.json)

All original eligible source/frame cells; reuse 976 existing and produce 2130 missing full periodic fixed-box FIRE quenches. Original MD and instantaneous relaxed labels are separately recorded.

- ID: `liquid-full-relaxation-20260928`
- Materials: Al
- Classification: **research**; role: relaxed_cells
- Location: `/home/ids/vmorozov/training-cache/liquid-predictability/full-relaxation-20260928`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.001 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Full frozen liquid-cohort Al FIRE cells",
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "relaxed_cells",
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
| atom_count | [70304] |
| timestep_ps | [0.001] |
| protocol | ["Full periodic cell, fixed box, generating potential; infinity-norm force convergence, no isolated-patch relaxation."] |

## Evidence

All 246 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/liquid-full-relaxation-20260928-592298fa.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-28T23:47:41.345301+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
