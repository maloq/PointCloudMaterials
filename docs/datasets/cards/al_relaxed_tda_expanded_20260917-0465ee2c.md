# Al: expanded relaxed/instantaneous TDA pairs

[All datasets](../README.md) · [Browsable card](al_relaxed_tda_expanded_20260917-0465ee2c.html) · [Full metadata](../records/al_relaxed_tda_expanded_20260917-0465ee2c.json)

Observed nearest-80 atom IDs retained after full-cell minimization; raw 144D TDA and centered float32 clouds. Respect training_eligible and shared ancestry.

- ID: `al_relaxed_tda_expanded_20260917`
- Materials: Al
- Classification: **building**; role: training_targets
- Location: `/home/ids/vmorozov/training-cache/relaxed_tda/al-expanded-20260917`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.142 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Al: expanded relaxed/instantaneous TDA pairs",
  "materials": [
    "Al"
  ],
  "role": "training_targets",
  "classification": "building",
  "description": "Observed nearest-80 atom IDs retained after full-cell minimization; raw 144D TDA and centered float32 clouds. Respect training_eligible and shared ancestry.",
  "evidence": [
    "docs/relaxed_tda_targets.md",
    "output/relaxed_tda/al-expanded-20260917/technical/plan.json"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| atom_count | [70304] |
| timestep_ps | [0.001] |
| protocol | ["Full periodic cell, fixed box, generating potential; infinity-norm force convergence, no isolated-patch relaxation.", "Full periodic fixed-cell FIRE; observed nearest-80 identities retained after minimization; raw 144D persistence image."] |

## Evidence

All 780 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al_relaxed_tda_expanded_20260917-0465ee2c.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-20T01:13:02.416719+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
