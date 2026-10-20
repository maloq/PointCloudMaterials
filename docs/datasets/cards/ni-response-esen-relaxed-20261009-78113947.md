# Matched relaxed Ni encoder observations of original MD

[All datasets](../README.md) · [Browsable card](ni-response-esen-relaxed-20261009-78113947.html) · [Full metadata](../records/ni-response-esen-relaxed-20261009-78113947.json)

Original Ni physical states, labels, rows and ancestry; quenched encoder observations and target geometries. Finite-amplitude paired simulator responses through actual whole-cell quenches.

- ID: `ni-response-esen-relaxed-20261009`
- Materials: Ni
- Classification: **research**; role: motion_training_and_evaluation
- Location: `/home/ids/vmorozov/training-cache/ni-response/relaxed-esen-original-md-20261009-v1`
- Present on this machine: True
- Potentials: Mendelev et al. 2012 Ni EAM/FS
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.557 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Matched relaxed Ni encoder observations of original MD",
  "materials": [
    "Ni"
  ],
  "potential_ids": [
    "ni-mendelev2012-eam"
  ],
  "role": "motion_training_and_evaluation",
  "classification": "research",
  "description": "Original Ni physical states, labels, rows and ancestry; quenched encoder observations and target geometries. Finite-amplitude paired simulator responses through actual whole-cell quenches.",
  "evidence": [
    "${storage:training_storage}/ni_response/relaxed-esen-original-md-20261009-v1/technical/plan.json",
    "docs/ni_relaxed_response.md"
  ],
  "limitations": [
    "The geometric velocity probe at a minimum is not the time derivative of the quenched trajectory.",
    "Finite-amplitude responses through minimization are not analytic infinitesimal derivatives.",
    "Quench numerical roundoff may change basin assignment."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 5 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ni-response-esen-relaxed-20261009-78113947.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
