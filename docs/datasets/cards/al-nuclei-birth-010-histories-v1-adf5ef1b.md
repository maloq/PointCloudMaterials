# Al nucleus births and crystalline precursor tails

[All datasets](../README.md) · [Browsable card](al-nuclei-birth-010-histories-v1-adf5ef1b.html) · [Full metadata](../records/al-nuclei-birth-010-histories-v1-adf5ef1b.json)

Same 295 PTM established-birth records; original roles and ancestry retained. Adds retrospective 12 ps precursor graphs, original embryo growth traces, atom excerpts and many-to-many links to saved first-crystal observations. No extra nucleus events or new simulations. Geometry-only sources without velocities remain excluded from motion training.

- ID: `al-nuclei-birth-010-histories-v1`
- Materials: Al
- Classification: **derived**; role: birth_histories
- Location: `/store/PERSO/vmorozov/experiments/nuclei_birth/al/histories/birth-tails-010-v1`
- Present on this machine: False
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.000 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Al nucleus births and crystalline precursor tails",
  "materials": [
    "Al"
  ],
  "role": "birth_histories",
  "classification": "derived",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Same 295 PTM established-birth records; original roles and ancestry retained. Adds retrospective 12 ps precursor graphs, original embryo growth traces, atom excerpts and many-to-many links to saved first-crystal observations. No extra nucleus events or new simulations. Geometry-only sources without velocities remain excluded from motion training.",
  "provenance": "${dataset:al-nuclei-birth-010-histories-v1}/manifest.json",
  "lineage": "110 pinned 0.1 ps source views; physical onset alias groups are explicit, not additional independent nuclei. Generating-potential unknowns remain in original source provenance.",
  "evidence": [
    "docs/datasets/nuclei_birth_histories.md",
    "configs/data/nuclei_birth_histories_20261007.json"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 0 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-nuclei-birth-010-histories-v1-adf5ef1b.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
