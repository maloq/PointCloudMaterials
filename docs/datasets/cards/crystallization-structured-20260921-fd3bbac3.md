# Symmetric 25-query MACE and historical GATr context timelines

[All datasets](../README.md) · [Browsable card](crystallization-structured-20260921-fd3bbac3.html) · [Full metadata](../records/crystallization-structured-20260921-fd3bbac3.json)

150 existing independent Al trajectories, original 90/15/15/30 lineage split, 16 tracked centers each. Center plus 12 cuboctahedral queries per 10 and 20 Angstrom shell; unique nearest real atom assignments, exact offsets, 3 ps grid. Frozen original transfer MACE and exact GATr VICReg step-3072 features; own-backbone future center targets. Per-source hashed completion receipts. No new simulations.

- ID: `crystallization-structured-20260921`
- Materials: Al
- Classification: **derived**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/crystallization-structured-20260921`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 10.120 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Symmetric 25-query MACE and historical GATr context timelines",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "derived",
  "description": "150 existing independent Al trajectories, original 90/15/15/30 lineage split, 16 tracked centers each. Center plus 12 cuboctahedral queries per 10 and 20 Angstrom shell; unique nearest real atom assignments, exact offsets, 3 ps grid. Frozen original transfer MACE and exact GATr VICReg step-3072 features; own-backbone future center targets. Per-source hashed completion receipts. No new simulations.",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "evidence": [
    "configs/crystallization_transfer/symmetric_mace_gatr_20260921.json",
    "output/crystallization_transfer/symmetric-mace-gatr-20260921/technical/plan.json",
    "output/crystallization_transfer/symmetric-mace-gatr-20260921/technical/encoder-provenance.json"
  ],
  "limitations": [
    "Queries are symmetric; real assigned atoms may deviate by up to 4 Angstrom and may change identity between frames.",
    "Frozen box-frame stencil has cubic symmetry, not arbitrary rotation invariance under resampling.",
    "Different original local encoder supports: MACE 7.94 Angstrom, GATr 16.87 Angstrom.",
    "Overlapping windows are not independent samples; physical targets remain original-MD targets.",
    "Collection may be partial until all 150 completion receipts exist."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 150 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/crystallization-structured-20260921-fd3bbac3.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T16:17:14.292955+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
