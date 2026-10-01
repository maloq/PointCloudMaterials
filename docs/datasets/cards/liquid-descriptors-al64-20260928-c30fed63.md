# Rich TDA, bond-order, CNA and geometry features for liquid distance prediction

[All datasets](../README.md) · [Browsable card](liquid-descriptors-al64-20260928-c30fed63.html) · [Full metadata](../records/liquid-descriptors-al64-20260928-c30fed63.json)

442 fixed patch features aggregated into 3536 invariant context features. Same 25 observed radius-8 patches, source roles and eligible rows as the sealed liquid-predictability assay.

- ID: `liquid-descriptors-al64-20260928`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/liquid-predictability/descriptors-al64-20260928`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 21.594 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Rich TDA, bond-order, CNA and geometry features for liquid distance prediction",
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "training_cache",
  "classification": "research",
  "description": "442 fixed patch features aggregated into 3536 invariant context features. Same 25 observed radius-8 patches, source roles and eligible rows as the sealed liquid-predictability assay.",
  "provenance": "src/research/liquid_predictability/descriptors.py and descriptor_data.py; no new simulations",
  "ancestry": "Inherited independent-melt roles unchanged; correlated extra atom centers retain their source.",
  "evidence": [
    "configs/liquid_predictability/descriptors_al64_20260928.json",
    "${storage:cache}/liquid-predictability/descriptors-al64-20260928/manifest.json",
    "docs/liquid_descriptors.md"
  ],
  "limitations": [
    "Conditional on a crystal existing outside consumed input atoms.",
    "CNA uses consumed geometry only; not stored PTM labels.",
    "Fixed test sources have informed earlier research."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 151 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/liquid-descriptors-al64-20260928-c30fed63.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-01T19:03:53.141383+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
