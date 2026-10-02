# Al64 transient crystal episodes and established-nucleus fates

[All datasets](../README.md) · [Browsable card](al64-nucleus-fates-v1-20261002-e3a0dfe6.html) · [Full metadata](../records/al64-nucleus-fates-v1-20261002-e3a0dfe6.json)

Original full-cell PTM/lineage fate audit; all 150 sources, new transient episode inventory and additional labels for unchanged original/relaxed birth rows. No new simulations.

- ID: `al64-nucleus-fates-v1-20261002`
- Materials: Al
- Classification: **derived**; role: label_audit
- Location: `/store/PERSO/vmorozov/experiments/nucleus_fates/al64-20261002`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.163 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Al64 transient crystal episodes and established-nucleus fates",
  "materials": [
    "Al"
  ],
  "role": "label_audit",
  "classification": "derived",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "Original full-cell PTM/lineage fate audit; all 150 sources, new transient episode inventory and additional labels for unchanged original/relaxed birth rows. No new simulations.",
  "provenance": "src/research/crystallization_origin/fates.py; original 20260925 full-cell PTM audit and 20260930 birth cohort",
  "lineage": "Same frozen Al64 melt/source ancestors and roles. No independent new trajectories; within-source episodes and event centers are correlated.",
  "evidence": [
    "configs/analysis/nucleus_fates_20261002.json",
    "docs/metrics/nucleus_fates.md",
    "${dataset:al64-nucleus-fates-v1-20261002}/technical/prepared.json",
    "${dataset:al64-nucleus-fates-v1-20261002}/technical/complete.json"
  ],
  "limitations": [
    "Operational PTM crystal episodes, not validated critical nuclei.",
    "Finite observation, coarse 0.75 ps cadence, conservative exclusion of any establishment-connected component.",
    "Four-to-seven-atom episodes inventoried but not atom-verified. New transient candidates lack classifier input-eligibility screening.",
    "Merging, residual crystal and right-censoring are explicit; no forced binary fate for uncertain events."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["al64-nucleus-fates-v1"] |
| cadence_ps | [0.75] |

## Evidence

All 127 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al64-nucleus-fates-v1-20261002-e3a0dfe6.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-02T12:18:52.028981+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
