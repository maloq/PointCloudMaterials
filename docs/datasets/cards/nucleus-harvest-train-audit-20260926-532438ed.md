# Al regional-emergence training-source harvest audit

[All datasets](../README.md) · [Browsable card](nucleus-harvest-train-audit-20260926-532438ed.html) · [Full metadata](../records/nucleus-harvest-train-audit-20260926-532438ed.json)

Exploratory atom-origin references, causal region eligibility, event/control outcomes, precursor bond order and visual review. Only the original 90 Al training ancestors; no fitting, simulation or held-out export.

- ID: `nucleus-harvest-train-audit-20260926`
- Materials: Al
- Classification: **derived**; role: label_availability_audit
- Location: `/work/PERSO/vmorozov/analysis/nucleus_harvest/train-audit-20260926`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.033 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Al regional-emergence training-source harvest audit",
  "materials": [
    "Al"
  ],
  "role": "label_availability_audit",
  "classification": "derived",
  "description": "Exploratory atom-origin references, causal region eligibility, event/control outcomes, precursor bond order and visual review. Only the original 90 Al training ancestors; no fitting, simulation or held-out export.",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "lineage": "Inherits exact fixed Al64 train source IDs and independent melt ancestry; references and event crops within a source are correlated.",
  "evidence": [
    "configs/analysis/nucleus_harvest_20260926.json",
    "${dataset:nucleus-harvest-train-audit-20260926}/technical/plan.json",
    "${dataset:nucleus-harvest-train-audit-20260926}/technical/state.json",
    "docs/metrics/nucleus_harvest.md"
  ],
  "limitations": [
    "Operational establishment candidates, not validated critical nuclei.",
    "Enriched event references and overlapping uniform controls are not a finalized probability-training population.",
    "Future-centered review crops are for inspection only, never model inputs."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["al64-crystallization-origin-audit-v2", "nucleus-harvest-training-audit-v1"] |
| seed | [20260926] |
| horizons_ps | [[3.0, 6.0]] |
| history_ps | [0.0, 12.0, 3.0, 6.0, [0.0, 3.0, 6.0, 12.0]] |
| lineage | ["independent_melt_114745743", "independent_melt_134462729", "independent_melt_13937749", "independent_melt_142641279", "independent_melt_144370470", "independent_melt_146234076", "independent_melt_147978527", "independent_melt_151871197", "independent_melt_15458297", "independent_melt_160380922", "independent_melt_184384186", "independent_melt_200290231"] … (91 values; see JSON) |
| cadence_ps | [0.75] |
| frame_count | [801] |
| atom_count | [70304] |
| timestep_fs | [3.0] |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |

## Evidence

All 92 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/nucleus-harvest-train-audit-20260926-532438ed.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-28T23:47:41.345301+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
