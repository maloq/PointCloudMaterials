# Raw Al/Mg/Ti/Ta local rich descriptors

[All datasets](../README.md) · [Browsable card](rich-descriptors-multimaterial-raw-20260929-5cf8ff14.html) · [Full metadata](../records/rich-descriptors-multimaterial-raw-20260929-5cf8ff14.json)

Derived 442 local geometry/bond-order/CNA/TDA targets and normalized nearest80 geometry from 13423868 raw dynamic training patches; static/relaxed views excluded. Original Al structural selection plus exact fixed Al64 calibration/test IDs. Scientific training uses a separately sealed compute-budget subset.

- ID: `rich-descriptors-multimaterial-raw-20260929`
- Materials: Al, Mg, Ta, Ti
- Classification: **prepared**; role: structural_pretraining
- Location: `/home/ids/vmorozov/training-cache/rich-descriptors/multimaterial-raw-20260929`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM; Mendelev 2008 Al EAM (Al1.eam.fs); Wilson–Mendelev 2016 Mg EAM (Mg1.eam.fs); Zhong 2014 Ta EAM; Kavousi 2019 Ni/Ti 2NN-MEAM, pure Ti component
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 31.556 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Raw Al/Mg/Ti/Ta local rich descriptors",
  "materials": [
    "Al",
    "Mg",
    "Ti",
    "Ta"
  ],
  "potential_ids": [
    "al-lee2003-meam",
    "al-mendelev2008-eam",
    "mg-wilson2016-eam",
    "ta-zhong2014-eam",
    "ti-kavousi2019-meam"
  ],
  "role": "structural_pretraining",
  "classification": "prepared",
  "description": "Derived 442 local geometry/bond-order/CNA/TDA targets and normalized nearest80 geometry from 13423868 raw dynamic training patches; static/relaxed views excluded. Original Al structural selection plus exact fixed Al64 calibration/test IDs. Scientific training uses a separately sealed compute-budget subset.",
  "provenance": "src/research/liquid_predictability/rich_multimaterial_data.py; existing immutable structural and Al64 releases.",
  "ancestry": "Original native Al train ancestors; external branches retain shared lineage metadata and remain train-only. No resplit.",
  "evidence": [
    "configs/liquid_predictability/rich_multimaterial_20260929.json",
    "docs/rich_multimaterial_encoder.md",
    "${storage:cache}/rich-descriptors/multimaterial-raw-20260929/plan.json"
  ],
  "limitations": [
    "All phases, not a liquid-only cohort.",
    "External materials have no independent validation/test sources in this release.",
    "442 local targets differ from the earlier 3536 context summaries."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| lineage | ["Al-archived-root", "Mg-archived-root", "Ta-archived-root", "Ti-archived-root", "al-1m-independent-melt-911001", "independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_135035943", "independent_melt_13937749", "independent_melt_139885636", "independent_melt_142641279"] … (155 values; see JSON) |
| material | ["Al", "Mg", "Ta", "Ti"] |
| split | ["selection", "train"] |
| protocol | ["fixed_material_cutoff_al_reference_v1", "rich_multimaterial_patch_descriptors_v1"] |

## Evidence

All 22411 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/rich-descriptors-multimaterial-raw-20260929-5cf8ff14.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-01T19:03:53.141383+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
