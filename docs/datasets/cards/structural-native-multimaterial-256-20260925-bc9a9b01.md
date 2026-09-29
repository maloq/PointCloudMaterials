# Native structural pretraining: 256 native centers plus 1% external cells

[All datasets](../README.md) · [Browsable card](structural-native-multimaterial-256-20260925-bc9a9b01.html) · [Full metadata](../records/structural-native-multimaterial-256-20260925-bc9a9b01.json)

Raw sharded geometry in Angstrom with material/atomic-number audit metadata. The model loader applies fixed training-material length scales, anchored to Al; the encoder receives geometry only with one constant atom channel. Fixed Al64 train/selection ancestry; calibration/test excluded. Existing relaxed Al teachers retained.

- ID: `structural-native-multimaterial-256-20260925`
- Materials: Al, Mg, Ta, Ti, Zr
- Classification: **prepared**; role: structural_pretraining
- Location: `/scratch/PERSO/vmorozov/PointCloudMaterials/training-cache/structural-pretraining/multimaterial-256-20260925`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM; Mendelev 2008 Al EAM (Al1.eam.fs); Wilson–Mendelev 2016 Mg EAM (Mg1.eam.fs); Zhong 2014 Ta EAM; Kavousi 2019 Ni/Ti 2NN-MEAM, pure Ti component
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 14.668 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Native structural pretraining: 256 native centers plus 1% external cells",
  "materials": [
    "Al",
    "Mg",
    "Ti",
    "Ta",
    "Zr"
  ],
  "role": "structural_pretraining",
  "classification": "prepared",
  "description": "Raw sharded geometry in Angstrom with material/atomic-number audit metadata. The model loader applies fixed training-material length scales, anchored to Al; the encoder receives geometry only with one constant atom channel. Fixed Al64 train/selection ancestry; calibration/test excluded. Existing relaxed Al teachers retained.",
  "potential_ids": [
    "al-lee2003-meam",
    "al-mendelev2008-eam",
    "mg-wilson2016-eam",
    "ta-zhong2014-eam",
    "ti-kavousi2019-meam"
  ],
  "evidence": [
    "configs/structural_pretraining/multimaterial_256_20260925.json",
    "${dataset:structural-native-multimaterial-256-20260925}/plan.json",
    "${dataset:structural-native-multimaterial-256-20260925}/manifest.json"
  ],
  "lineage": "Inherits fixed Al64 90 train / 15 selection roles. Additional archived families are train-only; branches and static frames may share ancestry. Million-atom melt and measurement share one lineage.",
  "limitations": [
    "Static generating potentials remain unknown; exact dynamic potential IDs are retained per source.",
    "This is a structural expansion; all64/legacy16 crystallization evaluation rows remain unchanged."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| split | ["selection", "train"] |
| material | ["Al", "Mg", "Ta", "Ti", "Zr"] |
| lineage | ["Al-archived-root", "Mg-archived-root", "Ta-archived-root", "Ti-archived-root", "Zr-archived-root", "al-1m-independent-melt-911001", "independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_135035943", "independent_melt_13937749", "independent_melt_139885636"] … (156 values; see JSON) |
| protocol | ["native_multimaterial_structural_v1"] |
| radius_A | [8.0] |
| seed | [20260925] |
| timestep_fs | [1.0, 2.0, 3.0] |
| frame_count | [1, 2401, 241, 3001, 4001, 7241, 801] |
| atom_count | [100000, 1000000, 10000422, 1024000, 1048576, 70304] |

## Evidence

All 22400 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/structural-native-multimaterial-256-20260925-bc9a9b01.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-28T23:47:41.345301+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
