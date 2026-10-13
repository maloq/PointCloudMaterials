# Spatial pretraining with Ni crystal interiors/interfaces and merged evaluation

[All datasets](../README.md) · [Browsable card](spatial-training-ni20-f686dc18.html) · [Full metadata](../records/spatial-training-ni20-f686dc18.json)

1,373,798 fitting neighborhoods, including 317,030 Ni (23.08%). Adds 52,838 Ni training observations: half crystal interiors, half interfaces. Validation/test merged into 248,081 evaluation rows, including 9,830 Ni after 1,638 phase-selected additions. 198 calculated targets, 197 active.

- ID: `spatial-training-ni20`
- Materials: Al, Mg, Ni, Ta, Ti
- Classification: **prepared**; role: structural_pretraining
- Location: `/home/ids/vmorozov/training-cache/spatial-pretraining/ni-enriched-eval-20261008`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM; Mendelev 2008 Al EAM (Al1.eam.fs); Wilson–Mendelev 2016 Mg EAM (Mg1.eam.fs); Mendelev et al. 2012 Ni EAM/FS; Zhong 2014 Ta EAM; Kavousi 2019 Ni/Ti 2NN-MEAM, pure Ti component
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 4.181 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Spatial pretraining with Ni crystal interiors/interfaces and merged evaluation",
  "materials": [
    "Al",
    "Mg",
    "Ti",
    "Ta",
    "Ni"
  ],
  "potential_ids": [
    "al-lee2003-meam",
    "al-mendelev2008-eam",
    "mg-wilson2016-eam",
    "ta-zhong2014-eam",
    "ti-kavousi2019-meam",
    "ni-mendelev2012-eam"
  ],
  "role": "structural_pretraining",
  "classification": "prepared",
  "description": "1,373,798 fitting neighborhoods, including 317,030 Ni (23.08%). Adds 52,838 Ni training observations: half crystal interiors, half interfaces. Validation/test merged into 248,081 evaluation rows, including 9,830 Ni after 1,638 phase-selected additions. 198 calculated targets, 197 active.",
  "provenance": "configs/data/spatial_ni_enriched.json; src/postprocessing/spatial_enrichment.py. Original prepared observations retained; full-cell periodic PTM supplies explicit structural sampling categories only.",
  "ancestry": "Original source ancestry retained: Ni 19 training and 34 held-out ancestors. Additions use 12 train and 21 held-out sources. Original selection/test roles are audit metadata within merged evaluation; no training/evaluation ancestry overlap.",
  "identity": "0325195ca0b4e060f6a41390823265b58f65267a3ae130148ffff28c4b688162",
  "normalization": {
    "protocol": "fixed_material_cutoff_al_reference_v1",
    "catalog_sha256": "9843ef97869feda393f174990db8123a958b7ea99aee7eeccfefa3b7ed995684",
    "scales_A": {
      "Mg": 10.148997617371739,
      "Al": 9.121389139452193,
      "Ti": 9.309864688636168,
      "Zr": 10.353008987883314,
      "Ta": 9.38733713271385,
      "Ni": 8.227955921292462
    },
    "reference_scale_A": 9.121389139452193,
    "model_inputs": [
      "normalized coordinates"
    ],
    "material_metadata": "used only for fixed length normalization and audit; never an encoder/head input",
    "ni_rule": "0.995 quantile of nearest-160 distances * 1.02; 4000 training points across up to 8 training sources"
  },
  "evidence": [
    "configs/data/spatial_ni_enriched.json",
    "docs/datasets/spatial_ni_enriched.md",
    "/home/ids/vmorozov/training-cache/spatial-pretraining/ni-enriched-eval-20261008/plan.json",
    "/home/ids/vmorozov/training-cache/spatial-pretraining/ni-enriched-eval-20261008/complete.json",
    "/home/infres/vmorozov/PointCloudMaterials/output/spatial-dataset/ni-enriched-eval-20261008/technical/final-audit.json"
  ],
  "limitations": [
    "Explicit phase-enriched spatial sample; not representative of MD phase occupancy.",
    "Rows share source and atom ancestry; not independent replicas.",
    "Only Al and Ni have held-out sources.",
    "Separate historical Al calibration rows are preserved outside pretraining evaluation.",
    "Compact target names remain pinned to the prior training schema; tda/n32_h2_image20 is now masked by the declared training SD threshold.",
    "No encoder training has been launched using this release."
  ],
  "packed_identity": "6a86e9102ceda0409b527a9058e1a0073f1f1e799ce2ccd3652fa05f34ee65b0",
  "completed_at": "2026-10-08T14:03:29.379705+00:00",
  "consumer": "/home/ids/vmorozov/training-cache/spatial-pretraining/ni-enriched-eval-20261008/consumer.json",
  "target_columns": 198,
  "active_columns": 197,
  "evaluation_role": "evaluation",
  "counts": {
    "train/Al": 593617,
    "train/Mg": 44369,
    "train/Ti": 57284,
    "train/Ta": 361498,
    "train/Ni": 317030,
    "evaluation/Al": 238251,
    "calibration/Al": 16848,
    "evaluation/Ni": 9830
  },
  "parent_prepared_identity": "e830847334cf10d2772fab37697a2390c441aaaa0339dfdb2637ee46125dad2c"
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| lineage | ["Al-archived-root", "Mg-archived-root", "Ta-archived-root", "Ti-archived-root", "al-1m-independent-melt-911001", "independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_135035943", "independent_melt_13937749", "independent_melt_139885636", "independent_melt_142641279"] … (208 values; see JSON) |
| material | ["Al", "Mg", "Ni", "Ta", "Ti"] |
| split | ["selection", "test", "train"] |
| protocol | ["fixed_material_cutoff_al_reference_v1", "spatial-ni-crystal-interface-merged-evaluation-v1"] |
| seed | [202610081] |
| ptm_rmsd_cutoff | [0.1] |
| timestep_fs | [1.0] |
| frame_count | [1151, 1171, 1181, 1191, 1241, 1331, 1351, 1421, 1471, 1481, 1541, 1601] … (31 values; see JSON) |
| preparation_seed | [261007402, 261007806, 261008210, 261008614, 261008701, 261008802, 261008903, 261009004, 261009018, 261009105, 261009307, 261009408] … (33 values; see JSON) |
| temperature_K | [1000, 1050, 1100, 1125, 1150, 950] |
| atom_count | [32000] |

## Evidence

All 797 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/spatial-training-ni20-f686dc18.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
