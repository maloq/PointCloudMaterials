# Six-frame exact 0.10-ps external distance study

[All datasets](../README.md) · [Browsable card](distance-encoder-md-dense6-010ps-training-20260926-55f1f221.html) · [Full metadata](../records/distance-encoder-md-dense6-010ps-training-20260926-55f1f221.json)

3,374,496 train, 377,496 selection and 3,061,560 test windows, all at .10-ps intervals and .50-ps span; no interpolation.

- ID: `distance-encoder-md-dense6-010ps-training-20260926`
- Materials: Al, Mg, Ta, Ti
- Classification: **derived**; role: training_geometry
- Location: `/home/ids/vmorozov/training-cache/distance-encoder/md-dense6-010ps-training-20260926`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM; Mendelev 2008 Al EAM (Al1.eam.fs); Wilson–Mendelev 2016 Mg EAM (Mg1.eam.fs); Zhong 2014 Ta EAM; Kavousi 2019 Ni/Ti 2NN-MEAM, pure Ti component
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 26.214 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Six-frame exact 0.10-ps external distance study",
  "materials": [
    "Al",
    "Mg",
    "Ti",
    "Ta"
  ],
  "role": "training_geometry",
  "classification": "derived",
  "description": "3,374,496 train, 377,496 selection and 3,061,560 test windows, all at .10-ps intervals and .50-ps span; no interpolation.",
  "potential_ids": [
    "al-lee2003-meam",
    "al-mendelev2008-eam",
    "mg-wilson2016-eam",
    "ti-kavousi2019-meam",
    "ta-zhong2014-eam"
  ],
  "lineage": "Train Al-million/Mg/Ti families; selection Al archive; test Ta archive. Entire ancestry groups stay together. This is a separate external assay, not a change to Al64 roles.",
  "evidence": [
    "configs/distance_encoder/md_dense6_010ps_20260926.json",
    "docs/metrics/distance_encoder_dense_external.md"
  ],
  "limitations": [
    "Selection/test each have one ancestry family; test is also transfer to unseen Ta.",
    "Not directly comparable to fixed-Al spatial-path warning metrics."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| cadence_ps | [0.1] |
| protocol | ["fixed_material_cutoff_al_reference_v1", "six_declared_cadence_md_observations_v3"] |
| material | ["Al", "Mg", "Ta", "Ti"] |
| lineage | ["Al-archived-root", "Mg-archived-root", "Ta-archived-root", "Ti-archived-root", "al-1m-independent-melt-911001"] |
| split | ["train"] |
| timestep_fs | [1.0, 2.0] |
| frame_count | [2401, 241, 3001, 4001, 7241] |
| atom_count | [100000, 1000000, 10000422, 1024000, 1048576] |

## Evidence

All 29 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/distance-encoder-md-dense6-010ps-training-20260926-55f1f221.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-02T12:18:52.028981+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
