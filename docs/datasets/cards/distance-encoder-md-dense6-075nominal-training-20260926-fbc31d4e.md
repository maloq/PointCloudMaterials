# Six-frame nominal 0.75-ps all-data distance training

[All datasets](../README.md) · [Browsable card](distance-encoder-md-dense6-075nominal-training-20260926-fbc31d4e.html) · [Full metadata](../records/distance-encoder-md-dense6-075nominal-training-20260926-fbc31d4e.json)

11,375,472 train and 190,080 selection windows. Native exact .75 ps; external exact .70 ps, explicitly accepted as nominal .75 ps. No interpolation.

- ID: `distance-encoder-md-dense6-075nominal-training-20260926`
- Materials: Al, Mg, Ta, Ti
- Classification: **retired**; role: provenance_only
- Location: `/home/ids/vmorozov/training-cache/distance-encoder/md-dense6-075nominal-training-20260926`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM; Mendelev 2008 Al EAM (Al1.eam.fs); Wilson–Mendelev 2016 Mg EAM (Mg1.eam.fs); Zhong 2014 Ta EAM; Kavousi 2019 Ni/Ti 2NN-MEAM, pure Ti component
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.004 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Six-frame nominal 0.75-ps all-data distance training",
  "materials": [
    "Al",
    "Mg",
    "Ti",
    "Ta"
  ],
  "role": "provenance_only",
  "classification": "retired",
  "description": "11,375,472 train and 190,080 selection windows. Native exact .75 ps; external exact .70 ps, explicitly accepted as nominal .75 ps. No interpolation.",
  "potential_ids": [
    "al-lee2003-meam",
    "al-mendelev2008-eam",
    "mg-wilson2016-eam",
    "ti-kavousi2019-meam",
    "ta-zhong2014-eam"
  ],
  "lineage": "Fixed native Al roles unchanged; all external ancestry stays train-only.",
  "evidence": [
    "configs/distance_encoder/md_dense6_075nominal_20260926.json",
    "docs/distance_encoder_history.md"
  ],
  "limitations": [
    "Nominal cadence is not a claim of identical physical intervals: actual native/external spans are 3.75/3.50 ps."
  ],
  "retirement_record": "docs/storage/cache_cleanup_20261002/receipt.json",
  "storage_status": "Cache arrays deleted at user request on 2026-10-02. Historical scientific description and provenance retained; rebuild before reuse.",
  "relaxation_source_availability_note": "Some historical relaxed-cell/neighborhood inputs were deleted at user request on 2026-10-03; retained downstream artifacts are unchanged. See docs/storage/relaxation_unverified_removal_20261003/receipt.json."
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| cadence_ps | [0.7, 0.75] |
| protocol | ["fixed_material_cutoff_al_reference_v1", "six_declared_cadence_md_observations_v3"] |
| material | ["Al", "Mg", "Ta", "Ti"] |
| split | ["selection", "train"] |
| lineage | ["Al-archived-root", "Mg-archived-root", "Ta-archived-root", "Ti-archived-root", "al-1m-independent-melt-911001", "independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_13937749", "independent_melt_142641279", "independent_melt_144370470", "independent_melt_146234076"] … (110 values; see JSON) |
| timestep_fs | [1.0, 2.0, 3.0] |
| frame_count | [2401, 241, 3001, 4001, 7241, 801] |
| atom_count | [100000, 1000000, 10000422, 1024000, 1048576, 70304] |

## Evidence

All 134 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/distance-encoder-md-dense6-075nominal-training-20260926-fbc31d4e.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
