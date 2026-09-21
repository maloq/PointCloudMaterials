# Al/Mg/Ti/Ta/Zr: 287,500 anchors with complete instantaneous TDA

[All datasets](../README.md) · [Browsable card](structural-neighbors-287500-full-tda-20260918-799b001d.html) · [Full metadata](../records/structural-neighbors-287500-full-tda-20260918-799b001d.json)

Immutable extension of the five-metal structural release: shooting anchors increase from 37,500 to 75,000, sampling every one of 456 training-eligible trajectories across 17 ancestral lineages. All supervised anchor/spatial/next-frame views have instantaneous TDA144. All parent observations, source splits and fixed material scales are retained; no new MD.

- ID: `structural-neighbors-287500-full-tda-20260918`
- Materials: Al, Mg, Ta, Ti, Zr
- Classification: **derived**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/structural_pretraining/broad-287500-full-tda-20260918`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM; Mendelev 2008 Al EAM (Al1.eam.fs); Wilson–Mendelev 2016 Mg EAM (Mg1.eam.fs); Zhong 2014 Ta EAM; Kavousi 2019 Ni/Ti 2NN-MEAM, pure Ti component
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 20.170 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Al/Mg/Ti/Ta/Zr: 287,500 anchors with complete instantaneous TDA",
  "materials": [
    "Al",
    "Mg",
    "Ti",
    "Ta",
    "Zr"
  ],
  "role": "training_cache",
  "classification": "derived",
  "description": "Immutable extension of the five-metal structural release: shooting anchors increase from 37,500 to 75,000, sampling every one of 456 training-eligible trajectories across 17 ancestral lineages. All supervised anchor/spatial/next-frame views have instantaneous TDA144. All parent observations, source splits and fixed material scales are retained; no new MD.",
  "evidence": [
    "${dataset:structural-neighbors-287500-full-tda-20260918}/plan.json",
    "${dataset:structural-neighbors-287500-full-tda-20260918}/manifest.json",
    "configs/shared_pretraining/broad_full_tda/data.json"
  ],
  "potential_ids": [
    "al-lee2003-meam",
    "al-mendelev2008-eam",
    "mg-wilson2016-eam",
    "ta-zhong2014-eam",
    "ti-kavousi2019-meam"
  ],
  "limitations": [
    "Static Al/Mg/Ta/Zr generating potentials are not established; their batches carry unknown-static provenance.",
    "Anchor records and shooting descendants are correlated; counts are not independent sources. Only the native Al cohort provides independent selection/test roles.",
    "No velocities or relaxed-TDA targets. Irregular saved-frame intervals retain their measured physical time offsets.",
    "The immutable plan contains the full native/shooting source lineage inventory beyond the top-level collection dependencies."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| seed | [1000095688, 1000756750, 100282052, 1005769362, 1006197591, 1007066282, 1008689840, 1009201367, 1009377897, 1011276288, 1012050372, 1012701400] … (1515 values; see JSON) |
| split | ["selection", "train"] |
| material | ["Al", "Mg", "Ta", "Ti", "Zr"] |
| lineage | ["Al-archived-root", "Mg-archived-root", "Ta-archived-root", "Ti-archived-root", "Zr-archived-root", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_002_velocity_35851", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_003_velocity_35863", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_000_velocity_35831", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_001_velocity_35839", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_002_velocity_35851", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_003_velocity_35863", "al_homogeneous_unseeded_2nn_meam_70304_500K_600ps_7velocityseeds_20260827/replicas/replica_000_velocity_35831"] … (127 values; see JSON) |
| timestep_fs | [1.0, 2.0, 3.0] |
| frame_count | [1, 113, 118, 135, 140, 149, 160, 161, 165, 17, 178, 188] … (55 values; see JSON) |
| protocol | ["structural_neighbors_full_tda_v1"] |

## Evidence

All 1517 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/structural-neighbors-287500-full-tda-20260918-799b001d.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T09:05:46.021728+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
