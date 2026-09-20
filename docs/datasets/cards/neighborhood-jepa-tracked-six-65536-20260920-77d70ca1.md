# Tracked six-neighbor space-time JEPA: 65,536 mixed-material training anchors

[All datasets](../README.md) · [Browsable card](neighborhood-jepa-tracked-six-65536-20260920-77d70ca1.html) · [Full metadata](../records/neighborhood-jepa-tracked-six-65536-20260920-77d70ca1.json)

Current-frame angular-coverage neighbor selection; same seven atom identities at previous/current/next saved frames. Local radius-8 complete graphs. Current/next center physical85 and instantaneous TDA144 are inherited from the full-TDA parent. 480 native-Al selection anchors remain held out. No static data, velocities, relaxed targets or new simulations.

- ID: `neighborhood-jepa-tracked-six-65536-20260920`
- Materials: Al, Mg, Ta, Ti
- Classification: **derived**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/neighborhood_jepa/tracked-six-65536-20260920`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM; Mendelev 2008 Al EAM (Al1.eam.fs); Wilson–Mendelev 2016 Mg EAM (Mg1.eam.fs); Zhong 2014 Ta EAM; Kavousi 2019 Ni/Ti 2NN-MEAM, pure Ti component
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 18.054 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Tracked six-neighbor space-time JEPA: 65,536 mixed-material training anchors",
  "materials": [
    "Al",
    "Mg",
    "Ti",
    "Ta"
  ],
  "role": "training_cache",
  "classification": "derived",
  "description": "Current-frame angular-coverage neighbor selection; same seven atom identities at previous/current/next saved frames. Local radius-8 complete graphs. Current/next center physical85 and instantaneous TDA144 are inherited from the full-TDA parent. 480 native-Al selection anchors remain held out. No static data, velocities, relaxed targets or new simulations.",
  "potential_ids": [
    "al-lee2003-meam",
    "al-mendelev2008-eam",
    "mg-wilson2016-eam",
    "ta-zhong2014-eam",
    "ti-kavousi2019-meam"
  ],
  "evidence": [
    "${dataset:neighborhood-jepa-tracked-six-65536-20260920}/plan.json",
    "${dataset:neighborhood-jepa-tracked-six-65536-20260920}/manifest.json",
    "configs/neighborhood_jepa/data_20260920.json"
  ],
  "limitations": [
    "Selection is native Al only; anchor and shooting counts are not independent trajectories.",
    "Physical time gaps depend on the recorded source cadence.",
    "Only center current/next views have fixed physical/TDA labels; neighbor targets are learned equivariant/invariant embeddings.",
    "Six neighbors provide approximate angular coverage, not an exact canonical cubic frame."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| seed | [1003525202, 1006197591, 100896184, 1009700132, 1016944708, 1017590640, 1018243358, 101956480, 1022281357, 1022348430, 1027852775, 103209158] … (735 values; see JSON) |
| lineage | ["Al-archived-root", "Mg-archived-root", "Ta-archived-root", "Ti-archived-root", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_002_velocity_35851", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_003_velocity_35863", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_000_velocity_35831", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_001_velocity_35839", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_002_velocity_35851", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_003_velocity_35863", "al_homogeneous_unseeded_2nn_meam_70304_500K_600ps_7velocityseeds_20260827/replicas/replica_000_velocity_35831", "al_homogeneous_unseeded_2nn_meam_70304_500K_600ps_7velocityseeds_20260827/replicas/replica_001_velocity_35839"] … (71 values; see JSON) |
| split | ["selection", "train"] |
| material | ["Al", "Mg", "Ta", "Ti"] |
| timestep_fs | [1.0, 2.0, 3.0] |
| frame_count | [113, 118, 135, 140, 161, 165, 17, 178, 2401, 241, 26, 29] … (32 values; see JSON) |

## Evidence

All 370 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/neighborhood-jepa-tracked-six-65536-20260920-77d70ca1.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-20T01:13:02.416719+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
