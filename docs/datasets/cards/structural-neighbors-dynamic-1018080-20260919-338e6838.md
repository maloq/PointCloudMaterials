# Fourfold dynamic structural expansion: 1,018,080 Al/Mg/Ti/Ta anchors

[All datasets](../README.md) · [Browsable card](structural-neighbors-dynamic-1018080-20260919-338e6838.html) · [Full metadata](../records/structural-neighbors-dynamic-1018080-20260919-338e6838.json)

Adds 763,560 dynamic anchors at previously unused source/frame combinations across all six training strata. Sampling balances ancestral lineages, then trajectories subject to unused-frame capacity. Retains all parent shards, the 480 selection anchors and fixed material scales. The dynamic-only training subset has 1,018,080 anchors; inherited static data (including Zr) is excluded by the training protocol. Instantaneous TDA144 covers every supervised anchor/spatial/future view; new shards use the local 8-normalized-unit support. No new simulations.

- ID: `structural-neighbors-dynamic-1018080-20260919`
- Materials: Al, Mg, Ta, Ti, Zr
- Classification: **derived**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/structural_pretraining/dynamic-1018080-full-tda-20260919`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM; Mendelev 2008 Al EAM (Al1.eam.fs); Wilson–Mendelev 2016 Mg EAM (Mg1.eam.fs); Zhong 2014 Ta EAM; Kavousi 2019 Ni/Ti 2NN-MEAM, pure Ti component
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 28.127 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Fourfold dynamic structural expansion: 1,018,080 Al/Mg/Ti/Ta anchors",
  "materials": [
    "Al",
    "Mg",
    "Ti",
    "Ta",
    "Zr"
  ],
  "role": "training_cache",
  "classification": "derived",
  "description": "Adds 763,560 dynamic anchors at previously unused source/frame combinations across all six training strata. Sampling balances ancestral lineages, then trajectories subject to unused-frame capacity. Retains all parent shards, the 480 selection anchors and fixed material scales. The dynamic-only training subset has 1,018,080 anchors; inherited static data (including Zr) is excluded by the training protocol. Instantaneous TDA144 covers every supervised anchor/spatial/future view; new shards use the local 8-normalized-unit support. No new simulations.",
  "evidence": [
    "${dataset:structural-neighbors-dynamic-1018080-20260919}/plan.json",
    "${dataset:structural-neighbors-dynamic-1018080-20260919}/manifest.json",
    "configs/shared_pretraining/mace_expanded_dual/data.json"
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
| seed | [1000007400, 1000095688, 100021905, 1000416918, 1000756750, 1000856622, 1000878560, 1001111905, 100115044, 1001685696, 1002514255, 1002779734] … (4892 values; see JSON) |
| split | ["selection", "train"] |
| material | ["Al", "Mg", "Ta", "Ti", "Zr"] |
| lineage | ["Al-archived-root", "Mg-archived-root", "Ta-archived-root", "Ti-archived-root", "Zr-archived-root", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_002_velocity_35851", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_003_velocity_35863", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_000_velocity_35831", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_001_velocity_35839", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_002_velocity_35851", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_003_velocity_35863", "al_homogeneous_unseeded_2nn_meam_70304_500K_600ps_7velocityseeds_20260827/replicas/replica_000_velocity_35831"] … (127 values; see JSON) |
| timestep_fs | [1.0, 2.0, 3.0] |
| frame_count | [1, 113, 118, 135, 140, 149, 160, 161, 165, 17, 178, 188] … (55 values; see JSON) |
| protocol | ["structural_neighbors_full_tda_v1"] |

## Evidence

All 4894 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/structural-neighbors-dynamic-1018080-20260919-338e6838.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T09:05:46.021728+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
