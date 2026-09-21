# Al/Mg/Ti/Ta/Zr structural neighbors: 250,000 training anchors

[All datasets](../README.md) · [Browsable card](structural-neighbors-250k-20260917-caca88b3.html) · [Full metadata](../records/structural-neighbors-250k-20260917-caca88b3.json)

Frozen five-metal positions-only structural pretraining release: 250,000 training anchor records and 480 Al selection records. Dynamic observations include three causal frames, a separate next-frame target and a nearby-center view. Per-material fixed cutoff normalization, geometry85, and masked instantaneous TDA144. Plan records every native/shooting ancestry, source manifest, target producer, calibrated radius and extraction task.

- ID: `structural-neighbors-250k-20260917`
- Materials: Al, Mg, Ta, Ti, Zr
- Classification: **derived**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/structural_pretraining/broad-250k-v2-20260917`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM; Mendelev 2008 Al EAM (Al1.eam.fs); Wilson–Mendelev 2016 Mg EAM (Mg1.eam.fs); Zhong 2014 Ta EAM; Kavousi 2019 Ni/Ti 2NN-MEAM, pure Ti component
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 16.673 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Al/Mg/Ti/Ta/Zr structural neighbors: 250,000 training anchors",
  "materials": [
    "Al",
    "Mg",
    "Ti",
    "Ta",
    "Zr"
  ],
  "role": "training_cache",
  "classification": "derived",
  "description": "Frozen five-metal positions-only structural pretraining release: 250,000 training anchor records and 480 Al selection records. Dynamic observations include three causal frames, a separate next-frame target and a nearby-center view. Per-material fixed cutoff normalization, geometry85, and masked instantaneous TDA144. Plan records every native/shooting ancestry, source manifest, target producer, calibrated radius and extraction task.",
  "evidence": [
    "${dataset:structural-neighbors-250k-20260917}/manifest.json",
    "${dataset:structural-neighbors-250k-20260917}/plan.json",
    "experiments/structural_pretraining_20260917/README.md"
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
| seed | [1000095688, 1000756750, 100282052, 1005769362, 1006197591, 1007066282, 1009201367, 1009377897, 1011276288, 1014781531, 1018243358, 1023116288] … (1011 values; see JSON) |
| split | ["selection", "train"] |
| material | ["Al", "Mg", "Ta", "Ti", "Zr"] |
| lineage | ["Al-archived-root", "Mg-archived-root", "Ta-archived-root", "Ti-archived-root", "Zr-archived-root", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_002_velocity_35851", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_003_velocity_35863", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_000_velocity_35831", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_001_velocity_35839", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_002_velocity_35851", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_003_velocity_35863", "al_homogeneous_unseeded_2nn_meam_70304_500K_600ps_7velocityseeds_20260827/replicas/replica_000_velocity_35831"] … (127 values; see JSON) |
| timestep_fs | [1.0, 2.0, 3.0] |
| frame_count | [1, 113, 118, 135, 140, 149, 160, 161, 165, 17, 178, 188] … (55 values; see JSON) |
| protocol | ["structural_neighbors_v1"] |

## Evidence

All 1013 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/structural-neighbors-250k-20260917-caca88b3.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T16:17:14.292955+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
