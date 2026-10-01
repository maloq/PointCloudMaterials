# Fixed 274-feature Al480 shooting predictive targets

[All datasets](../README.md) · [Browsable card](predictive-baseline-al480-274-20261001-37ba0228.html) · [Full metadata](../records/predictive-baseline-al480-274-20261001-37ba0228.json)

Training-source-frozen 9 first moments, 9 second moments and 256 RFFs of local crystal fraction/qbar4/qbar6 at 3/6/12 ps, with exact historical rows, weights and source roles.

- ID: `predictive-baseline-al480-274-20261001`
- Materials: Al
- Classification: **research**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/predictive-baseline/al480-274-20261001`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.129 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Fixed 274-feature Al480 shooting predictive targets",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "research",
  "description": "Training-source-frozen 9 first moments, 9 second moments and 256 RFFs of local crystal fraction/qbar4/qbar6 at 3/6/12 ps, with exact historical rows, weights and source roles.",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "evidence": [
    "docs/predictive_baseline.md",
    "${dataset:predictive-baseline-al480-274-20261001}/manifest.json"
  ],
  "lineage": "All 7661 local observations share 40 parents and 480 Langevin branches from 20 source trajectories; no new independent source, condition pooling recorded."
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["local_shooting_predictive_representation_v1"] |
| horizons_ps | [[3, 6, 12]] |
| seed | [20261001] |
| target_columns | [["crystalline_fraction@3ps", "bond_order/l4_qbar@3ps", "bond_order/l6_qbar@3ps", "crystalline_fraction@6ps", "bond_order/l4_qbar@6ps", "bond_order/l6_qbar@6ps", "crystalline_fraction@12ps", "bond_order/l4_qbar@12ps", "bond_order/l6_qbar@12ps"]] |
| parent_id | ["parent_000_T400_v35831_pre_nucleation_12ps", "parent_001_T400_v35831_pre_nucleation_3ps", "parent_002_T400_v35839_pre_nucleation_12ps", "parent_003_T400_v35839_pre_nucleation_3ps", "parent_004_T400_v35851_pre_nucleation_12ps", "parent_005_T400_v35851_pre_nucleation_3ps", "parent_006_T400_v35863_pre_nucleation_12ps", "parent_007_T400_v35863_pre_nucleation_3ps", "parent_008_T400_v35869_pre_nucleation_12ps", "parent_009_T400_v35869_pre_nucleation_3ps", "parent_010_T400_v35879_pre_nucleation_12ps", "parent_011_T400_v35879_pre_nucleation_3ps"] … (40 values; see JSON) |
| source_run_id | ["al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_000_velocity_35831", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_001_velocity_35839", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_002_velocity_35851", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_003_velocity_35863", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_004_velocity_35869", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_400K/replicas/replica_005_velocity_35879", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_000_velocity_35831", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_001_velocity_35839", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_002_velocity_35851", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_003_velocity_35863", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_004_velocity_35869", "al_homogeneous_unseeded_2nn_meam_70304_4temps_400-600K_600ps_6seeds_20260828/temperature_450K/replicas/replica_005_velocity_35879"] … (20 values; see JSON) |
| source_split | ["train", "validation"] |
| temperature_K | [400.0, 450.0, 500.0] |

## Evidence

All 1 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/predictive-baseline-al480-274-20261001-37ba0228.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-01T19:03:53.141383+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
