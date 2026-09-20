# Al · predictive-memory partial-observation cache

[All datasets](../README.md) · [Browsable card](predictive-memory-al-r17-20260917-91f24edf.html) · [Full metadata](../records/predictive-memory-al-r17-20260917-91f24edf.json)

Derived inputs or targets. Source lineages and target protocol remain part of the dataset definition.

- ID: `predictive-memory-al-r17-20260917`
- Materials: Al
- Classification: **derived**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/predictive-memory-al-r17-20260917`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 1.772 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Al \u00b7 predictive-memory partial-observation cache",
  "materials": [
    "Al"
  ],
  "role": "training_cache",
  "classification": "derived",
  "description": "Derived inputs or targets. Source lineages and target protocol remain part of the dataset definition.",
  "evidence": [
    "${dataset:predictive-memory-al-r17-20260917}"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["predictive_memory_partial_observation_v1"] |
| trajectory_id | [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 860, 861] … (150 values; see JSON) |
| root_lineage | ["independent_melt_114745743", "independent_melt_129223029", "independent_melt_134462729", "independent_melt_135035943", "independent_melt_13937749", "independent_melt_139885636", "independent_melt_142641279", "independent_melt_144370470", "independent_melt_146234076", "independent_melt_147978527", "independent_melt_151871197", "independent_melt_15458297"] … (150 values; see JSON) |
| parent_trajectory_id | [null] |
| split | ["test", "train", "val"] |
| material | ["Al"] |
| atomic_numbers | [[13]] |
| temperature_K | [400.0, 450.0, 500.0, 510.0, 520.0] |
| ensemble | ["NPT"] |
| thermostat | ["Nose-Hoover"] |
| thermostat_damping_ps | [0.3] |
| barostat | ["isotropic Nose-Hoover"] |
| barostat_damping_ps | [3.0] |
| pressure_bar | [0.0] |
| units | ["metal"] |
| timestep_ps | [0.003] |
| melt_seed | [114745743, 129223029, 134462729, 135035943, 13937749, 139885636, 142641279, 144370470, 146234076, 147978527, 151871197, 15458297] … (150 values; see JSON) |
| melt_duration_ps | [300.0] |
| equilibration_duration_ps | [15.0] |
| simulator_version | ["LAMMPS (22 Jul 2025 - Update 4)"] |
| coordinate_convention | ["wrapped relative to box_low; orthorhombic box_low/high arrays hashed above"] |
| future_lags_ps | [[0.75, 3.0, 12.0, 48.0, 96.0]] |
| radius_A | [17.0] |
| cutoff_A | [5.0] |
| seed | [20260917] |
| position_dtype | ["float16"] |
| velocity_dtype | ["float16"] |

## Evidence

All 5 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/predictive-memory-al-r17-20260917-91f24edf.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-20T01:13:02.416719+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
