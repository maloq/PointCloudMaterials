# Ti/Ta dynamics and recovery records

[All datasets](../README.md) · [Browsable card](ti_ta_crystallization_20260907-2bcfd479.html) · [Full metadata](../records/ti_ta_crystallization_20260907-2bcfd479.json)

Parent container for separately registered Ti source/branches, duplicate exports, additional shooting and Ta branches.

- ID: `ti_ta_crystallization_20260907`
- Materials: Al, Mg, Ta, Ti, Zr
- Classification: **mixed**; role: container
- Location: `/work/PERSO/vmorozov/simulations/ti_ta_crystallization_20260907`
- Present on this machine: True
- Potentials: Mendelev 2008 Al EAM (Al1.eam.fs); Wilson–Mendelev 2016 Mg EAM (Mg1.eam.fs); Zhong 2014 Ta EAM; Kavousi 2019 Ni/Ti 2NN-MEAM, pure Ti component
- Complete binary records with arrays present: 7; these are not independent-source counts.
- Stored frames: 25; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 3.296 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Ti/Ta dynamics and recovery records",
  "materials": [
    "Ti",
    "Ta"
  ],
  "role": "container",
  "classification": "mixed",
  "description": "Parent container for separately registered Ti source/branches, duplicate exports, additional shooting and Ta branches.",
  "evidence": [
    "docs/simulations/ti_ta_crystallization/README.md"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| pressure_bar | [0, 0.0] |
| sample_interval_ps | [0.1] |
| protocol | ["continuous melt-quench source", "ta-position-branches", "ti-source-then-branches"] |
| material | ["Al", "Mg", "Ta", "Ti", "Zr"] |
| temperature_K | [1250, 1250.0, 1900] |
| timestep_ps | [0.001, 0.002] |
| thermostat_ps | [0.1, 0.2] |
| barostat_ps | [1.0, 2.0] |
| mass_g_mol | [180.95, 47.88] |
| atom_count | [10000422, 1024000, 128] |
| velocity_seed | [189980460, 293653075, 533319719, 674707733, 80245850, 889339747, 891071441, 891072441, 891073441, 891074441, 891075441, 917101] … (17 values; see JSON) |
| melt_temperature_K | [3000] |
| melt_seed | [917001] |
| ptm_rmsd_cutoff | [0.15, 0.17] |
| branch_semantics | ["Position-conditioned NPT trajectories with fresh Maxwell-Boltzmann velocities; all six share one source lineage; not exact continuations."] |
| frame_count | [241, 3, 7] |
| root_lineage | ["/home/ids/vmorozov/simulations/ti_ta_crystallization_20260907/preflight/workflow_v2/source"] |
| coordinate_convention | ["positions are wrapped Cartesian coordinates in angstrom relative to box_low in the half-open periodic interval [0, box_high-box_low)"] |
| storage_dtype | ["float32"] |
| material_order | [["Al", "Mg", "Zr"]] |
| ensemble | ["isotropic Nose-Hoover NPT"] |
| duration_ps | [24.0] |
| material_temperature_K | [{"Al": 650.0, "Mg": 600.0, "Zr": 1250.0}] |
| material_timestep_fs | [{"Al": 1.0, "Mg": 1.0, "Zr": 2.0}] |
| scientific_scope | ["New position-conditioned paths from all six archived thermal coordinate snapshots per material. Source velocities and Nose-Hoover state are absent; these are not exact restarts."] |
| timestep_fs | [2.0] |

## Evidence

All 38 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ti_ta_crystallization_20260907-2bcfd479.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T12:07:34.550029+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
