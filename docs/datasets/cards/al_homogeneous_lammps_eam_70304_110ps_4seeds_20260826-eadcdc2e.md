# al homogeneous lammps eam 70304 110ps 4seeds 20260826

[All datasets](../README.md) · [Browsable card](al_homogeneous_lammps_eam_70304_110ps_4seeds_20260826-eadcdc2e.html) · [Full metadata](../records/al_homogeneous_lammps_eam_70304_110ps_4seeds_20260826-eadcdc2e.json)

Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.

- ID: `al_homogeneous_lammps_eam_70304_110ps_4seeds_20260826`
- Materials: Al
- Classification: **review_required**; role: raw_dynamics
- Location: `/store/PERSO/vmorozov/projects/PointCloudMaterials-20260913T174741Z/output/synthetic_data/al_homogeneous_lammps_eam_70304_110ps_4seeds_20260826`
- Present on this machine: True
- Potentials: Mishin 1999 Al EAM (Al99.eam.alloy)
- Complete binary records with arrays present: 4; these are not independent-source counts.
- Stored frames: 464; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 1.004 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "al homogeneous lammps eam 70304 110ps 4seeds 20260826",
  "materials": [
    "Al"
  ],
  "role": "raw_dynamics",
  "classification": "review_required",
  "description": "Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.",
  "evidence": [
    "${dataset:al_homogeneous_lammps_eam_70304_110ps_4seeds_20260826}"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| atom_count | [70304] |
| pair_style | ["eam/alloy/omp via the LAMMPS OPENMP suffix"] |
| ensemble | ["LAMMPS Nose-Hoover NPT"] |
| temperature_K | [500.0] |
| timestep_fs | [1.0] |
| coordinate_convention | ["positions are wrapped Cartesian coordinates in angstrom relative to box_low in the half-open periodic interval [0, box_high-box_low)"] |
| frame_count | [116] |
| storage_dtype | ["float32"] |

## Evidence

All 6 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al_homogeneous_lammps_eam_70304_110ps_4seeds_20260826-eadcdc2e.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T09:05:46.021728+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
