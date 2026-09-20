# Al · stopped 100,000-atom crystallization source

[All datasets](../README.md) · [Browsable card](al_meam_crystallization_100k_450K_20260911-8c8b56d9.html) · [Full metadata](../records/al_meam_crystallization_100k_450K_20260911-8c8b56d9.json)

Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.

- ID: `al_meam_crystallization_100k_450K_20260911`
- Materials: Al
- Classification: **review_required**; role: raw_dynamics
- Location: `/work/PERSO/vmorozov/simulations/al_meam_crystallization_100k_450K_20260911`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 23.898 GiB
- Missing metadata: ensemble

## Notes and relationships

```json
{
  "title": "Al \u00b7 stopped 100,000-atom crystallization source",
  "materials": [
    "Al"
  ],
  "role": "raw_dynamics",
  "classification": "review_required",
  "description": "Inspect current producer records and array headers; historical directory labels and planned counts are not proof of completion.",
  "evidence": [
    "${dataset:al_meam_crystallization_100k_450K_20260911}"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| pressure_bar | [0] |
| sample_interval_ps | [0.1] |
| protocol | ["al-source-then-branches"] |
| material | ["Al"] |
| temperature_K | [450] |
| timestep_ps | [0.001] |
| thermostat_ps | [0.1] |
| barostat_ps | [1.0] |
| mass_g_mol | [26.9815] |
| melt_temperature_K | [1325] |
| melt_seed | [911001] |
| atom_count | [100000] |
| ptm_rmsd_cutoff | [0.1] |
| branch_semantics | ["Position-conditioned NPT trajectories with fresh Maxwell-Boltzmann velocities; all six share one source lineage; not exact continuations."] |

## Evidence

All 3 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al_meam_crystallization_100k_450K_20260911-8c8b56d9.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-20T01:13:02.416719+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
