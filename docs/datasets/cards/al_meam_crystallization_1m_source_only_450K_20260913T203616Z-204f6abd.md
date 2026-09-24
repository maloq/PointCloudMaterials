# al meam crystallization 1m source only 450K 20260913T203616Z

[All datasets](../README.md) · [Browsable card](al_meam_crystallization_1m_source_only_450K_20260913T203616Z-204f6abd.html) · [Full metadata](../records/al_meam_crystallization_1m_source_only_450K_20260913T203616Z-204f6abd.json)

Retained earlier million-atom preparation/attempt. Inspect current status and arrays; do not substitute requested duration for completed data.

- ID: `al_meam_crystallization_1m_source_only_450K_20260913T203616Z`
- Materials: Al
- Classification: **incomplete_or_rejected**; role: raw_dynamics
- Location: `/scratch/PERSO/vmorozov/PointCloudMaterials/simulations/al_meam_crystallization_1m_source_only_450K_20260913T203616Z`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.000 GiB
- Missing metadata: ensemble

## Notes and relationships

```json
{
  "title": "al meam crystallization 1m source only 450K 20260913T203616Z",
  "materials": [
    "Al"
  ],
  "role": "raw_dynamics",
  "classification": "incomplete_or_rejected",
  "description": "Retained earlier million-atom preparation/attempt. Inspect current status and arrays; do not substitute requested duration for completed data.",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "evidence": [
    "${dataset:al_meam_crystallization_1m_source_only_450K_20260913T203616Z}/status.json"
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
| atom_count | [1000000] |
| ptm_rmsd_cutoff | [0.1] |
| branch_semantics | ["No branches: one continuous liquid-to-crystal source only."] |

## Evidence

All 2 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al_meam_crystallization_1m_source_only_450K_20260913T203616Z-204f6abd.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-24T10:36:51.012083+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
