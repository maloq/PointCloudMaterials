# al meam crystallization 1m 450K 20260913T203458Z

[All datasets](../README.md) · [Browsable card](al_meam_crystallization_1m_450K_20260913T203458Z-3d02e7ae.html) · [Full metadata](../records/al_meam_crystallization_1m_450K_20260913T203458Z-3d02e7ae.json)

MD payloads retired at user request on 2026-09-26. Historical plots, metrics, logs and provenance remain; no usable trajectory is retained here. See docs/simulations/cleanup_20260926/README.md.

- ID: `al_meam_crystallization_1m_450K_20260913T203458Z`
- Materials: Al
- Classification: **retired**; role: provenance_only
- Location: `/scratch/PERSO/vmorozov/PointCloudMaterials/simulations/al_meam_crystallization_1m_450K_20260913T203458Z`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.000 GiB
- Missing metadata: ensemble

## Notes and relationships

```json
{
  "title": "al meam crystallization 1m 450K 20260913T203458Z",
  "materials": [
    "Al"
  ],
  "role": "provenance_only",
  "classification": "retired",
  "description": "MD payloads retired at user request on 2026-09-26. Historical plots, metrics, logs and provenance remain; no usable trajectory is retained here. See docs/simulations/cleanup_20260926/README.md.",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "evidence": [
    "${dataset:al_meam_crystallization_1m_450K_20260913T203458Z}/status.json",
    "${dataset:al_meam_crystallization_1m_450K_20260913T203458Z}/retirement.json"
  ],
  "retirement_record": "docs/simulations/cleanup_20260926/receipt.json"
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
| branch_semantics | ["Position-conditioned NPT trajectories with fresh Maxwell-Boltzmann velocities; all six share one source lineage; not exact continuations."] |

## Evidence

All 2 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al_meam_crystallization_1m_450K_20260913T203458Z-3d02e7ae.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-02T12:18:52.028981+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
