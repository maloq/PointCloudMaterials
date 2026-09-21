# Al/Mg 24 ps parent collection

[All datasets](../README.md) · [Browsable card](zr_al_mg_initial_6x24ps-f64244b6.html) · [Full metadata](../records/zr_al_mg_initial_6x24ps-f64244b6.json)

Twelve retained NPT continuations: six Al and six Mg. Historical directory name includes Zr, whose reconstructed-potential trajectories were removed.

- ID: `zr_al_mg_initial_6x24ps`
- Materials: Al, Mg
- Classification: **research**; role: container
- Location: `/work/PERSO/vmorozov/datasets/zr_al_mg_initial_6x24ps`
- Present on this machine: True
- Potentials: Mendelev 2008 Al EAM (Al1.eam.fs); Wilson–Mendelev 2016 Mg EAM (Mg1.eam.fs)
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.001 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Al/Mg 24 ps parent collection",
  "materials": [
    "Al",
    "Mg"
  ],
  "role": "container",
  "classification": "research",
  "description": "Twelve retained NPT continuations: six Al and six Mg. Historical directory name includes Zr, whose reconstructed-potential trajectories were removed.",
  "evidence": [
    "${dataset:zr_al_mg_initial_6x24ps}/manifest.json"
  ],
  "lineage": "Position-conditioned branches from archived thermal snapshots; no source velocities or thermostat state; not exact restarts.",
  "removed_materials": [
    "Zr"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| material_order | [["Al", "Mg"]] |
| ensemble | ["isotropic Nose-Hoover NPT"] |
| pressure_bar | [0.0] |
| duration_ps | [24.0] |
| sample_interval_ps | [0.1] |
| material_temperature_K | [{"Al": 650.0, "Mg": 600.0}] |
| material_timestep_fs | [{"Al": 1.0, "Mg": 1.0}] |
| scientific_scope | ["New position-conditioned paths from all six archived thermal coordinate snapshots per material. Source velocities and Nose-Hoover state are absent; these are not exact restarts."] |
| material | ["Al", "Mg"] |

## Evidence

All 1 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/zr_al_mg_initial_6x24ps-f64244b6.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T16:17:14.292955+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
