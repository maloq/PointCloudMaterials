# Al: six 24 ps EAM continuations

[All datasets](../README.md) · [Browsable card](al-eam-six-24ps-c12da440.html) · [Full metadata](../records/al-eam-six-24ps-c12da440.json)

Six 1,048,576-atom trajectories at 650 K, 0.1 ps frames; positions only.

- ID: `al-eam-six-24ps`
- Materials: Al
- Classification: **research**; role: raw_dynamics
- Location: `/work/PERSO/vmorozov/datasets/zr_al_mg_initial_6x24ps/branches/Al`
- Present on this machine: True
- Potentials: Mendelev 2008 Al EAM (Al1.eam.fs)
- Complete binary records with arrays present: 6; these are not independent-source counts.
- Stored frames: 1446; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 8.506 GiB
- Missing metadata: ensemble

## Notes and relationships

```json
{
  "title": "Al: six 24 ps EAM continuations",
  "materials": [
    "Al"
  ],
  "role": "raw_dynamics",
  "classification": "research",
  "description": "Six 1,048,576-atom trajectories at 650 K, 0.1 ps frames; positions only.",
  "potential_ids": [
    "al-mendelev2008-eam"
  ],
  "evidence": [
    "${dataset:zr_al_mg_initial_6x24ps}/manifest.json"
  ],
  "lineage": "Shared archived source; position-conditioned, not exact restarts."
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| material | ["Al"] |
| atom_count | [1048576] |
| temperature_K | [650.0] |
| timestep_fs | [1.0] |
| duration_ps | [24.0] |
| sample_interval_ps | [0.1] |
| velocity_seed | [176387760, 602978450, 699107707, 728143657, 829122842, 891021440] |
| frame_count | [241] |
| coordinate_convention | ["positions decode to float32 wrapped Cartesian coordinates in angstrom relative to box_low in the half-open periodic interval [0, box_high-box_low)"] |
| storage_dtype | ["float16"] |

## Evidence

All 12 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-eam-six-24ps-c12da440.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-24T10:36:51.012083+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
