# Mg: six 24 ps EAM continuations

[All datasets](../README.md) · [Browsable card](mg-eam-six-24ps-ab3916d9.html) · [Full metadata](../records/mg-eam-six-24ps-ab3916d9.json)

Six 1,048,576-atom trajectories at 600 K, 0.1 ps frames; positions only.

- ID: `mg-eam-six-24ps`
- Materials: Mg
- Classification: **research**; role: raw_dynamics
- Location: `/work/PERSO/vmorozov/datasets/zr_al_mg_initial_6x24ps/branches/Mg`
- Present on this machine: True
- Potentials: Wilson–Mendelev 2016 Mg EAM (Mg1.eam.fs)
- Complete binary records with arrays present: 6; these are not independent-source counts.
- Stored frames: 1446; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 8.506 GiB
- Missing metadata: ensemble

## Notes and relationships

```json
{
  "title": "Mg: six 24 ps EAM continuations",
  "materials": [
    "Mg"
  ],
  "role": "raw_dynamics",
  "classification": "research",
  "description": "Six 1,048,576-atom trajectories at 600 K, 0.1 ps frames; positions only.",
  "potential_ids": [
    "mg-wilson2016-eam"
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
| material | ["Mg"] |
| atom_count | [1048576] |
| temperature_K | [600.0] |
| timestep_fs | [1.0] |
| duration_ps | [24.0] |
| sample_interval_ps | [0.1] |
| velocity_seed | [282514210, 324971279, 369006494, 668414018, 770041865, 81121196] |
| frame_count | [241] |
| coordinate_convention | ["positions decode to float32 wrapped Cartesian coordinates in angstrom relative to box_low in the half-open periodic interval [0, box_high-box_low)"] |
| storage_dtype | ["float16"] |

## Evidence

All 12 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/mg-eam-six-24ps-ab3916d9.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T12:07:34.550029+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
