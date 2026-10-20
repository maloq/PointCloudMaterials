# al-main-010ps-20260927-source0896

[All datasets](../README.md) · [Browsable card](al-main-010ps-20260927-source0896-12ee4fcc.html) · [Full metadata](../records/al-main-010ps-20260927-source0896-12ee4fcc.json)

Exact 0.1 ps sampled descendant of retained Al prepared liquid; 2 fs integration.

- ID: `al-main-010ps-20260927-source0896`
- Materials: Al
- Classification: **research**; role: raw_dynamics
- Location: `/store/PERSO/vmorozov/simulations/al-main-010ps-20260927-source0896`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 1; these are not independent-source counts.
- Stored frames: 3361; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 3.587 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "raw_dynamics",
  "classification": "research",
  "ancestry": "independent_melt_146234076",
  "split": "train",
  "source_id": 896,
  "sampling_ps": 0.1,
  "protocol": "al_dense_half_stop_010ps_v1",
  "description": "Exact 0.1 ps sampled descendant of retained Al prepared liquid; 2 fs integration.",
  "evidence": [
    "/store/PERSO/vmorozov/simulations/al-main-010ps-20260927-source0896/input_metadata.json"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["al_dense_half_stop_010ps_v1"] |
| material | ["Al"] |
| atom_count | [70304] |
| timestep_ps | [0.002] |
| ensemble | ["NPT"] |
| thermostat_ps | [0.3] |
| barostat_ps | [3.0] |
| pressure_bar | [0.0] |
| root_lineage | ["independent_melt_146234076"] |
| split | ["train"] |
| temperature_K | [400.0] |
| preparation_seed | [146234076] |
| velocity_seed | [307931848] |
| frame_count | [3361] |
| storage_dtype | ["float16"] |
| coordinate_convention | ["positions are wrapped float32 consumer coordinates relative to box_low in the half-open periodic interval [0, box_high-box_low)"] |
| source_run_id | ["al-main-010ps-20260927-source0896"] |

## Evidence

All 5 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-main-010ps-20260927-source0896-12ee4fcc.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
