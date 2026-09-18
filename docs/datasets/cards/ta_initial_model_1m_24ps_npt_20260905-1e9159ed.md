# Ta: 1,024,000-atom baseline, 24 ps

[All datasets](../README.md) · [Browsable card](ta_initial_model_1m_24ps_npt_20260905-1e9159ed.html) · [Full metadata](../records/ta_initial_model_1m_24ps_npt_20260905-1e9159ed.json)

One NPT trajectory from archived model_1m positions at 1900 K; fresh velocities.

- ID: `ta_initial_model_1m_24ps_npt_20260905`
- Materials: Ta
- Classification: **research**; role: raw_dynamics
- Location: `/work/PERSO/vmorozov/simulations/ta_initial_model_1m_24ps_npt_20260905`
- Present on this machine: True
- Potentials: Zhong 2014 Ta EAM
- Complete binary records with arrays present: 2; these are not independent-source counts.
- Stored frames: 252; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 1.846 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Ta: 1,024,000-atom baseline, 24 ps",
  "materials": [
    "Ta"
  ],
  "role": "raw_dynamics",
  "classification": "research",
  "description": "One NPT trajectory from archived model_1m positions at 1900 K; fresh velocities.",
  "potential_ids": [
    "ta-zhong2014-eam"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| material | ["Ta"] |
| atom_count | [1024000] |
| ensemble | ["isotropic Nose-Hoover NPT"] |
| temperature_K | [1900] |
| pressure_bar | [0] |
| duration_ps | [1, 24] |
| timestep_fs | [2] |
| sample_interval_ps | [0.1] |
| frame_count | [11, 241] |
| thermostat_damping_ps | [0.2] |
| barostat_damping_ps | [2.0] |
| velocity_seed | [891051440] |
| mass_g_mol | [180.95] |
| pair_style | ["eam/alloy"] |
| scientific_scope | ["One new position-conditioned trajectory; source velocities and Nose-Hoover state absent. Not an exact restart."] |
| storage_dtype | ["float16", "float32"] |
| coordinate_convention | ["positions are wrapped Cartesian coordinates in angstrom relative to box_low in [0, box_high-box_low) before storage quantization; decode to float32 and wrap again", "positions are wrapped Cartesian coordinates in angstrom relative to box_low in the half-open periodic interval [0, box_high-box_low)"] |

## Evidence

All 6 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ta_initial_model_1m_24ps_npt_20260905-1e9159ed.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-18T18:40:33.535128+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
