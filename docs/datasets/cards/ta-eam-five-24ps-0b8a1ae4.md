# Ta: five 10-million-atom branches

[All datasets](../README.md) · [Browsable card](ta-eam-five-24ps-0b8a1ae4.html) · [Full metadata](../records/ta-eam-five-24ps-0b8a1ae4.json)

Five 10,000,422-atom NPT branches at 1900 K, 24 ps, 0.1 ps frames, initialized from archived positions.

- ID: `ta-eam-five-24ps`
- Materials: Ta
- Classification: **research**; role: raw_dynamics
- Location: `/work/PERSO/vmorozov/simulations/ti_ta_crystallization_20260907/Ta`
- Present on this machine: True
- Potentials: Zhong 2014 Ta EAM
- Complete binary records with arrays present: 5; these are not independent-source counts.
- Stored frames: 1205; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 79.439 GiB
- Missing metadata: ensemble

## Notes and relationships

```json
{
  "title": "Ta: five 10-million-atom branches",
  "materials": [
    "Ta"
  ],
  "role": "raw_dynamics",
  "classification": "research",
  "description": "Five 10,000,422-atom NPT branches at 1900 K, 24 ps, 0.1 ps frames, initialized from archived positions.",
  "potential_ids": [
    "ta-zhong2014-eam"
  ],
  "evidence": [
    "docs/simulations/ti_ta_crystallization/README.md",
    "${dataset:ta-eam-five-24ps}/config.json"
  ],
  "lineage": "Ti descendants share one root source; Ta snapshots are not independent melts. Retain source/parent grouping."
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| pressure_bar | [0] |
| sample_interval_ps | [0.1] |
| protocol | ["ta-position-branches"] |
| material | ["Ta"] |
| temperature_K | [1900] |
| timestep_ps | [0.002] |
| thermostat_ps | [0.2] |
| barostat_ps | [2.0] |
| mass_g_mol | [180.95] |
| atom_count | [10000422] |
| velocity_seed | [891071441, 891072441, 891073441, 891074441, 891075441] |
| frame_count | [241] |
| coordinate_convention | ["positions are wrapped Cartesian coordinates in angstrom relative to box_low in [0, box_high-box_low) before storage quantization; decode to float32 and wrap again", "positions are wrapped Cartesian coordinates in angstrom relative to box_low in the half-open periodic interval [0, box_high-box_low)"] |
| storage_dtype | ["float16", "float32"] |

## Evidence

All 21 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ta-eam-five-24ps-0b8a1ae4.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T12:07:34.550029+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
