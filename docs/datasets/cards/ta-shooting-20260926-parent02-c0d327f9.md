# Ta shooting parent 2.8ns: 4/4 shots complete

[All datasets](../README.md) · [Browsable card](ta-shooting-20260926-parent02-c0d327f9.html) · [Full metadata](../records/ta-shooting-20260926-parent02-c0d327f9.json)



- ID: `ta-shooting-20260926-parent02`
- Materials: Ta
- Classification: **research**; role: raw_dynamics
- Location: `/store/PERSO/vmorozov/simulations/ta-shooting-20260926-parent02`
- Present on this machine: True
- Potentials: Zhong 2014 Ta EAM
- Complete binary records with arrays present: 4; these are not independent-source counts.
- Stored frames: 964; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 90.148 GiB
- Missing metadata: ensemble

## Notes and relationships

```json
{
  "materials": [
    "Ta"
  ],
  "potential_ids": [
    "ta-zhong2014-eam"
  ],
  "role": "raw_dynamics",
  "classification": "research",
  "title": "Ta shooting parent 2.8ns: 4/4 shots complete",
  "ancestry": "archived-Ta-unknown-common-preparation; all parents and descendants conservatively grouped",
  "parent": {
    "name": "2.8ns",
    "source": "/work/PERSO/vmorozov/datasets/Ta/initial_configurations/2.8ns.pos",
    "source_sha256": "7d20d67091eedf2f24a27978a5bc2e33c1755ed12f3ecd37ccfcfafff6b2376a",
    "source_step": 1400000,
    "atom_count": 10000422
  },
  "evidence": [
    "/store/PERSO/vmorozov/simulation-launches/ta-shooting-20260926/manifest.json",
    "/store/PERSO/vmorozov/simulations/ta-shooting-20260926-parent02/status.json"
  ],
  "limitations": [
    "Position-conditioned shots share an archived parent; original generating potential is unknown."
  ]
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
| velocity_seed | [241723527, 254269500, 47085415, 852288974] |
| parent_id | ["2.8ns"] |
| frame_count | [241] |
| coordinate_convention | ["positions are wrapped Cartesian coordinates in angstrom relative to box_low in [0, box_high-box_low) before storage quantization; decode to float32 and wrap again"] |
| storage_dtype | ["float16"] |

## Evidence

All 14 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ta-shooting-20260926-parent02-c0d327f9.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-02T12:18:52.028981+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
