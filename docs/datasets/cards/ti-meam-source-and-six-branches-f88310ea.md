# Ti: source and six 240 ps branches

[All datasets](../README.md) · [Browsable card](ti-meam-source-and-six-branches-f88310ea.html) · [Full metadata](../records/ti-meam-source-and-six-branches-f88310ea.json)

Pure Ti under the Ti component of the Ni/Ti potential, 100,000 atoms, 1250 K. All six branches descend from one source.

- ID: `ti-meam-source-and-six-branches`
- Materials: Ti
- Classification: **research**; role: raw_dynamics
- Location: `/work/PERSO/vmorozov/simulations/ti_ta_crystallization_20260907/Ti`
- Present on this machine: True
- Potentials: Kavousi 2019 Ni/Ti 2NN-MEAM, pure Ti component
- Complete binary records with arrays present: 7; these are not independent-source counts.
- Stored frames: 21647; duplicate-group records: 6
- Allocated storage, excluding registered nested datasets: 14.018 GiB
- Missing metadata: ensemble

## Notes and relationships

```json
{
  "title": "Ti: source and six 240 ps branches",
  "materials": [
    "Ti"
  ],
  "role": "raw_dynamics",
  "classification": "research",
  "description": "Pure Ti under the Ti component of the Ni/Ti potential, 100,000 atoms, 1250 K. All six branches descend from one source.",
  "potential_ids": [
    "ti-kavousi2019-meam"
  ],
  "evidence": [
    "docs/simulations/ti_ta_crystallization/README.md",
    "${dataset:ti-meam-source-and-six-branches}/config.json"
  ],
  "lineage": "Ti descendants share one root source; Ta snapshots are not independent melts. Retain source/parent grouping."
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| pressure_bar | [0] |
| sample_interval_ps | [0.1] |
| protocol | ["continuous melt-quench source", "ti-source-then-branches"] |
| material | ["Ti"] |
| temperature_K | [1250] |
| timestep_ps | [0.001] |
| thermostat_ps | [0.1] |
| barostat_ps | [1.0] |
| mass_g_mol | [47.88] |
| melt_temperature_K | [3000] |
| melt_seed | [917001] |
| atom_count | [100000] |
| ptm_rmsd_cutoff | [0.17] |
| branch_semantics | ["Position-conditioned NPT trajectories with fresh Maxwell-Boltzmann velocities; all six share one source lineage; not exact continuations."] |
| frame_count | [2401, 7241] |
| velocity_seed | [917101, 917103, 917107, 917111, 917117, 917123] |
| root_lineage | ["/home/ids/vmorozov/simulations/ti_ta_crystallization_20260907/Ti/source"] |
| coordinate_convention | ["positions are wrapped Cartesian coordinates in angstrom relative to box_low in [0, box_high-box_low) before storage quantization; decode to float32 and wrap again"] |
| storage_dtype | ["float16"] |

## Evidence

All 25 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ti-meam-source-and-six-branches-f88310ea.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-21T16:17:14.292955+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
