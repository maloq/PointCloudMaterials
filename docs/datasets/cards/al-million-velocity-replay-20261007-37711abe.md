# Al million-atom 400 ps velocity recovery

[All datasets](../README.md) · [Browsable card](al-million-velocity-replay-20261007-37711abe.html) · [Full metadata](../records/al-million-velocity-replay-20261007-37711abe.json)

Checkpoint-anchored replay; consumers must require complete status and fidelity gates.

- ID: `al-million-velocity-replay-20261007`
- Materials: Al
- Classification: **building**; role: raw_dynamics
- Location: `/store/PERSO/vmorozov/simulations/al-million-velocity-replay-20261007`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 27; these are not independent-source counts.
- Stored frames: 1107; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 19.926 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "title": "Al million-atom 400 ps velocity recovery",
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "raw_dynamics",
  "classification": "building",
  "lineage": {
    "parent_dataset": "al_meam_1m_450K_400ps_with_melt_20260913T205405Z",
    "independent_ancestors_added": 0,
    "role": "train; same million-atom parent used in structural training"
  },
  "observation_ps": 0.1,
  "description": "Checkpoint-anchored replay; consumers must require complete status and fidelity gates.",
  "evidence": [
    "${dataset:al-million-velocity-replay-20261007}/status.json",
    "${dataset:al-million-velocity-replay-20261007}/replay_manifest.json"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| frame_count | [41] |
| storage_dtype | ["float16"] |
| atom_count | [1000000] |
| coordinate_convention | ["wrapped float32 consumer coordinates relative to box_low; decode stored coordinates to float32 and wrap again"] |

## Evidence

All 82 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-million-velocity-replay-20261007-37711abe.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
