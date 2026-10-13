# Differentiable Lee2003 Al MEAM local response benchmark

[All datasets](../README.md) · [Browsable card](response-meam-torch-20261004-7180fd3f.html) · [Full metadata](../records/response-meam-torch-20261004-7180fd3f.json)

Exact pinned pure-Al MEAM port to PyTorch, AD forces/two tangents, validation against LAMMPS; same-GPU MACE comparison for current768-feature20/100fs targets,24A moving environments. Numerical/hardware experiment; no encoder training.

- ID: `response-meam-torch-20261004`
- Materials: Al
- Classification: **research**; role: response_engine_benchmark
- Location: `/store/PERSO/vmorozov/simulations/response-meam-torch-20261004`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM; MACE-MPA-0 medium checkpoint
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.004 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "title": "Differentiable Lee2003 Al MEAM local response benchmark",
  "materials": [
    "Al"
  ],
  "role": "response_engine_benchmark",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam",
    "mace-mpa-0-medium"
  ],
  "description": "Exact pinned pure-Al MEAM port to PyTorch, AD forces/two tangents, validation against LAMMPS; same-GPU MACE comparison for current768-feature20/100fs targets,24A moving environments. Numerical/hardware experiment; no encoder training.",
  "ancestry": "Original local80 parent indices0,1,2,45 and original two noise seeds; fixed Al64 e148b7ec215ba5e6d86fc57d21dac266bbd501f1e91320968266b5dbaeb8f44d. Four train ancestors only, no new independent roots; no relaxed inputs.",
  "evidence": [
    "${storage:repo}/configs/simulation/response_meam_torch_20261004.json",
    "${storage:repo}/docs/simulations/response_meam_torch_20261004.md"
  ],
  "limitations": [
    "Pure-Al/open-boundary specialization; exact potential-file hashes required.",
    "Hardware comparison changes potential physics; not an encoder performance result.",
    "Completion determined by numerical/timing receipts in the linked result bundle."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 0 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/response-meam-torch-20261004-7180fd3f.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
