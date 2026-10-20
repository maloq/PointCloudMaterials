# Differentiable CUDA MEAM precision, responses and seeded growth

[All datasets](../README.md) · [Browsable card](meam-gpu-blackwell-20261004-9cc24ddf.html) · [Full metadata](../records/meam-gpu-blackwell-20261004-9cc24ddf.json)

16384-atom450K200ps BAOAB descendant of existing native liquid; new explicitly seeded FCC/liquid slab and matched50ps float32-force/float64-force descendants. All integration states float64,1fs steps,0.1ps observations,1ps structure,5ps checkpoints. Pinned Lee2003 MEAM unchanged.

- ID: `meam-gpu-blackwell-20261004`
- Materials: Al
- Classification: **research**; role: gpu_meam_engine_validation
- Location: `/store/PERSO/vmorozov/simulations/meam-gpu-blackwell-20261004`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 8; these are not independent-source counts.
- Stored frames: 3712; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.631 GiB
- Missing metadata: ensemble

## Notes and relationships

```json
{
  "title": "Differentiable CUDA MEAM precision, responses and seeded growth",
  "materials": [
    "Al"
  ],
  "role": "gpu_meam_engine_validation",
  "classification": "research",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "description": "16384-atom450K200ps BAOAB descendant of existing native liquid; new explicitly seeded FCC/liquid slab and matched50ps float32-force/float64-force descendants. All integration states float64,1fs steps,0.1ps observations,1ps structure,5ps checkpoints. Pinned Lee2003 MEAM unchanged.",
  "ancestry": "One reused liquid ancestor, not another independent melt. Separate constructed interface with half fixed during20ps2500K BAOAB preparation, then immediate450K production quench; seeded growth is not spontaneous nucleation. Response benchmarks reuse four existing train parents and both original noise seeds.",
  "evidence": [
    "${storage:repo}/configs/simulation/meam_gpu_blackwell_20261004.json",
    "${storage:repo}/docs/simulations/meam_gpu_blackwell_20261004.md"
  ],
  "limitations": [
    "Completion and failure are per-arm receipts; registration does not establish completion.",
    "Initial mixed-liquid-200ps stopped at a periodic-bin boundary guard after10ps; preserved failed prefix, corrected production uses mixed-liquid-200ps-v2.",
    "No NPT density, melting point, independent-melt nucleation rate or new trained encoder is established.",
    "Float16 position observations; native float64 checkpoints preserve precision for divergence.",
    "Earlier interface preparation included5ps450K cooling, became99.84% crystalline, and failed the starting-interface gate; archived as interface-preparation-failed-v1.",
    "Both1325K interface preparations are hot-crystal controls, not proven liquids (mobile MSD0.219A\u00b2); verified2500K preparation has mobile MSD101.75A\u00b2 and0% crystalline mobile atoms."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| atom_count | [16384] |
| coordinate_convention | ["positions are wrapped Cartesian coordinates in angstrom relative to box_low in [0, box_high-box_low) before storage quantization; decode to float32 and wrap again"] |
| frame_count | [104, 11, 2001, 501, 82] |
| protocol | ["cuda_differentiable_meam_baoab_v1", "failed periodic-bin prefix; not production", "stopped hot-crystal control; excluded from liquid growth"] |
| storage_dtype | ["float16", "float32"] |
| duration_ps | [1.0, 200.0, 50.0] |
| temperature_K | [450.0] |
| timestep_fs | [1.0] |
| ptm_rmsd_cutoff | [0.1] |
| velocity_seed | [26100423] |

## Evidence

All 16 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/meam-gpu-blackwell-20261004-9cc24ddf.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
