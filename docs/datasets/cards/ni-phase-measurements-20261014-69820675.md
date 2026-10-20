# Dense Ni structural and velocity measurements with physical tangents

[All datasets](../README.md) · [Browsable card](ni-phase-measurements-20261014-69820675.html) · [Full metadata](../records/ni-phase-measurements-20261014-69820675.json)

244 measured values and 91 smooth responses in two initial perturbation directions at 0,5,...,100 fs for the same 630 sealed full-cell NVE teachers. Float64 observable arithmetic; no new MD or independent trajectories. Train-only normalizers. Fixed central atom and eighty atom IDs.

- ID: `ni-phase-measurements-20261014`
- Materials: Ni
- Classification: **complete**; role: motion_training_and_evaluation
- Location: `/home/ids/vmorozov/training-cache/ni-phase-measurements/observable-20261014-v1`
- Present on this machine: True
- Potentials: Mendelev et al. 2012 Ni EAM/FS
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 1.396 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Dense Ni structural and velocity measurements with physical tangents",
  "materials": [
    "Ni"
  ],
  "potential_ids": [
    "ni-mendelev2012-eam"
  ],
  "classification": "complete",
  "role": "motion_training_and_evaluation",
  "ancestry": "Same ninety Ni ancestors, 28 train and 62 pooled test; original roles retained",
  "description": "244 measured values and 91 smooth responses in two initial perturbation directions at 0,5,...,100 fs for the same 630 sealed full-cell NVE teachers. Float64 observable arithmetic; no new MD or independent trajectories. Train-only normalizers. Fixed central atom and eighty atom IDs.",
  "physical_anchors": 630,
  "training_anchors": 320,
  "test_anchors": 310,
  "measurement_channels": 244,
  "response_channels": 91,
  "evidence": [
    "configs/ni_latent_sim/phase_measurements_20261014.json",
    "docs/ni_phase_sim.md",
    "experiments/ni_phase_measurements_20261014/README.md"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 635 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ni-phase-measurements-20261014-69820675.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
