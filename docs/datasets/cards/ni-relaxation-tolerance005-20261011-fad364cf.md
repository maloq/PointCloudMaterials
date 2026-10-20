# Ni matched 0.01/0.03/0.05 force-tolerance diagnostic

[All datasets](../README.md) · [Browsable card](ni-relaxation-tolerance005-20261011-fad364cf.html) · [Full metadata](../records/ni-relaxation-tolerance005-20261011-fad364cf.json)

Sorted native FIRE at three force-component tolerances on the same twelve full32000-atom inputs from six sources as the speed pilot. All settings timed on one GPU, with float64 minimizer states and float32 forces; original identities, ancestry and boxes retained.

- ID: `ni-relaxation-tolerance005-20261011`
- Materials: Ni
- Classification: **research**; role: numerical_diagnostic
- Location: `/store/PERSO/vmorozov/simulations/ni-relaxation-tolerance005-20261011-v1`
- Present on this machine: True
- Potentials: Mendelev et al. 2012 Ni EAM/FS
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.224 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Ni matched 0.01/0.03/0.05 force-tolerance diagnostic",
  "materials": [
    "Ni"
  ],
  "potential_ids": [
    "ni-mendelev2012-eam"
  ],
  "role": "numerical_diagnostic",
  "classification": "research",
  "description": "Sorted native FIRE at three force-component tolerances on the same twelve full32000-atom inputs from six sources as the speed pilot. All settings timed on one GPU, with float64 minimizer states and float32 forces; original identities, ancestry and boxes retained.",
  "evidence": [
    "${storage:training_storage}/ni_relaxation/tolerance005-20261011-v1/technical/plan.json",
    "configs/simulation/ni_relaxation_tolerance005_20261011.json",
    "docs/simulations/ni_relaxation_speed_20261011.md"
  ],
  "limitations": [
    "Numerical repeats of the previous inputs, not new independent MD samples.",
    "Contains a convergence-stress source; not a population sample.",
    "Diagnostic minima do not replace historical training targets."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 0 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ni-relaxation-tolerance005-20261011-fad364cf.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
