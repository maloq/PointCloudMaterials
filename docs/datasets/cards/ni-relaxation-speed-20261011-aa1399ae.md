# Ni matched minimizer speed and structural fidelity pilot

[All datasets](../README.md) · [Browsable card](ni-relaxation-speed-20261011-aa1399ae.html) · [Full metadata](../records/ni-relaxation-speed-20261011-aa1399ae.json)

Twelve full32000-atom inputs from six existing response-bank sources, retaining original atom identities, Q,V ancestry and fixed boxes. FIRE, ABC-FIRE, tolerance and bounded ASE preconditioning comparisons; no new nucleus labels or independent MD samples.

- ID: `ni-relaxation-speed-20261011`
- Materials: Ni
- Classification: **research**; role: numerical_diagnostic
- Location: `/store/PERSO/vmorozov/simulations/ni-relaxation-speed-20261011-v1`
- Present on this machine: True
- Potentials: Mendelev et al. 2012 Ni EAM/FS
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.508 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Ni matched minimizer speed and structural fidelity pilot",
  "materials": [
    "Ni"
  ],
  "potential_ids": [
    "ni-mendelev2012-eam"
  ],
  "role": "numerical_diagnostic",
  "classification": "research",
  "description": "Twelve full32000-atom inputs from six existing response-bank sources, retaining original atom identities, Q,V ancestry and fixed boxes. FIRE, ABC-FIRE, tolerance and bounded ASE preconditioning comparisons; no new nucleus labels or independent MD samples.",
  "evidence": [
    "${storage:training_storage}/ni_relaxation/speed-20261011-v1/technical/plan.json",
    "docs/simulations/ni_relaxation_speed_20261011.md"
  ],
  "limitations": [
    "Five earlier repeatability sources plus one convergence-stress source; not a population sample.",
    "ASE methods use three-case bounded pilots, and incomplete cases do not count as speedups.",
    "Diagnostic minima do not replace historical training targets."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 0 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ni-relaxation-speed-20261011-aa1399ae.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
