# Ni fixed-atom motion sequences: original, liquid persistence and advancing fronts

[All datasets](../README.md) · [Browsable card](ni-latent-sequences-regimes3x-20261010-86fef16a.html) · [Full metadata](../records/ni-latent-sequences-regimes3x-20261010-86fef16a.json)

Preparing14400 local80 native0.1ps sequences:4800 preserved originals plus4800 liquid-persistent and4800 growing-front windows. One immutable central atom ID per sequence.60 original NPT source ancestors; roles retained; no new MD.

- ID: `ni-latent-sequences-regimes3x-20261010`
- Materials: Ni
- Classification: **research**; role: motion_training_and_evaluation
- Location: `/home/ids/vmorozov/training-cache/ni-latent-sim/regimes3x-20261010-v2`
- Present on this machine: True
- Potentials: Mendelev et al. 2012 Ni EAM/FS
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 1.628 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Ni fixed-atom motion sequences: original, liquid persistence and advancing fronts",
  "materials": [
    "Ni"
  ],
  "potential_ids": [
    "ni-mendelev2012-eam"
  ],
  "role": "motion_training_and_evaluation",
  "classification": "research",
  "description": "Preparing14400 local80 native0.1ps sequences:4800 preserved originals plus4800 liquid-persistent and4800 growing-front windows. One immutable central atom ID per sequence.60 original NPT source ancestors; roles retained; no new MD.",
  "evidence": [
    "configs/ni_latent_sim/regimes3x_20261010.json",
    "docs/ni_latent_sim.md",
    "experiments/ni_latent_sim_20261010/REGIMES3X.md"
  ],
  "limitations": [
    "Added regimes selected by future behavior; not natural occupancy or unconditional birth rates.",
    "Only one of four inner-selection sources supplies qualifying growth fronts.",
    "Short physical-teacher bank remains the original300 groups; no new long physical-response derivatives.",
    "Readiness is technical/data-complete.json; candidate preflight does not mean encoding/training complete."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 789 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ni-latent-sequences-regimes3x-20261010-86fef16a.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
