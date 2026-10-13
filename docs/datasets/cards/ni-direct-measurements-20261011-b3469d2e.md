# Ni direct BO/TDA/density/velocity motion measurements

[All datasets](../README.md) · [Browsable card](ni-direct-measurements-20261011-b3469d2e.html) · [Full metadata](../records/ni-direct-measurements-20261011-b3469d2e.json)

14400 fixed-central-atom native 0.1-ps sequences through 6 ps, with 244 direct measurements and no learned encoder. Original 60 ancestors and 3840/960/9600 roles retained. Also remeasures 300 archived 100-fs physical response anchors with observations every 5 fs.

- ID: `ni-direct-measurements-20261011`
- Materials: Ni
- Classification: **research**; role: motion_training_and_evaluation
- Location: `/home/ids/vmorozov/training-cache/measurement-sim/direct-20261011-v1`
- Present on this machine: True
- Potentials: Mendelev et al. 2012 Ni EAM/FS
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 2.754 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Ni direct BO/TDA/density/velocity motion measurements",
  "materials": [
    "Ni"
  ],
  "potential_ids": [
    "ni-mendelev2012-eam"
  ],
  "role": "motion_training_and_evaluation",
  "classification": "research",
  "description": "14400 fixed-central-atom native 0.1-ps sequences through 6 ps, with 244 direct measurements and no learned encoder. Original 60 ancestors and 3840/960/9600 roles retained. Also remeasures 300 archived 100-fs physical response anchors with observations every 5 fs.",
  "evidence": [
    "configs/ni_latent_sim/direct_measurements_20261011.json",
    "docs/measurement_sim.md",
    "docs/metrics/measurement_sim.md"
  ],
  "limitations": [
    "No new independent MD; parent liquid/front selection is outcome-enriched.",
    "Analytic output derivatives apply to 91 smooth channels; BO/TDA have value supervision and actual input secants.",
    "Physical teachers use fixed-box BAOAB and fixed80 identities; native paths are NPT with reselected neighbors.",
    "No eSEN outputs, broad spatial collection or temperature/time covariates."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 300 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ni-direct-measurements-20261011-b3469d2e.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
