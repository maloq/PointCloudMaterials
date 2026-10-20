# Ni100-fs paths with5-fs analytic response observations

[All datasets](../README.md) · [Browsable card](ni-dense-path-response-20261011-5003b20d.html) · [Full metadata](../records/ni-dense-path-response-20261011-5003b20d.json)

300 original full-cell anchors from60 Ni NPT sources continued with fixed-box BAOAB for100fs; two original noise shots, stored local80 positions/velocities/tangents every5fs. Float64 states and tangents, float32 forces. Original ancestry and roles retained.

- ID: `ni-dense-path-response-20261011`
- Materials: Ni
- Classification: **research**; role: motion_training_and_evaluation
- Location: `/store/PERSO/vmorozov/simulations/ni-dense-path-response-20261011-v1`
- Present on this machine: True
- Potentials: Mendelev et al. 2012 Ni EAM/FS
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.220 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Ni100-fs paths with5-fs analytic response observations",
  "materials": [
    "Ni"
  ],
  "potential_ids": [
    "ni-mendelev2012-eam"
  ],
  "role": "motion_training_and_evaluation",
  "classification": "research",
  "description": "300 original full-cell anchors from60 Ni NPT sources continued with fixed-box BAOAB for100fs; two original noise shots, stored local80 positions/velocities/tangents every5fs. Float64 states and tangents, float32 forces. Original ancestry and roles retained.",
  "evidence": [
    "configs/ni_latent_sim/dense_path_response_20261011.json",
    "docs/ni_latent_sim.md",
    "experiments/ni_latent_sim_20261010/DENSE_PATH.md"
  ],
  "limitations": [
    "All300 physical paths and encoded teacher packets complete; retains original ancestry, roles and physical shot seeds.",
    "No new independent source ancestors; physical short protocol differs from native NPT.",
    "Fixed initial80 atom IDs during teacher path; native long observations reselect neighbors about same center.",
    "Encoded targets are separately cached under IDS ni-latent-sim/dense-path-response-20261011-v1."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| seed | [100661259311376, 101118576584636, 101809906553240, 102730979339005, 103306510607479, 103321189688082, 104926938148963, 106647317176941, 106866462762878, 109026340872201, 109547808708683, 109806317641912] … (300 values; see JSON) |
| protocol | ["full-cell fixed-box BAOAB, actual saved Q,V, source T, no relaxation"] |

## Evidence

All 300 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ni-dense-path-response-20261011-5003b20d.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
