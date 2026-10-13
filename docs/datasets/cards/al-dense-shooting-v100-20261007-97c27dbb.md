# Additional dense Al velocity reshoots on V100

[All datasets](../README.md) · [Browsable card](al-dense-shooting-v100-20261007-97c27dbb.html) · [Full metadata](../records/al-dense-shooting-v100-20261007-97c27dbb.json)

40 planned shots, 20 existing melts; native NPT, 2 fs integration, 0.10 ps Q/V.

- ID: `al-dense-shooting-v100-20261007`
- Materials: Al
- Classification: **research**; role: raw_dynamics
- Location: `/scratch/PERSO/vmorozov/PointCloudMaterials/simulations/al-dense-shooting-v100-20261007`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.382 GiB
- Missing metadata: temperature_K, timestep_fs, ensemble

## Notes and relationships

```json
{
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "raw_dynamics",
  "classification": "research",
  "source_origin": "shooting",
  "sampling_protocol": "prepared_liquid_velocity_reshoot_v1",
  "title": "Additional dense Al velocity reshoots on V100",
  "description": "40 planned shots, 20 existing melts; native NPT, 2 fs integration, 0.10 ps Q/V.",
  "ancestry": "Inherited melt identities and original roles; no new independent preparations.",
  "evidence": [
    "/store/PERSO/vmorozov/simulation-launches/al-dense-shooting-v100-20261007/manifest.json",
    "/store/PERSO/vmorozov/simulation-launches/al-dense-shooting-v100-20261007/submission.json",
    "/store/PERSO/vmorozov/simulation-launches/al-dense-shooting-v100-20261007/status.json",
    "/store/PERSO/vmorozov/simulation-launches/al-dense-shooting-v100-20261007/technical/three-more-v100-20261007/submission.json"
  ],
  "execution_expansion": {
    "date": "2026-10-07",
    "total_concurrent_v100_workers": 4,
    "additional_worker_job": "1027414",
    "collector_job": "1027415",
    "branches_unchanged": 40,
    "receipt": "/store/PERSO/vmorozov/simulation-launches/al-dense-shooting-v100-20261007/technical/three-more-v100-20261007/submission.json"
  }
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 34 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-dense-shooting-v100-20261007-97c27dbb.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
