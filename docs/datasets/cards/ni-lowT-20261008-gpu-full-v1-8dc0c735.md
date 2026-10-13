# Ni low-temperature GPU birth search, all saved history

[All datasets](../README.md) · [Browsable card](ni-lowT-20261008-gpu-full-v1-8dc0c735.html) · [Full metadata](../records/ni-lowT-20261008-gpu-full-v1-8dc0c735.json)

All verified saved native 0.10-ps frames for the same 36 independent sources, with no analysis cap. Completed trajectories beyond 420 ps and retained converted tails are included; tails are archived on STORE without resimulation or requantization. The 400-ps simulation hold stop remains in effect. Original source roles, event rules and explicit zero-birth outcomes are retained; censoring uses each actual saved endpoint.

- ID: `ni-lowT-20261008-gpu-full-v1`
- Materials: Ni
- Classification: **queued**; role: derived_analysis
- Location: `/store/PERSO/vmorozov/experiments/nuclei_search/ni-lowT-20261008-gpu-full-v1`
- Present on this machine: True
- Potentials: Mendelev et al. 2012 Ni EAM/FS
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 2.240 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Ni low-temperature GPU birth search, all saved history",
  "role": "derived_analysis",
  "classification": "queued",
  "materials": [
    "Ni"
  ],
  "potential_ids": [
    "ni-mendelev2012-eam"
  ],
  "description": "All verified saved native 0.10-ps frames for the same 36 independent sources, with no analysis cap. Completed trajectories beyond 420 ps and retained converted tails are included; tails are archived on STORE without resimulation or requantization. The 400-ps simulation hold stop remains in effect. Original source roles, event rules and explicit zero-birth outcomes are retained; censoring uses each actual saved endpoint.",
  "source_campaign_identity": "ec82cb7932b179f693809805cc45ee69501ff3b24069f18d0ef86ceb3597ef03",
  "planned_independent_melts": 36,
  "native_cadence_ps": 0.1,
  "descriptor_backend": "gpu-extract/resident/voronoi",
  "ancestry": "Exactly the parent campaign sources and roles; no new simulations, resplitting or extra independent samples",
  "cache": "${storage:cache}/nuclei-search/ni-lowT-20261008-gpu-full-v1",
  "evidence": [
    "configs/analysis/ni_lowT_gpu_search_full_20261008.json",
    "/home/infres/vmorozov/PointCloudMaterials/output/ni_nucleation/campaign-lowT-20261008/technical/gpu-search-full-01/preparation.json",
    "/home/infres/vmorozov/PointCloudMaterials/output/ni_nucleation/campaign-lowT-20261008/technical/hard-stop-400ps/analysis-policy-amendment.json",
    "docs/simulations/ni_births_lowT_010_20261008.md",
    "/home/infres/vmorozov/PointCloudMaterials/output/ni_nucleation/campaign-lowT-20261008/technical/gpu-search-full-01/launch.json",
    "/home/infres/vmorozov/PointCloudMaterials/output/ni_nucleation/campaign-lowT-20261008/technical/gpu-search-full-01/queue-verification.json",
    "/home/infres/vmorozov/PointCloudMaterials/output/ni_nucleation/campaign-lowT-20261008/technical/gpu-search-full-01/full-history-verification.json",
    "/home/infres/vmorozov/PointCloudMaterials/output/ni_nucleation/campaign-lowT-20261008/technical/gpu-search-full-handoff-01/launch.json",
    "/home/infres/vmorozov/PointCloudMaterials/output/ni_nucleation/campaign-lowT-20261008/technical/gpu-search-full-handoff-01/queue-verification.json",
    "/home/infres/vmorozov/PointCloudMaterials/output/ni_nucleation/campaign-lowT-20261008/technical/rtx-handoff-01/restart-verified.json"
  ],
  "supersedes_analysis": "ni-lowT-20261008-gpu-400ps-v1",
  "analysis_extent": "full_saved_trajectory",
  "simulation_hold_cap_ps": 400,
  "launch_jobs": {
    "extract": "1028840",
    "track": "1028841"
  },
  "execution_handoff": "/home/infres/vmorozov/PointCloudMaterials/output/ni_nucleation/campaign-lowT-20261008/technical/rtx-handoff-01/launch.json"
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 145 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ni-lowT-20261008-gpu-full-v1-8dc0c735.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
