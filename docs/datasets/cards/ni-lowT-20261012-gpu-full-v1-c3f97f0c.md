# Additional 30 Ni melts: GPU birth search over full saved histories

[All datasets](../README.md) · [Browsable card](ni-lowT-20261012-gpu-full-v1-c3f97f0c.html) · [Full metadata](../records/ni-lowT-20261012-gpu-full-v1-c3f97f0c.json)

GPU resident/voronoi extraction, then unchanged CPU lineage search on all completed native 0.10 ps Q/V sources. No-birth histories remain valid censored observations; incomplete sources stay explicit. Primary PTM establishment is 64 atoms for 1.5 ps, with 8/32/128-atom thresholds also retained.

- ID: `ni-lowT-20261012-gpu-full-v1`
- Materials: Ni
- Classification: **queued**; role: derived_analysis
- Location: `/store/PERSO/vmorozov/experiments/nuclei_search/ni-lowT-20261012-gpu-full-v1`
- Present on this machine: True
- Potentials: Mendelev et al. 2012 Ni EAM/FS
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 1.822 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Additional 30 Ni melts: GPU birth search over full saved histories",
  "role": "derived_analysis",
  "classification": "queued",
  "materials": [
    "Ni"
  ],
  "potential_ids": [
    "ni-mendelev2012-eam"
  ],
  "description": "GPU resident/voronoi extraction, then unchanged CPU lineage search on all completed native 0.10 ps Q/V sources. No-birth histories remain valid censored observations; incomplete sources stay explicit. Primary PTM establishment is 64 atoms for 1.5 ps, with 8/32/128-atom thresholds also retained.",
  "source_campaign_identity": "5d4a689eb39d938a4d171c81ea5a3592ebf6c9f36ba7c3c7b16bbab64e5d1353",
  "planned_independent_melts": 30,
  "native_cadence_ps": 0.1,
  "descriptor_backend": "gpu-extract/resident/voronoi",
  "ancestry": "Same 30 independent parent sources and prespecified train/selection/test roles; no extra independent samples",
  "cache": "${storage:cache}/nuclei-search/ni-lowT-20261012-gpu-full-v1",
  "analysis_extent": "full_saved_trajectory",
  "simulation_hold_cap_ps": 400,
  "launch_jobs": {
    "extract": "1035586",
    "track": "1035587"
  },
  "evidence": [
    "configs/analysis/ni_lowT_gpu_search_20261012.json",
    "/home/infres/vmorozov/PointCloudMaterials/output/ni_nucleation/campaign-lowT-20261012/technical/gpu-search/launch.json",
    "docs/simulations/ni_births_lowT_010_20261012.md",
    "/home/infres/vmorozov/PointCloudMaterials/output/ni_nucleation/campaign-lowT-20261012/technical/gpu-search-recovery-01/recovery.json",
    "/home/infres/vmorozov/PointCloudMaterials/output/ni_nucleation/campaign-lowT-20261012/technical/gpu-search-recovery-01/launch.json",
    "/home/infres/vmorozov/PointCloudMaterials/output/ni_nucleation/campaign-lowT-20261012/technical/gpu-search-recovery-01/preflight.json"
  ],
  "execution_recovery": "/home/infres/vmorozov/PointCloudMaterials/output/ni_nucleation/campaign-lowT-20261012/technical/gpu-search-recovery-01/recovery.json",
  "initial_extraction_failure": "Missing inherited numerical audit file in frozen bundle; failed before extraction",
  "completed_sources_available": 30
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 121 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ni-lowT-20261012-gpu-full-v1-c3f97f0c.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
