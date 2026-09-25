# Al: precision campaign and unfinished sources

[All datasets](../README.md) · [Browsable card](memory-al-precision-campaign-20260917-c8104deb.html) · [Full metadata](../records/memory-al-precision-campaign-20260917-c8104deb.json)

Stopped 12-source preparation; three completed sources and retained failures are registered separately on STORE. This is not twelve completed independent trajectories.

- ID: `memory-al-precision-campaign-20260917`
- Materials: Al
- Classification: **mixed**; role: container
- Location: `/scratch/PERSO/vmorozov/PointCloudMaterials/simulations/memory-al-precision-20260917`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 1.552 GiB
- Missing metadata: timestep_fs, ensemble

## Notes and relationships

```json
{
  "title": "Al: precision campaign and unfinished sources",
  "materials": [
    "Al"
  ],
  "role": "container",
  "classification": "mixed",
  "description": "Stopped 12-source preparation; three completed sources and retained failures are registered separately on STORE. This is not twelve completed independent trajectories.",
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "evidence": [
    "docs/simulations/predictive_memory_precision_20260917.md"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["predictive_memory_precision_sources_v1"] |
| temperatures_K | [[500.0, 520.0]] |
| root_lineage | ["independent_melt_126814323", "independent_melt_346665285", "independent_melt_384688017", "independent_melt_407215734", "independent_melt_46641549", "independent_melt_580020242", "independent_melt_696676085", "independent_melt_746219836", "independent_melt_76476585", "independent_melt_817843753", "independent_melt_840271448", "independent_melt_860878647"] |
| parent_trajectory_id | [null] |
| temperature_K | [500.0, 520.0] |
| split | ["sealed_test", "train", "val"] |
| preparation_seed | [126814323, 346665285, 384688017, 407215734, 46641549, 580020242, 696676085, 746219836, 76476585, 817843753, 840271448, 860878647] |
| velocity_seed | [103667006, 112005714, 138027857, 30147642, 342374706, 607567658, 612992447, 622469642, 816449419, 826723107, 827508800, 840962199] |

## Evidence

All 4 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/memory-al-precision-campaign-20260917-c8104deb.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-25T13:22:08.459741+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
