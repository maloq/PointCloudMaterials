# Ni regional density and motion context at three spatial scales

[All datasets](../README.md) · [Browsable card](ni-spatial-context-20261015-55a2300f.html) · [Full metadata](../records/ni-spatial-context-20261015-55a2300f.json)

Derived 87-channel density/shell-motion/affine-motion measurements from complete periodic balls of radius 8, 12 and 20 Angstrom, without a nearest-atom cap. Reuses all 14,400 native kinetic windows and 19,800 birth-assay rows. Deduplicated extraction reads 52,578 frames from 87 relevant existing sources and measures 999,971 source/frame/center observations at all three radii. Normalization uses only native training observations; central244 measurements and targets remain unchanged. No new MD, interpolation or birth neighborhood filtering.

- ID: `ni-spatial-context-20261015`
- Materials: Ni
- Classification: **research**; role: motion_training_and_evaluation
- Location: `/home/ids/vmorozov/training-cache/ni-spatial-scales/radius-20261015-v1`
- Present on this machine: True
- Potentials: Mendelev et al. 2012 Ni EAM/FS
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 1.947 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Ni regional density and motion context at three spatial scales",
  "materials": [
    "Ni"
  ],
  "potential_ids": [
    "ni-mendelev2012-eam"
  ],
  "role": "motion_training_and_evaluation",
  "classification": "research",
  "status": "preparing",
  "description": "Derived 87-channel density/shell-motion/affine-motion measurements from complete periodic balls of radius 8, 12 and 20 Angstrom, without a nearest-atom cap. Reuses all 14,400 native kinetic windows and 19,800 birth-assay rows. Deduplicated extraction reads 52,578 frames from 87 relevant existing sources and measures 999,971 source/frame/center observations at all three radii. Normalization uses only native training observations; central244 measurements and targets remain unchanged. No new MD, interpolation or birth neighborhood filtering.",
  "ancestry": "Original native 60-source cohort: 16 train/44 test ancestors and 3840/10560 windows. Separate 90-source birth contract: 28 train/62 test; represented assay rows cover 21 train/42 test ancestors. The union requires 87 relevant raw sources; it does not create independent sources.",
  "base_plan_sha256": "ec26712af4b0113bea6c4b5dd95df0bc6baf200b33e251951bea29e8d03094be",
  "sample_interval_ps": 0.1,
  "radii_A": [
    8,
    12,
    20
  ],
  "context_channels": 87,
  "native_rows": 14400,
  "native_frames_per_row": 64,
  "birth_assay_rows": 19800,
  "birth_unique_states": 19652,
  "birth_history_frames": 4,
  "relevant_sources": 87,
  "source_frames": 52578,
  "unique_observations": 999971,
  "evidence": [
    "configs/ni_latent_sim/spatial_scales_20261015.json",
    "docs/ni_spatial_scales.md",
    "experiments/ni_spatial_scales_20261015/README.md",
    "${storage:training_storage}/ni_spatial_scales/radius-20261015-v1/technical/plan.json",
    "${storage:training_storage}/ni_spatial_scales/radius-20261015-v1/technical/context-benchmark.json",
    "${storage:training_storage}/ni_spatial_scales/radius-20261015-v1/technical/context-complete.json",
    "${storage:cache}/ni-spatial-scales/radius-20261015-v1/context.json"
  ],
  "limitations": [
    "Collection completion is determined by per-source receipts and context.json; planning and performance receipts do not establish completion.",
    "Native liquid-persistent and growing-front windows are outcome-enriched; their frequencies are not natural incidence or equilibrium occupancy.",
    "Wider observed regions may contain existing crystals or fronts; a context gain does not alone establish a liquid precursor.",
    "Regional support is reselected at every observation. Existing central birth history retains its fixed present-neighbor support; learned birth encoders use present input only.",
    "Regional TDA is not recomputed; compressed density and motion summaries do not retain complete atomic positions or connectivity.",
    "Autonomous latent generation receives only the initial four observed contexts and predicts central244. Independent local paths are not jointly generated atomistic nuclei."
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 87 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ni-spatial-context-20261015-55a2300f.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-18T23:47:27.639076+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
