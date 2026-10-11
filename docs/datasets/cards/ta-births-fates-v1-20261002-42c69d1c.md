# Ta full-cell births and nucleus fates

[All datasets](../README.md) · [Browsable card](ta-births-fates-v1-20261002-42c69d1c.html) · [Full metadata](../records/ta-births-fates-v1-20261002-42c69d1c.json)

30 complete24ps futures, six archived parents, exact0.5ps full-cell PTM/lineage fate audit. Reuses six historical PTM/graphs and computes24 shooting searches. No new MD.

- ID: `ta-births-fates-v1-20261002`
- Materials: Ta
- Classification: **derived**; role: label_audit
- Location: `/store/PERSO/vmorozov/experiments/nucleus_fates/ta-20261002`
- Present on this machine: True
- Potentials: Zhong 2014 Ta EAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 1.161 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "Ta full-cell births and nucleus fates",
  "materials": [
    "Ta"
  ],
  "potential_ids": [
    "ta-zhong2014-eam"
  ],
  "role": "label_audit",
  "classification": "derived",
  "description": "30 complete24ps futures, six archived parents, exact0.5ps full-cell PTM/lineage fate audit. Reuses six historical PTM/graphs and computes24 shooting searches. No new MD.",
  "provenance": "src/research/crystallization_origin/ta.py; checksum-bound original outcomes, trajectory arrays, historical external audit and20260926 shooting manifest",
  "ancestry": "One conservative archived-Ta common-preparation group; siblings and all six parents are correlated. No new split.",
  "evidence": [
    "configs/analysis/ta_births_20261002.json",
    "docs/metrics/ta_births.md",
    "${dataset:ta-births-fates-v1-20261002}/technical/plan.json",
    "${dataset:ta-births-fates-v1-20261002}/technical/complete.json"
  ],
  "limitations": [
    "Operational PTM FCC/HCP/BCC crystal definition, not all Ta ordered structures or validated critical nuclei.",
    "Exact0.5ps cadence can miss faster episodes; separate from the Al0.3ps search.",
    "Original float16 quantization and float32 boxes are retained.",
    "Candidate histories have not been screened for strict crystal-free predictor eligibility."
  ],
  "source_availability_note": "One or more raw Ta trajectory dependencies were deleted on 2026-10-03. Historical derived artifacts retain their original scientific definitions; see docs/storage/ta_simulation_removal_20261003/receipt.json."
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["ta-full-cell-births-fates-v1"] |
| cadence_ps | [0.5] |
| ptm_rmsd_cutoff | [0.1] |
| material | ["Ta"] |
| parent_id | ["2.7ns", "2.8ns", "2.9ns", "3.0ns", "3.60ns", "model_1m"] |
| velocity_seed | [145515637, 186999831, 241723527, 254269500, 32816276, 357008615, 376325111, 381745835, 395820186, 436100696, 47085415, 533918726] … (30 values; see JSON) |
| timestep_fs | [2.0] |
| storage_dtype | ["float16"] |
| atom_count | [10000422, 1024000] |
| frame_count | [49] |
| lineage | [{"interface_distance_A": 8.233252184899222, "local_birth_radius_A": 8.233252184899222, "maximum_missing_frames": 1, "minimum_overlap_fraction": 0.25, "minimum_shared_atoms": 4, "minimum_tracked_size": 4, "neighbor_cutoff_A": 3.70496348320465, "thresholds": [{"name": "primary", "persistence_frames": 4, "persistence_ps": 1.5, "size": 64}]}] |

## Evidence

All 33 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/ta-births-fates-v1-20261002-42c69d1c.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T00:11:05.385980+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
