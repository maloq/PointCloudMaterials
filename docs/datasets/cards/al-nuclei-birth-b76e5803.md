# Al nuclei birth dataset

[All datasets](../README.md) · [Browsable card](al-nuclei-birth-b76e5803.html) · [Full metadata](../records/al-nuclei-birth-b76e5803.json)

Current cumulative Al birth/fate catalogue: exact 0.10 ps saved-frame views and native-only uniformly saved 0.30 ps trajectories. Established births and failed embryos retain source/frame mappings and ancestry. Crystalline precursor tails and first-crystal observation labels are separately sealed supplements, not additional births.

- ID: `al-nuclei-birth`
- Materials: Al
- Classification: **derived**; role: event_dataset
- Location: `/store/PERSO/vmorozov/experiments/nuclei_birth/al`
- Present on this machine: True
- Potentials: Unknown / not applicable
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 3.797 GiB
- Missing metadata: generating potential identity

## Notes and relationships

```json
{
  "title": "Al nuclei birth dataset",
  "materials": [
    "Al"
  ],
  "role": "event_dataset",
  "classification": "derived",
  "potential_ids": [],
  "description": "Current cumulative Al birth/fate catalogue: exact 0.10 ps saved-frame views and native-only uniformly saved 0.30 ps trajectories. Established births and failed embryos retain source/frame mappings and ancestry. Crystalline precursor tails and first-crystal observation labels are separately sealed supplements, not additional births.",
  "provenance": "src.data.nuclei_birth; completed v3 searches, immutable upstream tracking receipts and source metadata",
  "lineage": "Original melt ancestry, source roles, shooting branches and historical prefixes remain explicit; related views are not independent samples.",
  "evidence": [
    "docs/datasets/nuclei_birth.md",
    "configs/data/nuclei_birth.json",
    "${dataset:al-nuclei-birth}/manifest.json"
  ],
  "limitations": [
    "No prediction-history eligibility or liquid controls are supplied.",
    "Mixed historical generating potentials stay recorded per source; unknown values remain unknown.",
    "New raw MD requires completed structural extraction and v3 tracking before refresh.",
    "Historical role conflicts require grouped selection in new experiments."
  ],
  "sampling_policy": {
    "010": "exact_saved",
    "030": "native_saved"
  }
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| lineage | ["unchanged CPU chronological tracking", {"continuation_fraction": 0.5, "durations_ps": [1.5, 3.0], "embryo_atoms": 8, "establishment_atoms": 64, "growth_atoms": 128, "interface_distance_A": 8.0, "minimum_shared_atoms": 4, "minor_parent_fraction": 0.2, "persistence_ps": 1.5, "prehistory_ps": 1.5, "protocol": "al_nucleus_lineage_v3", "reset_atoms": 4, "reset_ps": 1.5, "thresholds": [8, 32, 64, 128]}] |
| protocol | ["al_nucleus_lineage_v3"] |
| ptm_rmsd_cutoff | [0.1] |
| cadence_ps | [0.1, 0.3] |
| split | ["Historical roles unchanged; group by ancestor and review role conflicts before any fit"] |

## Evidence

All 3 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/al-nuclei-birth-b76e5803.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T00:11:05.385980+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
