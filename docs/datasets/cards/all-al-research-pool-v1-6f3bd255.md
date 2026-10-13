# All Al research pool: joined appearance/fate inputs and cohort inventory

[All datasets](../README.md) · [Browsable card](all-al-research-pool-v1-6f3bd255.html) · [Full metadata](../records/all-al-research-pool-v1-6f3bd255.json)

One metadata-selectable pool with paired raw/relaxed views, appearance and resolved-fate labels, exact physical observation protocols, and all-Al availability inventory. Only sealed eligible input banks are training-ready.

- ID: `all-al-research-pool-v1`
- Materials: Al
- Classification: **derived**; role: event_enriched_classification
- Location: `/store/PERSO/vmorozov/experiments/al_research/pool-v1`
- Present on this machine: True
- Potentials: Lee–Shim–Baskes 2003 Al 2NN-MEAM
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.971 GiB
- Missing metadata: None in the core fields

## Notes and relationships

```json
{
  "title": "All Al research pool: joined appearance/fate inputs and cohort inventory",
  "materials": [
    "Al"
  ],
  "potential_ids": [
    "al-lee2003-meam"
  ],
  "role": "event_enriched_classification",
  "classification": "derived",
  "description": "One metadata-selectable pool with paired raw/relaxed views, appearance and resolved-fate labels, exact physical observation protocols, and all-Al availability inventory. Only sealed eligible input banks are training-ready.",
  "provenance": "src/data/al_pool.py; retained upstream manifests and descriptor checksums; no relabeling or new MD",
  "ancestry": "Original roles and shared ancestor IDs preserved across older sources, shooting and dense descendants. Views are not additional independent samples.",
  "evidence": [
    "configs/al_research_pool.json",
    "docs/al_research_pool.md",
    "${dataset:all-al-research-pool-v1}/current.json"
  ],
  "limitations": [
    "Raw Al collections without task-specific eligibility remain inventoried but do not acquire prediction labels.",
    "Native 0.1 ps inputs enter on a new release only after sealing; 0.01 ps and million-atom coverage is separate metadata until eligible inputs exist.",
    "Potential ID describes current prepared descriptors; the wider inventory retains all Al potentials independently.",
    "No combined model was trained by pool construction; historical metrics keep their original populations."
  ],
  "relaxation_source_availability_note": "Some historical relaxed-cell/neighborhood inputs were deleted at user request on 2026-10-03; retained downstream artifacts are unchanged. See docs/storage/relaxation_unverified_removal_20261003/receipt.json."
}
```

## Recorded fields

| Field | Values |
| --- | --- |

## Evidence

All 2 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/all-al-research-pool-v1-6f3bd255.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-10-11T23:09:08.376791+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
