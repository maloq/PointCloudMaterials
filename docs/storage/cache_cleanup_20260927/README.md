# Forecast cache cleanup — 27 September 2026

Completed the user-authorized **683 GB cleanup**: **2,110 payload files**, totaling
**683,461,234,176 allocated bytes (683.461 GB)**, removed at 08:54 UTC.
This is the sum of allocated blocks of deleted single-link files, not a live quota delta.

| Physical location | Removed GB |
| --- | ---: |
| IDS: `training-cache/embedding-forecast-spatial-context-20260913` | 271.533 |
| STORE: `training-cache-archive-20260925/embedding-forecast-spatial-context-20260913` | 341.745 |
| STORE: `training-cache-archive-20260926/embedding-forecast-full-20260911` | 70.183 |
| Total | 683.461 |

Only the enumerated `neighbors.npy`, `radii_A.npy`, `pooled_embeddings.npy` and
`embeddings.npy` payloads were deleted. Metadata, IDs and timelines remain.
Each root now contains `retirement.json` and `RETIRED.md`; original completion
manifests are historical provenance, not evidence that payloads remain usable.
The two dataset entries are marked `payload_deleted`, `retired`, and
`provenance_only`. Rebuild these caches before reusing their historical protocols.

Before deletion, all planned file identities matched the reviewed proposal and
current jobs had no matching open files or memory maps. Verification preserved
1,355 nonpayload files by checksum, checked 965 protected arrays unchanged, and
retained all 125 reconstruction source trajectories and six managed encoder
feature-cache entries. Original checkpoint, configuration and matching producer
snapshots were verified. Checkpoints, predictions, metrics and source MD were
outside the deletion scope.

Evidence:

- [Completion receipt](receipt.json)
- [Exact applied file plan](applied_plan.csv)
- [Deletion journal](actions.jsonl)
- [Pre-deletion identities](identity_validation.json)
- [Current-job audit](process_audit.json)
- [Preserved file checksums](preserved_files.json)
- [Protected array identities](protected_arrays.json)
- [Previous registry entries](registry_before.json)
- [Original assessment and rebuild-source audit](../cache_usage_20260926/README.md)

The September 26 inventory remains a historical snapshot; subtracting this cleanup
from it is not a fresh measurement of all remaining caches.

Registry publication note: `configs/datasets.json` was updated and verified. The
full `scripts/project.py datasets --refresh` attempt encountered a concurrent
writer collision on a `.json.building` record. Another refresh was already
running after the catalog update; generated HTML/cards may lag until it finishes.
The authoritative retirement state and per-root tombstones are complete.
