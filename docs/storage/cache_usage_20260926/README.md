# Cache use and retirement candidates — 26 September 2026

**Later update:** the authorized [683.461 GB cleanup](../cache_cleanup_20260927/README.md)
completed on September 27. The assessment and sizes below describe the earlier snapshot.

**Assessment only: no caches were deleted.** Sizes refer to the preceding
[1,058.106 GB physical inventory](../cache_inventory_20260926/README.md), including
archival copies and relaxed structures. Active training and archive transfers can
change the current total. GB means 10^9 allocated bytes.

## Recent work

| Evidence | Size of corresponding cache collections |
|---|---:|
| Referenced in scientific experiment records within seven days | 270.863 GB |
| Referenced within three days (subset of the preceding row) | 217.156 GB |
| Declared dependencies of running jobs or current preparation/queues | 36.527 GB |

The windows end at **2026-09-26 15:19:29 UTC**. These are sizes of referenced
collections, **not a measurement of bytes read**. A reference to part of a collection
protects its whole inventory row. Most historical caches have no reliable
last-use counter, so absence of a recent reference does not establish disuse.
Filesystem access times were deliberately excluded: inventory reads and copies
also change them. Source-code and inventory references were excluded from the
recent-consumer scan. Consumer record modification dates are evidence of recent
references; they are not themselves a complete access log.

The shared geometry cache has a stronger producer record: **52.039 GB**, last
leased at **2026-09-26 13:49:21 UTC**. All **six** current generated encoder-feature
entries have producer-recorded use today; one had an active lease at the audit.
Keep all six under the user's retention policy. Running MACE training maps the
multimaterial structural arrays, and its frozen configurations declare the fixed
cohort, predictor population, evaluation geometry and features. New dense-history
preparation also reuses old coordinate arrays through hard links; deleting a
directory need not free those shared arrays.

The transfer-cache source and destination are protected while a separate,
previously started verified archive operation is running. Its **73.251 GB** row
total in the preceding inventory includes a partial archive copy; it is not all
additional independent data. Storage-copy activity is not scientific use.

Evidence: [collection assessment](usage_by_collection.csv),
[recent references](recent_references.csv), [summary](usage_summary.json),
[processes and mappings](active_processes.json), [Slurm](active_slurm.txt),
[declared dependencies](active_config_dependencies.json) and
[feature/geometry lease records](lease_usage.json).

## Concrete retirement options

These are alternative cumulative scopes, **not additive amounts**:

| Scope | Reclaimable payload GB | Consequence |
|---|---:|---|
| Remove the old spatial-context STORE archive only | **341.745** | Retain the complete IDS spatial cache and base embeddings; verify payload equality before calling the retained copy byte-identical |
| Remove both spatial-context copies | **613.278** | Retain base embeddings; rebuild neighbors, radii and pooled embeddings for any future old spatial-forecast replay |
| Also remove the old full-forecast embeddings | **683.461** | Recompute embeddings with the original frozen encoder, then rebuild spatial caches if needed |

Recommendation: start with the **342 GB archive-only scope** after verification.
The **683 GB scope** is a reasonable broader retirement of old derived forecasting
data if its future recomputation cost is acceptable. It would leave approximately
**375 GB** of the preceding inventory. Keep the other caches pending a separate
review of active/recent dependencies and rebuilding costs.

The spatial family has no current scientific reader in the allocated-node process
audit and no scientific consumer reference in the seven-day scan. Its original
September 13 configurations and cache manifests remain. Importantly, the STORE
archive is **incomplete**: it has 1,155 files versus 1,512 in IDS. Every STORE file
has an IDS counterpart of the same length; none is unique to STORE. The three
manifests match, but the k32 archive is missing files. Large payload checksums
have **not** been reread, so do not remove IDS while treating this archive as a
complete replacement. [Comparison evidence](spatial_archive_comparison.json).

For full-forecast rebuilding, the original checkpoint and Hydra configuration
match the cache's recorded SHA256 values. The selected-source configuration
also matches. All **125** selected source manifests match their recorded hashes
and all declared source arrays exist. The exact embedding producer hash was
found in three retained source snapshots; the earliest queue snapshot was an
older producer and was not accepted as equivalent. This verifies the retained
ingredients, not a newly executed or bitwise-proven rebuild.
[Source audit](forecast_rebuild_sources.json) ·
[Producer audit](forecast_producer_audit.json).

The [file preview](retirement_preview.csv) enumerates **2,110 payload files**;
[option totals and conditions](retirement_preview.json) exclude metadata and
any hard links surviving outside the candidates. The selected names are only
`neighbors.npy`, `radii_A.npy`, `pooled_embeddings.npy` and `embeddings.npy` in the
three explicit old forecast roots. Manifests, configs, completion/checksum
receipts, atom IDs, frames and timelines remain, together with source trajectories,
model checkpoints, predictions, metrics and producer snapshots. Before any later
deletion, recheck live dependencies and the recorded file identities.
