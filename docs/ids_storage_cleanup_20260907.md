# IDS cleanup execution — September 7, 2026

Executed the user-approved [storage review](ids_storage_review_20260907.md).
Detailed per-file records, hashes, verification logs, and disposable recovery
scripts are in `output/ids_storage_cleanup_20260907/`.

**Completed around 11:47 CEST:** net reclamation attributable to this cleanup is
**99.05 GiB** (106,354,878,464 allocated bytes), including the cost of the new Ta
binary. The IDS home now occupies **397.8 GiB** (427,126,990,336 bytes). The earlier
review's roughly 513 GiB also included a partial binary removed separately; that
separate cleanup is excluded from the 99.05 GiB figure. Concurrent training can
change the live total.

## Completed cleanup

| Operation | Allocated GiB reclaimed |
| --- | ---: |
| Failed Ta converter scratch array | 26.95 |
| Superseded attempt data and old Ta raw text | 21.69 |
| Duplicate preparation inputs and unused static context | 4.90 |
| Lossless compression of two retained interrupted trajectories | 2.45 |
| Total before large Ta conversion | 55.98 |
| Large Ta text removed, less newly retained binary | 43.07 |
| Net total | 99.05 |

- Checked the remote `ta-then-ti-20260907.service` on lamedell11: failed,
  `MainPID=0`, and both previously recorded PIDs absent. Its partial binary had
  already been removed by the separate storage-failure diagnosis documented in
  `experiments/ti_ta_crystallization_20260907/storage_failure_20260907.json`.
  That prior 15.95 GiB reclamation is **not** included in this cleanup's total.
- Verified canonical replacement binary checksums before removing archived
  nested/shooting attempt trajectories. Kept metadata and logs. Removed files
  have sizes and SHA-256 hashes in `deleted_files.json` (the abandoned scratch
  array has size/mtime only because it was incomplete disposable output).
- Verified all 241 old Ta frames against the existing float32 binary, including
  IDs, types, timeline, boxes, and the producer's coordinate wrapping. Deleted
  its raw dump and updated its outcome's conversion record. Restarts remain.
- Compared superseded preparation inputs byte-for-byte by SHA-256 with their
  canonical copies before pruning them; retained preparation metadata.
- Removed the old 512-point `static_context` arrays after checking their input
  hashes and that the current MACE queue uses `static_context80`. Preserved their
  JSON signatures under `static_context_records/` in the audit directory. Removed
  the original cache JSON records too, because this producer uses record existence
  to decide whether to rebuild. Original static data and sampling caches remain.
- Compressed the interrupted Ta initial attempt and source-28 text dumps with
  gzip level 1. Read each compressed stream back and compared its SHA-256 to the
  original before removing the text. Per-archive `trajectory_compression.json`
  records locate the compressed replacement. These are historical diagnostics,
  not canonical simulation trajectories.

Active training caches, checkpoints, accepted shooting/source binaries, original
material snapshots, and unrelated geospatial/Parquet research data were retained.

## Large Ta conversion recovery

The first 10,000,422-atom Ta branch completed its 24 ps dynamics but failed
conversion on the IDS user quota. Recovered conversion runs separately as
`ta-storage-conversion-20260907.service` on lamedell11, through the maintained
command:

```bash
python scripts/convert_trajectory.py elemental \
  /home/ids/vmorozov/simulations/ti_ta_crystallization_20260907/Ta/branches/2.7ns \
  --delete-source
```

The converter verifies checksums and decoded coordinate equality before deleting
text; it retains the source hash and reclaimed byte counts in
`binary_conversion.json`. Binary validation now examines one frame at a time to
avoid boolean allocations spanning the entire trajectory. Float32 storage and
coordinate semantics are unchanged. The new optional deletion flag is documented
in the existing conversion guide; it does not change automatic campaign defaults.

The recovery finalizer verifies the saved restart hash and publishes this branch's
complete outcome only after conversion and deletion succeed. Its completion is
recorded in `output/ids_storage_cleanup_20260907/ta_recovery.json`.
Recovery succeeded: all 241 frames / 10,000,422 atoms passed verification; the
raw dump and scratch array are gone, and `outcome.json` now records complete.
The new binary occupies 26.98 GiB; deleting its 70.05 GiB raw text yields a net
43.07 GiB saving for this conversion. No temporary build directory remains.
No dynamics are rerun. The remaining Ta/Ti campaign is not automatically submitted
by this storage cleanup; its space requirements still need to fit the user quota.

Validation: six targeted conversion/temporal-binary tests passed, including
elemental coordinate round-trip with and without source deletion.
`git diff --check` passed. Source-code changes are maintained conversion-tool
improvements; the scripts and inventories in the output directory are disposable
diagnostics, not new maintained commands.
