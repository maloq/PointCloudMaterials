# IDS storage review — September 7, 2026

Subsequent user-approved execution is recorded in
[ids_storage_cleanup_20260907.md](ids_storage_cleanup_20260907.md).

Read-only inspection of `/home/ids/vmorozov`, around 11:05–11:10 CEST.
Nothing deleted, compressed, or submitted. Sizes below are allocated GiB from
file block counts / `du`, not apparent file length. Inventory is not atomic.

## Inventory

| Directory | GiB |
| --- | ---: |
| Entire IDS home | 512.8 |
| simulations | 416.9 |
| experiments | 75.5 |
| data (other research) | 18.9 |
| models | 1.4 |

The shared filesystem has free space, but that does not establish the user's
quota. `quota` is unavailable here; no current quota value was verified.

## First cleanup candidates

All paths below are relative to `/home/ids/vmorozov`.

1. **12.63 GiB:**
   `simulations/al_meam_nested_shooting_pilot_70304_400-500K_20260902/branches/*/failed_censored_finalize_temperature_variable_20260902T084826Z/`.
   All 13 corresponding canonical branches now have complete outcomes. Keep
   failure metadata/logs and canonical outputs; the archived bulk trajectories
   and restart files are deletion candidates after checking accepted artifact
   references. Full checksums and cross-file dependencies were not re-audited.
2. **1.96 GiB:**
   `simulations/al_meam_position_shooting_70304_400-500K_48ps_8shots_20260830/branches/*/interrupted_attempt_wave_*/`.
   Old attempts from the completed shooting campaign. Apply the same
   canonical-artifact/reference check and retain small provenance records.
3. **7.10 GiB:**
   `simulations/ta_initial_model_1m_24ps_npt_20260905/branches/Ta/model_1m/trajectory.lammpstrj`.
   Complete outcome records a verified float32 binary, 241 frames and conversion
   checksums. Reverify that binary and its coordinate equivalence and consumers
   before deleting the text. This older Ta recipe is distinct from the newer
   elemental converter; do not blindly apply another producer's deletion command.

These three categories offer roughly **21.7 GiB**, slightly less when retaining
small diagnostic records. They are proposals, not unconditional deletion lists.
Superseded top-level preparation directories add about 0.48 GiB; preserve their
manifests and protocol-correction records before pruning copied bulk inputs.

## Large conditional candidates

The new `simulations/ti_ta_crystallization_20260907/Ta/branches/2.7ns/` contains:

| File | Allocated GiB |
| --- | ---: |
| trajectory.lammpstrj | 70.05 |
| conversion_positions.npy | 26.95 |
| .trajectory_binary_float32.building-3725716/positions.npy | 15.95 |

Metadata says dynamics_complete, but no completed conversion report exists.
The scratch file was last modified around 02:18 CEST, several hours before this
review. The sequence status still says running on the remote lamedell11 service;
SSH key authentication failed, so worker liveness is unconfirmed. **Do not delete
these files based on their names alone.** If that worker is dead, preserve the raw
trajectory and diagnose/restart conversion; its two intermediate arrays account
for about **42.9 GiB**. The current converter removes scratch only after successful
verification. After successful conversion and consumer checks, the 70 GiB raw
text becomes a separate deletion candidate. Creating the final binary needs
working space, so these savings are not all immediate or additive peak savings.

The new campaign also has 4.03 GiB of archived initial attempts. The old
high-temperature source-28 archive is 2.94 GiB. Keep both until their replacement
and diagnostic requirements are settled; archive compression is an alternative.

## Caches and other data

- `experiments/temporal_hypotheses_12h_20260906/cache`: **52.60 GiB**, derived
  training features. Current MACE ablation config explicitly references it as
  `source_cache`; preserve during the active queue.
- `experiments/pretrained_mace_spatiotemporal_20260906/uniform_training`:
  **5.54 GiB**, explicitly referenced by that queue as `reuse_prepared_cache`.
- `experiments/pretrained_mace_80_dt01_cosine_20260906`: **2.75 GiB**, also in use.
- `experiments/pretrained_mace_spatiotemporal_20260906/static_context`:
  **4.42 GiB**, derived analysis data; a later candidate after checking queued
  analysis configs. Keep checkpoints, results, and manifests.
- `data/saved_features_boost` contains **8.85 GiB** of related Parquet datasets,
  including a **2.20 GiB** `black_marble_stale_backup` file. This belongs to another
  research workflow; filenames alone do not establish duplicate content or
  permission to retire it. Do not treat differently engineered features as copies.

## Compression

Use lossless compression for cold diagnostic trajectories that must be retained.
A middle 8 MiB sample of the old Ta text compressed to 48.1% of its apparent
bytes with zlib level 1; a sampled training-cloud NPY compressed to 86.7%.
These are sample ratios, not whole-file savings. Filesystem allocated sizes
already differ from apparent sizes, so compare final allocated space directly.
Do not compress active memory-mapped NPY arrays in place: readers require their
current files. Avoid float32-to-float16 changes merely for storage savings; that
changes scientific data. Prefer verified text removal over retaining compressed
and binary representations indefinitely.

Only the GPU allocation appeared in Slurm during this review. That does not mean
the IDS datasets are idle: a MACE training queue is running locally, and the Ta/Ti
sequence is managed by a remote systemd service outside Slurm. No source jobs
were submitted or repaired as part of this storage review.

Disposable inventories and exact archived-attempt candidate paths are saved in
`output/ids_storage_review_20260907/`. This document is an operational review,
not a new tool or experiment runner.
