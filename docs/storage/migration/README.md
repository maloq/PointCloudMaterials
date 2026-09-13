# IDS storage migration — 2026-09-13

Move the six IDS research directories out of `/home/ids/vmorozov` while preserving dataset bytes, restart state, historical provenance and existing loader paths.

## Placement

| Directory | Destination |
| --- | --- |
| simulations | `/work/PERSO/vmorozov/simulations` |
| training-cache | `/work/PERSO/vmorozov/training-cache` |
| analysis | `/work/PERSO/vmorozov/analysis` |
| experiments (older result/cache trees) | `/store/PERSO/vmorozov/experiments` |
| data (geographic/fire datasets) | `/store/PERSO/vmorozov/data` |
| models (geographic/fire models) | `/store/PERSO/vmorozov/models` |

WORK exposes a 512 GiB filesystem limit. The six source directories contain 578,571,519,989 logical bytes (538.84 GiB), so all six cannot fit there. SCRATCH is not used for persistent datasets because it is automatically purged.

## Procedure and evidence

The explicit mapping is [technical/plan.json](technical/plan.json). Copy each listed source directory into its destination parent with `rsync -aH --info=progress2 --stats SOURCE DESTINATION_PARENT/`. Check the exit status. The existing inventory implementation `src.experiment_runner.cache_storage._inventory` computes SHA-256 for every regular file and records symlink targets. Compare source, destination, and a second source inventory; abort on any mismatch. Preserve the inventories before changing source paths. Replace each verified source directory with a compatibility symlink, and remove its displaced original only after the alias resolves correctly.

Maintained configs, simulation/reproduction configs and repository links use the new roots. Exact-resume forecast configs remain unchanged because the trainer requires byte-equivalent configuration values; their old storage paths forward through compatibility links. Historical manifests, hash-bound snapshots and submitted job scripts retain their original bytes and resolve through compatibility links. Repository source code remains in its existing checkout.

Slurm allocations 990987, 991149 and 991141 were inspected before copying: no Python training or LAMMPS process was present. WORK and STORE visibility was checked from both remote GPU allocations.

Logs, per-file verification records and final status are under [the migration output](/store/PERSO/vmorozov/projects/PointCloudMaterials-20260913T174741Z/output/storage_migration/ids-to-work-20260913/README.md). Migration completed at 2026-09-13T15:47:44.308598+00:00. All six directories passed verification and cutover; their old IDS paths remain usable as symlinks. Reclaimed 507.81 GiB on IDS. The completion receipt is [technical/completion.json](technical/completion.json). No scientific protocol or array representation changes are part of this migration.

This directory is an operational migration record. Copy logs and inventories in the output directory are operational artifacts; no new maintained command is introduced.

The TDA view loader and completed-cache verifier now compare resolved source directories, so a relocated cache remains usable without editing its hash-bound manifest. `tests/test_spatiotemporal_tda.py` passed (2 tests), including verification and loading after a move with a compatibility alias.
