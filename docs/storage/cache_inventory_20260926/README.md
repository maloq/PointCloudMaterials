# Research cache storage — 26 September 2026

**Later update:** the authorized [683.461 GB cleanup](../cache_cleanup_20260927/README.md)
completed on September 27. The assessment and sizes below describe the earlier snapshot.

Observed 2026-09-26T15:08:25.478556+00:00 to 2026-09-26T15:09:00.692408+00:00.

**1044.200 GB** of identified data/feature caches.
Plus **13.905 GB** of archived relaxed structures and preparation records: **1058.106 GB** combined.

GB means 1,000,000,000 allocated bytes, not GiB. One GNU `du -x -B1` invocation measured all disjoint roots, counting hard-linked files once and not following symbolic links. Distinct physical archive copies count toward disk usage; matching family names do not establish identical contents. Per-family attribution of shared hard links follows traversal order. This is an observation during active training and storage relocation, not an atomic quota measurement.

Scope: IDS training caches; SCRATCH training/geometry/features; both STORE training-cache archives; registered analysis caches and discovered cache directories/producer sidecars in current outputs, WORK analyses, STORE experiments and the archived repository; archived relaxed cells. It excludes raw MD trajectories, general model checkpoints/results, software package caches, and remote-only holdings. No cache was deleted.

| Cache family | Allocated GB |
|---|---:|
| Spatial-context forecasting | 613.298 |
| Crystallization transfer | 73.251 |
| Full embedding forecasting | 70.210 |
| Equivariant-context training | 58.628 |
| Structural encoder pretraining | 53.320 |
| Shared context geometry | 52.039 |
| Other training and derived data caches | 37.251 |
| Neighborhood JEPA | 31.588 |
| Relaxed structures and their training caches | 26.235 |
| Analysis and historical output caches | 17.559 |
| Distance encoder | 13.104 |
| Structured crystallization | 10.867 |
| Generated encoder features | 0.755 |
| Root metadata and links | 0.001 |

Dedicated cache roots:

| Location | Allocated GB |
|---|---:|
| `/home/ids/vmorozov/training-cache` | 496.006 |
| `/scratch/PERSO/vmorozov/PointCloudMaterials/training-cache` | 70.778 |
| `/store/PERSO/vmorozov/training-cache-archive-20260925` | 360.619 |
| `/store/PERSO/vmorozov/training-cache-archive-20260926` | 99.238 |

[Per-directory/file breakdown](collections.csv) · [Raw allocated-byte measurements](du_allocated_bytes.tsv) · [Summary](summary.json)

The largest family, spatial-context forecasting, is present on IDS and STORE. This report measures space; it does not establish which files can be safely retired. Generated encoder-feature caches are separately listed; the six-entry retention policy and active leases still apply. Preserved relaxed structures are expensive derived scientific data, not interchangeable with disposable inference features.

[Simulation cleanup and Al cadence options](../../simulations/cleanup_20260926/README.md) are recorded separately.
