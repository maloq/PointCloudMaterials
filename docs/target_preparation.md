# Spatial target preparation

The maintained producers are `patch_descriptors` and `patch_descriptor_batch` in
`src/research/liquid_predictability/descriptors.py`. Float64 bond-order reductions
use `src/postprocessing/descriptors/rich.py`; new cache preparation defaults to
native batched alpha persistence and summaries in `alpha3d.cpp`, loaded through
`alpha3d.py` in the same directory.
These are local **spatial encoder targets**, separate from full-cell PTM and
nucleus-search descriptors. Alpha complexes still run on CPU; no GPU is required.

The existing complete 442-column cache remains reusable. There is no reason to
recalculate it for a new encoder fit. New preparation uses the optimized producer
automatically; frozen execution snapshots retain their own code.

## What changed

- Four even spherical-harmonic orders share a compiled Legendre recurrence.
  Invariants, Wigner contractions and coherence summaries use compiled reductions.
- Both GUDHI alpha complexes, H0/H1/H2 persistence, log moments, Gaussian images
  and smooth Betti curves now share a native batch call. A smaller simplex-tree
  representation retains double-precision filtration values. This uses the same
  safe GUDHI `Alpha_complex` algorithm; no alternative filtration is introduced.
  The Python GUDHI binding already selected fixed 3D internally, so dimensional
  specialization alone does not explain the improvement.
- Current spatial training uses only 40 bond-order and 258 TDA columns. A declared
  two-family preparation skips 99 geometry and 45 CNA columns entirely. All 298
  retained names, ordering, neighborhoods and target formulas are preserved.
- CPU workers read the immutable plan once per worker. Existing bounded checkpoint
  blocks, input hashes, ancestry, material normalization and resumable shards remain.

No fast-math, filtration cutoff, downsampling or neighborhood approximation is
applied. Calculations stay float64 and stored targets float32. The support remains
the centered nearest-80 patch, cropped at radius strictly below 8 in the existing
material-normalized coordinates. No extra encoder input is introduced.

## Current target: bond order, CNA and active TDA

The requested CNA addition changes the new spatial objective to **198 outputs**:
40 bond-order, all 45 CNA, and 113 active TDA components. CNA includes the existing
fixed-3.2, fixed-3.6 and adaptive-12-neighbor fingerprints: coordination, bond
signature frequencies, common-neighbor counts, internal bonds and chain moments.
All 45 CNA components have nonzero training weight under the existing variation
threshold. Geometry descriptors remain omitted.

Use [the 198-column recipe](../configs/data/spatial_active_cna_preparation.json)
and its [training-only mask](../configs/data/spatial_active_cna_targets.json) for
new preparation. New training with this selection gives **one third of descriptor
weight to each family**; TDA retains equal weighting across its 18 groups. This is
an explicit objective extension, not a relabelling of the earlier bond-order/TDA
fits. The descriptor head resolves its output count from the same pinned names.
The training consumer can select them from the existing full cache, or load a new
compact 198-column cache. A new derived validation panel is required and checked.

```bash
python -m src.postprocessing targets plan --config configs/data/spatial_active_cna_preparation.json
python -m src.postprocessing targets prepare --config configs/data/spatial_active_cna_preparation.json --lane 0
python -m src.postprocessing targets seal --config configs/data/spatial_active_cna_preparation.json
```

The `targets selection` command now defaults to `bond_order cna tda`; pass
`--families bond_order tda` only to reproduce the earlier two-family selection.
The 198-column target array is 55.2% smaller than the original 442-column array,
or 42.3% smaller than calculating all 343 bond-order/CNA/TDA values. Coordinate
storage is unchanged. Existing full caches remain useful; no full re-extraction
is needed merely to add CNA to training, because those caches already contain it.

The final CNA-inclusive CPU audit used the same 936 Al/Mg/Ti/Ta patches and four
workers on nodecpu04, with three repetitions: original 442 values **3.965 s**,
native full 343 values **1.459 s**, active-only 198 values **1.413 s**. This is
**2.81× faster than the original producer**, and 3.28% more throughput than native
343-output preparation. These timings exclude I/O and warmup, as above. Retained
values are identical to full native outputs; CNA and bond order match the original
reference exactly. Maximum TDA drift remains 5.69e-14. Compact packing, equal
family weights, partial resume and threaded output also passed a 112-row audit.
See [the final numerical receipt](target_preparation_cna_20261008.json) and
`output/target-preparation/active-cna198-20261008/technical/`.
Only audits were executed; training and complete target regeneration were not launched.

## Earlier active-only comparison without CNA: October 8

The current spatial training transform activates **40 bond-order and 113 TDA
columns**, for **153 outputs**. The other 145 TDA columns have training SD below
1e-4 and zero objective weight. Their omission is based only on the frozen
training transform, not validation R2 or whether a target is easy to learn.
[The pinned selection](../configs/data/spatial_active_targets.json) records the
1,056,768-row fitting identity, transform checksum, names and ancestry. Its mask
matches the actual positive weights of the completed MACE/eSEN comparison.
This selection belongs to that training population; a different fitting cohort
needs its own training-only mask or an explicit decision to retain this objective.

Native preparation skips unused statistics, Gaussian image bins and Betti-curve
points inside the C++ kernel. Both n32/n80 alpha complexes and finite H0/H1/H2
persistence are still needed: all six neighborhood/dimension combinations retain
active targets. Outputs are compact; omitted values are not stored as zeros.

Matched four-worker CPU timings on nodecpu04, 936 patches, median of three:

| Preparation | Seconds | Speedup over original |
|---|---:|---:|
| Original 442 columns | 3.963 | 1.00× |
| Previous compiled reductions, spatial 298 | 2.029 | 1.95× |
| Native batch, spatial 298 | 1.418 | 2.80× |
| **Native batch, trained 153** | **1.368** | **2.90×** |

Active-only preparation adds **3.67% throughput** (3.54% less compute time) over
the native full-spatial batch path. The target matrix is **48.7% smaller than
298 columns**, or 65.4% smaller than 442; positions are unchanged. Shared alpha
construction/persistence dominates computation, so dropping 49% of outputs does
not halve preparation time. Retained float32 values match the native 298-column
path exactly across the audit sample. Against the original reference the largest
TDA difference remains 5.69e-14. A 112-row producer audit also covered compact
packing, actual spatial-consumer loading, resumability and two-thread equality.

The [two-family preparation recipe](../configs/data/spatial_active_preparation.json)
records this intermediate comparison; use the CNA recipe above for new work.
The complete corpus has not
been rebuilt. The numerical and timing receipts are in
`output/target-preparation/active153-20261008/technical/` and
[the durable active-target receipt](target_preparation_active_20261008.json).

```bash
python -m src.postprocessing targets plan --config configs/data/spatial_active_preparation.json
python -m src.postprocessing targets prepare --config configs/data/spatial_active_preparation.json --lane 0
python -m src.postprocessing targets seal --config configs/data/spatial_active_preparation.json
```

For another already-fitted training population, freeze its mask explicitly:

```bash
python -m src.postprocessing targets selection --packed-cache TRAINING_PACKED_CACHE --selection-output NEW_SELECTION.json --families bond_order tda
```

Bind the selection path and SHA256 under `descriptors.selection` as in the recipe.
The source mask is not recalculated from selection/test data. Compact caches need
their own fitting reference and transform; the existing packer verifies them
against their actual training rows. Spatial training consumers and descriptor
heads resolve the declared column names/count, preserving the existing block
weights. A 153-output head is for a new fit, not a replacement head for an existing
298-output checkpoint. Existing fitted models, caches and frozen results remain
unchanged. Full diagnostic descriptors are still available explicitly.

## Native batch measurements: October 8

936 existing patches from 117 Al/Mg/Ti/Ta sources; medians of three repetitions
on nodecpu04, Intel Xeon Gold 6154. Each comparison uses the same observations,
CPU allocation and worker count. Four processes each use one native thread.

| Preparation | One worker | Four workers | Speedup, four workers |
|---|---:|---:|---:|
| Original 442 columns | 14.459 s | 3.946 s | 1.00× |
| Previous optimized 442, safe alpha | 8.683 s | 2.373 s | 1.66× |
| Previous spatial 298, safe alpha | 7.420 s | 2.028 s | 1.95× |
| Native 442, scalar calls | 6.638 s | 1.808 s | 2.18× |
| Native spatial 298, scalar calls | 5.438 s | 1.485 s | 2.66× |
| **Native spatial 298, batched** | **5.070 s** | **1.399 s** | **2.82×** |

The new batch path is **1.45× faster than the previous spatial-only optimization**
(31.0% less computation time), with the same 298 outputs. The 2.82× comparison
also includes omitting unused geometry/CNA columns. It is not an equal-column
algorithm-only comparison. Native full-442 scalar preparation is 1.31× faster
than the previous full-442 path. These are steady-state descriptor timings,
including process dispatch/result transfer but excluding input loading, shard
hashing/writing, warmup and compilation; they are not full-corpus or training times.

All native variants matched the previous safe optimization's float32 outputs
exactly on this sample. Against the original producer, bond-order/geometry/CNA
were exact and TDA differed by at most 5.69e-14 raw, or 1.73e-10 training SD.
On this CPU the original producer also matched the sealed cache exactly.
Native compilation took 45.7 seconds once; subsequent processes reused its cache.

The actual preparation workflow also passed a fresh seven-shard, 112-row audit
across all four materials and Al train/selection/calibration/test roles: completed
reuse, byte-identical partial resume, changed-backend rejection, finite sealed
298-column loading, two-thread batch agreement and full-442 agreement.

Receipts: `output/target-preparation/native3d-20261008/technical/` and
[durable native benchmark receipt](target_preparation_native_20261008.json).
Slurm audit 1028605 completed; no corpus rebuild or training was launched.
An earlier attempt on nodecpu11 stopped with SIGILL during environment warmup,
before a native extension build was produced. Compatibility of that shared Python
environment on nodecpu11 remains unverified; it is not included in timing claims.

## Earlier reduction measurements: October 8

936 existing unrelaxed patches: eight spread through the first packed shard of
each of 117 training sources, covering Al/Mg/Ti/Ta. Medians of three repetitions
on nodecpu10, Intel Xeon E5-2643 v4; identical observations and worker counts.
Compilation, input sampling and publication are excluded; process scheduling and
result transfer are included in the four-worker timing.

| Preparation | One CPU worker | Four CPU workers | Speedup, four workers |
|---|---:|---:|---:|
| Original 442 columns | 9.487 s | 2.489 s | 1.00× |
| Optimized 442, safe alpha | 5.517 s | 1.481 s | **1.68×** |
| Spatial 298 only, safe alpha | 4.404 s | 1.202 s | **2.07×** |
| Spatial 298 only, fast alpha (opt-in) | 3.554 s | 0.968 s | 2.57× |

Safe-mode bond-order values matched the original producer exactly. The largest
TDA difference was 5.69e-14 in raw units, or 1.73e-10 of a target's training SD.
Geometry and CNA also matched exactly when requested. Single/process-pool output
was identical. These are preparation timings, not training-speed measurements or
full-corpus completion estimates. Dropping unused columns saves 32.6% of the target
matrix's storage; coordinate storage is unaffected.

Fast alpha had a largest TDA difference of 0.1113 (0.464 training SD), with RMS
standardized difference 0.00319 across TDA values. It can change tiny intervals
and count statistics. It remains explicitly opt-in and requires a cohort-specific
audit; a small average error does not guarantee small errors for every target.

## Preparing new targets

Reuse the existing preparation configuration's `structural_dataset`,
`fixed_dataset`, `evaluation`, `preparation` and `protocol` fields. Choose a **new
external cache path** and, for spatial-only targets, add:

```json
"descriptors": {
  "families": ["bond_order", "tda"],
  "alpha_precision": "safe",
  "alpha_backend": "native"
}
```

Omitting `descriptors` retains all four original families, with 442 columns,
using the native safe backend. The family set, backend and precision enter the
new cache identity. Changed code or
precision cannot resume a partial release. Historical sealed caches retain their
original identities and values.

```bash
conda activate pointnet-torch214
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/home/ids/vmorozov/training-cache/compiled/rich-targets
python -m src.postprocessing targets plan --config CONFIG.json
python -m src.postprocessing targets prepare --config CONFIG.json --lane 0
# Run each configured lane once, then:
python -m src.postprocessing targets seal --config CONFIG.json
```

For Slurm arrays, `prepare` uses `SLURM_ARRAY_TASK_ID` when `--lane` is omitted.
Allocate `preparation.workers` CPU cores per lane. These commands prepare data
only: they do not submit training or create W&B runs. CPU timings are a separate
`target-benchmark` command, never part of a training launch.

`CachedPatches` reads either family set, with train-only transforms. The spatial
consumer resolves its 298 columns by family and accepts either backing schema.
A new release needs its own fitting subset/transform and packed cache; do not
reuse a 442-column transform for 298 columns or change a historical data identity.

For the Python GUDHI reference path, explicitly set `alpha_backend: "gudhi"`.
This is also required for the optional `alpha_precision: "fast"`; native mode
supports safe precision only. Direct scalar `patch_descriptors` callers retain
the GUDHI backend unless they request native explicitly. Cache preparation uses
`descriptor_options`, which selects native by default, then runs bounded batches.

## Native dependency setup

Use `pointnet-torch214` for Python. GUDHI headers and their C++ dependencies live
in separate external directories; the loader does not install packages or download
source. Pin GUDHI 3.11.0 to revision
`84974f4ebb9e7c29cfc02b35edb8a9bd5ff4a55e` (tag `tags/gudhi-release-3.11.0`),
and install these conda-forge dependency builds into a separate build prefix:

```text
cgal-cpp=5.6.1=hd557e15_2
libboost-headers=1.86.0=ha770c72_5
gmp=6.3.0=hfd2156b_3
mpfr=4.2.2=ha2cb11d_1
eigen=3.4.0=h54a6638_2
```

Set local paths under the existing `execution` mapping in ignored
`machine.local.yaml`:

```yaml
execution:
  rich_alpha:
    gudhi_source: /external/software/gudhi-3.11.0-src
    dependencies: /external/software/tda3d-build
    cxx: g++
```

The compiler needs C++17/OpenMP; Python needs pybind11. Builds use `-O3` without
fast-math, double filtration values and CPU-specific compilation. Cached binaries
and locked build receipts live in `${storage:cache}/compiled/rich-alpha/`, keyed
by source, dependency metadata, compiler, Python ABI and CPU model/instruction
flags. Shards record their native build receipt and hash. Missing or changed
dependencies fail explicitly. CPU-specific binaries are not shared across unlike
nodes. Standalone batches can request `threads=N`; process-based preparation
keeps this at one to avoid multiplying worker and OpenMP thread counts.

## Conditioning issue found during validation

Normalizing a nearly zero order tensor amplifies tiny rounding differences into
visible coherence differences. When any tensor norm for an order is below 1e-8,
the optimized path calculates that order's harmonics with the original SciPy
arithmetic. The normalization floor remains 1e-14; the target is not redefined.

The unchanged reference itself differed from the sealed cache on nodecpu10 in
32 of 37,440 bond-order values: maximum difference 0.158, or 0.769 of that feature's
training SD. This is concentrated in nearly zero l=2 coherence. Geometry, CNA and
TDA matched the cache exactly. Optimization drift is therefore checked against
the reference **on the same CPU**, while reference-to-cache drift is reported
separately. On node60, optimized safe outputs matched all 936 cached rows within
1.73e-10 training SD. Existing target caches were not changed. A future scientific
target revision could define coherence only for sufficiently resolved tensors;
that would need a separate target contract and evaluation.

The actual producer audit prepared seven material/role shards (112 rows), sealed
and loaded the 298-column cache, verified completed-shard reuse, resumed a partial
shard with byte-identical outputs, and rejected changed precision. Full planning
preserved all 22,409 original tasks, counts and fixed material factors. Only these
small audits were computed; no corpus-wide rebuild or training was submitted.
The existing packing workflow and actual spatial-training consumer also loaded
the 298-column audit release successfully, with finite targets and unit-sum weights.

Receipts: `output/target-preparation/optimization-20261008/technical/`.
Disposable arrays: `${storage:cache}/target-preparation/optimization-20261008/`.
See [metric definitions](metrics/spatial_target_preparation.md).
The [durable numerical receipt](target_preparation_20261008.json) retains timing
repetitions, software versions, numerical errors and the preparation audit.
