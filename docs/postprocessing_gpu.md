# GPU structural extraction

GPU PTM is implemented. On the audited A100, the first complete CUDA extractor
was **10.6–26.0× faster than the eight-physical-core CPU implementation** on six
real Al/Ni frames, including million-atom Al. This measurement includes reading,
all four structural detectors, transfers and NPZ compression; it excludes disk
writes and chronological lineage. The bounded streaming measurements below
add writing and overlap CPU input/output with GPU work.

The preferred API is `descriptors.gpu.structure(..., backend="resident",
ptm_backend="voronoi")`. PTM labels stay on device until all component work is
finished. Correlation/histogram and solid-mask/count passes are fused; component
size counting aggregates atomics within warps and reuses resolved roots.
A shared-neighbor-search `joint` variant was measured separately: it adds little
benefit and is slower on the million-atom liquid case, so it is not the default.

## Scientific and numerical scope

The CUDA matcher retains OVITO 3.15.3's FCC/HCP/BCC reference templates, convex
hull graph matching, rotation fit, and RMSD threshold. It uses float64 geometry
and a GPU Voronoi-face clipping implementation to order the 18 nearest neighbors
by solid angle. See the pinned [source/licenses](../src/postprocessing/descriptors/gpu/NOTICE.md).
It exports structural types; orientation, strain, alloy types and other
templates are not implemented. Bond-order arithmetic remains float32.

User tolerance: up to 5% result differences may be acceptable for a substantial
speed gain. We report atom-label, rare-crystal, component and event differences
separately, rather than dividing all errors by the overwhelming liquid count.
The preferred `voronoi` path does not deliberately approximate neighbor ordering.
`distance` is a separate approximation; it missed 1.02% of reference crystal atoms
in one liquid million-atom frame and changed some per-event size traces. Its
small additional speed gain is unnecessary for reaching the 6× target.

Six-frame audit: Voronoi GPU PTM labels and all detector component labels were
identical to CPU. Bond histogram L1 count differences were 4, 0, 16, 8, 4, 0;
these are tiny bin-edge changes in the existing CUDA bond arithmetic.

Contiguous-window audit used **161 Al frames at 89.7–105.7 ps** and **61 Ni frames
at 295.0–301.0 ps**, both native 0.10 ps. It ran the real CPU lineage tracker on
both sets of extracted labels, with identical window boundaries and censoring.
PTM events, interactions, size histories, onset times and fates were exactly
equal. Al had five liquid-emergence records, one established emergence; Ni had
one dissolved emergence. These are window-specific records, not new catalogue
totals. The loose coherent detector changed Al birth records from 874 to 875
(0.114%); standard/strict coherent detectors and all Ni counts were unchanged.
This audit does not prove a <5% error bound for every future trajectory/material.

## First full-GPU timing

A100 PCIe 40 GB on node54, eight physical CPU cores allocated. Three alternating
repetitions, warmed JIT/filesystem, same original CPU producer and input frames.
The CPU reference includes its serial component work; these are application
wall times, not an estimate obtained by multiplying single-core speed.

| Frame | CPU, 8 cores (s) | First full GPU (s) | Speedup |
| --- | ---: | ---: | ---: |
| Al-70k liquid | 0.7737 | 0.0513 | 15.1× |
| Al-70k crystalline | 0.9239 | 0.0870 | 10.6× |
| Al-1m liquid | 14.3230 | 0.5499 | 26.0× |
| Al-1m crystalline | 15.7271 | 0.8897 | 17.7× |
| Ni-32k liquid | 0.3710 | 0.0277 | 13.4× |
| Ni-32k later liquid | 0.3977 | 0.0296 | 13.4× |

The final resident implementation also reproduced every saved array across the
222 audited birth-window frames, so the earlier event comparison still applies.

Later paired five-repeat comparisons found the resident/fused-reduction path
another **3.7–5.3% faster** than separate GPU stages on these cases. All exported
arrays matched the separate-stage GPU implementation exactly. Keep this paired
comparison distinct from the earlier CPU comparison; clocks and host contention
can change across runs. Chronological lineage and compression remain on CPU.

## Streaming measurements

The same exact frames/chunks were extracted sequentially and with bounded
prefetch/compression overlap. These timings include reading, extraction, NPZ
compression, file writes and output checksums. They exclude initial raw-input
verification and compilation. All saved arrays were exactly equal. One timing
per configuration/window; these are pilot throughput measurements, not confidence
intervals. Prefetch was 2 and chunks contained 8 frames.

| Native window | Frames | Sequential GPU (s) | Stream, 1 writer (s) | Extra speedup |
| --- | ---: | ---: | ---: | ---: |
| Al-70k liquid | 64 | 2.102 | 1.795 | 1.17× |
| Al-1m liquid | 32 | 14.940 | 12.010 | 1.24× |
| Al-1m crystalline | 32 | 26.044 | 17.086 | 1.52× |
| Ni-32k liquid | 64 | 1.115 | 1.018 | 1.09× |

Two compression workers improved the small Al/Ni windows to 1.30×/1.21× but
did not improve the million-atom cases. One writer remains the conservative
default; increase it explicitly when measurements justify the CPU cost.

## CPU-pool comparison

A separate throughput audit processed the same six adjacent million-atom liquid
frames (3125–3130) with **six simultaneous one-thread workers on six distinct
physical CPU cores**, then one A100 with its host process pinned to one core.
All workers were warmed first; timings include reads, extraction, transfers and
NPZ compression, but not output writes or lineage.

The CPU pool took **63.578 s**; the GPU took **2.794 s** (median of three batches),
a **22.76× throughput advantage over that entire six-worker pool**. CPU worker
completion times were 62.45–63.58 s. This is one CPU batch and one workload, not
a general conversion from GPUs to CPU-core counts or a whole-campaign ETA.

## Bounded streaming extraction

```bash
python -m src.postprocessing gpu-extract --config RESOLVED_GPU_CONFIG
```

The JSON arguments are `source` (the full pinned source record), `frames` (its
exact requested frame indices), `config` (the original descriptor config),
`output` (an external cache directory), and optional `chunk_frames` (16),
`prefetch` (2), `compression_workers` (1), `ptm_ordering` (`voronoi`),
`execution` (`resident`), and `input_receipt`. Use the existing native-cadence
view unchanged. No interpolation, subsampling or new event definition is applied.

A single reader prefetches a bounded number of frames. One GPU computes them
in order while bounded CPU writer threads compress completed chunks. This never
loads a whole trajectory into VRAM. Outputs use the same separate detector NPZ
members as CPU extraction, with distinct execution/configuration/input receipts.
Completed chunks are checksummed and reused after interruption; changed inputs,
settings or code require a separate output. Exceptions from readers/writers
propagate. Only completed chunks receive receipts.

`input_receipt` may point to the existing CPU source's `input-verified.json`. Its
input identity must match; the reused receipt and its hash are recorded. Without
one, raw arrays are verified once before extraction. Steady-state benchmark figures exclude verification and compilation. In a cold
CLI invocation, extraction seconds can include JIT/startup; raw verification
happens before that timer. Caches must live outside the repository.

Supports finite orthorhombic, periodic full cells with the declared four-detector
l=6 protocol. PTM search radius is twice the bond cutoff; each box length must
be at least three PTM search radii. The code fails on fewer than 18 PTM neighbors,
more than 64 bond neighbors, bins above 128 atoms, unbounded Voronoi faces or
nonfinite values. It never silently truncates neighborhoods or falls back to CPU.

Use `pointnet-torch214` and a CUDA toolkit matching PyTorch. Place the Torch
extension cache outside the repository. Benchmarks/evaluation create no W&B runs.
Existing production searches, caches, simulations and birth catalogues are unchanged.
This provides an opt-in extractor; it does not silently convert CPU Slurm lanes
into GPU reservations. The current numerical/performance evidence is for A100;
other GPU types, especially those with weak float64 arithmetic, need measurement.

The explicitly requested [low-temperature Ni search](simulations/ni_births_lowT_010_20261008.md#gpu-nucleus-search-october-8)
now uses this API through `ni-gpu-search`. It queues GPU extraction after the
simulations and unchanged CPU lineage afterward, with separate GPU caches and
birth exports. This opt-in production use preserves the historical CPU catalogues.

The measured source snapshots, configs, local audits and launch receipt are under
`output/postprocessing/gpu-ptm-20261008/technical/`. The [durable combined record](postprocessing_gpu_ptm_20261008.json) retains timings, numerical comparisons and implementation hashes. Historical hybrid measurements
and the reason PTM needed a port are retained below.

---

# Initial hybrid prototype (historical measurements)

The experimental backend lives in `src/postprocessing/descriptors/gpu/`.
It is opt-in and is not selected by the production search or history queues.
This is a feasibility and numerical-equivalence benchmark, not a new nuclei
dataset or a change to any detector threshold.

Two paths are compared with the existing CPU `descriptors.structure` producer:

| Backend | CPU work | GPU work |
| --- | --- | --- |
| `freud` | OVITO PTM, freud neighbors and l=6 harmonics | Ordered bond correlations, degrees, connected components, histogram |
| `cuda` | OVITO PTM | Periodic cell lists, neighbors, l=6 harmonics, correlations, degrees, connected components, histogram |

Component labels are canonicalized by the first atom in each component, as in
the reference CPU producer. Union operations link larger root indices to smaller
ones. Components below the existing minimum size are excluded without changing
the solid-atom counts. Directed edges and strict bond thresholds are retained.
There is no temporal subsampling or spatial cropping.

The CUDA harmonic kernel specializes the pinned freud/fsph arithmetic to l=6,
including float32 accumulation order, periodic wrapping, and the separate
query-image distance calculation. CUDA and CPU transcendental functions can
still differ near classification and histogram boundaries. Every comparison
records neighbor equality, q6/correlation differences, accepted-bond differences,
solid masks, component labels, counts and histograms. Exactness on these frames
does not establish exactness on every trajectory. See the source
[numerical references and licenses](../src/postprocessing/descriptors/gpu/NOTICE.md).

Supported input is one finite, wrapped orthorhombic frame with the current four
detectors. Each box length must exceed three neighbor cutoffs. The prototype
allows 64 neighbors per atom and 128 atoms per cell-list bin; overflow raises an
error and never truncates a neighborhood. It does not yet batch frames, overlap
CPU PTM with GPU work, implement GPU PTM, or move the chronological family state
machine to GPU.

## Reproduce the local audit

Use conda `pointnet-torch214`, a matching CUDA toolkit, and an allocated GPU.
Set `TORCH_EXTENSIONS_DIR` to an external cache directory. This command does not
create a W&B run or alter production caches:

```bash
python -m src.postprocessing gpu-benchmark --config RESOLVED_BENCHMARK_CONFIG
```

The resolved config contains `output`, one `threads` value, `repeats`, and
`cases`. Each case includes a source record from an existing pinned extraction
plan, its exact `frame`, a descriptive `name`, and `descriptor_config` from that
same plan. Use a fresh process for each CPU thread limit because OVITO reads its
limit at import. Outputs are synchronized wall times and numerical comparisons
in `results.json`, with input identities and implementation hashes.

The October 8 allocation and resolved inputs are under
`output/postprocessing/gpu-prototype-20261008/technical/`. Slurm job 1027981
reserves one A100 and four logical CPUs for this local benchmark only.
The ongoing Al/Ni production jobs retain their original code and caches.

## Measurement scope

`after_ptm` measures neighbors, harmonics, bond correlations, component labels,
counts and histograms with already-computed PTM types. Allocation and host/device
transfers are included. `full_frame` additionally reads the frame, computes fresh
CPU PTM and compresses the same output arrays into NPZ in memory. It excludes
shared-filesystem output writes, whole-trajectory checksum scans and subsequent
lineage tracking. Therefore `full_frame` is not a complete campaign wall-time
measurement. Compilation and startup are recorded separately. Final comparisons
use three repetitions in alternating order and their medians.

PTM-only precursor tails cannot inherit a speedup measured for the full
four-detector pipeline: PTM remains on CPU in both GPU variants.

## Results: 2026-10-08

On one NVIDIA A100 PCIe 40 GB, the `cuda` path reduced measured full-frame time
by **1.82–1.87×** across six real frames from three sources. Both paths used the
same four CPU threads on node04 (AMD EPYC 7302; the allocation comprised two
physical cores and their four SMT threads). This is not a comparison with four
physical CPU cores or an estimate for every GPU/CPU machine. No other work used
the allocated GPU. Times below are medians of three alternating repetitions.

| Frame | PTM-crystalline atoms | CPU full frame (s) | CUDA + CPU PTM (s) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Al-70304-0 | 37 | 1.3446 | 0.7403 | 1.82× |
| Al-70304-3000 | 60,272 | 1.4094 | 0.7614 | 1.85× |
| Al-1000000-3125 | 1,081 | 23.1555 | 12.6770 | 1.83× |
| Al-1000000-6000 | 827,608 | 24.5520 | 13.1001 | 1.87× |
| Ni-32000-0 | 0 | 0.5674 | 0.3079 | 1.84× |
| Ni-32000-3000 | 4 | 0.6161 | 0.3317 | 1.86× |

The remaining descriptor/component stage, with PTM already available:

| Frame | CPU (s) | CUDA (s) | Speedup |
| --- | ---: | ---: | ---: |
| Al-70304-0 | 0.5928 | 0.0048 | 123.0× |
| Al-70304-3000 | 0.6854 | 0.0275 | 24.9× |
| Al-1000000-3125 | 10.4225 | 0.0480 | 217.0× |
| Al-1000000-6000 | 11.9960 | 0.1594 | 75.3× |
| Ni-32000-0 | 0.2614 | 0.0035 | 74.3× |
| Ni-32000-3000 | 0.2816 | 0.0036 | 77.9× |

The large isolated-stage gains do **not** apply to complete searches or PTM-only
tails. CPU PTM remains the dominant cost. The fully reference-preserving `freud`
variant gives only **1.03–1.09×** full-frame speedup. CUDA tensor allocations
peaked near **1.00 GiB** for one million atoms; this excludes CUDA context and
allocator reservation overhead. No full trajectory was loaded into GPU memory.

### Numerical outcome

- All six frames: identical neighbor lists, accepted bonds at all three
  thresholds, solid masks/counts and atom-to-component labels for all four detectors.
- `freud` variant: every exported scientific array matched exactly.
- `cuda` variant: histogram arrays matched on two frames. Histogram L1 count
  differences were 4, 0, 16, 8, 4 and 0 in the table order. Maximum q6 difference
  was 2.3842e-7; maximum bond-correlation difference was 1.1102e-6.
- These sampled agreements do not establish unchanged event times/fates on a
  complete trajectory. The chronological lineage implementation was not ported
  or benchmarked here. Production extraction, caches and search jobs were untouched.

The evidence supports a useful GPU extraction prototype, with a numerical
boundary audit still required before production adoption. A substantial further
full-pipeline gain requires accelerating PTM itself; it is still the original
CPU OVITO call. PTM-only history extraction has not been accelerated by this work.

The [durable measurement record](postprocessing_gpu_20261008.json) contains all
repetitions, input identities, agreement results and exact implementation hashes.
The measured source snapshot, resolved source timelines and launch receipt remain
in `output/postprocessing/gpu-prototype-20261008/technical/`. The allocation was
released after successful completion. No new production jobs were launched.

## Original adoption decision: at least 6×

User requirement, 2026-10-08: GPUs are scarcer than CPUs; a production switch
needs at least a 6× speedup. The current 1.82–1.87× prototype does not qualify.

For the million-atom crystalline frame, the recorded CPU time is 24.5520 s,
the hybrid time is 13.1001 s, and its GPU descriptor/component stage is 0.1594 s.
Against that specific baseline, 6× requires at most 4.0920 s per frame. Even
eliminating the GPU stage entirely leaves about 12.9408 s, for only 1.90×.
Thus tuning the existing CUDA kernels cannot meet the target while the remaining
CPU work stays unchanged. The residual stage needs roughly 3.3× acceleration;
PTM is its dominant component. These are a timing-budget calculation, not a
measured GPU PTM result.

The next useful prototype is GPU PTM with the same active structures and RMSD
definition. A 5–10× PTM kernel target would provide headroom for transfers and
output costs, but is an engineering target, not an established result. Reference
PTM includes graph matching and best-fit template alignment, so it requires a
substantial implementation rather than a backend flag. See the
[reference source](https://github.com/pmla/polyhedral-template-matching) and
[algorithm description](https://docs.lammps.org/compute_ptm_atom.html).

Recommended next acceptance measurement: sustained completed frames/hour on a
multi-frame workload, including reads, PTM, descriptors, compression, writes
and lineage. Record actual physical cores and GPU host-CPU usage. Compare both
one normal production CPU worker and six concurrent such workers; the earlier
two-physical-core measurement alone does not establish a resource-efficient
switch from the available CPU pool. Preserve existing birth definitions and
audit classifications and event histories before adoption. This feasibility
calculation did not change production jobs or launch another GPU allocation.
