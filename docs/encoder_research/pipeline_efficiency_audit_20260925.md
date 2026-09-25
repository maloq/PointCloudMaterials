# Pipeline efficiency audit — 25 September 2026

The largest verified opportunity is stabilizing MACE's atom-array size. A
second verified opportunity is rearranging the hierarchical context contractions.
Neither requires changing the observations, model capacity or learning objective.

This audit covers the current native supervised MACE trainer and the new frozen
encoder/equivariant context workflow. Historical GeoFormer, BCR and history
trainers are not all reprofiled here. The previous
[runtime audit](runtime_refactor_20260925.md) and
[128/130 comparison](width128_vs130_20260925.md) remain separate evidence.

The initial standalone diagnostics ran on a separate RTX PRO 6000 Blackwell GPU
on node58, Slurm jobs **1008633** and **1008636**. They did not change scientific
jobs or production code. The three priority fixes have since been integrated
into the workspace; see the implementation follow-up below. Running source
snapshots, scientific configurations and W&B integration remain unchanged.

## Findings and priorities

| Priority | Finding | Evidence | Proposed change |
| --- | --- | --- | --- |
| 1 | Variable atom counts repeatedly trigger expensive cuEq kernel setup/selection | Fresh-shape update 1.183 s; padded prototype 0.0265 s; parity checked | Pad atom arrays to a declared capacity with disconnected zero-weight nodes; keep real edges unchanged |
| 2 | Hierarchical context materializes large pairwise tensor messages | Reordered block 1.89× faster at batch 512, 1.10× at 256; about 69% less additional allocated memory | Combine scale weights before contraction and avoid the pair × channel × irrep intermediate |
| 3 | Scalar context controls load and gather unused tensor fields | Code and exact array-size accounting: 108.55 MiB gathered per batch 512 versus 6.40 MiB needed | Give each variant an explicit required-field schema |
| 4 | Feature extraction uses an uncompiled method and per-patch GPU work | Source inspection, not a measured extraction speedup | Compile a dedicated field-export forward; vectorize degree-4/6 bond summaries |
| 5 | Some capacity widths are poorly aligned | Width 130 measured 50.4% slower than 128; 112/168/240 also fail divisibility by 32 | Benchmark aligned widths in new capacity studies; preserve existing recipes |
| 6 | Fixed context geometry is recomputed, including harmonics in every block | Source inspection | Compute once per batch; optionally cache immutable geometry per observation |
| 7 | CPU–GPU synchronization and repeated index transfers remain | Warm trace has eight stream synchronizations; source scalar conversions and CPU batch plans | Consolidate transfers and scalar checks; retain useful nonfinite failures |
| 8 | Evaluation and setup repeat reusable work | Source inspection | Resume by artifact identity; share identical controls and geometry; separate GPU exports from CPU analysis |

Measured speedups above concern particular stages. They must not be multiplied
together or presented as a full-campaign speedup.

## 1. Atom counts, not just batch size, cause costly specialization

The existing `torch.compile(fullgraph=True, dynamic=True)` captures one graph
for the diagnostic. Nevertheless, cuEquivariance does additional work inside
custom operations when the number of atoms changes. A constant batch of 256
patches is not a constant atom-array size: relaxed patches contain fewer than
80 atoms when cropping removes some neighbors.

The isolated test used width 128, export 128, batch/microbatch 256, FP32, fused
cuEq and real relaxed Al graphs. It deliberately sampled 20 batches with distinct
total atom counts. The prototype pads only `attrs`, `center`, `weight`, and
`group` to 256 × 80 entries. Dummy nodes have no edges and zero pooling/update
weight. It does not pad the edges, alter neighbors, resample examples or change
the normalization denominator.

| Input handling | Median fresh update | Median repeated update |
| --- | ---: | ---: |
| Existing variable atom count | 1.1834 s | 26.56 ms |
| Fixed atom capacity, dummy nodes | 26.51 ms | 26.35 ms |

Fresh medians exclude initial compilation and the instrumented batch; 18
unprofiled fresh visits remain. Warm medians use three blocks of eight updates.
The first-visit improvement is about **44.6×**, not a steady-state or full-run
44.6× gain. Edge counts still vary in both paths.

The instrumented fresh unpadded update launched **2,774 kernels**, taking
879 ms of summed kernel time, with **590 event synchronizations** and **303
stream synchronizations**. Its warmed counterpart launched **335 kernels**,
with 24.62 ms of summed kernel time and eight stream synchronizations. The
padded fresh trace launched 339 kernels and took 24.46 ms of kernel time.
Many extra first-visit kernels try different segmented-polynomial algorithms
and sorting paths. This supports a shape-dependent kernel-selection/autotuning
explanation, rather than repeated Dynamo graph capture alone. Kernel-duration
sums are not wall times; instrumented timings are excluded from the table.

The small CUDA output/gradient check found maximum output difference 4.66e-8
and maximum per-parameter relative gradient L2 difference 4.67e-7. Before
production integration, extend compiled parity checks to incomplete batches,
both domains, noisy coordinates, rotations/permutations and checkpoint resume.
Keep dummy contributions zero at every message/pooling path. For this capped
dataset padding is small; larger uncapped contexts need bounded size buckets.

The installed cuEq API includes batch and indexed-buffer extents in its problem
description. Upstream also documents JIT compilation and a persistent NVRTC
cache setting. A persistent cache may help compilation across processes, but
does not by itself establish that per-shape kernel selection disappears.
Do not turn on undocumented algorithm overrides or parallel compilation to
mask the issue. [NVIDIA release notes](https://github.com/NVIDIA/cuEquivariance/blob/main/CHANGELOG.md)
describe why parallel compilation was disabled by default in multi-GPU setups.

### What complete runs tell us

The already-completed width-128 base encoders used their recorded batch 512:

| Domain | Updates | Recorded training-loop time | Late ordinary update, block median |
| --- | ---: | ---: | ---: |
| Observed | 4,096 | 249.2 s | 51.32 ms |
| Relaxed | 4,096 | 477.5 s | 50.94 ms |

These are operational logs, not a controlled end-to-end benchmark. The timer
starts after initialization and initial selection; it includes later selection,
checkpointing and logging. Late medians exclude 32-update intervals following
the save/selection boundaries. Relaxed training spends much more time on early
shape visits but later reaches roughly the same steady update speed. Thus the
earlier fresh-batch observation must not be extrapolated to every update forever.
Extraction and short fits can remain especially sensitive to novel sizes.

Source: [GraphBank](../../src/models/encoders/graph_bank.py),
[spatial runtime](../../src/models/encoders/spatial_mace.py),
[training](../../src/research/supervised_onset/train.py).

## 2. Remove large intermediates without removing context

`HarmonicHierarchy` creates messages shaped `[batch, receiver, sender, channel,
irrep_component]`, then aggregates at three radii and mixes those aggregates.
For batch 512, 25 context nodes, 16 field channels and degree 6, one such
FP32 message tensor is **253.91 MiB** before gradients and other intermediates.

The scale mixing can occur first. Let `W[r,i,j]` be normalized geometric weights
and `a[i,r]` the learned scale mixture. Form

`W_eff[i,j] = sum_r a[i,r] * W[r,i,j]`.

Then separately contract the neighbor field and the directional injection with
`W_eff`. The latter needs a `[batch, receiver, sender, channel]` intermediate
(19.53 MiB at batch 512), followed by matrix multiplication with the harmonics.
There is no need to materialize the 253.91 MiB message. The scalar context
aggregation uses the same effective weights. This distributes linear sums;
the nonlinear updates, learned scales and geometric information remain intact.

| Batch | Original block forward/backward | Reordered | Speed ratio | Additional allocated memory, original → reordered |
| --- | ---: | ---: | ---: | ---: |
| 256 | 4.34 ms | 3.95 ms | 1.10× | 0.423 → 0.130 GiB |
| 512 | 7.74 ms | 4.10 ms | 1.89× | 0.846 → 0.260 GiB |

This is an isolated eager block benchmark using synthetic inputs with the
actual tensor shapes, not the whole predictor or scientific training. Four
alternating blocks of four forward/backward repetitions follow warmup. Memory
is peak allocation above the pre-step baseline. CPU comparisons checked outputs,
all parameter gradients, scalar/field input gradients and geometry gradients:
FP32 maximum output error 3.58e-7 and relative gradient error 2.41e-7; float64
errors were below 6e-16. Full predictor/GPU parity remains an integration check.

`TensorAttention` has related opportunities: contract neighbor fields and
directional couplings without first constructing the full pairwise field.
That variant is inspected but not benchmarked by this prototype.

Source: [context blocks](../../src/research/equivariant_context/model.py).

## 3. Load only the fields each context variant consumes

`ContextCorpus` loads, normalizes, uploads and gathers `z`, `actual`, `f1`, `f2`,
`f4`, and `f6` for every variant. `symmetric_invariant` consumes only `z` and
`actual`; the nominal stencil is shared separately.

| FP32 field | Gather size at batch 512 | Full 31,609-row population |
| --- | ---: | ---: |
| z and actual positions | 6.40 MiB | 0.386 GiB |
| f1 | 37.50 MiB | 2.261 GiB |
| f2 | 62.50 MiB | 3.768 GiB |
| f4 and f6 | 2.15 MiB | 0.130 GiB |

The scalar baseline therefore gathers about **94% unnecessary input bytes** and
retains **6.16 GiB of unused tensor fields**. These are exact shape/byte counts,
not measured latency savings. `vector_messages` needs only f1;
`tensor_attention` needs f1/f2; the harmonic variant needs all fields.

Implement a variant-to-field contract before loading and normalizing arrays,
not just before the model forward. Preserve identical fitting statistics for
every retained field and validate cache identities. This also reduces host
memory peaks from simultaneous raw, normalized and temporary arrays.

Source: [context corpus](../../src/research/equivariant_context/data.py).

## 4. The extraction path bypasses part of the acceleration

`extract()` constructs a new frozen model and `patch_features()` calls
`encoder.atom_features()` directly. It does not call the compiled encoder
forward used by the supervised trainer. Its cuEq fusion remains enabled, but
there is no compiled field-export wrapper.

Furthermore, f4/f6 use a Python loop over patches: each iteration transfers
vectors, evaluates harmonics and pools. A full chunk of 512 patches has 1,024
such loop iterations across the two degrees. That is a likely explanation for
bursty extraction, but its share of extraction wall time is not yet measured.

Build a tensor-only field-export module, use stable padded node buffers, and
compute harmonics for packed non-center atoms in one operation per degree.
Pool through graph IDs with the existing envelopes and fixed `n_ref`. Transfer
completed outputs in larger chunks. Keep the center exclusion, periodic-image
handling, cutoff definitions, atom identities and relaxed/observed semantics.
Representative atom IDs are already deduplicated within each source/frame;
do not claim that exact deduplication as a new optimization.

## 5. Alignment audit: where it matters and where it does not

The 128-versus-130 diagnostic establishes a real width sensitivity. Current
native/context defaults use 128, and scalar attention uses four heads of 32:
these are sensible aligned shapes. The earlier 500k/1M/2M capacity recipes use
112/168/240 channels. Benchmark 96/128, 160/192 and 224/256 as new alternatives;
these change capacity and must be recorded as new scientific configurations.
No speed benefit for these pairs is established by the 128/130 measurement.

Do not mechanically round every dimension to 32. Irrep component dimensions
3/5/9/13 are mathematically specified; five hazard bins define the target.
The context field width 16 uses ordinary PyTorch operations, so the cuEq
uniform-kernel divisibility guidance is not automatically its optimum.
The small radial input and five-output head are lower priorities until an
operator profile shows material cost. Likewise, increasing batch size beyond
256 has not given a substantial throughput gain in the measured MACE core.

## 6. Cache fixed context geometry, not learned features

Context positions and stencil are fixed for a cached observation, but each
forward recalculates displacements, distances, radial bases and envelopes.
Harmonics are calculated again in each message block. At minimum, compute them
once per forward and pass them to all blocks. Then measure caching only the
geometry needed by each variant. Full pairwise geometry caches trade arithmetic
for quadratic storage and memory traffic; larger context graphs require care.
Never cache learned scale weights, attention or atom activations across updates.

## 7. Synchronization, batch planning and precision

The current loss converts CUDA scalars to Python inside each microbatch.
CPU sampling is followed by repeated index-tensor transfers; `GraphBank.plan`
builds the flattened indices in NumPy. The warm trace contains eight stream
synchronizations and ten async memcpy calls. It also contains 64 scalar-item
calls, but these are **not 64 GPU barriers**: optimizer counters can be CPU
scalars. Time spent waiting for a stream overlaps useful GPU work and must not
be counted as wholly removable overhead.

Accumulate loss diagnostics on device, consolidate row/index transfers and
consider bounded GPU batch planning. Retain one useful fail-fast health check
per update and the finite-gradient check; do not silently remove correctness
checks. Existing fixed-pass plan caching is already implemented. Historical
packing/gather timings were under 1 ms at batch 256, so this is lower priority
than the demonstrated first-use cost.

The warmed profile uses FP32 SIMT GEMM kernels. Selective TF32 or BF16 warrants
a separately validated precision comparison, not unconditional enablement.
Check all gradients, accumulated updates, geometry invariance, noise response
and predictive scores; similar loss alone is insufficient. The current audit
does not establish a mixed-precision speedup or acceptable numerical error.

## 8. Setup and evaluation reuse

`make_banks()` builds both observed and relaxed banks even for a single-domain
arm. Construct the domains actually required by the arm/teacher. The geometry
cache is a benefit; rebuilding raw neighborhoods each update would regress it.

Final evaluation still re-exports encoder features if interrupted after export,
and descriptor controls are repeated across capacity studies. Key reusable
artifacts by checkpoint content, cohort, input/normalization definition and
probe recipe. Resume at completed exports/probes. In paired noise diagnostics,
the unchanged domain's compact bank is rebuilt for each noise fraction; it can
be reused. Perturbed geometry must still be rebuilt and re-encoded.

Calibration, source bootstrap and plotting should be CPU stages after GPU
exports. For AP bootstrap, sorting predictions once and reweighting precomputed
tie groups can preserve the statistic while avoiding repeated sorting. Retain
ties, source weights and zero-event-draw handling, and verify against sklearn.
This is evaluation acceleration, not AP-based optimization or selection.
Do not alter the declared likelihood selectors, source splits or diagnostics
to make the queue finish sooner. W&B ownership remains with the other agent.

## Integration order and receipts

1. Productionize bounded node padding, with compiled parity and short exact-
   resume tests; time fresh shapes and complete loops separately.
2. Make context loading variant-specific, then integrate the verified harmonic
   contraction reorder and test the entire predictor on GPU.
3. Vectorize/compile field export and reuse geometry within context forwards.
4. Add artifact-level evaluation resume and shared controls; then measure
   whether transfer planning, optimizer fusion or precision warrants more work.

Fresh-shape tests are deliberately diagnostic, not a training-sampling change.
Scientific default batch/microbatch remains 256; current 512 recipes retain
their recorded setting. No temperature, age or explicit time inputs were added.

Evidence: [summary and implementation hashes](../../output/encoder_supervised/pipeline-audit-20260925/technical/summary.json),
[shape timings](../../output/encoder_supervised/pipeline-audit-20260925/technical/shape-results.json),
[context block timings](../../output/encoder_supervised/pipeline-audit-20260925/technical/context-benchmark.json),
[contraction parity](../../output/encoder_supervised/pipeline-audit-20260925/technical/context-algebra.json),
[array accounting and completed-run timing logs](../../output/encoder_supervised/pipeline-audit-20260925/technical/static-audit.json).
Standalone scripts, submission scripts and four CPU/CUDA traces are in the
same `technical/` directory. No W&B runs were created for these diagnostics.

## Implementation follow-up

The first three fixes are now integrated in the workspace:

* `GraphBank(node_capacity=...)` pads attributes, center flags, weights and group
  IDs with zeros, adds no edges, and rejects capacity overflow. Supervised
  training, evaluation and noise banks use `80 * microbatch`; context feature
  export uses `80 * chunk`. Partial batches keep the same capacity. Physical
  graph count, center indices and pooling normalization are preserved.
* `ContextCorpus` requires a variant and loads/normalizes/uploads/gathers only
  its consumed fields. Whole-shard checksums still protect cache integrity.
  Each requested NPZ array is read once per shard. Field-order schemas also
  construct the predictor, so loading and model requirements share a definition.
  Checkpoints, completion receipts and preparation plans list the actual fields,
  including the nominal geometry supplied separately to every variant.
* `HarmonicHierarchy` combines normalized spatial-scale weights before the
  linear sums. Parameters, nonlinear updates and checkpoint keys are unchanged;
  receiver × sender × channel × component messages are no longer materialized.

Unit tests cover repeated graph IDs, short batches, overflow, dummy-node/edge
isolation, field loading with all unused arrays absent, train-only normalization,
output/gradient parity, rotation/reflection and permutation properties, export,
checkpoint round trips and resumed updates. Full-predictor hierarchy regression
uses the original contraction formula as an independent reference.

The user-declared numerical acceptance threshold for independently executed
resumed GPU updates is **absolute error 1e-6**. Serialized parameters and optimizer
state are checked separately; GPU reduction order need not be bitwise identical.
The first diagnostic attempts used an unnecessarily strict bitwise comparison;
their logs are retained in the verification directory.

Scientific settings, existing frozen source snapshots and W&B code are preserved.
The default effective batch/microbatch remains 256; recorded 512 ablations remain
512. New source identities require fresh output directories. These changes are
active for new runs through the existing workflows, not injected into running
workers. Feature-export vectorization/compilation is implemented in the follow-up
below. Cached context geometry and artifact-level evaluation resume remain later work.

### Production verification results

Separate node58 Slurm job **1008676** completed all checks. CPU suite: **66
passed**, five CUDA-only cases skipped. GPU suite: **38 passed**. There is overlap
between these suites; these are not 104 distinct tests. A frozen copy of the
checked implementation and source hashes accompany the receipts.

Actual current and relaxed Al graphs passed compiled-versus-eager output and
parameter-gradient comparisons at batch sizes 3 and 256. Maximum embedding
output difference was below 6e-8. The implementation also passed transformed
geometry and typed-field export tests. Checkpoint state restoration and the
next resumed update passed the declared numerical checks.

Width 128, exported width 128, batch/microbatch 256, FP32/TF32 off, RTX PRO 6000
Blackwell. Fresh-shape medians use 18 distinct atom counts per mode, excluding
initial compilation and batch 2 to match the initial audit's sampling:

| Production runtime | Fresh-shape update | Warm update |
| --- | ---: | ---: |
| Ragged atom arrays | 1.1965 s | 26.29 ms |
| Fixed-capacity atom arrays | 26.50 ms | 26.38 ms |

The **45.1×** reduction applies to these deliberately unseen-shape updates.
Warm throughput is essentially unchanged. Both runs captured one Dynamo graph
without graph breaks. This does not establish a 45× scientific-training speedup:
real runs revisit shapes, and initialization, evaluation, exports and logging
contribute separately.

The production hierarchy contraction was compared with the original expression
using identical parameters and alternating measurement order:

| Block batch | Original | Reordered | Speedup | Temporary memory, before → after |
| --- | ---: | ---: | ---: | ---: |
| 256 | 4.579 ms | 4.208 ms | 1.09× | 0.423 → 0.130 GiB |
| 512 | 7.684 ms | 4.308 ms | 1.78× | 0.846 → 0.260 GiB |

Memory here is the peak allocation above the pre-step resident baseline, not
whole-process VRAM. Variant-specific loading removes the unused-field allocations
identified above (about 6.16 GiB for the complete scalar-control cohort); the
verification checks actual NPZ loading and resulting batches, rather than
claiming an unmeasured full-fit timing improvement.

[Machine summary](../../output/encoder_supervised/pipeline-fixes-20260925/technical/summary.json),
[CPU log](../../output/encoder_supervised/pipeline-fixes-20260925/technical/cpu.log),
[GPU tests and timings](../../output/encoder_supervised/pipeline-fixes-20260925/technical/gpu.log),
and [frozen implementation hashes](../../output/encoder_supervised/pipeline-fixes-20260925/technical/implementation.json).
All diagnostics were local-only with no W&B runs; scientific jobs were not
restarted or modified.

## Data feeding follow-up

Read-only inspection after the production verification distinguishes training
from the feature-extraction stage still running with its frozen implementation.

The padded warm training trace has 24.374 ms of **union** GPU-kernel intervals
inside a 28.054 ms profiled step (about 87%); the first-to-last kernel span is
26.621 ms. Six host-to-device copies total 5.346 MiB and 0.129 ms of device copy
time. These are mostly index plans: graph geometry/radial/angular arrays are
already GPU-resident. Device copy time excludes CPU index construction, pageable
staging and synchronization; it is not the total cost of preparing a batch.
Nevertheless, the measured trace does not identify disk loading or PCIe bandwidth
as the principal training bottleneck. Frozen-context fits likewise upload their
selected normalized arrays once, then gather batches on the GPU.

On node59, a 20-sample, approximately one-second-interval observation found GPU 1
averaging 46.9% utilization (range 0–98%), with 5,579 MiB resident, while its cold
feature extractor used approximately one CPU core. GPU 0 was idle, with no
allocated memory: `lane-hot.json` confirmed completion of preparation and all
four predictor variants. This idle lane is a scheduling issue, not evidence that
its trainer is waiting on data. These observations are a short snapshot, not a
whole-run utilization average.

Extraction is serial across frames: read/verify trajectories, build the periodic
neighbor tree and patches, construct/upload a GraphBank, encode, compute degree-4
and degree-6 fields in per-patch loops, synchronously export arrays, then repeat.
There is no bounded background producer overlapping preparation of the next
frame with current GPU work. This source inspection plus utilization sampling
identifies a worthwhile pipeline target, but does not isolate filesystem time
from CPU geometry work or GPU launch overhead.

Next measurement/implementation priorities: phase timers for read/checksum,
geometry, transfer, encoding, bond fields and writes; vectorized bond-field
export; a bounded CPU preparation queue with pinned staging and asynchronous
transfer; and ready-task scheduling across GPUs once dependencies are met.
Keep ancestry validation and source identities. Increasing generic DataLoader
workers cannot address GPU-resident training or per-patch GPU launch overhead.

## Feature extraction implementation and measurement

The batched exporter and bounded frame prefetch are now implemented in
`src/research/equivariant_context/features.py` and used by campaign extraction.
The [operational guide](../equivariant_context.md#batched-feature-extraction)
describes the recorded `extraction` options and per-frame timing fields.

* Degree-4/6 fields use two batched harmonic evaluations and grouped reductions,
  preserving central-atom exclusion, taper radii and normalization.
* A reusable compiled typed forward exports z and l=1,2 fields. The scalar-only
  training forward is no longer mistaken for the typed extraction path.
* CPU workers validate and prepare upcoming frames, including graph edges and
  pinned transfer buffers, while the consumer submits GPU work. The queue is
  bounded and ordered, and worker errors propagate.
* Outputs stay on device until one packed host transfer per frame. Completed
  source shards still define the resume boundary; their integrity is checked.

Thirty-seven CPU tests passed (six CUDA cases skipped in that combined suite),
and all 36 equivariant-context GPU tests passed. These suites overlap. Tests
cover all fields, partial chunks, reuse, rotations, missing unused inputs,
bounded prefetch/ordering/overlap, worker and consumer failures, row/atom identity,
completed-source resume and corrupted-shard rejection. No W&B runs were started.

The real-frame benchmark used RTX PRO 6000 Blackwell on node60, Slurm job
**1008724**, width 128 and chunk 256. It read observed and relaxed geometry for
source 860, frames 64/80/128/176: 1,350 patch encodings per pass. Two timing
repetitions alternated baseline/new order. The baseline is the exact per-patch
export function captured before this change, including the earlier atom-padding
fix. Medians include reads/checks, patch/edge construction, transfers and GPU
extraction; they exclude checkpoint loading, initial compilation and final
source-shard writes. Filesystem caches were warm. The same trained observed
encoder weights were used on both geometries to isolate execution; these are
not new scientific observed-versus-relaxed prediction fits.

| Four-frame geometry input | Serial baseline | Optimized pipeline | Speedup |
| --- | ---: | ---: | ---: |
| Observed | 3.409 s | 0.592 s | 5.76× |
| Relaxed | 3.181 s | 0.596 s | 5.33× |

The first compiled export took 35.46 s. Compilation is therefore a trade-off for
very short exports; campaigns amortize it over many frames. `compile=false` is
an explicit option for short jobs. No full-cohort speedup is claimed from four
frames. CPU preparation still matters: consumer waits total approximately
0.36–0.38 s in these short optimized passes, including pipeline startup/drain.
Faster total extraction does not necessarily produce a higher utilization
percentage when the remaining CPU work dominates.

Numerical interpretation follows the user's updated **1e-5** tolerance. A
single-call comparison initially exceeded it, but the unchanged baseline itself
varied by about 1.8–2.0e-5 in z between GPU calls. Fitted normalization amplifies
small GPU reduction differences. Disabling compilation did not remove that
variation. The final check therefore records both individual differences and
four-call means. Maximum mean differences were **9.48e-6** (observed) and
**9.36e-6** (relaxed) for z, and below 4.3e-8 for every typed field. Individual
optimized-versus-baseline z differences were about 1.6–2.0e-5, comparable to
baseline repeat variation. This is not a claim of bitwise or per-call 1e-5
identity. The compiled path is enabled, with no scientific target/input changes.

[Summary](../../output/encoder_supervised/feature-extraction-20260925/technical/summary.json),
[raw timing/parity records](../../output/encoder_supervised/feature-extraction-20260925/technical/results.json),
[CPU log](../../output/encoder_supervised/feature-extraction-20260925/technical/cpu.log),
[GPU tests](../../output/encoder_supervised/feature-extraction-20260925/technical/gpu.log),
[final benchmark log](../../output/encoder_supervised/feature-extraction-20260925/technical/benchmark-final.log),
and [frozen implementation hashes](../../output/encoder_supervised/feature-extraction-20260925/technical/implementation.json).
The first benchmark attempt had a snapshot-relative data-path error after its
GPU tests passed; the harness was corrected. Intermediate numeric-comparison
logs remain archived. Existing scientific workers were not restarted or changed;
new workspace runs use the optimized path with fresh output identities.
