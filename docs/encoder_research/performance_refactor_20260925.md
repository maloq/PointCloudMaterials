# Encoder runtime audit and proposed refactor — 25 September 2026

**Policy update later on 25 September:** AP-specific training and selection were withdrawn by the user. The active trainer now uses predictive hazard likelihood; AP is diagnostic only. The AP replay measurements below describe the stopped historical pipeline. There will be no replay-cadence, sampled-AP or stochastic-AP tuning experiments. The remaining runtime priorities are tensor layout/fusion, compiled execution, shared spatial computation and evaluation reuse.

This audit concerns the then-active supervised AP3/AP6 capacity pipeline, with comparisons to the older dynamic-input and spatial-context implementations. Measurements are from an isolated RTX PRO 6000 Blackwell allocation. They are not H100/H200 measurements or completed quality comparisons. Production jobs and W&B implementation were not changed for this audit.

## What the current pipeline actually does

| Stage | Current implementation | Consequence |
| --- | --- | --- |
| Trajectories, relaxation, targets | Offline paired observed/relaxed caches, with original MD event labels and separate source roles | Training does not read raw trajectories or run relaxation every update. More simulation is unnecessary for this runtime audit. |
| Geometry preparation | `Corpus` loads graph arrays; `GraphBank` uploads both domains and caches parameter-free radial features and spherical harmonics | Most repeated geometry construction has already been eliminated. Geometry perturbations still require rebuilding their graphs/features. |
| Batch assembly | CPU NumPy builds flattened node/edge index vectors; small index tensors move to GPU; resident features are gathered | There is CPU work, but it is not the dominant measured cost for the 2M model. |
| Encoder | Two MACE interactions/products; correlation 2; degrees 0, 1, 2; cuEquivariance; FP32; center-conditioned atoms; smooth local taper | Large edge activations and tensor transformations matter much more to memory than the parameter file size. |
| Export | Final normalized scalar channels are pooled at the center and over its neighborhood; trainable projected residual exports 128 dimensions | The prediction head is small. Most computation happens before pooling. |
| Ordinary training | Importance-corrected hazard NLL, effective batch 256; microbatch accumulation | Frozen capacity jobs use microbatch 64. New workspace defaults are 256, as requested. |
| Ranking training | Every 32 updates: full fitting-population forward without gradients, AP-head backward, then full encoder forward/backward replay | This is the largest measured training cost. It gives exact chain-rule gradients for that full-population loss with bounded activation memory. |
| Selection/checkpoints | Selection every 256 updates; last checkpoint every 128; save best AP3 and AP6 states | Necessary scientific bookkeeping, but GPU/CPU/I/O stages should be separately timed. |
| Final evaluation | Export best3 and best6, fit frozen linear/nonlinear readouts, calibrate, bootstrap sources, measure trajectory spectra and input-noise response | Some reusable controls and identical selected states are recomputed. CPU statistical work need not hold an expensive GPU allocation. |

Source anchors: [training](../../src/research/supervised_onset/train.py), [capacity model](../../src/research/supervised_onset/model.py), [resident graph bank](../../src/research/structural_state/model.py), [evaluation](../../src/research/supervised_onset/evaluate.py), [backend](../../src/models/encoders/mace_backend.py).

The active input is one local coordinate snapshot: no velocity, temporal history, temperature or explicit time covariates. Relaxed arms consume the already-relaxed current structure. The capacity arms do not use a teacher. These input restrictions must remain explicit when refactoring.

## Measurements

### Effective batch 256, different microbatches

Real cached Al patches, 2M relaxed-input encoder, warmed shapes, alternating measurement order. Three blocks of eight ordinary updates per microbatch and two full ranking replays per microbatch.

| Measurement | Microbatch 64 | Microbatch 256 |
| --- | ---: | ---: |
| Ordinary update, median | 0.2194 s | 0.2290 s |
| Full ranking replay, median | 12.603 s | 13.265 s |
| Estimated steady update time: ordinary + ranking / 32 | 0.6132 s | 0.6435 s |
| Peak allocated memory during ordinary updates | 14.07 GiB | 38.47 GiB |

On this GPU and model, microbatch 256 gives **4.7% less throughput**, with about **2.7× peak allocated memory**, at the same effective batch. This does not establish the optimum on H100 or H200. Startup, selection, checkpoint writing and network logging are excluded. The default was not changed back by this audit.

The full-population term accounts for about **64%** of this steady training estimate. Its forward pass visits 10,825 rows, followed by another 10,825-row encoder replay, every 32 updates. The ordinary updates in that interval process 8,192 examples in total.

Raw timing receipt: [timing.json](../../output/encoder_supervised/runtime-audit-20260925/technical/batch-comparison.json).

### Separate ranking stages and GPU operators

A second diagnostic loaded an existing 2M observed-arm checkpoint and ran it on relaxed graphs solely for timings/gradients. No optimizer steps or scientific fits were performed. The different checkpoint/input pairing is immaterial to interpreting these as a runtime sample, but these are not validation metrics.

| Ranking stage, warmed repetition | Microbatch 64 | Microbatch 256 |
| --- | ---: | ---: |
| Full no-grad encoding | 3.651 s | 3.797 s |
| AP head objective and backward | 0.0024 s | 0.0029 s |
| Full encoder replay and backward | 9.046 s | 9.427 s |

The first head invocation cost 0.041 s; subsequent invocations were 0.002–0.003 s. The expensive part is repeatedly encoding the population, not computing AP ranks. Our Smooth-AP implementation compares positives against the population: 79 positives at 3 ps and 188 at 6 ps against 10,825 rows, approximately 2.89 million comparisons combined. It does **not** materialize a full population-squared comparison for each horizon.

One profiled ordinary forward/backward pass at each microbatch found:

| GPU trace measurement | Microbatch 64 | Microbatch 256 |
| --- | ---: | ---: |
| CUDA kernel count, effective batch 256 | 2,640 | 660 |
| Sum of kernel durations | 210.8 ms | 221.4 ms |
| Segmented-transpose kernel durations | 80.6 ms | 81.8 ms |
| Segmented-transpose share of kernel time | 38.2% | 36.9% |
| Median isolated batch packing/gather wall time | 0.264 ms | 0.638 ms |

The denominator includes only trace events with `cat=kernel`. Summing PyTorch key averages would double-count CPU associations and GPU annotations. Kernel-duration sums are not end-to-end wall time. Four times fewer kernels at microbatch 256 did not produce higher throughput: kernel launch count alone is not an adequate optimization target.

GraphBank preparation for both domains took 11.3 s, excluding earlier imports/model/checkpoint loading. Actual graph node counts are 80 throughout observed data and 70–80 for relaxed data. Directed edge counts vary: observed median 1,276, relaxed median 1,262. This variation matters for compilation even though node counts are nearly fixed.

The profiler process emitted one recoverable allocator-OOM warning when changing shapes; it completed both traces. Do not use this instrumented process as a peak-memory benchmark. The separate timing receipt above supplies the memory comparison. GPU utilization was not used to infer bottlenecks.

Raw stages, operator associations, hashes and traces: [runtime audit](../../output/encoder_supervised/runtime-audit-20260925/technical/profile.json), [64 trace](../../output/encoder_supervised/runtime-audit-20260925/technical/ordinary-micro64.json), [256 trace](../../output/encoder_supervised/runtime-audit-20260925/technical/ordinary-micro256.json). Trace-only aggregation was corrected after collection to avoid double counting; timings were not rerun or altered.

Items 1 and 3 are now implemented: [runtime results](runtime_refactor_20260925.md).
Item 4 has a [concrete implementation design](shared_spatial_implementation.md).

## Refactors in priority order

### 1. Give the MACE core a consistent cuEquivariance layout and fused message aggregation

The repository already uses cuEquivariance, but requests `layout='mul_ir'`, `group='O3_e3nn'`, `optimize_all=True`, **`conv_fusion=False`**. The installed MACE wrapper exposes a fused indexed tensor-product path. The existing custom scalar/vector/tensor slices assume the current layout.

Refactor the backbone into a tensor-only core with explicit irrep/layout ownership. Test indexed convolution fusion first, then carry a native layout through interactions, products, normalization and pooling. Convert at actual API boundaries, rather than repeatedly between operations. Inspect the trace again to establish which segmented transposes remain intrinsic to the contraction implementation. NVIDIA documents both layout conversion and indexed gather/scatter tensor products in its [MACE operations guide](https://docs.nvidia.com/cuda/cuequivariance/tutorials/pytorch/MACE.html).

**Benefit:** directly targets the observed memory-traffic hotspot; may also reduce edge-sized intermediates. **Trade-off:** more specialized backend code; layout/representation convention mistakes can silently alter the model. Removing all measured transpose time would be an unrealistic upper bound, not a forecast. The upstream library's advertised speedups compare different baselines; ours is already accelerated.

Acceptance: fixed inputs/weights, exported states, hazard logits, input-noise response and every trainable gradient; short optimizer trajectories; rotation/permutation checks; ordinary training and export performance. Retain O3/e3nn conventions. Do not copy Adam diagonal moments through a nontrivial change of parameter basis and call it an exact continuation.

### 2. Remove the AP-specific objective (implemented)

The new `supervised_onset_information_v4` protocol contains no AP loss, ranking replay, AP-selected checkpoint or AP-tuned readout/ensemble. Encoder and probe training use the existing importance-corrected first-event likelihood; selection uses natural source-weighted held-out likelihood. AP remains an evaluation metric. Old AP-trained checkpoints are historical evidence and cannot be resumed as new NLL-only fits.

The earlier recommendation to tune replay cadence or introduce stochastic AP is withdrawn. Removing replay eliminates its measured cost rather than making a different ranking optimizer. The arithmetic steady-training estimate drops from approximately 0.64 to 0.23 seconds/update at microbatch 256; this is not an end-to-end speed measurement of a new scientific fit. There is no guarantee of higher AP, nor is higher AP the design objective.

The research question is which observed information the state preserves for future crystallization. Use fixed-capacity likelihood-trained probes, input baselines, calibration, structure retention and state-sufficiency diagnostics alongside AP. Do not simply replace an AP leaderboard with claims that NLL measures all mutual information.

### 3. Separate immutable graph storage, batch plans and pure GPU execution

Replace repeated NumPy construction in `GraphBank.batch` with explicit reusable index plans. Fixed-order population passes can reuse fully prepared chunk plans across updates. Sampled batches can gather from GPU-resident index tables; node/edge lengths need not be recomputed over the entire bank for every call.

Then compile the encoder and backward outside Python packing. Start with supported dynamic dimensions or a small set of edge-count buckets; use fixed-buffer CUDA graph capture only where it fits. Mask padding out of **messages, normalization and pooling**, without changing the actual sampled population or weights. Maintain a tail-batch path.

**Benefit:** a stable foundation shared by training, selection and export, with fewer launches and allocations. **Trade-off:** padding consumes compute/VRAM; too many shapes cause expensive compilation; graph capture consumes workspace and needs stable execution. The existing shared-pretraining compilation helper contains lessons about lazy cuEq initialization, graph breaks and autocast backward. Reuse those lessons rather than copying its dynamic policy blindly. See [PyTorch compilation](https://docs.pytorch.org/docs/stable/generated/torch.compile), [dynamic shapes](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html) and [CUDA graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).

Packing alone is well below 1% of the measured 2M ordinary update. This is an enabling refactor, not evidence that a DataLoader rewrite will double current throughput. Compilation may matter more for the small model; it needs its own measurement.

### 4. Share atom computation across spatial contexts

For the next context-aware encoder, use one atom graph for a frame or overlapping-patch union, followed by many focal readouts and a sparse coarse context graph. Reuse atom computations where physical neighborhoods overlap. Keep broader regions at lower resolution rather than running 25 independent large encoders for every focal center.

This requires an architectural change: current atom features depend on the focal center indicator and patch-specific taper. They **cannot** be deduplicated across crops exactly. A reusable backbone would be center-independent; center specificity would enter its readout/context stage. A bounded union plus the required message-passing halo is often more appropriate than encoding an entire 70k-atom cell for a few centers. Measure the overlap/break-even point before choosing whole-cell execution.

**Benefit:** potentially the largest saving when scaling to many focal centers, histories and context slots. **Trade-off:** changes where center conditioning enters, receptive fields and available information; sparse coarse context can lose detail. Compare against independent crops at equal observation support. [LSR-MP](https://arxiv.org/abs/2304.13542) provides a relevant fine/coarse interaction design, not evidence of Al-onset AP gains. See [the spatial-context review](spatial_context_20260925.md).

### 5. Make evaluation a resumable dependency graph

Export each unique encoder state once, keyed by encoder tensor content, normalization, data identity and input protocol. The new workflow exports a single NLL-selected state. Historical best3/best6 states remain distinct historical artifacts. Save frozen features once for linear/MLP probes and downstream analysis.

Compute identical descriptor controls once per cohort/protocol/readout recipe, instead of repeating them in every width job. Reuse immutable results with explicit provenance. Run CPU calibration, source bootstrap, tables and plotting after GPU exports, on a separate CPU stage. Chunk/job boundaries should support resume without re-exporting completed data.

**Benefit:** avoids duplicated work and releases GPUs sooner. **Trade-off:** more artifact dependencies and strict cache identities. Do not reuse an export across changed normalization, changed noisy coordinates, changed checkpoint or changed inputs. Noise diagnostics still require re-encoding perturbed inputs. Keep source-bootstrap uncertainty; compute it more efficiently rather than dropping it to make evaluation look fast. These stage costs were inspected in code, not isolated in today's timings.

## Parameter and hardware choices

| Choice | Recommendation | Trade-off or qualification |
| --- | --- | --- |
| Effective batch | Retain 256 initially | Keeps the ordinary objective/sampling comparison fixed. Changing effective batch changes optimization, unlike merely partitioning it. |
| Microbatch | Measure 64/128/256 on each GPU; use explicit deviations from the requested default | 256 is already slower than 64 in the RTX sample. Larger VRAM does not imply larger microbatches are fastest. |
| Width | Keep the current capacity study; compare quality per GPU-hour | Parameters are not the dominant VRAM consumer: 2M FP32 weights occupy only about 8 MB, versus tens of GiB of activations/workspace. Tensor multiplicity, degree, edges and fusion determine cost. |
| Channel alignment | Test hardware-friendly nearby widths after the backend refactor | 112/168/240 were selected for approximate parameter budgets. Different widths may suit kernels better but change capacity; do not assume a multiple of 64 wins. |
| FP32 / TF32 / BF16 | Test TF32 and selective BF16 separately; keep AP/logit transforms and sensitive reductions in adequate precision | Faster matrix operations and lower activation memory may help, but cuEq kernels need separate coverage. An earlier BF16 pipeline showed substantial repeated-gradient variation despite similar losses; no blanket BF16 enablement based only on loss closeness. |
| MACE tensor degree / final layer | Compare smaller tensor multiplicities or a scalar final layer as architecture ablations | Final vectors/tensors currently affect scalar outputs through all-channel feature normalization. They are not simply unused tensors that can be deleted without changing the model. |
| Smaller cutoffs / fewer neighbors | Low priority before fusion/context redesign | Reduces cost by discarding geometric information and potentially precursor context; hard neighbor caps can worsen boundary sensitivity. |
| Extra VRAM | Cache immutable geometry, reusable batch plans and bounded compiled workspaces | Keeping trainable activations/features across optimizer steps makes them stale. Full-population activations will not fit merely because an H200 has more memory. |
| More GPUs | Parallel independent fits/ablations first | Per-model DDP adds communication; global AP is non-additive and requires correct global ranking/gradient handling. It may reduce latency while increasing GPU-hours. |

The prior dynamic shared-pretraining pipeline did benefit from CPU process prefetch: [the September 19 report](../shared_pretraining_mace_dual_optimization_20260919.md) records a 1.92× short-window improvement, with an explicitly imperfect timing comparison and a gradient reproducibility caveat. That pipeline constructed fresh inputs. The resident-geometry trainer measured here has a different bottleneck; the old speedup is not transferable.

A bolder alternative is a scalar/vector architecture such as [PaiNN](https://arxiv.org/abs/2102.03150), or efficient edge-frame convolutions inspired by [eSCN](https://arxiv.org/abs/2302.03655). These change model expressivity; eSCN's benefit at high angular degree does not establish a benefit for our degree-2 MACE. Test only after establishing the fused MACE reference, using AP and robustness at matched compute.

## Concrete implementation order and success criteria

1. Extract a shared, explicit tensor-only MACE runtime and graph-batch contract. Keep supervised AP, self-supervised and BCR objectives in separate scientific trainers. Remove the current dependency chain in which a capacity model reaches its spatial backbone through BCR and causal-model constructors. This reduces maintenance/initialization duplication; it is not itself a claimed steady-state speedup.
2. Add and validate fused convolution/layout execution, then compile/bucket the expensive pure-GPU path. Report prefill/startup, NLL, AP replay, selection, checkpoint and export separately.
3. Train the predictive-likelihood baseline without AP replay. Compare information/readout controls, predictive likelihood, calibration and AP3/AP6 at matched compute; AP must not control optimization or checkpoint selection.
4. Refactor reusable exports/controls and CPU evaluation dependencies.
5. Build shared fine/coarse spatial context as its own scientific experiment.

Preserve identical source splits, at-risk population, observed/relaxed definitions, causal labels and prediction-input ledgers. Compare present-information probes, 0.75 ps movement/stability, movement/dataset spectra and normalized input-noise response alongside AP3/AP6 when the architecture or precision changes. For numerical runtime changes, establish output/gradient parity first. For optimization changes, compare quality versus both examples processed and wall time. Use validation for choices; do not repeatedly select on the final test set.

Hardware profiling remains an explicit standalone diagnostic, outside the scientific training queue. Normal training should emit lightweight stage timings rather than repeatedly running benchmark suites. W&B ownership is outside this audit.

## Repository implementation links

* [Train / full-population replay](../../src/research/supervised_onset/train.py)
* [Capacity model](../../src/research/supervised_onset/model.py)
* [GraphBank and pooling](../../src/research/structural_state/model.py)
* [Evaluation](../../src/research/supervised_onset/evaluate.py)
* [cuEquivariance configuration](../../src/models/encoders/mace_backend.py)
* [Existing shared-pretraining compilation](../../src/training_methods/shared_pretraining/compilation.py)
* [Diagnostic script](../../output/encoder_supervised/runtime-audit-20260925/technical/profile_pipeline.py)
