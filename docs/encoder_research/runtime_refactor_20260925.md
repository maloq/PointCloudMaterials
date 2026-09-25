# Implemented spatial MACE runtime refactor

Items 1 and 3 of the [runtime proposal](performance_refactor_20260925.md) are now
implemented for the active supervised predictive-information workflow. AP remains
an evaluation metric: training, checkpoint selection and promotion use NLL.

**Performance limitation:** the fresh-random-batch check below found recurring
first-use costs inside the custom-operation path. The 4.36× result is warmed
repeated-batch throughput; a full training-run speedup is not established.
The later padding fix addresses these first-use costs; see the follow-up
integration section. A complete scientific-campaign speedup is still unmeasured.

## Changes

* [SpatialMACE](../../src/models/encoders/spatial_mace.py) builds the spatial
  backbone directly, without constructing BCR or causal/velocity models to
  extract their layers. The optimized backend keeps `ir_mul` layout throughout
  message passing and products, uses `O3_e3nn` conventions, and enables fused
  indexed convolution. All final tensor channels still contribute to feature
  normalization before scalar pooling. The invariant 128-dimensional export and
  parameter budgets are retained.
* [GraphBank](../../src/models/encoders/graph_bank.py) separates immutable GPU
  geometry from batch-index plans and their materialization. Lengths are
  calculated once. Declared repeated validation/export passes retain reusable
  GPU index plans in a bounded cache, default 256 MiB per domain. Random sampled
  batches do not populate/evict that cache. Passes larger than the cache are
  gathered directly rather than prebuilding an immediately evicted pass.
  Plans are bank-owned, preserve order/repeated examples, and reject use with
  another geometry bank. No learned feature or activation is cached across
  optimizer updates.
* The tensor-only encoder is compiled with `torch.compile(fullgraph=True,
  dynamic=True)`. NumPy planning and transfers stay outside compilation. Lazy
  cuEquivariance bases are warmed before tracing; failures and excessive
  recompilation propagate rather than silently falling back. Checkpoint
  parameter names remain ordinary module names. The follow-up padding fix below
  fixes the atom axis; edge counts remain dynamic. CUDA graph capture is not
  enabled by this change.
* [Active recipes](../../configs/supervised_onset/information_20260925/campaign.json)
  enable native layout, fusion and compilation, retaining effective batch and
  microbatch 256. They point to fresh `information-runtime-20260925` outputs.
  The source/input identities include both new runtime modules and record
  layout, fusion and compilation settings. Historical source snapshots remain
  unchanged.

The new factory consumes fewer random initializations because unused modules
are no longer constructed. The same numeric seed therefore does not promise
the old initialization. Numerical comparisons below explicitly copy matching
parameters. These recipes start fresh; they do not migrate an old optimizer
through a changed parameter representation.

## Measured performance

RTX PRO 6000 Blackwell, 2M model, actual cached relaxed Al graphs, effective
batch/microbatch 256. All three variants start from the same observed-arm
checkpoint parameters for this runtime diagnostic; these are not scientific
observed-versus-relaxed prediction results. Each performs importance-corrected
hazard NLL, backward, gradient clipping and AdamW updates. No AP loss is used.

| Runtime | Median update | Peak allocated VRAM | Speed versus reference |
| --- | ---: | ---: | ---: |
| `mul_ir`, unfused, eager | 0.2291 s | 38.51 GiB | 1.00× |
| `ir_mul`, fused, eager | 0.0927 s | 16.01 GiB | 2.47× |
| `ir_mul`, fused, compiled | 0.0526 s | 15.70 GiB | 4.36× |

Medians use three blocks of eight updates. Four initial warmup updates precede
the blocks; the first fused block still incurred lazy work on previously unseen
shapes (1.47 s/update), followed by stable 0.0927/0.0925 s blocks. Its raw value
is retained, not silently discarded. Initial warmup/compilation costs were
15.4/11.4/25.9 seconds respectively; they exclude model construction and bank
loading. One reference allocator retry warning was recoverable. The compiled
timing recorded one Dynamo graph and no graph breaks in the measured training
mode.

This is a short sequential timing comparison, not a complete training campaign
or a cross-GPU benchmark. Validation, checkpoints and W&B are excluded. AP/NLL
quality and H100/H200 speed are not established by these timings. Full-precision
runtime parity is a separate check below. There is no claim that all scientific
jobs or all older encoders become 4.36× faster.

Receipts: [raw timing](../../output/encoder_supervised/runtime-refactor-20260925/technical/timing.json),
[timing script](../../output/encoder_supervised/runtime-refactor-20260925/technical/timing_probe.py),
[parity receipt](../../output/encoder_supervised/runtime-refactor-20260925/technical/parity_probe.json).

## Batch scaling

Measured on the same RTX PRO 6000 and 2M relaxed-graph model. `batch_size` is
examples per optimizer update; `microbatch` is graphs processed together. The
trainer accumulates importance-weighted NLL gradients divided by the full batch
size, then clips and updates once. Changing the microbatch preserves that
objective up to floating-point accumulation; changing the effective batch can
change optimization behavior.

| Effective batch | Microbatch | Median update (ms) | Examples/s | Peak allocated GiB |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 64 | 54.75 | 4676 | 8.32 |
| 256 | 128 | 53.10 | 4821 | 10.78 |
| 256 | 256 | 52.39 | 4886 | 15.69 |
| 1024 | 256 | 207.57 | 4933 | 15.70 |
| 1024 | 512 | 204.40 | 5010 | 25.53 |
| 1024 | 1024 | 214.99 | 4763 | 45.17 |

Each setting first visits all 12 sampled batches, then measures three blocks
of eight repeated updates. These are warm repeated-shape measurements, not a
prediction-quality experiment. At effective batch 256, microbatch 256 improves
throughput by about 4.5% over 64 and 1.4% over 128. At effective 1024,
microbatch 512 gives about 1.6% more throughput than 256. Microbatch 1024 varies
from 208 to 244 ms/update, and does not establish a benefit; the earlier
ascending/descending sweep measured 205 ms after warmup. Both receipts are
retained. More VRAM does not imply proportionally higher throughput. Keep the
production batch/microbatch at 256.

New shape visits incurred substantial costs, even with one captured Dynamo
graph and zero graph breaks. Those costs recur across new settings, so a
four-update warmup was insufficient. The original 4.36× comparison must be
read as warmed repeated-batch throughput, not an established full-run gain.
The separate profiled fresh-random-batch check confirms a limitation: after
initial compilation, nine of eleven fresh batch visits took 1.63–1.77 seconds;
the first-visit median was 1.643 seconds. Repeating all twelve batches gave a
median of 0.05385 seconds. Only one Dynamo graph was captured, with zero graph
breaks. Python profiling attributes substantial time to cuEquivariance custom
operations (`uniform_1d` and the custom-operation backend); it does not yet
isolate the precise internal cause. Initial compilation under profiling took
50.2 seconds and is excluded from those medians. This is not ordinary one-time
startup alone. Resolving new-shape overhead is a higher priority than raising
batch size. No full scientific campaign was launched by this refactor.

See [fresh-batch timings](../../output/encoder_supervised/runtime-refactor-20260925/technical/fresh-batches.json)
and the [profiling script](../../output/encoder_supervised/runtime-refactor-20260925/technical/fresh_batches_probe.py).

Receipts: [complete warmed sweep](../../output/encoder_supervised/runtime-refactor-20260925/technical/batch-scaling-steady.json),
[summary and source hashes](../../output/encoder_supervised/runtime-refactor-20260925/technical/batch-scaling-summary.json),
[initial sweep including first-visit costs](../../output/encoder_supervised/runtime-refactor-20260925/technical/batch-scaling.json).

## Correctness and use

The independent width-32 CUDA check copied every trainable parameter from the
unfused reference. Maximum exported-state difference was 8.20e-8; the largest
per-parameter relative gradient difference was 2.94e-7. Compiled output and
gradient checks passed for changed atom/edge counts, repeated IDs, several batch
lengths and no-grad evaluation; that check captured two graphs without graph
breaks. The permanent runtime test also checks strict checkpoint reload.

Tests cover repeated/unsorted graph IDs, no cross-graph edges, cached-versus-
independent results, cache bounds, geometry-bank ownership and updated learned
weights. Existing structural/context tests verify that sharing the GraphBank
implementation preserves their behavior. The production preflight exercises
the enabled runtime on both observed and relaxed graphs with batch 256; its
enriched smoke-subset scores are not scientific results.

Validation complete: 36 supervised/runtime/tracking tests and 20 shared
regression tests passed. Four historical AP-specific tests were excluded from
the shared regression subset. All eight observed/relaxed preflights across
small, 500k, 1M and 2M sizes passed and match the final implementation identities.
See the [validation receipt](../../output/encoder_supervised/runtime-refactor-20260925/technical/validation.json).

Commands remain the existing workflow:

```bash
python -m src.research.supervised_onset.campaign check \
  --config configs/supervised_onset/information_20260925/campaign.json
python -m src.research.supervised_onset.campaign submit \
  --config configs/supervised_onset/information_20260925/campaign.json
```

Use `pointnet-torch214` and the existing documented CUDA environment. Testing
and timing do not create W&B runs. This implementation task does not submit a
new scientific campaign. Each worker owns one width; multi-width preflight
resets Dynamo between studies so unrelated width specializations do not exhaust
a single worker's recompile allowance.

For item 4, see [the shared spatial implementation design](shared_spatial_implementation.md).
It changes where center conditioning enters and therefore needs a separate
scientific comparison, even though it can reuse this runtime.

The follow-up [width 128 versus 130 comparison](width128_vs130_20260925.md)
measures the width effect separately, including fresh-batch overhead.

A later [pipeline audit](pipeline_efficiency_audit_20260925.md) isolates the
first-use problem with atom-count padding, checks outputs/gradients, and records
complete-loop timings showing that the overhead diminishes in longer runs.
That audit initially tested padding as a prototype; it is now integrated as
described below.

## Follow-up integration

The three priority fixes from the [pipeline audit](pipeline_efficiency_audit_20260925.md)
are implemented: fixed-capacity zero-weight atom padding, variant-specific
context field loading, and reordered harmonic hierarchy contractions. The
historical warmed/fresh timings above describe the earlier runtime; see the
audit follow-up for current verification and limits. Existing running source
snapshots and output identities are preserved.
