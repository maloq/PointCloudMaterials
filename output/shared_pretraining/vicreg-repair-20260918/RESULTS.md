# VICReg diagnosis and compiled restart

The preceding MACE run did learn useful structure, despite a nonzero VICReg floor.
It was stopped cleanly at update 1,067 (4.37 epoch equivalents); its physical/TDA
selection score improved to **0.25568**, against a training-group mean of 0.40472
(36.8% lower error). GATr had already stopped at update 640: its best score
0.40485 did not beat that baseline. Checkpoints and immutable source snapshots
were preserved. Unrelated CPU data preparation was not stopped.

## Why the VICReg plots looked stalled

For an exactly constant projector, this implementation reports **24.75**:
pair disagreement and covariance are zero, while the epsilon-stabilized variance
penalty is 0.99. GATr stayed close to that value; its last native-Al batches had
variance penalty about 0.96. MACE's native-Al VICReg decreased from about 24.7 to
18.9–19.3, while its variance penalty fell to about 0.65–0.68. Its curve therefore
was not flat. Raw training curves also mix material/potential/static populations;
per-group curves are necessary for interpreting their fluctuations.

The underlying conditioning issue is a large shared feature component with small
between-observation differences. Per-observation LayerNorm controls scale but
does not standardize those differences. The physical heads can learn nearly
constant predictions; the projector can concentrate its variation in one or two
directions. Earlier unrestricted BF16 further obscured small differences; the
protected-precision enlarged encoders fix that numerical issue, but a matched
pilot showed protected precision alone did not prevent GATr collapse.

The VICReg formula, pairing order and full-statistical-batch gradient cache were
checked; simply changing the logging would not repair the observed collapse.
A zero VICReg total is not a required scientific outcome: neighboring atomic
environments differ, and invariance, variance and covariance terms compete.

## Controlled repair pilots

These exploratory tests use one seed, the enlarged protected-precision encoders,
eight fixed **training-only native-Al batches of 128 pairs**, alternating spatial
and temporal views, repeated for 96 updates. The peak LR is 0.002 with a 16-update
warmup. They are short conditioning diagnostics, not the full broad-data schedule
or held-out test results. Selection uses the unchanged 480 observations from 15
native-Al selection sources. The same batches, initial encoder/head parameters
and optimizer schedule are used for the original-head versus correlation pair.
Several candidates reused this selection set, so the result is not an unbiased
estimate of final generalization improvement.

| Encoder | Variant | Mean VICReg, final 16 updates | Physical/TDA selection error |
| --- | --- | ---: | ---: |
| MACE | Original objective | 16.761 | 0.34907 |
| MACE | Correlation auxiliary | 17.945 | 0.35247 |
| GATR | Original objective | 24.750 | 0.40520 |
| GATR | Correlation auxiliary | 18.701 | 0.40387 |

The selected auxiliary adds 0.1 times a dimension-wise physical correlation loss,
while retaining physical MSE, instantaneous-TDA MSE and the original VICReg
coefficients. It supplies an informative contrast gradient near constant decoder
outputs; reconstruction MSE still determines amplitude. It uses training-batch
statistics only in the loss, never in encoder or decoder inference. See the
[exact definition](tables/METRICS.md). GATr's VICReg and feature spread improve;
its physical-error improvement is modest. MACE continues learning but the new
auxiliary does not improve its pilot point estimate. This is a guarded new
comparison, not evidence that final downstream quality is fixed.

Other retained ablations included projector BatchNorm, physical-only training,
direct-state VICReg, state-variance floors, a centered neighborhood readout,
zero-initialized physical heads, and a larger VICReg weight. Projector BatchNorm
reduced VICReg to roughly 14.5 while physical decoding stayed near the constant
baseline; it was not adopted. These tests demonstrate why a smaller VICReg value
alone is an inadequate selection criterion. Their scripts, weights and all
results remain in `technical/`.

![Loss and selection diagnostics](plots/diagnosis.png)

## Compiler and custom kernel

The new GATr Triton kernel fuses high/low BF16 decomposition and three tensor-core
products with FP32 accumulation, avoiding large cast/residual temporaries and
multiple GEMM launches. Backward uses the same approximation; long weight-gradient
reductions use deterministic split-K partials without atomics. This is first-order
training support, not higher-order force differentiation.

An isolated H100 operation profile attributed about 53% of GPU time to copies,
conversions and additions/subtractions, and about 47% to GEMM. The fused operation
was approximately 2.1–2.7x faster than the eager three-GEMM implementation in the
small shape sweep; that measurement includes Python/launch overhead and is not
an encoder speedup. Relative forward errors against FP32 were around 4.5e-6.

Compilation uses dynamic shapes and in-place module compilation, preserving
checkpoint names. It disables the upstream GATr einsum path cache. Crucially,
`backward_pass_autocast='off'` matches our backward passes outside autocast:
the default compiler assumption produced about 0.3–0.5% relative gradient
mismatch in the checks. This separate compiler issue did **not** cause the old
uncompiled loss plateaus. The corrected path passes gradient comparisons.
Library attention graph breaks remain; compiler counters are retained, and
recompilation-limit exhaustion raises rather than silently switching to eager.

Validation: 29 model, cache, checkpoint, loss and kernel tests passed, plus both
dynamic compiled forward/gradient/rotation/checkpoint tests (31 total on H100).
The 29 non-compiler tests also passed on RTX PRO 6000 Blackwell.
The fused GATr real-observation audit covers 72 observations across nine groups;
its worst rotation error is **0.0876% of within-group FP32 signal**, below the 1%
gate. These are correctness checks, not learned prediction-quality results.

## Matched update timings

Benchmarks use the same enlarged models, objective, seed, full batch 1,024 and
largest-support Ta/static observations. MACE uses microbatch 64 on H100; GATr
uses 256 on RTX PRO 6000 Blackwell. TF32 is disabled. Two warmup updates precede
five synchronized measured updates. The benchmark uses fixed LR 0.0003 and
AdamW default weight decay 0.01 in every mode; the live training uses its
scheduled LR and weight decay 0.0001. Transfer, both gradient-cache passes,
heads/losses, backward and AdamW update are included; CPU data/graph preparation
and compilation are excluded. The warmup column includes compilation and two
updates, not isolated compile time. Each timing runs without another GPU job
on the same device. Raw indices and source hashes are retained. [Timing CSV and frozen definitions](../vicreg-repair-timings-20260918/tables/update_timings.csv).

| Encoder/device | Execution | Median seconds/update | Warmup seconds | Peak allocated GiB |
| --- | --- | ---: | ---: | ---: |
| MACE | eager_fp32 | 4.921 | 14.6 | 25.39 |
| MACE | compiled_fp32 | 3.762 | 80.3 | 22.94 |
| MACE | compiled_fused | 3.600 | 77.2 | 22.50 |
| GATR | eager_fp32 | 6.341 | 28.2 | 24.20 |
| GATR | compiled_fp32 | 4.525 | 274.2 | 17.41 |
| GATR | compiled_reference | 4.817 | 210.0 | 19.86 |
| GATR | compiled_fused | 4.665 | 165.1 | 17.44 |

MACE compiled selective BF16 speedup over eager FP32: **1.367x**.

GATR compiled selective BF16 speedup over eager FP32: **1.359x**.


The custom GATr kernel reduces full-update time by **3.16%** and peak allocated
memory by **12.22%** relative to the compiled three-GEMM control. Its update
still takes **3.08% longer than compiled FP32**. Fusion removes much of the
conversion/launch overhead, but does not make compensated BF16 universally
faster; the complete encoder includes FP32 geometry and attention.

The preliminary GATr measurement is retained separately and excluded because
its process started before the final correlation-auxiliary source freeze.

## Detached restart

Recipes: `configs/shared_pretraining/vicreg_compiled_repair/`.
Fresh enlarged snapshot encoders, 12 epochs, batch 1,024, one seed, LR 0.002,
warmup/cosine schedule, online W&B, protected BF16 and compilation enabled.
The existing training-mean health gate remains at update 640. Selection still
excludes VICReg and the correlation auxiliary. No new data, relaxed-TDA targets
or causal-stage fits were added. The launcher freezes source/configs and saves
exact-resume checkpoints; see the [operational recipe](../../../docs/shared_pretraining_compiled_repair_20260918.md).

References: [VICReg reference implementation](https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py),
[PyTorch compiled backward semantics](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_backward.html),
[Triton matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).

Actual launches used the existing `queue worker` entry point with the same frozen source snapshot as the final benchmarks; workers were detached after each GPU completed verification.

- MACE: allocation 997799, detached launcher PID 1073610, launched 2026-09-18T09:51:03.630625+00:00; captured status running, step 176. [W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/mace-v5-compiled-0918).

- GATR: allocation 997864, detached launcher PID 1076808, launched 2026-09-18T10:02:15.600556+00:00; captured status running, step 1. [W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/gatr-v5-compiled-0918).
