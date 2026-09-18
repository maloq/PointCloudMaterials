# Compiled VICReg repair, September 18

The user requested stopping the preceding MACE/GATr runs, investigating the
VICReg plateau, restarting compiled training, and testing a custom GATr kernel.
The old MACE worker saved update 1,067 and exited; GATr had already failed its
update-640 baseline gate. Both old source snapshots and checkpoints are retained.

The new recipes are in `configs/shared_pretraining/vicreg_compiled_repair/`.
They use the existing broad 250k release, one seed (20260919), snapshot inputs,
12 epochs / 2,930 updates, batch 1,024, peak LR 0.002, 293 warmup updates and
cosine decay to 0.00002. MACE uses 64-observation microbatches on H100; GATr
uses 256 on RTX PRO 6000 Blackwell. The allocator cap is 40 GiB. The train-only
mean baseline must still be beaten by update 640. No new simulations or data.

The v4 enlarged encoders and physical/TDA/VICReg coefficients are retained.
A 0.1 physical-correlation auxiliary supplies a contrast-sensitive gradient
when the physical decoder is close to constant. The exact definition and its
raw-target variance eligibility are in [the metric contract](metrics/shared_pretraining.md#compiled-repair-protocol-v5).
It is training-only: exported encoders and inference remain observation-wise,
without batch normalization, held-out statistics, or dependence on other inputs.
Selection still measures physical MSE + 0.25 instantaneous-TDA MSE, excluding the
auxiliary and VICReg. Projector batch normalization was tested but not adopted:
it reduced VICReg without demonstrating better physical decoding in the pilot.

`compile_encoder=true` compiles in place with dynamic shapes, keeping checkpoint
names unchanged. GATr's documented einsum-cache switch is disabled; precision
casts are preserved; compiled backward autocast is off because backward runs
outside autocast. Compiler limit exhaustion raises an error instead of silently
falling back. PyTorch's [autograd semantics](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_backward.html)
explain this separate compiler requirement; the old uncompiled runs were not
caused by this setting. Library graph breaks remain permitted and recorded in
`technical/compilation.json`.

GATr's custom Triton operation fuses FP32-to-BF16 high/low decomposition and the
three tensor-core products, accumulating in FP32. Weight gradients use a
deterministic split-K reduction without atomics. Geometry, normalization,
readouts and heads stay FP32. First-order autograd uses the same compensated
map for input and weight gradients; higher-order differentiation is unsupported.
See [Triton's matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
and the original [compensated arithmetic method](https://arxiv.org/abs/1904.06376).

Benchmarks are separate processes, never a training stage. They compare the same
model, seed, largest-support batch and optimizer updates against eager FP32
(TF32 off), and retain compiled FP32 and three-GEMM controls for GATr. GPU timing
includes transfer, gradient caching, full-batch heads/losses, backward and update;
it excludes CPU graph preparation and compilation. Initialization/warmup time
and compiler counters are retained separately. See the [report](../output/shared_pretraining/vicreg-repair-20260918/RESULTS.md).

Submit from the repository root in `pointnet-torch214`:

```bash
python -m src.training_methods.shared_pretraining.queue submit \
  --plan configs/shared_pretraining/vicreg_compiled_repair/campaign.json
```

The launcher freezes code/configs and starts detached structural-only workers on
the existing allocations; no causal continuation is submitted. Status, exact
resume state, online W&B links and immutable launch receipts live under each
run's `technical/` directory. Do not resubmit an existing campaign directory.

## Actual launch receipt

The existing `queue worker` entry point was started detached for each architecture
after its GPU finished the separate benchmarks, using the same frozen source
snapshot as those measurements. The combined receipt is
`output/shared_pretraining/compiled-repair-campaign-20260918/technical/submissions.json`.
MACE uses H100 allocation 997799; GATr uses RTX6000PRO allocation 997864.
Both are fresh 12-epoch fits. Source/configs are frozen under the diagnosis
run's `technical/code/`, and the receipt records that exact path.
