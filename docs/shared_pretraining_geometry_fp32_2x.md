# Enlarged structural encoders with protected geometry

Use `conda activate pointnet-torch214`. The architecture revision is
`structural_v4_geometry_fp32_2x`. The shared trainer records protocol v4 and hashes
the precision implementation and CUDA dependencies. Old checkpoints require
their original frozen code; there is no weight or optimizer migration.

## Model sizes

The requested factor is relative to parameters in the previous **snapshot**
encoder, excluding auxiliary heads and GATr's previously unused temporal blocks.

| Snapshot encoder | Previous | New | Ratio | Widths |
| --- | ---: | ---: | ---: | --- |
| MACE | 73,491 | 147,139 | 2.0021 | 32 tensor channels, two interactions, readout hidden width 104 |
| GATr | 399,504 | 803,216 | 2.0105 | 8 multivector and 192 scalar channels, two spatial blocks |

Both export one 128-dimensional state. Readout hidden width 104 keeps MACE near
the doubled parameter budget while using 32 tensor channels. Physical/TDA heads
and the VICReg projector retain their previous widths. Complete stored models
have 275,880 (MACE) and 931,957 (snapshot GATr) parameters, including the unused
JEPA predictor. No parameters were added solely to reach the count.

Snapshot GATr now allocates only its spatial path. `StructuralModel('gatr',
history=True)` adds two interleaved causal temporal blocks; maintained trainers,
profiling and extraction set this from `history_frames`. A snapshot model rejects
multi-frame input explicitly. MACE remains snapshot-only in this protocol.

## Arithmetic

With `precision: bf16`, master weights and optimizer state remain FP32:

- MACE evaluates its invariant radial MLPs in BF16, then converts their outputs
  to FP32 before tensor products. Spherical harmonics, equivariant projections,
  products, residuals, normalization and multiscale pooling stay FP32.
- GATr keeps multivector projections, scalar/multivector mixing, geometric
  products and joins, norms, joint geometric attention and residuals in FP32.
  Scalar-to-scalar maps use compensated BF16 tensor-core arithmetic, described
  below. Attention is not reduced to independent scalar-only attention.
- Pooling/readouts, exported states, physical/TDA decoders, projector and latent
  predictor stay FP32 even under outer autocast. VICReg moments remain FP32.
- `precision: float32` disables scalar AMP as well. The training runtime disables
  TF32; precision/invariance checks must keep the same setting.

Ordinary BF16 scalar maps were insufficient on real observations: quantization
amplified tiny rotation-rounding differences relative to small within-material
signals. GATr therefore splits each scalar matrix operand into BF16 high and low
parts and evaluates `Ah @ Bh + Ah @ Bl + Al @ Bh`, with each product returning
FP32. This follows the decomposition principle of
[Henry, Tang and Heinecke (2019)](https://arxiv.org/abs/1904.06376).
It uses three products, not the speed/cost of an ordinary single BF16 GEMM.

PyTorch 2.14 does not supply a backward for `torch.mm(..., out_dtype=float32)`.
Our explicit backward evaluates the linear-map input and weight gradients with
the same compensated products. Like AMP, these approximate the FP32 map's
gradients rather than differentiating rounding discontinuities. Tests compare
both gradients and outputs to FP32 with relative norm error below 5e-5. This
custom operation supports first-order training, not higher-order force training.

## Validation and recipes

The [validation record](../output/shared_pretraining/geometry-fp32-2x-20260918/RESULTS.md)
contains exact tests, rotation errors, compilation results and operational timings.
The initial Blackwell batch-64 check shows no speed advantage over FP32: MACE is
approximately tied; mixed GATr is 16.2% slower in eager mode and 5.75% slower with
both encoders compiled. The priority achieved here is precision and capacity.
Do not extrapolate the earlier blanket-BF16 speedup to this arithmetic.
The rotation audit uses eight existing training observations from each of nine
material/potential/static groups: 72 observations covering Al, Mg, Ta, Ti and Zr.
It checks identity, an axis rotation and three seeded SO(3) rotations. Each group
uses its own FP32 between-observation RMS as the denominator, including separate
checks for z, projector and physical/TDA predictions. These are newly initialized
models; prediction quality still needs training and held-out evaluation.

Fresh, unsubmitted 12-epoch recipes are:

- `configs/shared_pretraining/geometry_fp32_2x/mace_vicreg_structural.json`
- `configs/shared_pretraining/geometry_fp32_2x/gatr_vicreg_structural.json`

They use the existing broad release, one seed, batch 1,024, peak LR 0.002, cosine
warmup, physical/instantaneous-TDA anchors, VICReg, online W&B and the existing
learning-health checks. Microbatch 64 is an initial operational choice for the
wider FP32 geometric path and must be profiled on the destination GPU. No new
simulation or label preparation is needed. No long training job was submitted
by this architecture change.

Use the existing separate profiler before increasing microbatch size:

```bash
python -m src.training_methods.shared_pretraining.profile \
  --config configs/shared_pretraining/geometry_fp32_2x/gatr_vicreg_structural.json \
  --microbatch 64 --memory-limit-gib 40 --precision bf16 \
  --output output/shared_pretraining/NEW-PROFILE/technical/profile.json
```

Compilation is validated separately with precision casts preserved, but remains
disabled in the trainer. GATr still has upstream graph breaks. The compiled
check does not establish a whole-graph implementation or end-to-end speedup.
Add `--compile` to the standalone profiler to measure this option; its recorded
`warmup_seconds` includes compilation and warmup updates.
