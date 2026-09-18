# Enlarged, geometry-protected snapshot encoders

Implemented and tested on 2026-09-18. These are implementation/precision results,
not evidence of improved learned prediction or TDA accuracy. No long fit was
started and no new data was generated.

| Encoder | Previous snapshot parameters | New | Ratio |
| --- | ---: | ---: | ---: |
| MACE | 73,491 | 147,139 | 2.0021× |
| GATr | 399,504 | 803,216 | 2.0105× |

Both export z128. MACE uses 32 tensor channels with readout hidden width 104;
GATr uses 8 multivector and 192 scalar channels. Snapshot GATr omits temporal
modules; the explicit history variant remains causal and trainable. Auxiliary
head sizes and scientific loss coefficients are unchanged.

## Precision and correctness

Geometry, geometric attention, residuals, pooling, readouts and exported states
are FP32. MACE radial MLPs use BF16. GATr scalar maps use compensated BF16 products
with FP32 outputs/accumulations. Details and the first-order gradient contract
are in [the implementation record](../../../docs/shared_pretraining_geometry_fp32_2x.md).

Twenty-six tests passed on each of H100 and RTX PRO 6000 Blackwell. Coverage:
upstream GATr FP32 equation/gradient agreement, actual tensor-core operand dtypes,
compensated output/input/weight-gradient accuracy, parameter budgets, rotational
invariance, packing, history causality, physical/TDA gradients, full-batch versus
cached gradients and exact optimizer/scheduler checkpoint continuation. An
additional identical local H100 test pass is retained separately, not counted
as an independent validation experiment.

The real-observation audit used 72 existing training observations, eight per
material/potential/static group, across nine groups spanning Al, Mg, Ta, Ti and
Zr. Each model had fresh initialization (seed 20260919). Checks used identity,
axis90 and three seeded proper rotations. Positions were rotated in FP64 then
rounded back to the input contract's FP32; masks and edges were preserved.

For each output (z, projector, physical decoder and TDA decoder), the relative
error is `RMS(rotated - original) / RMS(FP32_original - group_mean(FP32_original))`.
The denominator is computed separately within each eight-observation group.

| Check under mixed BF16 | MACE | GATr |
| --- | ---: | ---: |
| Worst rotation error / within-group FP32 signal, across all outputs | 0.0162% | 0.0682% |
| Largest absolute rotation RMS in z | 1.77e-7 | 8.25e-7 |
| Largest FP32-to-mixed difference in z / within-group FP32 signal | 0.339% | 0.399% |

Both pass the predeclared rotation gate of 1%. This finite audit does not
guarantee invariance for every future set of learned weights or observation.

Ordinary single-product BF16 scalar GATr failed the same audit: worst relative
rotation error 23.29%. It was replaced with compensated scalar products; the
failed measurements and source are retained in `technical/*single-bf16*`.

Blackwell compiled FP32/BF16 forward and backward checks passed with precision
casts preserved. Largest compiled-versus-eager BF16 differences in the synthetic
check were 4.77e-7 (MACE) and 2.06e-6 (GATr). Upstream graph breaks are allowed;
the trainer does not enable compilation automatically.

## Initial runtime check

All timings below used RTX PRO 6000 Blackwell, the same 64 largest-support Ta
static anchors, microbatch 64, two warmup updates and five measured full cached
updates. Each update includes two encoder forwards, physical/TDA/VICReg losses,
backward and AdamW. Data preparation is excluded. These disposable profiler fits
use LR 0.0003, not the prepared campaign's 0.002 peak. GPUs ran these profiles
sequentially without another workload on the measured GPU.

| Encoder / execution | FP32 seconds/update | Mixed BF16 seconds/update | Mixed speedup | Mixed peak allocated |
| --- | ---: | ---: | ---: | ---: |
| MACE, eager | 0.3853 | 0.3867 | 0.996× | 24.12 GiB |
| GATr, eager | 0.3760 | 0.4368 | 0.861× | 6.10 GiB |
| GATr, compiled encoder | 0.3581 | 0.3786 | 0.946× | 5.75 GiB |

The precision correction therefore **does not yet provide a speedup over FP32**
at this batch size. Compiled mixed GATr is 13.3% faster than eager mixed GATr,
but still 5.75% slower than compiled FP32. FP32 and mixed compilation/warmup took
63.5 and 41.6 seconds respectively, excluded from measured updates. These costs
are not isolated compiler times and may recur for new input shapes.

A candidate combining the three correction terms into one wider GEMM took
0.4373 seconds/update in eager mode, offering no improvement; it remains only a
disposable diagnostic. No claim is made about throughput at batch 1,024, larger
microbatches or H100/H200, or about relative speed against the smaller old models.

## Reproduction and status

Use `pointnet-torch214`. Fresh 12-epoch VICReg recipes are prepared under
`configs/shared_pretraining/geometry_fp32_2x/`, using the existing release,
batch 1,024 and peak LR 0.002 with cosine warmup. They are **unsubmitted**.
The diagnostic microbatch 64 fits within 40 GiB on Blackwell; larger operational
microbatches should be selected with the existing standalone profiler.

Run tests with:

```bash
python -m pytest -q tests/test_structural_precision.py \
  tests/test_structural_pretraining_model.py tests/test_shared_pretraining.py
```

Raw observations' row IDs, rotations, implementation identities, tests, timings,
profiling configurations and compiler checks are retained in `technical/`.
`summary.json` contains unrounded results. No historical research artifacts,
active jobs or exact-resume environments were changed.
