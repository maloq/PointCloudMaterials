# Al-only stability repair — September 18, 2026

The v5 GATr worker failed at update 2 with compiler cache exhaustion. MACE was
stopped cleanly at update 648. Its physical selection score improved overall,
but VICReg variance remained concentrated in roughly one direction and its
covariance term spiked on static Mg. Those checkpoints, logs, frozen source and
normalization definitions are preserved. See the [diagnosis and results](../output/shared_pretraining/stability-20260918/RESULTS.md).

The active recipes are `configs/shared_pretraining/al_stable/`. They select only
Al shards from the registered `broad-250k-v2-20260917` release: 104,864 MEAM,
15,016 EAM and 5,120 static training anchors. The static potential remains
unknown. No data is regenerated. Target moments are refitted on **Al training
endpoints only**, with the original producer's valid-TDA mask and 1e-4 scale
floor. Whole-source selection remains 480 observations from 15 held-out native
Al MEAM sources; this is not validation of EAM/static generalization. Physical
errors on this normalization cannot be compared numerically with old broad-data
scores without recomputation.

Each optimizer update still samples a homogeneous potential/static group, so
between-group differences cannot satisfy VICReg's variance requirement. Both
spatial and temporal neighbors remain views. Only instantaneous TDA is used.
The physical, TDA, VICReg and correlation coefficients are unchanged from v5.

The encoders retain 147,139 MACE / 803,216 GATr parameters, FP32 geometry and
selective BF16 scalar operations. The projector and physical/TDA heads now
condition feature differences across the **full statistical batch**, using input
BatchNorm (eps 1e-6). The projector has Linear–BatchNorm–ReLU–Linear after this
input conditioning. Physical/TDA heads retain their hidden LayerNorm and SiLU.
These batch statistics never enter the exported encoder. Gradient caching
recomputes only the encoder; heads and BatchNorm update once per optimizer step.

Moving encoder features make ordinary EMA statistics stale. Before every
selection pass, encode the same fixed, proportionally stratified sample of
1,024 **training** anchors. Fit input moments in float64; then fit the hidden
projector moments in inference order. Calibration uses no labels and no
selection/test inputs, changes no parameters, and is saved in validated best
checkpoints. Inference uses these frozen buffers and does not depend on batch
peers. `last.pt` is exact optimizer-resume state; `best.pt` is the calibrated,
validated model. The 128-dimensional exported encoder remains observation-wise.

One seed, batch 1,024, 12 epoch equivalents = 1,465 updates. Head peak LR is
0.002; encoder peak LR is 0.0002 (a 0.1 multiplier). Both use the same 147-update
warmup and cosine multiplier. Microbatches are 64 on H100/MACE and 256 on
RTX6000PRO/GATr; the allocator cap is 40 GiB. Online W&B logs both rates and
separates training curves by potential/static group and spatial/temporal view.
The health gate requires at least 5% improvement over the training-mean baseline
after update 256 and rejects three consecutive validations exceeding 1.5 times
the previous best (or losing the required baseline gain).

GATr compilation now uses a tensor-only equivalent of upstream attention,
removing its xFormers graph-break wrapper. Both GATr's einsum cache and
PyTorch's independent opt_einsum path search are disabled. The latter otherwise
specializes every atom count. A contiguous grade-zero scalar view and runtime
Triton strides avoid static saved-tensor stride assumptions in variable-size
backward. Full-graph compilation is required, including gradient-cache replay;
limit exhaustion raises rather than falling back. The workload gate exercises
all three Al groups, batch sizes 1,024/128, variable atom counts, evaluation,
no-grad caching and backward, before the detached fit.

Use the existing launcher, once the corresponding output directory is unused:

```bash
python -m src.training_methods.shared_pretraining.queue submit \
  --plan configs/shared_pretraining/al_stable/campaign.json
```

Source/config snapshots and exact launch receipts live in the campaign's
`technical/`. No causal stages, new allocations or simulations are launched by
this repair. Separate timing probes remain outside training.
