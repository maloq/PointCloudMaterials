# MACE paired alignment with geometric Epi regularization

Question: does replacing VICReg's variance/covariance regularization with the
existing geometric Epi objective preserve useful continuous liquid structure and
interfaces, without conditional JEPA prediction or physical-label supervision?

Three treatments, each from the same scratch encoder initialization and exact
anchor order within each of seeds123/456:

1. VICReg: `(25 I + 25 V + C)/51`.
2. Epi: `25 I/51 - 0.1 E/E_initial`.
3. Epi with variance floor: `(25 I + 25 V)/51 - 0.1 E/E_initial`.

`I` is the MSE between raw128-dimensional invariant exports of the same tracked
center at t and t+0.75ps. These temporal positives are an assumption to test;
they can suppress real rapid ordering events. Both views receive gradients.
No separate projector, JEPA predictor, physical/order/TDA reconstruction loss,
temperature input or phase stratification enters encoder optimization.

`E` averages the repository's RMS-controlled ridge/logdet score over both views.
The target reservoir is the existing frozen random width16 MACE with a fixed
128→64 projection. It is neither a trained teacher nor physical ground truth.
Initial score calibration uses four fitting batches only. The variance-floor
arm tests whether scale contraction under alignment limits the pure Epi arm.
Epi weight0.1 is predeclared, not optimized against development metrics.

Architecture: the existing neighborhood MACE width64 snapshot encoder, raw
invariant export, full native radius support, BF16 encoder/FP32 losses and
FP64 Epi ridge/logdet. Auxiliary equivariant outputs are not directly supervised
or used as exported analysis features in this study. The same architecture and
initial parameters apply to all three treatments.

Training: the frozen32,768-anchor native Al MEAM release (400/450/500/510/520K),
ancestry-disjoint from its480 development anchors. Fresh AdamW, encoder LR1e-4,
weight decay1e-4, 10% warmup/cosine decay, B512 with exact full-batch statistics,
24 complete shuffled passes =1536 updates. Each anchor occurs exactly once per
pass. This differs from the older JEPA study's sampled epoch equivalents.
Save fixed milestones0/4/12/24; no best-checkpoint selection.

Evaluation reuses the fixed Al structural screen, nonbulk/liquid-neighbor
supplement, perturbation sensitivity, coarse temporal response, and conditional
9ps residual-order /12ps onset prediction with source bootstrap intervals.
Dense spatial plots at12/24 passes retain8×samples and half-diameter markers.
Forecasting uses the existing15 reused development roots, not a fresh test set;
static references are a transductive screen. Two training seeds are distinct
from source-bootstrap uncertainty. Ta/Zr transfer is not tested in this Al run.

Recipe: `configs/mace_epi/campaign.json`.

```bash
python -m src.research.mace_epi.queue submit --config configs/mace_epi/campaign.json
```

Findings: all six fits and24 checkpoint evaluations completed. See the
[cross-study results](../../docs/encoder_research/results_20260924.md#mace-paired-alignment-epi-versus-vicreg).
Plain Epi improves the final present-structure readouts over VICReg but has lower
onset AP. The variance floor improves Epi's onset AP; none of the three objectives
establishes improved Brier or9 ps residual prediction beyond current physics.
Primary comparisons are matched seed/pass liquid-neighbor
NMSE, nonbulk fault/interface AP and spatial boundary AUC; report continuous
order recovery and perturbation response alongside them. Require calibrated
onset benefit beyond current physics before claiming better crystallization
prediction. Attractive UMAP separation or higher rank alone is not success.
