# Encoder initialization with directional spatial predictors

Question: does retaining geometry through structural pretraining improve the
information available to vector-message and harmonic-hierarchy onset predictors?

The [previous context study](../equivariant_context_20260925/README.md) motivates
those two predictors. Earlier onset-only fits overfit with prolonged training;
paired VICReg gave stronger 3 ps readouts than plain Epi, whereas Epi with a
variance floor retained more variation than VICReg. These observations motivate
the treatments; historical AP-selected and condition-using scores are not matched
baselines for this study.

| Treatment | Initialization before onset training |
| --- | --- |
| Scratch | Random native MACE, no pretraining |
| Physical | 12 epochs reconstructing current radial/count/angular structure on 1,157,760 observed neighborhoods |
| Paired VICReg | 12 epochs aligning 86,400 same-time observed/relaxed pairs with variance/covariance regularization |
| Paired Epi + variance | Same pairs and 12 epochs, geometric random-reservoir Epi score with a variance floor |

Each initialization produces two independently fine-tuned encoders, observed and
relaxed. Every encoder is trained for 24 complete epochs on the fixed Al64 onset
training population. NLL selects among checkpoints from epoch 12 onward; epoch-12
snapshots and all validation curves are retained. Earlier checkpoints are logged
to reveal overfitting, but are not substituted for the requested minimum training.
No AP objective, selector, promotion rule or fitted post-hoc ensemble is used.

Freeze each newly trained encoder, apply its shared weights to all 25 patches,
and train both vector messages and harmonic hierarchy for 24 epochs, also with
NLL selection from epoch 12 onward. This makes 3 structural initializations,
8 supervised encoder fits and 16 context fits, plus independent diagnostic probes.
Use one seed, width/export 128, batch/microbatch 256, native cuEquivariance and
the fixed source/sample contract. Paired pretraining has no onset labels, but
the final encoders belong to the supervised branch after fine-tuning.

Encoder input is current center-relative geometry in the same nearest-80/8 A
support, two 5 A message-passing blocks, no halo or motion/history. Prediction
uses the same 25-patch 0/10/20 A stencil, relative geometry and typed fields from
that encoder. Neither temperature, age nor explicit time covariates enter any
encoder, predictor, probe or baseline. Same-time relaxation is an explicit input
for the relaxed arm and a training view for the paired initializations.

The full evaluation includes all64 and legacy16 prediction tracks, calibrated
3/6 ps proper scores and AP, whole-source intervals, local linear/MLP and physical
controls, structural retention, whole-dataset/test spectra, normalized input-noise
response and observed dynamics at exactly 0.75 ps with movement rank/d95. Dense
relaxed dynamics are unavailable and are recorded as such. These historically
examined test sources do not provide fresh confirmatory evidence. Physical versus
paired pretraining differs in data size, total compute and objective.

[Recipe](../../configs/encoder_context/al64_20260925/campaign.json) ·
[Definitions](../../docs/metrics/encoder_context.md) ·
[Operations](../../docs/encoder_context.md).
