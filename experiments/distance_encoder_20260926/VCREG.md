# CD-MACE128: direct prediction and representation regularization

**CD-MACE128** names the current joint crystal-distance MACE encoder.
**CD-MACE128-VC** is a matched, one-seed regularized treatment. Both are supervised;
the baseline already predicts a continuous censored distance distribution and
multi-radius probabilities, rather than only a crystal class.

The question is whether a modest feature-diversity objective preserves information
that the distance head does not require. Compare the same initialization, 12.75M
training rows, material weights, 12 full epochs, optimizer and global batch 4096.
Selection stays predictive likelihood; test results never select coefficients.

## Literature and choice

[VICReg](https://arxiv.org/abs/2105.04906) combines agreement between paired views
with per-feature variance and covariance penalties to avoid collapse and reduce
redundancy. Its full paired-view objective would introduce another assumption
about which atomic configurations should share an embedding.

[Zhu et al., VCReg](https://arxiv.org/abs/2306.13292) adapt variance/covariance
regularization to supervised representation learning and report improved transfer
on image/video tasks. Their supervised setting is the closest match here. We use
the vanilla regularizer directly on the exported state, a weighted population
covariance for material balance, coefficients .05/.05 and a one-epoch warmup.
The materials benefit remains an experimental hypothesis.

[Supervised contrastive learning](https://arxiv.org/abs/2004.11362) pulls together
examples sharing a class. We do not choose it for this run: our task has censored
continuous distances, and forcing a whole class together could suppress useful
within-liquid differences. This is our task-specific judgment, not a claimed
general failure of supervised contrastive learning.

No augmentations, physical reconstruction or new projection head are added.
Only batch statistics of the existing 128-vector are required, with differentiable
gathering across both GPUs. No additional encoder pass is needed.

## Evaluation

Each model gets its original local distance head, a fresh local distance MLP and
frozen linear/MLP 3/6-ps onset probes, using the fixed whole-source splits. None
receives vector/harmonic context embeddings. Evaluate distance likelihood/errors,
probabilities >.5/.75/.95, misses, false alarms, visibility, calibration and AP as
a diagnostic. Also measure overall and locally PTM-clear rank, exact .75-ps motion
rank/stability, and input-noise response normalized by local neighbor spacing.

Keep provisional evaluations distinct from completed 12-epoch results. A higher
rank accompanied by worse likelihood or increased noise sensitivity is not an
improvement. The original context comparisons remain separate.

[Metric definitions](../../docs/metrics/distance_encoder_local.md) ·
[Training recipe](../../configs/distance_encoder/multimaterial_early_vcreg_20260926.json) ·
[Execution](../../docs/distance_encoder.md).
