# Why the temporal MACE pilot failed — September 9, 2026

Training was stopped at the user's request after 16 completed epochs. The last
complete checkpoint contains 288 of the planned 432 updates; the selected
checkpoint is epoch 15. No training was restarted. Frozen inference, linear
probes and derivative measurements were run on the preserved checkpoints.

The strongest evidence points to an inadequately balanced objective damaging
the usefulness of the temporal representation. It does not establish that
transformers or multi-frame inputs are unsuitable.

## 1. The regularizer overwhelmed the scientific target

The implemented objective was TDA MSE + 25 × variance penalty + covariance
penalty. From epoch one to epoch sixteen:

| Training measurement | Epoch 1 | Epoch 16 |
| --- | ---: | ---: |
| TDA MSE | 0.97454 | 0.90980 |
| Weighted regularization | 28.56987 | 6.90927 |
| Total loss | 29.54442 | 7.81907 |

Only 0.06474 of the 21.72535 reduction was TDA improvement: **99.70% of the
total loss reduction came from regularization**. A falling total loss therefore
gave a misleading impression of task progress.

Derivative measurements confirm the imbalance, rather than merely comparing
loss values with different scales:

| Regularization-gradient norm / TDA-gradient norm | Initial | Selected |
| --- | ---: | ---: |
| Full 1,536-anchor embedding batch | 234.6× | 22.4× |
| Encoder parameters, first 64 histories with full-batch derivatives | 613.1× | 42.0× |

The parameter-gradient cosine is −0.175 initially and −0.013 at selection;
the task and regularizer are not strongly aligned. These are measurements on
one fixed training batch, before Adam preconditioning, not an average over
training or a controlled causal ablation.

The 25/1 coefficients were carried into the new supervised protocol without
first calibrating their gradients. That was a poor experimental choice.
[VICReg's original objective](https://arxiv.org/abs/2105.04906) includes an
invariance term; its coefficients do not establish an appropriate balance
against this differently scaled TDA target.

## 2. The fusion stage became worse for topology decoding

All frozen ridge probes below use the identical 8,192 training anchors, all
1,024 validation anchors, the same standardized 32-component target, training-only
feature scaling and ridge alpha 1. No backbone or transformer is updated by them.

| Features / predictor | Validation MSE | Mean within-material R² |
| --- | ---: | ---: |
| Initial MACE, anchor features + ridge | 0.97247 | 0.2577 |
| Initial MACE, mean of five frame features + ridge | **0.93400** | **0.2896** |
| Initial temporal encoder + ridge | 0.95768 | 0.2653 |
| Trained temporal encoder + ridge | 1.12268 | 0.1045 |
| Actual trained TDA head | 1.17098 | 0.0529 |
| Trained MACE, anchor features + ridge | 0.97186 | 0.2583 |
| Trained MACE, mean of five frame features + ridge | 0.93287 | 0.2906 |

The temporal representation's probe error increased by 17.2% during training.
The MACE features before temporal fusion retained almost the same predictive
quality. This locates the observable degradation downstream of the backbone,
in the learned projection/temporal representation, rather than showing that the
backbone lost all useful structure. Linear decodability is the measured property;
this is not a proof about every possible nonlinear decoder.

A simple mean is only a control, not an assertion that arbitrary embedding
averages are meaningful. Here it outperforms a single observation by about 4%
under the same trained readout. There is useful information in the additional
observations; this objective did not learn to combine it effectively.

## 3. History barely helps the trained prediction

| Inference input | Validation TDA MSE |
| --- | ---: |
| Actual five-frame history | 1.17098 |
| Repeat anchor in all five slots | 1.17385 |
| Reverse four past frames; retain anchor | 1.17133 |

The benefit relative to repeated-anchor input is only 0.245%. Reversing past
order changes error by about 0.030%. Yet the corresponding embedding RMS changes
are 0.381 and 0.212. The representation responds to history much more than its
topology predictions benefit from it.

Attention is not numerically disconnected: all frames have nonzero gradients.
On the first 128 validation Al histories, the first block's average anchor
attention allocates 44.7% to the anchor; the second allocates 28.9%. Attention
weights alone are not explanations of predictive contribution.

## 4. Target scaling and source coverage make the task harder

The first 4 PCA coordinates contain 99.6407% of the global raw target variance;
16 contain 99.9935%. Whitening all 32 assigns similar training variance to much
smaller directions. The first/last component standard deviations differ by
1,454×, equivalent to about 2.11 million times the squared-error weighting per
unit raw displacement. [PCA whitening removes relative variance scales](https://scikit-learn.org/1.0/modules/generated/sklearn.decomposition.PCA.html).

Small variance does not prove that a coordinate is noise: it can encode the
subtle topology of interest. This needs a physical relevance/repeatability
assessment, not automatic removal of all small components. Nor is whitening
the entire explanation: Mg and Ta have large errors in leading coordinates too.

| Material | Training-material-mean predictor MSE | Trained head MSE | Head within-material R² |
| --- | ---: | ---: | ---: |
| Al | 0.79602 | 0.59407 | 0.1681 |
| Mg | 2.60495 | 2.58879 | −0.0595 |
| Ta | 0.97032 | 0.90699 | 0.0502 |

The Mg head is almost a mean predictor. Its validation target variance is 2.443
in the training-whitened basis, compared with 0.714 for Al and 0.955 for Ta.
Consequently Mg contributes about 55% of total validation error despite being
25% of validation anchors. The aggregate raw-144 R² is 0.809, but the mean
within-material raw-144 R² is −0.159: material differences conceal weak local
structural prediction. Overall raw R² would also be misleading here.

The 26,624 training anchors come from only nine source trajectories and one
target time per source (four Al, four Mg, one Ta). They are many neighborhoods,
not thousands of independent structural evolutions. Validation sources are
held out within campaigns, with Ta sharing a trajectory at later times. More
epochs repeat these same contexts.

The targets are full-cell minima, not small coordinate jitter: mean hot-to-relaxed
local neighbor displacement is 1.004 Å for Al, 0.788 Å for Mg and 0.496 Å for Ta.
These are differences of stored centered offsets, excluding the central atom;
periodic-image changes can contribute. This motivates checking target/basin
stability but does not by itself prove the 0.4 ps history is insufficient.

## What is established and what to change next

The gradient replay and identity checks passed, including exact agreement of
history anchors with the original paired cache. This is not explained by absent
history gradients or a demonstrated frame/target indexing error. The checks do
not establish that all architectural or physical assumptions are optimal.

The most useful next controlled experiment would freeze MACE, train only the
temporal module/readout with TDA supervision, and compare against the existing
frozen-MACE mean and anchor probes. Start with one material and meaningful
per-component metrics. Any representation regularizer should be added in a
separate small ablation with measured gradient influence; the full current
25/1 recipe should not simply be continued. Reassess target scaling and add
independent trajectory/time contexts before interpreting a larger run.

No such training has been launched. A GRU comparison would not isolate the
observed loss-design problem until these controls are in place.

## Evidence and reproduction

- [Diagnostic measurements](../../output/mace_temporal_transformer_20260909/diagnosis/diagnostics.json)
- [Frozen-probe report and plots](../../output/mace_temporal_transformer_20260909/analysis/RESULTS.md)
- [Failure diagnosis plot](../../output/mace_temporal_transformer_20260909/diagnosis/failure_diagnosis.png)
- [Training stop summary](../../output/mace_temporal_transformer_20260909/training_summary.json)
- [Diagnostic implementation](diagnose.py)

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 conda run -n pointnet python -m \
  experiments.mace_temporal_transformer_20260909.diagnose \
  --config experiments/mace_temporal_transformer_20260909/training.json
# Re-render figures from completed JSON without model inference:
conda run -n pointnet python -m experiments.mace_temporal_transformer_20260909.diagnose \
  --config experiments/mace_temporal_transformer_20260909/training.json --figures-only
```

The report and diagnostic are versioned experiment records. Generated measurements,
figures and failure/retry logs stay in the run output. The diagnosis used no neural
optimizer updates. The failed initial diagnostic attempts (module invocation,
then an unsupported retained compiled backward graph) and successful retry are
preserved; the latter recomputes each graph for gradient measurement.
