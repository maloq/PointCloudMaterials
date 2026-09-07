# Task-trained atomic encoders: September 5, 2026

All nine representations now have task-relevant training and validation tuning.
MACE, SchNet-style, density MLP and GeoFrame were trained from scratch on
continuous current structure and repeated shooting outcomes. SOAP, density PCA,
TDA and coarse-order descriptors received trained nonlinear heads. No OVITO/PTM
phase labels were used.

The sweep completed **45 trials and 27 selected checkpoints**, with three
learning-rate candidates and three final seeds per representation. It used
4,972 sampling epochs and approximately 50 minutes on the node's H100. Each
selected run trained for 80–229 epochs; checkpoint selection used validation
sources. All passed the audited plateau criterion. The audit corrected a
stopping-rule issue involving accumulated small improvements and extended one
GeoFrame seed from 207 to 229 epochs before evaluating the test set. Its selected
checkpoint stayed at epoch 189.

## Forecast comparison

Scores are percentage reductions in MSE relative to a **trained nonlinear
coarse-bond-order + temperature baseline**; positive is better. Results average
three seeds and 12/24/48 ps horizons within each target family. They are not
classification accuracies.

| Representation | Future topology | Future bond order | Future mobility |
|---|---:|---:|---:|
| Reference MACE | +2.08% | -14.31% | +3.16% |
| SchNet-style | +3.29% | -13.06% | +2.97% |
| Smooth-density MLP | **+7.05%** | -6.63% | +0.34% |
| GeoFrame v2 | +3.11% | -15.18% | +2.99% |
| SOAP + PCA | +5.69% | -6.64% | -2.19% |
| Smooth-density PCA | +6.63% | -8.21% | -1.40% |
| TDA, 16 PCs | +6.47% | -14.80% | **+7.15%** |
| TDA, 128 PCs | -50.16% | -146.98% | -439.45% |
| Coarse-order baseline | 0.00% | **0.00%** | 0.00% |

Density MLP has the largest topology point estimate, with a 95% source-bootstrap
interval of [+3.34%, +9.99%] against the baseline. Compact TDA has the largest
mobility point estimate, [+0.41%, +14.13%]. These intervals do not establish
that either model beats every other model. The coarse-order baseline has the
best bond-order point estimate. Several model-versus-baseline intervals include
zero; the complete report retains them rather than declaring one overall winner.

The nonlinear heads take an embedding and temperature. They do not also receive
the explicit coarse-order inputs. A separate matched linear probe appends those
inputs to every representation, as in the previous benchmark. Thus a negative
direct forecast score does not prove that an embedding has no additional useful
information.

## What changed for MACE and the embeddings?

Under the original coarse-augmented linear probe, MACE's topology/order/mobility
scores improve from **+0.81/+1.05/+4.68%** with jitter VICReg to
**+1.24/+1.44/+5.54%** with task supervision. Paired source intervals support
small improvements in topology and mobility, while the order improvement's
interval includes zero. Longer, task-relevant training helps, but does not close
the gap to the density representations in this benchmark.

MACE's held-out effective rank remains about 4.9 of 128 dimensions, and its
matched-neighbor future-agreement gain is -0.35% [-1.01%, +0.31%]. Low effective
rank alone is not proof of failure, but the retrieval assay does not show useful
separation beyond the matched coarse-order neighbors. Density MLP reaches rank
43.4 and improves matched-neighbor future agreement by **4.79%
[3.55%, 5.95%]**. This makes it the stronger current candidate for organizing
subtle environments by their future behavior under this protocol.

Compact TDA's nonlinear mobility result is useful evidence that the readout
matters: its original linear probe still scores -6.22%. The 128-PC variant
generalizes poorly despite tuned, converged heads. A readout audit also found
that applying the heads' affine BatchNorm before the supposedly matched probe
changed its variance-floor behavior on near-null TDA coordinates. The final
matched probes use the original fixed descriptors; the first analysis is
preserved under `analysis_before_fixed_probe_correction/`. No model was retrained
or selected using test results for this correction.

## Stability remains an architecture issue

The following is the worst per-material, per-seed 95th-percentile embedding
change, normalized by the corresponding material's training embedding spread.
The controls use the same environments as the preceding benchmark.

| Encoder | Rotation | 0.0001 Å coordinate noise | 0.02 Å coordinate noise |
|---|---:|---:|---:|
| Reference MACE | 0.000014 | 0.00121 | 0.242 |
| SchNet-style | 0.000103 | 0.00086 | 0.174 |
| Density MLP | 0.000003 | 0.00097 | 0.209 |
| GeoFrame v2 | **0.271** | **0.303** | **1.211** |

Corrected MACE passes the final GPU rotation control. GeoFrame still has large
outliers, including Mg environments under rotation and Al/Mg environments under
tiny coordinate perturbations. This training has not repaired its canonical-frame
discontinuities. These are perturbation controls, not a new full-trajectory
smoothness measurement.

## Scope and remaining uncertainty

Current-geometry training includes Al, Mg and Ta with material-balanced losses.
Independent future supervision and the primary test are **Al only**: 3,054
initially noncoherent centers from six source runs, with eight shooting futures
per parent. This follow-up reuses a previously examined test split. Whole sources
remain separated, and all fitting and validation selection finished before test
evaluation.

Geometry targets are BOO, compact alpha-persistence and SOAP measurements, not
ground-truth phase identities. One static Al validation environment contributes
about 87% of the current TDA target energy relative to the training mean; this
heavy tail affects the geometric part of selection. Targets and samples were
retained, and independent mobility and retrieval assays are reported alongside
topology. Neighborhoods and capacities also differ between architectures. This
is controlled task training and learning-rate tuning, not an exhaustive search
or an energy/force-trained MACE potential comparison.

Three training-metric tests passed, all nine models passed forward/backward
preflight, and the final MACE rotation control passed. All generated artifacts
are physically in the repository.

- [Complete methods, source intervals, seed variation and training curves](../output/predictive_encoder_training_20260905/RESULTS.md)
- [Forecast comparison CSV](../output/predictive_encoder_training_20260905/comparison.csv)
- [Matched linear probes](../output/predictive_encoder_training_20260905/linear_probe_comparison.csv)
- [Paired improvements over earlier training](../output/predictive_encoder_training_20260905/improvement_over_previous_training.csv)
- [Selected checkpoints](../output/predictive_encoder_training_20260905/selected_runs.json)
- [Experiment recipe and configuration](../experiments/predictive_encoder_training_20260905/README.md)
