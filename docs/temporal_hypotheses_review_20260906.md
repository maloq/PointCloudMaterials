# Temporal encoder campaign review — September 6, 2026

Predictive density embeddings are the strongest current compromise between
smoothness, structural information and future prediction. Proper temporal MACE
training now gives useful improvements, including a clear gain over its previous
descriptor-supervised version. GeoFrame remains unsuitable for this objective
in the tested configuration. Temporal invariance produces a useful but very
low-dimensional structural representation; it should not be called a complete
predictive state.

## Execution and recovery

All **33 scheduled training runs completed**: 24 fifteen-minute screening trials
and nine confirmation trials of approximately 36.1 minutes. Training consumed
11.42 hours and 2.783 billion sampled examples, including repeated draws. The
cache has 6.14 million local clouds, 5.64 million assigned to training. Those
numbers are not counts of independent simulations.

The original report stage stopped after 6 of 18 selected models, leaving stale
`running` status files. Both supervisor and worker were absent when checked.
There was no Python traceback or supervisor exit record. Scheduler accounting
was unavailable, so the cause is **unknown**; a timeout or scheduler cleanup
cannot be established from the available evidence. The server had not rebooted.
Detachment does not protect a process from the end of a Slurm allocation.

All selected checkpoints were intact. The completed evaluation was replayed
without training or reselection, and the original partial artifacts were moved
to `interrupted_analysis/`. The twelve already-selected screening winners were
also evaluated separately to provide comparisons at the same training budget.
All 18 primary evaluations and all 12 screening evaluations completed; the
three existing temporal-campaign tests passed. Training provenance is unchanged;
analysis changes are recorded separately.

## Main held-out comparison

Scores are percentage reductions in MSE relative to the original **linear
coarse-bond-order + temperature baseline**. Each frozen embedding is appended
to those same baseline inputs, and ridge readouts are fitted on training sources
and tuned on validation sources. These numbers cannot be compared directly with
the earlier report's nonlinear-head table, which used a different baseline.
TDA, SOAP, bond-order and PTM targets did not train or select these encoders.

The test contains 3,054 initially noncoherent Al environments from six source
simulations, with future targets at 12, 24 and 48 ps. Geometry descriptors are
continuous assays, not phase labels or proof of a new liquid structure.

| Encoder/objective | Training tier | Future topology | Future order | Future mobility | Future-neighbor agreement |
|---|---|---:|---:|---:|---:|
| Density, predictive | 3 confirmation seeds | +6.91% | +7.96% | **+14.55%** | +4.71% |
| MACE, predictive | 3 confirmation seeds | +3.43% | +4.32% | +9.21% | +3.58% |
| SchNet, predictive | 3 confirmation seeds | +3.26% | +3.97% | +6.22% | +0.80% |
| Density, temporal invariance | 1 screening seed | **+8.46%** | **+8.22%** | +8.92% | **+5.03%** |
| Density, static | 1 screening seed | +5.76% | +6.20% | +10.35% | +3.55% |
| GeoFrame, predictive | 1 screening seed | -0.64% | -0.66% | -1.04% | -1.48% |

For predictive density, the source-bootstrap 95% intervals are topology
[+2.59%, +10.30%], mobility [+9.65%, +18.09%] and neighbor agreement
[+3.49%, +5.81%]. Its mobility scores span only +14.31% to +14.72% across
confirmation seeds. MACE's mobility interval is [+7.43%, +10.74%] and neighbor
interval [+2.51%, +4.61%]. Temporal invariance's mobility interval crosses zero,
[-3.69%, +16.86%]. Intervals resample six sources and condition on the selected
seeds; they do not include model-selection uncertainty or correct for the many
exploratory comparisons. The test sources have been examined in earlier work.

![Future-assay scores and source intervals](../output/temporal_hypotheses_12h_20260906/comparison_intervals.png)

## What the controlled ablations say

The following comparisons use **only the equal-budget screening models**,
each selected from two learning rates and trained for fifteen minutes. Relative
error reductions here use the named reference model as denominator, rather than
the coarse baseline used above. Their intervals remain conditional on one seed.

- **Prediction helps density:** versus static training, topology error falls
  1.46% [0.86%, 1.93%], mobility error 4.95% [3.02%, 6.43%], and neighbor error
  1.09% [0.02%, 1.94%]. The result does not rely on longer confirmation training.
- **Prediction helps MACE's dynamics:** versus static MACE, mobility error falls
  1.48% [0.42%, 2.90%] and neighbor error 1.81% [1.26%, 2.66%]. Topology and
  order differences at this budget are not resolved. Confirmation improves
  MACE further, although equal-step convergence was not tested.
- **More data helps modestly:** full-data predictive density versus the 1/16
  center subset reduces topology/order/mobility error by 1.65/1.09/1.27%.
  The neighbor-error interval includes zero. Smaller-data runs develop markedly
  worse validation scores with continued training, consistent with overfitting.
- **Explicit motion gradients add no demonstrated benefit:** relative to the
  otherwise predictive density model, topology/order scores slightly worsen
  and mobility/neighbor differences are unresolved. Keep the simpler predictive
  objective as the reference.
- **The tested smoothness penalty is inconclusive:** it does not improve future
  assays. At its selected checkpoint its loss contribution is only about
  0.000079 versus roughly 23 total loss, so this is not a strong test of explicit
  smoothness regularization. Smaller perturbation scores alone do not establish
  an effect of the penalty.
- **Wider/larger MACE did not help at equal wall time:** the 6 Å model's neighbor
  error is 2.46% worse than 4 Å MACE; doubling channels makes it 2.01% worse.
  They processed fewer examples: approximately 704 and 2,084 examples/s,
  respectively, versus about 4,600 for standard predictive MACE. Larger capacity
  or support is not disproven; the available compute allocation did not pay off.

## Can we predict embeddings, and are they smooth?

The separate latent assay uses held-out Al pairs at 0.3, 1.2, 6 and 12 ps and
each model's frozen EMA teacher as target. Static/temporal controls simply
persist the current target embedding. The predictive model receives coordinates,
material, lag and the Al shooting temperature, not velocities or history.

| Representation | Error reduction vs persistence | Error reduction vs shuffled futures | Teacher effective rank / 128 |
|---|---:|---:|---:|
| Predictive density, 3 seeds | 36.80% | 30.06% | 18.35 |
| Predictive MACE, 3 seeds | 39.27% | 24.17% | 7.97 |
| Predictive SchNet, 3 seeds | 45.53% | 6.97% | 8.65 |
| Predictive GeoFrame, 1 seed | 49.49% | **0.14%** | 21.85 |
| Temporal-invariance density, 1 seed | Persistence control | 88.61% | **2.14** |
| Static density, 1 seed | Persistence control | 20.76% | 52.60 |

Density and MACE show useful conditional latent prediction. GeoFrame's apparent
49% improvement over persistence is misleading: correct futures are scarcely
better than condition-matched shuffled futures, and its current TDA/SOAP/order
probe scores are essentially zero. SchNet also has much weaker separation of
correct and shuffled futures than density or MACE. Since targets differ between
models, the persistence percentage is not a universal model-ranking metric.
These controls do not yet establish long-rollout accuracy or superiority to a
fully trained condition-only distributional predictor.

Temporal invariance strongly compresses the covariance spectrum. Its good
topology and retrieval results show that this is **not an uninformative constant
embedding**. It may encode a useful slow structural coordinate, but its two
effective dimensions and uncertain mobility gain are poor evidence for a rich
general-purpose predictive state. Effective rank describes covariance variance,
not the exact number of nonzero coordinates or recoverable signals.

The robustness assay applies rotations and coordinate noise to the same 192
validation environments, balanced across Al/Mg/Ta. Values below are the worst
per-material/per-seed p95 displacement, normalized by that material's embedding
spread in these probe samples.

| Encoder/objective | Rotation | 0.0001 Å noise | 0.02 Å noise |
|---|---:|---:|---:|
| Density, predictive | 0.000006 | 0.000897 | 0.171 |
| MACE, predictive | 0.000030 | 0.001484 | 0.277 |
| SchNet, predictive | 0.000660 | 0.005572 | 0.858 |
| GeoFrame, predictive | 0.000002 | **0.084858** | **1.619** |

MACE passes its rotation control. GeoFrame also passes this particular sampled
rotation control, but tiny coordinate perturbations still produce much larger
outliers. We should not claim that every old GeoFrame failure reproduces on this
small new sample. These are local perturbation tests, not a full static-Al
analysis or a densely sampled temporal jump/rollout audit.

## Improvement over the previous trained models

Using exactly the old **matched linear** assay and the same test samples:

| Model | Topology, previous → now | Mobility, previous → now | Neighbor agreement, previous → now |
|---|---:|---:|---:|
| Density | +7.62 → +6.91% | +9.60 → +14.55% | +4.79 → +4.71% |
| MACE | +1.24 → +3.43% | +5.54 → +9.21% | -0.35 → +3.58% |
| SchNet | +2.04 → +3.26% | +4.14 → +6.22% | +1.48 → +0.80% |

Paired source intervals support MACE improvements in all four measured future
assays. Density improves order and mobility, with no resolved topology or
neighbor improvement. Thus twelve hours did not improve every quantity.
Training data volume, objective and compute all changed; this historical
comparison cannot attribute gains to removing descriptor supervision alone.

## Next experiments suggested by these results

Use predictive density as the main baseline and retain MACE as a serious
alternative. Keep temporal-invariance density as a candidate slow-coordinate
model. A useful next hypothesis is to separate a compact slow structural state
from a richer state used for prediction, applying invariance only to the former.
This is a proposed experiment, not a demonstrated architecture improvement.

Before another broad sweep, test per-horizon latent skill against a trained
condition-only baseline and repeated rollouts; examine temporal jumps on dense
trajectories. Test stronger, explicitly measured smoothness regularization if
needed. Additional independent liquid source trajectories and shooting outcomes
are more informative than only resampling existing atoms. Ta still lacks an
independent validation trajectory, Mg/Ta source coordinates are float16, and
the independent predictive test in this campaign is Al only. Neither this table
nor TDA regression establishes discovery of new pre-crystalline motifs.

## Artifacts and reproduction

- [Campaign recipe and explicit analysis commands](../experiments/temporal_hypotheses_12h_20260906/README.md)
- [Full twelve-hypothesis results](../output/temporal_hypotheses_12h_20260906/RESULTS.md)
- [Equal-budget screening table](../output/temporal_hypotheses_12h_20260906/screen_analysis/comparison.csv)
- [Paired equal-budget intervals](../output/temporal_hypotheses_12h_20260906/equal_budget_pairs.csv)
- [Paired historical comparison](../output/temporal_hypotheses_12h_20260906/previous_training_pairs.csv)
- [Training audit](../output/temporal_hypotheses_12h_20260906/training_audit.csv) and [curves](../output/temporal_hypotheses_12h_20260906/training_curves.png)
- [Representation diagnostics](../output/temporal_hypotheses_12h_20260906/representation_summary.csv)

This document is a versioned research report. Shared analysis and orchestration
remain in `src/`; the dated experiment record documents the existing entry point.
Plots, CSVs, replay provenance, logs and archived partial results are generated
artifacts in the repository output directory. Large training caches and optimizer
checkpoints remain on IDS.
