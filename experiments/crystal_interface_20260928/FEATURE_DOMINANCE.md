# Does the interface model over-rely on strong features?

**It strongly relies on a crystal/order direction, but the measured reliance is
not specific to training sources.** The evidence supports a phase-dominated local
representation, rather than a few exploding channels or clear feature overfitting.
This still matters: strong current-phase cues can dominate a representation while
it remains weak at localization before an interface becomes observable.

This audit concerns the completed **CIV-MACE128 distance + direction + VCReg**
checkpoint, epoch 16, SHA prefix `8f384e83e602`. It does not describe the new dense,
interface-invisible batch-512 fit. The encoder and original predictor remain frozen
throughout these diagnostics. No new scientific training or W&B runs were created.

## The dominance is real, but spread across channels

Training-only PCA, applied unchanged to held-out sources:

| Export | Statistic | Train | Test |
|---|---|---:|---:|
| Local 128-dimensional embedding | Variance in first PC | 58.59% | 54.57% |
| Local embedding | Variance in first eight PCs | 90.41% | 89.81% |
| Local embedding | Largest individual channel's variance share | 0.93% | 0.97% |
| Context 128-dimensional state | Variance in first PC | 76.71% | 78.18% |
| Context state | Variance in first two PCs | 96.14% | 96.16% |
| Context state | Largest individual channel's variance share | 3.66% | 3.59% |

The leading local direction has absolute correlation **0.860 / 0.851** with
crystal membership on train/test, and **0.812 / 0.786** with the radius-8 Å,
degree-4 bond-order power. Its correlation with interface visibility is
**0.650 / 0.632**. These are label-side diagnostics, not newly supplied inputs.
The same physical association transfers to held-out sources.

This is redundancy across many channels, not one channel with an enormous scale.
The first two context PCs explain nearly all variance, but variance is not the
same as predictive information; the readouts below demonstrate that distinction.

## Does the model actually rely on that direction?

We encoded **4,320** uniformly sampled contexts: 2,880 train, 480 selection and
960 test, preserving every source in those roles. For all 25 patches, we altered
the scalar/vector inputs immediately before the **original frozen context
predictor**. The scalar PCA here is fitted on training patch inputs, distinct from
PCA of the final context state. The predictor was not refitted.

| Frozen predictor input | Train predictive loss ↓ | Test predictive loss ↓ |
|---|---:|---:|
| Original features | 3.907 | 3.651 |
| Clip scalar tails at three training standard deviations | 3.907 | 3.650 |
| Zero vector fields | 4.000 | 3.714 |
| Keep only leading scalar PC, retain vectors | 4.229 | 3.881 |
| Keep eight leading scalar PCs, retain vectors | 3.945 | 3.665 |
| Remove leading scalar PC, retain vectors | 8.168 | 8.165 |
| Replace scalars with training mean, retain vectors | 9.265 | 9.259 |

Loss is the original distance/direction/proximity predictive objective, excluding
VCReg. Removing the leading scalar direction increases test loss by **4.514**
(paired source-bootstrap 95% interval **4.013–5.023**). Keeping just eight scalar
PCs changes it by **+0.014** (**−0.015 to +0.041**). Thus most of this predictor's
useful scalar dependence is concentrated in a small subspace, and it is useful
on both train and test.

Clipping extreme values changes test loss by only **−0.0014**, with interval
**−0.0043 to +0.0009**. Zeroing vectors has a much smaller effect than removing
the leading scalar direction: **+0.063**, interval **−0.016 to +0.141**. That does
not prove vectors are universally unnecessary; this is one trained model and one
localization task. Feature interventions can leave the training distribution, so
their damage measures reliance, not whether the removed feature is spurious.

## Do weaker features retain useful information?

Separate logistic readouts were fitted on training embeddings only, using the
same declared regularization and training-standardized PC coordinates. These
measure accessible information; they are not the original predictor. Target is
**interface distance ≤20 Å**, a present-distance event, not a future time horizon.

| Export and readout input | Train log loss ↓ | Test log loss ↓ |
|---|---:|---:|
| Local: all 128 PCs | 0.3797 | 0.3864 |
| Local: leading PC only | 0.4020 | 0.4060 |
| Local: remaining 127 PCs, leading PC removed | 0.6514 | 0.6389 |
| Context: all 128 PCs | 0.1318 | 0.1291 |
| Context: leading two PCs only | 0.2484 | 0.2488 |
| Context: leading eight PCs only | 0.1417 | 0.1388 |
| Context: remaining 127 PCs, leading PC removed | 0.5475 | 0.5332 |
| Training-prevalence constant | 0.6744 | 0.6657 |

The local leading PC captures about **93%** of the all-PC readout's test log-loss
improvement over the constant baseline. For the context state, however, two PCs
holding 96% of variance are insufficient: lower-variance directions substantially
improve prediction. Discarding them just to make the embedding look lower-rank
would lose useful information.

The all-PC local readout has a train/test gap of only **0.0068**; the context readout
is slightly better on test. Source populations differ, so these gaps are not a
proof of no overfit. They nevertheless offer no strong evidence that the dominant
features work on train and fail on held-out sources.

Native-channel-standardized probes are also exported. Their regularization geometry
differs from standardized-PC probes, so PC subset comparisons above use the all-PC
reference rather than interpreting a basis change as information gain or loss.

## What remains concerning

The model's good global behavior does not establish useful information in the
hard interface-invisible population. In the complete saved test population,
P(distance ≤32 Å) averages **0.1931** for invisible contexts, against observed
prevalence **0.0899**. The earlier [audit](OVERFITTING.md) also found poor ≤20 Å
calibration before visibility. This is consistent with excessive dependence on
the easier phase/visibility signal, but it does not isolate a causal shortcut.

In the smaller frozen-head intervention sample, invisible test loss improves by
**0.081** when keeping four scalar PCs (interval **0.029–0.129 improvement**).
Treat that as exploratory: this sample has only **535** invisible test rows,
**zero** ≤20 Å positives and **54** ≤32 Å positives. It cannot establish improved
early detection at 20 Å. Full saved-feature probes use all **42,160** original
invisible test rows; no fixed benchmark rows are dropped.

VCReg regularizes the **patch exports**, not the final context state. At the last
logged training update, its weighted contribution was **0.00877**, versus predictive
loss **4.16097**; the scalar covariance penalty before its 0.01 multiplier was
**0.3467**. Its variance floor is largely satisfied, but correlated phase directions
remain. Loss magnitudes alone do not measure gradient influence, and simply
increasing VCReg is not established as a remedy by this audit.

The new interface-invisible training run is a relevant next check: whether useful
distance information improves within that restricted population while local
features retain more than a phase indicator. Keep proper likelihood/calibration
and information readouts as the criteria; rank or feature importance alone should
not decide which model is better. A subsequent controlled feature-dropout or
stronger-decorrelation experiment would need to demonstrate that benefit before
being adopted. This audit does not change the active training recipe.

### Follow-up: the intended liquid-only distance task

The user clarified that the intended signal is changes in liquid structure related
to distance from a crystal outside the observed input. Interface absence alone
does not establish that population: crystal interiors can have no visible boundary.
The queued adaptation currently restricts interface visibility, not all confirmed
crystal visibility, so it does not fully isolate this research question.

A direct check on the completed model conditions the original test population on
all three saved fields: `~inside_crystal`, `~crystal_visible_context`, and
`interface_exists`. This retains **12,923 rows from 28 test sources**: liquid query,
no confirmed crystal in any input patch, and an interface elsewhere in the cell.
For every finite target in these rows, the saved interface distance equals the
saved nearest-crystal distance exactly. The score is capped at 64 Å, consistently
with the existing point-error evaluation.

| Predictor on this restricted population | Test distance RMSE, Å ↓ |
|---|---:|
| Completed joint encoder/predictor | 15.227 |
| Constant training-population mean distance | 10.711 |

The constant **40.3468 Å** is fitted only on the same restricted training population.
For each role, start with half fixed/half uniform and equal-source weights within
each population; then condition and renormalize. Thus this comparison does not use
test targets to fit the baseline. The model-minus-constant test MSE difference is
**+117.15 Å²**, paired whole-source bootstrap 95% interval **+86.68 to +155.53 Å²**
(1,000 resamples, seed 20260928, 28 eligible test sources).

This is concrete evidence that the completed model has not learned a useful
conditional-mean distance estimate for this population. It does not demonstrate
that the input liquid structure lacks information, or rule out useful ranking or
a better probabilistic model. A snapshot also provides structural signatures rather
than directly observed temporal transformations.

For comparison, the broader no-crystal-visible population has 42,128 test rows and
67.30% censored mass. Including cells without an interface substantially changes
the problem and can hide weakness in localization when a crystal actually exists.
Report crystal absence separately from distance to an existing external crystal.

## Scope and reproducibility

All PCA bases, scales and readout coefficients use training rows only. Scores give
equal source mass within each fixed/uniform population and half mass to each
population. Source-bootstrap intervals use 1,000 paired resamples of 30 test
sources; they exclude seed and within-source sampling uncertainty and do not
correct for exploring several interventions. One encoder seed was inspected.

Rebatched BF16 frozen-head inference matched saved probabilities to mean absolute
error **0.000466**, maximum **0.00957**. This is relevant when interpreting tiny
effects such as scalar-tail clipping. Definitions, hashes, inputs and raw
predictions are preserved in the [analysis bundle](/work/PERSO/vmorozov/analysis/crystal_interface/al64-random-20260928/distance_direction_vcreg/analyses/feature-dominance-v3/).
Revision v1 had invalid saturated-probability log scores; v2 corrected float64
scoring, and v3 added the matched all-PC readout reference. The reported results
are v3; unchanged frozen-head results were imported from v2 with provenance.

- [Concentration and readout figure](/work/PERSO/vmorozov/analysis/crystal_interface/al64-random-20260928/distance_direction_vcreg/analyses/feature-dominance-v3/plots/feature-concentration-and-readouts.png)
- [Frozen-predictor intervention figure](/work/PERSO/vmorozov/analysis/crystal_interface/al64-random-20260928/distance_direction_vcreg/analyses/feature-dominance-v3/plots/frozen-predictor-interventions.png)
- [Readout scores](/work/PERSO/vmorozov/analysis/crystal_interface/al64-random-20260928/distance_direction_vcreg/analyses/feature-dominance-v3/tables/probe-scores.csv)
- [Cue associations](/work/PERSO/vmorozov/analysis/crystal_interface/al64-random-20260928/distance_direction_vcreg/analyses/feature-dominance-v3/tables/cue-associations.csv)
- [Frozen-head scores](/work/PERSO/vmorozov/analysis/crystal_interface/al64-random-20260928/distance_direction_vcreg/analyses/feature-dominance-v3/tables/frozen-head-scores.csv)
- [Metric definitions](../../docs/metrics/crystal_feature_dominance.md) · [Execution](../../docs/crystal_interface_unseen.md#feature-dominance-audit).
