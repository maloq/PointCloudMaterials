# Latest MACE quality results — 26 September 2026

All eight checkpoint evaluations completed without a failed stage, in about
16 minutes on one RTX6000 PRO. Completion hashes and all five geometric checks
were verified for every model. Outputs contain 24 material/model plot panels,
24 structural summary rows and 72 predictive-readout rows. See the
[protocol](README.md), [definitions](../../docs/metrics/encoder_quality.md) and
[output location](../../docs/encoder_quality.md).

## Main comparison

Epi+variance preserves more liquid structure than the other current recipes,
but no model improves every scientific criterion. Spatial boundary discrimination
remains weak, and physical descriptors still add predictive information to
several embedding-only readouts.

The first three metric columns below average three relaxed Al snapshots.
Prediction uses the exact same 45291 Al64 test windows and 30 test sources,
with a fresh frozen-embedding MLP and separate calibration role. Observed and
relaxed prediction inputs are different observation contracts. Lower liquid
error/log loss is better; higher classification accuracy/boundary AUROC is better.

| Encoder | Prediction input | Liquid-neighbor NMSE | Nonbulk balanced accuracy | Boundary AUROC | 6 ps calibrated log loss |
| --- | --- | ---: | ---: | ---: | ---: |
| Scratch, supervised | Observed | 1.498 | 0.624 | 0.563 | 0.07010 |
| VICReg → supervised | Observed | 1.463 | 0.643 | 0.552 | 0.06991 |
| Epi+variance → supervised | Observed | 1.417 | 0.650 | 0.549 | 0.06916 |
| Scratch, supervised | Relaxed | 1.496 | 0.634 | 0.567 | 0.05530 |
| VICReg → supervised | Relaxed | 1.507 | 0.641 | 0.552 | 0.05557 |
| Epi+variance → supervised | Relaxed | 1.451 | 0.642 | 0.545 | 0.05536 |
| VICReg, pretrained epoch12 | Observed | 1.486 | 0.645 | 0.543 | 0.06989 |
| Epi+variance, pretrained epoch12 | Observed | 1.404 | 0.651 | 0.538 | 0.07073 |

## What the checks establish

1. **Liquid information improves modestly.** Among the observed-input supervised
   encoders, Epi reduces liquid-neighbor error versus scratch by 5.4% in Al,
   3.3% in Ta and 5.6% in Zr. Its Al nonbulk balanced accuracy increases from
   62.4% to65.0%. Epi's pretrained export has still lower neighbor errors:
   Al1.404, Ta1.161, Zr1.441. These are exploratory within-snapshot measurements,
   not independent dynamic material-transfer demonstrations.

2. **Interfaces are partly accessible, but spatial organization is unresolved.**
   The Al nonbulk classifier's mean solid/liquid-boundary recall rises from
   67.0% for scratch to71.6% for observed-input Epi; ordered-liquid candidate
   recall rises from50.0% to52.8%. Planar-fault recall does not improve uniformly;
   the first frame has no supported fitting examples for that class. Epi's
   distance-matched boundary AUROC is lower than scratch across all three
   materials: Al0.549 versus0.563, Ta0.598 versus0.624, Zr0.533 versus0.542.
   Shuffled controls are near0.5. Thus a supervised classifier extracting a
   distinction is different from that distinction organizing latent distances.
   K=7 colors do not constitute independently validated interface species.

3. **Fine-tuning produces a trade-off, not wholesale erasure.** For Epi,
   supervised observed-input fine-tuning worsens liquid-neighbor error by
   about0.9–1.5% across Al/Ta/Zr and lowers the mean source correlation between
   latent and physical change from0.612 to0.485. Its boundary AUROC increases,
   and6 ps log loss improves from0.07073 to0.06916. The paired improvement is
   −0.00157, 95% source-bootstrap interval[−0.00219, −0.00095]; the corresponding
   Brier interval includes zero. This compares a label-free pretrained endpoint
   with subsequent onset-supervised fine-tuning, not epochs of an unchanged
   VICReg objective. The temporal statistic is an association, not a
   displacement-matched rearrangement test.

4. **Relaxed observations produce the largest predictive improvement here.**
   Scratch6 ps log loss changes from0.07010 observed to0.05530 relaxed, a21.1%
   reduction. Its paired difference is−0.01480 [−0.01909, −0.01089]. Both Brier
   and log loss improve at3/6 ps. Full-cell relaxation changes the observation;
   this is not evidence of an architectural gain or a pure local denoising
   intervention. On relaxed inputs, Epi does not establish a log-loss advantage
   over scratch and has worse calibrated Brier at3/6 ps; at6 ps the Brier
   difference is+0.000418 [0.000109,0.000721].

5. **Descriptor add-back exposes a remaining predictive limitation.** For
   observed-input Epi, adding32 current-geometry descriptors to z reduces
   calibrated6 ps log loss from0.06916 to0.06743, paired difference−0.00173
   [−0.00249,−0.00106]. Brier also improves. For relaxed Epi, log loss changes
   from0.05536 to0.05307 and Brier from0.01464 to0.01381. These results show
   useful information beyond what the tested z-only readout accesses. They do
   not distinguish information absent from z from nonlinear accessibility,
   finite optimization, or the larger joint-input predictor.

6. **Latent distances do organize some future information.** Observed-input
   Epi's fitting-neighbor forecast has calibrated6 ps log loss0.07127 versus
   0.08411 for descriptor-distance neighbors, with paired interval for the
   difference[−0.01534,−0.01043]. The raw scores agree in direction. This is
   evidence beyond the bulk liquid/crystal UMAP picture. Neighbor count was
   selected by validation likelihood, never AP.

## Uncertainty and interpretation

The observed-input Epi-versus-scratch3 ps calibrated log-loss difference is
−0.00120 [−0.00209,−0.00047], while its6 ps interval crosses zero. One fitted
encoder seed per recipe, reused test sources and many descriptive contrasts
limit causal and confirmatory claims. Source-bootstrap intervals condition on
these trained models; they do not include training-seed uncertainty or adjust
for multiple comparisons. Static frames do not supply independent-root error
bars. Higher latent rank or lower noise response alone is not success.

Linear readouts have poor raw calibration under this fixed24-pass recipe;
their shared3/6 ps calibrator can also degrade12 ps extrapolation. Do not equate
their failure with absence of information in the encoder. The table above uses
the declared MLP readout, and all linear results remain exported.

These labels predominantly measure **arrival from existing crystal**, not
nucleus birth. Static Ta/Zr ordered-liquid proxies cannot establish a precursor's
future. Dense relaxed dynamics were explicitly unavailable. Keep Epi-pretrained,
fine-tuned Epi and scratch as complementary controls in subsequent mechanism
experiments; there is no evidence for one universal winning recipe.

Machine evidence is retained under the configured run output:
`technical/evaluations/*/{static-metrics,predictive-metrics,temporal-metrics,geometry-checks}.json`,
paired cross-model contrasts in `technical/review-paired.json`, and the
original metric-contract snapshots. Cross-model predictions were checked for
identical sample IDs, sources, roles and event labels before pairing.
