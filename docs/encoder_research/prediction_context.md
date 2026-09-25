# Prediction inputs and context ledger

[Handbook](README.md) · [275 comparison rows](../../output/encoder_research/prediction-context-20260925/tables/prediction-context.csv)
· [Evidence and checksums](../../output/encoder_research/prediction-context-20260925/technical/contexts.json)
· [MACE sizes](mace_sizes.md)

## Rule from25 September2026

Do not feed temperature, simulation age, absolute time or explicit time
covariates to an encoder, forecasting head, frozen probe or baseline unless the
user explicitly requests them again. This includes one-hot temperatures and
polynomials/normalizations of age. Keep timestamps for ordering observations,
splits and labels; keep temperature for dataset audits and stratified reporting.
Neither is a model input. Fixed output bins at0.75/3/6/9/12 ps are labels, not
time features. The `ap_temperature` hyperparameter is a ranking-loss softness,
not the material's temperature or an input covariate.

The live supervised protocol is now `supervised_onset_capacity_v3`.
Its head accepts only `logits(z)`. Its fresh linear/MLP probes use only the
exported representation; descriptor baselines use only descriptors. The old
conditions-only baseline becomes a learned constant-risk baseline. There is
no concatenation of the cached seven condition columns. Tests poison metadata
with NaNs/changed ages and verify identical probe predictions. The old conditioned
recipe is rejected by the live runner; its frozen source remains available for
historical reproduction. The subsequent [capacity campaign](../supervised_capacity.md)
uses this policy with a fixed128-D projected export at every tested width.

This policy is also recorded in [AGENTS.md](../../AGENTS.md). Other historical
training families retain their original code and input definitions; do not reuse
them for new training without applying this policy and writing a new contract.

## What the completed AP results actually used

| Comparison | Encoder observation | Onset predictor receives | History / velocity |
| --- | --- | --- | --- |
| Larger supervised AP3/AP6, six arms | Observed, relaxed, or paired current patches; at most80 atoms including center;8 A crop;2 MACE layers with5 A edges | State + five temperature indicators + age/600 ps + its square | None / none |
| Its descriptor controls | No learned encoder;237 observed,237 relaxed or474 paired descriptors | Descriptors + the same seven conditions | None / none |
| Robust-onset eight-arm screen | Observed or relaxed current geometry; variable atom count in8 A | State + five temperature indicators | None / none |
| Spatial hierarchy four-arm screen | Current focal8 A geometry plus smooth regional summaries out to8,12 or16 A; early/late fusion varies by arm | State + five temperature indicators | None / none |
| Geoformer, native-screen, parameter-search, MACE/Epi **onset probes** | Relaxed current geometry; native radius or nearest80, recorded separately for each row | Representation + five temperature indicators +8 current-order features +89 relaxed-geometry descriptors | None / none |

The275-row ledger joins by `(collection, model, readout)`:246 comparisons from
the horizon review and29 from the larger supervised study. All275 used temperature;
29 used simulation age;210 supplied97 additional physical descriptors alongside
the representation. For example, a128-dimensional encoder in the last row had
a230-input linear onset predictor, verified against saved weights. This is a
**conditional prediction assay**, not a measurement of prediction from z alone.
The9 ps future-residual regression in that same evaluation used z alone, so its
input contract must not be inferred from the onset probe's contract.

The larger supervised study's seven columns were checked against all31,609 cached
rows, not just against comments or configuration. The frozen producer derives
age from the original MD frame index and0.75 ps cadence. Its `conditions` control
therefore measured temperature **and age**. The accompanying
[erratum](../../output/encoder_supervised/ap36-large-20260924/tables/CONTEXT_ERRATUM_20260925.md)
records this correction without rewriting historical scores.

Spatial support also needs qualification. The larger cohort has80 observed
atoms per patch and70–80 relaxed atoms after the8 A crop. Median outer radii
are7.03 A and7.46 A respectively. There is no added message-passing halo.
Two5 A layers cannot access atoms absent from this input graph. Relaxation was
performed on the full periodic current cell before extracting a patch; relaxed
inputs can therefore depend indirectly on surrounding atoms. A relaxed teacher
is training-only for the observed-input distilled arm. Paired observations are
two views of the same time and location, not a longer history or a larger radius.

These tables describe **prediction-time inputs**. Earlier pretraining may have
used different query heads, targets and conditions; a null
`training_predictor_context` means that training-stage contract was not audited
here. For instance MACE/Epi's neighborhood training is distinct from its frozen
onset readout. The ledger is not a complete audit of all archived studies or
remote H200 artifacts supplied only as prose summaries. Unverified inputs must
remain unknown rather than being reported as absent.

## Contract for every new experiment

Write `technical/prediction-context.json` before training, alongside the immutable
configuration/source identity. Record:

1. Encoder inputs: geometry/species/center flags, observed versus relaxed views,
   atom cap, radial support, edge cutoff, number of layers and computational halo.
2. Spatial context: regional radii, token or atom representation, fusion point,
   whether the environment enters through full-cell preprocessing.
3. Temporal inputs: frame count, historical duration, velocity/displacement inputs,
   causal selection and any explicitly requested time features.
4. Each training and evaluation predictor: embedding dimensions, every physical
   side-input block, external conditions and their exact transformations. Keep
   encoder, predictor, probe and baseline contracts separate.
5. Training-only teachers, paired views and auxiliary targets; none should be
   described as a prediction-time observation unless inference uses them.
6. Evidence: data producer, configuration, code/checkpoint hashes and actual input
   widths. Record unknown values explicitly; never trust a name such as
   `conditions` to specify its contents.

The new supervised runner creates this record during binding and refuses a
changed context in the same output directory. Each result JSON/CSV links to it;
readout checkpoints record input width and `external_inputs=[]`. Geometry still
contains physical clues correlated with temperature or age; the policy removes
explicit metadata inputs, not those physical correlations.

Rebuild the dated historical ledger without fitting anything:

```bash
conda run -n pointnet-torch214 python -m src.research.prediction_context_audit \
  --config configs/analysis/prediction_context_20260925.json
```

Its AP values are copied unchanged. Check the original metric definitions and
cohorts before comparing rows; adding a context log does not make the older small
development cohort comparable to the larger reused test cohort.
