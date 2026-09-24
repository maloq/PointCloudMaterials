# How encoder performance was measured

[Handbook](README.md) · [Metric contracts](../metrics/) · [Results](results.md)

There is no single encoder accuracy. Every comparison needs an **observation
contract**, an exported feature identity, a target, a readout, a population and a
split. The exact exported `tables/METRICS.md` takes precedence over this cross-study
explanation, especially for historical results.

## What is being evaluated?

| Question | Our measurements | What a positive result does not establish |
| --- | --- | --- |
| Does inference respect the intended physics/computation? | Rotation, permutation, translation/boost and velocity-parity controls; periodic wrapping; atom-ID alignment; batch/reorder replay; native-checkpoint equivalence | Better physical information or forecasting |
| Is the representation continuous? | Matched-coordinate perturbations, interpolated paths, frame/membership interventions and cutoff crossings | Low natural-time change, or useful response to real rearrangements |
| Does it retain current information? | Actual training heads; frozen ridge and nonlinear probes of radial/angular/order/density/TDA/motion targets | A closed dynamical state or well-organized latent distances |
| Does distance organize meaningful states? | Physical-neighbor retrieval, conditional add-backs, matched-temperature/order neighborhoods, spatial coherence | Distinct thermodynamic phases or causal mechanisms |
| Does it represent dynamics usefully? | Declared-lag normalized jumps, within-source rank, change readouts, physical future prediction, history add-backs | A Markov state from a nonsignificant add-back result |
| Does it forecast crystallization? | Sustained local-onset hazards, AP, likelihood/Brier, calibration, alarms, timing plus misses | Global nucleation, a committor, or current-state recognition |
| Is it cheaper to train/use? | Same-workload warmed timings, end-to-end throughput and memory | Better scientific results at matched updates or wall time |

## Observations, targets and splits

Record positions versus positions+velocities; number/timing of historical frames;
tracked atom IDs; cutoff/taper/halo; potential/material/temperature; observed
versus relaxed geometry; coordinate precision; and whether relaxation used the
full cell. An80-neighbor normalized GeoFrame crop, a physical17Å partial
observation and a relaxed8Å MACE patch are different inputs. Casting float16
coordinates to float32 does not recover lost precision.

Instantaneous TDA, relaxed TDA, radial/angular moments, original-MD order and a
learned future embedding are different targets. The target radius need not equal
the encoder radius. A future geometry oracle can establish label observability;
it cannot be counted as a causal forecast.

Split by the highest independent **source/root**, including all its shooting
branches, overlapping windows and relaxations. Random windows from the same
trajectory do not form an independent test. Earlier GeoFrame Al/Mg/Ta temporal
validation separated times and center IDs within trajectories; Ta had one source.
Later native studies used150 sources with90/30/30 roles. The repaired structural
screen uses45 roots with25/5/15 roles. Those15 are explicitly **reused development**,
not a new final test. Separate selection and probability calibration when the
protocol supplies distinct splits; the small structural screen reuses its five
tuning roots for both and records that limitation.

Fitting-only normalization is essential. Do not estimate feature scales, PCA,
regression residuals, target variances, clipping bounds or prototype centers using
test observations. Fix the reference population when comparing checkpoints.

## Present reconstruction and frozen probes

For a standardized target coordinate, error is `(prediction-target)^2` in the
producer's fitted units. But aggregation varies:

- The relaxed-TDA comparison balances **three blocks**, H0/H1/H2, despite their
  16/64/64 coordinates. Its balanced MSE is not raw144-coordinate MSE.
- The older causal encoder balances six physical blocks. Predictive memory
  averages coordinates and lags of a different packet. Equal-looking numbers
  cannot be compared between them.
- Structural-state physical probes average coordinate errors within each example,
  examples within each root, then roots equally. The native geometry loss balances
  radial17, l2-Gram36 and l4-Gram36 blocks.
- RMSE is the square root of the named MSE, not an interchangeable percent-change
  measure. A31.5% RMSE increase implies a different MSE increase.

A **training-head score** measures the encoder and the decoder trained together.
A **fresh ridge probe** tests linearly accessible information after freezing the
encoder. A stronger nonlinear probe tests whether a linear arrangement is the
limitation. Keep capacity, tuning budget, conditions and selection rules matched.
Appending temperature or current descriptors changes the question from direct
prediction to incremental information beyond those inputs.

Report the native head and frozen probes together. Fresh rescaling can hide tiny
raw feature amplitudes and a highly sensitive decoder; the structural-state v1
failure demonstrated this. Compare initial/final encoder weights, a training-mean
predictor and actual head norms/spread. Step0 selection can be legitimate, but it
must be labelled as a retained baseline rather than successful learning.

Global R² uses variation around the evaluation-population mean. **Within-frame**
R² removes each source/frame mean in the denominator and tests local differences.
Negative within-frame R² can coexist with good global MSE because predicting
material, temperature or mean phase explains much of the aggregate variance.
Always report liquid/noncrystalline subsets with the exact label rule: `PTM Other`
sometimes means code0, and sometimes the complement of types1/2/3. These are not
identical populations and neither is a proof of equilibrium liquid membership.

Contracts: [topology/static analysis](../metrics/analysis.md),
[causal MACE](../metrics/mace_causal.md), [structural state](../metrics/structural_state.md).

## Smoothness, dimensionality and neighbors

The [September 24 dynamics supplement](embedding_dynamics.md) adds complete
dataset, within-track, uncentered temporal-movement and centered-fluctuation
spectra. It reports participation/entropy ranks and dimensions retaining
90/95/99% energy, with sample ceilings and exact physical-lag matching. A
constant-velocity line has movement rank one but zero fluctuation variance.
These are linear spectral dimensions; they are not nonlinear intrinsic manifold
dimensions. [Exact definitions](../metrics/embedding_dynamics.md).

The causal normalized RMS jump uses a declared physical lag and covariance
reference, schematically

\[
J(\Delta)=\sqrt{\frac{\mathbb E\|z_{t+\Delta}-z_t\|^2}
 {2\,\operatorname{tr}\operatorname{Cov}(z_t)}}.
\]

The producer's source-wise reduction and covariance population matter. A ratio
of global averages need not equal an average of source ratios. Other legacy
“relative drift” and “change/variance” scores use different normalizations.
Do not apply the0.10 criterion to an unrelated metric. Under stationary moments,
this particular normalized form corresponds to `J²=1-correlation`; real
transitions can require genuine motion in an informative state.

Inspect full jump distributions, p95/outliers, raw spread and transition response,
not only a mean. Adjacent history windows overlap; apparent smoothness can come
from shared inputs. Forecast horizons beyond the history span test new evolution.
Averaging, bent-path penalties and low-dimensional motion constraints can suppress
real signal as well as noise.

Participation rank `(sum eigenvalues)^2/sum(eigenvalues^2)` and entropy effective
rank `exp(-sum p log p)` are distinct. Raw covariance rank and correlation-matrix
rank also differ. Retain the producer's formula, precision, centering and cohort;
do not rename them all “rank.” A higher rank can spread nuisance variation.

Our recent structural-state retrieval uses five fitting neighbors, matched on
temperature and coarse current phase, with raw exported Euclidean distance.
Targets are compared on **query-neighbor pairs**, not to the average neighbor
prediction. Other studies use31 neighbors or density/coarse-order matching.
Standardization, whitening and physical residual metrics are controlled changes
to distance, not new encoders. The liquid-geometry study keeps frozen regression
probes unchanged across these distance interventions.

## Fixed future physical outcomes and state-use tests

A physical forecast MSE is comparable across encoders only if targets, scales,
horizons, rows and aggregation match. Compare persistence, conditions-only,
current physical packet, history packet, trained repeated-frame control and
initial encoder when applicable. Prediction of `Y(t+τ)-Y(t)` is different from
prediction of the endpoint. A decoded physical score is more portable than raw
latent MSE because each encoder defines its own target space.

The shooting assays predict conditional means or future-law features across
branches. Repeated futures reduce ambiguity about the conditional distribution;
an individual realized future remains noisy. Coarse-augmented ridge and a direct
nonlinear predictor of `z+temperature` answer different questions even on the
same trajectory cohort.

Predictive memory scores a four-component continuous future-path density. NLL is
normalized per coordinate/lag; it can be negative for some continuous densities.
Hazard event NLL is a different likelihood. A mixture's mean MSE and density NLL
can disagree; neither alone proves useful uncertainty estimates. Inspect coverage,
state shuffling/mean-code interventions and simple fixed-packet readouts.

A history add-back compares a predictor of `Yfuture|z` to a matched predictor
with access to the original observed history as well. Improvement exposes
predictive information absent or inaccessible in `z`. No improvement can also
reflect limited diagnostic optimization or power; it is not proof of sufficiency.
The distance/future factorial's residual label removes only its declared linear
present baseline, not all nonlinear present predictability.

## Crystallization: recognition, risk and timing

Current-state classification is easier and scientifically different from onset
prediction. In recent assays, local onset is the tracked atom's **first** PTM1/2/3
state sustained for three saved frames. Eligible origins precede onset and meet
the declared recent noncrystalline condition. Future confirmation frames can define
the label but must never enter encoder inputs. Preserve enough follow-up for
confirmation at the horizon boundary.

Discrete hazards give `h_k=P(event in bin k | survived earlier bins, z, conditions)`.
Survival through K bins is `product(1-h_k)`. Event NLL includes earlier survival
and the event-bin hazard; right-censored examples contribute observed survival.
Bin ends differ across protocols. Recent structural screens use0.75/3/6/9/12ps;
older local-predictability studies used a longer six-horizon grid.

| Score | Interpretation and required companion |
| --- | --- |
| Average precision (AP) | Event ranking across thresholds; compare with the same population's source-weighted prevalence. Report positives and source count. |
| AUROC | Ranking of positives against negatives; can look favorable with rare events while precision is weak. |
| Brier | Mean squared probability error; report constant-risk baseline and prevalence. |
| Binary log loss | Horizon-specific probability error; differs from joint event-time NLL. |
| Joint event-time NLL | Entire hazard distribution, including survival/censoring; keep bins and follow-up fixed. |
| Recall/precision/FPR | Operating point at a threshold selected without test labels. A tuning5% FPR constraint does not guarantee5% FPR on test. |
| Timing MAE among detections | Conditional timing error; always report misses and all-positive timely-detection rate too. |
| Fixed-lead recall | Warning at a specified lead, not merely onset sometime inside a horizon. Positive-only AP is uninformative. |

With source-equal weighting, each row carries inverse source row count. Aggregate
AP must be recomputed from weighted predictions; it is **not** an average of
per-source AP. Include complete tied score groups. No-positive AP/recall and
no-detection timing are undefined, not zero. Oversampled transitions require
correction when estimating natural-population risks.

The repaired screen's18 positive12ps windows represent correlated origins from10
sources, not18 independent nucleation events. The larger relaxed assay's338
local atom onsets from27 event-bearing sources are also correlated. Never compare
AP0.2675 on the small development screen directly with AP0.2529 on the larger
fixed-grid test to infer model improvement.

Contracts: [local onset](../metrics/local_predictability_native_onset.md),
[relaxed large test](../metrics/relaxed_encoder_large_test.md),
[repaired onset review](../metrics/structural_state_onset_review.md),
[distance/future factorial](../metrics/structural_state_future.md).

## Uncertainty and fair comparisons

Use paired whole-source resampling with the producer's temperature strata and
sampling multiplicities. Average errors within roots; recompute weighted AP for
each draw. Bootstrap source intervals condition on the fitted models. They do
not measure training-seed uncertainty, model-search uncertainty or unseen domain
shift. Record invalid no-positive draws and the procedure used to exclude them.

Distinguish absolute score differences, relative error reductions and relative
error increases: `candidate-reference`, `1-candidate/reference`, and
`candidate/reference-1` have different signs/scales. Standard deviation across
seeds is not a95% confidence interval. Reused test sources, post-hoc comparisons
and unadjusted multiple testing should remain visible beside the numbers.

Before comparing: match observation/target domains, rows/ancestry, normalizers,
readout capacity/conditions, training budget, checkpoint rule, horizon, population
and metric implementation. If one differs, label the comparison as external or
confounded. More epochs, wider models, more sampled centers and different GPUs
are not automatically independent tests of an architectural hypothesis.

## Material-specific interface and liquid references

The [GeoFrame evolution contract](../metrics/geoframe_evolution.md) adds full-cell
PTM/RMSD sensitivity (0.08/0.10/0.12), Al planar faults/twins, equal-weight q4/q6,
normalized w4/w6, averaged q6 and bond coherence. Its mixed-neighborhood class is
an operational boundary proxy, including grain boundaries and isolated motifs;
it is not a unique solid–liquid surface. HCP Zr is not an Al stacking fault.
Ordered/five-fold liquid categories are descriptor proxies. Reference thresholds
are fitted only on the fitting region, separately by material. These descriptors
are not Hu–Tanaka's Voronoi-weighted quantities.

Report class counts, unsupported classes and spatially held-out per-class AP,
within-liquid continuous-order/topology R² and density-conditional gains. Inspect
K=7 cluster/context contingencies and nonbulk adjusted mutual information in the
native representation; a probe can succeed while clustering mixes liquid types.
Our proposed boundary score asks whether embedding distance distinguishes a
reference-label change across a nearest-neighbor bond, conditioned on physical
bond-length bins. Shuffled and collapsed controls are required. It is a diagnostic
of the declared proxy, not a universal score for meaningful spatial structure.

Encoder and projector must have separate rows. UMAP/t-SNE remain exploratory:
refitted maps cannot measure latent motion. Evaluate perturbations, rank and
physical change response together; a constant state is smooth. Compare prediction
only with matching present-physics conditions, source splits and selection rules,
including selected-step0 flags, Brier/NLL and paired-root uncertainty.
