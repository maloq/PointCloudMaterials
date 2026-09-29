# What would establish a useful atomic-environment encoder?

Evidence and scientific checks, 26 September 2026. This is a research decision
framework. Its implemented subset is now evaluated in the
[latest native MACE comparison](../../experiments/encoder_quality_20260926/README.md):
static structure, geometric/noise controls, temporal descriptor-change association,
frozen likelihood readouts, descriptor add-back, and latent-neighbor forecasts.
The stronger displacement-matched and regional-birth checks below remain proposals.
Existing historical results and exported metric definitions remain unchanged.

## Working definition

A useful encoder preserves distinctions between local structures and their
possible evolution while responding appropriately to irrelevant changes of
coordinates. Nearby embeddings should describe similar environments and, to the
extent the observation contains predictive information, similar future outcome
distributions. Smoothness, physical fidelity, latent-distance organization and
prediction require separate evidence. A representation can pass one and fail
another. A continuous ordering process need not form isolated clusters.

The claim is relative to the observation contract: current geometry, surrounding
support, history and relaxation are different observations. Invariant scalar
exports must respect rotations; typed directional exports must transform
equivariantly. Neither should lose relative orientations needed by a context
predictor. Geometry-only inputs cannot uniquely determine all stochastic or
velocity-dependent futures.

## What our completed comparisons support

- In the matched 24-epoch MACE study, final Epi liquid-neighbor error is 1.410
  versus VICReg 1.679; within-liquid order R² is 0.409 versus 0.278. Epi improves
  nonbulk interface/ordered-liquid readouts but not every fault class.
- Epi largely retains initial nonbulk spatial coherence rather than improving
  it. VICReg loses much of that signal. Neither higher rank nor lower training
  loss establishes appropriate spatial organization.
- Epi plus variance has better predictive proper-score point estimates than
  the other two objectives, but none establishes added calibrated benefit over
  the historical current-physics baseline. Those forecasts included explicit
  conditions; they are not a condition-free comparison.
- Matched Geoformer VISReg reduces raw-encoder liquid-neighbor error by about
  14% in both seeds. Encoder and projector can move in opposite directions.
  FactorVAE removal has not shown consistent improvement.

These are conditional findings for the recorded recipes and populations, not
proof of a universally best loss. See the [verified results and curves](../../output/encoder_research/mace-vicreg-epi-20260923/RESULTS-20260926.md).

One training hypothesis deserves a direct intervention: temporal alignment may
suppress real structural evolution along with unwanted fluctuations. The
completed objective comparison does not isolate that mechanism. Compare
alignment strengths with identical paired observations; separately compare
same-frame symmetry views against different-time views. A predictor of the
future state is another distinct treatment, not equivalent to forcing the two
states to coincide. Relaxed/observed pairing is also a distinct physical view,
not an exact symmetry. Keep covariance/variance treatment fixed in each pairing
comparison, then test the regularizer interaction if supported.

[VICReg](https://arxiv.org/abs/2105.04906) explicitly addresses collapse through
variance and covariance constraints. The additional question here is which
physical distinctions the paired-view objective retains. Avoiding a constant
representation does not answer that question.

## Scientific checks and interpretations

| Requirement | Check | What would support the claim / expose failure |
| --- | --- | --- |
| Correct geometry | Rotate, translate, permute and periodically wrap identical observations; compare declared invariant/equivariant exports; replay batching and precision | Agreement to the measured numerical tolerance. Failure diagnoses implementation or input conventions, not an optimization benefit |
| Retained liquid information | Raw-embedding nearest-neighbor physical error and matched linear/nonlinear probes, within noncrystalline atoms; density/coarse-order controls reported separately | Improvement over initialization and simple controls on held-out sources; nonlinear success with linear failure indicates difficult accessibility, not necessarily absent information |
| Distinct interfaces and defects | Nonbulk confusion matrices, per-class readouts, native-space cluster/context contingencies and threshold sensitivity | Interfaces, faults and liquid motifs remain distinguishable with recorded class coverage. Good bulk crystal/liquid separation alone is insufficient |
| Appropriate spatial structure | Distance-matched boundary AUROC; inspect labels across interfaces and within homogeneous regions; collapsed and shuffled controls | Latent distances identify structural boundaries without fragmenting comparable environments. Merely making neighboring atoms alike can erase the boundary |
| Appropriate dynamics | Fixed-atom exact-lag trajectories plus noise, cutoff/membership and real-rearrangement controls; raw spread and reference-normalized increments | Robustness to benign perturbations alongside detectable true changes. A small jump alone can be achieved by collapse or over-smoothing |
| Useful future information | Frozen z-only linear/MLP likelihood probes, constant/initial/descriptor baselines, and descriptor-plus-z add-back comparisons | Lower held-out log loss and Brier with calibration and source uncertainty; structural readout gains alone are insufficient |
| Information accessible in the original observation | Compare z-only against a matched diagnostic given z plus current physical descriptors or original geometry; declare larger context/history separately | A reproducible add-back benefit exposes information absent or inaccessible to the z-only readout. No benefit is not proof of sufficiency |
| Reproducible effects | Matched initialization, data, support, updates and readout budgets across training seeds; source-group uncertainty and material/potential transfer | Effects replicate across seeds/sources. Many correlated atoms/windows do not replace independent simulations |

PTM and [averaged bond-order parameters](https://arxiv.org/abs/0806.3345) provide
complementary structure references. [PTM](https://arxiv.org/abs/1603.05143)
supports template identification and orientation information; it is not a full
classification of liquid motifs or a definition of a critical nucleus. Keep
continuous order/coherence, local density, topology, crystal connectivity and
material-appropriate defects alongside template labels. Al HCP-like faults and
HCP Zr must not share an unqualified defect label. Image colors are not ground
truth without atom assignments.

Existing checks: [evaluation guide](evaluation.md),
[liquid/interface definitions](../metrics/encoder_parameter_search.md),
[noise-response definitions](../metrics/embedding_noise.md). Existing static
spatial splits are exploratory; new independent tests use source separation,
or explicitly buffered spatial holdouts when separate sources are unavailable.

## Two additional diagnostics to specify before implementation

**Response to real rearrangements.** Compare latent movement for persistent
ordering changes against reversible fluctuations at matched coordinate RMS
displacement, physical lag and current coarse order. Recompute neighbor lists.
Use train-defined thresholds and several independent physical descriptors;
report continuous descriptor-change relationships as well as thresholded pairs.
Retain the direct noise curves and information scores so that a larger response
to every perturbation cannot count as improvement. This tests sensitivity to
meaningful change; it does not assume all thermal motion is irrelevant.

**Future outcomes among latent neighbors.** Use nearest neighbors from fitting
sources to estimate probabilities of predeclared future outcomes for each
held-out query. Choose neighbor count/smoothing using only readout-selection
sources and likelihood. Evaluate log loss and Brier against the fitting-prior
and descriptor-distance controls on identical rows. This tests the native
embedding geometry, complementary to a learned MLP that can rearrange it.
Report the full population and coarse-order-matched subsets separately. It is
a supervised evaluation readout, never a self-supervised encoder loss or selector.

The [native MACE quality contract](../metrics/encoder_quality.md) now implements
the full-population latent-neighbor forecast and a limited temporal association
diagnostic with frozen populations, scales, weights and source hashes. It does
not yet implement displacement-matched rearrangement controls or the
coarse-order-matched forecast subset. Those stronger diagnostics still require
their exact definitions and populations to be frozen before scientific exports.

## The nucleation target needs a separate population

The [completed ancestry audit](../../experiments/crystallization_origin_20260925/RESULTS.md)
finds 98.5% existing-crystal arrival among Al64 positive windows at both 3 and
6 ps, and zero regional-birth training windows on the current prediction grid.
Consequently, current onset performance primarily tests arrival. This does not
mean the full trajectories lack births: there are 206 operational isolated
establishment candidates in the training sources.

Preserve Al64 as the arrival benchmark. Use the separately proposed
[regional-establishment population](../../experiments/crystallization_origin_20260925/HARVEST_PROPOSAL.md)
to evaluate precursors, with causal origin-time eligibility that permits small
ordered embryos. Compare eventual establishment, dissolution and crystal
arrival as distinct outcomes. Never choose input atoms from future nucleus
membership or remove an origin because of its future outcome. Labeling may use
future persistence; encoder observations and risk eligibility may not.

At matched present size/order/density, ask whether z distinguishes embryos with
different future behavior. This conditional subset asks about information beyond
simple order; it does not replace a representative natural-population evaluation.
Static Ta/Zr motifs are candidate structures until their futures are observed.
Ta's current external audit has 13 branch-local establishment candidates in
one preparation-ancestry group; static Zr supplies no temporal outcomes.

Bond-orientational fluctuations and competing fivefold motifs have been
implicated in hard-sphere crystallization by
[Russo and Tanaka](https://arxiv.org/abs/1109.0107). This motivates checking
multiple kinds of order, not assuming that increasing any order parameter
raises nucleation probability in our metallic systems.

For a stronger eventual assessment, repeated unbiased trajectories from matched
configurations can estimate the probability of reaching a declared crystal basin
before returning to liquid, with a stated velocity/randomness ensemble. A finite
3/6 ps establishment probability is a different target. Such committor analyses
have identified multi-variable reaction coordinates in
[Ni3Al](https://arxiv.org/abs/2004.01473); that result does not transfer its chemical
inputs or thresholds to geometry-only Al/Ta/Zr. Local geometry may omit relevant
surroundings, so full-system fate does not imply local-state sufficiency.

## Practical comparison and decision rule

Start by applying the existing checks to saved initial and 4/12/24-epoch exports;
the regional outcome diagnostic awaits its separate versioned population. For
new controlled training comparisons, predeclare 24 full epochs, retain earlier
diagnostics, use at least three matched seeds and the same source/sample release.
Record examples seen and optimizer updates as well as epochs. Keep final
checkpoints fixed for the comparison; any additional within-run selector must
use the branch's declared objective. Three seeds improve replication but do not
guarantee statistical power.

Follow [the fixed Al64 roles](../datasets/fixed_al64.md) for matched arrival
comparisons; never resplit or omit model-specific rows. An event-rich regional
population needs a separate release inheriting source roles. Existing historical
test sources remain exploratory evidence; changing sample density does not make
them untouched confirmation. Evaluate candidates across held-out preparation
ancestries when available, with an eventual untouched confirmation population.

Report paired effects per training seed and source-resampled intervals, minimum
meaningful improvements/noninferiority margins declared from validation and
measurement repeatability, and all unsupported classes. Do not interpret a
nonsignificant difference as equivalence. Do not combine these questions into
an arbitrary weighted score or accept a gain on a favored visualization.

Use the current policies: no explicit temperature/age/time model inputs; no AP
losses or AP selection; no physical-reconstruction pretraining. Physical
descriptors remain evaluation controls. Self-supervised encoders use label-free
training/selection, while supervised encoders and frozen predictive readouts use
declared likelihood objectives. New MACE defaults remain geometry-only, width
128/export 128, batch/microbatch 256, with online scientific W&B tracking.

Call an encoder structurally improved only when its liquid fidelity/interface
gains replicate without unacceptable continuity or defect regressions. Call it
predictively improved only when proper scores and calibration support that
claim for the correct target. Call it a precursor representation only after
regional-birth evidence. None of these finite checks proves a universally
sufficient or Markovian state.
