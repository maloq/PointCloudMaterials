# Rich liquid descriptors: distance likelihood and selection

The population, original row IDs, source roles, and conditional proposal weights
are inherited unchanged from the frozen liquid-predictability configuration.
All queries are liquid, all consumed patches exclude established crystal, and an
established crystal exists elsewhere. No new samples or trajectories are created.
Calibration and test sources never fit normalization, trees, networks or selection.

Each patch uses its cached nearest 80 candidates with radius strictly below 8 Å.
Only these coordinates enter descriptors. Twenty-five patches extend to a maximum
query-centered support of 32 Å. No cell-level neighbor, PTM class, lineage, material,
time, temperature, velocity, crystal distance, or clearance enters a feature.

**442 patch features:** 99 radial/neighbor-distance, pair-distance, angular and
shape invariants; 40 bond-order invariants; 45 common-neighbor fingerprints; and
258 persistence features. Bond order includes q2/q4/q6/q8, neighbor averages,
neighbor-order dispersion, normalized w4/w6 and neighbor-averaged versions, and
bond coherence summaries. Neighbor averaging is restricted to observed atoms.
CNA uses bonded center-neighbor pairs with cutoffs 3.2 and 3.6 Å and a close-packed
adaptive cutoff `(1+sqrt(2))/2 * mean(first 12 neighbor distances)`. For every bond
it counts common neighbors, their mutual bonds, and longest simple chain/ring.
It summarizes 421/422/444/666/555/544/433/other bond fractions and moments. This
adaptive fingerprint is not a full adaptive FCC/HCP/BCC phase classifier.

TDA uses GUDHI safe-precision alpha complexes of the nearest 32 and all consumed
80-or-fewer atoms. Squared filtration radii become radii in Å; essential infinite
H0 is omitted, every finite H0/H1/H2 interval is retained. Features include log
persistence moments, entropy, H0 Gaussian death curves, H1/H2 6×6 surfaces weighted
by tanh(lifetime), and smooth Betti curves. There is no hard finite-death exclusion.
Large boundary intervals contribute log moments even outside image grid support.
The nearest-k and radius membership boundaries themselves are still hard.

For each descriptor, context features are means and standard deviations in three
patch-center shells [0,4], (4,14], (14,24] Å, plus the norm of its first spatial
moment and Frobenius norm of its traceless second spatial moment. Moments use
centered descriptor values and relative patch offsets divided by 24 Å. There are
**3,536 full context features**, all rotationally invariant. Every patch uses the
same feature producer. No laboratory-direction coordinates reach a predictor.

**Likelihood.** Every new arm uses distance edges
`[0,16,20,24,28,32,40,48,56,64]` Å: nine finite uniform-density bins plus a censored
tail. Below 64 Å, NLL is `-log(p_bin)+log(bin_width)`; at or above 64 Å it is
`-log(p_tail)`. Finite upper edges belong to the lower bin, except 64 Å belongs to
the censored tail. Probabilities are clipped at 1e-12 and renormalized for scoring.
The no-input prior uses weighted training bin frequencies plus 1e-8 per bin.
CatBoost minimizes weighted MultiClass log loss; the MLP uses equivalent random
draws from the declared training weights with cross-entropy. The bin-width term
is model-independent on validation, so categorical loss selects the same checkpoint
as distance NLL. No class reweighting, AP loss, or proximity loss is used.
This differs from the earlier continuous lognormal mixture family; comparisons
must retain that fact. Density NLL has the same Å reference measure, but histogram
resolution is a modeling limitation. The matched histogram prior isolates the
benefit of descriptors within the new distribution family.

RMSE compares the predicted mean of capped distance (bin midpoints, tail=64) to
min(target,64). Brier scores at 20/32/48 Å use summed bin probabilities and D≤r.
Scores use the inherited conditional weights. Reliability tables report ten
fixed probability bins with mass, weighted mean probability and event frequency.
Distance subgroups and query distance above the full 32 Å observation envelope
are diagnostics only. All original held-out rows are retained.

Choose the best descriptor arm by full-validation distance NLL, then report its
untouched test scores and all other predeclared arms. Also record if the prior wins.
Bootstraps resample independent sources as units, preserving their conditional
weight mass: 2,000 draws, seed 20260928. Positive NLL gain and positive relative
RMSE reduction favor descriptors. Ordinary 95% intervals and Bonferroni bounds
across nine non-prior arms are exported. Source intervals exclude training-seed
uncertainty; the study uses one seed and a previously used test cohort.

Active boosting fits use CatBoost GPU MultiClass on one dedicated GPU per fit.
The proposed CPU `rsm=0.7` was removed before any fit started because feature
subsampling is unsupported for this GPU objective. Each declared family subset
therefore supplies all its features. GPU floating-point reductions need not be
bitwise deterministic. Plain boosting and Bayesian bootstrap (temperature 1)
are explicit; resolved parameters are saved. Descriptor extraction is unchanged.

Training feature importance is CatBoost PredictionValuesChange, not evidence of
causal physical influence or of held-out utility. Family fits/ablations supply
the validation comparison. All descriptor controls remain locally tracked.

Derived sensitivity/relaxation protocols may reuse these fitted-model calculations
on explicitly sealed derived cohorts. Their prediction-context records name the
input domain and original versus synthetic versus relaxed-label target. Pairing,
synthetic generators and cold membership rules are defined in liquid_controls.md;
historical exported definitions remain frozen.


## Execution refactor

The code-cleanup revision consolidates artifact export, preparation, checkpoint
and execution helpers. Scientific formulas, rows, weights, fitting populations
and selectors are unchanged. New table exports include a per-table hash and
definition binding. Historical exported definitions and frozen source snapshots
remain authoritative; changed implementation hashes require a new export revision.
