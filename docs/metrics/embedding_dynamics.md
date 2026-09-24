# Embedding dimension and temporal stability, version 1

## Matched 0.75 ps update (September 24)

The `encoder_dynamics_lag075_20260924.json` recipe uses **only 0.75 ps** movement.
It adds fresh native checkpoint exports to the existing dense cohort: 40 tracked
atoms, 801 frames each, 10 evaluation sources (32,040 states / 32,000 pairs),
plus 420 normalization observations from five separate reference sources. New
Geoformer and native MACE exports use these same observed MD coordinates,
identities and row order. No relaxation, interpolation, model fitting or new
simulation is performed. Geoformer is implemented by the repository's
`GeoFrameTransformerV2`; the public tables use the user's name Geoformer.

`native_dense.py` checks the stored input hashes, uses nearest 80 neighbors with
atom-ID tie breaks for Geoformer, and preserves each MACE checkpoint's 8 Angstrom
crop and native feature producer. Producer/checkpoint hashes and repeat-inference
differences are recorded. Current MACE was trained on relaxed geometry, so its
use of observed coordinates here is explicitly an input-domain transfer. This
is a descriptive common-input dynamics assay, not a repeated relaxed AP assay.
Native supports, pretraining and precision still differ between architectures.
The reference/evaluation source split is inherited from the historical assay;
the covariance reference is not a claim that every model was trained on it.

`comparison.csv` provides readable names, dataset/noncrystalline/evaluation
participation ranks, movement participation rank and d95, normalized RMS jump,
lag, pair count and source count. Encoder and projector remain separate rows.
Dataset rank includes all 32,460 available exported rows with equal source
weights; movement uses the 32,000 evaluation pairs only. Historical coarse-time
exports retain their original definitions below and their frozen result files.

**Movement d95:** the smallest integer k such that the first k eigenvalues of
the uncentered temporal-increment second moment account for >=95% of its trace.
Equivalently, projection onto those k principal directions retains >=95% of
the source-weighted squared displacement energy. These are learned combinations
of channels, not the first k coordinates and not a nonlinear manifold dimension.
Dividing every displacement by the common 0.75 ps does not change this fraction.

Producer: `src/research/trajectory_stability/spectrum.py`; adapters/export:
`src/research/trajectory_stability/audit.py`. Calculations use float64. These
measure **linear spectral dimension**, not nonlinear intrinsic manifold dimension.
A curved one-dimensional path can span many linear directions. Neither high nor
low dimension establishes retained information, forecast skill or good dynamics.

## Populations, time and weights

`ranks.csv` separates the complete available exported **dataset** (descriptive;
includes fitting rows), the fitting **reference**, held-out **evaluation**, and
evaluation observations after subtracting each atom trajectory's mean
(`within_track`). Within-track means are recomputed for each domain. This last
measurement removes differences between atoms/sources; it is descriptive, never
an inference transformation or training input. This is the rank of the exported
assay dataset, not a claim about every training observation or simulation atom.

Each source/root has equal weight, and rows/pairs within a source have equal
weight. Spectra are computed on this weighted population, not by averaging ranks.
Per-track metrics weight that track's observations equally. Coverage reports
rows, sources, observations per track and algebraic rank ceilings. Sources with
no eligible pair do not contribute to that lag. No confidence intervals are
inferred from correlated frames or single seeds.

Rows join by actual source/root and atom identity. Only exact requested physical
lags (absolute tolerance 1e-8 ps) form pairs. No interpolation, cross-atom steps
or cross-source steps. A gap may contribute at its actual requested lag; it is
never treated as one saved frame. Duplicate track timestamps fail explicitly.
Reference and evaluation source IDs must be disjoint.

Domains include the full population, each temperature and noncrystalline rows.
The dense assay defines noncrystalline as PTM not in {1,2,3}; the current
structural screen already stores this crystal membership as a Boolean `phase`.
This is not an equilibrium-liquid classification. Current-screen diagnostics
also stratify on eligible onset within 12 ps versus no onset within 12 ps.
Those future labels are for analysis only. Both endpoints must satisfy a domain
for its pair metrics; equal endpoint labels do not guarantee no intervening event.

The native structural screen contains only frames 64,224,368,512, from a verified
0.75 ps source timeline: separations **120,108,108 ps**. It supports coarse-time
108/120 ps comparisons, not sub-ps jitter, short-time response delay or reliable
per-trajectory dimension (four observations imply state rank <=3). The separate
historical v6 dataset supplies 801 frames at 0.75 ps. It evaluates older MACE/GATr
and descriptor exports, not the latest checkpoints. Do not mix these cohorts.

## Spectra and dimension

For centered states, C = sum_i w_i (z_i-mu)(z_i-mu)^T. Let lambda_j be its
nonnegative eigenvalues, p_j=lambda_j/sum(lambda).

- `participation_rank = 1/sum(p_j^2)`.
- `entropy_rank = exp(-sum(p_j log(p_j)))`, using only positive p_j.
- `d90`, `d95`, `d99`: minimum number of descending components explaining
  at least 90%, 95%, 99% of energy.
- `numerical_rank`: count(lambda_j > 1e-10*max(lambda)). This declared tolerance
  is not an estimate of physical manifold dimension.
- `rank_ceiling`: min(D,N-1) for centered observations; min(D,N) otherwise.
- `total_energy`: trace(C) for centered states; second-moment trace otherwise.

Spectra use raw exported coordinates, without test-fitted standardization or
whitening. They are invariant to a common nonzero scale and orthogonal basis
change; centered state spectra also ignore constant translation. Arbitrary
coordinate rescaling changes the metric and can change effective dimension.
`eigenvalues.csv` retains the complete spectra, including sample-limited tails.
Zero-energy spectra have numerical rank 0 and undefined effective ranks/energy
dimensions (blank/null). Zero reference variance is a fatal normalization error,
not evidence for successful smoothing.

## Temporal movement

At lag tau, v_i = (z(t+tau)-z(t))/tau. `movement` uses the **uncentered**
second moment sum w_i v_i v_i^T: this includes persistent drift. `fluctuation`
uses the covariance after subtracting the source-weighted mean v. Both report
the same spectral metrics above. Straight, constant-velocity motion has movement
rank one and zero fluctuation variance. Centering alone would erase that motion.
Pooled movement ranks can exceed any individual track's movement rank because
different atoms can move in different directions.

`drift_energy_fraction = ||sum w_i v_i||^2 / sum w_i ||v_i||^2` (undefined for
zero motion); `velocity_rms = sqrt(sum w_i ||v_i||^2)` has raw embedding units/ps.
Per-track movement uses all observed adjacent velocities, with actual time
division; its `adjacent_lags_ps` explicitly records when durations differ.
Population movement spectra always separate the requested lags.

## Stability

Let V be the centered, source-weighted covariance trace of **fitting** reference
observations, fixed for a given exported checkpoint. This compares each model
relative to its own fitting-population spread, not a common raw latent metric.

- `rms_jump = sqrt(sum w_i ||Delta z_i||^2/(2V))`.
- `p50_jump`, `p95_jump`, `p99_jump`: inverse weighted empirical CDF quantiles of
  ||Delta z||/sqrt(2V), without interpolation between quantiles.
- `domain_reference_rms_jump` uses the corresponding fitting domain's V instead.
  It is null if that fitting domain is absent or collapsed; the pooled-reference
  jump remains separately visible. This guards against apparent smoothness
  caused by large separation between phases/temperatures.
- `zero_increment_fraction` counts exact zero movements with source weights.
- Per-track `velocity_roughness` = sum ||v_next-v||^2 /
  sum (||v_next||^2+||v||^2), using only successive intervals of equal duration.
  Linear motion gives 0; iid equal-cadence states approach 1.5; exact alternation
  gives 2. This is not acceleration in ps^-2.
- Per-track `increment_cosine` averages cosine(v_next,v); `reversal_fraction`
  counts negative cosines. Zero-norm pairs are excluded. Missing direction or
  zero-energy roughness stays null, never zero. `turn_pairs` reports equal-lag
  candidate pairs before zero-norm exclusion.

`stability.csv`, `ranks.csv`, `per-track.csv`, `eigenvalues.csv` accompany the
full JSON and input checksums in `technical/`. Existing exported definitions are
preserved: use a fresh output directory for this supplementary analysis.
