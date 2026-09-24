# GeoFrame checkpoint evolution, reference assay v1

This is a static, transductive physical-reference assay. The original encoder
training includes these snapshots. Fitting/testing below refers to the frozen
probe's spatial regions, not a held-out encoder training collection. Colors and
K=7 clusters are never used to define physical labels. No reported static score
establishes future nucleation or temporal persistence.

Each frame has 4,096 uniformly sampled interior anchor atoms plus their nearest
neighbor, in that order. Coordinates have a two-normalization-radius boundary
margin; no periodic box is inferred. Fitting centers lie left of mid-x minus
1.1 radii; test centers right of mid-x plus 1.1 radii. The intervening gap is
excluded from probes. Receptive fields contain 80 nearest atoms including center,
divided by the historical material radius. Encoder and once-applied VICReg
projector use float32, evaluation BN statistics and deterministic FPS.

## Reference measurements

Full-snapshot OVITO PTM enables FCC/HCP/BCC/ICO. Best-template RMSD is retained;
cutoffs 0.08, 0.10 (primary), and 0.12 produce three reference assignments.
Solid means accepted FCC/HCP/BCC, never ICO. `solid_fraction` is the accepted
solid fraction of the center's 14 nearest neighbors. Full-snapshot Al planar
fault labels use PTM cutoff 0.10 with orientations/interatomic distances.

For k=12 and k=14, compute equal-weight complex spherical-harmonic averages for
the center and each of its k neighbors using each atom's own k nearest bonds.
q_l = sqrt(4 pi/(2l+1)) ||q_lm||. `hat_w_l` is the Wigner-3j cubic contraction
divided by ||q_lm||^3, zero for norm <= 1e-14. This **normalized** invariant is
not Hu–Tanaka's unnormalized w_l. qbar6 averages complex q6m over center and
neighbors before taking the norm. Coherence is their mean normalized complex
inner product with the center; connection counts use >0.65/0.70/0.75.
`density_knn` = k/(4 pi r_k^3/3). Primary order uses k=12 for Al, k=14 for Ta/Zr.

Context is an operational proxy, with priority from last applicable assignment:

| ID | Definition |
|---|---|
| 0 | Noncrystalline center, neighbor solid fraction <=0.1, neither candidate below |
| 1 | Crystalline center with neighbor solid fraction >=0.8 |
| 2 | Crystalline center outside interior, or neighbor fraction in (0.1,0.8) |
| 4 | Noncrystalline center with neighbor solid fraction >=0.8 |
| 6 | Noncrystalline, fraction <=0.1, qbar6 >= fitted material threshold |
| 5 | Noncrystalline, fraction <=0.1, accepted ICO or hat_w6 < -0.08 |
| 3 | Al intrinsic fault/coherent twin/multilayer fault, overriding other contexts |

The qbar6 threshold is the 75th percentile of fitting-side noncrystalline,
low-solid-fraction **anchors**, pooled across the three frames per material.
Neighbor copies and test centers do not calibrate it. Candidate 5 is a five-fold
proxy; candidate 6 is ordered liquid compatible with further investigation,
not a proven BCC embryo. The -0.08 rule is a declared exploratory normalized
invariant threshold, not a literature-derived universal phase boundary.
Continuous descriptors and the independent PTM motif axis remain available.
Class 4 does not distinguish grain boundaries from other nontemplate defects.

## Reported scores

- `collapse.effective_rank`: covariance participation ratio, (sum lambda)^2 /
  sum lambda^2, with population centering. Constant embeddings have rank 0.
  `variance` is the covariance trace; `pair_rms` = sqrt(2 trace).
- `context`, `ptm`, `planar_fault`: fitting-side StandardScaler and L2 logistic
  regression (C=1, balanced fitting weights, max_iter=2000). Classes require 20
  fitting observations. Balanced accuracy/confusion concern supported test
  classes; test coverage is explicit. AP is one-vs-rest over all test anchors,
  undefined with fewer than 20 positives or no negatives. Fault types preserve
  OVITO IDs 0–4; contexts preserve IDs above. Cutoff scores refit the probe for
  each reference sensitivity assignment. They are not independent replicates.
- `liquid_order`: ridge alpha=10 reconstructs q4/q6/hat-w4/hat-w6/qbar6/coherence
  within noncrystalline anchors whose solid fraction <=0.1. Input and target
  scalers fit only fitting liquid. Constant targets (variance <=1e-12) excluded.
  At least 40 fit and 40 test centers are required. `r2` is held-out R² by active
  descriptor, possibly negative. Density-only and density+embedding fits share
  rows/scaling; conditional gain is mean(R²_joint - R²_density).
- `liquid_topology`: identical fit on 1,024 uniformly selected anchor indices,
  using existing `liquid_structure.persistence_image` (144D alpha-complex
  descriptor on 65 atoms in Å). This is an independent descriptor assay;
  fidelity to it does not prove that a topology cluster is a physical phase.
- `spatial.auc`: proposed boundary-aware coherence diagnostic. Test-side spatial
  nearest-neighbor pairs must have contexts stable across PTM cutoffs at both
  endpoints. Embedding distance predicts whether endpoint contexts differ.
  Split physical pair distances into five quantile bins; average valid-bin
  AUROCs weighted by pair count. A bin needs 10 equal-context and 10 changed-
  context pairs. A constant distance distribution uses one bin. At least 100
  eligible pairs needed. `nonbulk_spatial` excludes crystal-interior endpoints.
  `same_label_normalized_distance` divides mean equal-context bond distance by
  anchor population RMS pair distance; undefined for a collapsed embedding.
  Shuffling embeddings is a fixed-seed null. Constant embeddings produce AUROC
  0.5 where defined, never perfect coherence.
- `continuity`: first 256 outcome-blind anchors, identical independent Gaussian
  displacement directions at per-coordinate std 1e-4, 0.01, 0.1 Å; fixed outer
  80 atoms, recomputed internal grouping/frames. Median and p95 embedding changes
  divide by those anchors' original population RMS pair distance. Repeated
  unperturbed inference must match exactly. These perturbations are not MD time.
  `triad_index_switch_fraction_1e4_A` measures primary/secondary frame-axis index
  changes at 1e-4 Å while holding group membership fixed; it excludes sign-only
  flips and therefore is not an exhaustive frame-discontinuity detector.
- `nonbulk_cluster_ami`: KMeans K=7, n_init=5, fixed seed, fitted only on fitting
  anchors in native embedding coordinates. Adjusted mutual information against
  reference context on test anchors excluding crystal interior. The 7x7 count
  matrix shows test cluster/context associations; this is not a seven-phase
  ground truth. UMAP is refitted for descriptive plots and never scores motion.

## Reused Al future and temporal assays

Use the verified `structural-state/screen-20260922` cache: 2,880 observations from
45 independent roots, split 25 fitting / 5 tuning / 15 development. Crop each
centered **relaxed** radius-8 Å patch to its nearest 80 atoms and divide by the
historical Al normalization radius 9.192189 Å. Require at least 80 stored atoms
and the 80th inside the support, avoiding a truncated neighborhood. This is an
input-domain transfer from the original static collection; future labels remain
the original MD labels from the existing producer, not relaxed geometry labels.

`future_residual_9ps` reuses the fitting-only present-state baseline and target
normalization from `structural_state.dynamics.targets`: current order8, relaxed
geometry89, temperature and time/time². Ridge alpha=10 fits standardized
embeddings to that standardized residual. MSE is averaged equally per source,
then across sources; per-source and PTM-stratum errors are retained. The zero
residual prediction is the declared present-state baseline.

`conditional_hazard` reuses the existing five-bin sustained-onset labels, source
weights, optimizer and tuning-NLL selection, including its step-zero constant
risk option (1,024 updates, fixed seed, linear hazard head). All readouts receive
fitting-standardized temperature indicators + current order8 + relaxed geometry89
as conditions. Compare embedding+conditions against a dimension-matched zero
embedding+conditions control. This adds current physics to the previous assay's
temperature-only conditions; **the resulting AP is not directly interchangeable
with the previously reported MACE AP**. Report 12 ps source-weighted Brier/AP,
all five horizons and saved predictions, not accuracy on all mostly-liquid atoms.

`temporal_response` pairs adjacent recorded frames of the same root and atom ID
(108 or 120 ps lags). Physical change is Euclidean distance of current order8,
with scaling fit only on fitting sources. Its fitting-pair 25th/75th percentiles
define quiet/changing extremes. On development pairs report Spearman correlation
of physical change with embedding distance, and AUROC distinguishing changing
from quiet pairs. Also normalize their mean distances by development population
RMS pair distance. Report all pairs and pairs whose two endpoint PTMs are OTHER.
These are sparsely sampled **relaxed** states; this is change responsiveness,
not sub-picosecond thermal continuity. Ta/Zr temporal scores are not inferred
from these Al measurements.

Epoch summaries average frame-level scores within each material, without an
independence claim or confidence interval. Epochs are correlated observations.
Undefined values remain null/blank, never zero. CSVs export numeric leaves;
per-descriptor vectors, counts and confusion matrices remain in JSON. Frozen
implementation/document hashes accompany each export.


Table export: 2026-09-23T15:13:57.656114+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
