# Matched structural trajectory stability, version 1

This is a separate descriptive protocol, `structural_trajectory_stability_v1`.
It does not reuse the within-source normalization of the causal native MACE J
score. All calculations below use float64, including covariance, increments and
PCA, because the native states have large means relative to their variation.

## Population and weighting

The config freezes two randomly selected test sources at each of 400, 450, 500,
510 and 520 K from the pre-existing Al independent-source cohort. Four centers
per source are selected from its fixed center pool with the recorded seed,
independently of outcomes. All 801 frames, 0–600 ps with 0.75 ps cadence, are
used. Tracks retain atom identity. Differences never cross a source or atom
boundary. There are 32,040 test observations and 32,000 adjacent pairs.

Training-reference observations are four seeded centers, frames 0,40,...,800,
from one separate training source per temperature: 420 observations, balanced
by source, temperature, center and time. No reference moments use test data.
The test source lineages are checked against each encoder release's training
and selection lineages. Historical test use remains exploratory.

Means first average time and centers within each source, then weight each source
equally. Bootstrap draws resample whole sources within temperature (2,000 draws,
percentile 2.5/97.5). All tracks/overlapping pairs stay together. Intervals are
conditional on the fixed training-reference moments and trained model seed;
two sources per temperature is a small cohort. Paired ratios use the same source
draws. Pooled p95 jumps are descriptive quantiles, with no bootstrap; equal
trajectory lengths make source weights equal in that pooled sample.

## Representations and support

- MACE and GATr: raw exported z128 from the selected native snapshot VICReg
  checkpoints; no head, projector, feature fitting or temporal smoothing.
  Fixed training Al scale 9.121389139452193 Å, reference scale 9.192189 Å.
  Native support 17 model units = approximately 16.87 Å; taper 15–17 model
  units. MACE uses directed edges at 5 model units. Compiled native BF16 with
  protected FP32 geometry and precision casts preserved; TF32 disabled.
- TDA: repository `persistence_image` on the nearest 80 atoms including the
  tracked center, ties broken by atom ID. 144D alpha-complex H0/H1/H2 descriptor
  in physical Å. H0 has 16 death-radius channels; H1/H2 each have an 8×8
  lifetime-weighted image. Finite deaths above 3.5 Å are excluded. This is
  instantaneous TDA, not relaxed or spatiotemporal TDA.
- SOAP: DScribe single-species Al SOAP; nonperiodic local periodic chart,
  nominal cutoff 7 Å, n_max=8, l_max=6, sigma=0.3 Å, raw 252D power spectrum,
  no PCA or per-sample normalization. Full encoder neighborhood is supplied,
  retaining the Gaussian-density tail outside the nominal cutoff.
- Bond order: first six channels from the checksum-verified existing source
  cache: q4, q6, normalized w4/w6, qbar6 and mean q6 coherence. The producer
  uses 12 nearest bonds of the center and of its 12 nearest neighbors.
- Radial: first 32 channels of repository `geometry_packet`, normalized
  Gaussian radial basis, physical distances 0–7 Å, smooth 5–7 Å taper.
- Angular: channels 64:80 of the same packet, weighted pairwise Legendre
  moments of order 1–16, the same support and taper.

These supports differ. This compares actual representations and does not
isolate architecture, locality, training seed or checkpoint-update effects.
PTM labels and qbar6 are reused only as physical display/conditioning context,
with source manifest, atom, time and cache checksum checks.

## Formulas

For each method let C be the population covariance of the fixed training
reference and V=trace(C). Each temporal displacement is Δz=z(t+lag)−z(t).

- `rms_jump`: sqrt(mean_sources(mean_time,centers(||Δz||²))/(2V)). This is
  the glossary's training-reference normalized RMS jump: 1 corresponds to the
  RMS distance between independent observations from that reference population.
  `jump2_mean` and `lag2` are its per-source squared quantities. Lag curves use
  all overlapping pairs at each declared physical lag.
- `raw_increment_mse`: mean over time, center and feature of squared increments.
- `p95_jump`: pooled 95th percentile of ||Δz||/sqrt(2V).
- `standardized_rms_jump`: sensitivity analysis after dividing each coordinate
  by its training standard deviation, then sqrt(mean coordinate squared
  increment / 2). Retain coordinates with variance > max variance × 1e−12;
  report the retained count. Unlike covariance-trace normalization, this
  diagnostic depends on the feature basis and is not full whitening.
- `rms_acceleration`: sqrt(mean_sources(mean(||z[t+1]−2z[t]+z[t−1]||²))/(6V)).
  This is a normalized second difference, not acceleration per ps²; at the
  fixed cadence no time division is used. Independent reference-distributed
  frames give an expectation of 1 for its square.
- `roughness`: within each source, mean(||Δz[t]−Δz[t−1]||²) divided by
  mean(||Δz[t]||²+||Δz[t−1]||²), then average sources equally. Linear paths
  give 0; independent identically distributed frames approach 1.5; exact
  alternation gives 2. This captures temporal bending/backtracking relative
  to movement size, including actual thermal/structural motion.
- `increment_cosine`: average cosine between consecutive nonzero increments,
  first within source, then across sources. `reversal_fraction` is the fraction
  of these pairs with negative dot product. `zero_increment_fraction` reports
  exact zero steps; undefined direction/roughness stays null, never zero.
- `jump_075_to_12_ratio`: ratio of source-weighted RMS jumps at 0.75 and 12 ps.
- `reference_effective_rank`: (sum eigenvalues(C))²/sum eigenvalues(C)²,
  computed by SVD of centered training observations. This is a participation
  ratio, not matrix rank or evidence by itself of useful retained information.
  Zero reference variance is a fatal error, not a smoothness success.
- `rms_jump_ratio`: candidate/reference RMS jump, using each method's own
  training-reference distance scale. Bootstrap ratios are paired by source.

Phase-conditioned per-source `jump2_mean` uses only adjacent endpoints with
the same specified PTM class: 0 unclassified, 1 FCC, 2 HCP, 3 BCC. Empty groups
have null means and zero pairs. Unclassified is not a validated liquid label;
equal endpoint labels do not prove the interval had no structural transition.

`mean_80_atom_replacement_fraction` averages 1−|intersection(nearest80[t],
nearest80[t+1])|/80. `jump_churn_spearman` correlates this fraction with
per-pair normalized jump within each source. It describes association, not
causation or numerical artifact.

## Displays and numerical controls

Raw time series show the first sampled source per temperature and first sampled
atom. Each method's PC1 is fitted to training observations only and divided by
sqrt(2V); axes are different projections and not physical coordinates. No
smoothing is applied. The standalone interactive explorer includes every track,
with display-only rounding to five decimal places; metric arrays retain their
original precision. Heatmaps contain every test step with a common color
limit equal to the pooled 99th percentile over four displayed representations;
saturation is declared. Tables preserve unsaturated values.

Repeated-input and reversed-batch inference compare the first batch from every
source, with recorded maximum absolute error and MSE. They bound those checked
execution effects only, not input quantization or physical dynamics. Full-box
coordinates are the existing float16 exports; local subtraction/scaling matches
the training producer. The audit makes no sub-cadence or full-precision claim.

The shared structural loader now also exposes a context-only `previous` snapshot
and an optional dynamic-only subset with refitted training target moments.
Trajectory-stability inputs and metric equations do not use or change with these
training options; v7 training curvature is defined separately in
[shared pretraining](shared_pretraining.md#mixed-material-dynamic-triplets-v7).

## Local structural support (v10)

Current structural MACE/GATr observations use fixed material normalization
`x_model = x_A * 9.192189 / scale_material`, crop to radius <8 before packing,
and quintic C2 weights equal to one through radius 6 and zero at radius 8. There
is no outer halo. MACE uses 5-unit edges, two layers and pooling tapers 0–3,
3–5, 6–8; GATr globally attends only within the cropped sphere and scales its
weighted count by 100. Training, static inference and trajectory inference share
`src/data/structural_pretraining/support.py`. Geometry baselines using the
encoder's support and radial controls now also use that local support. Existing
85-component physical and 80-point instantaneous-TDA targets are unchanged.

The revision is incompatible with previous large-support checkpoints. Historical
exported metric contracts and results retain their original support definitions;
reproduction of those runs requires their frozen code. Current within-domain
VICReg, selection, bond-order and temporal-only curvature metric formulas are
unchanged. Curvature weights are recalibrated at initialization using training
batches under the declared 2%-loss / 10%-encoder-gradient policy. See
[local protocol](../shared_pretraining_local_structure_20260918.md).
