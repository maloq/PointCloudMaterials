# Liquid geometry and crystallization diagnostics

Implementation: `src/research/liquid_geometry/`. The run freezes the recipe,
input hashes, source population and metric contract. Undefined quantities are
blank, not zero. All uncertainty below conditions on the trained weights, fitted
reference population and one training seed. No UMAP coordinates or learned
cluster labels enter any calculation.

## Expanded frozen-encoder experiment

The completed `relaxed_encoder/expanded-20260921` population has 18,771 rows,
150 independent simulation sources and ten cached 128D representations. Source
roles are inherited unchanged. Current PTM Other (`label==0`) defines the primary
population; this is an operational noncrystalline definition, not proof that an
environment belongs to bulk liquid. A supplementary q6<0.35 query mask tests
sensitivity to conspicuous order. It does not exclude interfaces or control all
physical confounding. The fixed source cap256 exceeds the observed maximum224,
so this recipe retains every eligible row. Original MD order8 and future targets
are shared between hot/cold input domains. Role, atom, time, physical descriptor,
event-bin, ancestry and encoder population identities are checked explicitly.

Physical targets are q4, q6, w4, w6, averaged q6, q6 coherence, density_r12 and
smooth coordination. Future means the same atom16 saved frames later (12ps).
Train-only population moments standardize targets and probe inputs; constant
training channels use unit scale and are recorded by the metric kernels. Fits
weight training rows equally; reported errors weight test sources equally.

Four distance interventions use the same frozen encoder:

1. `raw`: subtract training mean, leaving Euclidean distances unchanged.
2. `standardized`: also divide by training population standard deviation.
3. `whitened`: multiply by `(C + 0.001 tr(C)/d I)^(-1/2)`, with training sample
   covariance C. This is shrinkage whitening, not an information gain.
4. `physical_metric`: standardized-input ridge (summed squared loss + alpha1
   squared coefficient norm, unpenalized intercept) predicts five standardized
   physical residuals. Targets q4/w4/w6/averaged-q6/coherence first have a
   training-fitted ridge prediction from temperature one-hot, elapsed time/time²,
   q6, density and coordination removed. Distances use predicted residuals only.
   Test targets, future targets and onset labels never fit this metric.

`rank=(tr C)^2/tr(C²)` is the covariance participation ratio, calculated by
float64 SVD. `trace` is covariance trace. A perfectly constant representation
has rank0/trace0 by convention. `global_liquid_rank` pools test rows;
`conditional_rank` is the equal-source mean of within-source ranks, which removes
between-temperature/source means. Rank measures spread, not physical usefulness.
The physical metric has only five coordinates; low rank there is not collapse.

Every query retrieves31 training neighbors at the **same temperature**, excluding
its simulation source. `neighbor_order_mse` is mean squared standardized-order
distance across neighbors and eight channels, not error of the neighbor mean.
`random_order_mse` uses31 fixed-seed random training rows at the same temperature.
`neighbor_order_gain=1-neighbor_order_mse/random_order_mse`, computed per source
then averaged. The latest-study arm below instead has an analytic source-balanced
random comparator; its numbers should not be pooled with this arm.

`future_neighbor_mse` predicts future order8 by neighbor-mean future order8.
`metric_untrained_topology_mse` predicts hot-input144D persistence images by their
neighbor mean. Topology was **not used to fit this new metric**, but was used in
encoder pretraining; it is not an unseen pretraining target. Hot-input topology
as a common evaluation target can favor encoders preserving thermal details.
All errors use training-standardized target channels.

`physical_probe_mse` decodes current physical residuals from standardized raw
features by fixed-alpha1 ridge. Its baseline is zero residual (the nuisance
model). `future_probe_mse` decodes five future residuals after a baseline fitted
to **all eight current order descriptors** plus temperature/time/q6/density/
coordination. These incremental linear probes measure accessible information;
their values deliberately repeat across all distance interventions.
`physical_probe_gain` and `future_probe_gain` are one minus the ratio of
source-averaged model and baseline MSE. They are not estimates of total mutual
information or nonlinear sufficiency.

`knn_brier`/`knn_logloss` use onset probability `(positive neighbors+0.5)/(31+1)`
and the saved original-MD12ps event (`event<5`). Distance fitting is independent
of these labels; the training neighbor outcomes supply the forecast. The
within-role/temperature shuffled-feature control breaks sample identity while
preserving conditional feature distributions; targets remain fixed.

`normalized_jump` uses exact same-source/same-atom pairs48frames (36ps) apart,
restricted to eligible rows at both endpoints. For each test source, mean squared
embedding displacement is divided by twice the equal-source mean training
within-source covariance trace at that temperature; source results are averaged.
Missing pairs give an undefined result and a zero pair count. This is sparse
36ps drift, not timestep jitter, and low drift alone is not a success criterion.

Saved linear and MLP forecasts are joined by exact original test indices.
Their five hazard logits have horizons0.75/3/6/9/12ps;12ps probability is one minus
the product of all five survival probabilities. Forecast log loss and Brier are
source-equal means. Average precision uses each row weight1/(rows in its source).
Probability clipping at1e-12 affects log loss only up to numerical endpoints.
`distance-interventions.csv` reports source-paired differences from raw distance,
with500 temperature-stratified source bootstrap draws; lower errors are better.

Associations across the fixed ten encoders use Spearman correlations of
source-averaged diagnostics versus matched frozen forecast scores. Bootstrap
draws resample the same sources for all encoders and retain temperature counts.
AP is recomputed from pooled predictions using source multiplicities as weights.
Hot/cold domains are reported separately; pooled associations are confounded by
input domain. Fewer than four encoders or a constant axis yields no correlation.
Multiple related encoders, transforms and readout heads are not independent
training replicates. These exploratory intervals have no multiplicity correction
and do not establish causation or generalization across random training seeds.

## Latest completed contextual forecast bridge

The exact restricted population from `symmetric-relaxed-reuse-20260921` contains
23,396 windows (7,654 test windows/30 test sources). Current-center observed and
relaxed MACE128D features are reconstructed using their actual frame lookups.
Source cache checksums, forecast row/event identities and dense Brier replay
against the original exports must pass. Current PTM fractions are exported;
all rows in this fixed cohort are Other. Both encoders use original MD order8
as the physical target. Only training moments fit standardization.

`effective_rank` is within-source raw-feature participation ratio.
`physical_neighbor_mse` uses the same31-neighbor definition above;
`physical_reconstruction_mse` is error of the mean neighbor target instead.
The random comparator analytically averages squared target distance when a
training source at the same temperature is chosen uniformly, then a row in it
uniformly. `physical_neighbor_ratio` divides neighbor MSE by this comparator.

The saved128-bin CDF has0.75ps cadence through96ps. `dense_brier` averages squared
CDF error over every bin and test row within a source; `brier12`/`logloss12` use
bin15 and event<16. Log-loss probabilities are clipped at1e-7. Positive paired
error improvements mean observed minus relaxed; rank change is relaxed minus
observed. Higher rank is not automatically an improvement.

For each of four forecast heads separately, associations compare paired-source
geometry/rank changes with paired-source forecast error improvements. Spearman
is calculated after centering both changes within temperature. Bootstrap500
resamples paired sources within temperature and recenters each draw. This is
heterogeneity across sources for **two encoders**, not eight encoder replications.
The contextual heads also see spatial context and descriptor histories, so the
bridge cannot attribute all forecast performance to a current-center embedding.

## Historical same-run checkpoint experiment

Original shared-pretraining MACE128→5860 and GATr512→5860 are extracted using
their actual frozen eager-FP32 producer and checkpoint-native support. The
encoder has128 and projector64 dimensions. The native Al dynamic release has
no usable PTM field for these sampled atoms: **this arm is all-phase** and must
not be called a liquid-only result. Probe training samples2048 source-balanced,
outcome-blind rows; selection has480 available rows. The same rows and native
future views are used at both checkpoints and for both architectures. Source
lineage is disjoint, but selection previously informed encoder selection; this
is a development diagnosis, not an untouched final test.

Current and future native physical85 targets comprise radial32, pair32,
angular16 and moments5 channels. Train-standardized fixed-alpha1 ridge predicts
them from encoder/projector features. Each block reports source-equal MSE and
`skill=1-MSE/training-mean-constant-MSE`. Future blocks also compare with the
actual current-descriptor persistence forecast. Undefined zero-denominator
skills remain blank. Native positive lags are verified from same atom IDs,
integration steps, source timestep and saved times, and exported explicitly.
Squared same-atom drift is source-averaged and normalized by twice the **pooled
training** covariance trace. This historical normalization differs from the
within-temperature primary arm. Rank and physical/predictive skills must accompany
drift to guard against a constant representation appearing temporally stable.
