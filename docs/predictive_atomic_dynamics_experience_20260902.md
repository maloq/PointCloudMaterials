# Predictive atomic dynamics: experience and conclusions so far

Date: 2026-09-02

## Executive conclusion

We are trying to learn a compact state representation in which two instantaneous
atomic configurations are close when they have similar future dynamics. For the
nucleation problem, the most concrete current version is:

> Given one instantaneous, periodic 70,304-atom Al configuration, predict the
> probability that it reaches a persistent crystalline basin before returning to
> a liquid basin.

The input is the parent state at one time. The main position-only task uses atomic
positions, cell, temperature, and deterministic local-to-mesoscopic structural
features. Velocities and forces are separate ablations because including velocities
changes the question from structural propensity to prediction of one phase-space
trajectory. The target is estimated from repeated LAMMPS futures launched from the
same parent positions.

The work so far establishes five points.

1. The frozen GeoFrame representation contains real future information. The newer
   GeoFrameTransformerV2 checkpoint improves future regression substantially over
   V1.
2. Static GeoFrame PCA is a strong baseline. Linear VAMP and several learned
   predictive bottlenecks have not yet beaten it by a scientifically meaningful
   margin on rigorous cross-run retrieval.
3. Mesoscopic spatial context and multiple stochastic futures both help, but in
   different ways. Context greatly improves regression; representing the full
   future distribution gives the first reproducible improvement in neighbor
   geometry.
4. Extra model capacity, encoder fine-tuning, initial velocities, and a two-frame
   history did not solve the problem with the current data.
5. The dominant limitation is now data design: too few independent parent
   configurations, and especially too few parents with intermediate committor.
   Thousands of frames, atoms, or branches do not replace independent transition-
   boundary parents.

The current result is therefore encouraging but not yet a successful predictive
reaction coordinate. The most useful next investment is an adaptive boundary-
shooting campaign, followed by a censor-aware committor/survival model. A larger
neural model should come only after that dataset exists.

## 1. How the scientific question evolved

### 1.1 Linear temporal representation

The initial hypothesis was deliberately general: local environments should be
considered similar when the same atom has a similar future local environment. For
temporal pairs

\[
(X_i(t), X_i(t+\tau)),
\]

the point clouds were encoded by a frozen GeoFrameTransformer and a regularized
linear VAMP model was fitted to the present and future embeddings. The kinetic-map
coordinates scale the left singular functions by their singular values, so
Euclidean distance emphasizes predictable modes.

This was a good minimal test. It required no decoder, preserved rotation invariance,
and made the statistical object explicit. It also exposed an ambiguity: in an
ordinary MD trajectory, the single observed future mixes structural propensity,
thermal noise, global phase progression, and run identity.

### 1.2 Position shooting and isoconfigurational futures

We then used position-conditioned forward shooting. A parent configuration is held
fixed while new momenta and Langevin-noise seeds generate sibling trajectories.
This is called *shooting* by analogy with transition-path sampling: trajectories are
shot forward from a selected parent state.

The siblings approximate

\[
p(X_{t+\tau}\mid X_t, T),
\]

where the conditioning state contains positions but not one particular set of
momenta. This makes the position-only target an isoconfigurational structural
propensity rather than an individual deterministic future.

### 1.3 From future embeddings to a nucleation committor

Predicting a future embedding is a useful general representation-learning task, but
it is indirect for crystallization. The current primary target is the committor

\[
q(X_t)=P(\text{crystal basin B is reached before liquid basin A}\mid X_t).
\]

Each future is followed until a persistent basin hit or a temperature-dependent
time limit. A branch that reaches neither basin is censored, not silently labeled
liquid. This turns the scientific question into an interpretable and falsifiable
binary first-passage problem.

## 2. Representation and physical context

The strongest encoder used in the latest experiments is the frozen
GeoFrameTransformerV2 factor/spectral-normalization checkpoint:

```text
output/detached/vicreg_geoframe_v2_factor_sn_grouped_scratch_20260831_160541/
GF_V2_FACTOR_SN_GROUPED_SCRATCH_G0.1_N0.1_B16384-epoch=34.ckpt
```

The checkpoint-selected `vicreg_projector` is a 128-dimensional invariant
representation. Embedding extraction is deterministic and uses the repository's
PBC-aware environment construction and deterministic farthest-point sampling. The
local environment contains 160 neighbors and uses a 9.192189 A normalization
radius. Earlier VAMP experiments used the best V1 epoch-159 GeoFrameTransformer;
the V1/V2 shooting comparison subsequently justified moving to V2.

A single uniformly sampled local cloud is insufficient for a nucleation problem: a
20-atom nucleus occupies only about 0.028% of a 70,304-atom system. We therefore
tested progressively richer context:

- a central local embedding;
- deterministic satellite environments around the central atom;
- mean and standard deviation of 8 or 16 satellite embeddings;
- radial/multiscale summaries and q4/q6 descriptors;
- a 17-token invariant spatial transformer retaining satellite offsets;
- for the committor pilot, separate summaries of the largest PTM crystalline
  cluster, its interface, and reproducibly sampled bulk background.

The last representation is targeted using only the parent state. Future outcomes
do not choose the input regions.

## 3. Data used and what the counts mean

### 3.1 Ordinary MEAM trajectories

The compatible current catalog contains 29 long, unseeded 70,304-atom MEAM source
runs at 400, 450, and 500 K. Twenty-eight are 600 ps and one is 999 ps, for
17,799 ps in total and approximately 5,962 frames at the stored 3 ps cadence. The
latest nine runs contribute three runs per temperature.

These trajectories are valuable for temporal pretraining and for finding candidate
transition parents. Dense frames from one run are strongly correlated, however.
They do not provide thousands of independent nucleation histories.

The September 1 temporal-pretraining ablation used 18 compatible runs, 466 anchors,
and 29,824 center-atom rows after its strict source-seed split. It predicted changes
at 6, 12, and 24 ps from one present configuration.

### 3.2 Fixed-duration shooting data

Four compatible shooting roots now contain 480 complete, unique-seed branches.
Every branch has 70,304 atoms, 161 frames at 0.3 ps cadence, and 48 ps duration.
Together this is 77,280 stored frames and 23.04 ns of simulated branch time.

The main September 1 ablation ladder used the first 440 branches: 40 parent
configurations with 11 sibling futures each. The important independent count is 40
parents from 20 source runs, not 440 branches and not the number of per-atom rows.
Its split contained 22 parents/11 source runs for optimization, 6/3 for model
selection, and 12/6 for final validation.

The extra 40 completed branches were not part of that frozen comparison and should
not be retroactively mixed into its reported numbers.

### 3.3 Nested first-passage shooting pilot

The authoritative campaign is:

```text
/home/ids/vmorozov/simulations/
al_meam_nested_shooting_pilot_70304_400-500K_20260902
```

It contains 36 immutable position parents and 144 branches. Each parent has two
independent momentum draws and two independent thermostat-noise futures per
momentum, giving four futures per parent. The split is 26 optimization parents, 4
model-selection parents, and 6 final-validation parents, inherited from source-run
identity.

The model uses exactly one input time, `t=0`. The branch is then monitored every
100 LAMMPS steps, or 0.3 ps with the 3 fs integration step, until first passage or:

| Temperature | Maximum duration | LAMMPS steps |
| ---: | ---: | ---: |
| 400 K | 72 ps | 24,000 |
| 450 K | 48 ps | 16,000 |
| 500 K | 36 ps | 12,000 |

The completed campaign contains 11,968 stored frames and 2,540.4 ps of actual
trajectory time. It produced 131 resolved outcomes: 29 liquid and 102 crystal,
plus 13 censored branches. Only two transition parents had an empirical resolved
committor between 0.2 and 0.8. That last number, rather than the frame count, is the
critical limitation.

## 4. Experimental results in chronological order

### 4.1 Linear VAMP: useful modes, small controlled gains

Linear VAMP on ordinary temporal pairs produced stable spectra and plausible
low-dimensional trajectories. Temperature-conditioned post-hoc fits sometimes
improved future-neighbor distance over static PCA by roughly 1--3%; the largest
exploratory gain was 2.55% at 400 K and 24 ps.

Those gains weakened under the more important cross-run controls. With spatial
context at 12 ps, a 12-dimensional VAMP map improved distance over 12-dimensional
PCA by about 1.78% when candidates were matched by run, temperature, approximate
event time, and moderately similar crystallinity. Under the strictest matching it
was about 1.02% worse. The corresponding strict differences at 6 and 24 ps were
approximately 0% and +0.30%.

Interpretation: VAMP finds time-correlated directions, but much of the apparent
geometry can be explained by temperature and shared run-level phase progression.
Linear one-future temporal pairing alone does not isolate a strong transferable
local propensity.

### 4.2 First 400 K shooting predictor

The first controlled shooting experiment had 12 parents and 96 branches. Its small
MLP predicted the branch-averaged future GeoFrame embedding with held-out
standardized R2 = 0.078 overall. Per-horizon R2 was 0.046, 0.253, 0.034, and -0.060
at 6, 12, 24, and 48 ps.

The 12 ps result showed that the input contains conditional-future signal, but the
learned 16-dimensional bottleneck did not improve the combined neighbor geometry
over static context. With only six optimization parents, model conclusions were
necessarily weak.

### 4.3 GeoFrame V1 versus V2 on 440 branches

Using identical 40-parent data and training logic, the V2 factor/SN encoder was a
clear improvement for future regression:

| Encoder | All | 6 ps | 12 ps | 24 ps | 48 ps |
| --- | ---: | ---: | ---: | ---: | ---: |
| GeoFrame V1 | 0.033 | 0.085 | 0.058 | 0.022 | -0.074 |
| GeoFrame V2 factor/SN | **0.108** | **0.167** | **0.151** | **0.067** | **-0.009** |

These are held-out standardized future-prediction R2 values. V2 beat the metadata
group-mean baseline overall, but its within-temperature R2 was positive only at
400 K: 0.131 at 400 K, -0.073 at 450 K, and -0.091 at 500 K. The strongest cell was
400 K, 12 ps before nucleation, with R2 = 0.243.

Better prediction did not automatically produce a better metric. Relative to its
own static 16-dimensional PCA, the V2 bottleneck changed future-neighbor distance by
+0.15%, -1.50%, -1.11%, and -3.87% at 6, 12, 24, and 48 ps. This is why V2 was kept
as the encoder, while the mean-absolute-future bottleneck was abandoned.

### 4.4 Controlled predictive-state ablations

All later ablations retained the same V2 encoder, source-run split, and 40 parents.
They changed one idea at a time and evaluated held-out, cross-run, temperature- and
parent-phase-matched future-neighbor retrieval.

| Experiment | Main observation | Decision |
| --- | --- | --- |
| Future-change target | Removes much static persistence from the target. | Retained |
| 16-satellite mean/std context | Ridge future-change R2 rose from 0.633 to 0.795, but raw retrieval became worse. | Information present; geometry inadequate |
| 425k-parameter spatial transformer | Validation R2 = 0.701; representation still lost to local PCA. | Capacity alone rejected |
| Distributional RFF target | Predicted future signature beat local PCA by 1.84%, 1.28%, and 1.05% at 6, 12, and 24 ps; 1.15% jointly. | First accepted geometry gain |
| Direct geometry loss | Scratch run collapsed scale; corrected training selected the unchanged distributional model. | Rejected |
| Ordinary temporal pretraining | Joint gain rose from 1.15% to 1.50%; 12/24 ps gains became 1.88%/1.88%. | Retained, but below target |
| Final GeoFrameV2 block fine-tuning | Joint gain fell from 1.50% to 1.44%. | Rejected |
| 3 ps structural history | Parent R2 rose only 0.65364 to 0.65543; joint retrieval changed +0.111%. | Negligible |
| Initial velocity descriptors | Individual-branch R2 and retrieval changed by about 0.0001 and -0.089%. | Rejected at 6--24 ps |
| Short 0.3--3 ps prediction | Position-only future-signature gain was 2.08% jointly; velocity added only 0.10%. | Useful control, still sub-threshold |

The distributional target represents every sibling future by random Fourier
features of its future change and averages those features. It therefore
distinguishes future distributions with similar means but different spread or
modes. Split-shot correlations of 0.927, 0.910, and 0.914 show that 11 futures per
existing parent already give a reproducible target. More shots of these same,
mostly saturated parents have diminishing value.

The preregistered success threshold was at least a 5% retrieval improvement over
static GeoFrame PCA at 6 and 12 ps, stable across seeds with a parent/source-run
bootstrap interval excluding zero. No model has met it.

### 4.5 Why velocities did not help

The fixed-duration shooting protocol uses a Langevin damping time of 0.3 ps. The
main futures at 6--24 ps are 20--80 damping times away, so the initial momentum is
expected to be largely forgotten. Even at 0.3--3 ps, invariant velocity descriptors
added only 0.10% to retrieval. A force-free ballistic rollout was strongly out of
distribution and degraded direct retrieval by 7.6--9.2%.

The nested campaign supports the same conclusion. Only 3 of 63 resolved futures
that differed only in thermostat noise disagreed in outcome, and only 1 of 34
parents had different empirical committors between its two momentum draws. There
was almost no independent momentum-conditioned signal for a phase-space head to
learn.

This does not mean velocity is universally irrelevant. It means this thermostat
and these prediction horizons rapidly erase the chosen initial-velocity signal.

### 4.6 Endpoint labels from fixed-duration shooting were saturated

A PTM endpoint audit of the 440 fixed-duration branches found 403 crystal, 19
liquid, and 18 censored outcomes. Only 12 of 40 parents had mixed outcomes, with the
useful variation concentrated in six 400 K parents selected 12 ps before
nucleation. The 450--500 K and 3 ps-before-nucleation strata were almost entirely
crystal.

Training a committor on this dataset would mostly learn temperature, parent offset,
and class prevalence. This motivated the nested first-passage campaign rather than
inventing a favorable label from the learned embeddings.

### 4.7 Nested committor pilot

The primary final-validation results on transition candidates were:

| Model | Branch Brier | NLL | ROC AUC | Parent pB MAE |
| --- | ---: | ---: | ---: | ---: |
| Training prevalence | **0.1271** | **0.4253** | 0.792 | 0.2305 |
| Collective variables | 0.1424 | 0.7122 | **0.917** | 0.1498 |
| GeoFrame regional structure | 0.1652 | 0.6660 | **0.917** | 0.2076 |
| GeoFrame + collective variables | 0.1620 | 0.6584 | **0.917** | 0.2024 |
| GeoFrame + collective variables + force | 0.1340 | 0.5008 | **0.917** | **0.1219** |
| Phase space | 0.2419 | 0.6706 | 0.708 | 0.2570 |

The prevalence model predicts approximately 0.789 for every final state. It wins
Brier score only because 12 of 14 resolved final transition futures crystallize; it
is not a reaction coordinate. The best structured model ranks parents well and
almost halves parent-level committor MAE, but it predicts 0.032 for the sole mixed
final parent whose empirical committor is 0.5.

There are only four final-validation transition parents. The exact parent bootstrap
95% interval for the structured model's Brier difference from prevalence is
[-0.0445, 0.0995], and the estimated probability that it is better is only 0.316.
The pilot therefore cannot establish calibrated improvement.

The failure is boundary coverage, not absence of all structural information. At
400 K, optimization parents with largest initial crystalline clusters of 38 and 46
atoms are fully censored; the likelihood sees no resolved examples between a
20-atom liquid result and a 64-atom crystal result. The sole 29-atom mixed parent is
in final validation. Model selection is also all crystal among its resolved
transition futures, so it cannot select calibration.

## 5. Engineering lessons and infrastructure now available

The project now has a reusable path from LAMMPS output to held-out predictive
evaluation:

- strict simulation catalogs preserve run, temperature, time, potential, random
  seeds, cell, atom IDs, split, crystallinity, and provenance;
- only direct `outcome.json` records with `state: complete` admit shooting data;
  interrupted attempts and stale status files are never treated as trajectories;
- float32 `.npy` trajectories are opened as read-only memory maps;
- a measured five-frame read improved from 442 ms for text to 3.27 ms for binary,
  about 135 times faster before neighborhood construction;
- atom IDs are tracked exactly across time and all neighborhoods use minimum-image
  PBC geometry;
- point-cloud embeddings and downstream features are cached, so the H100 is used
  for the encoder pass while small linear heads can be iterated cheaply;
- deterministic inference, repeated-encoding checks, rotation checks, and explicit
  cache identities prevent stochastic or stale features from changing a result;
- splits are by source run, never by atom rows or overlapping frames;
- censored first passages remain explicit;
- metrics and uncertainty are aggregated over independent parents or source runs,
  not over the much larger correlated atom-row count.

These safeguards changed several conclusions. In particular, cross-run matching
showed that some VAMP gains were shared phase progression, and strict outcome
handling showed that several apparently liquid branches were actually unresolved.

## 6. What we learned about objectives

### Absolute future embeddings are too static

At short lags, the future local environment resembles the current one. Predicting
or retrieving an absolute future embedding rewards static structural persistence.
Static PCA is therefore difficult to beat even when a model learns some dynamics.
Future *changes*, cage-relative motion, bond survival, and basin outcomes are more
direct targets.

### A conditional mean is incomplete

Sibling trajectories are samples from a distribution. Their mean discards
variance and multimodality. Random-feature kernel means produced the first
reproducible geometry improvement, confirming that the distribution is a better
object than its mean. For nucleation, the committor is an even cleaner scalar
summary of that conditional distribution.

### Prediction accuracy does not guarantee useful Euclidean geometry

The 16-satellite model achieved R2 = 0.795 while making nearest-neighbor retrieval
worse. A prediction head may decode information from a distorted hidden state.
The operational representation should therefore be whichever output actually
retrieves similar held-out future distributions—often the predicted future
signature rather than an arbitrary bottleneck.

### Geometry losses can create misleading collapse

The first direct-neighbor loss run shrank the representation standard deviation to
0.008. Tiny distances preserved some rankings while destroying prediction. This
run is invalid, not a 3.2% improvement. Variance diagnostics and independent model
selection are essential whenever a metric-learning objective can rescale its
coordinates.

### Position-only and phase-space prediction are different tasks

Position-only models estimate structural/isoconfigurational propensity. Adding a
specific velocity field predicts a particular realized future. These results must
be reported separately; averaging an easier branch-conditioned prediction cannot
be presented as an improvement in structural propensity.

## 7. What not to repeat with the current dataset

- Do not fit a substantially larger MLP or VAMPnet to the same 40 parents. It adds
  estimation variance without supplying missing boundary examples.
- Do not fine-tune more of GeoFrameV2 on the current shooting split. Fine-tuning
  the last block already made held-out performance worse.
- Do not add many more futures to parents that always crystallize or always melt.
  Eleven-shot distribution targets are already reproducible.
- Do not evaluate on random atom/frame splits or allow same-run neighbors. This
  measures shared trajectory progression, not transfer to a new run.
- Do not turn censored branches into liquid labels.
- Do not use a fixed parent offset across temperatures. Nucleation speeds and useful
  transition bands are temperature dependent.
- Do not use free ballistic propagation as a dynamics model for this Langevin
  system.
- Do not claim success from AUC alone on four parents or from a prevalence predictor
  on a highly imbalanced final split.

## 8. The data that should be simulated next

The goal is not simply “more MD.” It is more *independent uncertainty at the
decision boundary*.

### Stage 1: independent source histories

Generate at least 20--30 genuinely independent source runs per temperature at 400,
450, and 500 K for the next campaign. Independence should vary the equilibrated
liquid configuration or melt/quench history as well as velocities. Reusing one
validated position field with new Maxwell-Boltzmann velocities gives independent
dynamics, but less structural diversity than independent liquid histories.

Store frames every 0.3--1 ps near the transition. Online or post-hoc PTM analysis
should identify candidate parents, but parent selection should also diversify:

- GeoFrame regional structure;
- largest-cluster size and crystalline fraction;
- cluster compactness and surface geometry;
- number of competing clusters;
- crystal structure type and interface character;
- density and established order parameters.

Do not select every parent from one narrow scalar threshold.

### Stage 2: cheap screening shots

Initially select about 40--60 parents per temperature, preferably no more than one
or two nearby parents from any source run. The observed initial largest-cluster
bands worth oversampling are approximately:

| Temperature | Candidate boundary band |
| ---: | ---: |
| 400 K | 25--55 atoms |
| 450 K | 25--40 atoms |
| 500 K | 18--30 atoms |

Launch four futures per parent using the current 2-momentum by 2-thermostat nested
design. This is enough to reject clearly liquid or clearly crystalline parents
cheaply.

### Stage 3: adaptive expansion at the boundary

Expand only mixed, uncertain, or censored parents to 16--32 futures, for example
4--8 momentum samples and four thermostat-noise samples per momentum. The immediate
objective is tens of mixed parents; the serious nonlinear-model target is at least
about 100 independent boundary parents per temperature.

Use uninterrupted, adaptive first-passage simulations rather than a short fixed
endpoint. Practical hard maxima to test are approximately 144 ps at 400 K,
72--96 ps at 450 K, and 48--72 ps at 500 K, with early termination immediately
after a persistent basin hit. Preserve censoring and event times.

### Stage 4: split before selecting parents

Assign source runs to optimization, model-selection, and final-validation before
parent selection. Every split at every temperature must contain liquid-like, mixed,
and crystal-like parents. Keep all parents and branches from one source run in one
split.

At `t=0`, preserve positions, velocities, forces, IDs, cell, temperature, source
run/time, PTM labels, cluster membership and geometry, and split. For every branch,
preserve nested random seeds, first basin, first-passage time, censoring reason,
trajectory cadence, and a restart. This supports position-only, phase-space, and
survival analyses without rerunning the campaign.

## 9. Recommended next model

For the data already available, the next principled extension is a censor-aware
competing-risks or survival model. It can use all 13 censored nested branches and
their observation times instead of dropping them from a binary likelihood. The
existing regularized collective-variable and regional GeoFrame logistic models
should remain the baselines.

Once the expanded boundary dataset exists, use a small hierarchical model:

1. keep GeoFrameTransformerV2 frozen initially;
2. retain individual tokens from nucleus, interface, and bulk regions with their
   PBC-correct relative geometry;
3. aggregate them with a small invariant attention/message-passing network;
4. predict basin-specific hazards or a binomial committor from all sibling futures;
5. optionally retain the multi-horizon distributional future-change head as an
   auxiliary representation objective;
6. validate on entirely unseen source runs and report parent-level calibration,
   ranking, and bootstrap intervals.

Only after that model generalizes should the final GeoFrame block be unfrozen at a
much smaller learning rate. The main milestone is not lower training loss; it is a
calibrated, source-run-transferable committor that improves on collective variables
and prevalence for mixed boundary parents.

## 10. Reproducibility and detailed records

The stage-specific documents remain the source of implementation and reproduction
details:

- [Linear temporal VAMP](temporal_vamp.md)
- [Fixed-duration shooting predictor](shooting_predictive_training.md)
- [Predictive-state research and ablation rationale](predictive_representation_research_20260901.md)
- [Nested first-passage committor pilot](nested_committor_pilot_20260902.md)
- [LAMMPS position-shooting campaign](lammps_position_shooting.md)

The latest committor configuration and entry point are:

```text
configs/nested_committor_geoframe_v2_20260902.yaml
scripts/run_nested_committor.py
```

Its immutable derived artifacts are under:

```text
/home/ids/vmorozov/experiments/nested_committor_geoframe_v2_20260902
```

They include the resolved configuration, feature caches, exact model parameters,
coordinates and predictions, metrics, and plots. The current result should be read
as a campaign-design pilot with evidence of structural ordering—not as a validated
committor model.
