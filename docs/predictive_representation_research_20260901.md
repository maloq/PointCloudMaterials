# Research recommendation: a spatial, distributional predictive-state model

Date: 2026-09-01

## Bottom line

The next model should not be a larger version of the current bottleneck and should
not start with another VAMPnet. The highest-probability improvement is a small
**spatial context network trained against the distribution of future changes**:

1. keep the current GeoFrameTransformerV2 checkpoint as the local encoder;
2. retain the individual central and satellite embeddings and their relative
   positions instead of reducing the satellites to mean and standard deviation;
3. represent the sibling shooting outcomes with a kernel mean embedding, so that
   all stochastic futures contribute and distances compare future distributions;
4. train the predictive coordinates directly to reproduce future-neighbour
   relations, with multi-horizon latent prediction and VICReg regularization;
5. pretrain the context network on the much larger ordinary trajectory collection,
   then fine-tune it on the smaller shooting ensemble.

This is a predictive-state representation: two present configurations are close
when their conditional future distributions are close. It remains decoder-free and
keeps the point-cloud encoder reusable.

## Why the current method plateaus

The September 1 GeoFrameV2 run is useful evidence, not a failed experiment. It
shows that the input contains future signal: held-out future-regression R2 is 0.108
overall and 0.167/0.151 at 6/12 ps, above the temperature/phase mean baseline
overall. It also isolates why that signal did not become a better metric.

### 1. The target is dominated by static persistence

The current target is the mean **absolute future embedding** over sibling branches,
followed by PCA. A local environment often remains recognizably itself at short
lags. Static PCA can therefore retrieve similar absolute future embeddings without
learning what changes. This is particularly problematic because absolute future
embedding distance is also the retrieval evaluation target.

Use a dynamic target instead:

\[
r_{s,h}=P_h\left[z_{s}(t+h)-\widehat{E}(z(t+h)\mid z(t))\right],
\]

where the simplest baseline is either \(z_s(t+h)-z(t)\), or a ridge prediction of
future embedding from the current local embedding fitted on optimization runs.
Supplement it with physical dynamic quantities that are inexpensive and meaningful:
PBC-correct cage-relative displacement, neighbour/bond survival, changes in
Steinhardt order, and future local crystalline fraction. Absolute-future retrieval
should remain a secondary compatibility metric; future-*change* and outcome
retrieval should be primary.

This mirrors the closest atomistic work. The DeepMind glass model predicts the
isoconfigurational mean particle displacement rather than a future structural code,
and GlassMLP predicts mobility/bond-breaking propensity.

### 2. Spatial arrangement is discarded

The current context is

\[
[z_{central},\;\operatorname{mean}(z_{satellites}),\;
\operatorname{std}(z_{satellites})].
\]

It cannot distinguish two contexts with the same satellite histogram but different
spatial organization. Satellite offsets and satellite identity are also absent from
the shooting cache. This is exactly the information needed to distinguish a compact
incipient nucleus, an interface, and dispersed crystalline motifs.

Both major glass-dynamics results emphasize extended context. The DeepMind model
uses recurrent message passing and reports increasing performance with more
recurrences and a larger effective context. GlassMLP coarse-grains structural
descriptors over 16 length scales. Work comparing linear models and GNNs also finds
that the first generation of neighbour-averaged descriptors gives a large gain and
the second a smaller additional gain. The lesson is that the representation of
context matters at least as much as raw network size.

### 3. Eleven futures are collapsed to one mean

Different sibling branches are samples from
\(p(X_{t+h}\mid X_t)\). Their mean does not describe multimodality, variance, or a
mixture of crystallizing and non-crystallizing outcomes. The measured reliability
of the current 11-shot mean is already 0.83--0.86, so adding more shots to the same
parents is no longer the main issue. We should use the existing shots more
informatively.

### 4. The learned geometry is only indirectly tied to retrieval

The current pairwise loss matches normalized scalar Euclidean distances between a
16-dimensional bottleneck and a concatenated mean-future target. It does not match
neighbour rankings or conditional distributions, and one scalar per pair discards
most relational information. The prediction head can also use a distorted
bottleneck as long as the final linear layer partially repairs it.

### 5. The nominal sample count overstates the independent data

The last run contains 40 parent configurations, 20 source runs, and 440 branches.
The predictor split is only 22 parents/11 runs for optimization, 6/3 for selection,
and 12/6 for validation. The 1,408 optimization atom rows are spatially and
configuration correlated; they are not 1,408 independent dynamical experiments.
This is far smaller than representative propensity studies: one comparison uses
100 initial configurations with 50 isoconfigurational trajectories per
configuration, and GlassMLP uses 300 initial structures.

## Proposed model

Working name: `GeoFrameDistributionalPredictiveState`.

### Present-state encoder

For every queried atom, construct 17--33 context tokens: the central atom and
16--32 deterministically selected satellite centres over multiple radial shells.
Each token contains:

- frozen GeoFrameV2 local invariant embedding;
- radial basis features of its offset from the queried atom;
- optional inexpensive physical scalars (q4/q6/q8, shell population, local
  crystalline fraction, and per-atom energy if it is available from the producer);
- temperature as a conditioning variable, not as a prediction target.

Use two or three width-128 attention/message-passing blocks with four heads and a
pair-distance RBF bias, followed by attention pooling around the central token.
Only distances and invariant node features enter the outer context network, so the
output stays rotation and translation invariant. This should be well below one
million trainable parameters. The repository's GeoFrame token encoder already has
the needed pairwise-geometry-bias and pooling pattern; this is an outer context
module, not a new point-cloud architecture.

Before that network, run a cheap diagnostic baseline: concatenate the central
GeoFrame embedding with one and two generations of distance-weighted neighbour
averages. If this already improves substantially, the context transformer should be
judged against it, not only against mean/std.

### Future-distribution target

For every shot \(s\) and horizon \(h\), form a compact future-change vector
\(r_{s,h}\), initially using 16--32 PCA dimensions fitted only on optimization
runs. Draw fixed random Fourier features at several RBF bandwidths:

\[
\phi(r)=\sqrt{2/D}\cos(Wr+b), \qquad
\mu_h(X_t)=\frac{1}{S}\sum_{s=1}^{S}\phi(r_{s,h}).
\]

Then

\[
\|\mu_h(X_i)-\mu_h(X_j)\|^2
\]

approximates maximum mean discrepancy between the two empirical future
distributions. Unlike a raw mean, this target distinguishes distributions with the
same mean but different spread or modes. A practical first setting is 256 features
per horizon, with bandwidths at 0.5, 1, and 2 times the optimization-set median
distance. Cache these fixed targets.

With only 11 shots, also report split-shot stability and test simple shrinkage of
the empirical kernel mean toward the temperature/phase mean. Do not learn the
future encoder or kernel bandwidth on validation runs.

### Objective

Let \(\xi_i=f_\theta(X_i)\) be a 32- or 64-dimensional context state and
\(g_h(\xi_i)\) the predicted kernel mean at horizon \(h\).

1. **Distribution prediction**

   \[
   L_{pred}=\sum_h w_h\|g_h(\xi_i)-\mu_{i,h}\|_2^2.
   \]

2. **Future-neighbour distillation**

   Within batches containing different source runs, define a teacher neighbour
   distribution from future MMD distances and a student distribution from latent
   distances:

   \[
   p_{ij}\propto\exp[-d^2_{MMD}(i,j)/T_p],\qquad
   q_{ij}\propto\exp[-\|\xi_i-\xi_j\|^2/T_q],
   \]

   and minimize \(\sum_i KL(p_i\|q_i)\). Exclude the same parent and same source
   run from candidate neighbours. This is the continuous-target analogue of
   neighbourhood component analysis and directly optimizes the scientific
   retrieval question.

3. **Anti-collapse regularization**

   Reuse the repository's tested VICReg variance and covariance terms on \(\xi\).
   A stop-gradient/EMA target branch is useful during temporal pretraining; the
   shooting kernel targets themselves are fixed and do not need an EMA.

Use separate predictor heads and validation scores for 6, 12, and 24 ps. Leave
48 ps out of the first fit: its present-only predictability is approximately zero in
the current run and it can dominate optimization noise. Add it only if the shorter
horizons improve. Do not force the first model into 2--16 coordinates. First show
that a 32/64-dimensional predicted state improves retrieval, then compress it with
PCA/PLS or a smaller bottleneck and measure the information loss.

### Two-stage training

1. **Temporal pretraining:** use all valid ordinary trajectories with run-level
   splits. Predict multi-horizon future-change GeoFrame targets from context tokens,
   using the SPR/JEPA pattern of a predictor, stop-gradient or momentum target, and
   VICReg regularization. This supplies thousands of distinct parent frames even
   though each has only one realized future.
2. **Shooting fine-tuning:** fit the future-distribution and neighbour-distillation
   objectives on complete shooting outcomes. Freeze GeoFrameV2 initially. If the
   context model clearly generalizes, unfreeze only the last GeoFrame token block at
   a 10--50 times smaller learning rate.

The last predictor did not use the long ordinary trajectory roots. At the time of
this audit, 18 complete `trajectory.npz` files are accessible in the 400/450/550 K
campaign and seven more in the completed 500 K campaign. These are immediately
useful for pretraining after deduplication and run-level splitting. The new
12-temperature IDS campaign is still in `prepared` state, and the 40-branch
shooting extension currently has only 14 complete outcomes; neither should be
silently counted as a complete new dataset. The older four-temperature summary
also claims a completed 600 K group whose directory is currently absent, so the
catalog must resolve that inconsistency before admission.

## Nucleation-specific head: learn a committor

If the main scientific target is nucleation rather than a general future geometry,
add a committor head. Define liquid basin A and crystalline basin B using the
repository's PTM largest-connected-cluster coordinate. For parent \(i\), if
\(k_i\) of \(n_i\) valid shots reach B before A, optimize the exact binomial
negative log likelihood

\[
-k_i\log q_i-(n_i-k_i)\log(1-q_i).
\]

The logit of \(q_i\) is an interpretable one-dimensional reaction coordinate. A
multinomial head handles several terminal states. This is statistically preferable
to regressing noisy empirical fractions with MSE, and the repository already has
jumpy-FFS/PTM basin infrastructure. It should be an auxiliary readout from the
shared context state, not a replacement for the general distribution target unless
all of the science is reduced to one A/B event.

## Position-only versus phase-space prediction

These should be reported as two different questions:

- **Position-only input** estimates isoconfigurational propensity: what the
  structure makes likely after randomizing momenta. This is the clean test of the
  original hypothesis.
- **Position plus velocity, or two-frame history** predicts an individual future
  branch. It will usually be substantially more accurate because it observes part
  of the phase-space state, but it no longer measures purely structural propensity.

Train the position-only model first. A velocity/history-conditioned residual head
is then a valuable upper-bound experiment and can decompose predictable variation
into structural propensity and momentum-conditioned residual.

## Ranked ablation plan

| Order | Experiment | Question answered |
| ---: | --- | --- |
| 0 | Re-score static PCA and current bottleneck on future changes, bond survival, and local crystallinity | Is the old conclusion caused by an absolute-future metric? |
| 1 | Ridge/MLP on GeoFrame plus 1--2 hierarchical neighbour averages and q4/q6 | How much gain comes from cheap multiscale context? |
| 2 | Spatial context transformer, mean future-change target | Does retaining spatial arrangement improve prediction? |
| 3 | Replace the mean target with multi-bandwidth RFF kernel means | Do multiple futures and distribution shape matter? |
| 4 | Add soft future-neighbour KL plus VICReg | Can the learned Euclidean geometry beat static PCA? |
| 5 | Pretrain the context model on ordinary trajectories | Is shooting performance data-limited? |
| 6 | Fine-tune the last GeoFrameV2 block | Is the frozen static encoder the remaining bottleneck? |
| 7 | Add velocity/two-frame history as a separate upper bound | How much uncertainty is irreducibly hidden from positions? |
| 8 | Add a binomial committor head | Can the representation predict the nucleation event itself? |

Do not combine these changes in the first run. The ladder identifies which idea
actually creates the gain.

### Ablation 5 implementation

The ordinary-trajectory pretraining ablation is implemented by
`scripts/run_shooting_temporal_pretraining_ablation.py` with
`configs/shooting_temporal_pretraining_ablation5_geoframe_v2_20260901.yaml`.
It reads the migrated `trajectory_binary_float32` arrays by memory map, tracks the
same 64 shooting center atom IDs, and caches GeoFrameV2 tokens for a central local
environment, 16 spatial satellites, and central futures at 6, 12, and 24 ps.

The split is intentionally stricter than an ordinary random train/validation
split. Velocity seeds 35869 and 35879 are excluded because they define the final
shooting holdout; seed 35863 is used only for pretraining early stopping; seeds
35803, 35831, 35839, 35851, and 35897 are optimization data. Four converted EAM
runs are rejected by the catalog's MEAM parameter hash. With an eight-frame
(24 ps) anchor stride, the current 18-run pool contains 466 anchors and 29,824
center-atom rows.

Run the complete experiment with:

```bash
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet
cd /home/infres/vmorozov/PointCloudMaterials
PYTHONPATH=. python scripts/run_shooting_temporal_pretraining_ablation.py \
  --config configs/shooting_temporal_pretraining_ablation5_geoframe_v2_20260901.yaml \
  --stage all
```

Extraction is resumable through per-anchor shards. `--stage extract`,
`--stage pretrain`, and `--stage train` allow restarting later phases without
re-encoding point clouds. Temporal pretraining predicts PCA modes of deterministic
same-trajectory embedding changes. Only the spatial-transformer backbone is
transferred: the distributional RFF shooting head is newly initialized, and all
shooting fitting and held-out retrieval settings are identical to ablation 3.

The 2026-09-01 run completed with all three pretraining seeds reaching nearly the
same ordinary selection MSE (0.5358--0.5364; standardized zero-prediction MSE is
1). Transfer improved the primary predicted-kernel retrieval gain over local PCA
from 1.279% to 1.877% at 12 ps and from 1.049% to 1.881% at 24 ps. At 6 ps it was
effectively unchanged/slightly worse (1.841% to 1.807%). Across all horizons the
gain increased from 1.146% to 1.500%. The hidden representation improved more
consistently, but the absolute result remains below the predeclared 5% success
criterion. Ordinary temporal pretraining therefore helps longer-horizon transfer,
but is not by itself the large predictive breakthrough sought here.

### Ablation 6 result: final GeoFrameV2 block fine-tuning

`scripts/run_shooting_encoder_finetune_ablation.py` implements the next controlled
test. It caches the float32 outputs of frozen GeoFrameV2 layers 0--4 and the fixed
pair-geometry tensors for all 40 shooting parents, then trains only transformer
layer 5 and the final transformer normalization together with the same
ordinary-pretrained spatial predictor. The patch encoders, pair-geometry network,
layers 0--4, VICReg projector, distributional target, split, and retrieval
evaluation are unchanged. Reconstructing the initial embeddings through this
cache differed from the original frozen cache by at most 4.77e-6.

At the preregistered encoder learning rate of 1e-5, selection losses worsened for
seeds 11 and 22 and improved slightly for seed 33; the selected seed remained 22.
Its predicted-kernel gain over local PCA changed from 1.807% to 1.768% at 6 ps, 1.877% to 1.679% at
12 ps, 1.881% to 1.842% at 24 ps, and 1.500% to 1.443% jointly. The hidden
representation also declined jointly from 1.155% to 0.967%. This ablation is
rejected: ablation 5 remains the accepted checkpoint. With only 28 independent
shooting optimization parents, supervised encoder adaptation adds variance rather
than useful predictive information.

### Ablation 7 result: two-frame history and branch velocity

`scripts/run_shooting_dynamical_ablation.py` evaluates three additions on top of
the accepted ablation-5 predictor without changing its target or split:

- a frozen GeoFrameV2 encoding of the same central and 16 satellite atom IDs 3 ps
  before the shooting parent;
- rotation-invariant local position/velocity couplings at branch time zero;
- their combination.

This experiment deliberately separates two scientific questions. History is a
parent-level structural-propensity feature and is fit to the unchanged ensemble of
11 sibling futures. Velocity differs across siblings, so it is fit and evaluated
against each branch's realized future. Averaging the velocity-conditioned branch
predictions provides a secondary comparison to the parent-level distribution; it
is not used to make the easier individual-outcome problem look like an improvement
on structural propensity.

The run used all 40 parents, 440 complete branches, 64 tracked centers per parent,
and the same 6, 12, and 24 ps RFF future signatures as ablation 5. Reconstructing
the current embedding through the new cache differed from the existing cache by at
most 4.77e-6. The history residual raised held-out parent R2 only from 0.65364 to
0.65543. Its primary all-horizon parent retrieval gain over ablation 5 was 0.111%;
the per-horizon changes were +0.059%, -0.087%, and +0.080% at 6, 12, and 24 ps.

For individual stochastic branches, held-out R2 was 0.22216 for position only,
0.22277 with history, 0.22270 with velocity, and 0.22269 with both. The joint
all-horizon future-neighbour gain was -0.018% for history, -0.089% for velocity,
and -0.054% for both. Averaging velocity-conditioned predictions over siblings
changed parent retrieval by -0.001%. These effects are far below both sampling
noise and the 5% success threshold, so ablation 7 is rejected and ablation 5
remains the accepted checkpoint.

The velocity result has a useful physical interpretation rather than indicating a
loader failure: the shooting Langevin damping time is 0.3 ps, while the earliest
evaluated future is 6 ps (20 damping times). Initial momenta should therefore be
mostly forgotten. A short-horizon 0.3--3 ps control would test whether the velocity
features carry the expected transient signal, but it is not likely to improve the
6--24 ps structural-propensity representation. A committor head is deferred until
the branches have a preregistered basin/outcome label and enough observation time;
inventing a label from the same learned embeddings would not be a valid committor
experiment.

Run or reproduce this ablation with:

```bash
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet
cd /home/infres/vmorozov/PointCloudMaterials
PYTHONPATH=. python scripts/run_shooting_dynamical_ablation.py \
  --config configs/shooting_dynamical_ablation7_geoframe_v2_20260901.yaml \
  --stage all
```

The complete result is stored under
`/home/ids/vmorozov/experiments/shooting_dynamical_ablation7_geoframe_v2_20260901`.

### Ablation 7b result: short-horizon momentum and ballistic controls

The follow-up `scripts/run_shooting_short_horizon_ablation.py` uses the same 440
branches, source-run split, 64 centers, frozen GeoFrameV2 checkpoint, and
ordinary-pretrained spatial architecture, but extracts futures at 0.3, 0.6, 1.2,
and 3 ps. These horizons span one to ten Langevin damping times. A new output head
is trained because the ablation-5 head has a different 6/12/24 ps target. The
deterministic current embeddings agree with the original cache to 4.40e-6.

The selected position-only seed reached held-out parent-distribution R2=0.5886.
Its predicted-signature retrieval gains over local GeoFrame PCA were 2.635%,
1.651%, 2.578%, and 1.995% at 0.3, 0.6, 1.2, and 3 ps, or 2.075% jointly. Thus the
same model has clearer—but still sub-threshold—predictive geometry at short times.

Adding the 75 invariant velocity descriptors changed joint individual-branch R2
from 0.20308 to 0.20317 and joint retrieval by +0.100%. To distinguish a weak
descriptor from a genuinely absent force-free signal, every complete time-zero
velocity field was also propagated as `x(t)=x(0)+v(0)t` under PBC, converted back
to local point clouds, and encoded by the same frozen GeoFrameV2 model. This
atomwise ballistic control was strongly out of distribution: direct retrieval
changed by -7.58%, -8.00%, -9.17%, and -9.11%, and direct R2 was negative. A
selected PCA/ridge calibration reduced the damage but added only +0.051% jointly.
Initial velocities therefore do not explain the missing 0.3--24 ps predictive
signal in this Langevin solid; force-free rollout is not an appropriate dynamics
model even at the first saved 0.3 ps frame.

`scripts/analyze_shooting_branch_outcomes.py` then audited whether a finite-horizon
committor target is supported. It applies the repository PTM definition to the
last three 0.3 ps frames: a persistent largest crystalline cluster of at least 100
atoms is crystal, all frames below that threshold and 1% crystalline fraction are
liquid, and everything else is censored. Outcomes were 403 crystal, 19 liquid,
and 18 censored. Only 12/40 parents had mixed outcomes; useful variation was
concentrated in the six 400 K, 12 ps pre-nucleation parents. The 450--500 K and
3 ps pre-nucleation strata were nearly all crystal. A classifier on these data
would largely learn temperature/offset and class prevalence, not a transferable
committor, so ablation 8 is not trained on this saturated endpoint set.

The next shooting campaign should adapt parent time or progress-coordinate
interface separately by temperature until sibling success probabilities lie
roughly in 0.2--0.8. It should supply at least tens of independent source runs and
preferably 50--100 mixed-outcome parents per temperature. Fixed-duration failures
should either be extended until they reach a basin or retained as explicitly
censored outcomes. That data would support a binomial committor head without
changing the encoder architecture.

Run the completed short-horizon and outcome analyses with:

```bash
PYTHONPATH=. python scripts/run_shooting_short_horizon_ablation.py \
  --config configs/shooting_short_horizon_ablation7b_geoframe_v2_20260901.yaml \
  --stage all
PYTHONPATH=. python scripts/analyze_shooting_branch_outcomes.py \
  --config configs/shooting_short_horizon_ablation7b_geoframe_v2_20260901.yaml
```

Results are under
`/home/ids/vmorozov/experiments/shooting_short_horizon_ablation7b_geoframe_v2_20260901`.

## Evaluation contract

The primary score should be cross-run-only neighbour retrieval on held-out source
runs, with candidates matched by temperature and approximate time/crystallinity.
Measure the distance between **future distributions of changes**, using held-out
shots where possible. Bootstrap confidence intervals over parent configurations or
source runs, never over atom rows alone.

A serious success criterion is:

- at least a 5% reduction in future-distribution neighbour distance relative to
  static GeoFrame PCA at 6 and 12 ps;
- positive improvement for all training seeds and a run-bootstrap 95% interval
  excluding zero;
- positive within-temperature future-change R2, not merely an aggregate gain over
  temperatures;
- stability to kernel bandwidth, context radius, and covariance/variance loss
  weights;
- rotation and repeated-encoding checks retained from the existing pipeline.

Also report the direct predicted future signature. If `g_h(xi)` retrieves better
futures than `xi`, use the predicted signature as the operational representation;
there is no scientific reason to prefer an arbitrary hidden bottleneck.

## Data recommendation

The next simulation budget should prioritize **more distinct parent
configurations**, not many more sibling shots of the current parents. The present
11-shot target is already reasonably reliable. A useful target is 150--300 parents
drawn from at least 30--50 independent source runs, stratified across temperature,
time before/after nucleation, crystallinity, and spatial regions, with roughly
8--12 shots per parent. Include stable-liquid and clearly crystallized controls as
well as transition-region parents. Sample several parent times from a run, but keep
the entire run in one split.

For the ordinary trajectories, use dense frames for pretraining but down-weight
highly overlapping frames and batch across runs. The effective experimental unit
remains the source trajectory. A large number of atoms from one global state helps
learn spatial fields, but it does not replace independent global histories.

## What not to prioritize yet

- A larger unstructured MLP: it cannot recover discarded spatial arrangement or
  future-distribution shape.
- Nonlinear VAMPnets: they optimize slow/autocorrelated modes and do not by
  themselves fix the isoconfigurational target or retrieval mismatch.
- A full coordinate decoder, diffusion model, or trajectory generator: much more
  data and compute, with no need for the current hypothesis test.
- End-to-end MACE/NequIP training on the shooting rows alone: the independent
  parent count is too small and the existing GeoFrameV2 features already contain
  measurable signal.
- More shots on identical parents before adding parents: current ensemble-mean
  reliability says this has diminishing returns.

## Code reuse map

Repository components to reuse:

- `src/models/encoders/geo_frame_transformer.py`: invariant local tokens,
  pairwise geometry bias, and attention pooling;
- `src/training_methods/contrastive_learning/vicreg.py`: tested variance and
  covariance regularization;
- `src/data_utils/temporal_lammps_dataset.py`: run-aware temporal sampling and
  satellite atom IDs/offsets;
- `src/baselines/descriptor_baselines.py`: Steinhardt, SOAP, and CNA baselines;
- `src/data_utils/synthetic/atomistic/jumpy_ffs.py`: basin/shooting provenance for
  a committor readout;
- the current shooting loader's strict `outcome.json` admission and source-run
  split checks.

External reference implementations inspected:

- [DeepMind glassy dynamics](https://github.com/google-deepmind/deepmind-research/tree/master/glassy_dynamics): recurrent graph context and isoconfigurational mobility targets;
- [SPR](https://github.com/mila-iqia/spr): multi-step normalized latent prediction with a momentum target encoder;
- [V-JEPA](https://github.com/facebookresearch/jepa): decoder-free target-feature prediction and attentive frozen probes;
- [Set Transformer](https://github.com/juho-lee/set_transformer): compact permutation-invariant attention and pooling modules;
- [MICo](https://github.com/google-research/google-research/tree/master/mico): sampled behavioural-distance matching with stop-gradient targets and robust angular/norm distances;
- [VICReg](https://github.com/facebookresearch/vicreg): explicit variance/covariance anti-collapse regularization.

These repositories are design references. The required modules are small enough to
implement against this project's concrete data types without adding their old
framework dependencies.

## Primary literature

1. Bapst et al., [Unveiling the predictive power of static structure in glassy systems](https://doi.org/10.1038/s41567-020-0842-8), Nature Physics 16, 448--454 (2020).
2. Jung, Biroli, and Berthier, [Predicting dynamic heterogeneity in glass-forming liquids by physics-inspired machine learning](https://arxiv.org/abs/2210.16623), Physical Review Letters 130, 238202 (2023).
3. Alkemade et al., [Comparing machine learning techniques for predicting glassy dynamics](https://arxiv.org/abs/2202.09173) (2022).
4. Schwarzer et al., [Data-Efficient Reinforcement Learning with Self-Predictive Representations](https://openreview.net/forum?id=uCQfPZwRaUu), ICLR (2021).
5. Tang et al., [Understanding Self-Predictive Learning for Reinforcement Learning](https://proceedings.mlr.press/v202/tang23d.html), ICML (2023).
6. Zhang et al., [Learning Invariant Representations for Reinforcement Learning without Reconstruction](https://openreview.net/forum?id=-2FCwDKRREu), ICLR (2021).
7. Castro et al., [MICo: Improved representations via sampling-based state similarity for Markov decision processes](https://proceedings.neurips.cc/paper/2021/hash/fd06b8ea02fe5b1c2496fe1700e9d16c-Abstract.html), NeurIPS (2021).
8. Gretton et al., [A Kernel Two-Sample Test](https://www.jmlr.org/papers/v13/gretton12a.html), JMLR 13 (2012).
9. Rahimi and Recht, [Random Features for Large-Scale Kernel Machines](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html), NeurIPS (2007).
10. Lee et al., [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html), ICML (2019).
11. Goldberger et al., [Neighbourhood Components Analysis](https://proceedings.neurips.cc/paper/2004/hash/42fe880812925e520249e808937738d2-Abstract.html), NeurIPS (2004).
12. Jung et al., [Machine-guided path sampling to discover mechanisms of molecular self-organization](https://doi.org/10.1038/s43588-023-00428-z), Nature Computational Science 3, 334--345 (2023).
