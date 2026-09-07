# Predictive atlas for local atomic dynamics: method and current progress

Status date: 2026-09-04

## Research question

We want a representation of a local atomic state in which two states are close
when they induce similar **distributions of future evolution**, rather than only
when their instantaneous point clouds look alike.

For a parent configuration (p), central atom (i), shooting branch (b), and
future horizon \(\tau_\ell\), let

\[
Y_{pib\ell}=\text{local environment of atom }i\text{ at }\tau_\ell
\]

under branch (b). Multiple shooting branches start from the same positions but
use independent momentum/noise realizations. They are samples from the
conditional future law given the present state.

The current method learns

\[
S_{pi}\longmapsto z_{pi}\in\mathbb{R}^{32}
\longmapsto \widehat\mu_{pi}\in\mathbb{R}^{768},
\]

where (z_{pi}) is a low-dimensional predictive state and
\(\widehat\mu_{pi}\) approximates a kernel mean embedding of the distribution of
future representation paths.

The primary predictive distance is

\[
d_{\mathrm{pred}}(S,S')=
\left\|\widehat\mu(S)-\widehat\mu(S')\right\|_2.
\]

This is a distributional prediction problem. It is not a regression to one
realized future and it does not use future cluster labels for training.

## Construction of the future-law target

### Frozen future encoder

Every present and future local cloud is encoded with the repository's pretrained
GeoFrameTransformerV2 checkpoint:

`output/detached/vicreg_geoframe_v2_factor_sn_grouped_scratch_20260831_160541/GF_V2_FACTOR_SN_GROUPED_SCRATCH_G0.1_N0.1_B16384-epoch=34.ckpt`

The teacher encoder is frozen. Its invariant output has 128 dimensions. Point
cloud extraction is deterministic, periodic boundary conditions are handled by
minimum-image neighborhoods, and atom IDs identify the same central atoms across
time and shooting branches.

The present implementation predicts laws in this frozen representation space.
It does not claim that the representation is injective over raw point-cloud laws.

### Joint future paths

For each branch, the future change at 6, 12, and 24 ps is computed in teacher
embedding space:

\[
\Delta u_{pib\ell}=E(Y_{pib\ell})-E(S_{pi}).
\]

Each horizon is standardized using optimization data, the three horizons receive
equal normalized weights, and the changes from the same branch are concatenated:

\[
v_{pib}=
[w_1\widetilde{\Delta u}_{pib1},
 w_2\widetilde{\Delta u}_{pib2},
 w_3\widetilde{\Delta u}_{pib3}]
\in\mathbb{R}^{384}.
\]

Using several horizons from the same branch preserves their pathwise dependence;
the target is not merely a collection of unrelated future marginals.

### Random Fourier features and empirical mean embedding

A fixed random Fourier feature map approximates a mixture of three RBF kernels:

\[
\phi(v)\in\mathbb{R}^{768}.
\]

There are 256 features at each bandwidth. The fixed bandwidths are approximately
5.973, 11.947, and 23.894 in standardized path space. The same feature map and
normalization are reused across the newer runs so their future-law distances are
on the same scale.

The supervised target for one parent atom is the branch average

\[
\mu_{pi}=\frac{1}{B}\sum_{b=1}^{B}\phi(v_{pib}).
\]

With a characteristic kernel and infinitely many features and samples, this mean
embedding identifies the future-path distribution. Here it is a finite-sample,
finite-feature approximation. Split-shot agreement is reported to quantify its
sampling noise.

## Present-state model

### Spatial context

The input contains the selected central environment plus 16 deterministic
satellite environments within 9.192189 Å. Every environment contains 160 atoms
and is encoded independently by the invariant GeoFrame teacher. The model retains
the satellite embeddings, relative offsets, radial-shell descriptors, and central
token identity instead of reducing the neighborhood to a mean and standard
deviation.

A two-block spatial attention network uses pairwise distance RBF features. The
central output token is projected to a 32-dimensional atlas coordinate.
Temperature is provided as an explicit conditioning variable.

### Temporal context

The accepted temporal model tracks the same central and satellite atom IDs at
four past times:

\[
t-12,\quad t-9,\quad t-6,\quad t-3\ \mathrm{ps}.
\]

For each spatial token, a GRU encodes the sequence of past-minus-current frozen
embeddings. A learned gate adds this kinetic information to the current spatial
token before spatial attention. Only past and current states are inputs; no
future information enters the predictor.

### Atlas and decoded geometry

The central spatial/temporal representation is mapped to a 32-dimensional chart
(z). A small MLP decodes (z) to the 768-dimensional conditional mean
embedding. Training minimizes mean squared error to the empirical mean embedding,
with weak variance and covariance regularization on (z) to discourage collapsed
coordinates.

The decoded Euclidean distance is the primary global geometry. The decoder
Jacobian also defines a local pullback metric

\[
G(z)=J_g(z)^\top J_g(z),
\]

which is saved as a diagnostic of locally predictive directions. The current
models usually have effective rank near the full 32 dimensions at the configured
relative cutoff, so a substantially lower intrinsic rank has not yet been
demonstrated.

## Data and leakage control

### Earlier fixed-duration shooting data

The first conditional-law experiment used 40 pre-nucleation parents and 12
complete futures per parent, assembled from four compatible shooting campaigns.
Parents were split by independent source MD run. It used 512 atom centers per
parent in the compute-scaled run.

This target is relatively reliable: alternating split-shot halves have correlation
0.922. However, these parents follow fairly similar crystallization progressions,
so this benchmark is easier and less diverse than the transition-balanced pilot.

### Transition-balanced fixed-horizon data

The current principal dataset is:

`/home/ids/vmorozov/simulations/al_meam_nested_shooting_pilot_70304_400-500K_20260902_fixed24ps_float16_compatible`

It contains:

- 36 parent configurations;
- 12 parents at each of 400, 450, and 500 K;
- 4 independent shooting futures per parent;
- 144 complete trajectories;
- 81 frames per trajectory, from 0 to 8000 LAMMPS steps;
- 6, 12, and 24 ps future targets;
- explicit source-run splits: 26 optimization, 4 model-selection, and 6 final-validation parents.

The parents include 30 transition candidates and six liquid/crystal controls.
The controls were selected at source frame zero and therefore have no past
history. The fair position-versus-history comparison uses the 30 transition
parents only:

- 22 optimization parents = 11,264 central-atom training states;
- 4 model-selection parents = 2,048 central-atom selection states;
- 4 untouched final-validation parents = 2,048 central-atom validation states;
- 120 future trajectories in total.

All atom states from the same parent and source run remain in the same split.
The effective number of independent units is the number of source runs, not the
number of atom centers.

Four shots are noticeably noisy: split-shot target correlation is 0.801 on the
30 transition parents, versus 0.922 with 12 shots in the earlier campaign.

### Data completion recovery

The expensive physics trajectories were complete, but 104 compatibility workers
failed after composing their 24 ps binary because they attempted to delete an NFS
directory while numpy memmaps were still open. They had not written final outcome
records. The recovery command accepts only this exact cleanup failure, verifies
the self-contained composed binary and immutable source outcome, and explicitly
records that it neither reconstructed frames nor reran LAMMPS. The campaign then
passed strict summarization with all 144 branches. Future workers preserve the
small intermediate binary and remove only the large text dump.

### Data still in progress

At the last scan, the independent-source campaign
`al_meam_independent_sources_70304_400-500K_30perT_float16_20260902` had 16 of 90
runs complete, all in the 400 K optimization split. It had no strict
`summary.json`, so none of it was admitted to these results. These trajectories
will add independent parent diversity, but a single trajectory from a state is
not by itself a conditional future distribution; repeated fixed-horizon shooting
from selected parents is still required.

## Evaluation protocol

The primary test is future-neighbor consistency on unseen source runs. For each
query, candidates must come from a different source run and match temperature,
parent role, and global crystalline fraction. A static-GeoFrame PCA caliper is
applied before any candidate representation chooses its nearest neighbors. This
asks which representation best distinguishes future behavior among states that
already look approximately similar.

Reported baselines are:

- static local GeoFrame PCA;
- mean/std spatial-context PCA;
- linear VAMP at 12 ps;
- the 32-dimensional atlas latent;
- decoded marginal future embeddings;
- decoded joint-path conditional mean embeddings;
- the empirical conditional-law oracle.

The teacher distance is the Euclidean distance between empirical RFF mean
embeddings. Exact RBF MMD is computed on sampled pairs as a check on the RFF
approximation. The current nested run obtains Spearman 0.990 between approximate
and exact biased MMD.

## Results so far

### Earlier 12-shot pre-nucleation benchmark

The compute-scaled frozen-encoder temporal atlas achieved:

- future-law retrieval distance 0.30036 versus 0.30636 for static PCA;
- +1.957% gain over static PCA;
- source-bootstrap 95% interval +1.659% to +2.224%;
- pairwise future-law Spearman 0.603 versus 0.522 for static PCA;
- empirical oracle gain +16.37%.

This established that a learned conditional-law geometry can outperform static
encoder geometry, but only modestly.

### Harder transition-balanced benchmark

The table below uses the valid matched final-validation subset: the two independent
400 K transition sources. Lower distance is better.

| Representation | Future-law distance | Gain over static PCA |
|---|---:|---:|
| Static local PCA | 0.45715 | 0.000% |
| Linear VAMP, 12 ps | 0.47130 | -3.095% |
| Position-only predictive atlas | 0.45662 | +0.117% |
| Four-frame temporal predictive atlas | **0.45515** | **+0.437%** |
| Empirical future-law oracle | 0.36538 | +20.074% |

Adding temporal context gives a direct +0.320% improvement over the identical
position-only atlas. Its source-bootstrap interval is +0.295% to +0.346%, and
pairwise Spearman increases from 0.3310 to 0.3351. The temporal model's held-out
target R2 is about 0.455.

The improvement is real within this test but scientifically small. In particular,
the retrieval uncertainty calculation contains only two independent matched
source runs. Thousands of atom-level queries do not turn those into thousands of
independent experiments. This result should be described as a promising pilot,
not conclusive evidence of generalization across trajectories.

### Transition predictability audit by target, horizon, and input

The next experiment used the same 30 transition parents to separate three
questions that the single retrieval score conflates:

1. Which statistical object is predictable: one realized future, the conditional
   mean, the conditional future-law embedding, or branch-to-branch variance?
2. How does predictability change between 6, 12, and 24 ps?
3. Which present information is actually helping under the current model class?

The audit uses 512 fixed atom IDs, hence 15,360 parent/atom states and 61,440
realized branch/atom futures per horizon. It retains the source-run split of 22
optimization, four model-selection, and four final-validation parents. The frozen
GeoFrame encoder is unchanged. Dense probes have a 32-dimensional bottleneck and
are selected only on the model-selection sources.

| Horizon | Individual future R2 | Conditional mean R2 | Future-law R2 | Log-variance R2 | Mean reliability | Law reliability | Variance reliability |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 ps | 0.433 | 0.708 | 0.540 | 0.017 | 0.771 | 0.652 | 0.029 |
| 12 ps | 0.405 | 0.667 | 0.536 | 0.012 | 0.781 | 0.632 | 0.021 |
| 24 ps | 0.377 | 0.646 | 0.558 | 0.016 | 0.763 | 0.632 | 0.025 |

This provides a more precise answer than the aggregate retrieval score. The
conditional mean and future-law embedding are reproducibly predictable across all
three horizons. As a descriptive ratio, future-law R2 reaches approximately 83%,
85%, and 88% of split-shot reliability at 6, 12, and 24 ps. Conditional variance
is not resolved: both its reliability and model R2 are near zero with four shots.

The individual-future comparison uses a finite-sample leave-one-out sibling mean
as its reference. With only three remaining sibling futures this is noisy and is
not a theoretical ceiling. A model trained across many parent/atom states can
legitimately outperform that finite-sibling reference by denoising across states.

The controlled input ladder for the joint 6/12/24 ps future-law target is:

| Present input | Final-validation R2 |
|---|---:|
| Metadata only | -0.001 |
| Local GeoFrame + temperature | 0.465 |
| Local GeoFrame + global metadata | 0.468 |
| Central/satellite summary statistics | 0.463 |
| Flattened spatial tokens | 0.418 |
| Flattened spatial tokens + four-frame history | 0.394 |
| Same input with matched shuffled history | 0.359 |

The local input won the preregistered model-selection comparison and was therefore
used for the horizon audit. The slightly higher final-validation number for local
plus global metadata was not used to revise that decision. Real history beating
matched shuffled history by 0.035 R2 demonstrates temporal information, but the
10,987-dimensional flattening is statistically inefficient and performs much
worse than the compact local input. This is a model-form failure, not evidence
that spatial or temporal context is irrelevant.

Across 20,000 held-out, cross-source, temperature-matched pairs, Spearman
agreement with empirical future-law distance is 0.690 for static GeoFrame-PCA
distance and 0.732 for predicted future-law distance, an absolute improvement of
0.042. This pair test is effectively a two-source 400 K result because final
validation contains only one source at 450 K and one at 500 K. The predictive map
improves global ranking but remains undercalibrated on some rare divergent pairs;
one static-near example has empirical distance 0.872 but predicted distance only
0.114.

The complete audited result was copied into the repository at:

`output/predictability_map_nested_transition_geoframe_v2_20260903_135104`

Its `ANALYSIS.md`, `metrics.json`, CSV tables, saved probes, plotted arrays, and
figures provide the exact numerical record.

### Shooting evolution and embedding-trajectory animations

The new result folder also contains paired animations at 400, 450, and 500 K. For
each temperature, the displayed final-validation atom maximizes terminal
four-shot GeoFrame dispersion without using model errors for selection.

- The structure GIFs display the same central atom in all four sibling futures,
  using the 160 nearest atoms and periodic minimum-image relative coordinates.
- Each GIF contains 41 exact frames from 0 to 24 ps at 0.6 ps intervals; no
  structural frame is interpolated.
- The embedding GIFs compare frozen-GeoFrame PCA trajectories with trajectories
  through the local predictive bottleneck.
- Applying the parent-trained predictive map to later frames is explicitly a
  diagnostic out-of-training-distribution use. It is not a learned propagator.
- The 400 and 450 K examples include both censored and crystallizing siblings. All
  four selected 500 K siblings crystallize, but their first-passage times range
  from 15.9 to 26.7 ps.

The paths are not uniformly smooth. The predictive bottleneck compresses much of
the large frame-to-frame GeoFrame-PCA motion, but these animations alone do not
establish Markov closure or a dynamically smooth latent manifold. The GIF index
and underlying numerical trajectory arrays are in
`output/predictability_map_nested_transition_geoframe_v2_20260903_135104/gifs/`.

### Finest-timestep GeoFrame stability audit

The September 4 audit directly measures how quickly the frozen GeoFrame
representation changes under physical evolution. It uses the original nested
shooting trajectories rather than the 24 ps compatibility files because the
original producer stores an early high-cadence segment. The LAMMPS integration
timestep is 3 fs, but coordinates were written every 10 steps; therefore the
smallest observable interval is **30 fs**, not 3 fs.

The audit follows the same 32 deterministic atom IDs through 11 exact frames from
0 to 0.30 ps in all 144 branches. This covers 36 parents, four siblings per
parent, 400/450/500 K, 30 transition candidates, three liquid controls, and three
crystal controls. In total, 50,688 periodic 160-point local environments were
encoded. Two outputs of the accepted epoch-34 checkpoint were tested on exactly
the same clouds: the 128-dimensional VICReg projection used by the predictive
pipeline and the direct 128-dimensional GeoFrame encoder output.

The cross-state scale is the median embedding distance between random local
states from different transition parents at the same temperature. Same-atom
top-1 retrieval searches for the correct time-zero atom among the 32 candidates
from the same branch; chance is 3.125%.

| Representation | Lag | Mean cosine | Median drift / cross-state median | Same-atom top-1 |
|---|---:|---:|---:|---:|
| VICReg projector | 0.03 ps | 0.9284 | 0.773 | 8.57% |
| GeoFrame encoder output | 0.03 ps | 0.9910 | 0.841 | 18.27% |
| VICReg projector | 0.30 ps | 0.9013 | 0.919 | 5.10% |
| GeoFrame encoder output | 0.30 ps | 0.9882 | 0.954 | 6.60% |

The direct encoder's high absolute cosine is not by itself evidence of a useful
stable coordinate: different-parent states also have mean cosine 0.987. After
only 30 fs, physical embedding drift is already 77--84% of the corresponding
same-temperature cross-parent distance. By roughly 0.09--0.15 ps, median drift
has reached about 90--95% of that scale, and same-atom retrieval is approaching
chance. The sibling spread rises almost as quickly as single-branch displacement
from the common parent, showing that randomized momentum and thermostat noise
rapidly affect the instantaneous representation.

This is not inference noise or a failed invariance property. Repeated encoding
and point permutation produce bitwise-identical embeddings. Independent global
rotations change the embeddings by only 2.5e-6--2.9e-6 of the cross-state
distance. At 30 fs, the atom-matched local environments have moved by 0.249 A RMS
while retaining 98.1% of their initial nearest-160 atoms. At 0.30 ps those values
are 0.640 A and 95.3%. Thus most early drift occurs while neighborhood membership
is still nearly fixed and reflects sensitivity to fast physical thermal motion.

The result explains the jagged short-time GeoFrame trajectories seen in the
animations and argues against treating a one-frame embedding as a smooth
dynamical state. The next encoder-side ablation should explicitly suppress fast
thermal motion using short-history temporal pooling or a shooting-law-aware
denoising/slow-feature objective. Such a model must still be selected by held-out
future-law prediction rather than by temporal smoothness alone: a constant
representation would be perfectly smooth but useless.

The complete comparison is in
`output/geoframe_temporal_stability_comparison_finest_30fs_20260904/`. Full
per-lag arrays and stratified results are in
`output/geoframe_temporal_stability_finest_30fs_20260904/` for the VICReg
projector and
`output/geoframe_encoder_temporal_stability_finest_30fs_20260904/` for the direct
encoder output.

## Accepted and rejected model changes

### Accepted

- Original frozen GeoFrameTransformerV2 encoder.
- Explicit 17-token spatial context rather than context mean/std alone.
- Joint 6/12/24 ps path targets from the same shooting branch.
- Fixed multi-band RFF conditional mean embeddings.
- Four past frames for the same spatial token identities.
- Source-run optimization/model-selection/final-validation splits.

### Rejected or retained only as baselines

- **Linear VAMP:** remains an important baseline, but is 3.10% worse than static
  PCA on the hard matched nested test.
- **Ordinary-MD temporal encoder training:** 468,480 overlapping rows improved
  prediction of one realized ordinary-MD future by 0.95%, yet degraded shooting
  future-law retrieval by 0.175%. The objective learned trajectory-specific
  evolution rather than the stochastic conditional law.
- **Direct final-block encoder fine-tuning on the old campaign:** every seed chose
  epoch -1.
- **Direct final-block encoder fine-tuning on the new conditional-law data:** all
  three seeds again chose epoch -1. Its retrieval change relative to the frozen
  temporal atlas is -0.025%, with interval -0.070% to +0.019%. The experimental
  checkpoint is rejected.
- **More epochs alone:** early stopping repeatedly selects very early or unchanged
  models. Compute is not the current limiting variable.

## Current interpretation

The method is functioning as intended: repeated shooting futures define a
distributional target, temporal history helps relative to current positions, and
the evaluation can distinguish predictive retrieval from static resemblance.

The dominant limitation is statistical rather than a shortage of optimizer steps.
Four futures per parent produce noisy law estimates, conditional variance is
effectively unidentified, and only two held-out sources currently meet the strict
matched-retrieval conditions. The +20% oracle retrieval gap shows that future-law
geometry contains much more useful structure than the present network recovers,
but part of this oracle advantage is itself estimated with noisy four-shot means.

The target audit adds an important distinction: the conditional mean and kernel
mean embedding are already predictable at useful held-out R2, even though the
retrieval gain of the structured atlas is small. Therefore the entire task is not
unpredictable. The remaining problem is to turn that regression signal into a
well-calibrated geometry that separates rare, static-near states with different
future laws. Naively flattening more spatial and temporal context is not the
solution; it loses to the compact local representation despite a positive
real-history versus shuffled-history control.

The finest-step audit identifies an additional representation bottleneck. The
accepted static encoder is exactly deterministic and rotation invariant, yet its
instantaneous coordinates lose most same-atom identity on a 0.03--0.15 ps scale.
This makes temporal pooling or thermal denoising a scientifically motivated next
step, not merely a larger context model. It also means that future-law targets
constructed from isolated future frames contain substantial fast-motion
variation that the kernel and branch averaging currently have to absorb.

The current evidence supports the following limited claim:

> On fixed-horizon Al MEAM shooting ensembles, a spatially and temporally
> conditioned mean-embedding atlas retrieves slightly more similar held-out
> future-path distributions than static GeoFrame PCA, while linear VAMP does not.
> On transition parents, a compact local predictor explains roughly 0.54--0.56 of
> held-out future-law variation across 6--24 ps, but four shots do not resolve the
> conditional variance.

It does not yet support a claim of a universally predictive low-dimensional
state or a quantitatively large improvement.

## Highest-value next data

1. Generate at least 12--16 fixed-horizon shots per parent; 24--32 is preferable
   near ambiguous transition states.
2. Increase the number of independent parent source runs, especially in final
   validation. Each temperature/parent-role cell needs several independent
   sources, not one.
3. Select many parents across the transition tube, including growth, dissolution,
   and genuine mixed-outcome states. Do not concentrate only on monotonic global
   crystallization progress.
4. Ensure every parent intended for temporal modeling has at least 12 ps of valid
   prehistory. Controls at frame zero cannot test a history model.
5. Keep source-run splits fixed before shooting and keep all descendants of a
   source in the same split.
6. Retain positions, velocities, atom IDs, cell, physical timestep, temperature,
   source run, parent ID, shot ID, split, and structural diagnostics.

The most useful next benchmark would contain at least 20 independent final-
validation parent sources per temperature and enough shots to form two stable
held-out empirical laws. That would make source-level confidence intervals and
temperature-stratified retrieval scientifically meaningful.

## Highest-value next model work

After more repeated-future data are complete:

1. Add a controlled short-time stabilization ablation: pool a dense 0.03--0.30 ps
   history or train a shooting-law-aware denoising/slow-feature head, while
   preserving sensitivity to held-out future-law differences. Compare against
   simple temporal averaging before changing the full encoder.
2. Replace flat context concatenation with a parent-balanced hierarchical token
   model: temporal attention or a small recurrent encoder within each tracked
   spatial token, followed by permutation-aware spatial attention with relative
   offsets and distances. Keep the compact local model as the mandatory baseline.
3. Train one parent-balanced atlas on the union of the original 12-shot campaign
   and the transition-balanced campaign, without allowing 512 correlated atom
   rows from one parent to dominate a batch.
4. Weight the target loss by estimated split-shot reliability or model branch
   sampling noise explicitly.
5. Add a metric-calibration or pair-ranking term on optimization parents, while
   retaining mean-embedding regression as the primary objective. This directly
   targets the rare static-near/future-divergent failures seen in the audit.
6. Fine-tune more of the present encoder only with a mixed objective that preserves
   static invariance while directly optimizes the repeated-future law. Continue to
   include epoch -1 as a selectable candidate.
7. Add velocities or short displacement fields as explicit equivariant kinetic
   inputs and compare them fairly with the four-frame history representation.
8. Test whether the 32-dimensional atlas can be reduced after target noise falls;
   the current pullback spectrum does not justify claiming a much smaller rank.

Nonlinear diffusion or generative future models are not the next priority. The
conditional mean-embedding atlas already isolates the key scientific question at
far lower cost and with clearer evaluation.

## Reproduction

The accepted transition-history model is:

`/home/ids/vmorozov/experiments/predictive_atlas_nested_fixed24_transition_history_geoframe_v2_20260903`

Run the position-only and temporal comparisons with:

```bash
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet
export PYTHONPATH=.

python scripts/run_predictive_atlas.py frozen \
  --config configs/predictive_atlas_nested_fixed24_transition_geoframe_v2_20260903.yaml \
  --stage all

python scripts/run_predictive_atlas.py history \
  --config configs/predictive_atlas_nested_fixed24_history_geoframe_v2_20260903.yaml \
  --stage all
```

The direct encoder fine-tuning rejection can be reproduced with:

```bash
python scripts/run_predictive_atlas.py finetune \
  --config configs/predictive_atlas_nested_fixed24_transition_encoder_finetune_geoframe_v2_20260903.yaml \
  --stage all
```

The transition target/input audit and shooting GIFs use:

```bash
python scripts/run_predictability_map.py \
  --config configs/predictability_map_nested_transition_geoframe_v2_20260903.yaml

python scripts/render_shooting_dynamics_gifs.py \
  --config configs/shooting_dynamics_gifs_nested_transition_20260903.yaml
```

The finest-timestep stability audit and direct-encoder control use:

```bash
python scripts/analyze_geoframe.py stability \
  --config configs/geoframe_temporal_stability_finest_30fs_20260904.yaml

python scripts/analyze_geoframe.py stability \
  --config configs/geoframe_encoder_temporal_stability_finest_30fs_20260904.yaml

python scripts/analyze_geoframe.py compare-representations \
  --projector-result output/geoframe_temporal_stability_finest_30fs_20260904 \
  --encoder-result output/geoframe_encoder_temporal_stability_finest_30fs_20260904 \
  --output-dir output/geoframe_temporal_stability_comparison_finest_30fs_20260904
```

Both configurations contain explicit output paths and the runners refuse to
overwrite completed results. Set a new output directory before reproducing them.

Principal implementation files are:

- `src/temporal_vamp/predictive_atlas.py`: RFF targets, atlas model, training,
  retrieval evaluation, MMD checks, witnesses, and pullback metric;
- `src/temporal_vamp/shooting_history.py`: deterministic past-token extraction;
- `src/data_utils/shooting_dataset.py`: strict campaign snapshots and source-run
  metadata;
- `src/data_utils/shooting_binary_dataset.py`: periodic binary trajectory loading
  and neighborhood construction;
- `src/temporal_vamp/predictive_atlas_finetune.py`: controlled present-encoder
  fine-tuning;
- `src/temporal_vamp/predictability_map.py`: target/horizon reliability audit,
  input ladder, dense predictive probes, distance calibration, and figures;
- `src/temporal_vamp/shooting_gifs.py`: exact periodic local-structure and
  embedding-trajectory animations;
- `src/temporal_vamp/geoframe_stability.py`: finest-step identity, sibling,
  physical-neighborhood, and invariance metrics and plots;
- `scripts/analyze_geoframe.py stability`: exact nested-campaign
  stability runner;
- `scripts/analyze_geoframe.py compare-representations`: direct encoder versus
  VICReg-projector comparison;
- `scripts/run_lammps_campaign.py nested-fixed-horizon-compatibility`: physical 24 ps
  continuation, strict validation, and exact cleanup-failure recovery.

The detailed chronological log remains in `docs/predictive_atlas_20260902.md`.
