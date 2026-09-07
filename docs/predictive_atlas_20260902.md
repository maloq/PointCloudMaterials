# Conditional mean-embedding predictive atlas

This experiment replaces single-realization future supervision with the empirical
distribution of future paths produced by the Al MEAM shooting campaigns. The
scientific claim is deliberately limited to future laws in the frozen GeoFrameV2
representation space; the target encoder is not known to be injective on raw local
point clouds.

## Data contract

The run merges four completed, immutable campaigns under
`/home/ids/vmorozov/simulations`:

- `al_meam_position_shooting_70304_400-500K_48ps_8shots_20260830`
- `al_meam_position_shooting_70304_400-500K_48ps_1shot_local_20260831`
- `al_meam_position_shooting_70304_400-500K_48ps_2shots_local_followup_20260831`
- `al_meam_position_shooting_70304_400-500K_48ps_40branches_local_20260901`

The strict merged snapshot contains 40 parent states, 20 source MD runs, 64
selected atoms per parent, and 480 complete branches. Every parent has exactly 12
independent velocity/thermostat seed pairs. A branch is admitted only through the
repository's complete-outcome and float32-binary trajectory contracts. The run
does not read campaigns still in progress.

Parents from validation source runs are never used to fit target normalization,
the path kernel, PCA/VAMP baselines, or network parameters. Within the remaining
training runs, configured velocity seeds form a source-run-level model-selection
split. Evaluation queries and candidates come from different source runs and are
matched by exact temperature and parent phase, global crystalline fraction, and a
static GeoFrame-PCA caliper.

## Model and target

The frozen encoder is:

```text
output/detached/vicreg_geoframe_v2_factor_sn_grouped_scratch_20260831_160541/
GF_V2_FACTOR_SN_GROUPED_SCRATCH_G0.1_N0.1_B16384-epoch=34.ckpt
```

For every branch and selected atom, the target is the same-branch concatenation
of full 128-dimensional GeoFrame embedding changes at 6, 12, and 24 ps. Each
horizon is standardized using optimization source runs and receives equal total
weight. A three-band RBF kernel is approximated with 3 x 256 fixed random Fourier
features. Averaging those features over the 12 branches estimates the conditional
mean embedding of the joint future-path distribution.

The input network receives a central GeoFrame token, 16 spatially arranged
satellite tokens, invariant q4/q6/first-shell descriptors, pair distances, and
temperature. Its spatial transformer is initialized from the accepted ordinary
temporal-pretraining backbones. It produces a 32-dimensional latent state and a
nonlinear decoder maps that state to the 768-dimensional conditional mean
embedding. The primary predictive distance is Euclidean distance between decoded
mean embeddings, not unqualified Euclidean distance in the latent chart.

## Reproduction

Use the repository conda environment and the H100 when available:

```bash
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet
export PYTHONPATH=.

python scripts/run_predictive_atlas.py frozen \
  --config configs/predictive_atlas_geoframe_v2_480branches_20260902.yaml \
  --stage extract

python scripts/run_predictive_atlas.py frozen \
  --config configs/predictive_atlas_geoframe_v2_480branches_20260902.yaml \
  --stage train
```

`--stage all` performs both steps. Extraction is deterministic and cached. The
configuration has an exact 40-parent/480-branch guard so silently training on an
incomplete campaign is impossible.

The run directory is:

```text
/home/ids/vmorozov/experiments/predictive_atlas_geoframe_v2_480branches_20260902
```

It contains the resolved config and dataset snapshot, immutable embedding/token
caches, `model.pt`, `path_kernel.npz`, `vamp_baseline.npz`, all coordinates and
predictions, witnesses, metrics, and plots.

## Current result

On held-out source runs, the selected atlas has target reconstruction R2 = 0.661.
For cross-run nearest-neighbor retrieval, lower future-path distance is better:

| Retrieval space | Mean held-out distance | Gain vs static PCA |
|---|---:|---:|
| Static local GeoFrame PCA, 32D | 0.35127 | reference |
| Predicted marginal future means | 0.35067 | +0.17% |
| Predicted joint-path mean | **0.34622** | **+1.44%** |
| Empirical joint-path oracle | 0.31375 | +10.68% |
| VAMP, 32D at 12 ps | 0.39125 | -11.38% |

The joint-path gain has a source-run bootstrap 95% interval of +0.74% to +2.13%
and a probability of positive gain of 1.0 in 5,000 bootstrap samples. In sampled
held-out pairs, Spearman correlation with empirical future-law distance is 0.734
for the predicted joint embedding, 0.707 for the marginal target, and 0.659 for
static PCA. Thus joint temporal coupling adds reproducible signal, but most of the
oracle headroom remains.

The random-feature distance is a faithful approximation to the configured exact
kernel MMD on sampled pairs (Spearman 0.995; 4.0% mean relative absolute error).
Split-shot halves have flattened target correlation 0.922, so 12 shots provide a
useful but still noisy empirical law. Shuffling horizon alignment within each
parent moves the validation target by 0.107 on average, confirming that the joint
target contains information absent from separate horizon marginals.

Two limitations are important. Euclidean distance in the raw 32D latent chart is
not yet a useful replacement for decoded atlas distance: its retrieval gain is
-1.19%. Also, the decoder pullback has effective rank 32/32 for every sampled state
at the configured 1e-4 relative cutoff. This run therefore supports conditional
future-distribution learning, but does not yet demonstrate a much smaller
intrinsic predictive dimension. New, more outcome-diverse parents and additional
shots should be evaluated before increasing model complexity.

## Tests

`tests/test_predictive_atlas.py` checks that the joint kernel distinguishes
correlated and anticorrelated two-time processes with identical marginals, checks
RFF against exact RBF MMD, verifies rotation invariance of the spatial input, and
checks pullback-spectrum construction.

```bash
PYTHONPATH=. pytest -q \
  tests/test_predictive_atlas.py \
  tests/test_shooting_predictor.py \
  tests/test_temporal_vamp.py
```

## Four-frame past-context ablation

The follow-up run at
`/home/ids/vmorozov/experiments/predictive_atlas_history4_geoframe_v2_480branches_20260902`
adds explicit structural history at 12, 9, 6, and 3 ps before every parent. For
each current central and satellite token, the same atom ID is tracked backward in
the source MD trajectory and its local point cloud is encoded by the same frozen
GeoFrame model. A shared GRU processes each token history from oldest to newest;
the resulting temporal token is fused before the spatial attention blocks. The
model is warm-started from the selected position-only atlas and retains that
checkpoint if temporal fine-tuning does not improve the source-run selection
loss.

Alignment checks found a maximum source/parent position error of 7.63e-6 and a
maximum independently re-encoded current-token error of 1.36e-6. This rules out
frame or atom-ID misalignment as an explanation for the result.

Temporal context raises validation target reconstruction R2 from 0.66134 to
0.66379 and pairwise Spearman correlation with future-law distance from 0.73379
to 0.73592. It does not improve the primary matched retrieval result: future-law
distance changes from 0.346221 for the position atlas to 0.346581 for the history
atlas. The direct gain is -0.104%, with a source-run bootstrap 95% interval of
-0.328% to +0.110%. The history atlas still beats static PCA by 1.335%, but is not
better than the position-only atlas.

This is a controlled negative result for 3 ps-sampled position history on the
current parents. It does not establish that history is generally irrelevant:
these parents are strongly matched by phase/progress and come from only 20 source
runs, while much of the atomistic momentum memory is lost well before the first
6 ps target horizon. A useful future test needs parents whose recent growth or
dissolution rates differ despite matched instantaneous structure.

## Expanded overlapping-center and encoder fine-tuning experiment

The next run reuses every complete fixed-horizon shooting branch but increases
the deterministic centers from 64 to 512 atoms per parent. This creates 20,480
parent/atom states and 245,760 realized future paths (20,480 states times 12
shooting branches). The local clouds and their satellite neighborhoods overlap;
this is intentional data augmentation over spatial positions, not 20,480
independent dynamical initial conditions. The number of independent parents and
source runs remains 40 and 20, respectively, and all splitting and bootstrap
evaluation therefore remain source-run based.

The experiment retains the four past frames at -12, -9, -6, and -3 ps and the
17 spatial tokens at each frame. It also freezes the original 64-center RFF path
kernel and target normalization, so increasing the center count cannot make the
prediction task easier by redefining the target. The frozen-encoder atlas is fit
for up to 1,200 epochs with source-run early stopping and three seeds. A second
stage exposes only GeoFrameV2 transformer block 6 and its final normalization
(198,970 parameters) for present-central-token fine-tuning at learning rate
2e-6; the future encoder, past tokens, and satellite tokens remain frozen. The
unchanged epoch -1 model is an explicit candidate in model selection.

Reproduce it with:

```bash
python scripts/run_predictive_atlas.py finetune \
  --config configs/predictive_atlas_expanded512_history_finetune_geoframe_v2_20260902.yaml \
  --stage all
```

The completed run is
`/home/ids/vmorozov/experiments/predictive_atlas_expanded512_history_finetune_geoframe_v2_20260902`.
It contains the expanded embedding/history/activation caches, both model files,
coordinates and predictions, witnesses, metrics, and plots.

The expanded frozen atlas reaches validation target R2 = 0.68267. Its decoded
joint-path mean-embedding distance retrieves held-out future laws at mean distance
0.30036 versus 0.30636 for static local PCA: a +1.957% gain with source-bootstrap
95% interval +1.659% to +2.224%. Pairwise Spearman correlation with held-out
future-law distance is 0.603 for the decoded prediction and 0.522 for static PCA.
The 32D latent alone gives a smaller +0.611% retrieval gain. The empirical target
oracle gives +16.37%, so substantial learnable headroom remains. The target is
more reliable at this spatial sample size: its split-shot correlation remains
0.922 while the estimate covers eight times as many centers.

All three encoder fine-tuning seeds select epoch -1. Fine-tuning therefore changes
neither retrieval nor validation R2; its measured gain over the expanded frozen
model is exactly zero. The implementation reconstructs the original current
embedding to maximum absolute error 1.91e-6 before optimization, so this rejection
is not caused by a cached-activation mismatch. The accepted scientific result is
the expanded, longer-trained atlas with the encoder frozen.

## Newly completed simulation scan

The nested shooting pilot at
`/home/ids/vmorozov/simulations/al_meam_nested_shooting_pilot_70304_400-500K_20260902`
is complete and strict-valid: 36 parents, 144 branches, and source-level splits
of 104 optimization, 16 model-selection, and 24 final-validation branches. Its
outcomes are 29 liquid, 102 crystal, and 13 right-censored; only two parents have
mixed resolved outcomes. These trajectories stop on first basin entry and thus
do not all contain the fixed 6/12/24 ps observations required by the current
joint-path target. They must be used as a separate censored committor/outcome
auxiliary task unless fixed-horizon continuations are generated; terminal frames
must not be carried forward as fabricated future observations.

The independent-source campaign
`al_meam_independent_sources_70304_400-500K_30perT_float16_20260902` was still
running at the final scan and had no `summary.json`, so it was not admitted to
this experiment. Once complete, its principal value is additional independent
parent trajectories and leakage-resistant source-run splits, rather than more
correlated centers from the existing 40 parents.

## True temporal GeoFrame encoder pretraining

The earlier ordinary-trajectory ablation trained only the downstream spatial
context network on frozen GeoFrame embeddings. The new
`scripts/run_temporal_encoder_pretraining.py` stage instead updates GeoFrameV2
transformer blocks 5--6 and the final normalization (397,684 parameters). A
frozen copy of the original checkpoint supplies PCA targets for embedding changes
at 6, 12, and 24 ps. A prediction head is first fit with the encoder frozen; joint
fine-tuning must then beat this frozen-head baseline on independent velocity-seed
runs. Embedding distillation with weight 0.25 limits loss of the strong static
representation. The exported checkpoint retains the repository's original
Lightning/Hydra format and loads through `load_frozen_encoder`.

The first run used 18 ordinary MD runs, 466 non-overlapping 24 ps anchors, and 64
centers: 29,824 rows. All three seeds accepted nonzero updates, but the selected
ordinary-MD selection improvement was only 0.198%. In the fixed-teacher shooting
test, the resulting atlas was 0.041% worse than the previous atlas, with 95%
source-bootstrap interval -0.106% to +0.056%.

The compute-scaled run implements the proposed overlapping-data test. It uses the
same 18 independent runs but 915 anchors at 12 ps stride and 512 deterministic
centers, producing 468,480 rows (15.7 times more). It caches the final-two-block
upstream activations on the H100 and trains with batch size 8,192 for 300 joint
epochs. H100 utilization during the run was approximately 90--98%. All three
seeds reproducibly improve the ordinary-MD held-out prediction:

| Seed | Frozen-head MSE | Fine-tuned MSE | Best epoch |
|---:|---:|---:|---:|
| 11 | 0.553883 | 0.548620 | 293 |
| 22 | 0.553930 | 0.548781 | 289 |
| 33 | 0.553998 | 0.548650 | 299 |

The selected ordinary-MD gain is 0.950%. Nevertheless, its fixed-teacher shooting
atlas reaches future-law distance 0.300889, compared with 0.300362 for the
previous frozen-encoder atlas. This is a direct -0.175% gain (a degradation), with
95% source-bootstrap interval -0.265% to -0.093% and zero positive bootstrap
probability. Pairwise future-law Spearman correlation falls from 0.60296 to
0.59577. The temporally trained static PCA also becomes 0.130% worse than the
fixed teacher PCA under the same caliper.

This is a useful negative result: more overlapping data and more compute do train
a better predictor of the single future realized by an ordinary trajectory, but
that signal does not transfer to the stochastic conditional future distribution.
The accepted encoder therefore remains the original frozen GeoFrameV2 checkpoint.
Further encoder training should use repeated shooting-law targets (or a mixed
ordinary-MD plus shooting-law objective), and requires more independent,
outcome-diverse parents rather than still more correlated ordinary trajectory
rows.

Reproduction configurations are:

- `configs/temporal_encoder_pretraining_geoframe_v2_468480_20260902.yaml`
- `configs/predictive_atlas_temporal_encoder_scaled_input_cache_20260902.yaml`
- `configs/predictive_atlas_temporal_encoder_scaled_history_20260902.yaml`
- `configs/predictive_atlas_temporal_encoder_scaled_20260902.yaml`

The main artifacts are under
`/home/ids/vmorozov/experiments/temporal_encoder_pretraining_geoframe_v2_468480_20260902`
and
`/home/ids/vmorozov/experiments/predictive_atlas_temporal_encoder_scaled_20260902`.

## Transition-balanced fixed-horizon rerun (2026-09-03)

The fixed-horizon compatibility campaign at
`/home/ids/vmorozov/simulations/al_meam_nested_shooting_pilot_70304_400-500K_20260902_fixed24ps_float16_compatible`
is now strict-complete: 36 parents, four futures per parent, 144 paths, and exact
6/12/24 ps targets. Of these, 104 paths required physical LAMMPS continuation
after first passage and 40 already contained 24 ps. A worker-finalization bug had
left the 104 composed, checksum-valid 81-frame binaries without `outcome.json`:
numpy memmaps were still open when an NFS intermediate directory was deleted.
`recover-completed-compositions` now accepts only that exact cleanup failure,
verifies the composed binary and immutable source outcome, and records that no
frames were reconstructed and LAMMPS was not rerun. The campaign then passed its
strict summarizer. Future workers preserve the small intermediate binary and
delete only the large text dump.

The atlas loader now consumes this producer directly and preserves its explicit
source-run splits: 26 parents for optimization, four for model selection, and six
for final validation. All experiments use 512 deterministic central atoms, 17
spatial tokens, the original frozen GeoFrameV2 checkpoint, and the fixed RFF
kernel from the 480-branch experiment. This makes the future-law distance scale
comparable while moving to substantially harder, transition-balanced parents.

The 36-parent position-only atlas reaches final-validation target R2 = 0.629. On
the valid matched-retrieval subset--the two independent 400 K transition parents--
its future-law distance is 0.45545 versus 0.45712 for static PCA, a +0.364% gain.
VAMP is 3.676% worse than static PCA, whereas the empirical conditional-law oracle
is +20.07% better. The RFF approximation remains accurate (Spearman 0.990 against
exact biased MMD over 512 sampled pairs). The uncertainty estimate is narrow over
local states but has only two independent source runs, so it must not be read as a
broad source-population confidence interval.

Past context is unavailable for the six liquid/crystal controls because those
controls were deliberately selected at source frame zero. The fair history
comparison therefore uses exactly the 30 transition parents: 22 optimization,
four model-selection, and four final-validation parents, with 120 future paths.
The position-only target distance is 0.45662; adding the same 17 token identities
at t-12, t-9, t-6, and t-3 ps reduces it to 0.45515. This is a +0.320% direct gain
over the identical position model (source bootstrap +0.295% to +0.346%) and a
+0.437% gain over static PCA. Pairwise future-law Spearman improves from 0.3310
to 0.3351. The accepted new result is therefore the frozen-encoder history atlas,
not the position-only or VAMP representation.

Exposing the final GeoFrame block (198,970 parameters) directly to the repeated-
shooting conditional-law target does not improve model-selection loss: all three
seeds choose epoch -1. Its held-out retrieval is 0.45527, a -0.025% change versus
the frozen history atlas (95% interval -0.070% to +0.019%). This experimental
encoder is rejected. The remaining +20% oracle gap and split-shot correlation of
only 0.723 show that four futures per parent are still a noisy conditional-law
target; more compute cannot substitute for more independent futures and parents.

Reproduction configurations are:

- `configs/predictive_atlas_nested_fixed24_geoframe_v2_20260903.yaml`
- `configs/predictive_atlas_nested_fixed24_transition_geoframe_v2_20260903.yaml`
- `configs/predictive_atlas_nested_fixed24_history_geoframe_v2_20260903.yaml`
- `configs/predictive_atlas_nested_fixed24_transition_encoder_finetune_geoframe_v2_20260903.yaml`

The accepted model and metrics are under
`/home/ids/vmorozov/experiments/predictive_atlas_nested_fixed24_transition_history_geoframe_v2_20260903`.
The independent-source campaign is not yet admissible: at the final scan only 16
of 90 runs were complete, all in the 400 K optimization split, and no strict
`summary.json` existed.

## Finest-timestep GeoFrame stability audit (2026-09-04)

To determine whether the frozen teacher itself provides a smooth instantaneous
state, the original nested shooting campaign was re-encoded at its finest stored
cadence. LAMMPS integrates at 3 fs, but the trajectory stores the early segment
every 10 steps, so the finest observable interval is 30 fs. The exact audit uses
36 parents, all 144 four-sibling branches, 32 fixed atom IDs, and 11 frames from
0 to 0.30 ps: 50,688 local environments in total.

Both the checkpoint's VICReg-projector output and its direct GeoFrame encoder
output are exactly deterministic under repeat encoding and point permutation.
Independent rotations change either representation by less than 3e-6 of its
median cross-state distance. The temporal changes below are therefore physical.

| Representation | Lag | Mean cosine | Median drift / cross-state median | Same-atom top-1 |
|---|---:|---:|---:|---:|
| VICReg projector | 0.03 ps | 0.9284 | 0.773 | 8.57% |
| GeoFrame encoder output | 0.03 ps | 0.9910 | 0.841 | 18.27% |
| VICReg projector | 0.30 ps | 0.9013 | 0.919 | 5.10% |
| GeoFrame encoder output | 0.30 ps | 0.9882 | 0.954 | 6.60% |

Same-atom retrieval searches among 32 within-branch candidates, so chance is
3.125%. Cross-state distances use different transition parents matched by
temperature. The direct encoder has high cosine because the entire representation
is angularly concentrated: even different-parent states have mean cosine 0.987.
Its high absolute cosine therefore does not imply strong dynamical identity.

At 30 fs, atom-matched local coordinates have moved by 0.249 A RMS while 98.1%
of the initial nearest-160 atom set remains. At 0.30 ps the corresponding values
are 0.640 A and 95.3%. Median embedding drift reaches approximately 90--95% of
the cross-state scale by 0.09--0.15 ps, and sibling futures diverge nearly as
quickly as a branch moves away from its parent. The frozen teacher is invariant
and numerically reproducible, but one-frame embeddings retain substantial fast
thermal motion and should not be assumed to form a smooth dynamical coordinate.

This motivates short-history pooling or a shooting-law-aware thermal-denoising
objective as the next controlled ablation. Smoothness alone is not the selection
criterion; the stabilized representation must improve held-out conditional
future-law prediction and retrieval.

Artifacts are in:

- `output/geoframe_temporal_stability_comparison_finest_30fs_20260904/`;
- `output/geoframe_temporal_stability_finest_30fs_20260904/`;
- `output/geoframe_encoder_temporal_stability_finest_30fs_20260904/`.

The runner, comparison, configurations, and core metrics are respectively in
`scripts/analyze_geoframe.py stability`,
`scripts/analyze_geoframe.py compare-representations`,
`configs/geoframe_temporal_stability_finest_30fs_20260904.yaml`,
`configs/geoframe_encoder_temporal_stability_finest_30fs_20260904.yaml`, and
`src/temporal_vamp/geoframe_stability.py`.
