# Predictive representation training from shooting ensembles

## Why this is called shooting

The terminology comes from transition-path sampling. A configuration near a
transition is selected as a parent and several trajectories are *shot forward*
from it after changing momenta or stochastic forces. The repository campaign is a
position-conditioned forward-shooting ensemble: sibling branches share the exact
parent positions and periodic cell, draw independent Maxwell-Boltzmann momenta and
Langevin noise, and then evolve for 48 ps.

Consequently, siblings are samples from a future distribution conditioned on the
parent positions and temperature. They are not repeated futures of an identical
phase-space state because their initial velocities differ.

## Binary trajectory input

Shooting training is binary-only. Each complete branch must advertise a
`pointcloudmaterials.shooting_trajectory` artifact named
`trajectory_binary_float32` in `outcome.json`. The arrays are standard `.npy`
files opened with read-only memory maps: positions and velocities have shape
`[frame, atom, xyz]`, while timesteps, periodic box bounds, atom IDs and atom types
are separate typed arrays. An unmigrated complete branch is an error; training never
falls back to parsing `trajectory.lammpstrj`.

`ShootingBinaryEnvironmentDataset` selects exact timesteps and central atom IDs,
constructs PBC-correct local/context clouds in data-loader workers, and prefetches
ordered branch batches while the previous batch is encoded on the GPU. Position-only
loading avoids paging the equally large velocity array for the frozen geometric
encoder. Each batch also carries the campaign/branch/parent/run identity, split,
temperature, phase, random seeds, parent crystallinity, source and branch times,
atom IDs/types, and PBC-correct center coordinates. Direct velocity consumers use
`ShootingBinaryTrajectory.load_frames`.

The two throughput controls are independent:

```yaml
embedding:
  point_cloud_batch_size: 2048   # point clouds per GPU encoder call
  environment_batch_size: 2      # trajectories collated per loader batch
  environment_num_workers: 4     # CPU mmap/KD-tree workers
```

ASCII parsing remains only in `shooting_text_conversion.py`, which is used by the
one-time conversion/migration scripts. The measured five-frame read benchmark on a
70,304-atom branch was 3.27 ms from float32 memory maps versus 442 ms from text
(135x faster), before neighborhood construction.

## Current experiment

`configs/shooting_predictive_geo_frame_70304_400K.yaml` uses the authoritative IDS
campaign directly. It never copies trajectories to the repository filesystem. A
branch is admitted only when its `outcome.json` has `state: complete`, its metadata
matches `manifest.json`, and its trajectory and restart remain nonempty with the
recorded sizes. `status.json` and `interrupted_attempt_*` directories are not data.

The first run deliberately selects only 400 K. This is presently the only completed
temperature block containing eight-branch parent ensembles in both source train and
validation splits. It contains 12 parents and 96 branches:

- eight source-training parents and four source-validation parents;
- two parent times per source run, 12 ps and 3 ps before detected nucleation;
- eight independent futures per parent;
- 64 deterministic central atom IDs per parent;
- future horizons of 6, 12, 24 and 48 ps.

Within the eight source-training parents, source velocity seed 35863 is reserved for
early stopping. The official validation source seeds 35869 and 35879 are untouched
until final evaluation. Both parent times and all sibling branches inherit the split
of their source run.

## Representation and objective

The frozen epoch-159 GeoFrameTransformer encodes the parent state exactly as in the
context-VAMP experiment: a 128-dimensional central embedding plus the mean and
standard deviation of eight satellite embeddings, giving 384 input features. Future
states use the central 128-dimensional embedding so the target remains local.

For each parent atom and horizon, the eight sibling future embeddings are averaged.
The training target is the first eight PCA coordinates of these ensemble-mean futures
at all four horizons. A small MLP maps the 384-dimensional parent feature through a
16-dimensional bottleneck and a linear prediction head. Its loss combines

1. mean-squared multi-horizon future prediction; and
2. agreement between pairwise bottleneck distances and pairwise ensemble-future
   distances.

The prediction head is not a point-cloud decoder. It is used only to force the
bottleneck to retain information about the conditional future. Model selection is
repeated for three random initialization seeds. Ridge regression, parent-context
PCA and the raw context embedding are retained as baselines.

The principal evaluation retrieves neighbors from a different source MD run at the
same temperature and the same parent phase (`pre_nucleation_12ps` or
`pre_nucleation_3ps`). It compares the distance between their branch-averaged future
embeddings. Thus thermal branch noise is averaged rather than counted as predictable
error.

## Run

```bash
conda activate pointnet
python scripts/run_shooting_predictor.py \
  --config configs/shooting_predictive_geo_frame_70304_400K.yaml \
  --stage all
```

Stages `extract` and `train` allow the expensive frozen-encoder pass to be reused.
All derived caches, checkpoints, logs, coordinates and plots are written below
`/home/ids/vmorozov/experiments/shooting_predictive_geo_frame_70304_400K_20260831`.

When the complete campaign has validated parent ensembles in both source splits,
extend `data.temperatures_K` to `[400, 450, 500]`. Mixing incomplete temperature
blocks into the first run would confound temperature with train/validation identity.

## First 400 K result

The tuned run selected initialization seed 33 using only the held source seed's
prediction-plus-geometry loss. On the untouched four-parent validation set, the
neural model has standardized future-prediction R2 = 0.078, compared with -0.008
for the optimization-set mean and -0.173 for selected ridge regression. Broken down
by horizon, neural R2 is 0.046, 0.253, 0.034 and -0.060 at 6, 12, 24 and 48 ps.

For cross-run, temperature/phase-matched future-neighbor retrieval, lower distance
over matched random is better. At 12 ps the predicted future reaches 0.8331 and the
16-dimensional bottleneck 0.8347, versus 0.8385 for the full context encoder and
0.8356 for its 16-dimensional PCA. At 24 and 48 ps the static context remains better.
The combined-horizon bottleneck score is 0.8336 versus 0.8270 for the context.

This is evidence for a learnable conditional-future signal, especially near 12 ps,
but not yet a generally better predictive geometry. The principal limitation is the
number of independent inputs: eight sibling shots reduce target noise, but the
optimization set still has only six parent configurations (three source runs at two
parent times), each sampled at 64 atoms. The measured between-parent/atom fraction
of total future variance is 0.67, 0.67, 0.70 and 0.53 across the four horizons, so
the weak result is not explained by branch noise alone.

Exact aggregate and retrieval metrics are in `metrics.json`; concise per-horizon
prediction and branch-variance diagnostics are in `diagnostics.json` under the IDS
experiment directory.

## Full-data encoder comparison

The September 1 comparison merges three independently seeded shooting campaigns
whose parent coordinate hashes, source provenance, atom count and simulation
protocol are identical. Their velocity/thermostat seed pairs are disjoint:

- the original eight-shot campaign: 320 branches;
- the completed one-shot campaign: 40 branches;
- the completed two-shot follow-up: 80 branches.

This gives exactly 11 futures for each of 40 parents, or 440 branches. Configuration
guards require both counts, so a partially completed campaign cannot silently enter
the run. The newer `40branches_local_20260901` campaign is not included because it
was incomplete when this comparison snapshot was defined.

The matched configurations are
`shooting_predictive_geoframe_v1_440branches_20260901.yaml` and
`shooting_predictive_geoframe_v2_factor_sn_440branches_20260901.yaml`. They differ
only in output directory and encoder checkpoint. The V1 run uses the original
epoch-159 GeoFrameTransformer. The V2 run uses epoch 34 from the factor-VAE,
spectral-normalization training directory because repository static evaluation
identified it as that run's best checkpoint. Both export the checkpoint-selected
128-dimensional VICReg-projector representation.

### Full-data result

Both detached runs completed on 40 parents, 440 branches and 64 atom IDs per
parent. Held-out standardized future-prediction R2 was:

| encoder | all | 6 ps | 12 ps | 24 ps | 48 ps |
| --- | ---: | ---: | ---: | ---: | ---: |
| GeoFrame V1 | 0.033 | 0.085 | 0.058 | 0.022 | -0.074 |
| GeoFrame V2 factor/SN | **0.108** | **0.167** | **0.151** | **0.067** | **-0.009** |

A temperature/phase group-mean baseline, fitted only on optimization parents,
obtains R2 = 0.117 for V1 targets and 0.048 for V2 targets. Therefore the V1
predictor does not beat metadata-level progression. V2 does, but within-temperature
R2 is 0.131 at 400 K, -0.073 at 450 K and -0.091 at 500 K. The strongest resolved
cell is 400 K, 12 ps before nucleation, where V2 obtains R2 = 0.243.

The learned metric does not beat static PCA generally. Relative change in matched
future-neighbor distance for bottleneck versus its own static 16-dimensional PCA
(positive is better) is:

| encoder | 6 ps | 12 ps | 24 ps | 48 ps | all horizons |
| --- | ---: | ---: | ---: | ---: | ---: |
| GeoFrame V1 | -4.00% | -4.34% | -5.33% | -4.84% | -4.16% |
| GeoFrame V2 factor/SN | **+0.15%** | -1.50% | -1.11% | -3.87% | -2.04% |

A cross-encoder evaluation uses the concatenated, standardized V1 and V2 future
signatures as one common target. V1 static PCA is still the best of the compared
spaces at every horizon. Thus V2 is a materially better input for future regression,
but this mean-target bottleneck still fails to turn that signal into a useful nearest-
neighbor geometry.

With 11 shots, estimated reliability of each ensemble-mean target is 0.82--0.91
for V1 and 0.83--0.86 for V2 across horizons. Insufficient shot averaging is no
longer the primary limitation. Exact self-target, temperature, phase and common-
target results are saved in the joint `comparison.json` artifact.

## Tests

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. conda run -n pointnet \
  pytest -q tests/test_shooting_predictor.py tests/test_temporal_vamp.py
```

The tests cover strict outcome admission, conversion-only text parsing with
velocities, float32 memory maps, multiprocessing binary environment loading,
periodic local environments, sibling grouping and recovery of a synthetic
multi-future conditional mean.

## Ordered predictive-state ablations (September 1)

The next experiment series uses the same factor/SN GeoFrameV2 checkpoint, the same
40 parents and 440 complete branches, and the same source-run split. The primary
score is held-out cross-run neighbor retrieval with exact temperature and parent
phase matching. Its target is future *change* rather than the absolute future.

Run the stages in order:

```bash
conda activate pointnet
python scripts/evaluate_shooting_ablation0.py \
  --experiment-dir /home/ids/vmorozov/experiments/shooting_predictive_geoframe_v2_factor_sn_440branches_20260901 \
  --output-dir /home/ids/vmorozov/experiments/shooting_ablation0_future_change_geoframe_v2_20260901
python scripts/run_shooting_multiscale_ablation.py \
  --config configs/shooting_multiscale_ablation1_geoframe_v2_20260901.yaml
python scripts/run_shooting_spatial_ablation.py \
  --config configs/shooting_spatial_ablation2_geoframe_v2_20260901.yaml
python scripts/run_shooting_distributional_ablation.py \
  --config configs/shooting_distributional_ablation3_geoframe_v2_20260901.yaml
python scripts/run_shooting_geometry_ablation.py \
  --config configs/shooting_geometry_ablation4_fixed_geoframe_v2_20260901.yaml
```

Ablation 1 caches 16 individual satellite embeddings, their PBC-correct offsets,
q4/q6 and first-shell size. It evaluates local, q-augmented, old eight-satellite,
new 16-satellite mean/std, and three-scale radial features. Wider context raises
held-out future-change PCA R2 from 0.633 for local Ridge to 0.795 for 16-satellite
mean/std Ridge. It nevertheless makes raw Euclidean retrieval worse, establishing
that the added context has predictive information but not the desired geometry.

Ablation 2 replaces averaging with a 425k-parameter, two-block invariant context
transformer. Pairwise distances between the 17 PBC-correct token offsets provide
the attention bias. Its validation R2 is 0.701, and its representation still loses
to local PCA in retrieval. More spatial model capacity alone is therefore not the
answer on this dataset.

Ablation 3 is the first accepted retrieval improvement. Each branch change is
projected to 16 optimization-fitted PCA modes, mapped through 128 random Fourier
features at 0.5, 1 and 2 times the optimization median bandwidth, and averaged over
all 11 sibling shots. The predicted kernel mean beats local PCA by 1.84%, 1.28%,
1.05% and 1.15% at 6, 12, 24 ps and jointly. Split-shot kernel-mean correlations
are 0.927, 0.910 and 0.914, so the distributional target is reproducible with the
current number of shots. The improvement is real but remains below the registered
5% success threshold.

The first scratch ablation-4 run is retained for audit but is invalid: adaptive
student temperature allowed representation scale to collapse (mean coordinate
standard deviation 0.008), destroying prediction while preserving tiny-distance
rankings. The corrected run fixes temperature from the non-collapsed ablation-3
state, strengthens VICReg variance, and warm-starts from ablation 3. Independent
selection rejects every geometry fine-tuning epoch, so the accepted checkpoint
remains ablation 3. Do not report the collapsed run's apparent 3.2% gain as a model
result.
