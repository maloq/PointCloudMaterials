# Variant C: hot/relaxed MACE — 2026-09-09

## Recovery — September 9, 15:18 Paris

The initial queue failed during data preparation, before any training update.
Al 175 ps frame 24 reached the 10,000-iteration limit at fmax 0.13385 eV/Å.
The first 17 completed paired frames are retained. The failed frame and prior
status/configuration files are archived under `recovery_20260909_1518/` in the
first run output; original registry execution records remain intact.

The current configuration increases the CG budget to 100,000 iterations,
500,000 force evaluations and a 3,600-second per-frame timeout. The generating
potential, fixed cell and 0.01 eV/Å convergence requirement are unchanged.
Previously converged frames remain valid and are reused. No unconverged frame
is accepted. Both queues are detached on current H100 allocation 986459,
ending September 10 at 06:00:55 Paris, with a 05:30:55 safety deadline.
The 12-epoch/TDA6 workflow runs first, followed by the fresh 24-epoch/TDA1
workflow after its training and analysis finish. Online W&B initializes at
training start; no training result is claimed during data preparation.

The original launch specifications below record the first attempt. Recovery
uses the maintained runner and this explicit specification:

```bash
python scripts/experiment_registry.py run --spec experiments/mace_thermal80_20260909/recovery_run_spec.json
```

Live queue status stays in the original output root. Recovery launch and
execution metadata are stored separately in `recovery_20260909_1518/`.
These added specifications and notes are experiment records; generated retry
logs and verification files are output artifacts. No new runner was introduced.

Question: can shared embeddings of instantaneous and relaxed configurations
retain structural topology while suppressing thermal fluctuations?

This pilot uses the same small pretrained MACE and full 80-atom mean-pooling
architecture as plain80. The existing trainer supports the explicit `thermal80`
protocol; no separate trainer or legacy implementation is introduced.

## Data and physical protocol

Use ordinary Al/Mg/Ta continuation sources only, first stored anchor per source:
Al 166/170/174/175 ps ancestors for training, 177 ps for validation;
Mg 940/960/980/990 ps ancestors for training, 1000 ps for validation;
Ta uses disjoint center IDs and a later validation time in the same source.
Al shooting branches are excluded from this pilot.

| Material | Train anchors/epoch | Validation anchors |
|---|---:|---:|
| Al | 16,384 | 512 |
| Mg | 8,192 | 256 |
| Ta | 2,048 | 256 |
| Total | 26,624 | 1,024 |

Twelve uniform epochs, batch 1,536, microbatch 512: 18 updates/epoch,
216 updates total, 319,488 anchor exposures and 1,916,928 training view exposures.
This pilot is not data/budget matched to the larger plain80 run. Source precision
also differs where the old local caches preceded global float16 conversion:
we rebuild all hot neighborhoods from the current complete frames so neighbor
identities and relaxation inputs have exact correspondence. Full-frame source
quantization is inherited, and the hot/relaxed contrast includes its removal.

Every required full periodic snapshot is minimized at fixed box with its
original generating potential: Al1.eam.fs, Mg1.eam.fs or Ta_Zhong2014 EAM.
LAMMPS CG uses infinity-norm force tolerance 0.01 eV/Å; incomplete or unconverged
frames fail loudly. No thermostat, pressure relaxation, ideal-crystal template,
or isolated 80-atom minimization is used. Potential checksums and final forces
are recorded per frame.

For each center, select 80 nearest atom identities in the instantaneous frame.
Use those same identities, centered on the same atom, in the relaxed frame.
Compute relaxed TDA from the stored centered float16 offsets. Both members of
the pair receive this identical target. Full relaxed snapshots are separately
stored as verified float16 artifacts; neighborhood extraction precedes this
coarser global quantization. Source text is removed only after verified storage.

## Objective and training

Each anchor has six training views: hot anchor/spatial/temporal and their three
relaxed partners. A fourth physical future view and its relaxed partner are
stored solely for frozen post-training prediction probes.

Loss = 25 × mean(spatial MSE, temporal MSE) + 25 × hot/relaxed MSE
+ 25 × variance-floor penalty + covariance penalty + TDA MSE from epoch six.
Spatial/temporal MSE averages the hot and relaxed domains; variance/covariance
averages all six views, keeping one shared regularization budget. TDA head has
zero LR and no gradients for five epochs. All six inputs predict relaxed TDA.
No forecasting or nuisance loss, teacher, adaptive loss weights, or element
balancing. TDA PCA is train-only; the fixed initial feature scaler fits both
hot and relaxed training anchors.

AdamW encoder/head peak LR 1e-4/1e-3, weight decay 1e-5, gradient clip 5.
Per-update cosine LR, one-epoch encoder warmup; half-epoch TDA head warmup at
its activation. Compensated BF16 radial matrices and exact gradient replay.
Select lowest fixed validation loss among epochs 6–12.

## Reproduction and execution

Activate conda `pointnet`, then:

```bash
python scripts/experiment_registry.py run --spec experiments/mace_thermal80_20260909/run_spec.json
```

The existing allocation-aware queue prepares the paired cache, performs the
real-MACE preflight, trains, fits frozen probes, then runs the existing full
static Al analysis using encoder output. Allocation 984861, node53 H100;
safety deadline 05:25 Paris on September 9. Online W&B is required.

`python -m src.data_utils.mace_relaxed --config .../training.json` runs just data
preparation. `python scripts/convert_trajectory.py relaxation FRAME_DIR
--delete-source` verifies and stores an already-converged relaxed frame.

The GPU LAMMPS build uses the official stable_22Jul2025_update4 source and bundled
Kokkos, CUDA 12.9, `BUILD_MPI=OFF`, `PKG_KOKKOS=ON`, `PKG_MANYBODY=ON`,
`Kokkos_ENABLE_CUDA=ON`, `Kokkos_ARCH_HOPPER90=ON`, Release mode and bundled
`lib/kokkos/bin/nvcc_wrapper` as C++ compiler. Build with CMake and copy `lmp` to
the configured output bin directory. Source/binary checksums and build logs are
in the output directory. CPU and GPU starting energy/fmax agree numerically.
The full million-atom Al preflight converged in about 75 seconds.

## Results and code ownership

[Queue/status](../../output/mace_thermal80_20260909/status.json) ·
[Preparation](../../output/mace_thermal80_20260909/data/status.json) ·
[W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/therm809)

Eleven focused tests passed before launch, including unchanged plain80 behavior,
fixed atom identities when neighbors exchange, and six-view cached/direct
gradients on both sides of TDA activation. The real-MACE preflight is required
before training; no trained-result claim is made during preparation.

Maintained implementations: `src/simulation/relaxation.py`,
`src/data_utils/conversion/relaxation.py`, `src/data_utils/mace_relaxed.py`, and
small extensions of the existing trainer/objective/preflight/analysis modules.
This directory contains experiment records/configs. Logs, generated LAMMPS
inputs, relaxed frames, caches, binaries, checkpoints and diagnostics are
output artifacts, not new maintained launch scripts.

Energy minimization can change liquid motifs and basins. Evaluate displacement,
hot/relaxed consistency, held-out topology and future probes, and spatial
interface contrast. Static Al includes training ancestors and is descriptive.
No PTM/HCP label is used as definitive ground truth. Later neighbor reselection
and minimization-tolerance studies are needed before claiming noise invariance
or precursor preservation.

Launch verified at 01:29 Paris: detached queue is preparing data; training is
pending complete paired data and the real-MACE preflight. The first paired
frame passed convergence, identity and storage checks. Mean local-offset change
was 1.05 Å, so this relaxation is not merely tiny jitter; precursor retention
must be assessed. See [launch verification](../../output/mace_thermal80_20260909/launch_verification.json).

Recovery verified at 15:21 Paris: the failed Al frame converged after 11,723
iterations in 158 seconds, fmax 0.0066689 eV/Å. Paired extraction and verified
float16 conversion completed; preparation advanced to frame 40. Eleven focused
training/numerical tests passed. Both detached queues remain active. GPU
preflight and actual training are still pending completion of the paired cache.
[Recovery evidence](../../output/mace_thermal80_20260909/recovery_20260909_1518/recovery_verified.json).
