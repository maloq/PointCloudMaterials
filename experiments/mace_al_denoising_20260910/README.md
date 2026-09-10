# Al topology denoising — September 10, 2026

The [cross-experiment problem review](../../docs/encoder_tda_relaxation_problems_20260910.md)
consolidates encoder, TDA, relaxation and evaluation issues, with confirmed
findings separated from open questions.

## Current state: available-data results and detached work

The matched audit completed at 14:31 CEST. The subsequent
[detailed potential comparison](POTENTIAL_DIFFERENCES.md) separates force-field,
structural, TDA and predictor effects. On the same nine early source frames,
the endpoint displacement averages 0.956 Å and the first-12 identity overlap is
82.2%. EAM produces narrower first shells and more local ordered motifs, while
both quenched states remain predominantly disordered. A global mean shift
accounts for 81% of the squared TDA difference; a source-held-out offset reduces
EAM prediction error from 0.849 to 0.132, but local descriptor correlations are
only 0.18–0.20. Full-cell force evaluations confirm that each potential prefers
its own endpoint. The 9× uncorrected prediction-error ratio should therefore
not be read as a 9× structural difference or proof of physical superiority.
`analyze_potential_differences.py` and the report are experiment records;
derived plots, metrics, arrays and generated force-evaluation inputs/logs are
in `available_mixed/potential_audit/deeper_analysis/`.
At 16:10 CEST all 90 full-data target frames had finished and job 988317 was
caching their frozen MACE features. The launch history follows below.

The user authorized mixed potentials and requested using prepared data immediately.
The 90-frame preparation was stopped at nine completed FIRE frames, with the
partial tenth relaxation archived. `training_available.json` freezes fourteen
existing shards: nine distinct MEAM frame contexts (preferring the original CG
target in the five duplicate contexts) and five older Al1 EAM/FS continuations.
No input is duplicated and targets are not averaged. The records preserve each
potential, minimizer, source lineage and actual history spacing.

The available-data split has 13,056 training, 4,864 validation and 1,280 test
neighborhoods. MEAM sources 000/001/002 are assigned train/validation/test before
fitting. EAM continuations 166/170/174 ps train, 175 ps validate and 177 ps test.
This is exploratory: the EAM continuations share an ancestor, and 177 ps was used
as validation in earlier experiments. Potential also covaries with temperature
(EAM 650 K, MEAM 400 K) and snapshot spacing (0.1 versus 0.75 ps). Temporal
attention now receives per-history physical times using a common 0.75 ps scale.
Exported mixed-cadence encoders require explicit `(B, T)` frame offsets in ps.

All 36 runs finished at 13:57 CEST, and initial analysis finished at 13:59 CEST.
The [available-data report](../../output/mace_al_denoising_20260910/available_mixed/analysis/RESULTS.md)
contains the full comparison. Balanced test MSE is 0.22570 for atom-temporal,
0.23151 for its matched atom-anchor control, 0.23485 for the TDA-only balanced
transformer, 0.40728 for mean pooling, and 0.68754 for the anchor MLP. The
source-averaged gain from atom histories over the atom-anchor control is 3.07%,
with an exploratory two-trajectory bootstrap interval crossing zero. The
relaxed-input ridge reference reaches 0.06980, so the target is decodable from
relaxed MACE features in this cohort. These results do not establish a universal
temporal advantage or a causal potential effect. Nineteen focused encoder/history
tests passed; the subsequent eight-test denoising run also covers the exact
paired statistical test.

`potential_audit.json` declares nine other independent melt sources, three each
at 400/450/510 K, unused in this available-data fit or evaluation. After training,
the same full cells are minimized with both potentials using FIRE and the same
force threshold. Both targets use the same centers and neighbor identities.
The primary test compares source-averaged prediction errors from the fixed
atom-temporal model using all 512 paired label swaps; intervals resample sources
within temperature. Secondary H0/H1/H2 tests use Holm correction. The prespecified
practical effect is 5% relative prediction error. This tests the potential used
to define the relaxed target on fixed MEAM-generated observations, not the effect
of changing the potential generating the trajectories. Existing CG/FIRE target
differences are reported separately for the five paired contexts.

The user subsequently requested detached work and continued use of allocation
988064. The potential audit uses a detached Slurm step within that existing L40S
allocation on node50; its configuration deadline is 18:55 CEST, before the
allocation ends at 19:13. The larger target preparation has its own L40S batch
allocation: job **988317**, running on **node39**, with a 12-hour limit and a
15-minute safety margin. It reuses the nine verified FIRE shards and prepares
the remaining contexts and frozen MACE cache. No additional audit allocation
is requested. Closing the IDE does not terminate the detached step; the existing
Slurm allocation must remain active for that step.
The audit started at 14:08 CEST as **988064.1**. Its orphaned `srun` client has
its own session, `/dev/null` input and a persistent log in
`available_mixed/detached_audit/audit.log`; `launch.json` records the exact command.

The batch launch uses the existing Slurm submission helper and family command:

```bash
PYTHONPATH=. python experiments/mace_al_denoising_20260910/submit_slurm.py \
  --plan experiments/mace_al_denoising_20260910/slurm.json --job prepare_full
```

Its generated job script, immutable config copy, runtime deadline config and
submission record are in `output/mace_al_denoising_20260910/fire/slurm/prepare_full/`.
The runtime deadline is computed when the batch job starts, so queue time does
not consume its budget. The potential-audit command is:

```bash
python -m src.training_methods.pretrained_mace \
  --config experiments/mace_al_denoising_20260910/training_available.json \
  --stage potential-audit
```

The new reuse producer and potential-audit implementation under `src/` are
maintained code behind the existing family command. The new configurations,
Slurm plan and submission recipe in this directory are experiment records.
Backend diagnostics, generated scripts, runtime configs, logs and outputs stay
under the corresponding output directories. The earlier chronology below is
retained to explain the original CG failure and FIRE recovery.

Research question: can several thermal snapshots denoise the topology of the
relaxed anchor better than one observation? Relaxed-input prediction is a
diagnostic reference. The user authorized the loss-only ablation, balanced
topology supervision, residual temporal fusion, atom-level temporal fusion and
use of the larger Al dataset.

## Data and split

`source_inventory.json` records 112 completed independently melted Al sources
available at selection time. `training.json` freezes 30 sources at 400, 450 and
510 K: six training, two validation and two test sources at each temperature.
These preserve the producer's optimization, model_selection and
final_validation splits. Sources are selected in manifest order without using
their crystallization outcomes. The 500 K pool did not yet have completed final
validation sources; 520 K did not have completed optimization sources.

Each source contributes frame indices 40, 240 and 640 (30, 180 and 480 ps), with
256 uniformly selected atom centers per frame: 13,824 training, 4,608 validation
and 4,608 test neighborhoods across 90 source/time contexts. Five observations
are spaced by the actual 0.75 ps storage cadence and span 3 ps. Every frame
uses the same hot-anchor-selected 80 identities. This differs from the earlier
EAM Al pilot's 0.1 ps cadence and cannot be compared by absolute score.

Targets use the source's Lee2003 Al MEAM potential and converged full-periodic,
fixed-cell CG relaxation at maximum force 0.01 eV/Å. Potential checksums must
match each campaign. The existing relaxation, pairing, history and persistence
image implementations are reused. Centered float16 neighborhoods are extracted
before global relaxed-position quantization. The maintained converter writes
verified float16 relaxed frames with float32 boxes and exact identities.
Source trajectories are read only. Relaxation text and conversion provenance
are retained in this output while the experiment is being validated.

## Architecture and controlled comparisons

MACE remains frozen in every run and is cached once. Pooled features are float32;
per-atom scalar features are cached as float16 with recorded quantization error.
MACE's two scalar blocks are exposed before pooling without changing its weights.
The frozen feature interface feeds the original temporal transformer directly.

The configuration declares twelve variants and three seeds before test analysis:

- Anchor and five-frame-mean MLPs with the existing 32 whitened PCA targets.
- The original 128-wide, two-block transformer with either 25/1 variance/covariance
  regularization or TDA supervision alone. These two differ only in their loss.
- The same transformer with balanced H0/H1/H2 supervision over the full 144D target.
- Anchor and mean MLPs with the balanced target, plus a continued anchor control.
- Anchor features plus a zero-initialized temporal correction.
- Temporal attention per atom identity followed by learned spatial pooling;
  its matched anchor-only control repeats current atom features in all slots.
- Relaxed-anchor features predicting their own TDA, as a diagnostic reference.

Residual and atom variants start from the selected anchor decoder. The continued
anchor control receives the same warm start and extra optimization budget. Each
run uses up to 32 epochs and patience eight, selected by validation topology
loss; the initial checkpoint remains eligible. All variants use the same shuffled
training examples and effective batch size, with accumulation for atom attention.
Only the regularized comparator includes representation penalties.

Block target scales use within-Al training variation, with a floor of 5% of the
largest block standard deviation added in quadrature. The three block MSEs have
equal weight regardless of their grid sizes. PCA and feature transforms also
fit training data only. Full test analysis is deferred until all configured
training runs finish. It reports per-homology/per-temperature errors, within-frame
R², source-bootstrap comparisons, and repeated-anchor/reversed-past interventions.

## Reproduction

Use the `pointnet` environment and an allocation with enough time remaining;
the dated safety deadline is explicit in the configuration.

```bash
python -m src.training_methods.pretrained_mace \
  --config experiments/mace_al_denoising_20260910/training.json --stage prepare
python -m src.training_methods.pretrained_mace \
  --config experiments/mace_al_denoising_20260910/training.json --stage train
python -m src.training_methods.pretrained_mace \
  --config experiments/mace_al_denoising_20260910/training.json --stage analysis
```

`--stage all` performs all stages in sequence. Completed cache shards and completed
variants are reused only under their recorded configuration. Interrupted training
is reported explicitly; it is not silently restarted. Existing frozen encoder
exports use `PretrainedMACEDenoising` and include their scaling and decoder weights.

The GPU MEAM binary is built from the same official stable_22Jul2025_update4
source as the preceding relaxation experiment, source SHA256
`411088d9c03339e025f6a975e0a5741bb9e3f351cc39eda220ab22ac318fe2fb`.
The CMake settings are Release, BUILD_MPI=OFF, BUILD_OMP=OFF, PKG_KOKKOS=ON,
PKG_MEAM=ON, Kokkos_ENABLE_CUDA=ON, Kokkos_ARCH_ADA89=ON,
CUDAToolkit_ROOT=/usr/local/cuda-12.1, CMAKE_CXX_COMPILER set to bundled
`lib/kokkos/bin/nvcc_wrapper`, and NVCC_WRAPPER_DEFAULT_COMPILER=/usr/bin/g++-11.
DOWNLOAD_POTENTIALS, WITH_GZIP, WITH_PNG and WITH_JPEG are OFF.
The CPU/GPU comparison uses all 70,304 atom force vectors of the same source
frame before any minimization, with absolute/relative tolerance 1e-8 eV/Å.
Build logs, the numerical audit and binary checksum are in the run output.

## Outputs and findings

Original output: `output/mace_al_denoising_20260910/`. The CG preparation attempt
failed after five of 90 completed frames. On source 001 at 400 K, frame 640,
the minimizer repeated energy -231793.067503096 eV and maximum force
0.089092840166581 eV/Å for thousands of iterations. The frame did not meet the
0.01 eV/Å requirement. Its exact LAMMPS child was stopped; the preparation and
dependent pipeline both failed explicitly. The stop audit and original logs are
retained. No training or held-out result exists.

The recovery runs on allocation 988064 (node50, L40S). The old Kokkos binary
only supports CG with quadratic line search;
attempted FIRE and quickmin diagnostics failed immediately as unsupported.
`training_fire.json` declares a separate FIRE protocol with a 0.001 ps initial
timestep, the same fixed cell, potential and force threshold, and a 900 second
per-frame limit. The binary is pinned to official `patch_2Sep2026`, with build
commands, checksums and logs in `relaxation_recovery_benchmark/`. This build uses
CUDA 13.2 and g++-13, retaining double precision and ADA89 targeting. Its source
SHA256 is `df89defbca87aad40f6c55ab6fca05b910089d831774219ba5a8ddbb9fa4c5dc`.
All 70,304 initial atomic force vectors agreed with the prior CPU calculation
within 2.7e-15 eV/Å. On the identical stalled input (verified by checksum), FIRE
converged in 74.22 seconds to fmax 0.00912141654 eV/Å and energy
-231795.453033821 eV. The disposable validation command and its result are
`relaxation_recovery_benchmark/validate_fire.py` and `validation.json`.
The six focused history/denoising tests passed after the configuration change.

At 13:25 CEST, the full recovery preparation → training → analysis pipeline was
relaunched using `run_spec_fire.json`. All 90 frames use FIRE; the five CG labels
remain separate. No new model training result is claimed yet. The recovery
output is `output/mace_al_denoising_20260910/fire/`. The safety deadline remains
18:55 CEST, before the allocation ends at 19:13. Completion time is uncertain.
The first recovery controller launch failed before starting the command because
its spec omitted the required empty `dependencies` list. That failed record is
retained under `fire/execution/`. The corrected controller records are under
`fire/controller/`, with the actual pipeline log in `fire/controller/command.log`.
Scientific data and configuration were unchanged by this launcher correction.

All 22 focused tests passed. A disposable check exercised all twelve predictor
variants on eight actual training-source histories with finite backward updates.
The three exported-fusion paths matched direct GPU inference within 1.2e-5.
The original bitwise GPU pooling check was too strict: repeated MACE evaluations
differed by up to 2.4e-7. The numerical-tolerance rerun passed, and both logs are
retained. These are implementation checks with discarded updates, not research
results. CPU/GPU MEAM forces agreed within 2.7e-15 eV/Å; the first full-cell
relaxation reached fmax 0.00964 eV/Å in 155 seconds.

The maintained launch command for the queued sequence is:

```bash
python scripts/experiment_registry.py run \
  --spec experiments/mace_al_denoising_20260910/run_spec.json \
  --wait-for-dependencies-until 2026-09-10T18:30:00+02:00
```

That spec references the failed immutable CG preparation attempt as its
dependency and is retained as an execution record; it is not an active queue.
The recovery uses the same maintained command with `training_fire.json` and
`--stage all`, tracked through `run_spec_fire.json`:

```bash
python scripts/experiment_registry.py run \
  --spec experiments/mace_al_denoising_20260910/run_spec_fire.json
```

Its eventual report will be
[fire/analysis/RESULTS.md](../../output/mace_al_denoising_20260910/fire/analysis/RESULTS.md).

New `src/data_utils/mace_denoising.py`, `src/models/encoders/mace_denoising.py`,
`src/training_methods/mace_denoising.py` and `src/analysis/mace_denoising.py` are
maintained implementations behind the existing family command. This directory
contains experiment records. Build artifacts, logs, checkpoints, cached features,
relaxation snapshots and measurements are generated run artifacts under output.
