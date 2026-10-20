# Running and inspecting the Ni latent-flow queue

The [scientific protocol](../experiments/ni_latent_flow_20261018/README.md) defines
51 fits covering 17 treatments and three seeds. The
[active recipe](../configs/ni_latent_sim/flow_20261018.json) fixes the two-GPU,
12-hour queue. Read the run's actual receipts for current status.

## Launch verified on 2026-10-18

The queue launched at **02:09:38 Europe/Paris** on node60 in allocation 1045373.
Its fixed deadline is **14:09:38 Europe/Paris**. The two detached GPU workers
are recorded in
[`submission.json`](../output/ni_latent_flow/native-20261018-v1/technical/submission.json).
Both RTX PRO 6000 GPUs were verified training, with the linear-VAMP block model
past update 500 and the VAE block model past update 100; both had saved resumable
checkpoints and online W&B receipts. All 24 common birth-control readouts had
already completed. These startup checks are not a scientific result.

The GPU lane 0 worker briefly claimed the CPU controls before the dedicated CPU
worker acquired its lock, then moved to training after approximately 30 seconds.
This did not change scientific settings or require a job restart. The original
submission receipt and frozen source are preserved.

Before launch, real-data integration checks covered six model variants, final
checkpoint reloads, seeded sampling, native evaluation and interrupted/resumed
training. The resumed model, EMA and training history matched uninterrupted
training exactly. Debug runs remained local. Receipts are under
[`technical/preflight/`](../output/ni_latent_flow/native-20261018-v1/technical/preflight/).

## Dependencies and preparation

Use conda `pointnet-torch214`. The existing e3nn installation requires the
repository's established `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` environment setting
for its trusted packaged constants. The queue automatically supplies that setting
and conservative BLAS/OpenMP thread counts to frozen workers.

The producer reuses the sealed `ni_kinetic_latent/native-20261015-v1` normalized
native bank and PTM labels, the completed `ni_spatial_scales/radius-20261015-v1`
regional cache, the original birth-history measurements and declared source
checkpoints. It verifies source-plan and payload hashes. This workflow does not
collect new MD or recalculate the 244/87 measurements. Storage tokens resolve
through the project's `machine.local.yaml`; see [portability](portability.md).

Preparation binds the full 14,400-window cohort, roles, source identities,
native timelines and checkpoint provenance into `technical/plan.json`. Existing
training/test ancestry and central normalization remain unchanged. A different
configuration requires a new run revision; do not edit a submitted source
snapshot or overwrite its plan.

From the repository root:

```bash
conda activate pointnet-torch214
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -m src.research.ni_latent_flow.queue prepare \
  --config configs/ni_latent_sim/flow_20261018.json
```

The configured output is
`${storage:training_storage}/ni_latent_flow/native-20261018-v1`; the configured
cache namespace is `${storage:cache}/ni-latent-flow/native-20261018-v1`.
Source arrays remain in their existing sealed parent caches. The run stores
training checkpoints, analysis arrays, diagnostics and provenance in its own
output tree.

## Detached launch and allocation boundaries

With two allocated GPUs visible, launch once:

```bash
python -m src.research.ni_latent_flow.queue launch \
  --config configs/ni_latent_sim/flow_20261018.json
```

Launch verifies the new metric family, freezes `src/`, `configs/`, metric
documents and machine mapping into `technical/source/`, then starts detached
workers. A dedicated CPU process starts the common birth controls early, while
two GPU workers claim independent model stages. The queue does not depend on a
chat wakeup or scheduled review. `technical/submission.json` records the actual
worker PIDs, device assignment, source snapshot, allocation and global deadline.
A submission receipt prevents accidental duplicate launch.

The 12-hour deadline is measured from launch. Workers also respect their actual
Slurm allocation boundary, with shutdown/checkpoint reserves. If the allocation
ends before the global queue deadline, an owned continuation may be submitted
to the combined alternative partitions `RTX6000PRO,A100,A40,L40S`. These are
alternative compatible GPUs, not a request to occupy each GPU type. Continuation
receipts are `technical/continuation-<lane>.json`; they preserve the frozen run
and its original deadline.

Each stage has at most two execution attempts. Per-arm locks prevent workers
from fitting/evaluating the same arm concurrently. Exit75 means a resumable
deadline pause, not successful completion. A hard deadline terminates only the
process group created for that stage; no unrelated simulation/job is stopped.
After a failed independent fit, workers continue to other eligible work. Readout
and comparison failures remain visible.

## Training and completion receipts

For an arm such as `linear_block-s20261025`, inspect
`analyses/linear_block-s20261025/technical/`:

| Artifact | Meaning |
| --- | --- |
| `progress.json` | Latest saved stage/update, target budget and update timing |
| `last.pt` | Resumable model, optimizer, RNG, stage and predictor EMA state |
| `final.pt` | Fixed-budget final checkpoint; block inference uses its EMA predictor |
| `complete.json` | Training completed; binds checkpoint SHA and completed stages |
| `wandb.json` | Stable resumable online run identity in teshbek/PointCloudMaterials |
| `train-*.log`, `evaluate-*.log`, `births-*.log` | Execution output for individual attempts |
| `<stage>-attempt-*.json` | Attempt PID, command, timing and exit code |
| `<stage>-failed.json` | Stage exhausted its bounded execution attempts |
| `execution-*.json` | Allocation, hardware, precision, input dimensions and resume point |

Scientific training stays online in W&B. Local benchmarks and frozen readouts
do not create training runs. Final evaluation summaries update the recorded
scientific run. Authentication/network errors are not silently changed to an
offline training mode.

Inherited/direct arms train their predictor for 3,200 updates. Twelve new
SelfTICA/FMRC arms additionally train 12 representation epochs and 12 fresh
decoder epochs, each 9,180 updates at batch256. The new encoders are then frozen
for predictor fitting. This difference is intentional and recorded, rather than
pretending that all 51 arms have the same total compute.

## Independent evaluations and useful outputs

Native evaluation uses `assessment.evaluate`; birth evaluation uses
`assessment.evaluate_births`. A trajectory failure does not prevent a completed
encoder from receiving its independent birth readouts. Exactly 21 representative
encoder fits are declared for that assay; unchanged present features in other
predictor variants receive a skipped/reuse receipt rather than duplicate probes.

`technical/evaluation-complete.json` is the shared native assessor receipt. It
can record a numerical generation failure and may precede completion of the
new all-path validity accounting. **The queue's native completion marker is
`technical/flow-evaluation-complete.json`.** The adapter continues counting every
requested path even after a numerical rollout failure, unless it must pause at
the deadline. `validity-partial.pt` and `validity-progress.json` retain that work.
Native scoring never replaces invalid trajectories with successful resamples.

The independent birth marker is `technical/birth-evaluation-complete.json`.
Read its state: an unchanged-feature arm may intentionally skip a new fit.
Common-control completion is
`analyses/controls/technical/controls-complete.json`. All 19,800 birth rows remain
in the existing assay; no new neighborhood filter is applied.

Per-arm scientific outputs include:

- `tables/training.csv`: stage objectives and optimizer diagnostics.
- `tables/retention.csv` and `latent_spectrum.csv`: information retained in the
  present representation, with continuous-order and phase-readout diagnostics.
- `tables/trajectory.csv`: connected-path, increment and correlation scores,
  temporal spread, coverage, local ordering and local persistent crystal entry.
- `tables/validity.csv`: all-path numerical/necessary-measurement-bound failures
  and excursions outside training support.
- `plots/order_trajectories.png`, `latent_order.png`, `local_crystallization.png`:
  native visual diagnostics when the corresponding evaluation is defined.
- `technical/likelihood-availability.json`: the flow objective is not relabeled
  as a teacher-forced likelihood; zero-noise and MDN-mean controls are distinguished.

Nucleus predictions and readouts live under
`births/analyses/<arm>/<space>/<task>/lead-<lead>ps/`. Paired source-level native,
validity and birth summaries are exported to `analyses/comparison/tables/`.
Each numerical table binds [the metric definitions](metrics/ni_latent_flow.md)
and implementation hashes. Follow the common
[result publication workflow](research_results_system.md) to expose existing
outputs without recomputing or relabeling them.

## Status and resumption

At run level, inspect `technical/worker-<lane>.log`, `lane-<lane>.json`,
`worker-start-*.json`, `worker-exit-*.json` and `queue-finished.json`.
The final receipt distinguishes pending stages and comparison completion.
An elapsed deadline with unfinished arms is a partial campaign; it is not
evidence that every declared treatment was evaluated. Generation failure is
likewise a scientific outcome to report alongside available retained-information
and birth diagnostics, not a successful physical trajectory.

For explicit later resumption, use the recorded frozen configuration, not the
mutable repository recipe. On an authorized allocation, a worker command has
the following form; replace the placeholder with the actual run directory:

```bash
cd /absolute/run/directory/technical/source
export PYTHONPATH="$PWD"
export PCM_PROJECT_ROOT="$PWD"
python -m src.research.ni_latent_flow.queue worker \
  --config config.json --lane 0
```

It honors the original submission deadline and will not silently extend a
finished 12-hour campaign. Extending a campaign or changing its budget requires
a declared new execution decision. Individual `train`, `evaluate`, `births`
stages also accept `--name` and an absolute Unix `--deadline`; use the frozen
configuration and preserve ownership/locks when scheduling those stages.

To aggregate already completed stages, use the same frozen configuration:

```bash
cd /absolute/run/directory/technical/source
export PYTHONPATH="$PWD"
export PCM_PROJECT_ROOT="$PWD"
python -m src.research.ni_latent_flow.queue compare \
  --config config.json
```

This does not train a model or choose a checkpoint. It retains missing seeds,
undefined comparisons and recorded generation failures. A comparison exception
is recorded as `technical/comparison-failed-*.json` and must not be reported as a
completed scientific comparison.

The existing local model predicts independently sampled measurement paths. It
does not evolve coupled atomic coordinates, generate a spatially connected
nucleus, or establish natural nucleation rates. A SupraTITO-inspired collective
stage remains a separate deferred experiment; see the scientific protocol.
