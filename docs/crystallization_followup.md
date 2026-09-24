# Literature-guided frozen-MACE forecast queue

Use conda `pointnet-torch214` at the repository root. The workflow reads existing
original and relaxed caches, verifies their hashes, and stores each timeline once
per GPU worker. Dense descriptors are gathered from actual observed MD frames;
the original future-target tensor is never used to construct these inputs.

```bash
python -m pytest -q tests/test_crystallization_followup.py
python -m src.research.crystallization_followup.queue prepare --config configs/crystallization_transfer/literature_followup_20260922.json
python -m src.research.crystallization_followup.queue verify --config configs/crystallization_transfer/literature_followup_20260922.json
python -m src.research.crystallization_followup.queue submit --config configs/crystallization_transfer/literature_followup_20260922.json
python -m src.research.crystallization_followup.queue report --config configs/crystallization_transfer/literature_followup_20260922.json
```

`verify` needs one allocated GPU. It verifies the real150-source joins, shared
targets, unchanged inputs after future-target corruption, all12 forward/backward
paths and a complete one-epoch checkpoint/evaluation smoke. Its results are not
scientific comparisons.

`submit` uses two GPUs and12 CPUs in the current Slurm allocation. It freezes the
executable code and launches one detached `srun` step with two tasks and one GPU
bound to each task. File locks distribute fits dynamically. It does not cancel
the user's bash allocation or request new allocation time. The frozen code can
continue after workspace edits; implementation fingerprints must match validation.

Each worker checks the allocation end time, saves exact training resumes and exits
before expiration. The queue is finite. Continue unfinished work by running its
**frozen** `technical/code` module and config with `worker --lane NAME` in another
allocated GPU step, with `PCM_PROJECT_ROOT` pointing to that snapshot. Failed fits
require inspecting their recorded traceback before restarting.

Output: `output/crystallization_transfer/literature-followup-20260922/`.
`RESULTS.md` and `tables/comparison.csv` update after each completed fit.
`technical/lane-*.json`, `lane-*.log` and `runs/*/status.json` report progress;
checkpoints, predictions, full metrics and immutable population IDs remain under
`technical/`. Metric definitions and hashes accompany CSV exports. No hardware
benchmark is run as part of training and no WandB credentials/settings are changed.

Initial launch on22 September2026: allocation1004167, node58, two RTX PRO6000
Blackwell GPUs. A single two-task Slurm step binds one GPU to each worker;
`technical/launches.json` retains the exact launch command. The user's bash
allocation remains running. Final scientific interpretation is pending completion.

## Second wave

The same queue entry point accepts
`configs/crystallization_transfer/literature_optimization_20260922.json` (16 fits)
and `configs/crystallization_transfer/crystal_front_20260922.json` (12 fits).
The latter requires the registered `crystallization-observed-front-20260922` cache.

The detached front producer runs:

```bash
python -m src.research.crystallization_followup.front --config configs/crystallization_transfer/crystal_front_20260922.json --after output/crystallization_transfer/literature-optimization-20260922
```

It derives current and3 ps past descriptors from existing MD using eight CPU
processes, retaining checked per-frame and per-source receipts for continuation.
It then waits for the optimization queue, runs the full front GPU preflight, and
submits the two-GPU front queue only if validation passes and allocation time
remains. `technical/front-preparation.json`, `preparation.log` and `handoff.json`
record progress/failures. No simulations or quenches are started. Preparation
uses an immutable snapshot under `technical/orchestration/code`; GPU fits receive
their own immutable snapshot. The user's bash allocation is preserved.
