# Output curation and shared run tracking — 2026-09-07

The [dashboard](../output/registry/index.html) is now the entry point for training
results, simulations, storage and research ideas. It indexes **98 local output
groups and 74 IDS groups**. The local inventory fell from **48.94 GiB to 35.96 GiB**
of allocated file storage, excluding the small generated registry itself.

## Retention results

- Removed **78 explicitly verified files**, reclaiming **12.69 GiB**: regenerable
  temporal neighbor/inference caches, their archived paired metadata, and unused
  periodic training checkpoints.
- Packed **3,349 old experiment logs and W&B journals into 42 lossless archives**,
  reclaiming a further **0.29 GiB**. Every member was reread and SHA-256 verified
  before deleting its original. Simulation logs were excluded.
- Verified that all **4,316 pre-existing report, plot, config, metric and source
  files** remain, including **1,665 visual artifacts**. All **27 checkpoint paths
  explicitly referenced by configurations/reports/recipes** remain. Distinct
  model/seed weights, best/final comparisons, scalers and last checkpoints remain.
- Preserved original artifact paths, source trajectories, simulation restarts,
  identity/timeline arrays and unique scientific arrays. The remaining roughly
  36 GiB is not all disposable clutter: it includes about 23 GiB of simulation
  artifacts and temporal datasets, plus retained scientific results and models.

The deletion audit is
[`output/registry/cleanup_applied.jsonl`](../output/registry/cleanup_applied.jsonl).
The original inventory, explicit plan, preserved cache specifications and
verification reports are alongside it. `CACHE_RETENTION.md` in each affected
analysis directory explains full-inference regeneration. Historical metrics and
plots were not rerun or altered by this cleanup.

## Shared system

The maintained registry command and its implementation provide static HTML/JSON
navigation, config snapshots, explicit verified pruning, lossless log packing,
execution tracking and a small versioned ideas backlog. There is no database
service. New execution attempts record exact configs and commands, git commit,
dirty patch, source archive including untracked research code, package versions,
host/PID/job identity, exceptions and outcomes. Recorded historical commits remain
unknown where no run provenance exists; today's commit is only the catalogue commit.

The experiment runner, registered Hydra training commands, MACE workflow,
spatiotemporal trainer and elemental simulation run command write this provenance.
Other maintained commands can use an explicit execution spec. Their scientific
and restart protocols remain separate. No submitted launcher was moved, and no
external batch script was rewritten.

New default Hydra runs use `output/runs/training/<experiment>/<timestamp>/`.
Checkpoint retention defaults to one best and one latest periodic snapshot;
explicit top-k configurations remain honored. The spatiotemporal best/last/final
comparison remains intact. Requested post-training analysis failures now fail
the workflow instead of only printing a warning.

The [guide](output_registry.md) documents refresh, status inspection, launch specs,
idea updates, archive recovery and the next config-organization step. Configs and
experiment recipes have verified content-addressed snapshots with retained capture
indexes; the actual `configs/` migration has not been performed.

## Progress observation and limits

The dashboard records filesystem status and timestamps. It does not claim that a
historical `running` JSON establishes scheduler liveness. Local tracked process
identities can be checked; remote identities remain explicitly unverified. SSH to
the former GPU node rejected access because there was no active allocation there;
the accessible GPU login host did not expose `squeue`. No simulation launcher was
retired, and simulation files/jobs were excluded from destructive cleanup.

During the final refresh, the existing Ta/Ti recovery wrote a new failure at
**18:23 CEST**. Position conversion has a completed recovery record, but the
subsequent `--resume-ta` attempt failed with **`OSError: [Errno 122] Disk quota
exceeded` while writing the existing Ta runner log** during verification of an
already completed branch. Its `run_record.json` also captured this failure.
This is shown as blocked work in the ideas backlog; the sequence must not be
reported as healthy/running. No additional simulation was started for this task.

## Validation and file roles

**26 focused tests passed**, covering destructive cleanup boundaries, complete
preflight before deletion, retained prerequisites, verified log recovery, failed
command logs, attempt history, checkpoint recovery, workflow commands and format
conversion. Python compilation and whitespace checks passed. The retained source
files required by all removed inference caches are present.

New `scripts/experiment_registry.py`, `src/experiment_runner/registry.py` and
`tracking.py` are maintained tools. `experiments/registry.json` and `ideas.json`
are maintained organization records. This report and the guide are versioned
documentation. Dashboard, inventories, test logs and cleanup manifests are
generated records under `output/registry/`; retained provenance and verified log
archives must be kept when pruning generated output later.
