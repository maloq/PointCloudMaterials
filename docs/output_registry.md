# Experiments, simulations and ideas

Start at [the searchable registry](../output/registry/index.html), or its
[Markdown index](../output/registry/README.md). The registry is a generated static
page and JSON, with no database server or service to maintain. Open the HTML in
a browser; all plots, including representatives, latent analysis and spatial
views, link to their original artifacts. Image galleries load on demand.

The **Experiments**, **Simulations**, **Datasets & caches**, **Maintenance** and
**Ideas** views separate scientific runs from storage and operational records.
Search matches run names, metric names, configs and artifact paths. Expand a run
for numerical metric previews, reports, checkpoints, config files and plots.

Add questions to `experiments/ideas.json`; each needs an ID, title, question,
state, concrete next action, and a link to evidence. Update an existing idea with
`experiment_registry.py idea --id ID --state planned --next-action "ACTION"`.
Updates retain a small decision history. Rebuild the dashboard to display them.

The [variant C pilot](../experiments/mace_thermal80_20260909/README.md) prepares
matched hot/relaxed full-cell data on node53, then runs 12 epochs and static Al
analysis through the same maintained MACE queue.

The current MACE recipe restores the [original VICReg pipeline](../experiments/mace_original_vicreg_20260909/README.md):
pretrained small MACE, normalized 80-point geometry without element inputs,
the original Lightning module/projector and pure spatial/temporal VICReg.
It runs 24 epochs followed by the standard static Al analysis.
The [matched VICReg + TDA run](../experiments/mace_original_vicreg_tda_20260909/README.md)
is queued after both finish, using the same protocol with TDA from epoch 1 and
its own static analysis. Its detached dependency wait is visible in
`output/mace_original_vicreg_tda_20260909/queue_status.json`.
The baseline completed training but its first analysis hit Dynamo's radial
recompilation limit. Analysis was restarted with eager radial layers, and the
TDA queue now waits on that analysis recovery record. Both recoveries use the
original training/analysis output locations and preserve failed-attempt records.
The completed [plain80](../experiments/mace_plain80_20260909/README.md) and thermal
runs retain their results and original execution source/config snapshots.

## Files and responsibilities

| File / directory | Role |
| --- | --- |
| `experiments/<question>_<date>/README.md` | Versioned research question, protocol, reproduction commands and findings |
| `experiments/ideas.json` | Small versioned backlog: question, state, next action and evidence |
| `experiments/registry.json` | Explicit IDS storage roots, scanned read-only |
| `output/registry/experiments.json` | Generated inventory and results index |
| `output/registry/index.html` | Generated navigation, progress and plot galleries |
| `output/registry/config_snapshot/` | Verified config/recipe snapshots and capture indexes |
| `<run>/run_record.json` | Latest tracked execution, not a replacement for scientific status |
| `<run>/tracking/<timestamp>/` | Immutable execution attempt, config copies, source archive, git patch and outcome |

The implementation lives in `src/experiment_runner/registry.py` and
`tracking.py`; the maintained entry point is `scripts/experiment_registry.py`.
The registry and cleanup manifests are disposable generated records, except that
**retained cache metadata and diagnostic archives must be preserved**: they are
the remaining provenance of removed artifacts. The experiment JSON files above
are maintained research organization records, not generated run outputs.

## Refresh and inspect

```bash
conda run -n pointnet python scripts/experiment_registry.py build
conda run -n pointnet python scripts/experiment_registry.py status \
  --record output/runs/training/QUESTION/RUN/run_record.json
```

Building refreshes both local outputs and configured IDS roots without loading
large arrays. IDS artifacts are linked through `output/registry/storage/`.
No external data is moved or deleted. If a configured root is unavailable, the
build raises an error instead of silently dropping it.

If your editor does not render the dashboard, serve it locally from the repository:

```bash
python -m http.server 8765 --bind 127.0.0.1
# Open http://127.0.0.1:8765/output/registry/index.html
```

The dashboard is a snapshot, not a background monitor. Refresh it after important
events. Status JSON is displayed with its source; an old `running` record is not
proof of scheduler liveness. The `status` command checks local PID identity and
detects a dead/reused process. Remote process status is explicitly unverified.
Simulation branch completion and source ancestry remain defined by the existing
campaign manifests and verifiers. A successfully submitted job is not completed
science, and a cancelled campaign may still contain useful completed sub-runs.

## New runs and provenance

The experiment-plan runner, registered Hydra training entry points, MACE workflow,
spatiotemporal trainer and elemental simulation `run` command now record execution
attempts automatically. Resume semantics remain owned by their existing commands.
Every attempt captures the exact command, config bytes, Python/environment identity,
host/PID/Slurm job ID, package versions, git commit, dirty patch, and a compressed source snapshot
including untracked repository research code. Exceptions are recorded and reraised.
Requested post-training analysis failures now fail the workflow explicitly.

Default Hydra training runs use:

```text
output/runs/training/<experiment_name>/<timestamp>/
```

Explicit run directories and submitted jobs retain their original paths. For new
simulation protocols, use the configured IDS simulation root; the run record lives
beside the simulation artifacts. Keep derived training datasets on IDS, not under
the repository. Do not move active campaigns or rewrite their submitted scripts.

For another maintained command, use an explicit JSON execution spec:

```json
{
  "kind": "analysis",
  "question": "What does the retained encoder separate on the static Al sample?",
  "output": "output/runs/analysis/static_al/review_01",
  "cwd": "/home/infres/vmorozov/PointCloudMaterials",
  "configs": ["configs/analysis/static.yaml"],
  "command": ["/home/infres/vmorozov/miniconda3/envs/pointnet/bin/python", "-m", "src.analysis.pipeline", "configs/analysis/static.yaml"],
  "dependencies": [],
  "completion": "process_exit"
}
```

This is a template: set the scientific config's output path to the same run before
launching. The wrapper does not rewrite configs or scientific command arguments.
Save a real spec with the experiment recipe, then run:

```bash
conda run -n pointnet python scripts/experiment_registry.py run --spec PATH.json
```

Wrapper records go under `<output>/execution/` to coexist with commands that
already track themselves. Its log is `<output>/command.log`. A repeated spec
cannot overwrite a tracked execution. Dependencies are explicit `run_record.json`
paths that must have `command_succeeded`. For a command that only submits jobs,
set `completion: submission_only`; its terminal state is `submitted`, and it
cannot satisfy a completed dependency. This command does not invent a generic
restart protocol or automatically launch proposed ideas.
Interrupts are forwarded to the launched command's process group and recorded;
an uncatchable termination is reported as stale when its local PID disappears.

Historical runs are different: a catalogue build records today's commit only as
the **catalogue commit**. It never invents a historical training commit. Existing
run configs, source snapshots and hashes remain the evidence. Missing provenance
is displayed as not recorded.

## Retention and cleanup

Keep the checkpoints used for each reported model/seed comparison, including
distinct best/final models when compared. Keep selected weights and the scalers
needed to load them. The default Lightning top-k count is now one; explicitly
configured top-k counts still apply. Periodic saving retains the latest periodic
checkpoint rather than all epochs. The spatiotemporal trainer also retains best,
last and final because those are explicitly audited and compared.

Keep metrics, reports, resolved configs, source provenance, representative/latent/
spatial plots, and their scientific data. Trajectories, identity/timeline arrays,
simulation restarts and unique raw observations are not disposable run clutter.
New simulation position storage remains verified float16 with float32 box bounds
and unchanged integration/restart precision.

Regenerable neighbor and inference caches can be removed after retaining their
inputs and specifications. Inference cache metadata must be archived and removed
with its NPZ; an orphan sidecar makes the current loader fail. Rebuild caches using
the original full analysis with `figure_set.figure_only=false` before figure-only
analysis. Per-directory `CACHE_RETENTION.md` records this requirement.

Cleanup is an explicit list of paths, SHA-256 hashes, sizes, reasons and retained
prerequisites. It is never a wildcard deletion policy:

```bash
conda run -n pointnet python scripts/experiment_registry.py prune --plan PLAN.json
conda run -n pointnet python scripts/experiment_registry.py prune --plan PLAN.json --apply
```

All prerequisites and candidates are verified before the first deletion; each
candidate is checked again immediately before removal. Symlinks, external paths
and registry deletions are rejected. Every unlink is logged. Stop if a file has
changed; inspect the producer instead of editing the expected hash blindly.
Quiescence must be checked before making a plan; hashes do not lock out future jobs.

Old experiment logs and W&B event journals can be packed losslessly:

```bash
conda run -n pointnet python scripts/experiment_registry.py pack-logs \
  --before 2026-09-06T00:00:00+02:00
# Add --apply after inspecting the generated preview and verifying those runs are inactive.
```

Each archive is reread and every member verified before originals are removed.
The adjacent manifest records member hashes and the archive checksum. Extract
into a separate directory to inspect logs or recover W&B journals for replay.
Simulation and maintenance logs are excluded, except the standalone W&B folder.

## Next: configs

Keep reusable Hydra composition under `configs/`; keep question-specific launch
specs beside the research recipe; keep immutable resolved configs with each run.
`config_snapshot/index.json` maps current original paths to content-addressed
copies and SHA-256 hashes. Previous capture indexes are retained, so a later config
reorganization cannot overwrite this evidence. Build the registry before moving
configs, update composition/import/command references together, and preserve paths
embedded in active jobs. The actual configs-folder migration is a separate step.
