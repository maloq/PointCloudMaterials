# Research records and result storage

For a new question, create `experiments/<question>_<YYYYMMDD>/README.md`. State the
question, scientific protocol, configuration, exact reproduction command, output
location and findings. Keep implementation in the relevant `src/` package; faithful
older research workflows live in `src/research/<method>/`. Keep dated commands out
of `scripts/`; its [index](../scripts/README.md) lists maintained entry points.

```text
experiments/question_YYYYMMDD/
  README.md                 question, protocol, commands and findings
  RESULTS.md                optional longer interpretation
  technical/                run specs, inventories, config and queue plans

output/question/run-name/
  README.md                 start here
  index.html                analysis gallery, where applicable
  plots/                    umap.png, spatial-166ps.png, forecast_scores.png
  tables/                   metrics.csv, model-comparison.csv
    METRICS.md              definitions frozen at table export
  technical/                JSON, arrays, caches, logs and intermediate stage trees
```

Use one question and one readable run name (e.g. `history-seed3`); do not add
`runs/training/question/timestamp/default/variant/` to hand-written output paths.
Training-plan stages and exact execution-attempt directories retain their existing
semantics. Source snapshots must remain immutable even when their names are technical.
Reusable Hydra composition stays under `configs/`; historical standalone configs live
with their experiment. Do not move a Hydra default without updating its composition.

New default Hydra training puts its checkpoints, logs and source/config snapshots
under `technical/`; post-training analysis publishes at the run root. Explicitly
configured run directories retain their original training paths.

New standard analyses put their intermediate artifacts under `technical/` and
publish short plot names and metric tables at the run root. New embedding forecasts
store configs, model states and prediction arrays under `technical/`; their plot and
CSV are one level below the root. Plan collection writes CSVs under `tables/`, summary
JSON under `technical/`, and plots directly under `plots/`. Existing analyses and
forecast resumes are explicitly recognized and preserve their previous artifact paths.
`outputs/` is a legacy root: readable storage inspection and cleanup include it; new
runs should use singular `output/`. Do not bulk-move historical runs or submitted jobs.

Recent queue configs and simulation launcher paths have deliberately not moved until
Slurm quiescence can be checked. Older experiment configs were moved into `technical/`;
repository imports, reproduction commands and config references were updated together.
The September 12 change is recorded in [the cleanup report](repository_cleanup_20260912.md).

## Metrics and their definitions

The maintained standard-analysis, topology-comparison, forecast and experiment-plan
exporters save a CSV with `tables/METRICS.md`. Definitions include weighting, target
normalization, train/test separation, seed spread and the source bootstrap. Raw JSON
stays available under `technical/`. Each export saves the exact source/doc hashes in
`technical/metric-contract.json`; old exports retain their old documents.

When changing a metric, update its description in `docs/metrics/` in the same change.
Update the matching SHA-256 entries in `docs/metrics/contracts.json` after reviewing
both code and documentation (e.g. compute each with `sha256sum PATH`). The
`experiment_registry.py metrics-docs` command and `test_research_layout.py` reject
unreviewed drift. Analysis/aggregation include topology definitions; update those
copies when changing topology. Preserve old exported docs; do not rewrite historical
results to imply they were evaluated under a new formula. Retired workflows retain
their dated methods descriptions and source snapshots rather than invented glossaries.

## Storage and quick cleanup

```bash
conda run -n pointnet python scripts/experiment_registry.py storage
conda run -n pointnet python scripts/experiment_registry.py clean --root output/QUESTION/RUN
# After checking this run has no active or queued readers/writers:
conda run -n pointnet python scripts/experiment_registry.py clean --root output/QUESTION/RUN --apply --inactive
```

`storage` writes `output/maintenance/storage/STORAGE.md` and `large-files.csv` using
allocated disk bytes. It does not follow external dataset/cache symlinks. `clean`
selects only the current inference-cache producer with completed analysis metrics and
an unchanged, retained local checkpoint. It archives the metadata, writes a plan, then
uses the existing size/hash-verified prune command. Its preview never deletes files.
Inspection, plans and retained sidecars live under `output/maintenance/`. Keep the
sidecars: they explain how to recreate removed caches. Changed/missing checkpoints and
external storage are reported and excluded; malformed/orphan caches fail explicitly.

Keep selected checkpoints, exact-resume state for unfinished runs, scalers, paired
test errors/predictions, research reports and source/config snapshots. Keep unique
trajectories, IDs, timelines, raw observations and simulation restarts. Larger
trajectory exports may be removed only through the existing verified conversion
commands. New positions remain float16, boxes float32, identities/timelines exact;
LAMMPS integration and restart precision are unchanged.

MACE templates discard inference arrays after successful analysis; automatic removal
now preserves reconstruction sidecars in `technical/retention/`. MACE configs also
disable duplicate paper/figure sets and omit validation prediction arrays. Generic
static/temporal templates retain their caches because existing spatial-comparison
workflows consume them after analysis; clean them only after those comparisons finish. Other protocols retain
their explicit settings. Large PNG collections and array files are reported, not
assumed disposable. `pack-logs` remains available for verified lossless log compression.

## External research storage (2026-09-13)

Existing simulation inputs and datasets live on WORK; caches live on IDS.
New simulation output starts on SCRATCH and completed elemental campaigns publish
verified copies to STORE. SCRATCH is not backed up and has a 30-day inactivity purge;
archive stopped failures/restarts explicitly. WORK still reports a 512 GiB capacity.
Use named storage roots and dataset IDs, preserving old aliases and immutable
manifests. See [portable setup and retention](portability.md) and the
[initial migration record](../experiments/storage_migration_20260913/README.md).
