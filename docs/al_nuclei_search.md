# Al nucleus searches with separate observation protocols

Current code and commands: [shared post-simulation module](postprocessing.md).
Dated execution receipts below retain their recorded implementations.

The [revised lineage search](al_nuclei_lineage.md) reuses this release's structural
classifications. The original `established` counts require confirmation before
any split/merge and must not be interpreted as all physical nucleus births.
Version 3 separates birth origin from later growth and interaction fate.

The new full-cell search separates the event catalogue from prediction-data
eligibility. Small and disappearing embryos remain visible even if they lack a
strictly crystal-free input history or matched persistent-liquid controls.
Historical event labels and prediction datasets retain their original definitions.

The [recipe](../configs/analysis/al_nuclei_search_20261003.json) defines exact
0.75 ps, 0.10 ps and 0.30 ps searches. Native 0.1 ps trajectories are also searched
at 0.3 ps using every third frame. Shooting retains its native 0.3 ps observations.
Every source enters every exactly compatible grid; no interpolation is permitted.
The million-atom Al trajectory, older sources, dense descendants and available
additional campaigns are included through the shared registry. Incompatible,
unavailable, derived and relaxed inputs remain explicitly inventoried.

Four parallel detectors compare PTM with loose/reference/strict coherent bond
order. Physical-time tracking retains brief embryos, persistent establishment,
documented growth, dissolution, parent crystals, interfaces and uncertain
interactions. The [metric contract](metrics/al_nuclei_search.md) defines every
threshold, censoring rule and paired-event comparison. The detector has not been
validated as a critical-nucleus estimator.

## Execution

Use `pointnet-torch214` with the added `freud-analysis==3.4.0` dependency.

```bash
python -m src.research.crystallization_origin.search prepare --config configs/analysis/al_nuclei_search_20261003.json
python -m src.research.crystallization_origin.search submit --config configs/analysis/al_nuclei_search_20261003.json
```

CPU extraction shares verified frame classifications between cadence views.
Tracking and final analysis are separately resumable Slurm stages with frozen
code and configuration. Completed chunk/event receipts are verified before reuse.
No online training run is created for this scientific data audit.

The 2026-10-03 launch contains 1,042 source records, including related historical
prefixes: 179 exact 0.75 ps views, 43 exact 0.10 ps views and 892 exact 0.30 ps
views. All 43 records with 0.10 ps views also have matched stride-three 0.30 ps
views. These are source/view counts, not independent-melt counts. Static/relaxed
collections, unavailable inputs, incompatible cadences and unknown materials are
recorded in coverage; ongoing simulations require a later release.

Extraction job array **1020339** uses 24 lanes of four CPU workers. The separate
tracking array hit Slurm's submission-count limit, so **1020363** runs all sixteen
tracking lanes in groups of four and then collects results, after extraction
succeeds. Scientific settings and frozen code are unchanged. The launch receipt
preserves the initial scheduler error and its resolution; `track-collect.sbatch`
records the exact combined command. The 48-hour postprocessing limit is an
execution allowance, not an estimated completion time.

Results are at `${storage:training_storage}/nuclei_search/al-all-20261003-v2`;
disposable extraction arrays are at `${storage:cache}/nuclei-search/al-all-20261003-v2`.
`technical/inventory.json` and `technical/plan.json` freeze population and exclusions.
`technical/launch.json` records submitted jobs; per-lane and per-source progress
separates queued, running, failed and completed work. The final bundle is
`analyses/catalogue-v2` with events, counts, coverage and matched comparisons.

`python -m src.research.crystallization_origin.search progress --config CONFIG`
publishes only completed results into a separately labeled progress bundle.
Absence of an event in an unfinished source is not a negative outcome. New sources
that finish simulation after the inventory snapshot require a later search release.

Related: [shared Al pool](al_research_pool.md),
[earlier strict dense prediction assay](birth_dense010.md),
[literature and unresolved physical definitions](encoder_research/nucleation_literature_and_open_questions.md).
