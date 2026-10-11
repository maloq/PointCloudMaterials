# Six frame appearance prediction from dense Al trajectories

This collection belongs to the [shared Al pool](al_research_pool.md). Its sealed
inputs are added as a native-cadence cohort, retaining the distinct observation
offsets and independently searched labels.

This experiment tests recent structural history on the native 0.1 ps data.
It first searches the 24 completed 520 K descendants for isolated births and
failed embryos, then compares original-MD descriptors at six observations
against matched snapshot controls. It does not reuse the sparse-grid births
or start new simulations. The native event count is unknown until the search
finishes; zero events from this collection entered the earlier 228-event
relaxed results or the 247-event shooting extension.

The separate million-atom Al catalogue contains 42 isolated-birth candidates
from a trajectory saved at 0.1 ps, but its search ran at 0.5 ps. Those candidates
are outside this first native-cadence cohort; their old labels are not silently
treated as 0.1 ps onset labels.

The input offsets are minus 0.6 through minus 0.1 ps before first observed
local crystal, in 0.1 ps steps. The span is 0.5 ps. Crystal-free screening,
four matched persistent-liquid controls, proper-score selection and ancestry
separation follow the [metric definition](metrics/birth_dense010.md).
Snapshot, early snapshot, real history and repeated current-frame controls
share every evaluation row. Logistic regression and CatBoost give 40 outer
fits with five-fold nested CV and one seed. Report both training and held-out
errors, AP as a diagnostic, calibration and paired ancestry uncertainty.

## Detached execution

Use conda pointnet-torch214:

```bash
python -m src.research.birth_prediction.dense submit \
  --config configs/birth_prediction/dense010_20261003.json
```

Eight CPU lanes with eight workers each compute native full-cell PTM. Six CPU
lanes with four workers then trace ancestry/fates and prepare descriptor rows.
A sealing stage verifies eligibility, timeline and ancestry support. One GPU
then performs readout fitting and final evaluation. Dependencies require each
stage to complete successfully. Frozen code, input hashes, progress and Slurm
IDs are saved in the run technical folder; completed chunks/sources/fits resume
without recomputation. Search/follow-up thresholds are not loosened when a
source has few usable events. No time or temperature enters predictors.

Results live at
`${storage:training_storage}/birth_prediction/dense010-20261003`.
Generated patches/descriptors live outside the repo at
`${storage:cache}/birth-prediction/dense010-20261003`.
`analyses/coverage-v1` holds native counts and exclusions;
`analyses/nested-cv-v1` holds scores, uncertainty and saved predictions.
`technical/inventory.json` freezes source availability and actual cadence.
Use `technical/launch.json`, PTM stage progress and source graph progress to
distinguish queued work, completed detection and completed prediction fits.

## October 3 launch

Submitted PTM array 1020147, ancestry/descriptor array 1020148, sealing job
1020149 and fitting/evaluation job 1020150. The first native 32-frame PTM chunk,
its lineage graph and six real-frame descriptor layouts passed local input
checks. The frozen bundle matches all recorded scientific producer hashes.
Six PTM lanes started immediately; the remaining lanes wait for CPU resources.
