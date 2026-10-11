# Al shooting appearance and fate experiments

Inputs and labels are now joined in the [shared Al pool](al_research_pool.md),
with cohort selection and an overview of all Al work. This page describes only
the frozen shooting comparison.

This workflow extends the Al appearance assay using the completed shooting
search and adds a separate question: does the crystal-free local structure
predict whether an observed nucleus later dissolves or shows continued growth?
Ta is excluded. Earlier results and the separate relaxed-input study are preserved.

The shooting audit found 77 isolated establishments and 30 brief dissolved
candidates. Of the establishments, 31 have documented growth and 46 unresolved
endings. Eligibility screening determines how many can supply crystal-free
histories and four matched liquid controls. Counts and exclusions are saved;
unresolved endings never become negative dissolution labels.

Appearance prediction compares earlier-only, shooting-only and combined
training on common outer holdouts. Fate prediction uses resolved nuclei only,
with dissolution positive and continued growth negative. Both compare logistic
regression and CatBoost using a snapshot or four observations at −6, −4.5, −3,
−1.5 ps before appearance. These offsets match exactly across the two source
cadences. No time or temperature covariates are used. This first extension uses
original coordinates; the older relaxation queue continues separately.

Five outer ancestry folds and five inner NLL-selection folds produce 80 final
fits, using one seed. All branches from a preparation ancestor stay together.
The earlier outer assignment remains fixed. Readouts are diagnostic fits, so
tracking stays local. [Metric definitions](metrics/birth_shooting_appearance.md)
describe weights, proper scores, uncertainty, label resolution and unequal
follow-up. These are retrospectively selected sites, not a prospective alarm.

## Execution and outputs

Use conda `pointnet-torch214`:

```bash
python -m src.research.birth_prediction.shooting_appearance submit \
  --config configs/birth_prediction/shooting_appearance_20261003.json
```

Eight CPU lanes prepare descriptors from existing coordinates and PTM records.
One CPU job seals the data and checks fold support, then one GPU runs readouts
and exports final comparisons. Every stage runs detached through Slurm, with
an immutable code/config bundle and a local launch receipt. No new simulations,
PTM runs or relaxation are needed. The frozen commands can resume completed
source preparation or fits without repeating them.

Scientific outputs are under
`${storage:training_storage}/birth_prediction/shooting-appearance-20261003`.
The generated input cache is under
`${storage:cache}/birth-prediction/shooting-appearance-20261003`, outside the repo.
`technical/launch.json` records jobs; source progress and source receipts are
in the cache. `analyses/coverage-v1` holds fold counts and fate eligibility;
`analyses/transfer-and-fate-v1` holds train/test errors, paired comparisons,
calibration and held-out predictions. `technical/design.json` freezes ancestry
assignments and the exact readout recipe.

## October 3 launch and screened population

Slurm preparation array1019840 completed all92 candidate-containing sources.
Sealing job1019841 and fitting/evaluation job1019842 form the detached dependency
chain. The preparation screen retained19 shooting events: six establishments
and13 brief dissolved episodes. They provide53 positive center histories and
212 matched controls from10 ancestors and12 parent configurations.

Of those shooting events, three have documented growth,13 are dissolved and
three remain right censored. The three growth cases share one ancestor, so
shooting-only fate conclusions will be particularly uncertain. The binary fate
experiment uses combined earlier and shooting resolved cases, preserving outer
ancestry separation; the shooting subset is a diagnostic population, not an
independently well-balanced fate benchmark. Censored cases still enter appearance
prediction. Most searched candidates could not supply the required crystal-free
prehistory; no label or eligibility threshold was loosened to retain them.

The sealed union contains247 events,3935 case/control rows and129 ancestors.
Fate classification retains236 resolved events:146 dissolved and90 documented
growth events, represented by766 center histories. It excludes eight unresolved
merges from the earlier pool and three censored shooting events. All outer and
inner ancestry/class-support checks passed. The fit worker started on node51;
training and final evaluation continue detached under job1019842.
