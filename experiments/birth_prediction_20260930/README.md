# Can crystal-free histories distinguish future isolated establishments from liquid controls?

Requested 30 September 2026. This is a new, event-enriched classification study
on the original fixed Al source roles. It preserves the earlier arrival benchmark.
Its labels are operational isolated establishments, not validated critical nuclei.

New follow-up: [merged evaluation with training-only CV selection](MERGED_TEST.md).

Earlier follow-up: [descriptor-family information on original/relaxed snapshots](FEATURE_FAMILIES.md).

Earlier follow-up: [fixed-endpoint history and persistent-site controls](TEMPORAL_SITE.md).
It preserves the retained cohort and distinguishes changing input length from
changing the prediction endpoint, with same-site score and descriptor diagnostics.

Completed follow-up: [leakage and structural-feature audit](../../docs/encoder_research/birth_leakage_features.md).
All 11,793 inputs replay as past, PTM-clear geometry. Positive locations/endpoints
and persistent-liquid controls are retrospectively selected, so these scores
remain annotated-site discrimination. Matched feature interventions most
consistently support local void topology and bond orientational order. No new
encoder or predictor was fitted for that audit.

## Question and matched population

Can history ending before **any observed PTM crystal** distinguish a region that
will establish a crystal from one that remains liquid over matched follow-up?
Does that information remain when the newest one, two or three frames are removed?

Use the complete 150-source audit, with original 90/15/15/30 train/selection/
calibration/test roles. Cases use up to four atom centers near each isolated
establishment. Center selection is explicitly retrospective; inputs are ordinary
observations around each tracked atom at its actual observed positions.

Search up to 24 ps before establishment for the first visible PTM crystal in
the moving 8 A sphere. Require association with an eventual birth-core atom,
complete history and no crystal anywhere in that sphere during the input.
Boundary appearances and unrelated crystal contact remain explicit exclusions.
This does not assert the first crystalline atom ever in the trajectory.

Each case receives four control centers proposed uniformly from the **same
source at the same endpoint**, accepted only if their moving 8 A sphere remains
PTM-clear across the whole input history and through matched birth confirmation,
with at least 6 ps follow-up. Thus “liquid all the way” means the complete declared
interval. Controls are not assumed indefinitely liquid. Source/time matching
reduces preparation and global-transformation confounding without passing time
or temperature to the classifier.

All models use exactly the same accepted cases and controls. The full 8 A sphere
is audited, while every encoder/descriptor consumes the same nearest-80 patch,
with a radius-8 mask. No atom deletion creates the liquid-only observations.
There is one readout seed, no new trajectories or encoder fitting.

## History treatments

Let A be the observed appearance frame. The common start is A−8.

| Newest frames removed | Observations | Endpoint | History span | Lead before appearance |
| ---: | ---: | --- | ---: | ---: |
| 0 | 8 | A−1 | 5.25 ps | 0.75 ps |
| 1 | 7 | A−2 | 4.50 ps | 1.50 ps |
| 2 | 6 | A−3 | 3.75 ps | 2.25 ps |
| 3 | 5 | A−4 | 3.00 ps | 3.00 ps |

These are the user's requested truncated histories. Earlier endpoint and shorter
history are intentionally coupled; a fixed-length shifted history would be a
different experiment. Actual original source timelines must be exact 0.75 ps.

## Predictors and frozen encoders

Eleven predictors × four endpoints = **44 fits**:

- Training-prevalence intercept.
- Boosted trees on all rich descriptors, TDA alone, bond order alone, and
  geometry/CNA together.
- Linear logistic controls on all rich descriptors and on geometry alone.
- Boosted trees and linear logistic probes on each of the two MACE exports.

The rich bank reuses the current 442-feature local producer: geometry, CNA,
harmonic order/coherence and alpha-persistence statistics, images and Betti curves.
Every bank receives chronological feature concatenation, mean, standard deviation
and last-minus-first change. This gives learned and conventional features the
same observation support and temporal readout construction.

**Rich MACE:** the retained RH2 256-channel, three-interaction checkpoint at
epoch14/update1806, selected by Al selection-source descriptor Gaussian NLL.
It predicts rich structural/TDA descriptors during training; no birth label
selected that encoder. The prior run stopped at update5486; its best retained
checkpoint is the comparison input, not a claimed completed60-epoch fit.

**VICReg MACE:** the completed R2 128-channel epoch24 reference from the matched
encoder-mechanisms study, seed20260926, selected by its predeclared fixed endpoint.
This is a strong retained VICReg reference, not proof of an optimal checkpoint
across all historical studies. No AP-based choice or birth-driven encoder search
is performed. Model widths, objectives and training populations differ, so any
contrast measures the complete retained treatments rather than objective alone.

Both are frozen, including recorded normalization. Each runs against its original
native source files. Shared scalar exports are used; descriptor heads and vector
fields are not classifier inputs. New history readouts use the same seed and
predictive likelihood rule. No conditions, explicit time covariates, species,
material IDs, relaxation or physical-reconstruction pretraining are added.

## Selection and evidence

Train boosted trees with binary Logloss and validation Logloss early stopping.
Linear C is chosen by validation NLL using train-only standardization. Separate
calibration sources fit a monotone logit calibration and lock a 5% liquid-control
false-alarm threshold. Test outcomes do not select any fit or threshold.

Report raw/calibrated NLL, Brier, AUROC, AP and recall with achieved false-positive
rate. Event-balanced weights prevent many anchors from dominating; source
bootstrap intervals account for correlated event/anchor observations. There is
one training seed, so intervals omit training-seed uncertainty. Retain paired
model comparisons and all four endpoint predictions on identical row IDs.

Report 3/6 ps **establishment-lead strata** as secondary diagnostics on complete
matched sets. These are not separately trained horizon-risk forecasts.

The 1:4 enriched case/control ratio gives 20% prevalence. AP and calibration
refer to this artificial classification population, not natural nucleation
frequency. Representative prospective evaluation remains a separate task.
Past inspection of held-out birth coverage is disclosed; this is exploratory
benchmark development. Minimum support is10/3/3/5 distinct events in train/
selection/calibration/test. A failed gate publishes counts and stops fitting.

## Reproduction and results

[Recipe](../../configs/birth_prediction/preappearance_20260930.json) ·
[Exact metrics](../../docs/metrics/birth_prediction.md) ·
[Execution](../../docs/birth_prediction.md).

```bash
python -m src.research.birth_prediction.queue bind --config configs/birth_prediction/preappearance_20260930.json
python -m src.research.birth_prediction.queue submit --config configs/birth_prediction/preappearance_20260930.json
```

Scientific results: `${storage:training_storage}/birth_prediction/preappearance-20260930/`
with `analyses/coverage-v1`, `analyses/classification-v1/<predictor>/minus-N`
and `analyses/comparison-v1`. Numerical CSVs retain frozen metric definitions
and implementation hashes. The result report must distinguish catalogue births,
eligible events, correlated histories and fitted classification performance.

## Submitted release and support

The 30 September preparation retained **95 distinct isolated establishments**,
295 positive histories and 1,180 matched liquid histories (1,475 total). Original
source roles remain 90/15/15/30; sources with no eligible observations retain explicit
coverage records.

| Split | Eligible births | Positive histories | Liquid controls |
| --- | ---: | ---: | ---: |
| train | 59 | 193 | 772 |
| selection | 8 | 23 | 92 |
| calibration | 13 | 39 | 156 |
| test | 15 | 40 | 160 |

All 44 fits and comparison stages were submitted detached. The output
`technical/launch.json` records jobs 1015346–1015350 and 1015358–1015360,
the frozen implementation and recovery from a transient Slurm submit limit.
Descriptor extraction and frozen encoder export precede readout fitting.
Classification scores are pending; these counts are support, not performance.
