# Pre-appearance birth classification (v1)

Producers: `src/research/birth_prediction/{data,features,infer,fit,queue}.py`.
The shared fitting function also accepts declared, disjoint source partitions
for the separate [truncation/CV protocol](birth_prediction_extension.md). Its
original fixed-source likelihood and scoring definitions below are unchanged;
historical exports retain their frozen implementation.
The same fitter also accepts explicit observation transforms for the separate
[fixed-endpoint/site protocol](birth_prediction_temporal.md), whose exports use
their own metric family. The original default truncation path is unchanged.
This is a deliberately event-enriched case/control classification experiment,
not a natural-risk nucleation probability benchmark or a critical-nucleus assay.

## Population and labels

Use the original fixed Al64 release's 90/15/15/30 source roles and exact 0.75 ps
observations. A separate derived event-centered track is recorded; it is not the
historical all64 center/window evaluation population. Original definitions,
artifacts and source splits remain unchanged. Only primary isolated establishments
from the complete full-cell PTM/ancestry audit are cases.

At birth, uniformly permute atoms within 8 A of the recorded birth centroid;
scan at most 64 to retain up to four eligible centers per event. Follow their
actual atom identities and positions. Within the preceding 32-frame search,
find the first PTM-crystalline atom in the center's moving 8 A sphere. That
appearance must include at least one atom in the eventual birth core. Boundary
appearances are left censored; no appearance, incomplete history, prior visible
crystal or unrelated appearance are explicit exclusions. This is bounded-window
appearance, not a claim of the first crystalline atom ever in the full simulation.

Each positive's eight input frames end immediately before appearance. Every
input frame must contain **zero FCC/HCP/BCC PTM atoms in its entire moving 8 A
sphere**, conservatively including atoms beyond the nearest-80 consumed inputs.
The model receives only the centered nearest-80 original coordinate patch with
radius-8 consumption. No future center, membership, PTM, event identifier, time,
temperature, species or material channel becomes a feature.

For every case retain four uniformly proposed centers from the same source and
same endpoint, accepted only when their moving 8 A sphere remains PTM-clear
through the entire eight-frame history and follow-up. Follow-up is at least
6 ps and extends through the matched case's establishment confirmation.
This means liquid throughout the declared observed interval, not indefinitely
liquid or liquid everywhere in the full cell. Candidate/control proposal limits
and zero-yield events/sources are retained. Past/future PTM is used for dataset
labeling and exclusions only. Cases are identified retrospectively, so this
population cannot estimate absolute prospective risk or population prevalence.

Negative coordinates and labels are deduplicated by source/atom/endpoint.
One event contributes equal total scoring weight: each row has inverse count
of rows associated with that source/event. Complete matched sets preserve 1:4
case/control prevalence. Sources with zero yield retain coverage records.
The predeclared support gate requires 10/3/3/5 distinct train/selection/calibration/
test events. This permits an exploratory comparison, not precise rare-event
performance estimates. Previously inspected held-out event catalogues are not
a new untouched discovery dataset.

## History endpoints and features

The four fits retain the same cases, controls and starting frame. Remove zero,
one, two or three newest frames, giving 8/7/6/5 observations, spans
5.25/4.50/3.75/3.00 ps, and endpoint-to-appearance leads 0.75/1.50/2.25/3.00 ps.
Both lead and history length change; this is not a pure fixed-length lead study.
Actual per-source observation times are checked by the original raw producer.

Descriptors reuse `liquid_predictability.descriptors.patch_descriptors`: geometry,
bond order, CNA and rich alpha-persistence features in Al units. All finite
intervals, persistence summaries/images and smooth Betti curves use their
existing construction. Descriptor neighbor averaging stays within the same
nearest-80 patch. The learned features are frozen scalar exports from pinned
rich-descriptor MACE and VICReg MACE checkpoints, including their recorded
pooling/trunk normalization. Inference runs against each checkpoint's exact
recorded native producer; no fine-tuning or new physical pretraining occurs.

Every bank receives the same temporal aggregation: chronological frame-feature
concatenation, mean, population standard deviation and last-minus-first change.
Lengths are identical between cases and controls within each experiment.
No explicit time/condition tensor is appended. The same frozen checkpoints and
one readout seed serve all endpoints.

## Fitting, calibration and measurements

Boosting uses binary Logloss, selection-source Logloss early stopping, fixed
capacity and no class-balancing/AP loss. Linear logistic controls use train-only
standardization and choose C by selection-source binary NLL. Calibration uses
separate calibration sources and a regularized monotone affine logit map.
All frozen probes and descriptor controls stay local; no online training run
is created or restarted for this diagnostic study.

Metrics use event-balanced row weights normalized within each evaluated split:

- NLL: weighted mean binary negative log likelihood; probabilities clipped to
  [1e-7,1-1e-7] for scoring.
- Brier: weighted mean squared probability error.
- AUROC and AP: sklearn weighted binary definitions; AP is diagnostic only and
  depends on the deliberately enriched 20% prevalence.
- Recall at a locked false-alarm threshold: weighted 95th percentile of calibrated
  probabilities among calibration liquid controls; alarms use `p > threshold`.
  Report achieved test false-positive rate as well as recall.
- Coverage: distinct source/event births, histories, sources and each rejection
  reason; counters can overlap and are not independent physical event counts.

Raw and calibrated scores are retained separately. Bootstrap intervals are
2.5/97.5 percentiles from resampling whole test sources (2,000 draws), conditional
on the fitted models/one seed. Matched contrasts subtract per-row calibrated
log losses using the same source draw. They do not quantify training-seed
uncertainty or correct for multiple comparisons.

The 3/6 ps establishment-lead strata keep complete matched sets whose associated
case establishes within the stated lag from that experiment's endpoint. These
are case/control subgroup diagnostics, **not horizon-specific calibrated risk**.
Unsupported/one-class strata report unavailable AP/AUROC; no model-specific
evaluation-row exclusions or test-driven selection occur.

Persist sample identities, source roles, model/source hashes, raw/calibrated
predictions, transforms, fitted readouts and selection records. Generated neural
states use the shared six-entry leased cache; predictions and metrics are permanent.
Every CSV freezes this definition and implementation hashes via metric_docs.
