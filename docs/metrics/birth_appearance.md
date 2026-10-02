# Appearance prediction with established and failed embryos

Producers: `birth_prediction/{appearance_data,appearance,appearance_analysis}.py`,
with the shared NLL-selected readout fitter in `merged_fit.py`. No encoder is
trained. Descriptor controls and frozen-predictor evaluation stay local, without
new W&B runs. Existing models and historical outcome definitions remain frozen.

## Cohort and target

The user explicitly chose **positive for any crystal appearance**. Established
births and verified failed embryos are positive; matched continuously liquid
histories are negative. The original predictor still learned established birth
versus liquid, so its failed-embryo scores measure transfer to a broader target.

The fate catalogue has 184 isolated, single-origin, strongly linked, atom-verified
failed episodes peaking at ≥8 atoms. The 28 strong episodes are a **subset**, not
an additional 28: the disjoint remainder is 156. Strong episodes contain ≥8 atoms
in two adjacent saved observations. See `nucleus_fates.md` for the complete finite-
observation definition, exclusions and cadence limitations.

All 150 fixed-Al64 source and melt-ancestor roles are unchanged. Original train
sources supply all fitting and five-fold NLL selection. Former selection,
calibration and test are merged for evaluation, as explicitly requested earlier.
Candidates on training sources are never presented as held-out results. Their
scores are separately labeled `train_sources`; for the original-pool model these
are new episodes on previously exposed sources, not necessarily fitted samples.

Original 1,475 rows remain untouched, in their original order. Candidate input
eligibility is checked before fitting and identical across models/domains. At most
four local atom centers are retained per candidate, as in the original assay.
For each center, search the 32 frames before the tracked component first appears
for the first PTM crystal within the moving 8 Å sphere. Its crystalline atoms
must include a member of this episode's first component. The search must have a
liquid left boundary. Require eight original-MD crystal-free observations ending
one saved frame before appearance. Actual spacing is exactly 0.75 ps (history
span 5.25 ps); no interpolation. Input centers are chosen retrospectively, which
does not establish prospective detection performance.

Select four controls from the same source and observation endpoint, PTM-clear
throughout the history and through the later of the candidate's disappearance
confirmation or six ps beyond the input endpoint. The historical established
controls retain their original follow-up through establishment confirmation or
six ps. These are declared event-specific finite follow-up rules, not infinite
liquid survival. No input includes labels, future centroids or future geometry.

Existing `(source,atom,end_frame)` histories cannot be duplicated by candidates.
New episodes have negative event IDs derived from their graph node; established
event IDs stay positive. Sample IDs retain the original source/atom/endpoint/label
format. Strong candidates enter combined fitting once. Exclusions and accepted
center counts are published for **every** one of the 184 candidates; no substitution
or model-specific dropping is allowed. Eligibility can reduce all counts.

Each event and all its matched sets carry total weight one, shared equally by
their rows. Each set has one positive and four controls, giving weighted prevalence
0.20. This is event-enriched classification, not natural nucleation probability.
Repeated centers and events within a source are not independent samples.

## Inputs and fitting

Use the same 442 descriptors (geometry 99, bond order 40, CNA 45, TDA 258), with
logistic and CatBoost readouts. Input variants are current frame and eight-frame
history. History concatenates chronological descriptors, mean, standard deviation
and last-minus-first, matching the original producer. No temperature, age,
absolute time, velocity, material/species input, or extra surrounding patches.

Compare original MD and full-cell-relaxed coordinates separately. Relaxation
minimizes each input frame with its generating Al MEAM potential in a fixed box;
the retained original nearest-80 identities, rows, labels and source roles stay
unchanged. Local offsets are saved in float32 before full-cell archival
quantization. Descriptor code then re-sorts by radius and clips at 8 Å, so the
effective retained descriptor subset can differ after relaxation. Convergence
failures block the paired result rather than silently removing examples.

Replay eight frozen original-pool readouts (two coordinate domains × two readouts
× two histories), verifying their probabilities against saved original predictions.
Fit eight combined-pool counterparts on established plus eligible failed events
from training sources only. Use the retained five source folds, one seed, original
logistic C grid and CatBoost settings. Weighted pooled training-fold binary NLL
selects C or a common tree count; full-train refitting never uses test early
stopping. Training normalization uses training rows only. No calibration map.
The chosen training-CV score is a selection diagnostic, not unbiased nested CV.

The shared fitter requires an explicit metric family so each cohort exports its
own definitions; old merged-test executions retain their frozen code/contracts.

## Evaluation and uncertainty

Evaluate both predictors on established, strong failed, other failed, all failed,
and combined populations. Every population includes its matched liquid controls:
AP/AUROC are not computed on positive-only candidates. Export event/source/row
counts, weighted NLL, Brier, AP, AUROC, matched AUC and case-control probability
gap. NLL/Brier/AP/AUROC use the existing `fit.scores`; matched AUC is the average
of four case-versus-control comparisons with half credit for ties. The probability
gap is case probability minus mean matched-control probability. Event weighting
and source resampling match `temporal_analysis.metric_bundle`.

The alarm threshold is the weighted 95th percentile of training OOF negative
probabilities, with strict `>` for alarms. Report positive recall and observed
false-positive rate at that frozen threshold, and mean positive probability.
The source bootstrap resamples complete sources 2,000 times, retains their event
weights and matched sets, and reports percentile 95% intervals. Paired model
changes use identical source draws. It measures source uncertainty conditional
on these fitted models, not training-seed uncertainty. Only a small number of
strong events are available on held-out sources; counts accompany all metrics.

Train-source errors are separate from held-out errors. Ten fixed probability bins
provide weighted mean predicted probability and observed positive fraction.
AP remains diagnostic; no AP-based selection, fitting or ensemble promotion.
Previously inspected source roles make these comparisons exploratory.

## Artifacts

- `coverage-v1/tables/support.csv`: retained event/source/row support by original
  train versus merged-test role and candidate stratum.
- `candidate-eligibility.csv`: all 184 candidates, accepted history counts and
  rejection-reason counts across scanned centers (reasons are not independent events).
- `comparison-v1/tables/train-and-test.csv`: all model/pool/domain/history/stratum
  scores. Undefined uncertainty for descriptive train errors is left empty.
- `paired-comparisons.csv`: combined minus original-pool performance on identical
  held-out rows, with source-bootstrap intervals.
- `training-selection-cv.csv`: NLL-selected training CV diagnostics.
- `calibration.csv`: fixed-bin reliability on the matched population.
- Saved probabilities, model hashes and PNG NLL/AP comparison plots preserve
  provenance and make subsequent analyses possible without refitting.
