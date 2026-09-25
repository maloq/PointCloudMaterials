# Three-picosecond onset prediction: next experiments

Proposal based on the completed encoder screens, updated24 September2026.
**Primary: source-weighted sustained-local-onset AP at 3 ps.** AP6 is secondary;
AP12 is contextual. Trajectory stability keeps the exact0.75 ps lag. One seed
per new fit. This proposal does not submit a new training queue.

## What changed when we inspected the correct horizon

| Existing model/readout | AP3 | AP6 | AP12 |
| --- | ---: | ---: | ---: |
| Onset supervision B, joint | 0.1593 | 0.1385 | 0.1280 |
| AP-ranking C, joint | 0.2578 | 0.1919 | 0.1901 |
| Relaxed teacher D, joint | 0.2599 | 0.1912 | 0.1427 |
| Noise-trained teacher E, joint | 0.2606 | 0.1708 | 0.1347 |
| Relaxed-input H, joint | **0.6576** | **0.3431** | 0.1890 |
| Relaxed-input H, fresh linear probe | **0.7616** | **0.3922** | 0.1946 |
| Wider context, late fusion, joint | 0.0130 | 0.0951 | 0.1548 |
| Hierarchy to12 Å, joint | 0.0158 | 0.2064 | 0.1270 |

H's source-bootstrap AP3 interval is[0.142,1.000]; its paired improvement over B
is[0.120,0.992], conditional on nonempty-positive resamples. Of2000 draws,52 have
no3 ps events and are excluded, not assigned AP0. The original small cohort has
11 fitting,2 tuning and 3 development3 ps positive windows. Thus H is a strong
candidate, not a reliably measured0.66-AP predictor. The joint checkpoints were
selected on AP12, and the fresh probes on NLL; none was selected retrospectively
by development AP3.

The matched Epi study also changes interpretation: final two-seed AP3 means are
VICReg 0.0481, Epi 0.0288 and Epi+variance 0.0269. Improving the Epi variant at 12 ps
did not improve it at 3 ps. Pause this objective sweep for short-horizon onset.

Evidence: [all replayed evaluations](../../output/encoder_research/onset-horizons-20260924/tables/all-models.csv),
[paired uncertainty](../../output/encoder_research/onset-horizons-20260924/tables/paired-AP3.csv),
[historical selectors and definitions](../../docs/metrics/onset_horizons.md).

## First: reuse the larger existing cohort

The existing registered `relaxed-encoder-large-test-20260921` collection has
current observed/relaxed inputs and saved original-MD delays. Reading its retained
`technical/assay/population.npz` gives the following counts; this uses the actual
delay field, not division of 12 ps event counts by four.

| Existing role | Eligible windows | Onsets by 3 ps | Event-bearing sources at 3 ps | Onsets by 6 ps |
| --- | ---: | ---: | ---: | ---: |
| Train | 10,825 | 79 | 46 | 188 |
| Selection | 5,306 | 39 | 13 | 79 |
| Calibration | 4,222 | 35 | 15 | 85 |
| Historical test | 11,256 | 53 | 24 | 135 |

These are available counts **before** auditing overlap with each encoder's
pretraining and the newer45-root screen. Preserve the original assignments;
publish which roots remain eligible for each independent claim. Already used
test sources remain reused evaluation. Use one common eligible cohort for every
matched comparison. If exclusions leave too few events, freeze a new cohort
from existing independent trajectories by root before fitting any model. No new
MD is required for this plan. Existing12 ps observation cadence can miss many
3 ps precursor windows; a fixed3 ps cadence from existing trajectories is a
subsequent data-extraction extension, not outcome-based positive sampling.

Retain causal noncrystalline eligibility and sustained confirmation. At 3 ps,
confirmation may require later frames than the prediction endpoint; these are
labels only. AP within 3 ps does not require3 ps advance warning. Also report
misses, event-level recall and first-alarm lead time to expose near-onset
recognition. Do not invent independent events by counting overlapping windows.

## Ordered experiment queue

Stage boundaries prevent spending the budget on architecture before verifying
input information and readout selection. The initial queue is deliberately
small:12 frozen-head fits, then six matched encoder fits, then three independent
single-change extensions only if supported. Reuse selected stages rather than
retrain controls for each extension.

| Stage | Matched experiment | Purpose and decision |
| --- | --- | --- |
| 1: twelve cheap head fits | Six inputs × linear and one-hidden-layer width64 heads: temperature-only; current physical descriptors; relaxed physical descriptors; frozen C export; frozen H export; concatenated observed/relaxed descriptors. | Determine whether H's advantage survives more events and whether descriptors already contain the useful information. All heads receive the same temperature conditions; other physical descriptors enter only their declared input arm. |
| 1, selectors from the same fits | Retain tuning-NLL-selected and tuning-AP3-selected states from each fitting trajectory. Use root-grouped fitting/selection folds when feasible, then reserve calibration roots. | Diagnose the repeated constant-risk MLP selections without expanding encoder capacity. Select on AP3, not development scores. Report each selector separately. |
| 2: six encoder fits | Three input/teacher variants × two ranking objectives: observed C; observed D with a relaxed teacher; relaxed H. Compare the previous AP12 surrogate with the same surrogate using3 ps labels/scores. All six use AP3 tuning selection and identical physical retention. | Isolate the ranking horizon from checkpoint selection and test whether the teacher transfers the relaxed-input benefit. Reuse width32,128-dimensional export,cuEquivariance and 2048 updates; one seed. |
| 3a: one new encoder, reuse Stage2 observed baseline | Observed-input relaxation-displacement teacher: predict center-relative, identity-matched observed-to-relaxed atom displacements, with invariant relaxed summaries as an auxiliary target. | A stronger test of whether useful quench information can be learned from observed geometry. Match atom IDs from existing full-cell pairs; do not subtract unordered cropped clouds. Prediction runs on observed inputs only. This is a hypothesis, not an established gain. |
| 3b: one new encoder, reuse Stage2 relaxed baseline | Gated observed+relaxed atom features, pooled once into the exported state. | Test whether instantaneous distortion adds information to relaxed structure. Keep exported width fixed and use present-time pairs only. Distinguish full-cell relaxation's broader information access from a learned local surrogate. |
| 3c: one new encoder, reuse Stage2 observed baseline | Weak gated late-fused surroundings, initialized to zero influence, with AP3 supervision. | Check whether surroundings can add short-horizon information without degrading the local encoder. Prioritize this over another expensive interleaved hierarchy because all previous context models were weak at 3 ps. |

For Stage1 use a single predeclared head learning rate1e-3, weight decay1e-4,
1024 updates, and natural/source-balanced loss. Compare selection rules without
different hyperparameter budgets. Include constant-risk and original frozen-head
predictions without fitting. Do not score a tuned readout and call it an encoder
improvement. For Stage2 keep the five-bin hazard NLL and 3/9/12 ps physical tasks
fixed; only the AP surrogate's cumulative score/positive bin changes. Tuning
AP3 is primary for all candidates, with NLL and earlier step as tie breakers.
Apply the physical-retention gate before AP selection. The current implemented
selector retains the earlier step on exact AP ties; the NLL tie-break is a
proposed refinement for the larger-cohort run, not a retroactive selection rule.

The ranking surrogate is our source-weighted adaptation of
[Smooth-AP](https://arxiv.org/abs/2007.12163), originally evaluated on retrieval
benchmarks; it supplies a differentiable ranking loss, not a crystallization
performance guarantee. AP/PR evaluation is particularly useful under class
imbalance; preserve and report natural prevalence when changing the cohort
([Saito and Rehmsmeier,2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)).

## Common outputs and promotion

- Primary AP3, secondary AP6 and contextual AP12, positive-window/root counts,
  Brier/NLL, root-bootstrap intervals, recall and achieved FPR at a
  calibration-selected5% threshold. No threshold is selected on test data.
- The actual exported embedding gets matched fresh probes and present-structure
  decoding. Calibration cannot turn a weak ranking into a useful one.
- Exact0.75 ps jumps and responsiveness near onset, dataset/within-track/movement
  spectra, and d95. These remain linear spectral dimensions, not nonlinear
  intrinsic-manifold estimates. Reject smoother but less responsive candidates.
- Input-noise tests at 0.1/0.5/1/3%3D displacement RMS relative to local spacing,
  with both raw Å and normalized latent response. Rebuild edges and support.
- Promote on the common source-held-out cohort, not this three-event screen or
  whichever horizon gives the largest number. Keep all comparisons and controls
  in the report, including unsuccessful or constant-risk selections.

Do not prioritize more width, stronger whitening, tensor pooling, pure Epi,
or aggressive slowness now. The observed evidence points first to relaxed
structure, horizon-aligned ranking and adequate event coverage.

## Implemented support versus proposed work

The shared trainer now accepts explicit `primary_horizon_ps`. It controls the
AP surrogate's labels/probabilities, tuning AP, primary reporting and bootstrap.
New ready-to-edit recipes are `configs/robust_onset/screen_ap3_20260924.json` and
`configs/spatial_hierarchy/screen_ap3_20260924.json`; they still target the small
screen and have not been submitted. The existing-corpus AP3 metrics were already
present;246 saved model/readout predictions have been replayed and verified.
The larger-cohort exports, head fitting and new architectures above are proposed,
not already implemented or trained.
