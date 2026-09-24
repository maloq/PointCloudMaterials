# Experiments aimed at better crystallization AP

Proposal, September 24, 2026. No new training campaign is submitted by this plan.
One seed for screening, as requested; seed replication would be a later explicit
confirmation step. Use existing observations and relaxation products first.

## Why change the objective?

The latest native MACE geometry run reaches 12 ps AP **0.1684 / 0.1782** in two
seeds, versus **0.1090 / 0.1245** at the lower encoder learning rate. These are
frozen, current-physics-conditioned linear hazards on the reused 15-root
development split. The encoder learns present geometry/order; the readout is
trained and selected by hazard likelihood. Neither is directly optimized for AP.
See [matched results](../../output/encoder_research/parameter-search-20260923/tables/training-comparison.csv)
and the [actual checkpoint selector](../../src/research/structural_state/evaluation.py).

This does not establish that the linear head or likelihood selection causes the
low AP: the following controls distinguish readout, objective and missing input.

Earlier evidence suggests useful starting points:

| Evidence | Result | How it informs this plan |
| --- | --- | --- |
| [Larger relaxed assay](../../output/relaxed_encoder/large-test-20260921/RESULTS.md) | Relaxed geometry MLP AP **0.3911** at 12 ps; cold encoder MLP **0.2529** | Reproduce fixed relaxed descriptors as a strong baseline; do not assume learned embeddings retain all useful precursor information |
| [Descriptor study](../../output/local_predictability/research-summary-20260917/RESULTS.md) | At 9 ps: snapshot **0.2499**, history **0.3508**, repeated frames **0.2580**, broad context **0.4586** | Test history and spatial context separately before scaling an atom-history model |
| [Repaired structural assay](../../output/structural_state/repaired-review-20260923/README.md) | Relaxed structural encoder MLP about **0.268** at 12 ps | Include a stronger readout and relaxed input control |

These are different horizons, input access, readout conditions, fitting exposure
and evaluation grids. They are **not** evidence that 0.39–0.46 is achievable on
the current small cohort, nor directly comparable improvements over 0.168.

## Priority experiment sequence

| Stage | Matched variants | Scientific question / promotion |
| --- | --- | --- |
| **1. Frozen readout screen** | Current physics alone; best geometry encoder alone; encoder + physics; relaxed geometry alone. Each with linear and one-hidden-layer width-64 readouts; fixed regularization grid. Train each hazard once and retain both NLL-selected and AP12-selected checkpoints. | Is precursor information already accessible, and is the selection criterion limiting ranking? Reuse cached embeddings; no encoder training. |
| **2. Ranking objective** | On the best two Stage-1 input families: binary 12 ps BCE baseline, focal loss (gamma 2), one AP-surrogate objective. Keep readout capacity, fitting rows and tuning budget equal; AP-select all three. | Does aligning the loss with rare-event ranking improve AP beyond changing checkpoint selection? Do not confuse generic pairwise AUC optimization with AP optimization. |
| **3. Information available to the predictor** | Winning snapshot packet; actual 12 ps descriptor history; separately trained repeated-current packet of matching dimension; wider current spatial context; history + wider context only if each contributes. Relaxed-only vs observed-only vs concatenated observed/relaxed packets. | Does new input add predictive information? All historical samples must precede the origin. Full-cell relaxation uses current geometry but accesses broader context; label it explicitly. |
| **4. Train the encoder for onset** | Frozen-head winner; fine-tune native MACE with onset loss; same fine-tuning plus present-geometry/order retention; observed and relaxed input as separate branches. | Does task supervision improve the exported state rather than only the attached head? Start from the same checkpoint and give matched update budgets. |

Stage 1 is the first experiment I would run. Use head learning rates {1e-3,3e-4}
and weight decay {1e-4,1e-2}, width 64 for the nonlinear head; leave all other
hyperparameters fixed. This is 4 input families × 2 readouts × 4 settings = 32
cheap fits, with two selectors computed from the same training trajectory. Avoid
expanding this grid before seeing its source-level variability.

Stage 2 adds 2 × 3 = 6 head fits at the Stage-1 selected setting. Focal loss
downweights easy examples; it is an ablation, not an AP guarantee ([original
paper](https://arxiv.org/abs/1708.02002)). For direct AP optimization, evaluate a
source-weighted version of the published stochastic AP surrogate, with audited
sampling weights and positive-example state ([Qi et al., NeurIPS 2021](https://proceedings.nips.cc/paper_files/paper/2021/hash/0dd1bc593a91620daecf7723d2235624-Abstract.html)).
Its algorithm name “SOAP” is unrelated to the atomic SOAP descriptor. A vanilla
uniform-example implementation optimizes a different population from our
source-weighted metric; test this before accepting the implementation.

Stage 4: initialize from high-LR geometry MACE; test encoder LR {1e-5,1e-4},
head LR 3e-4, 4096 updates, one seed. Start with onset-only and onset + present
retention; calibrate the retention coefficient from fitting loss/gradient scales,
then fix it before validation. Keep the frozen-head control. Only if this works
should we train an expensive interleaved atom-history model. Do not start with
larger width, an extra information bottleneck, or aggressive slowness penalties.

## Fix the evaluation before searching

1. Primary metric: **source-weighted AP for sustained local onset within 12 ps**,
   at natural eligible-window prevalence. Keep current-state recognition separate.
   Report 3/6/9 ps as secondary horizons, not as a route to choose the best-looking
   number. Preserve three-frame confirmation and end-of-trajectory follow-up.
2. Use the existing larger natural 12 ps grid where matching inputs are already
   available: 11,256 historical test windows / 338 local onsets / 30 sources.
   Reproduce descriptor and encoder baselines on exactly the same row IDs. For
   head-only comparisons fit on the same training exposures too. New exports of
   existing coordinates are allowed; no new molecular dynamics is required.
3. Audit parent/root ancestry against every encoder's pretraining. A frozen
   encoder trained on a proposed test root makes that test transductive. Use only
   disjoint roots for an independent confirmation claim. All repeatedly inspected
   test/development sets remain **reused exploratory evaluation**; do not rename
   them fresh holdouts. New untouched confirmation data can be selected later
   from existing collections after that audit.
4. Select hyperparameters and checkpoints on tuning roots only, primarily by
   AP12; break exact ties by NLL, then earlier checkpoint. Prefer root-grouped
   cross-validation within the fitting/selection pool to repeatedly optimizing
   five small tuning roots. A fold with no events has undefined AP; report it,
   pool out-of-fold predictions, and retain whole-root splits. Never tune on the
   current 15 development roots or historical test scores.
5. Preserve source balance. Natural/source-balanced BCE is the baseline. If
   oversampling positives for optimization, log inclusion probabilities and
   restore the declared source/population weights for the intended loss. A
   deliberately class-weighted loss is a different objective, and its raw
   sigmoid scores are not calibrated event probabilities.
6. Calibrate risk on separate natural-prevalence calibration roots after ranking
   selection. Calibration cannot repair a bad ordering. Report AP, PR curves,
   Brier/NLL, recall at the calibration-chosen 5% false-positive-rate threshold,
   precision at that threshold, lead time, missed events, and timing-with-misses.
   Cluster overlapping positives by local atom onset for event-level recall.
7. Compare candidates with paired temperature-stratified **whole-root bootstrap**
   intervals for AP differences, including valid bootstrap replicate counts.
   Empty-positive replicates are undefined, never zero. One seed does not give
   training-seed uncertainty. Prioritize a useful absolute AP gain (e.g. +0.03,
   an engineering target, not a promised effect) and inspect error cases.

## Preserve dynamics while improving prediction

Use the [new stability/dimension metrics](../metrics/embedding_dynamics.md) next
to AP. Report whole-export covariance rank, within-track rank, uncentered movement
rank, centered fluctuation rank, spectral tails, per-track coverage, and physical
lag jump distributions. Separate noncrystalline, temperature and pre-onset groups.
Do not select a smoother or lower-rank embedding if it loses onset responsiveness
or present information. The current four-snapshot screen cannot establish
short-time smoothness; the promoted checkpoints need dense inference on existing
0.75 ps trajectories with matched atom IDs before that conclusion is possible.

Reproduction of the added metrics and existing-export results is documented in
[the dynamics guide](embedding_dynamics.md).
