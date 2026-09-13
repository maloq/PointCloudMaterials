# Proposal: improve local transition prediction

Status: proposed on 2026-09-13; no new training launched by this proposal.
The goal remains forecasting full local embedding trajectories, with useful predictions
of when their tracked center first becomes crystalline. Keep the existing PTM rule,
source splits and original benchmark alongside each new experiment.

The evidence motivating the order is in [RESULTS.md](RESULTS.md): AR/direct 9 ps
transition F1 is 0.502/0.498, while applying the current readout to actual future
embeddings gives 0.844. This is a diagnostic reference, not a proven achievable ceiling.
The observed embedding readout has 98.3% balanced state accuracy. Full-size models
optimize only embedding MSE and select checkpoints on validation embedding MSE.
Predicted change amplitudes are about two-thirds of observed amplitudes. Imminent
onset occurs in only 1.91% of the evaluated at-risk test windows; the training event
prevalence must be computed separately.

**Priority 1: predict the distribution of event time explicitly.**
Reuse the existing GRU history encoder and full-trajectory decoder. Attach a small
head using their history context and forecast features. Output 12 conditional onset
probabilities, one per future 0.75 ps interval. They define 13 outcomes: onset in
one of the 12 bins, or no onset through 9 ps. For hazard h_k,

    P(T = k) = h_k * product_{j < k}(1 - h_j)
    P(T > 12) = product_{j <= 12}(1 - h_j).

Train with negative log likelihood of the observed event bin or no-event outcome.
For partially observed follow-up, use the likelihood of surviving through the last
confirmed interval; do not call an unobserved future a negative. Existing complete
windows reserve the two subsequent frames needed to confirm the three-frame target.
Event losses use pre-first-onset windows; state/embedding losses may also use already
crystalline windows. No ground-truth future information enters model inputs.

This adapts the minibatch neural discrete-time survival likelihood described by
[Gensheimer and Narasimhan](https://pmc.ncbi.nlm.nih.gov/articles/PMC6348952/).
The resulting cumulative event probability gives logically consistent 3/6/9 ps risks.
Select alert thresholds on validation. Report a conditional event-time median and
interval with the probability of any event; do not force a time estimate on a
low-probability or censored case. The median minimizes absolute error for the model's
conditional distribution; it cannot establish that that distribution is correct.

First train only the new head with the existing forecaster frozen. This tests how
much the current single-frame-threshold readout leaves unused. Then fine-tune the
shared forecaster. Keep separate evaluations of (a) the new event head and (b) the
unchanged physical readout applied to predicted embeddings: an improved event head
alone does not establish better predicted embedding trajectories.

**Priority 2: make transition information matter in the training loss.**
Use the proposed joint objective

    L = L_embedding_MSE + lambda_event * L_event_NLL
        + lambda_crystal * L_crystal_state.

For a physically directed embedding auxiliary loss, first fit a differentiable binary
crystal classifier on real training embeddings and freeze its weights. Apply it to
predicted embeddings and compare with future PTM labels using binary cross entropy.
This classifier must produce logits; the existing ridge margins are not calibrated
logits and should not be substituted without fitting a suitable mapping. This loss
provides a gradient through the predicted embedding in a physically relevant direction.
Monitor whether it damages latent error or pushes predicted embeddings away from the
observed distribution. It is an ablation with a potential tradeoff, not a guaranteed
improvement. Retain an independent frozen evaluation readout for comparability.

Start with mean embedding loss, per-origin event likelihood, and mean state loss;
try lambda_event = 1 and lambda_crystal = 0.1 as initial settings, not established
optima. Inspect their training gradient scales before the paired pilot. Any subsequent
coefficient selection uses validation only. A free-standing state classifier on hidden
features is a useful cheaper comparison, but it does not directly constrain predicted
embeddings and must be described separately.

**Priority 3: expose the learner to transitions and difficult negatives frequently.**
Construct an index over the existing labeled training centers. A proposed sampler
allocates 50% of draws to onset within 9 ps, 25% to difficult negatives, and 25% to
background negatives. Difficult negatives include transient crystal episodes failing
the persistence rule and confirmed onsets just beyond the 9 ps horizon. Future
labels may define training strata, never prediction inputs. Define mutually exclusive
strata and preserve representation of all training sources and distinct atoms.

For probability training, weight each draw by its training-population stratum
probability divided by its sampling probability. This increases encounter frequency
and reduces sampling noise while preserving the intended population objective; it
does not magically increase the expected positive weight. Uncorrected balanced/focal
training changes that objective and would need a separate calibration experiment.
Validation and test retain natural prevalence. Continue embedding-only training on
the full existing unlabeled center cache where helpful; currently PTM labels cover
64 centers/source, not all 1,024 embedded centers/source.

**First pilot: four conditions, two seeds each.**

| Condition | What changes | Question |
|---|---|---|
| A | Freeze direct forecaster; fit event head | Can a learned temporal readout outperform the existing threshold? |
| B | A plus joint fine-tuning with embedding MSE + event NLL | Does transition supervision improve the shared forecast? |
| C | B plus frozen-classifier state loss on predicted embeddings | Do embedding trajectories preserve crystal information better? |
| D | C plus stratified, importance-weighted sampling | Does more reliable exposure to rare transitions help? |

Use the completed direct checkpoint first because it is slightly better at latent
MSE/state accuracy and quicker to run. Repeat the best protocol on AR afterward.
Begin with 6 ps history and 9 ps future to preserve the current target. Use two fixed
seeds and 2,000 optimizer updates per condition, validation every 200 updates, and
matched training source/center populations. Batch size and wall time should be set
from a short throughput measurement for the new losses; do not inherit a large
unlabeled batch size without checking the number of distinct labeled trajectories.
The initial coefficients above define the pilot; expand their validation grid only
if needed. These commands do not exist yet: implementation would extend
`src/training_methods/embedding_forecast/`, using the existing training entry point
with explicit protocol/configuration fields rather than copying a runner.

Select checkpoints on validation transition average precision, with event likelihood,
calibration and latent MSE also reported. Compare A–D and both original forecasters
on identical origins, with validation-selected operating points. A small extra
history-only event-head control can test whether forecast features improve the head
beyond using the same observed history directly.

**Later experiments, after the supervised baseline works.**

1. Compare 6 versus 12 ps history under the new transition objective. Earlier context
   experiments only established a 0.7–0.8% latent-MSE gain; that does not establish
   the size of the transition benefit. Avoid mixing context changes into A–D.
2. Extend the future to 18 ps in a separate matched experiment, retaining 3/6/9 ps
   event scores. Assess timing with true onset inside the predicted interval so
   late forecasts can be observed. The present exact-9-ps-lead assay puts onset at
   the last forecast frame, forcing detected errors to be early. An evaluation
   change alone is not model improvement; preserve the original 9 ps benchmark.
3. Add observations of the surrounding environment at the origin and in its past:
   a larger spatial neighborhood or nearby local embeddings. Test whether proximity
   to an approaching crystalline region supplies information absent from a single
   small local environment. This is a hypothesis requiring new feature preparation.
   No future neighbor choices or future structural labels may define the input.
4. If deterministic paths remain too smooth, predict a small mixture of complete
   trajectories with a shared component identity over all future frames. Train a
   mixture likelihood and derive transition risk from the weighted paths. This
   represents different possible futures rather than requiring their average to
   look like a realized transition. The rationale follows
   [Bishop's mixture-density formulation](https://www.microsoft.com/en-us/research/publication/mixture-density-networks/).
   The repository's single low-rank Gaussian is a useful uncertainty baseline but
   is still one mode. Mixtures need checks for component collapse and calibration;
   best-of-K trajectory error alone is not a sufficient success criterion.
5. Test lower embedding noise/frame dropout as a separate augmentation ablation.
   The existing augmentation may suppress weak precursors; this has not been shown.
   Keep true sampling times and forbid time reversal/warping as physical equivalents.

**Judge improvements on transitions, timing and calibration together.**

- Primary: event average precision at 3/6/9 ps and recall with thresholds selected
  for 0.5% and 1% validation false-positive rates; also report actual test false-positive
  rates, which need not equal validation values.
- Retain event F1, precision/recall curves, event/no-event likelihood and Brier scores
  at each horizon. Threshold calibration is fitted on validation, never test.
- Report conditional timing MAE alongside the fraction of all eligible events both
  detected and timed within 1.5 ps. Include one-origin-per-event fixed-lead results;
  overlapping windows are not independent events.
- Keep embedding MSE, physical crystal-state metrics of the predicted embeddings,
  and three-/nine-frame persistence sensitivity. State accuracy alone is insufficient.
- Use paired source bootstrap and both training seeds. A gain must survive source
  variation and cannot be claimed solely from a changed threshold, easier target or
  more favorable cohort.

The 27 existing test simulations have informed this proposal, so future comparisons
on them are retrospective benchmarks. Freeze choices using validation and confirm
with new independent simulation seeds before making a fresh generalization claim.
All proposed metric calculations need their own reviewed code/docs contract when
implemented; existing exported metric definitions must remain intact.
