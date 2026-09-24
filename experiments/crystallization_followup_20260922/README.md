# Which observed information improves relaxed-MACE crystallization forecasts?

The [literature review](LITERATURE.md) motivates complementary original/relaxed
observations, actual-time structural changes and dense recent history. This
first wave tests information and objectives with frozen encoders before another
encoder architecture or generative-model sweep.

## Fixed protocol

Reuse exactly the archived relaxed comparison's 150 independent Al sources,
90/15/15/30 train/selection/calibration/test split and 9,396/3,625/2,721/7,654
eligible windows. Inputs remain causal. The main branch has three real archived
frames within 72 ps and 25 symmetric queries at center, 10 A and 20 A. Frozen
MACE checkpoints and their historical supports are unchanged. Labels are sustained
onset in the original MD, not labels reassigned after minimization. No new
simulation, relaxation or encoder extraction is needed.

One seed (20260919), batch128, head width128, two spatial/temporal blocks, AdamW
LR1e-4 with one-epoch warmup/cosine, 24-epoch ceiling, patience6 after epoch6.
All models predict the same 265-channel original-MD/reference-embedding state
every3 ps through96 ps and a 0.75 ps onset CDF. Teacher forcing decays over6 epochs;
selection and evaluation are fully open-loop.

Select every checkpoint using **source-balanced integrated Brier through12 ps**.
This differs from the old96 ps selection rule, so retrain controls and do not treat
an improvement over a historical selected checkpoint as a pure input ablation.
Physical and 96 ps performance remain reported; advance models only if they improve
the declared short score without more than10% degradation in physical validation
MSE relative to the newly trained cold control. There is no automatic test-based
promotion or stopping.

## Executable first wave

| Experiment | Motivation / controlled question |
|---|---|
| cold-control | Establish the relaxed reference under the shared short-horizon selection rule. |
| original-control | Compare original and relaxed inputs on exactly the same origins and new training budget. |
| cold-rates | Test whether physical-time secants and their actual lags improve irregular-history interpretation. |
| cold-dense-repeat | Measure the contribution of original current descriptors with the dense-history branch's capacity. |
| cold-dense-real | Test genuine original history at -12,-6,-3,-.75,0 ps against repeated current observations. |
| cold-dual-repeat | Measure extra spatial/temporal predictor capacity using a second independently trained relaxed branch. |
| cold-dual-original | Test complementary original embeddings against the same-capacity repeated-relaxed control. |
| cold-quench-difference | Test original-minus-relaxed physical/order/shell descriptors in their shared physical basis. |
| cold-dense-rates | Test whether dense observations and time-aware sparse history are complementary. |
| dual-dense-rates | Test whether original embeddings still add information beyond the dense physical descriptors. |
| dual-dense-rates-short | Add correctly censored12 ps likelihood and short physical losses to test horizon emphasis. |
| dual-dense-rates-no-clock | Remove absolute simulation age to test reliance on a protocol-specific condition. |

All single-branch variants instantiate the same auxiliary MLP. Dense real/repeat
have identical input width and parameter count; dual original/repeat also have
identical parameter counts and separate training-only normalizers. The dual branch
adds features from another checkpoint, not a subtraction between latent spaces.
Descriptor quench differences are valid because both sides use the same named
physical basis. Current fields include bond-order and coherence scalars already;
none of these variants supplies an explicitly reconstructed crystal front.

Report AP, Brier, event NLL and calibration at .75,3,6,9,12 ps, recall at a
calibration-selected5% FPR with realized test FPR, timing and missed-positive counts,
restricted-time error including survivors, physical path errors and the existing
sampled-center spatial scores. Paired Brier intervals resample whole sources1000
times. One seed and reused historical test sources make this exploratory; confirm
any selected improvement on previously untouched independent sources.

## Reproduction and findings

Config: [literature_followup_20260922.json](../../configs/crystallization_transfer/literature_followup_20260922.json).
Run `python -m src.research.crystallization_followup.queue prepare|verify|submit|report --config configs/crystallization_transfer/literature_followup_20260922.json`.
Execution details: [workflow](../../docs/crystallization_followup.md).
Results: [live comparison](../../output/crystallization_transfer/literature-followup-20260922/RESULTS.md).

Implementation verification precedes submission. Small smoke fits validate the
pipeline and are not scientific results. **Update this record after all12 fits
complete**, including paired contrasts, misses, physical trade-offs and actual
epochs. Do not infer winners from partially completed queues.

**All12 first-wave fits completed.** See [results and interpretation](RESULTS.md)
and the [28-fit optimization/ordered-region follow-up](SECOND_WAVE.md).

Validation passed:45 unit/regression tests, actual150-source paired joins,
future-target input-independence checks, finite forward/backward updates for all12
variants, and a complete checkpoint/evaluation smoke. The linked first-wave report
contains completed comparisons. Follow-up training is detached.
