# Completed consecutive-motion comparison

**Discarded approach (16 September 2026):** replacement-embedding training on
frozen encoder features is no longer pursued. Scientific results and exact recipes
are retained. Embedding forecasting and native encoder training remain active.
See [scope and historical reproduction](../../docs/discarded_frozen_encoder_maps.md).

The 44 fits and evaluation completed in 1,701.6 seconds. Of 40 learned-map
candidates, **none passed the validation information-retention gate**, and none
met the requirement that both overall and low-order original-pair RMS jumps be
at most 0.10. The saved validation selection is null. No replacement is accepted.

This was the earlier frozen-feature experiment: MACE was fixed, and a 32D/64D
snapshot map was trained on its structural features. It must not be presented as
a smoothness sweep of the actual new MACE encoder. The separate native-encoder
data-amount study updates MACE itself with one fixed set of motion-loss weights.

## Controlled comparison

These are development-test results for the complete **64D / eight-direction**
comparison. Ranges show both seeds, not confidence intervals or a selection of
the most favorable configurations. Reference retains 256 dimensions. Every learned
map receives physical supervision. Direction and bending additions below also
include direct slowness.

| Training constraints | Overall RMS jump | Low-order RMS jump | Worst physical-error ratio | Motion energy in eight learned directions |
| --- | ---: | ---: | ---: | ---: |
| Unchanged reference | 0.461 | 0.742 | 1.00x | 76.0–76.1% |
| Physical supervision only | 0.626–0.634 | 0.860–0.867 | 1.07–1.09x | 61.1–61.7% |
| Direct slowness | 0.470–0.485 | 0.738–0.800 | 1.12–1.16x | 72.2–75.3% |
| Slowness + local directions | 0.449–0.460 | 0.701–0.724 | 1.14–1.15x | 90.6–91.1% |
| Slowness + bending | 0.381–0.437 | 0.662–0.783 | 1.25–1.27x | 81.6–83.7% |
| All three constraints | 0.366–0.419 | 0.637–0.738 | 1.30x | 94.9–95.1% |

Jump columns use the original matched pairs at 0.75 ps and original training
reference identities. Low-order selects both endpoints and the reference
population using group qbar6 <0.30; this is a disordered proxy, not a PTM phase
classification. The direction column uses the separate consecutive-sequence
cohort and predicts directions from the current state only. Physical ratios take
the worst of four target families, two populations and two cohorts, relative to
the matching reference. The permitted ratio is 1.10; acceptance is decided on
validation, where even the closest candidate had a ratio of 1.123. Seven
candidates passed the physical gate on development test alone, which does not
override their validation failure.

## What improved and what failed

Shared direction constraints reliably concentrated displacement energy: the
combined 64D models reach approximately 95% in eight predicted directions. This
is evidence about motion organization in the learned representation, not proof
of a physical manifold or of a smooth path along it.

The combined models reduce some original-pair jumps but lose information. Their
largest physical penalty is low-order instantaneous H1 topology, about 30% higher
error on development test and 38–40% on validation. The best validation low-order
jump anywhere in the sweep belongs to `combined-d32-r4-t0.03-s20260917`: validation
jump 0.504 and worst error 1.596x; development-test jump 0.627 and worst error
1.454x. It is an illustrative failed tradeoff, not an accepted model.

Sequence bending does not establish improved liquid-like smoothness. In the
combined 64D models its low-order normalized RMS is 1.312–1.762, versus 1.253 for
the unchanged reference. Their velocity-change ratio also remains near 2.96–2.98,
versus 2.96 for the reference. Confining changes to a few directions has not made
the path consistently turn or accelerate less. This comparison does not prove
that physical evolution sets an irreducible smoothness floor.

## Evidence and next distinction

Results: `/store/PERSO/vmorozov/analysis/mace_local_motion/sequences-20260916/`.
`tables/comparison.csv` contains both splits and all variants;
`tables/sequence_motion.csv` contains physical-lag motion statistics. Each variant's
`technical/.../evaluation.json` preserves per-family errors and independent probe
results. Exported `tables/METRICS.md` preserves the exact protocol and source hashes.

For the actual new encoder, a controlled comparison of physics-only, direct
slowness, direction and bending additions remains necessary to isolate their
effects. The ongoing [data-size study](../mace_data_amount_20260916/README.md)
holds these weights fixed and cannot replace that comparison. No additional
training was launched while producing this status report.

Definitions: [normalized jump](../../docs/research_glossary.md#normalized-rms-jump),
[local motion basis](../../docs/research_glossary.md#local-motion-basis), and
[temporal bending](../../docs/research_glossary.md#temporal-bending-penalty).
