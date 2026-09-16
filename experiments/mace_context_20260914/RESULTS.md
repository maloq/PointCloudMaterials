# Completed MACE context and center-readout pilot

Both requested changes remove the finite nearest-neighbor membership jump in
the tested crossings. Smooth inner pooling is the stronger candidate for relaxed
patch topology. The tracked center is stronger for several first-shell physical
changes. Neither improves instantaneous hard-80 TDA reconstruction over the
original encoder.

The [smoothness follow-up](SMOOTHNESS.md) quantifies temporal variation separately:
at 0.75 ps, smooth inner pooling reduces normalized squared embedding change
44.21%, while the center has 2.81 times the control's change despite removing
the membership jump. The direction survives normalization by within-frame spread.

All three matched eight-epoch VICReg continuations completed on the two L40S
GPUs in node57 allocation 992489. The evaluation covers four frozen readouts
and three trained readouts, 5,760 anchors, six held-out test simulations,
144 tracked temporal series, and 72 controlled boundary crossings. The separate
controllers completed; both GPUs were verified idle afterward.

| After matched training | Hot TDA test MSE | Relaxed TDA test MSE | Crossing / natural 0.75 ps embedding energy |
|---|---:|---:|---:|
| Original 80-node mean | 0.017150 | 0.035616 | 0.03671 |
| Complete context, smooth inner mean | 0.041476 | 0.030984 | 1.41e-9 |
| Complete context, tracked center | 0.078912 | 0.046926 | 5.40e-10 |

TDA errors are balanced H0/H1/H2 descriptor MSEs; lower is better. The crossing
uses epsilon 0.0001 Angstrom. The smooth representations' squared differences
decrease approximately quadratically with displacement, while the original
encoder approaches a finite jump. This result survives continued training.

Smooth inner pooling reduces relaxed-TDA MSE **13.01%** versus the matched
trained control; the paired six-source bootstrap 95% interval is **9.82–16.00%**.
Every test source improves. Hot-TDA error increases to **2.42 times** the control.
The center's relaxed-TDA error is 31.75% higher and its hot-TDA error is 4.60
times the control. Eight additional VICReg epochs do not recover the lost
hard-80 topology information. Smooth-inner frozen and trained errors differ by
less than 0.3%; center training improves its own relaxed error by 5.89%, while
remaining worse than the mean control.

The TDA information generalizes across sources. Smooth-inner relaxed-TDA
train/validation/test errors are **0.026533 / 0.029628 / 0.030984**, versus
**0.858168** test error after shuffling training labels. Removing frame-wide
differences still leaves **87.28%** of within-frame relaxed-TDA variation explained,
versus **85.97%** for the control. These controls support real descriptor
information rather than simple sample memorization. They do not establish that
VICReg learned all this information: the earlier initialization audit found
substantial topology information before VICReg training.

The physical tradeoff depends on the observable. At 0.75 ps, the center reduces
q6 increment error versus persistence by **21.17%**, compared with **11.17%**
for the control. Density increment reduction is **62.21% versus 26.60%**;
mean first-shell distance is **96.95% versus 82.26%**. Smooth inner pooling is
weaker for these immediate local changes. For hot-TDA increments, reductions are
82.55% / 61.54% / 27.43% for control / inner / center. Longer lags through 12 ps
are retained in the machine summary.

The exact TDA target also retains a discontinuity at the controlled rank swap:
its jump approaches **2.51%** of natural 0.75 ps TDA increment energy. A perfectly
smooth representation cannot exactly reproduce both sides of a discontinuous
hard-80 target. This accounts for part of the target-alignment issue, not the
entire observed error difference.

Implementation uses full weight to 5 Angstrom and a nonnegative quintic taper to
zero at 7 Angstrom. The native two 5 Angstrom message layers have complete
context; the 18 Angstrom candidate sphere includes an augmentation margin.
Only the exact ancestors of readout nodes are computed. Both scalar blocks are
retained, so every representation is 256-dimensional. Center identity is recovered
from tracked atom IDs; periodic geometry and original neighbor ties are verified.

Training uses the original spatial/temporal VICReg loss, projector and augmentation
settings, a fresh AdamW optimizer at constant 1e-4, batch 256 and gradient replay
in microbatches of eight. All modes receive matched augmentation draws. Lowest
validation VICReg loss selects epochs **6 / 8 / 8** for control / inner / center.
No TDA labels enter encoder optimization. Ridge coefficients and scaling use
training sources, with penalties selected using validation sources only. All final
optima lie inside the extended 1e-14 to 1000 penalty grid.

Validation: all **17 selected tests** passed across local and two-GPU execution,
including the non-default-device loading test. Real-checkpoint GPU checks verify
full/pruned graph equivalence, rotation, permutation, extra context and replay
gradients. Full-graph relative squared differences are about 1e-13; replay's
relative L2 gradient difference is below 0.007%. All temporal point sets match
the original labels, including 84 harmless point-order differences from tied
distance queries. The smallest box length is 106.21 Angstrom, safely larger than
twice the candidate radius.

This is a one-initialization warm-start pilot on an exploratory cohort used in
earlier diagnostics. It is not a full training-seed study or a downstream forecast
benchmark. For forecasting, smooth inner pooling is a candidate when relaxed
structure is the priority; the center readout is a candidate for local atomic
observables. A newly identified embedding cache and fitted forecaster are needed
to establish prediction gains.

![Completed comparison](../../output/mace_context/forecast-seed20260910-pilot-20260914/plots/context-comparison.png)

[All scores](../../output/mace_context/forecast-seed20260910-pilot-20260914/tables/comparison.csv),
[TDA label crossings](../../output/mace_context/forecast-seed20260910-pilot-20260914/tables/label-crossing.csv),
[full machine summary](../../output/mace_context/forecast-seed20260910-pilot-20260914/technical/summary.json),
[metric definitions](../../docs/metrics/mace_context.md),
[checkpoint/API usage](../../docs/mace_context_encoder.md),
and [frozen-only findings](FROZEN_RESULTS.md).
