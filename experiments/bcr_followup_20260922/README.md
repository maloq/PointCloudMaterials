# BCR: sample-specific conditioning and structural retention

Protocol fixed before this follow-up: 22 September 2026. One seed (20260921), one
GPU, eight hours. Uses existing BCR checkpoints and existing paired relaxations.
The independent-root G1 result remains a failure of its original 5% engineering
gate; this study neither revises that threshold nor claims an encoder improvement.

The review motivating this experiment observes that BCR reconstruction improves
while radial readouts worsen, and that much of the decoder advantage survives a
wrong code. The questions are whether conditioning is largely a global adjustment,
whether the final readout hides useful intermediate information, and whether the
code remains useful with a freshly initialized decoder.

## Fixed comparisons

1. Export the 64-dimensional center-plus-pooled scalar representation and its
   128-dimensional final readout at encoder updates 0, 1,000, 3,000 and 10,000.
   Verify final exports against the original assay. Fit radial (17), angular (4)
   and moment-Gram (144) readouts: 24 ridge/residual pairs, 48 readouts.
2. At updates 1,000 and 10,000, freeze the original decoder and compare true code,
   training mean, four strict and four unrestricted donor assignments, one fitted
   global code, and training-mean-centered amplitudes 0, 0.5, 1 and 2. This is an
   intervention assay; no encoder/decoder parameter changes. Fit the global code
   for at most 1,000 updates, selecting on two training tuning roots. Replay the
   original 384 anchors, two corruption draws and five noise levels exactly.
3. Audit transfer to paired observed and relaxed neighborhoods as specified below:
   36 ridge/residual pairs, 72 readouts. No future or crystallization labels.
4. Freeze encoders at updates 0, 1,000 and 10,000. Train three new conditional
   decoders from identical original decoder initialization, with matched 10,000
   updates, batch 256, microbatch 32, root-balanced sample streams, noise streams,
   optimizer and schedule. All 12 original training roots are used. Frozen encoder
   tensors are checked unchanged. Compare on the original corruption bank.

The queue follows this order. Decoder training checkpoints resume exactly; completed
probe pairs are reused. An interrupted small probe is rerun. If the allocation
expires, report incomplete work without shortening budgets or presenting it as a
matched completed comparison. There is no matched VICReg checkpoint in the pilot
artifacts, so none is claimed here.

## Probe selection and populations

On the original pilot, ten encoder-training roots fit scaling and readouts; the
remaining two training roots choose ridge penalty and residual duration. The six
development roots never enter fitting or selection. Ridge uses 11 log-spaced
penalties from 1e-6 to 1e4, with unpenalized intercept. A two-hidden-layer width-128
SiLU residual network starts with exactly zero output, is trained for up to 5,000
updates, and is evaluated every 100 updates on the tuning roots. Step zero (ridge)
is always eligible. A nonlinear readout cannot be selected with worse tuning
error than ridge, although its held-out error may still be worse.

Publish each radial measurement and moment-Gram degree l=0,2,4,6 separately,
plus errors by temperature and root. The original 384-anchor all/liquid populations
are identical because all satisfy observed q6<0.35; they are not independent
confirmations. Root-bootstrap intervals for denoising comparisons condition on
one training seed and do not estimate seed uncertainty.

## Paired relaxed structural audit

Use the existing expanded paired-relaxation collection, not newly simulated data.
Select outcome-blind, seeded, fully available roots from historical training and
selection roles at 400, 450, 500, 510 and 520 K. Per temperature, six training roots
supply five fitting roots and one tuning root; three historical selection roots
supply development observations. Totals: 25 fitting, five tuning, 15 development
roots. Exclude original BCR development roots from transfer fitting, and verify no
transfer development root was used for BCR encoder training. Historical test and
calibration roots are untouched. Freeze selected roots before job submission;
later producer completions cannot alter the sample.

Use frames 64, 224, 368 and 512 (48, 168, 276 and 384 ps), and the first 16 tracked
centers from the existing outcome-blind plan. This produces 2,880 pairs: 1,600
fitting, 320 tuning and 960 development observations. Observed and relaxed patches
are independently extracted at full radius 8 Å around the same atom identity,
retaining all neighbors inside support and validating the 256-atom cap. The older
nearest-80 clouds do not define this audit's support.

At frozen encoder updates 0 and 10,000, compare both feature representations on
observed→observed, relaxed→relaxed and observed→relaxed structural targets. The
latter measures prediction of same-frame relaxed structure from observed structure,
not temporal prediction. Scaling and readout selection use only the designated
transfer fitting/tuning roots. Liquid subsets use observed-domain q6<0.35 for all
three tasks. Report temperature-specific results alongside pooled results so
between-temperature differences do not stand in for local information.

Relaxed full cells are archived float16 coordinates with exact identities and
float32 boxes, fixed-box FIRE tolerance <=0.01 eV/Å and Lee2003 Al MEAM provenance.
Verify source manifest, timestep, atom IDs, potential checksums, cell dimensions,
archive checksums and conversion records. This precision is adequate for a
separate structural audit, not a substitute for the original high-precision weak-
noise denoising population. Full-cell quenching includes context outside the local
patch; observed→relaxed recovery need not be deterministic from local inputs.

## Decision after diagnostics

If pooled features remain useful while exports deteriorate, investigate the final
readout first. If stronger probes recover the difference, distinguish accessibility
from information loss. If deterioration persists, design a separately labeled
three-arm comparison of BCR, code-only geometry reconstruction and their hybrid,
retaining independent withheld structural measurements. Do not automatically train
that objective before interpreting this audit, or use a trained target as its own
independent validation.

[Active recipe](../../configs/bcr/followup_20260922/study.json),
[execution and reproduction](../../docs/bcr_followup.md),
[metric definitions](../../docs/metrics/bcr_followup.md).

[Completed results and interpretation](RESULTS.md).
