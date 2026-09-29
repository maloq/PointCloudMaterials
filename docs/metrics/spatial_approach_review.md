# Spatial approach: post-hoc review (26 September 2026)

This analysis reuses immutable predictions from the five completed spatial
readouts. It fits nothing, chooses no checkpoint and does not change the original
8 Å alarm protocol or its exported metrics. It verifies source/sample/path
alignment and input checksums. The original definitions are in
`spatial_approach.md` and its frozen per-run export.

`pointwise.csv`: six-bin distance NLL gives each test source equal weight, with
95% percentile intervals from 5,000 bootstrap draws of the 30 source means.
Brier and AP at distance ≤8 Å are the original equal-source diagnostics.
`paired-nll.csv`: subtract reference NLL from model NLL within each of the same
30 sources, then average. Negative is better. Bootstrap resamples paired source
differences. All intervals condition on a single training seed, fixed fitted
models and the original calibration data; they omit training-seed uncertainty
and are not corrected for multiple comparisons.

`visibility-subsets.csv`: each subset is identical across models and uses the
context visibility flags, even for local-only models. Source weighting is
recomputed inside each subset; these numbers are not additive contributions to
the full NLL. Missing PTM crystal means no FCC/HCP/BCC-labelled atom in the actual
input patches, not proof of an ideally disordered liquid. Zero positive examples
within 8 Å prevents measuring near-crystal recall in that subset.

`alarm-radius.csv` and `paths.csv`: for each of all five fitted cumulative
probabilities P(distance ≤ R), R=4/8/12/20/32 Å, independently set a threshold
from the same calibration away paths. Threshold is the higher empirical 95th
percentile of the maximum consecutive-pair minimum probability. Alarm is strict
exceedance at the second observation. No test tuning or score/threshold selection
is performed. Radii other than 8 Å are exploratory diagnostics suggested after
seeing the original results; report every radius, not a best-on-test radius.

Pooled warning recall uses all 495 test approach paths, including misses. False
alarms use all 292 test away paths. These paths span 28 sources. Separate
source-mean rates and bootstrap intervals average within source first; their
intervals must not be attached to pooled-path rates. Conditional median warning
distance excludes missed paths and must be read with the miss rate. Early clear
alarms require distance >8 Å and both contributing observations clear of either
reference crystal or all PTM crystal. Raw counts use all approach paths as their
denominator, not only visible/clear opportunities. First-reference-visible
distance is an instantaneous label-side diagnostic, not an operational predictor
or an upper bound on liquid-structure information. It uses one observation,
whereas alarms require two. Away paths stay beyond 32 Å; they do not characterize
false alarms for tangential or near-miss scans.

`scan-support.csv`: point counts quantify the availability of clear observations
on held-out approach scans, with a separate count for 8<distance≤32 Å. Repeated
positions are not independent trials. `fixed-versus-scan.csv`: unweighted mean
8 Å probabilities in distance bands, reported separately for the original
at-risk observations and controlled approach scans. Infinity is excluded from
these finite-distance bands, and scan endpoints include crystalline atoms
outside the original currently-liquid training population. These are descriptive
distribution-shift diagnostics, not matched population comparisons.

The figure shows source-bootstrap NLL intervals and pooled recall curves for
the original 8 Å and exploratory 20 Å alarms, with actual held-out false-alarm
rates. Source-bootstrap seed is 20260926 throughout. No inference of temporal
lead time, nucleation prediction or autonomous-navigation performance is made.
