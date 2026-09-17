# Partial seed-20260918 comparison at user-requested stop

Only the original-loss H0 and H12 models completed 12,000 updates. H48 and the
repeated-current control were not run. This two-model analysis is not the full
four-model collector. It uses exact matched source/center/anchor rows and the
existing 500-resample source bootstrap, with seed 20260918.

`paired_gain_over_snapshot` is snapshot minus H12 for each score; positive
favors H12. Test NLL gain is −0.0024351 [−0.0228464, +0.0187907]; future MSE
gain is +0.0104141 [+0.0030149, +0.0189585]. Source intervals do not cover
training-seed uncertainty. The primary NLL comparison is unresolved.

[Metric table](tables/partial-comparison.csv), [definitions](tables/METRICS.md),
[input hashes](technical/inputs.json), and
[full stop update](../../research-summary-20260917-stopped/RESULTS.md).
