# Observed history improves forecasts, with diminishing gains after 6–12 ps

All 24 fits completed successfully on September 13, 2026. Training ran from
11:57:32 to 13:43:07 UTC; automatic comparison completed at 13:43:35 UTC.
Results average two fitted seeds and equally weight 27 held-out sources, using
663,552 identical test windows per fit. Smaller is better for standardized MSE.

| Observed history / ps | AR path MSE | Direct path MSE | AR gain vs trained anchor | Direct gain vs trained anchor |
| --- | ---: | ---: | ---: | ---: |
| 0, anchor only | 0.225769 | 0.228534 | 0% | 0% |
| 1.5 | 0.214840 | 0.216167 | 4.84% | 5.41% |
| 3 | 0.211791 | 0.212719 | 6.19% | 6.92% |
| 6 | 0.209411 | 0.210087 | 7.25% | 8.07% |
| 12 | 0.207835 | 0.208616 | 7.94% | 8.72% |
| 24 | 0.207421 | 0.208161 | 8.13% | 8.91% |

At 24 ps, the paired source-bootstrap 95% gain interval versus trained anchor is
7.31–9.03% for AR and 8.11–9.80% for direct. These intervals condition on the two
fitted seeds. The complete curves improve monotonically in this tested range,
but extending 6 to 12 ps reduces full-path error by only 0.75% for AR and 0.70%
for direct. Extending 12 to 24 ps reduces it by another 0.20% and 0.22%.
Six ps obtains about 89%/91% of the respective total anchor-to-24 ps improvement.

The +9 ps errors likewise fall from 0.243112 to 0.223363 for AR and 0.246494 to
0.224322 for direct. The separate late-bin (6,9] mean errors fall from 0.112489
to 0.093150 and from 0.115831 to 0.094139. Longer history helps late predictions;
24 ps is not uniformly best at every individual future frame: its first-step
error is slightly higher than 12 ps for both models.

AR has lower full-path error at every tested context, but its advantage over
direct at 24 ps is just 0.36%, and AR has more parameters. This is not a
capacity-matched architecture test.

The models do more than repeat an unweighted history mean. That baseline is best
at 6 ps among the tested lengths (0.240929 MSE) and worsens at 24 ps (0.259456),
where learned AR/direct predictions improve further. Reversing the past while
preserving the anchor increases 24 ps model errors by 6.81%/7.84%. This demonstrates
input-order sensitivity under an intervention; it is not a substitute for a
separately trained mean-only control when claiming that temporal ordering is
necessary.

Twelve ps is a practical next context for a larger-model comparison: it retains
almost all the measured benefit of 24 ps with fewer observed frames. Twenty-four
ps achieves the lowest average path error in this sweep. These conclusions remain
conditional on compact models, sparse matched anchors, two seeds and the fixed
16-epoch training budget. Most selected checkpoints are at the final epoch, so
the sweep does not establish asymptotic convergence or an optimal physical memory
length. Test sources were previously examined; findings are exploratory.

Evidence: [complete machine-readable scores](../../output/embedding_forecast/context-pilot-20260913/tables/context-quality.csv),
[paired comparisons and checkpoint provenance](../../output/embedding_forecast/context-pilot-20260913/technical/comparison.json),
[context plot](../../output/embedding_forecast/context-pilot-20260913/plots/context-quality.png),
[horizon plot](../../output/embedding_forecast/context-pilot-20260913/plots/horizon-errors.png),
and [exported metric definitions](../../output/embedding_forecast/context-pilot-20260913/tables/METRICS.md).
Incremental gains above are `1 - longer_context_mse / shorter_context_mse` from
the linked table; the fraction of total benefit is
`(anchor_mse - six_ps_mse) / (anchor_mse - twenty_four_ps_mse)`.
See [README.md](README.md) for exact configurations and reproduction commands.
