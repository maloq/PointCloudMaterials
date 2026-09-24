# Completed BCR follow-up

The one-seed queue completed all stages in 83.1 minutes. The analysis concludes that
BCR improves reconstruction with a fresh decoder but makes radial structure harder
to recover from the embedding. A training-fitted constant almost reproduces the
original decoder's performance. Stronger probes and paired relaxed-data transfer
preserve the structural concern. Intermediate pooled features reduce, but do not
remove, the deterioration.

- Exported radial RMSE: +31.5% on the original melt assay, +20.7% on observed
  transfer data and +16.0% on relaxed transfer data. All development roots worsen
  in each of these three comparisons.
- Fresh-decoder noise MSE at σ/d0=0.12: 3.07% lower with the final trained encoder
  than with the initial encoder; the 1,000-update encoder already gives 2.78%.
- Correct code versus one optimized constant: 0.166% noise-MSE reduction in the
  original final decoder at σ/d0=0.12.
- Pooled observed→relaxed radial readout improves modestly (0.97%); exported
  observed→relaxed radial improvement is not established.

The original G1 outcome is unchanged. Root intervals condition on one fitted seed.
All/liquid populations coincide even in the transfer assay; this is not independent
phase-specific or crystallization validation. Recommend the fresh-decoder constant
control, followed by a matched BCR/code-only-geometry/hybrid comparison with withheld
structural measurements. No new encoder training was launched by this analysis.

[Full report, plots, uncertainty and limitations](../../output/bcr/conditioning-audit-analysis-20260922/README.md).

Reproduce numerical analysis with:
`python -m src.research.bcr_followup_analysis --config configs/analysis/bcr_followup_20260922.json`.
