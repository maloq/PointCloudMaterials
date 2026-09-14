# Observed-history forecast sweep

All 24 compact fits completed successfully in **990987.1 on nodesumo01 / H100**: observed
histories 0, 1.5, 3, 6, 12 and 24 ps; direct and autoregressive predictors;
two seeds, 16 epochs each. All predict the same next 9 ps.

- [Live training log](technical/training/command.log) and [execution record](technical/training/execution/run_record.json).
- [Automatic comparison status](technical/comparison/queue_status.json).
- [Experiment protocol and commands](../../../experiments/forecast_context_20260913/README.md).
- [Matched-window verification](technical/window_pairing.json) and [tests](technical/tests.log).

Read [RESULTS.md](RESULTS.md),
[context-quality.csv](tables/context-quality.csv), [metric definitions](tables/METRICS.md),
[context-quality.png](plots/context-quality.png) and [horizon-errors.png](plots/horizon-errors.png).
Training and automatic comparison finished at 13:43 UTC on September 13.
[Interpretation and limitations](../../../experiments/forecast_context_20260913/RESULTS.md).
