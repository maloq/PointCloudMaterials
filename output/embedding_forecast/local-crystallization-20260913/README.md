# When does a tracked local environment become crystalline?

[Read the completed research report](../../../experiments/forecast_crystallization_20260913/RESULTS.md).

AR has useful local transition skill: 9 ps event F1 0.502 versus 0.196 for persistence.
Warning precision is 49.0%, recall 51.5%, and correct-warning timing MAE 2.39 ps.
These pool origins at different distances from onset. At exactly 9 ps lead, only
25.7% of eligible events are detected and 4.0% are also timed within 1.5 ps.
The held-out assay covers 1,728 local trajectories in 27 independent simulations.

- [Prediction comparison](plots/prediction-quality.png)
- [Tracked local examples](plots/local-trajectories.png)
- [Future-state accuracy](tables/future-state.csv)
- [Upcoming local transitions](tables/local-onset.csv)
- [One forecast per event at fixed lead](tables/fixed-lead.csv)
- [Temperature breakdown](tables/by-temperature.csv)
- [Event and persistence census](tables/local-events.csv)
- [Paired event F1 improvements](tables/paired-event-f1.csv)
- [Exact metric definitions](tables/METRICS.md)

`technical/` retains labels with exact tracked atom IDs, frozen selected checkpoints,
the training-only crystal readout, paired forecast score paths, configs, scientific
source copies, software versions, provenance hashes and execution logs. Files are
research artifacts; source trajectories and original training checkpoints remain intact.
