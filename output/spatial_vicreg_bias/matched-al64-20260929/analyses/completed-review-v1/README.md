# Completed spatial VICReg mechanism study

All nine encoders completed 24 epochs, with all 63 checkpoint assays and six
classical controls. [Scientific findings](../../../../../experiments/spatial_vicreg_bias_20260929/RESULTS.md).

![Clustering versus continuous readouts](plots/mechanism-summary.png)

The shaded regions show the range across three paired seeds. Liquid TDA skill
is error reduction against the training-mean predictor on fixed held-out
crystal-free input observations. A collapsing global liquid cluster does not
establish that all liquid information was erased: the continuous readout improves.

[Metric definitions](tables/METRICS.md) · [Numerical table](tables/completed-study.csv)
· [Full summary and source-paired intervals](technical/summary.json).

This review refits no encoder, clusterer or readout. New source-paired summary
intervals were computed from retained errors in CPU Slurm job 1013898.
Figures, tables and summary receipts here are actual copies; original per-checkpoint
outputs remain in `/work/PERSO/vmorozov/analysis/spatial_vicreg_bias/matched-al64-20260929/`.
