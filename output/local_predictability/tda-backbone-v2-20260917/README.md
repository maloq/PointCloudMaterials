# Frozen MACE/GATr instantaneous TDA retention

**Complete.** [Results and proposed GATr follow-up](RESULTS.md).

One-seed source-held-out readout experiment on all 38,400 frozen native windows.
Targets describe the nearest 80 observed atoms, including the center, at the current anchor.
Both physical snapshot encoders stay frozen; ridge and nonlinear residual probes test retained topology.

- [Scientific protocol](../../../experiments/local_predictability_20260917/TDA.md)
- [Current progress](technical/status.json)
- [Metric definitions](../../../docs/metrics/backbone_tda.md)
- [Completed metrics](tables/topology.csv) and [machine-readable results](technical/metrics.json).

Validation: five focused tests and eight result-layout tests passed. CUDA smoke checked
both checkpoint identities and frozen export parity (maximum absolute difference <8e-7),
and completed ten readout optimizer updates on RTX PRO 6000.
