# Forecast MACE encoder: completed diagnostics

The exact forecast encoder carries generalizable local TDA information: held-out
local skill is 93.92% for observed topology and 85.97% for relaxed topology.
Numerical variability is negligible at the measured cadence; changing patch
membership and unresolved stochastic evolution both matter.

Read [the findings](RESULTS.md), [metric definitions](tables/METRICS.md). Tables retain the measured precision,
physical-lag, sibling and generalization comparisons. The full WORK run holds the figure and technical arrays, preserving
coordinates, atom IDs, embeddings, labels, trained readouts and provenance.

The source protocol and reproduction commands are in
`experiments/mace_encoder_diagnostics_20260914/README.md` in PointCloudMaterials.
The full run lives at
`${storage:analysis}/mace_encoder_diagnostics/forecast-seed20260910-20260914/`.
