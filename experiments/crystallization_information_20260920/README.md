# Which observed information improves a frozen snapshot's short-horizon onset prediction?

Primary diagnostic: compare z with z+feature-group on the same natural at-risk
origins at .75,3,6,9,12ps. Use existing Al sources, exact onset labels and frozen
SIGReg-direct, Epi-direct, VICReg-direct MACE plus older VICReg MACE/GATr exports.
No encoder is selected using this diagnostic's test scores.

Add groups separately: radial/pair geometry, angular structure, bond ordering and
coherence, density/coordination, velocities/local deformation, geometry/order
changes over past3/6/12ps, and spatial shell geometry/motion out to25Å. Include
combined descriptors, condition-only/descriptor-only baselines and a shuffled
full-block negative control. All input slots and parameter counts are matched
within each readout family. Linear and nonlinear probes test accessibility;
stronger embedding-only/full probes check whether readout capacity explains gains.

Decode physical groups from frozen embeddings independently. Weak decoding and
positive add-back gain suggest a representation deficiency; gains from velocity,
history or outer shells can instead indicate unavailable input context. Current
bond-order neighborhoods also differ from the encoder crop. This is a finite
probe diagnostic, not a proof of representation sufficiency or causality.

Metrics: source-paired NLL/log-loss/Brier gains, AP, AUROC, recall at calibrated
false-alarm rate, actual false alarms, conditional timing error with misses, and
physical decoder R². One seed; historical held-out sources; exploratory comparisons.
See [definitions](../../docs/metrics/crystallization_information.md).

Recipe: `configs/analysis/crystallization_information_short.json`.
Run: `python -m src.research.crystallization_information --config configs/analysis/crystallization_information_short.json`.
Results: `output/crystallization_information/short-horizon-20260920/RESULTS.md`.
Execution uses cached features on CPU so the new multi-horizon GPU queue continues.
