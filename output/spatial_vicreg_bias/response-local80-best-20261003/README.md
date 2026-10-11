# Response encoder · interactive structural comparison

- [Held-out Al MD](heldout-al.html): 24,960 projection observations; 13 complete snapshots of 70,304 atoms.
- [Six relaxed Al static snapshots](al-static.html): 166, 170, 174, 175, 177 and 240 ps; 24,000 projection observations and 684,723 dense grid centers.

The selected **response-weight-1 MACE128**, seed 20261002, update 250, minimizes validation future-feature Gaussian NLL among the 12 completed response follow-up fits: **0.9949387311935425**. This selection did not use cluster agreement, test labels or plot appearance. The training objective predicts short-time 20/100-fs features and simulator responses; its validation rank does not establish best crystallization prediction.

Choose the response model or the retained RH2 checkpoint under **Neural**. The pages include paired 3D PaCMAP, dense MD, optimal cluster color matching, cluster examples with lattice overlays, and spatial/embedding paths. The slider changes both PaCMAP and MD; the interface toggle only emphasizes existing points.

The descriptor fits, projections, dense assignments and RH2 neural results were recovered from checksum-verified archived JSON without numerical changes. The response model was evaluated frozen, on exactly reconstructed original nearest-80 patches; no neural training or descriptor refit was performed. Some historical GeoFormer assets were deleted in the October 2 cleanup, so its complete viewer is unavailable.

[Selection receipts](technical/checkpoint-selection.json) · [Inference/input record](technical/inference.json) · [Archive recovery](technical/viewer-recovery.json) · [Held-out correspondence definitions](analyses/matched/tables/METRICS.md) · [Static correspondence definitions](analyses/static/tables/METRICS.md)

The two interactive HTMLs are real files at this run's top level. Their assets also reference the restored sibling `mace-rh2-best-20260930` and `general-descriptors-20260930` result folders; retain those folders together when moving this viewer.
