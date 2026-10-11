# Native local80 response encoder in the interactive structural comparison

This revision adds the October 3 local80 response follow-up checkpoint to the
existing paired neural/descriptor 3D PaCMAP and dense MD viewer. The selected
response-weight-1 fit, seed 20261002, has minimum recorded validation future-feature
Gaussian NLL among the 12 completed response-weight/schedule fits: update 250,
NLL 0.9949387311935425. All candidates completed 400 updates. Selection uses neither
held-out scores, AP, cluster correspondence nor appearance of the plots. The
candidate receipts, exact checkpoint SHA256 and frozen training source are retained.

The native Predictor.encode exports 128 coordinates before the learned future-feature
head. It consumes one center-relative nearest 80-atom patch, constant atom channel,
center indicator, smooth support taper at 8 Å and 5 Å edges. Its 128-channel MACE has
two interactions, angular order 2 and correlation 2. The learned training normalizers
are preserved. Al length factor is 1; inference uses float32, TF32 disabled, no
autocast, batch 128. No temperature, time, motion, history, species/material identity
or teacher halo enters the encoder. The training-only MACE-MPA teacher used a moving
24 Å environment, 450 K and 20/100 fs future Fourier values/directional responses. The
inference receipt keeps the original separate encoder, predictor and teacher input
records. Static inputs are already relaxed; training patches were unrelaxed.

The fixed Al64 release remains
e148b7ec215ba5e6d86fc57d21dac266bbd501f1e91320968266b5dbaeb8f44d,
all64 structural track. K7 MiniBatchKMeans fits the same 74,880 uniform training
rows: raw native embeddings, seed 20260929, batch 4096, n_init 3, max_iter 200,
reassignment_ratio 0. Validation/calibration/test rows never fit clusters.
Every one of the 24,960 held-out displayed row and 24,000 static row retains its atom/frame/source
identity. Thirteen source 908 snapshots show 70,304 atoms each. Six static snapshots
(166, 170, 174, 175, 177, 240 ps) retain 684,723 total grid centers and their full-source
nearest 80 context. Static source atom indices do not establish identities between
snapshots, and static coordinate bounds are not periodic boxes.

October 2 removed the executable JS assets and dense inference patches, while
explicit JSON viewer payloads survived. Restoration checks the archived checksum
inventory and reserializes those payloads without recalculating historical values.
The original source coordinates reconstruct the response model's input patches;
every reconstructed array must reproduce its historical .npy SHA256. Original
assay patches replace periodic nearest-neighbor ties before that check. Static grid
centers are identified exactly from preserved coordinates in the original full
snapshot. Sampled and dense response cluster labels must agree exactly.

All four descriptor partitions, their scaling/projections, dense labels, local
examples and lattice overlays are reused from the all-training reference.
Older RH2 neural coordinates, labels, examples and embedding paths are also reused.
Historical GeoFormer full MD and sample/lattice/travel JS assets were removed;
its complete older viewer cannot be restored from the remaining archive. No model or descriptor
fit is recreated to substitute for those deleted historical results.

New neural PaCMAP 0.9.1 is a descriptive 3D projection of the full display population:
20 neighbors, MN 0.5, FP 2, iterations 100/100/250, Euclidean distance, apply_pca=false,
PCA initialization, faiss, seed 20260929. This retains the earlier viewer protocol;
it is distinct from the separate train-landmark/held-out-transform response 2D study.
The displayed maps do not define clusters or measure predictive performance.

For each descriptor family, a one-to-one Hungarian assignment maximizes raw
membership overlap. Its reference is all 24,960 held-out displayed rows or all 684,723
static dense centers pooled over the six snapshots. The permutation is fixed
across frames. Table `reference`, `display` and `snapshots` entries report:

- `rows`: number of compared centers.
- `matched_rows`, `matched_fraction`: assigned contingency mass and its fraction.
- Per-pair `intersection`, `neural_rows`, `descriptor_rows`, `union` and `iou`:
  ordinary membership counts and intersection/union.
- `neural_fraction`, `descriptor_fraction`: intersection divided by the respective
  cluster count. Empty denominators are null.
- `adjusted_rand_index`: sklearn adjusted Rand agreement, invariant to permutation.

These exported numbers describe the new response encoder. The RH2 viewer choice
retains its archived statistics. Shared colors are assigned pairs, not proof of
identical physical states. Counts are atoms, not independent trajectories; no
confidence interval or downstream crystallization superiority is inferred.

The interface toggle retains all observations and correspondence counts. Points
within 12 Å of the saved PTM boundary use full opacity, 1.4× size and a 0.6 px black
outline; others retain normal appearance. Examples are up to five uniformly
sampled actual environments per new cluster, deterministic by snapshot/model/cluster.
Sparse connections and PTM-fitted lattice overlays use the existing renderer.
Travel reuses exactly the saved observed-atom subset and spatial graph; only the
new 128D neural vectors and their 16-nearest-neighbor graph are computed. Both
spatial and embedding paths show observed structures, not interpolated atoms.


Table export: 2026-10-03T15:59:52.697731+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `technical/metric-contracts/response_cluster_comparison.json` relative to the analysis root.
