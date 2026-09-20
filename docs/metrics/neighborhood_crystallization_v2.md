# Matched frozen-encoder crystallization assay

Input: one current radius-8 normalized local snapshot, no encoder history or spatial
context aggregation. All frozen 128-dimensional invariant states receive the same
training-source-weighted feature standardization and known temperature/elapsed-time
conditions. Compare a linear hazard and a 128-hidden-unit SiLU/LayerNorm hazard,
1,024 AdamW updates, source-balanced training batches of 512, LR .0005 with warmup
and cosine, seed 20260920. Head biases start at the source-weighted training hazard; output weights use 1% of their default initialization. Head selection minimizes source-equal development event
NLL. Encoder checkpoints are selected by their original development rule, never by
crystallization test results. Older models execute under their frozen producer code.

Use all 150 independent native Al sources and their existing train/development/
calibration/test roles. All natural eligible at-risk origins and 16 tracked centers
are retained (191,688 windows across all roles). No event oversampling. A local
sustained crystalline event is three consecutive PTM crystalline frames; read the
inherited `crystallization_transfer` implementation for exact risk exclusions.
Discrete hazard endpoints: .75,3,9,24,48,96 ps. Origins are 3 ps apart.

This is the historical assay test cohort, already used for older studies, not a
new untouched test set. Pretraining ancestry is explicitly checked against the
calibration/test roots for every checkpoint. No final-test score changes the queue.

Reported quantities are inherited exactly from `crystallization_transfer.metrics`:
source-equal event NLL, AP, AUROC, Brier, ECE, precision/recall/FPR and classification
losses. The alarm threshold targets at most 5% FPR on independent calibration
sources. Whole-source bootstrap intervals are conditional on one training seed;
windows/centers are not independent bootstrap units. Condition-only and fixed
current-descriptor readouts provide non-encoder baselines. The historical
`geometry-baseline` actually contains 136 geometry+motion+order features: all
128 physical-packet entries plus eight bond-order entries. It has velocity inputs
that snapshot encoders do not have. `geometry-only-baseline` selects packet
indices 0:80 and 112:117 plus all eight order entries (93 features). Both append
the same seven temperature/time conditions. The order neighborhood is computed
from the original trajectory and can extend beyond the encoder crop.

Timing MAE/bias is conditional on detected event windows, using conditional mean
bin-midpoint onset time. Always report missed windows and missed event centers,
event recall, timing-within-3ps recall, lead time and false alarms alongside MAE.
Spatial metrics describe sampled at-risk centers only: crystal-fraction MAE,
Jaccard and pair-difference RMSE within 25 Å. They are not full-cell phase maps.

Each readout saves logits, exact row/source indices, its train-only transform,
optimizer/checkpoint selection, source-level metrics and pinned encoder hash.
`CRYSTALLIZATION.md` updates as assays finish. No missing score is a zero.
