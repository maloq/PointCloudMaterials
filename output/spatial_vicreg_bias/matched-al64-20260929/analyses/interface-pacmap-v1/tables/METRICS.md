# Interface PaCMAP visualizations

This is a visualization of frozen feature spaces. Neural encoder training runs:
**zero**. Checkpoints, encoder weights, descriptor clusterers and neural cluster
assignments are unchanged. Standard nonparametric PaCMAP optimizes 2D/3D point
coordinates; no parametric/neural PaCMAP or predictor is fitted.

## Inputs and populations

Use exactly the 24,960 uniform held-out observations from the completed matched
Al study (30 original test source ancestors, unchanged roles and atom identities).
The second population includes all of these observations within 20 Å of the
crystal-side boundary atom layer, as defined by the separate interface
correspondence reference. There is no per-model subsampling, no source resplit,
and no removal of rows based on neural features or cluster appearance.

Raw encoder and projector spaces cover all nine trained combinations of
alpha=0/0.5/1 and seeds 17/29/43 at epochs 4/12/24. The non-neural spaces are
the full active TDA, bond-order, CNA and family-balanced joint vectors from the
interface-focused descriptor models. Their means, standard deviations, active
columns and family weights were fitted exclusively on training sources. They
are applied unchanged here; no features are reduced to a scalar prediction.
Neural vectors retain their native Euclidean metric, with no per-coordinate
whitening or standardization introduced by this visualization.

Consumed encoder inputs remain observed, periodic, centered 80-atom geometry,
with the original fixed length normalization. No history, motion, temperature,
absolute time or physical teacher is added. Source/frame/atom IDs, PTM,
interface distance and crystal fraction are coloring/filter/hover metadata only.

## Frozen inference and cache provenance

Where needed, forward inference regenerates held-out neural features on the declared Slurm
device from the exact saved checkpoint recipe/state and a copy of the original
training producer's model modules, VICReg projector and PairEncoder definition.
The network is in eval mode, all parameters have requires_grad=False and the
forward pass is inside torch.inference_mode. No optimizer, backward pass, W&B
run or training method is invoked. Checkpoint hashes match completed evaluation
receipts; epoch, completed-pass offset, sample identity and data identity are
verified. CPU float32 may differ slightly from the prior CUDA float32 result:
nearest saved K=7 centroid assignments are checked on every displayed atom, with
at most two disagreements allowed per representation (recorded explicitly),
otherwise execution fails. No labels are replaced by replay assignments.
The original six-entry cache policy and active leases protect generated vectors.
Projected coordinates and input/projection receipts are durable output artifacts.

## PaCMAP and displays

Pinned pacmap 0.9.1, FAISS backend; n_neighbors=20, MN_ratio=0.5, FP_ratio=2,
iterations=(100,100,250), random_state=20260929 and PCA initialization. Both
dimensions are independently fitted on the same rows. apply_pca=False retains
the complete input metric when building neighbors; PCA initializes coordinates
only. PaCMAP's common translation and scalar normalization are retained. These
are transductive descriptive visualizations of the held-out population, not
train-fitted predictions or held-out performance measures. Parameters and
package versions are frozen in each receipt. No favorable seed/layout selection.

Each feature space/population exports 2D PNG panels, an offline interactive HTML
with side-by-side 2D and rotatable 3D plots, and numerical coordinates with exact
atom/source/frame/original-row identities. Colors include the original neural
K=7 assignments, independent TDA/bond-order/CNA/joint K=7 labels, PTM, physical
region, signed atom-layer distance and input crystal fraction. Descriptor-only
pages use the respective descriptor labels as their default. Neural-cluster
numbers from different models are arbitrary and not aligned by their numeric ID.
Distance color saturates at ±20 Å; missing interfaces remain missing/gray.

Interactive region/source/frame filters only hide points; they do not refit the
layout. Physical-region codes describe the near-interface 12-Å diagnostic,
whereas the interface-focused layout includes a broader 20-Å context. The initial
marker size is 2 pixels and adjustable. Static panels use marker area 0.7.
Full-context and interface-focused layouts are separately fitted, with arbitrary
coordinate orientation. Visible distances/gaps and cluster compactness after
dimension reduction are not direct physical-state or clustering evidence.
Correspondence metrics remain calculated in the original feature spaces.

Coverage CSVs describe sample counts, input dimensions and execution time, not
scientific quality scores. There are 116 paired views / 232 projections:
(27 checkpoints × 2 exports + 4 descriptor spaces) × 2 populations × 2 dimensions.
The gallery publishes finished views incrementally as real file copies, using
a bundled local Plotly asset rather than a remote service or CDN.

Implementation reference: [official PaCMAP repository](https://github.com/YingfanWang/PaCMAP).

## GPU replay revision

The original CPU sweep failed its predeclared assignment check (seven differences for same-center seed 17, epoch 4). A separate CPU replay still differed at six rows using both the exact original K-means kernel and float64 distances. This establishes a difference before the cluster-label calculation; it does not isolate its architectural or numerical cause. No threshold was relaxed and no rows were dropped.

Revision v2 uses frozen CUDA float32 inference on node58 with the original A+B batch construction and original 256-parent offsets, including neighbor inputs needed to reproduce batch layout. TF32 is disabled as in the original producer. Checkpoint seeds and the original model implementation are restored. The original `_labels_inertia_threadpool_limit` producer kernel supplies the verification labels. The same at-most-two disagreement gate remains; original saved assignments remain the colors. Only the A representations are visualized. No training/optimizer/gradient step occurs.

Completed non-neural descriptor projections are copied unchanged from v1, with per-file inheritance hashes. Neural projections are regenerated under this declared GPU replay; original v1 artifacts and definitions remain intact. Both revisions publish through the same user-facing gallery. A separate MD renderer enriches HTML with exact saved physical coordinates without changing any projected coordinate, cluster label or metric.


Table export: 2026-09-29T13:05:57.428371+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `technical/metric-contracts/interface_pacmap.json` relative to the analysis root.
