# Local structural MACE and GATr

Active recipe: [`configs/shared_pretraining/local_structure/`](../configs/shared_pretraining/local_structure/).
It replaces the oversized structural observation; there is no large-support
compatibility mode. Historical runs require their frozen source or old commit.

Coordinates retain the training-only material scale: multiply physical offsets
by `9.192189 / material_scale`. Keep points strictly inside radius 8, with full
weight through radius 6 and a quintic C2 taper from 6 to 8. Crop before attention,
edge construction and packing. No extra halo and no fixed-k truncation.
The support and revision are recorded in training/checkpoint identities.

The full audit covers 1,275,000 views from all 1,385 dynamic training/selection
shards of the existing full-TDA release:

| Material | Mean input atoms | Mean sum of weights | Outer radius, Å |
| --- | ---: | ---: | ---: |
| Al | 122.37 | 78.56 | 7.938 |
| Mg | 124.35 | 79.74 | 8.833 |
| Ti | 125.18 | 79.71 | 8.102 |
| Ta | 122.22 | 78.35 | 8.170 |

These are local neighborhoods at the approximate GeoFrameTransformerV2 scale;
they do not reproduce its fixed 80-point, recentered input exactly. Actual
counts range from 92 to 139. The old inputs averaged 1,138–1,176 atoms.
Every supervised 80-point instantaneous-TDA neighborhood and every 7 Å
physical target is contained in the new crop. Raw immutable caches are reused;
outer points are discarded before either encoder sees them. Target definitions,
source splits, temporal neighbors and existing spatial partner choices are
unchanged. Spatial partners in this existing cache were selected within 4.25
normalized Å; newly prepared local releases retain the same 4.25-unit partner radius. A 2-unit
partner cutoff would fall inside the nearest-neighbor shell and is not used.

GATr attention covers only this sphere. Its count feature is now divided by 100.
MACE retains a 5-unit edge cutoff and two message-passing layers, truncated to
the same sphere. Its three pooling tapers are 0–3, 3–5 and 6–8 units. The
invariant output remains 128-dimensional. MACE's bond-order head predicts q4m/q6m from the center's learned l=2
features. GATr now also predicts q4m/q6m: per-atom learned multivector streams
are converted to even l=4/l=6 tensors, smoothly pooled, then decoded by an
equivariant linear head. Both use the same 12-nearest-neighbor targets and 0.1
loss weight; neither head receives the invariant embedding or raw target vectors.
GATr forms tensors before pooling to retain ordering when mean vectors cancel.
Its training-only feature cache has 128+704 channels; export remains z128 and
bond supervision adds no additional encoder forward pass.

Both corrected fits start from scratch with seed 20260919, statistical batch
2,048, compiled BF16 with FP32 geometric operations, mixed material/potential
batches, physical and instantaneous-TDA anchors, within-domain VICReg, and
backtracking only on temporal updates. The cosine schedule peaks at 0.002 for
heads and 0.0002 for the encoder, with 10% warmup. Static data remain excluded.
GATr: 5 epochs / 622 updates on H100 allocation 997799. MACE: 5 epochs /
622 updates on node58 allocation 999600. Microbatches are 1,024 and 512,
respectively; statistical batch size is unchanged.

Curvature coefficients are recalibrated on three training-only temporal batches
at the new initialization, aiming at 2% loss contribution and capped at 10% of
the base encoder-gradient norm, then fixed for the run. The initial preflights used 3.1 for GATr without bond supervision and 0.69 for MACE: the
encoder-gradient cap binds, giving about 0.77–0.80% and 0.63–0.67% of the
initial temporal loss respectively. Preflight is separate from production training.

Preflight after compilation, B=2048: H100 GATr took 1.57–1.71 s for spatial
updates and 1.89–2.02 s for temporal updates, with about 11.9 GiB peak allocated.
Node58 MACE took 1.02–1.11 s and 1.45–1.47 s, with about 14.1 GiB peak allocated.
These are six-update smoke measurements, not full-training throughput estimates.
On temporal updates, measured data wait reached 0.46 s for GATr and 0.40 s for
MACE; four preparation threads and next-update prefetch are enabled. The smaller
models expose some remaining CPU preparation cost; full GPU saturation is not
claimed. All measured objective values and gradients were finite.

Oversized runs were stopped gracefully at GATr update 1,046 and MACE update
147. Their checkpoints and historical metrics are retained, but not loaded by
these fits. The old v7-to-v8 migration code has been removed.

Audit, calibration, GPU checks and test logs:
[`output/shared_pretraining/local-structure-checks-20260918/technical/`](../output/shared_pretraining/local-structure-checks-20260918/technical/).

Launch from the repository in conda `pointnet-torch214`:

```bash
python -m src.training_methods.shared_pretraining.queue submit --plan configs/shared_pretraining/local_structure/campaign.json
```

The existing launcher freezes code and metric contracts, detaches workers,
logs online to W&B, and checkpoints before allocation expiry. Re-submission to
the same campaign is rejected. Each run's `technical/status.json` records progress.

Validation before launch: 90 tests passed on H100, including compiled BF16
rotation/gradient checks, local training/static/trajectory input parity,
material normalization, target coverage, gradient replay and queue tests.
Node58 additionally passed 10 MACE/bond-order/precision tests.

## GATr bond-order update

The initial local GATr launch was stopped at update 1 when the user requested
five epochs and bond supervision for it too. Its output is retained at
`output/shared_pretraining/gatr-local-20260918`; do not resume it. The replacement
starts fresh at `output/shared_pretraining/gatr-local-bond-20260918` with W&B ID
`gatr-local-bond-0918`. Its protocol is `shared_pretraining_local_gatr_bond_v11`.
MACE continues its existing five-epoch local run in frozen code.

Only launch the replacement GATr with:

```bash
python -m src.training_methods.shared_pretraining.queue submit --plan configs/shared_pretraining/local_structure/gatr_bond_campaign.json
```

The original two-run campaign is already submitted. Its frozen config records
the original GATr budget. The current GATr config contains the revised request.
The new GATr curvature coefficient is recalibrated with the bond objective
included in the base loss; inspect `technical/gatr_bond/calibration.json` in the
local checks output.

The bond-supervised GATr update passed 46 targeted tests, including compiled
BF16 covariance/backpropagation, true-batch versus gradient-cache updates,
current/partner-only supervision, local input parity, and fresh local-shard
spatial-partner availability. The new GATr head also retains nonzero l=4/l=6
features for an FCC shell whose mean vector is exactly zero.

The completed bond-supervised GATr preflight selected coefficient 3.1. Its six
full-B=2048 updates took 1.69–2.32 seconds, used about 12.9 GiB peak allocated,
and had finite losses/gradients. Five of six input waits were below 2 ms; one
was 0.27 seconds. These smoke timings exclude cold compilation.
