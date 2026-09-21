# Multi-horizon JEPA embedding prediction

Predict the same tracked atom's future **invariant and equivariant embeddings**
at **0.75, 3, 6 and 9 ps** from its current snapshot embedding, temperature and lag.
Future local snapshots are encoded only as teacher targets, with gradients through
both branches of the jointly trained encoder. Physical/TDA prediction remains an
auxiliary anchor. This is not a TDA-only forecast experiment.

Three added fits: SIGReg, VICReg variance/covariance, and EpiJEPA-inspired
regularization, all with the MLP projector and bond-order anchors. They match the
corresponding original 0.75 ps runs in initial checkpoint, seed, parameters, batch,
updates and total future-loss weights. Three extra future-center views bring the
training total to 17 per anchor; exported inference remains a single snapshot.

Use the existing 32,768/480 anchor split (90 training /15 development native Al
Lee-MEAM sources). Valid 3/6/9 ps training targets: 32,768/32,448/32,112. All 480
development examples cover all horizons. Right-boundary missing targets are
masked; no new simulations were run.

Each fit uses width64, B512, compiled BF16 and 768 updates. Each selected encoder
then gets the standard frozen linear and nonlinear crystallization readouts.
Readouts and per-horizon physical/latent errors are reported separately; compare
latent prediction with each encoder's own persistence baseline.

[Scientific protocol](../../../experiments/neighborhood_jepa_multihorizon_20260920/README.md)
· [Metric definitions](../../../docs/metrics/neighborhood_jepa_multihorizon.md)
· [Original regularizer comparison](../regularization-20260920/README.md)
· [Launch records](technical/launches.json)

Update this report after the three fits and six readouts complete.

Current pending jobs: **1001356, 1001357, 1001358**, using the immutable
`vram-20260920` release. They replace1001326–1001328 (loader/validation release),
which replaced the original1001305–1001307. All are waiting for `QOSMaxGRESPerUser`;
they start automatically when the existing eight-GPU campaign releases quota.
The scientific tasks and budgets are unchanged. New execution retains encoder
activations and caches packed inputs on GPU; large/small memory tiers use16/6
retained microbatches. Larger microbatches were tested, but128 was fastest here.

Thirty-eight focused tests passed. A compiled BF16 B512 smoke verified
interruption, optimizer/RNG resume, validation, checkpoint saving and metric
export. [Measured execution improvements](../../maintenance/neighborhood-vram-20260920/README.md).
