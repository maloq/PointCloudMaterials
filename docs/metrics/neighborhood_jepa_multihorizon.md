# Multi-horizon neighborhood JEPA

Extension of `neighborhood_jepa_regularization.md`, with the same 32,768 training
and 480 development anchors, source splits, Al Lee-MEAM potential, warm checkpoint,
width64 model, B512, seed20260920, 768 updates and checkpoint rule. New horizons
are3,6,9ps in addition to.75ps. Actual raw integration timesteps define the lag;
we require exact recorded observations at the requested physical time.

The predictor receives only the current snapshot invariant128/equivariant120
state and known temperature. It predicts invariant AND equivariant embeddings
of the same tracked atom's future local neighborhood. A zero-displacement center
query with the requested physical lag uses the existing shared JEPA prediction
heads; no future geometry is an input. The same jointly trained encoder processes
future target snapshots, and target gradients are retained (no stop-gradient/EMA).
The input-view count rises from14 to17. Deployment remains snapshot-based.

All three new arms retain their original SIGReg, VICReg VC or Epi regularizer,
MLP projector, present/order/geometry anchors and present/.75ps neighbor tasks.
The future-center latent family keeps total weight.1, divided equally across
.75,3,6,9ps. Invariant error is mean squared128D error. Equivariant error averages
component errors using the same train-only per-degree/radial moment scales as v2.
The fixed-future anchor weight likewise remains.25, divided equally over four
lags; each lag uses physical85 error +.25 TDA144 error decoded from the predicted
invariant embedding. These are auxiliary physical anchors, not a replacement
for JEPA embedding prediction. Neighbor-family weights remain unchanged.

At3/6/9ps, training has32768/32448/32112 valid anchors and development480/480/480.
Right-boundary missing observations have explicit masks and no loss/gradient.
A current-frame graph supplies a shape-only placeholder, never an accepted future
target. Training averages valid observations separately per horizon, then equally
across horizons; a wholly censored minibatch contributes zero for that horizon.
No missing validation score is replaced by zero.

Future graphs use the existing normalized local support, tracked atom identities,
periodic chart and edge cutoff. Physical/TDA anchors use the original producers in
Angstrom: radius support from the parent scale; TDA uses the nearest80 points
with stable distance/identity ordering. Fixed moment targets use normalized graphs.
The inherited training-only normalizers remain fixed; future/test statistics do
not normalize targets. Raw trajectory files are not changed and no new MD is run.

Per-lag development metrics: invariant MSE, equivariant scaled MSE, corresponding
current-state persistence errors and their ratios, decoded physical85 and TDA144
error against fixed targets, and physical/TDA persistence. Aggregation averages
within source then equally over sources with valid observations. Latent errors
are representation-dependent; use within-model persistence ratios and decoded
physical/frozen crystallization results across encoders. Exact predictions, target
states, masks and source indices are saved for subsequent source-level comparison.

Checkpoint selection remains present physical+.25TDA, unchanged from the paired
single-horizon regularizer runs. Frozen crystallization probes use the same linear
and MLP protocol in `neighborhood_crystallization_v2.md`. They are separate from
JEPA training. The existing historical test cohort and one seed limit claims.

The performance release reuses the current invariant/equivariant selection state
for all horizons and encodes each future center once: four views per selection
anchor in total. Right-boundary masks and every metric formula remain unchanged.
