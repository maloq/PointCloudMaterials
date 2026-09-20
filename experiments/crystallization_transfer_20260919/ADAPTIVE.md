# Corrected trainable encoders and spatial/temporal context

Question: after correcting the stale normalization interface, can pretrained
fine-tuning and scratch MACE recover useful geometry-dependent onset predictions?
Which attention organization and optimization schedule best uses the observations?

## Population and controls

Reuse the pinned parent checkpoint, registered 150-source Al observation cache,
90/15/15/30 train/selection/calibration/test split, 109,838 eligible training
windows, six hazard horizons and all original event definitions. No new data or
simulations. Context still comprises the same seven geometry-selected centers
inside 25 Å; local MACE remains the original normalized radius-8 snapshot encoder.
Spatial representatives are rebuilt per frame and are not tracked atoms across
frames. Temporal attention therefore acts on frame summaries or on the complete
observed token set, without asserting persistent neighbor identities.

The main recipe uses batch 64, physical observations at -12, -3 and 0 ps, and one
seed. Full epochs shuffle every eligible training window once. Per-window weights
preserve equal training-source importance. Fine-tuning and scratch receive the
same total sample budget; head-only warmup is included in that budget.

## Corrected optimization

The invariant input to the prediction head is normalized with differentiable
mean and variance over the complete statistical training batch. Smooth radial
weights exclude out-of-support tokens; each observation has equal total mass
before applying its source weight. Variance uses epsilon 1e-8 (with a 1e-6
ablation). Gradients pass through moments, cancelling a shared encoder shift
instead of amplifying it through stale constants. Tensor variants normalize
rotation-invariant contractions, never raw vector components.

Before every validation, inference moments are recomputed from eight fixed,
training-only windows per source (720 total), using the current encoder and the
actual context/history settings. Double precision moment estimation avoids
cancellation at small native feature variance. Evaluation uses these fixed
moments, with no dependence on evaluation-batch composition. Moments are saved
with each checkpoint; selection, calibration and test data never estimate them.

Fine-tuning defaults to one epoch of head-only fitting, followed by encoder
unfreezing with a separate warmup. Scratch trains both modules immediately.
Encoder and head gradients are clipped separately (norm 1 and 5 respectively).
Head and encoder have separate learning rates; cosine schedules finish at 5%
of peak, with warmup. There is no VICReg or structural reconstruction objective
in these task-specific copies. The frozen structural parent remains unchanged.

All head moments and gradients use the full statistical batch. Direct backward
retains encoder microbatch graphs until the single head loss backward, using the
available 96 GB GPUs and avoiding a repeated encoder forward. A separately tested
gradient-replay implementation gives the same objective with less activation
memory. Replay is an execution option, not a distinct scientific treatment.
Native MACE still uses the established compiled mixed-precision/cuEquivariance
path; context heads and invariant normalization operate in FP32.

## Attention hypotheses

All variants share a pointwise projection and the same output-head form; the
attention variants add the specified blocks. They are matched in hidden width,
not total parameter count (reported in each fit).

| Context | Spatial processing | Temporal processing |
| --- | --- | --- |
| Mean | Smooth weighted mean of projected tokens per frame | Mean of frame summaries |
| Spatial | Geometry-aware self-attention independently within each frame | Mean of frame summaries |
| Temporal | Smooth weighted mean per frame | Causal attention over summaries |
| Factorized | Spatial attention per frame | Causal attention over summaries |
| Joint | Causal attention over all observed space–time tokens | Read the latest frame after joint blocks |

Every output also receives the latest center token and known temperature/time
conditions. Time inputs use physical offsets. Temporal attention masks later
keys from earlier queries; all observed frames precede or equal the forecast
origin. Spatial pair bias uses squared periodic relative distance; temporal/joint
bias also uses signed lag and squared lag. The bias ablation removes spatial
pair information from factorized attention while retaining radial support.
Tensor variants retain directional contractions and tensor alignment in spatial
attention. Full residual attention blocks include a feed-forward sublayer.

## Predeclared queue and selection

52 six-epoch screens: 24 fine-tuned, 22 scratch and six frozen controls. The
reference is factorized attention, width 128, four heads and one block in each
axis. One-factor changes cover:

- Encoder LR: fine-tune 3e-7/1e-6/3e-6; scratch 3e-6/1e-5/3e-5/1e-4.
- Head LR: 1e-4/5e-4/1e-3.
- Five attention organizations; depth two, width 256, eight heads; spatial bias off.
- Context-center radius 12/18/25 Å; history 0/3/12/48 ps and repeated-current 48 ps control.
- Normalization epsilon 1e-6 versus 1e-8; fine-tuning warmup 0/1/2 epochs.
- Tensor-context fine-tuning and frozen controls.

After every screen completes, rank screens by selection NLL within training mode.
Promote the top two fine-tuned recipes, top two scratch recipes and top frozen
recipe to independent **12- and 24-epoch** runs: ten additional fits, 62 total.
They restart from their prescribed initial weights and use the longer cosine
schedule, rather than continuing a fully decayed six-epoch optimizer. Predictions
on test sources never determine promotion. Promotions retain selected screen IDs
and selection scores. Incomplete or failed screening blocks automatic promotion.

Report the existing physical event/classification/timing/spatial metrics, source
uncertainty conditional on one seed, selected update versus total budget, and
training-only input-sensitivity/feature-spread diagnostics. Longer runs or more
hyperparameter candidates do not provide training-seed confidence. This remains
an exploratory study on previously examined sources.
