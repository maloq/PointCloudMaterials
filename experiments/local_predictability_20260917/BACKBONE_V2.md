# Does a geometric transformer improve the native physical state?

The first comparison is a one-seed snapshot screen: accelerated native MACE
against an axial geometric algebra transformer (GATr), using the same native
windows, 128-dimensional state, seven conditions and present/future physical
heads. This is a new v2 experiment; previous onset receipts and fitted weights
do not establish its learnability or predictive performance.

Both models see center-relative positions and velocities. GATr uses the
[official implementation](https://github.com/Qualcomm-AI-research/geometric-algebra-transformer/tree/6afc26f26b8fcf51136ae8c1d264a36e14b6e497),
two spatial/temporal block pairs, eight multivector and 128 scalar channels. Point
and oriented-plane embeddings encode position and velocity; invariant scalar
inputs contain smooth support, squared radius/speed, signed radial motion and
the frame's smooth weighted count. Its state is projected from the observed
current-center scalar channels. Atom IDs are correspondence only. The spatial
graph is still built by the common input producer, although GATr ignores edges.

For both backbones, snapshot, 12 ps history and repeated-current-frame variants
have the same parameter inventory. A zero-initialized scalar gate multiplies each
entire temporal block change. History and repeated-frame continuations can load
their own architecture's snapshot parent. GATr uses causal temporal masks, physical
offsets and a smooth 12 ps age envelope inside normalization. Its join reference
is a fixed origin point; neither future observations nor other batch windows can
contribute. Absent keys have zero attention weight, and spatial taper weights
enter the attention denominator. Current centers are identified from positive
support and zero displacement, not array position.

The physical population is all 38,400 native windows, regardless of crystallinity.
Frozen full-timeline packets supply present and 0.75/3/9/24/48/96 ps targets.
Normalization is the existing train-only release normalization. Training loss is
present MSE plus mean future MSE. The separate onset option retains the original
risk set and six-bin hazard likelihood; it is implemented but not in this queue.
See [exact metric definitions](../../docs/metrics/local_predictability_backbone_v2.md).

The staged screen is:

1. Correctness tests for symmetries, causal masking, independent windows,
   zero-gate nesting and shared gradients, identical task heads, packet joins,
   microbatch gradients and exact optimizer/sampler resume.
2. For each architecture, the established 32-window, eight-source present-only
   fitting gate, up to 2,000 updates. Its diagnostic normalization is separate
   from the release. Pass requires MSE <= 0.10 and every block <= 0.25 together.
   Failure blocks larger fits of that architecture; a focused fitting diagnosis
   is the next research action, not a model sweep.
3. Explicit real snapshot and history workload profiling, FP32, effective batch
   eight. Save cold producer/graph costs, training/validation throughput, input
   waits, peak VRAM and actual masked attention kernel traces.
4. Fresh snapshot fits, 2,048 updates each, seed 20260919, alternating 256-update
   segments. Effective batch eight and every source/window draw are matched.
   Selection uses 64 fixed windows per selection source, independent of outcomes.
   Record matched-example and elapsed-time learning curves, then export full
   selection, calibration and test predictions without interpreting them.

This budget is a screen, not a claim of convergence or equal wall-clock exposure.
Temporal and repeated-frame fits, onset fits, matched native-row descriptor
rescoring and frozen readouts follow only after the screen is inspected. No
architecture is selected automatically using test data. No new simulations,
contrastive losses, mixed precision or compilation are required.

Reproduce with `python -m src.research.local_predictability.backbone_v2 --config
configs/local_predictability/backbone_v2/rtx6000_screen.json --stage screen`.
See [execution and dependency setup](../../docs/backbone_v2.md).

## H100 screen and completed-MACE repeats

The September 17 H100 extension repeats the 2,048-update physical snapshot screen
and measures both backbones' snapshot/history throughput on the same H100 NVL.
Both now use execution microbatch eight and a 24 GiB observation cache, with
20 synchronized timing updates. The effective batch remains eight throughout.
Thus cross-device timings also differ in execution configuration; the primary
speed ratio is within the H100 comparison.

The next comparison repeats the already completed native MACE onset assay with
GATr: an independent snapshot parent trained for 4,096 updates, then 4,096 updates
each for snapshot, 12 ps history and repeated-current-frame control. This is two
substantive comparisons (snapshot and history) plus their necessary history
control, at the original one-seed example budget. The parent is never shared
between architectures. Data identities, validation selection and training-sampler
state must match the completed MACE fits. The onset eligibility filter remains
separate from the all-state physical experiment.

The current-state crystallinity classifier and older mixture-likelihood memory
objectives are not part of this repetition. Speed and prediction quality are
reported separately; a faster network need not predict more accurately. See
[comparison metric definitions](../../docs/metrics/backbone_comparison.md).
