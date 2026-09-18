# Mixed-material triplet implementation checks

28 relevant tests passed, covering grouped normalization, train-only sampling,
static filtering, true timestamps, backtracking gradients, and cached/full update
agreement.

The real H100 preflight used the production batch 2,048 and microbatch 256 on
254,520 dynamic anchors, compiled selective BF16, and both spatial and temporal
pairs. All four updates had finite loss/gradients. Peak allocated memory was
25.6 GiB; the compiler captured two graphs with zero graph breaks.

Initial held-out physical/TDA score: 0.8030738831. After four updates: 0.7783627138.
Training-group mean baseline: 0.5097569488. This is startup verification, not a
validated predictive improvement. Production starts again from scratch.

Temporal updates after compilation took 16.2–16.3 seconds. The first spatial
update included compilation, so it is not a steady-state timing. Full curves,
per-group terms, calibration references and counters are in technical/result.json.

Detached production was submitted on allocation 997799 at 2026-09-18 17:29 UTC.
The run has 1,492 updates, twelve epoch equivalents; no extra GPU job was queued.
Its frozen code and submission receipt are in the sibling
gatr-mixed-triplets-campaign-20260918/technical directory.
