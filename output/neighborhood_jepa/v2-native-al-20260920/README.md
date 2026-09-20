# Causal neighborhood JEPA v2 and frozen crystallization evaluation

The new native-Al comparison is running detached. Node52: job **1000974**, two
L40S GPUs, eight-hour budget. Node53: existing **1000616**, two H100 GPUs (one
joins immediately; one after its legacy fit). Node59: existing **1000818**, two
RTX PRO 6000 GPUs joining after their existing legacy fits. No old fit was stopped.

**Five MACE arms**, common seed/initialization, 1,280 updates each, batch 256,
encoder/head maximum LR .0002/.002, cosine with warmup, compiled selective BF16.
8,995 native-Al training anchors in 36 independent lineages; 480 development
anchors in 15 lineages; fixed .75 ps lag and Lee MEAM potential. No new simulations.

Corrections: independent raw invariant exports in train/eval, no target-dependent
normalization; all equivariant channels directly anchored to smooth fixed moments;
explicit sample-normalized SIGReg at weight .1; family-balanced prediction losses;
only required views encoded (2,2,8,8,14 instead of 21 for every arm).

[Live crystallization comparison](CRYSTALLIZATION.md) evaluates **34 encoder
checkpoints**, each with a matched frozen linear and nonlinear onset readout:
26 v1 models, three earlier VICReg MACE/GATr checkpoints, and five new v2 models.
Two non-encoder baselines add four fits: 72 readouts in total. New encoder results
are pending; no scientific improvement is claimed from the smoke test.

The assay retains all 150 sources and original train/development/calibration/test
roles, natural at-risk windows, .75–96 ps horizons, calibration-selected alarms,
classification/calibration/timing/missed-event and sampled-center spatial metrics.
It uses one current snapshot, no history/context aggregation. This is the existing
historical test cohort, not newly untouched data. Test scores do not select the queue.

Validation: **39 tests passed**, including actual CUDA FP32/BF16, compiled replay,
optimizer/projection-counter resume, causal train/eval inputs, joint target gradients,
FCC/cubic/parity/perturbed structure tests, and the legacy amplitude-rescaling witness.
A 32-update compiled real-data smoke completed; exact legacy MACE, GATr and JEPA
frozen-producer extraction passed. **1,386 cached views** were independently rebuilt
from raw trajectories, including atom identity, target and periodic-image checks.

Preflight findings and changes: the first disposable smoke launcher lacked the
spawn main guard (corrected); a test attempted unsupported deepcopy of a CUDA
cuEquivariance module (replaced by a fresh module plus state loading). SIGReg 2.56
was too dominant in the smoke and was reduced to .1 before the final comparison;
final tests use .1. Hazard probes initialize to training-only empirical hazard so
rare events do not start at an artificial 50% hazard. Uncalibrated probe preflight
results remain separately archived under `technical/checks/`.

Deferred: three-seed confirmation (user requested one seed), history/control and
richer tensor-product architecture ablations, retrieval and comprehensive TDA/noise
continuity assays. No automatic promotion is enabled. Coordinate storage is float16;
observed coordinate spacing reaches .0625 Å, limiting fine-scale MD noise claims.

Status: `technical/lane-*.json`, `technical/runs/*/status.json`,
`technical/crystallization/*/*/status.json`. Config/source/checkpoint/data identities
are saved per fit. New metric definitions/hashes are exported; the old contract
checker remains disabled. The scientific protocol is in
`experiments/neighborhood_jepa_v2_20260920/README.md` and execution notes in
`docs/neighborhood_jepa_v2.md`.
