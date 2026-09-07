# Task-supervised encoder training — September 5, 2026

Question: after task-relevant training, which atomic representation best retains
continuous local structure and predicts independent shooting futures?

The earlier jitter-only VICReg experiment is preserved. This experiment trains
MACE, SchNet-style, smooth-density MLP and GeoFrame from scratch using continuous
geometry and future outcomes, without PTM labels. Geometry warm-up precedes joint
training. Current-geometry losses are balanced across Al/Mg/Ta. The predictive
targets are the existing eight-shot means at 12/24/48 ps. These are supervised
targets, not an unlabeled representation-learning protocol.

Each representation gets three learning-rate trials on seed 123, checkpoint and
hyperparameter selection using validation sources, and two additional seeds at
the selected learning rate. Training uses plateau learning-rate reductions and
early stopping after a minimum 80 epochs, with a 300-epoch cap. Convergence flags
and learning curves are retained. Fixed SOAP, TDA, density PCA and coarse-order
descriptors receive the same trainable nonlinear prediction heads and tuning.
GeoFrame is also trained here; its old frozen checkpoint is no longer substituted
for matched training. The supplied checkpoint provides architecture settings only.

Source splits, future target scaling and operational liquid selection stay fixed
from the earlier benchmark. This reuses a previously examined test set, so the
results are a controlled follow-up rather than a new blind benchmark. Training
and hyperparameter selection never use test losses. All model selection finishes
before this experiment evaluates test predictions. The independent future test
is Al; Mg/Ta contribute current-structure supervision and auxiliary evaluation.

Current targets are continuous bond order (8), compact alpha-persistence scores
(16), and SOAP scores (64), standardized within each material using training
inputs only. Their three losses have equal weight. Future topology, order and
mobility losses also have equal weight. These complementary geometric targets
are training measurements, not evidence that a particular phase exists.

Configuration: [config.json](config.json). Implementation is shared in
`src/training_methods/predictive_structure.py`; this directory contains the
experiment entry point and protocol. All outputs reside physically under
`output/predictive_encoder_training_20260905`.

```bash
conda activate pointnet
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python experiments/predictive_encoder_training_20260905/run.py --config experiments/predictive_encoder_training_20260905/config.json
```

The run is launched detached; its PID, command, per-trial logs, checkpoints,
learning curves and final comparison are saved with the results.

Completed: 45 trials, 27 selected checkpoints, 4,972 sampling epochs. The final
convergence audit extended one GeoFrame seed to account for accumulated small
improvements. All selected runs satisfied the audited plateau criterion before
test evaluation. Fixed-descriptor matched probes retain their original inputs;
the trained heads' normalization must not change the original variance floor.

[Findings and interpretation](../../docs/predictive_encoder_training_20260905.md)
and [full result tables](../../output/predictive_encoder_training_20260905/RESULTS.md)
are saved in the repository. Shared scientific code is in `src/`; this folder
holds the experiment record. Generated diagnostics and analysis-replay scripts
remain in the output directory.
