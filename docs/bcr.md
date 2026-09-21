# BCR execution

Use conda `pointnet-torch214`. The maintained entry point is
`python -m src.training_methods.bcr STAGE --config CONFIG`.
Stages: `prepare`, `verify`, `train`, `evaluate`, `probes`, `select`.
Implementation lives entirely in `src/training_methods/bcr/`.

The real-data verification recipe is `configs/bcr/real_overfit.json`. It has eight
full-radius full-precision Al patches. `prepare` creates an immutable cache and
refuses replacement. Preparation records root/source/center/frame identities,
physical times and cells, periodic image offsets, provenance/potential checksums,
noise calibration and native precision. It accepts explicitly declared repository
`shooting_binary` and `temporal_lammps_binary` producers. All other formats fail.

`verify --device cuda` runs the correctness suite and a two-patch fixed-noise
optimization diagnostic, saving `technical/gate.json` and verification metrics.
It is separate from training and never runs implicitly inside it. Large runs
(>256 updates) require this receipt for the exact encoder/decoder/tested model.
No Slurm submission is hidden inside training. Budget preparation, all control
fits and held-out evaluations together before scheduling an overnight study.

Create the real study from a new data recipe; set training `encoder.d0` and
`encoder.n_ref` to its training-only manifest values. The supported `arm` choices
are `bcr`, `unconditional`, `frozen_random`, `vicreg`, `frozen_vicreg`, `denoising`.
Frozen VICReg additionally requires `vicreg_checkpoint` from the matched encoder,
data identity and support. Old VICReg/JEPA checkpoints are deliberately rejected.
All controls must use equal exposure, budgets and seed/noise streams.

`train --device cuda` supports `microbatch` accumulation for reconstruction arms.
VICReg uses the full statistical batch. Set `updates`, `batch_size`, `lr`, `min_lr`,
`seed` and a dedicated output directory for each arm. FP32 eager only is released.
Re-running resumes `technical/last.pt` after exact config/data-identity checks;
optimizer, learning-rate position, source/block sampler and noise generators are
saved. CPU replay is bitwise tested; GPU arithmetic uses the declared FP32 tolerance.
Code/environment/version manifests include actual source hashes. Fresh scientific
jobs should use the repository's immutable code snapshot workflow.

For frozen checks use `evaluate|probes --checkpoint PATH --split development|test`.
Final test access requires `test_status: untouched_lineages` in the frozen data
manifest; an operator must establish that status from actual provenance. Evaluation
uses deterministic per-anchor noise keys; `evaluation.noise_bank_seed` selects a
second independent bank. `evaluation.unconditional_checkpoint` supplies a separately
trained control; omitting it leaves unconditional gain explicitly undefined.

`select` reads `technical/checkpoint-assessments.json`: each record needs
`checkpoint`, `structural_retention_pass`, `robustness_pass`, and
`development_nmse`. Assess these against the predeclared matched control margins;
no G2/G3 result is synthesized from reconstruction error or rank. See the scientific
protocol and `docs/metrics/bcr.md` for measurements and unresolved research gates.
