# Overnight information/context queue

Use pointnet-torch214. Maintained entry:
`python -m src.research.context_night.queue submit --config configs/context_night/night_20260921.json`.
Stages `freeze`, `worker`, `execute`, `report` share the same recipe. Submission is
one-shot and records allocations, process IDs and frozen source/config/metric code.
No metrics-contract gate is enabled. Existing relaxation workers are not altered.

The allocated node53 H100 performs encoder fits and their probes. Two additional
12-hour one-GPU jobs process path screens, development-selected longer fits and
new-encoder transfers. File locks prevent duplicate tasks. Failed stages are
recorded with logs and block only dependent tasks; other ready work continues.
Subprocesses release models and GPU memory between tasks. Before allocation expiry,
existing training runtimes save optimizer/RNG checkpoints and defer evaluation.

Current state: `technical/tasks/*.json`, `technical/lane-*.json`, `technical/logs/`,
`technical/launches.json`. Scientific comparisons regenerate RESULTS.md after each
completed task. Source-held-out descriptor normalization uses original trajectories
already registered in DATASETS.md; no new dataset or raw simulation is produced.
Residents store source timelines once on GPU, not all overlapping input windows.
The encoder uses existing graph caches and retained/replayed activation profiles.
