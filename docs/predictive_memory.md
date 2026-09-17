# Predictive-memory pilot workflow

Use conda `pointnet` from the repository root. The module entry points reuse the
existing experiment tracker and allocation queue; no new simulations are needed.

```bash
python -m src.data.predictive_memory.prepare --config configs/predictive_memory/pilot.json
python -m src.training_methods.predictive_memory.train --config configs/predictive_memory/pilot.json --history-ps 48 --velocity
```

The first command audits and derives a separate immutable cache on the configured
cache storage root. It refuses to overwrite a completed release. `source_plan`
references the previous cache's observation inventory and inherited splits;
its outcome-bearing tensor shards are not loaded. Source paths are resolved by
the existing machine configuration. Release artifacts include dataset manifest,
lineages parquet, source splits, precision limitations and a data card. The first
release also includes `DATA_CARD_ERRATA.md` clarifying center selection.

Fit variations use `--history-ps {0,12,48}`, optional `--velocity`, and optional
`--repeat-anchor`. `--resume` requires the exact scientific configuration and
release checksum. Latest checkpoints include optimizer, sampling and Torch/CUDA
random states; best checkpoints are selected only by physical validation NLL.
`--deadline-utc` checkpoints and exits before the allocation reserve. Inspect
`technical/status.json`, `training.jsonl`, and `best.pt`/`latest.pt` in each fit.

The H100 recipe batches all observed frames into disjoint spatial graphs. The
equivariant temporal value sum is algebraically factored to avoid the expanded
time-by-time-by-atom-by-feature tensor. `encoder.frame_chunk` limits simultaneous
spatial frames and `activation_checkpoint` trades recomputation for VRAM. Neither
changes the data, neighbor list, temporal resolution, or gradients through past
frames. The selected H100 setting is 65 frames and no recomputation; a measured
48 ps example used about 16 GiB. Smaller GPUs can change these execution settings
in a new run config; exact continuation still requires identical config.

Scientific rationale and declared limitations: [experiment protocol](../experiments/predictive_memory_20260917/README.md).

The follow-up [velocity-input seed replicate](../configs/predictive_memory/replicate-xv-seed20260918.json)
reuses the release and training implementation with seed 20260918. Run its four
fits using `--velocity` at H=0,12,48 and H=48 with `--repeat-anchor`; compare them
with `python -m src.training_methods.predictive_memory.compare --config
configs/predictive_memory/replicate-xv-seed20260918.json --modalities xv`.
Its detached tracked command waits for the primary pilot to succeed before
using the GPU. Allocation plans and live queue state remain in each output's
`technical/` directory.

The separate [H200 capacity task](predictive_memory_h200.md) assigns eight
width-32 velocity-input fits across two seeds, using a portable bundle of the
same observations. It supersedes the previous crystallization-oriented H200
assignment for new work.
