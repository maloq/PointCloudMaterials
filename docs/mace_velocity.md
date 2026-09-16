# Running and loading the velocity extension

Active recipe: `configs/analysis/mace_velocity.json`. Run and status directory:
`output/mace_velocity/all-velocity-20260915/`. The local output is a symlink to
the configured WORK analysis root. Training caches are under the configured IDS
cache root, `mace-velocity-all-20260915`.

The detached `python -m src.research.mace_velocity.queue` waits for preparation,
extracts fixed structural teacher features, and runs `coordinates_velocity` on
GPU 0 and `coordinates_only` on GPU 1. This uses the current Slurm allocation;
detachment does not extend the allocation. The recipe stops before its deadline.
It does not submit or alter external jobs. `technical/queue-launch.json` records
the process and allocation; `technical/queue-status.json` records progress.
The companion `queue --finish-paused` process evaluates retained best checkpoints
if training reaches the deadline before all eight requested epochs finish. It
waits for both trainers to stop, uses the remaining allocation margin, and records
`complete_with_partial_training` so evaluated checkpoints are not mistaken for a
fully completed epoch budget. Exact continuation still uses each `last.pt`.

Each variant saves `technical/VARIANT/{config.json,status.json,history.json,best.pt,last.pt}`.
The corresponding `technical/VARIANT.log` contains training steps. Completed
evaluation writes `tables/VARIANT.csv`, frozen metric definitions, per-source
errors and paired test predictions. Keep normalization, source/cache manifests,
exact-resume states and code hashes with each checkpoint.

## Available inputs in this run

The audit scanned 64 simulation/dataset locations, including the unpublished
simulation location. Retained discovery
files enumerate actual paths and headers, not filenames assumed to contain velocities.
All usable measured velocities found are Al, in A/ps. The static Zr analysis data
has no velocities and is not treated as a velocity training example.

The accepted plan contains 1,125 trajectory records: 1,114 shooting-binary records,
10 verified legacy NPZ trajectories, and one legacy paired-dump trajectory requiring
conversion. Records include continuations and are not independent preparations.
There are 150 independently melted sources and one shared older preparation
lineage. Training uses 1,065 records; 30 independently melted sources are validation
and 30 test. Old shooting split names are superseded by this explicitly separate
preparation-level protocol; all descendants of the older shared liquid stay in train.

Every retained record contributes up to three stratified current/previous frame
pairs and four tracked centers, rather than processing every atom of every frame.
Old binary copies, repeated prefixes, quarantined runs, interrupted attempts and
restart-only audits have explicit reasons in `technical/inventory.json`.
Original source artifacts remain untouched. Source-balanced weights prevent
large collections of siblings from dominating the objective.

The ten NPZ velocity producers are traced to
`src/simulation/campaigns/unseeded_meam_crystallization.py::analyze`, which validates
the position/velocity timelines and sorted atom IDs. The remaining paired dump is
converted through `scripts/convert_trajectory.py paired-velocity`; it validates
atom/time/box alignment, retains a small exact source, verifies float16 arrays,
and records separate coordinate/velocity quantization errors and source hashes.

## Encoder loading

`best.pt` uses protocol `mace_local_phase_space_v1`, with `encoder_state`,
`head_state`, target/feature normalization, initialization identity and implementation
hashes. Instantiate `MACEVelocityEncoder` with the original architecture's MACE
module and the retained feature mean/scale, then load `encoder_state` strictly.
The experiment's `train.setup` demonstrates the exact architecture reconstruction;
it deliberately also validates the dataset/teacher provenance for evaluation.

For inference without loading the training cache:

```python
from src.research.mace_velocity.inference import load_encoder, embed_local_groups

encoder, config = load_encoder("PATH/coordinates_velocity/best.pt", device="cuda:0")
embedding = embed_local_groups(encoder, config, position_halos, velocity_halos)
```

The retained initialization checkpoint supplies the exact MACE architecture;
all trained parameter values are then replaced from `encoder_state` strictly.

For forward inference, call `make_context_graph(clouds, 'halo_inner', device=...)`
and pass the concatenated float32 velocities in **the same atom order** as the
unpruned input clouds to the encoder. Positions are Angstrom offsets, index zero
is the tracked center, and each input includes a complete 18 A candidate halo.
The returned `[B,304]` tensor consists of structure `[0:256]`, activity `[256:288]`,
flow `[288:304]`. Normalize or weight these blocks explicitly for downstream
distances; do not infer a scientifically calibrated clustering metric from their
raw concatenation. Velocity-free inputs can use the structural block, but no
missing velocities are fabricated for motion analysis.

The encoder describes the instantaneous local phase-space state. Recorded recent
frames regularize the structural block during training; they are not an extra
history input at inference. Temperature, elapsed time, source identity, and
future outcomes are never encoder inputs.
