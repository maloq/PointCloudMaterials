# Causal native MACE workflow

Use the existing native-encoder entry point in `pointnet`:

```bash
conda run -n pointnet python -m src.research.mace_velocity causal-prepare --config configs/mace_causal/pilot.json
conda run -n pointnet python -m src.research.mace_velocity causal-train --config configs/mace_causal/pilot.json --variant D --device cuda:0
```

Train `A`, `B`, `C`, `D`, and `repeated_anchor` with the same recipe for the matched
comparison. After D completes, `E` initializes from its best checkpoint and uses
the recipe's information constraints. Use a distinct output/cache for changes
to source selection or geometry. Changing widths or the Gaussian head requires a
distinct output, but may reuse the same verified graph cache. No command submits
a Slurm job. `--resume` requires unchanged config, implementation and cache and
restores optimizer and sampler state from the last validation checkpoint.

After a fit, `causal-probe --config configs/mace_causal/pilot.json --variant D`
trains simple linear and nonlinear physical readouts from the frozen exported
state. It also fits two predictors with identical trainable architectures: one
gets z and the original raw history, the other gets z and a single constant
training history. Only z varies in that control. A held-out physical prediction
gain from raw history indicates discarded information; a null result does not
prove state sufficiency. Heads and diagnostic encoders never replace exported z.

Run the same frozen-probe budget for every encoder being compared. A's direct
future and hazard heads are untrained, so their raw exports are not a fair
forecasting baseline. E uses additional updates after D; it is a second-stage
tradeoff study, not an equal-total-budget comparison with A–D.

Preparation reads the retained sequence-label cache and the actual source
positions, velocities, IDs and cells. It verifies their producer identity before
building causal graphs. The pilot adds 6.75 ps of explicit follow-up and computes
new labels with the same physical producer, retaining the original nine-frame
labels unchanged. This supports 0.75, 3 and 9 ps targets with sustained-event
confirmation. Set `additional_followup_ps` to zero to reuse only existing labels;
unavailable or off-cadence requested times fail explicitly. Caches go to IDS; readable output follows
`output/mace_causal/RUN-VARIANT/{tables,plots,technical}`. Frozen probe runs use
`RUN-VARIANT-probe-MODE` at the same level. Preserve checkpoints,
source snapshots, normalization, cache manifests and paired predictions.

Inference needs no original foundation checkpoint, target cache or auxiliary head:

```python
from src.training_methods.mace_causal.train import load_encoder
from src.data_utils.causal_history import build_history

encoder, config = load_encoder("output/mace_causal/RUN-D/technical/best.pt", "cuda:0")
history = build_history(positions, velocities, boxes, times_ps, atom_ids, atomic_numbers,
    center_atom_id, cutoff_A=5., context_radius_A=9., spatial_layers=2)
z = encoder(history.to("cuda:0"))  # [1,128]
```

Supply only observed frames, ending at the anchor. Arrays are `[T,N,3]`, box
lengths `[T,3]`, time `[T]`, and persistent ID/species `[N]`. Orthorhombic periodic
cells and a consistent atom ordering are required. Positions/velocities use A and
A/ps. The builder subtracts the tracked center and handles periodic images;
relative fractional displacements are unwrapped backwards from the anchor.
As with any sampled trajectory, displacements beyond half a box between retained
frames cannot be uniquely unwrapped from positions alone. This pilot uses the
actual elemental Al type mapping from its verified producer, never guessed IDs.

This architecture uses MACE's native tensor product blocks with scalar, vector
and rank-two channels retained through every block. Each spatial block receives
the preceding temporal output; there is only one final multiscale pooling stage.
Motion changes the main state. Attention scores are invariant and values retain
equivariant tensors. The learned spatial radial weights include signed relative
motion; the current frame has a residual path through every temporal block.
See the [upstream implementation](https://github.com/ACEsuit/mace/blob/main/mace/modules/models.py).

Read the [exact metric contract](metrics/mace_causal.md) before comparing models.
The causal architecture and correctness tests are not evidence of improved
forecasting or crystallization skill. A full comparison requires adequate training
budgets, multiple seeds, whole-source uncertainty and event-rich held-out data.

Validation on 16 September 2026: 48 focused/regression tests passed in `pointnet`.
A real-data H100 smoke completed all six variants, all four frozen-probe modes,
and the diagonal-Gaussian head on six independent sources with 72 windows and
0.75/3/9 ps targets. Fits used six updates and smaller 4-channel/16-dimensional
encoders. A separate real-input forward/backward check verified the main
16-channel/128-dimensional architecture and gradients through both spatial and
temporal blocks. Replayed cached physical targets matched exactly (676 values).
See the [validation record](../output/mace_causal/smoke-20260916/README.md).

These are execution/correctness checks. They do not establish forecasting gains,
state sufficiency or successful smoothness: the small smoke fits did not meet
the declared 0.10 low-order jump criterion. Longer training and the controlled
scientific comparison remain necessary before selecting a research encoder.

The three-seed GPU pilot uses `pilot.json`, `pilot-seed20260917.json`, and
`pilot-seed20260918.json` in `configs/mace_causal/`. Each uses 1,000 encoder updates
and 500 updates per frozen probe, with 90/30/30 independent train/validation/test
sources. After all fits, run
`python -m src.research.mace_causal_comparison --config configs/mace_causal/comparison.json`.
The collector verifies exact held-out target pairing and exports seed-averaged
physical errors and paired whole-source bootstrap intervals. These intervals
measure source uncertainty; they do not estimate training-seed uncertainty.
