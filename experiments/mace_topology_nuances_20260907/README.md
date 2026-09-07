# Topology-sensitive MACE continuation — 2026-09-07

User-requested replacement of the remaining ablations: analyze completed work,
audit topology-target stability, test a larger frozen decoder, then train with
topology-aware attraction and topology-distance preservation. Actual input stays
80 atoms; temporal VICReg uses verified 0.1 ps pairs for Al/Mg/Ta.

## Completed diagnostics

The completed LR5 all-objective model has its standard full static analysis in
`output/pretrained_mace_ablations_lr5_20260907/runs/all_objectives/static_analysis/`.
The interrupted no-temporal trial is analyzed separately at its latest saved
checkpoint; it is not a completed or matched four-epoch comparison.

The training-only stability audit used 384 anchor patches. Whitened TDA MSE at
0.005 Å perturbation was 0.00401 with fixed neighbors and 0.00758 after selecting
the nearest 65 atoms from the available 80. Neighbor membership changed for 6%
of patches at that amplitude. Rotation error was zero at stored target precision;
the tested float16 round-trip error was 0.0000614. These results do not establish
a physical noise floor; they calibrate an explicit small geometric tolerance.

A 256→512→512→32 decoder on frozen embeddings did not improve over the existing
256→256→32 head: held-out MSE 0.6336 versus 0.6234. Its inner-validation error was
also worse. The current head is retained. This single-seed diagnostic does not
prove that additional decoder capacity can never help.

## New training protocol

[Training configuration](training.json), [controller plan](plan.json),
[standard analysis](static_analysis.yaml), [descriptive comparison](comparison.json).

- Warm-start encoder, TDA head, forecast head and fixed scalers from the completed
  all-objective model; fresh optimizer and schedule. No teacher or EMA network.
- Batch **1,536 quadruplets /6,144 views**, doubled from 768. Encoder chunks are
  also 1,536. A real gradient/memory/peak-LR preflight must pass before training.
- Peak encoder LR **3e-4**, head LR **3e-3**, twice the previous LR5 run. One epoch
  of linear warmup, then cosine decay after every optimizer update toward 1e-6.
  Maximum eight epochs /5,632 updates, three hours, or validation early stopping;
  the controller shortens the time cap if needed to reserve analysis time.
- Reliability weights are fixed from training-only within-material variance and
  perturbation error, clipped to [0.05,1]. Observed weights are approximately
  0.94–0.99, so no PCA component is discarded. They weight TDA reconstruction
  and topology distances. Unweighted TDA MSE remains logged for comparison.
- Spatial/temporal attraction is multiplied by an exponential of the reliable
  target distance, calibrated to one-half at each material's training median
  pair distance, with a minimum of 0.05. Variance/covariance regularization remains.
- Add a Huber loss matching squared latent distances to reliable squared TDA
  distances between same-material anchors with local density within 3%. Density
  uses the radius of the twelfth neighbor. Its weight is 5. No clusters are used.

This changes several training settings together, so the final comparison is
descriptive, not a controlled attribution to one loss. Density matching does not
separate liquid/crystal environments or establish precursor sensitivity. Static
Al contains ancestors of training continuations and is not an independent test.

## Reproduction and detached execution

```bash
conda run -n pointnet python -m src.analysis.topology_nuances \
  --plan experiments/mace_topology_nuances_20260907/plan.json --stage stability
conda run -n pointnet python -m src.analysis.topology_nuances \
  --plan experiments/mace_topology_nuances_20260907/plan.json --stage decoder
conda run -n pointnet python -m src.training_methods.topology_campaign \
  --plan experiments/mace_topology_nuances_20260907/plan.json
```

The detached controller launch and current PID are recorded in the output's
`launch.json` (it was restarted after correcting the small gradient-test setup). It analyzes the
interrupted checkpoint, verifies the new objective and doubled batch, trains,
runs common frozen probes, and executes the standard static pipeline on the
selected encoder. Online W&B is requested for training. It stays within existing
allocation 983527, with safety cutoff currently 18:22 Paris. Failures write an
explicit failed status and traceback. This is a fresh workflow, not a resume CLI;
do not rerun it over completed output directories.

## Outputs and file roles

`output/mace_topology_nuances_20260907/status.json` is the controller status.
Target/decoder audits and preflight results are in the same directory. Training,
selected encoder and static outputs are under `training/`; incremental results
are under `comparison/`. Optimizer checkpoints use
`/tmp/vmorozov_mace_topology_nuances_20260907/`, while selected weights and reports
remain in the repository. Node-local intermediate checkpoints are temporary.

This directory contains experiment records. Maintained implementation is in
`src/analysis/topology_nuances.py`, `src/training_methods/topology_objective.py`
and `src/training_methods/topology_campaign.py`; existing MACE training/analysis
commands are reused. Launch records, logs and generated diagnostics are
disposable run outputs. Final training findings are pending.
