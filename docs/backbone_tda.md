# Running frozen-backbone TDA readouts

Use conda `pointnet` on a CUDA worker. The recipe references the completed H100
physical snapshot screen; it may run on RTX6000 because parent exports are
checked against fresh extraction. Existing H100 onset runs are independent.

```bash
python -m src.research.backbone_tda --config configs/local_predictability/backbone_v2/tda_snapshot.json
```

The initial worker is Slurm job 996727 on node58 with 12 allocated CPUs and an
RTX PRO 6000. The recipe stops work by 18:40 UTC on September 17, before the
allocation expires at 19:02 UTC. Use the maintained
`scripts/experiment_registry.py run --spec SPEC` tracker for detached execution;
the run's `technical/execution-spec.json` records the exact launch and dependency
on the completed H100 screen. No hardware benchmark runs in this workflow.

Target production uses eight spawned CPU workers. Encoder extraction feeds both
frozen models the same prefetched GPU batch, then the two small readouts train
on resident states. Targets/states are cached in IDS via `${storage:cache}`.
Per-source NPZ/JSON artifacts are atomically completed and checksummed, so a
deadline or interrupted extraction can resume finished sources. Restart the
same command; only `deadline_utc` may change. Interrupted readouts restart from
their deterministic beginning. Any changed scientific config, source producer,
checkpoint or completed artifact raises an error; use new output/cache paths for
a different experiment. A file lock prevents simultaneous writers to one run.

Read `technical/status.json` for progress, `tables/topology.csv` and
`tables/METRICS.md` for results, and `technical/metrics.json` for nested metrics.
`paired_data.npz`, `predictions.npz`, target scales and readout files support
independent reanalysis. Linear NPZs apply `(z-mean)/scale @ coefficient.T + intercept`.
Nonlinear checkpoints add their block-scaled residual to the corresponding
frozen ridge prediction before transforming back to raw TDA units.

Scientific protocol: [TDA retention](../experiments/local_predictability_20260917/TDA.md).
