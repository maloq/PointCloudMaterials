# Running the consecutive local-motion experiment

Use the existing module in conda `pointnet`:

```bash
python -m src.research.mace_local_state.run --config configs/analysis/mace_local_motion.json --stage motion-all
```

`motion-all` plans exact consecutive windows, starts one preparation worker per
configured GPU, assembles the verified feature cache, runs disjoint model variants
on the GPU workers, then evaluates them. `motion-prepare --lane N` and
`motion-fit --lane N` are worker stages, not alternate independent protocols.
`motion-evaluate` reuses completed fits. Smoke configuration uses the same
implementation and separate cache/output, with named sources and three epochs.

Runtime configuration names the actual node, Slurm allocation and a stop time
before allocation expiry. Detachment does not extend that allocation. Workers
stop at completed source units or checkpoint intervals; the controller records
`paused`, not success. To continue, explicitly update runtime allocation settings
and rerun `motion-all`; completed data units/epochs are hash checked and reused.
The scientific config and implementation must remain unchanged for exact resume.

The output is `output/mace_local_motion/sequences-20260916/`, linked to the
STORE analysis directory (`${storage:archive}/analysis/`) because WORK quota was
exhausted at launch. Existing inputs remain read from WORK; feature/source caches
are on IDS. Monitor
`technical/prepare-lane*.json`, `technical/fit-lane*.json`,
`technical/evaluation-status.json`, `technical/status.json`, and worker logs.
The detached controller PID, allocation and command are retained in
`technical/launch.json`. Do not modify running source files or compare partial
evaluation tables as if the sweep were complete.

Completed tables have frozen metric definitions and implementation hashes.
See [protocol](../experiments/mace_local_motion_20260916/README.md) and
[calculations](metrics/mace_local_motion.md).

Current launch: controller PID `833590`, node51 allocation `994149`, stop time
2026-09-16 14:44:52 Europe/Paris (five minutes before allocation expiry).
Output: `/store/PERSO/vmorozov/analysis/mace_local_motion/sequences-20260916/`.
