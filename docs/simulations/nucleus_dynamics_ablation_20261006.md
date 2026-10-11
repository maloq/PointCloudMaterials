# Box and thermostat continuation controls

Follow-up to the [completed birth-rate calibration](nucleus_continuation_bias_20261005.md).
That present-selected cohort gave 13/64 saved births, 14/64 NPT births and
20/128 BAOAB births by 12 ps; BAOAB mean largest cluster was 35% smaller than
the saved reference. The new comparison separates fixed-box effects from
damping-strength effects without selecting new states by those outcomes.

Use all 64 existing initial full cells. Add native fixed-box NVT at 3 fs and
1 fs (64 each), plus float32 BAOAB with 300/1000 fs damping times (128 each).
Reuse NPT 3 fs and BAOAB 100 fs references; weak-damping shots reuse their
per-state noise seeds. Every arm uses source temperature and 12 ps duration,
native 0.75 ps observations and the frozen common-float16 nucleus detector.
See the [scientific definitions](../metrics/nucleus_dynamics_ablation.md).

The existing validated FastAl implementation is unchanged. NVT uses the same
native LAMMPS callback as NPT, with float64 integration and float32 force
arithmetic; this explicit control does not change the float32 response backend.
No new potential gate or training run is introduced. NVT-versus-BAOAB retains
differences in splitting, precision and momentum handling.

```bash
python -m src.research.nucleus_controls.ablation prepare --config configs/simulation/nucleus_dynamics_ablation_20261006.json
CUDA_VISIBLE_DEVICES=0,1 python -m src.research.nucleus_controls.ablation launch --config configs/simulation/nucleus_dynamics_ablation_20261006.json
python -m src.research.nucleus_controls.ablation summarize --config configs/simulation/nucleus_dynamics_ablation_20261006.json
```

Use conda `pointnet-torch214`. Launch detaches two workers on the explicitly
selected free local GPUs and requests four single-GPU Slurm helpers. A shared
nine-hour deadline begins at submission and includes queue wait; each helper
requests six hours within it. Workers obey the earlier of allocation and
campaign deadlines, reserving 15 minutes to checkpoint/publish. No automatic
extension; only process groups created by this chat's supervisor are stopped.
Other simulations and the containing allocation are not modified.

Output is `nucleus_recovery/dynamics-ablation-20261006/analyses/ablation-v1`
under training storage. SCRATCH holds new integration work; verified float16
trajectories, original-precision checkpoints and native NVT restarts go to
STORE under `simulations/nucleus-dynamics-ablation-20261006`. Native NVT
commands/logs are archived. The dataset registration is
`al-nucleus-dynamics-ablation-20261006`; it creates descendants of the same 64
ancestors, not new independent preparations. Frozen code, launch ownership,
hardware receipts and exact deadline live under `technical/`.

`technical/progress.json` reports new completion separately from the 192 reused
branches and 64 saved references. `tables/paired-comparison.csv` contains
matched-state contrasts, including partial-coverage denominators. Summaries
hold a lock and retain frozen metric definitions/hashes. Dense/shooting,
million-atom, relaxed and other-cadence populations are not included here.

## Launch receipt

Launched October 6 at 00:48:10 Paris time. Shared absolute deadline:
**October 6, 09:48:10 Paris time**. Local supervisor PID/PGID `3613857`
owns two GPU workers in node60 allocation `1022218`. Additional four-GPU
helper array: `1023266_[0-3]`, each requesting six hours within the shared
deadline. The containing allocation and unrelated jobs are not owned by this
campaign. Exact process ownership and frozen source are recorded in
`technical/launch.json`; per-worker execution and hardware records are separate.
