# CSLD restart boundary diagnosis

Question: why does the 24-rank 15-to-24 ps continuation smoke diverge?

Run from the repository root, in `pointnet`, inside a 24-rank CPU Slurm
allocation:

```bash
python experiments/restart_audit_20260905/diagnose_restart.py \
  --smoke-root /home/ids/vmorozov/simulations/al_meam_predictive_dynamics_fixed15_smoke_1parent_16branches_float32_20260904 \
  --output-root /home/ids/vmorozov/simulations/restart_boundary_audit_20260905/boundary_001
```

The output directory must be new. The recorded run used Slurm job 981566;
its batch recipe is `../run.sbatch` relative to that output directory.
Inputs come from smoke branch 0 and its existing 15 ps restart. The diagnostic
reruns the first 5000 steps, then compares two further steps with three restart
variants: original restart, same-job restart, and same-job restart without
sorting. Each variant uses 24 ranks and the original thermostat parameters.
The reused input template's `final_24ps.restart.bin` name in the uninterrupted
directory is historical: this diagnostic stops at step 5002, not 8000.

Results in `comparison.json`: the reproduced midpoint restart has the same
SHA-256 as the original. Original and same-job restarts preserve atom ownership
and local atom order, yet velocities differ at step 5001 (maximum 2.847426
Angstrom/ps); positions differ at step 5002. Disabling sorting does not help.
Thus a different node, corrupt checkpoint, or changed atom order alone cannot
explain the baseline failure. This is a boundary diagnosis, not a passing
24 ps acceptance test.

The LAMMPS 22 July 2025 source offers a likely explanation:
[`RanMars::gaussian`](https://raw.githubusercontent.com/lammps/lammps/stable_22Jul2025/src/random_mars.cpp)
caches `save` and `second`, which `get_state`/`set_state` omit.
[`FixTempCSLD`](https://raw.githubusercontent.com/lammps/lammps/stable_22Jul2025/src/EXTRA-FIX/fix_temp_csld.cpp)
uses that state serialization. The link to this run's divergence is an inference,
not an instrumented confirmation. An engine fix needs a separate build and full
comparison; existing checkpoints do not contain the omitted cache values.
Do not enable the extension gate on the strength of this diagnostic.
