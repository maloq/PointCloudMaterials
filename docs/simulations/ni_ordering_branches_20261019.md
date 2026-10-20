# Ni ordering: small-velocity-perturbation NPT pilot

A separate nine-hour collection tests physical continuation variability around
saved Ni states. It supplies no training data to the concurrent history/context
queue. There are24 origins from distinct ancestors,12 train/12 test inherited,
with eight origins in each present-liquid qbar6 stratum. Stratum boundaries are
training-only tertiles. Select origins by seeded identity hash, without future
outcomes. This is a present-state stratified pilot, not natural-rate sampling.

Each origin has16 perturbed branches plus one unperturbed control:408 tasks.
Positions and box are identical. Velocity noise is Gaussian, center-of-mass-free,
with vector RMS1% of initial peculiar-velocity RMS; rescale peculiar velocities
to preserve original kinetic energy and retain original COM motion. This is a
specified initial-state uncertainty ensemble, not Maxwell resampling, stochastic
thermostat noise or the exact future distribution of one complete deterministic
state. All branches initialize new NPT thermostat/barostat variables identically.
Saved float16 Q/V and float32 boxes cannot reconstruct original extended NPT state;
this pilot is explicitly reconstructed continuation, not native restart replay.

Potential: unchanged Ni1_Mendelev_2012.eam.fs SHA256
acc8bd87f61dd67ecd8f72a6f181d6aafe52ca45b29fe6c15cec7c39c0bf76df.
Original native LAMMPS isotropic NPT settings:1fs, source target temperature,
0bar, thermostat0.1ps/barostat1ps, float64 states and float32 FastNi GPU forces.
Six ps perbranch, observations every0.1ps. The same central atom is followed,
nearest80 reselected each frame. Store local Q/V and244measurements, full-cell
PTM center label, central connected-component size, total crystalline fraction,
and largest component. Component sizes are instantaneous; they are not automatically
new persistent tracked births. No generated full-cell structure is inferred from
independent neural local paths.

Native full-precision checkpoints are written every1ps and on deadline pause,
published with checksums to STORE. New simulation work is on SCRATCH; source and
checkpoint identities, local measurements and phase observations publish to STORE.
Only local float32 neighborhoods are retained, not a new full-cell trajectory.
The restart is resumed with the same NPT fix ID and unchanged GPU callback.
Interleave origins acrossbranch numbers, so a partial night covers all strata.
Two independent GPU workers use compatible GPU partitions as alternatives.
Bounded retries and per-branch failures do not terminate other branches.

Recipe: `configs/simulation/ni_ordering_branches_20261019.json`.
Launch: `python -m src.simulation.campaigns.ni_ordering_branches launch --config configs/simulation/ni_ordering_branches_20261019.json`.
Operational output: `${storage:training_storage}/ni_ordering_branches/pilot-20261019-v1`.
Archive: `${storage:archive}/simulations/ni-ordering-branches-20261019-v1`.
Inspect actual completion/progress receipts;408planned tasks are not completed
physical continuations, and16branches cannot precisely estimate sub-percent events.

## Launch receipt

Submitted October19 at00:36:26 Paris; fixed deadline09:36:26. Jobs1047132/1047133 requested two GPU workers and initially wait for resources. The first unperturbed control advanced0.1ps and resumed to0.2ps; native LAMMPS confirmed restoration of fix integrate NPT state. The pilot retains that exact checkpoint. CUDA architecture targets8.0/8.6/8.9/9.0/12.0 are fixed for shared-cache portability. See `technical/submission.json` and `technical/preflight.json`; allocation/branch completion must be read from live receipts.

## October19 failure audit and bounded recovery check

All408 production tasks exhausted their two attempts before MD integration:
the inherited `CUDA_HOME=/usr/local/cuda-12.9` pointed to a nonexistent `nvcc`
on the allocated workers. There are no completed production continuations or
measurement exports. The previously recorded0.2ps local preflight checkpoint
is the only partial trajectory; this is an operational failure, not a finding
that physical continuations do not crystallize. Original frozen source,
logs and failure receipts remain unchanged.

The maintained launcher now uses
`machine.execution.gpu_meam_portable_cuda_root`, matching the established Ni
birth campaign launcher. Each worker compiles/loads the force extension once
before claiming independent branches and writes a runtime failure receipt if
that common dependency fails. This avoids repeating a shared compiler error
across every branch; the force kernel and physical protocol are unchanged.

A separate bounded integration/restart/export check resumes the preserved
unperturbed control on node60 GPU1. Its output is
`${storage:training_storage}/ni_ordering_branches/pilot-20261019-v1-recovery-check`,
with copied frozen source plus the operational launcher fix. The original plan
identity and native checkpoint are preserved; only operational output paths
change. The check had a600-second wall-time cap and did not relaunch the408-task
campaign. It completed successfully in209seconds including cold compilation,
resuming from0.2ps to6ps. Native LAMMPS reported restored NPT fix state. All61
observations at exact0.1ps offsets exported finite local states `(61,80,6)` and
measurements `(61,244)`; the archived file matches its completion checksum.
Receipts: `technical/validation-exit.json`, `technical/validation-complete.json`
and the branch `complete.json`. This is one recovered unperturbed control,
not a recovered408-task ensemble or an estimate of event probabilities.
