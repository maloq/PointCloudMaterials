# Force-only response engines, 2 October 2026

Recipe: [response_force_engines_20261002.json](../../configs/simulation/response_force_engines_20261002.json).
Implementation: [response_force_engines](../../src/research/response_force_engines/).
Results: [force-engines-20261002](../../output/response_atlas/force-engines-20261002/).

Two persistent CPU adapters implement short local-response trajectories:
calorine4.0 CPUNEP with the official UNEP-v1-main model, and LAMMPS20250722
with the original Lee2003 Al MEAM library/parameters. The UNEP model download
was checked against the upstream MD5 and is SHA256-pinned in the recipe and
dataset registry. Source: [Zenodo11533864](https://zenodo.org/records/11533864).
The running MACE local80 collection remains a separate frozen experiment.

All force calculations remain in-process. External float64 BAOAB uses Al mass
26.9815385,450K,1fs steps,100fs friction and20/100fs observations. One central
plus four perturbed trajectories produces a256-value/two-direction response
sample. Epsilon trials0.03/0.01/0.003Å share the central path. All paths share
the original full-cell CUDA-generated initial momentum/thermostat noise, gathered
by the original atom rows. Noise is generated once on GPU during preparation;
workers thereafter run entirely on CPU, one thread per engine.

Features reuse the original `LocalFeatures.observed` and Fourier projection,
including each path's own perturbed initial anchor. Its actual float32 CUDA RFF
coefficients are saved and promoted to float64. CPU/GPU observable agreement is
recorded. Dynamics and forces are float64; this is an explicit new recipe, not a
claim of bitwise equality with the float32 MACE reference.

CPUNEP ignores ASE nonperiodic flags. Its atoms therefore occupy a box with12Å
vacuum margin on each face, and each step checks that at least the6Å model
cutoff remains between every atom and each face. A second12Å box expansion
must leave the initial forces unchanged. LAMMPS uses fixed nonperiodic boundaries,
ID-ordered gather/scatter, and rebuilds neighbors for every external force call.
Neither engine installs an internal thermostat/integrator. Directional energy
finite differences check forces, and one complete path per engine is repeated.

Four predetermined **training** parents and two existing gate seeds per parent
retain the fixed Al64 source/sample identity.18/24Å candidates are compared with
every larger environment up to36Å, separately for both feature prefixes. All
included atoms move. New potentials must pass their own boundary and epsilon
convergence checks. No fitted predictor or new W&B run is created by this
engineering benchmark. ACE distillation is not implemented by this workflow;
it would require a separately fitted and validated potential.

Use conda `pointnet-torch214`, with calorine4.0 installed without changing its
dependencies. Existing e3nn requires `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for its
trusted package constants.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -m src.research.response_force_engines.benchmark prepare --config configs/simulation/response_force_engines_20261002.json
python -m src.research.response_force_engines.benchmark launch --config configs/simulation/response_force_engines_20261002.json
```

`prepare` needs CUDA solely to reproduce existing streams. `launch` freezes code
and metric definitions, then detaches two single-thread CPU processes inside the
current allocation. `technical/launch.json` records commands, PIDs and allocation.
Each scientific payload is written to SCRATCH, copied to STORE and hash-verified
before completion. A stopped process resumes at whole-shot boundaries using its
frozen `run --engine meam|unep --config .../technical/code/config.json` command.
Do not rerun prepare over an active experiment or edit its frozen source.
Original source tensors, generated noise tapes, future feature vectors and
derivative tensors are saved; no new trajectory position arrays are written.

The local registry ID is `response-force-engines-20261002`. Simulation paths use
machine storage aliases; output, execution receipts and named metric bundles use
the standard result layout. Metrics are documented and hashed by
[`response_force_engines`](../metrics/response_force_engines.md).

The persistent multi-process adapter dispatches all five paths concurrently,
without changing their noise or initial anchors. The first training parent at24Å
and both seeds supply a three-repeat latency comparison and require exact-to-
tolerance agreement with the saved serial results. Three workers plus the two
single-thread serial campaigns fit alongside the two-thread MACE collector in
the existing eight-CPU allocation. Its metric family is
[`response_force_parallel`](../metrics/response_force_parallel.md).

```bash
CUDA_VISIBLE_DEVICES='' python -m src.research.response_force_engines.parallel --config configs/simulation/response_force_engines_20261002.json --engine meam --workers 3
CUDA_VISIBLE_DEVICES='' python -m src.research.response_force_engines.parallel --config configs/simulation/response_force_engines_20261002.json --engine unep --workers 3
python -m src.research.response_force_engines.publish --config configs/simulation/response_force_engines_20261002.json
```

`publish --watch` waits for both detached serial workers to complete, then
exports their final tables, updates the plot and reports eligible radii at the
declared epsilon0.01Å. A new smaller halo is eligible only after every
parent/seed/prefix comparison passes; no automatic encoder training or teacher
replacement follows. Partial plots use the same first parent and two seeds for
all alternatives, while final numerical gates use all four train parents.
