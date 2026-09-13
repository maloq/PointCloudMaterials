# Al crystallization with the Ti source-and-branches workflow

Prepared 2026-09-11. Production protocol: generate a continuous pure-Al
crystallization trajectory and conditional 240 ps futures from distinct stages,
using the exact Al shooting campaign potential and the Ti campaign design.

## Protocol

- Potential: Lee, Shim and Baskes (2003) Al 2NN-MEAM, exact repository files and
  SHA256 checksums from `meam_predictive_dynamics_fixed15_20260904.yaml`.
- 100000 Al atoms; FCC lattice 4.0446507884 angstrom, 25 x 25 x 40 cells.
- Independent melt at 1325 K for 300 ps, following the established Al melt
  temperature/duration. The full melt must pass PTM <1% crystalline and
  mean-squared-displacement growth >10 square angstrom over its last half.
- Quench to 450 K, then continuous isotropic zero-pressure NPT, as in the Ti
  workflow. Timestep 1 fs, temperature damping 0.1 ps, pressure damping 1 ps.
  This matches Ti's physical workflow; the older Al shooting campaign uses a
  different NVT thermostat. Only its potential files and Al PTM cutoff are reused.
- Full-system PTM with RMSD cutoff 0.10, as used in the Al shooting campaign.
  Stop at >=94% FCC+HCP+BCC in two consecutive assessments, 4 ps apart, after a
  minimum 24 ps. Hard maximum source time 2000 ps; failure to reach the criterion
  is reported as incomplete, not accepted as crystallization completion.
- Six distinct chronological parents nearest crystalline fractions
  [0, 0.1, 0.3, 0.5, 0.7, 0.9], selected after source completion. These selections
  are conditioned on the completed source and share its root lineage.
- Six 240 ps branches from those positions, with distinct fresh velocity seeds.
  They are position-conditioned futures, not exact restart continuations.
- Sample every 0.1 ps. Retain verified float16 positions, float32 boxes, exact
  integer identity/timeline arrays, and full-precision LAMMPS restarts.
  Record quantization error before removing verified text trajectories.

Temperature 450 K is the default selected for this Ti-style analogue. The prior
Al independent-source and shooting datasets remain separate campaigns.

## Reproduce

Run from the repository root in conda environment `pointnet`:

```bash
python scripts/run_lammps_campaign.py elemental run \
  --config docs/simulations/al_crystallization/technical/al.json
```

The maintained elemental runner shares the source and branch implementation
between explicit Al/FCC and Ti/BCC protocols. The `assess-ti` CLI remains an alias
for older generated Ti inputs; new inputs use `assess-source`. Slurm CPU bindings
are derived from the current allocation for full campaigns as well as individual
branches. The campaign runs the source and then its six branches sequentially
on the same 48 allocated CPUs.

Output:
`/home/ids/vmorozov/simulations/al_meam_crystallization_100k_450K_20260911`, linked
as `datasets/al_meam_crystallization_20260911`. Generated preflight and campaign
batch scripts, logs and submission receipts live there. The full campaign has
a 72-hour allocation on CPU/cpu-high and remains detached after logout.

## Validation and file roles

`src/simulation/campaigns/al_crystallization_preflight.py` is the recorded validation recipe using the maintained
melt input and execution functions. It tests a 4000-atom hot crystal at 450 K
and a 50 ps melt at 1325 K, measuring PTM separation and liquid diffusion.
Production retains its separate 300 ps melt and repeats the liquid checks on all
100000 atoms. Preflight findings are recorded under `preflight/validation.json`.
Both potential files matched the shooting campaign's recorded SHA256 hashes.
Twenty elemental conversion, resumption, input and binary-reader tests passed,
including FCC/BCC atom-count checks. A 16 MiB output write/fsync probe passed;
this establishes writeability, not available NFS quota.

`technical/al.json` and this README document the simulation. The preflight implementation is in `src/simulation/campaigns/`. Elemental runner
changes and tests are maintained implementation. Generated jobs, logs and
validation outputs are disposable operational diagnostics in the output root.

Submitted on 2026-09-11: preflight job **989326**, full campaign **989327** with
`afterok:989326`. The full 48-CPU campaign starts only if preflight passes and
Slurm grants resources. At submission, preflight was running on nodecpu12;
production was pending its dependency. See output-root `launch.json` for the
receipt. Initial lattices are fully melted and liquid-validated before quenching;
no crystal seed is inserted into either the Ti or Al crystallization source.
