# Million-atom Al campaign — 2026-09-13

User requested the recent 100,000-atom Al protocol at one million atoms.
Recipe: `configs/simulation/al_crystallization_1m.json`. The only physical changes
from `al_crystallization.json` are the exact atom count (1,000,000) and FCC repetitions
(50 × 50 × 100 instead of 25 × 25 × 40). This changes the box aspect ratio slightly.
Potential hashes, seeds, temperatures, integration, liquid validation, 94% stopping
criterion are preserved. The user subsequently capped the crystallization stage at
400 ps, after the unchanged 300 ps melt (at most 700 ps total). The user subsequently requested no branches;
both branch-target and branch-seed lists are explicitly empty.

Submitted as detached Slurm job **991395**, one exclusive 96-CPU node in `cpu-high`,
192 GiB RAM, five-day allocation. Jobs 991364 and 991366 were cancelled during early melting to apply the user's source-only and 400 ps-limit corrections. Their early files are retained. No GPU is requested.
The larger calculation may exceed that allocation: full-precision checkpoints are
retained, but this elemental workflow does not automatically resume Al after a
walltime interruption. Do not claim crystallization completion from a timed-out run.

Run ID: `al_meam_1m_450K_400ps_with_melt_20260913T205405Z`.
Output starts at
`/scratch/PERSO/vmorozov/PointCloudMaterials/simulations/al_meam_1m_450K_400ps_with_melt_20260913T205405Z`.
The sibling `-launch` directory contains the batch script, request and Slurm receipt.
Completed output is converted to verified float16 and published to STORE by the
existing elemental command; integration and restart precision remain unchanged.

Reproduce with a new run name inside a 96-CPU allocation:

```bash
python scripts/run_lammps_campaign.py elemental run \
  --config configs/simulation/al_crystallization_1m.json \
  --run-name UNIQUE_NAME --ranks 96
```

Both potential files passed their recorded SHA256 checks. The generated lattice
input was checked to contain 4 × 50 × 50 × 100 = 1,000,000 atoms. No runner was copied.

Concurrent request for remaining top-ups: direct manifest/outcome audit confirmed
160/160 accepted 15 ps branches complete, 40/160 older 48 ps branches complete, and
36/60 independent 510/520 K sources complete. The user clarified the 520 K sources; the remaining 24 were submitted as array
991371 (see ../independent_al_sources/RECOVERY_20260913.md). The old 48 ps top-up
was not relaunched.

The explicit `source_limit_policy: save_state` finishes normally at the duration cap
without claiming completed crystallization. `source_progress.json` and `status.json`
record `stop_reason: duration_limit` and `crystallization_complete: false` if the
94% criterion was not reached. The existing 94% early stop remains active.

Post-melt state is retained as `melt/liquid.restart.bin` (native full-precision
LAMMPS state) plus `melt/liquid.lammpstrj`; `melt_state.json` records the restart
checksum and liquid validation. The final 400 ps state is saved separately under
`source/final.restart.bin`. All are included in final STORE publication.
Sixteen elemental tests passed, including duration-limit versus actual-crystallization
outcome semantics and retention/checksum of the melt restart.

The user also requested the full melt trajectory. `save_melt_trajectory: true`
saves every 0.1 ps during the 300 ps melt, converting it to verified float16 under
`melt/trajectory_binary_float16` before starting crystallization. Native
`melt/liquid.restart.bin` is retained unchanged. Job 991392 was replaced during
early initialization to record the entire melt from step zero. Seventeen elemental
tests passed, including actual melt conversion and exact liquid-restart retention.
