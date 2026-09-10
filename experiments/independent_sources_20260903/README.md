# Independent 510/520 K Aluminum sources

Question: extend the independent-source pool to 510 and 520 K before calibrating
basin boundaries and selecting shooting parents. The launcher preserves the
original 30-per-temperature split, seeds, diagnostic boundary bands and manifest
checksums. These are experiment-specific assumptions, not new default settings
for the maintained independent-source campaign.

Shared generation and Slurm machinery is in
`src/simulation/campaigns/independent_meam_source.py`. This recipe customizes it
and ensures generated Slurm jobs invoke the same specialized launcher.

```bash
conda run -n pointnet python experiments/independent_sources_20260903/independent_meam_510_520K_sources.py --help
conda run -n pointnet python experiments/independent_sources_20260903/independent_meam_510_520K_sources.py prepare \
  --campaign-root /path/to/new_source_campaign
conda run -n pointnet python experiments/independent_sources_20260903/local_source_queue.py --help
```

`local_source_queue.py` is the experiment's cutoff-controlled local execution
recipe for already prepared, unsubmitted sources. Its required arguments remain
explicit; do not use it as a generic scheduler.

Original output:
`/home/ids/vmorozov/simulations/al_meam_independent_sources_70304_510-520K_30perT_float16_20260903`.
See [simulation context](../../docs/simulation_context_for_agents.md) for run
status and interpretation. The source manifest explicitly prohibits production
parent selection from future outcomes before calibration. No new findings were
produced by relocating the recipe.

## Recovery on 2026-09-09

User requested resuming independent Al sources. Filesystem audit found 75/90
completed sources at 400/450/500 K and 27/60 at 510/520 K: 48 remain. No Al
source/controller jobs were present in the expanded Slurm queue. Incomplete
low-temperature source 75 and high-temperature sources 23/24 had stale September
7 runtime files. These were preserved using same-filesystem moves into each
run's `interrupted_attempt_recovery_20260909T203221Z` directory, with original
inputs copied alongside them. Original seeds and manifests remain unchanged.

Recovery controller **987764** is queued with `afterany:987762`, waiting for the
six Ti shooting jobs to release submitted-job slots. It then invokes the
existing `independent-meam-source submit-next-wave` at low-temperature index 75
and the specialized recipe's `submit-next-wave` at high-temperature index 23.
Existing CPU batch paths are preserved. The configured waves use up to 3 low-
and 2 high-temperature jobs, 48 MPI ranks per source, with successor controllers.
Completed sources encountered later in either chain are skipped by the producer.
Sources retain their original 600 ps measurement histories and float16 storage.

The generated batch script, archived-status copies and launch receipt are in
`/home/ids/vmorozov/simulations/al_independent_sources_recovery_20260909T203221Z`.
The experiment record is `recovery_20260909.json`. The recovery script uses only
the existing maintained commands; no new runner was added. Source submissions
execute on the CPU controller; inherited GPU-allocation Slurm variables were
removed from its submission environment.

The maintained conflict check now queries live user jobs rather than expired
job IDs, while retaining duplicate-job detection. Three independent-source tests
passed, including the expired-receipt regression. A 16 MiB storage write/fsync
passed; this does not quantify remaining NFS quota. This recovery resumes source
production only; exact-restart and production-shooting scientific gates remain.
