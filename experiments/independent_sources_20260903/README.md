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
