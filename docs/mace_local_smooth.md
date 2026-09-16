# Running the frozen local-state smoothness sweep

Use `python -m src.research.mace_local_state.run --config
configs/analysis/mace_local_smooth.json --stage smooth-all` in `pointnet`.
The `smooth-prepare`, `smooth-fit`, and `smooth-evaluate` stages are separate.
The older affine comparison's stages and config remain distinct.

The run is `output/mace_local_smooth/velocity-frozen-20260915/`, linked to the
configured analysis root. Frozen feature pieces and checksums use the IDS cache.
Producer metadata is read literally; only recipe paths use the config resolver.

Check `technical/prepare-status.json`, `technical/fit-status.json`, and
`technical/evaluation-status.json`. Per-variant directories retain `best.pt`,
completed-epoch `last.pt`, histories and evaluation arrays. `tables/comparison.csv`
and `plots/smoothness_information.png` are written after the sweep completes.
Keep the cache, checkpoint provenance and exact config for resumption.

A detached process still needs a valid allocation. Longer work needs a batch job
or another authorized allocation; launch records/logs go under `technical/`.

Scientific scope: [experiment](../experiments/mace_local_smooth_20260915/README.md).
Terms: [normalized RMS jump](research_glossary.md#normalized-rms-jump) and
[information retention](research_glossary.md#information-retention-and-smoothness-tradeoff).

The first sweep is complete; see [findings](../experiments/mace_local_smooth_20260915/RESULTS.md).
The longer follow-up uses `configs/analysis/mace_local_smooth_capacity.json` with
`--stage smooth-all`, submitted detached as job 994010. Its output is
`output/mace_local_smooth/velocity-frozen-capacity-20260915/`; the exact submission
command and batch script are retained in its `technical/` directory.
