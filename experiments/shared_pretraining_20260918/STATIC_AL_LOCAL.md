# Latest local MACE and GATr: static Al

Question: how do the final local bond-order MACE and GATr representations partition the standard six Al snapshots?

Use the completed September 18 local MACE v10 and GATr v11 runs, final `last.pt` step 622 for both (best-selected states were step 576). Checkpoints are frozen and SHA-pinned. Native input, output, batch-reordering and independent compiled replay verification passed for both.

Keep snapshots 166, 170, 174, 175, 177 and 240 ps, the cached interior sampling grid and the previous seven-cluster spherical-k-means protocol. Preserve each encoder's trained local support: model radius 8, taper from 6 to 8, equivalent to approximately 7.9384 Å for Al. Thus spatial support differs from earlier wider-context models. Features are native z128. Cluster labels are independently assigned and this is a descriptive analysis, not a held-out generalization test.

Recipes: `configs/analysis/static_structural_{mace,gatr}_local_latest_al.yaml`; exports: `configs/analysis/structural_{mace,gatr}_local_latest_static.json`. Run `python -m src.analysis.pipeline ABSOLUTE_RECIPE_PATH` in each result's `technical/runtime` with conda `pointnet-torch214`. Each runtime preserves the checkpoint's actual frozen training source and historical metric contract; `technical/runtime-manifest.json` records its source hashes. The launcher registry also records the live orchestration repository, which is distinct from the inference runtime.

Results: [MACE](../../output/structural_static/mace-local-step622-al-20260919/) and [GATr](../../output/structural_static/gatr-local-step622-al-20260919/). Plots and tables are generated during the full pipeline; launch is not completion.
