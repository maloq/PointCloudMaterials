# Direct Epi-inspired invariant embedding: static Al

Analyze the `epi-direct-order` encoder using its development-selected `best.pt`,
step 768. This is the completed width64 MACE treatment with identity projector,
Epi regularizer weight .1 and order-anchor weight .25. Export the native invariant
128 channels including its trained per-observation LayerNorm; the angular120
channels and conditional prediction heads do not enter static clustering.

Use the same six snapshots (166, 170, 174, 175, 177, 240 ps), cached interior
centers and seven-cluster settings as the earlier MACE/GATr static analyses.
Preserve the trained local support (radius8 in model units, approximately
7.9384 Å for Al), edge cutoff5 and native eager BF16 execution. This comparison is
descriptive; independently fitted cluster numbers do not identify matched phases.

The checkpoint is frozen under
`output/structural_static/epi-direct-order-step768-al-20260920/technical/source-checkpoint/`.
Its recorded producer files are verified and copied into the analysis runtime.
Verification requires exact checkpoint weights and native collated input tensors,
then agreement with independently loaded eager native invariant outputs and
reordered batches before the full pipeline starts.

Reproduce in the result's `technical/runtime`, using conda `pointnet-torch214`:

```bash
python -m src.analysis.neighborhood_adapter --config configs/analysis/epi_direct_static.json --stage export
python -m src.analysis.neighborhood_adapter --config configs/analysis/epi_direct_static.json --stage verify
python -m src.analysis.pipeline configs/analysis/static_epi_direct_al.yaml
```

The export command requires a fresh output. Recipes pin the frozen checkpoint;
source provenance and the executed batch script are retained in `technical/`.
[Results](../../output/structural_static/epi-direct-order-step768-al-20260920/)
contain plots and tables once those analysis stages complete.

Rerun on allocation1001439/node58: the original compiled run failed the full-frame singleton check (maximum difference .00133). A diagnostic found eager batch32/singleton differences of .00000301, versus .000962 compiled; compiled/eager batch32 differed by .00181. Eager inference preserves the intended native operations. Original logs and adapter are retained under `technical/attempt-1001420/`; tolerances remain unchanged.

On the next rerun, the user explicitly disabled the full-frame batch-replay stopping assertion after one of768 values failed on170ps (maximum absolute difference2.62e-6). Differences remain diagnostic outputs; the assertion is removed only from this preserved analysis runtime. See `technical/rerun-no-batch-check.json`.
