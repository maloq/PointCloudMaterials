# Broad native snapshot encoder evaluation

The [live result page](../output/encoder_research/screen-20260923/index.html) and
[summary CSV](../output/encoder_research/screen-20260923/tables/summary.csv) update as
models finish. This queue compares 40 new pinned checkpoints and reuses 37 completed
GeoFrame checkpoint evaluations. Counts represent checkpoints, not independent
architectures or training seeds. See the [scientific protocol](../experiments/encoder_screen_20260923/README.md)
and [metric definitions](metrics/encoder_screen.md).

Use conda `pointnet-torch214`. Native checkpoints execute under their recorded
producer code; orchestration/evaluation is snapshotted before detaching workers.
One worker per allocated GPU shares an advisory-lock queue. A failed model is
recorded and other independent models continue. Failed models require diagnosis
and explicit receipt archival before retry; there is no silent fallback loader.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -m src.research.encoder_screen.prepare --config configs/encoder_screen/screen_20260923.json
python -m src.research.encoder_screen.queue --config configs/encoder_screen/screen_20260923.json --lane gpu0
python -m src.research.encoder_screen.report --config configs/encoder_screen/screen_20260923.json
```

Preparation is once per new output. Existing completed embeddings are reused only
under the pinned task/reference identity. Outputs and failures live in
`technical/evaluations/<name>/`. Each GPU lane records status in `technical/lane-*.json`.
The queue stops taking new tasks near allocation expiry; completed work is reusable
by a continuation allocation. Resume using the frozen config/code named in
`technical/launch.json`, not a modified live checkout. No new encoder training or
simulation is submitted by this workflow.

The explicit engineering benchmark compares the same frame, all metric fields
and cluster assignments with default and bounded threading. Its measured timings
apply to CPU metric evaluation, not total campaign speed. UMAP and spatial figures
use cached embeddings in a separate figure pass. Historical metric files are not
rewritten to make them look consistent with a changed protocol.

## Eightfold spatial plot density

At the user's request, spatial panels now use eight times as many atoms within
the exact original slice. The original slice atoms and their cluster assignments
are retained; seven times as many additional atoms are sampled from the same
interior slice. Full-cell PTM and the original material order threshold label the
new atoms. Recomputed labels on the original slice must match exactly. KMeans is
refitted on the unchanged original fitting embeddings, its original assignments
are checked exactly, and it predicts labels for the additional embeddings.
UMAP, contingency tables and numerical metrics keep their original fixed cohort.

A separate frozen visualization worker performs this additional inference and
redraws each completed encoder's spatial panels. It records exact before/after
counts and input/checkpoint identities under `technical/dense8-inputs/` and
`technical/evaluations/<model>/dense8-figures.json`. This does not change the
already-running numerical queue or its definitions.

The second density increase retains every atom from the fourfold plots, then
adds the same number again. Marker diameter is halved: Matplotlib scatter area
changes from5 to1.25 points². Fourfold PNGs and their input cache are preserved.
