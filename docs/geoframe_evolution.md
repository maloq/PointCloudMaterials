# Running the GeoFrame checkpoint-evolution study

Use conda `pointnet-torch214`. The [scientific protocol and literature](../experiments/geoframe_evolution_20260923/README.md)
define the requested epoch-34 recipe and independent reference targets. Active
recipes are under `configs/geoframe_evolution/`.

The reproduction reuses the existing 1,261,253-patch five-material static cache.
Its original 80/20 split gives 1,009,002 training patches; drop-last at batch
16,384 yields 61 model updates per pass and 999,424 patch presentations. An
epoch is one such shuffled traversal, not an arbitrary number of sampled
updates. FactorVAE adds a discriminator update from epoch 5, so Lightning's
global step is not the encoder's update count. The original epoch-34 checkpoint
has 3,965 global steps: 2,135 model + 1,830 discriminator updates.

Keep the 160-epoch cosine/warmup clock and stop after 35 passes. Preserve each
periodic checkpoint plus the initial and final states. Only operational settings
differ from the archived recipe: four loader workers, no shared-memory cache
copy, disabled optional model summary (incompatible with this PyTorch compile
runtime), and additional retention/assays. This reproduces the recipe on the
current implementation; it is not a claim of bitwise historical training.

```bash
python -m src.research.geoframe_evolution.train \
  --config configs/geoframe_evolution/epoch34.yaml \
  --output output/geoframe_evolution/epoch34-reproduction-20260923 --passes 35

OVITO_THREAD_COUNT=2 QT_QPA_PLATFORM=offscreen python -m src.research.geoframe_evolution.reference \
  --config configs/geoframe_evolution/assay.json \
  --output output/geoframe_evolution/epoch34-reproduction-20260923

python -m src.research.geoframe_evolution.queue \
  --config configs/geoframe_evolution/campaign.json

python -m src.research.geoframe_evolution.precursors \
  --reference-output output/geoframe_evolution/epoch34-reproduction-20260923 \
  --output output/geoframe_evolution/structured-liquid-regions-20260923
```

For this launch training uses GPU 0 and analysis GPU 1 inside allocation 1005857
on node61. No new allocation is needed. Frozen training source and evaluation
source are separate under the run's `technical/`; evaluation runs each checkpoint
in a child process and refreshes `README.md`, `index.html`, curve plots and CSVs.
The detached launch receipts record exact processes, source snapshots and logs.
Plot milestones: epochs 0,4,9,11,19,29,34 and the archived reference. Every epoch
receives the full numerical assay. Raw encoder and projector are both measured.

The reference preparation intentionally refuses to overwrite existing frame
records. An amended scientific protocol needs a new reference output. Evaluation
resume skips a completed checkpoint only after checking its content hash. A
failed evaluation writes its stage and traceback; it does not stop the independent
training process. The queue can be resumed with its frozen command after the
reported issue is corrected and the scientific export version is preserved.

Training resume uses `--resume PATH` with the retained full-state periodic
checkpoint and the same configured total passes. Lightning's post-fit `last.ckpt`
may carry the *next* epoch number; verify model-update count and the last periodic
checkpoint, not the filename or that number alone. Metric documentation and
implementation hashes accompany exports. Plots/weights remain ignored by Git.

The dense Ta/Zr candidate gallery is a separate output and metric family. It
refines full neighborhoods around high-order liquid candidates away from existing
crystal, showing mutually q6-coherent components and truncation flags. These
are targeted structural examples, not a nucleation rate or future-fate result.

The completed endpoint review is under
`output/geoframe_evolution/epoch34-review-20260923/RESULTS.md`. Recreate its
source-level paired comparisons with `python -m
src.research.geoframe_evolution.review --source OUTPUT --output REVIEW_OUTPUT`.
`python -m src.research.geoframe_evolution.render --output OUTPUT` redraws spatial
plots with discrete named category legends, verifying saved cluster contingency
counts exactly and preserving the numerical exports.
