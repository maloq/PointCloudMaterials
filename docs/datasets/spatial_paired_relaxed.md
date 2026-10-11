# Paired relaxed spatial training, October 8

Recipe: `configs/data/spatial_paired_relaxed_20261008.json`.
Implementation: `src.postprocessing.relaxed_spatial` and its inventory module.

This release includes paired dynamic Al/Ni relaxations and 24 already-relaxed
static Mg/Al/Zr/Ta inherent configurations explicitly confirmed by the user.
Dynamic observations require retained matching unrelaxed originals. Exact local
hot/cold pairs qualify even where retired raw Al trajectories have been removed.
Original geometry is audit-only. Static data are an explicit exception to this
pairing requirement; their generating potentials and minimization settings
remain unknown. Missing dynamic originals, missing relaxation products and
deliberately removed unverified relaxations are excluded with recorded counts.

Retained indexed float32 patches are preferred. Native/fixed-cohort paired
caches contribute additional centers from available indexed relaxed cells;
their float16-derived precision is explicit. One source/frame/center is kept
across caches. The original source ancestry and roles are preserved. Selection
and test are pooled for monitoring, with calibration and diagnostic data kept
separate. No held-out ancestor is used for fitting or normalization.

Static sources retain their original train-only material ancestry groups and
native-unit cached neighborhoods: Mg 62,916, Ta 510,260 and Zr 61,440. Six Al
static sources without that cache contribute 256 interior centers each, 1,536
total. Their point-cloud geometry uses no invented periodic cell. These data
have unknown preparation ancestry within each archived material family and
must not be treated as independent MD replicas. The held-out monitoring pool
contains Al/Ni; this run does not establish Mg/Ta/Zr held-out generalization.

All 6,571 planned native Ni relaxation frames are a dependency, including those
still running at queue submission. There are 256 fixed centers per Ni source,
and each paired observation retains the original nearest80 atom membership.
The cohort contains birth-centered windows, so it is not an unbiased estimate
of phase frequency in the underlying MD trajectories. No new quench is launched
by target preparation. Original arrays and relaxation receipts are checked before
use; native restart extraction is CPU-only.

The exact Small eSEN target selection and fixed material normalization are
copied from `esen_small_ni_20261008.json`: 40 bond-order, 45 CNA and 113 TDA
outputs. Target statistics are refitted on the new training data. Training
retains VICReg, 12 full epochs, batch/microbatch256, BF16 neural operations and
one-epoch warmup followed by cosine LR .001 to .00001. W&B remains online.

```bash
conda activate pointnet-torch214
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  # existing e3nn/checkpoint profile
python -m src.postprocessing.relaxed_spatial plan --config configs/data/spatial_paired_relaxed_20261008.json
python -m src.postprocessing.relaxed_spatial audit --config configs/data/spatial_paired_relaxed_20261008.json
python -m src.postprocessing.relaxed_spatial submit --config configs/data/spatial_paired_relaxed_20261008.json
```

The queue prepares paired Al and static targets immediately, prepares Ni after verified full
relaxation completion, seals and packs the training set, then starts one GPU
worker on any compatible partition. Training checkpoints resume across its own
allocation requeues. `technical/launch.json` is the ownership receipt;
`technical/paired-inventory.json` records inclusion/exclusion counts, and
`technical/training-config.json` is generated only after all target shards seal.
The execution bundle freezes code and metric contracts before submission.

October 9 completion: all 6,571 Ni relaxations and 2,670,500 target rows are
sealed. The fitting set has 1,339,157 rows, including 487,680 Ni. The first GPU
job (1029531) stopped in the numerical probe before any training update: its
rotation check inadvertently rounded rotated positions to BF16. The corrected
probe computes that coordinate transformation in FP32, retaining the original
2% tolerance and BF16 neural operations. The fresh retry configuration is
`configs/spatial_architecture/esen_small_relaxed_static_20261009.json`, with
output `output/spatial_architecture/esen-small-relaxed-static-20261009-v3/`.
It reuses the same sealed packed data, moments and monitoring panels. The
previous failed execution and its source snapshot remain unchanged.
