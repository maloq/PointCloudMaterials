# Broad full-TDA GATr continuation

The active recipe is `configs/shared_pretraining/broad_full_tda/`. It continues
the completed Al-only GATr's best checkpoint (step 1216) for three epoch
equivalents, with all weights trainable and fresh optimizer/scheduler state.
The parent is copied into the new run's `technical/parent.pt`; its SHA256 and
completed-run status are preserved in `parent-provenance.json`.

## Data

| Training stratum | Anchors |
| --- | ---: |
| Al native | 62,500 |
| Al shooting | 75,000 |
| Other Al (million-atom MEAM/EAM and static) | 25,000 |
| Mg | 37,500 |
| Ti | 37,500 |
| Ta | 37,500 |
| Zr static | 12,500 |
| Total | 287,500 |

The immutable parent release's 250,000 training anchors and 480 native-Al
selection observations are preserved. An additional 37,500 shooting anchors
are balanced over 17 eligible ancestral lineages and distributed across all
456 training-eligible trajectories. Added source/frame combinations exclude
those sampled in the parent. These trajectories are correlated descendants,
not 456 independent sources. Other metals retain the original frozen sample.

Every supervised anchor/spatial/next-frame view receives instantaneous TDA144
using the same physical nearest-80 AlphaComplex descriptor. No relaxed targets
or new simulations are produced. Context-only history frames remain unlabeled;
this run uses snapshot inputs. Existing coordinate arrays are hard-linked into
the new cache, and TDA arrays are atomically replaced there, leaving the parent
unchanged. Interrupted preparation resumes from completed full-TDA receipts.
The cache is registered as `structural-neighbors-287500-full-tda-20260918`.
Potential/source metadata and unknown static potentials follow its parent.

## Training and evaluation

Batch 1024, encoder microbatch 256, 40 GiB allocator cap, compiled selective
BF16 with FP32 geometry, one seed. Three epochs yield 843 updates. Cosine
schedule with 10% warmup uses head peak LR 0.002, encoder peak LR 0.0002 and
1% final multiplier. VICReg pairs and all objective coefficients remain as
in the Al stability run. Batches remain homogeneous in material/potential and
static/dynamic group; restoring materials does not let between-material means
satisfy the VICReg variance requirement.

New training-only target moments are fitted. Decoder final layers are rebased
to preserve physical-unit outputs across the normalization change, while encoder
and projector weights transfer unchanged. Training-only head calibration is
retained. A step-zero selection pass provides the parent baseline in these
new units; it can remain the best checkpoint. Selection remains 15 native-Al
sources and does not validate generalization to other metals. See the
[metric contract](metrics/shared_pretraining.md#broad-full-tda-structural-continuation).

## Detached execution

```bash
conda run -n pointnet-torch214 python -m src.training_methods.shared_pretraining.queue submit \
  --plan configs/shared_pretraining/broad_full_tda/campaign.json
```

The existing queue freezes source/configs, submits a 16-worker CPU preparation
job (96 GiB, four-hour limit), then a dependent GPU fit (eight-hour limit) to
RTX6000PRO/H100/L40S. The GPU job starts only after successful data preparation;
training checks complete TDA coverage before loading the model. No hardware
benchmark runs in this pipeline. Submission is one-shot; inspect the receipt
before attempting any resubmission. Existing allocations/jobs are not stopped.

Receipt/logs: `output/shared_pretraining/broad-full-tda-campaign-20260918/technical/`.
Fit: `output/shared_pretraining/gatr-broad-full-tda-20260918/technical/`.
Preparation status: `${storage:cache}/structural_pretraining/broad-287500-full-tda-20260918/status.json`.
W&B run: `teshbek/PointCloudMaterials/gatr-broad-tda-0918`.

Validation before submission: 25 CPU tests passed, including target reproduction,
parent immutability, training-only source expansion, decoder-unit transfer,
exact optimizer resume and CPU-to-GPU dependency behavior. Real-data preparation
also passed on an inherited Mg static shard and a newly sampled Al shooting shard.
