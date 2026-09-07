# MACE: 0.1 ps temporal pairs and per-step cosine — 2026-09-06

User-requested continuation of the strict 80-atom MACE experiment: replace
validation-triggered learning-rate reduction with per-optimizer-step warmup and
cosine decay, and use the available 0.1 ps Al continuations for temporal VICReg.
Launched detached on September 7 at 00:00 Paris time in allocation 983527.

## Reproduction and initialization

```bash
conda run -n pointnet python -m src.training_methods.pretrained_mace \
  --config experiments/pretrained_mace_80_dt01_cosine_20260906/config.json
```

[Configuration](config.json), [standard static analysis](static_analysis.yaml).
The prior run was stopped at the user's request. This phase initializes from
its saved step **2,600**, retaining encoder weights, both prediction heads, and
the fixed feature/TDA scalers. `initialization.pt` contains those weights and
source metadata without optimizer state. AdamW and its schedule start afresh;
this is a changed training protocol, not an exact optimizer resume. The MACE
lineage originates in the official small MACE-MP-0b2 MLIP checkpoint. No teacher
or EMA model is used. Repeated runs require a new output directory and W&B ID.

## Data and objectives

Actual inputs remain center +79 neighbors, with the established smooth 6.5 Å
compact context and a 256D scalar encoder. The 512-point backing caches are
sliced before the model receives data. All four objectives remain: spatial
VICReg, temporal VICReg, latent-to-TDA prediction, and future-latent prediction.

Al continuations from `datasets/zr_al_mg_initial_6x24ps/branches/Al` add training
sources 166/170/174/175 ps (4,096 center IDs per source, four anchors each).
The 177 ps continuation supplies validation (512 IDs, four anchors); 240 ps is
excluded. Metadata and same-atom frame differences verify **0.1 ps** for every
enabled temporal VICReg pair in Al, Mg and Ta. This is the separation of encoder
views; the underlying MD integration timestep is unchanged.

Al shooting records remain useful for spatial/TDA/forecast objectives. Their
0.3 ps pairs are explicitly excluded from both temporal invariance and temporal
spread losses. Their third view can still receive TDA supervision. Forecast
horizons remain 1.2/6/12 ps for shooting, 0.4/2/4 ps for ordinary continuations,
and 0.4/1/2 ps for the later Ta validation interval.

| Material | Training quadruplets | Distinct neighborhood states |
|---|---:|---:|
| Al | 245,760 | 847,096 |
| Mg | 32,768 | 117,338 |
| Ta | 8,192 | 29,334 |
| Total | 286,720 | 993,768 |

There are 1,146,880 view slots before repeated sampling. Distinct states count
source/frame/atom IDs, not independent samples. Every 768-example batch has
128 Al shooting, 128 Al continuation, 256 Mg and 256 Ta examples. Thus **640
temporal pairs** contribute: 128 Al, 256 Mg, 256 Ta. Other objectives use the
whole batch. Independently shuffled pools cycle to give 1,408 updates per epoch.

The warmup lasts one epoch (1,408 updates), increasing encoder LR from 1.5e-6
to 3e-5 and head LR from 1.5e-5 to 3e-4. Cosine decay then updates after every
optimizer step, reaching an absolute minimum of 1e-6 at the configured
28,160-update endpoint. Validation no longer changes LR. The existing 20-epoch,
eight-hour and validation-patience stopping limits still apply; an early stop
need not reach the cosine endpoint.

## Verification and analysis

```bash
PYTHONPATH=. conda run -n pointnet python \
  experiments/pretrained_mace_80_dt01_cosine_20260906/verify.py
```

Schedule checks passed for warmup, per-step decay and the final LR. Temporal
loss/gradient checks passed, including zero contribution from excluded shooting
pairs. A real balanced batch had shape `(768, 4, 80, 3)` and 640 eligible pairs;
masked gradient caching matched full backpropagation within 1.42e-7. Data audits
verified compact context coverage, sampling counts and temporal pair identity.

After training the existing standard static pipeline runs on the selected
encoder alone, on all **772,953 Al centers**, with results in the repository.
The static frames 166/170/174/175 ps are ancestors of training continuations;
177 ps is a validation ancestor. These visualizations are descriptive and are
not a fully independent test. Imposed clusters and spatial coherence alone do
not establish physical phases.

## Outputs and file roles

- Online W&B: <https://wandb.ai/teshbek/PointCloudMaterials/runs/sbviicv4>.
- Run, status, logs, provenance, selected weights and eventual static results:
  `output/pretrained_mace_80_dt01_cosine_20260906/`.
- Bulk data/checkpoints: `/home/ids/vmorozov/experiments/pretrained_mace_80_dt01_cosine_20260906/`;
  existing prepared shards are reused without copying.
- This directory contains versioned experiment records and verification code.
  Shared implementation remains in `src/data_utils/pretrained_mace.py`,
  `src/training_methods/pretrained_mace.py`, and `src/utils/training_utils.py`.
  Generated diagnostics and logs belong to the run output directory.

Training and analysis findings are pending; the workflow writes `RESULTS.md`
after completing the standard analysis.

Five [matched objective ablations](../pretrained_mace_ablations_20260907/README.md)
are scheduled in a detached controller after this workflow completes training,
static analysis, and process exit.
