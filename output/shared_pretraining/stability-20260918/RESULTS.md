# Al-only VICReg stability repair

Captured 2026-09-18T11:27:21.582731+00:00.

## What failed

GATr v5 failed at update **2** with `FailOnRecompileLimitHit`, not a nonfinite
loss. Its upstream attention wrapper broke graphs; contraction-path planning
also specialized every atom count despite `dynamic=True`. A real full-batch
regression test then exposed a backward saved-stride specialization. The fixes
are tensor-only native attention, disabling both einsum path optimizers,
contiguous grade-zero scalar views and runtime Triton strides. Full-graph
compilation is required. The final RTX workload completed **eight updates**
across all three Al groups, batches 1024/128, varying atom counts, evaluation,
no-grad caching and backward with **two graphs and no graph breaks**.

MACE v5 was stopped cleanly at **648/2930**. Physical selection was improving
overall: 0.7926 at update 64, best 0.2473 at 576, 0.3366 at 640 and 0.2938 at
648. Those are the **old broad-normalized units**, not comparable to the scores
below. Its VICReg representation was concentrated in roughly one direction;
static Mg produced raw VICReg **153.575**, almost entirely covariance
(**151.687**). That was a real instability, not evidence that every physical
objective had diverged. The preceding short Al-only pilot did not cover it.

## Conditioning experiments

All values below use the new **Al-training-only target normalization** and
480 selection observations from 15 held-out native Al MEAM sources. Lower is
better. The constant training-group-mean score is **0.8598**. Each diagnostic
fit used one seed, 384 updates, 128 pairs/update, all three Al training groups,
physical + 0.25 instantaneous TDA + 0.1 VICReg/51 + 0.1 correlation, encoder LR
0.0002 and head LR 0.002 with 40 warmup updates.

| Backbone | Projector conditioning only | Physical/TDA conditioning, stale statistics | Same weights, refreshed training statistics |
|---|---:|---:|---:|
| MACE | 0.8579 | 1.7372 | **0.3220** |
| GATR | 0.8603 | 1.0356 | **0.3011** |

Projector conditioning alone lowered VICReg but left physical predictions
nearly constant. Conditioning the physical/TDA heads exposed useful signal;
ordinary moving averages then lagged the moving encoder. The mean error was
6.56 native feature standard deviations for MACE and 28.29 for GATr. Refreshing
moments using only training inputs recovered the held-out decoding above.
This was detected in the new diagnostic candidate; the old v5 heads used
LayerNorm and did **not** have this BatchNorm running-statistics issue.

The paired stale/refreshed columns use identical learned weights. The audit
used a deterministic proportional sample of approximately 512 training
anchors drawn from fixed 512-row samples per Al group. New production runs
use 1024 fixed proportional training anchors. Neither uses held-out moments
or labels. Native-only calibration was also checked diagnostically, but is
not the adopted all-Al calibration. Several candidates reused this selection
set, so this is exploratory diagnosis, not an untouched test-set comparison
or a backbone ranking.

![Readout conditioning diagnostic](plots/conditioning.png)

## Fresh detached fits

- **MACE**: running, update 32/1465. [W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/mace-v6-al-0918).
- **GATR**: running, update 32/1465. [W&B](https://wandb.ai/teshbek/PointCloudMaterials/runs/gatr-v6-al-0918).

Only **Al**: 104864 MEAM, 15016 EAM and 5120 static training anchors. The static
potential remains unknown. Inputs and target normalizers exclude every other
metal. Held-out validation currently tests native Al MEAM; EAM/static
out-of-source generalization is not established. No new data was generated.

Both fits use one seed, 12 epoch equivalents (**1465 updates**), batch 1024,
selective BF16 with FP32 geometry, compiled encoders, peak head LR **0.002**,
peak encoder LR **0.0002**, 147 warmup updates and cosine decay. Homogeneous
potential/static batches preserve within-group VICReg statistics. Spatial
and temporal neighbors remain separate logged views. Instantaneous TDA only.
The exported encoder contains no BatchNorm and remains independent of batch
peers. The validated full checkpoint stores frozen training-fitted head
moments; `last.pt` is optimizer-resume state. A 5% baseline-gain gate after
update 256 and sustained-regression checks replace reliance on a good old
checkpoint alone. The full 12-epoch outcomes and future-prediction skill are
not yet known.

![Current Al-only training](plots/training.png)

## Verification and reproduction

- 40 tests passed on H100, including full-graph variable-size gradients and
  rotation checks; 38 noncompiler tests repeated on RTX.
- Material filtering and training-only calibration tests verify that held-out
  or excluded-material records cannot enter sampling or fitted moments.
- Decoder outputs are invariant to evaluation batch peers and survive exact
  state-dict reload. Encoder weights are unchanged by calibration.
- Full-vs-cached FP32 gradient discrepancies were measured, not hidden by a
  loss-only assertion: 0.0326% for MACE snapshot, 0.000062% for GATr snapshot,
  and 0.098% in the separate uneven-chunk causal test. Tests bound the whole
  gradient error to 0.2%; per-weight tolerances also remain enforced.
- The final compiler workload and failed intermediate reproducers are retained
  under `technical/`. Timing in these correctness tests is not a new matched
  FP32 speed benchmark.

[Execution recipe](../../../docs/shared_pretraining_al_stability_20260918.md) ·
[Metric definitions](tables/METRICS.md) · [Metric CSV](tables/stability.csv) ·
[Launch receipt](../al-stable-campaign-20260918/technical/submissions.json).
Frozen production code/configuration is in the launch receipt. Diagnostic
scripts and stopped-run traces are under `technical/`.

The compiler investigation follows [PyTorch's recompilation guidance](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model/recompilation.html)
and the [pinned upstream GATr attention implementation](https://github.com/Qualcomm-AI-research/geometric-algebra-transformer/blob/6afc26f26b8fcf51136ae8c1d264a36e14b6e497/gatr/primitives/attention.py).
