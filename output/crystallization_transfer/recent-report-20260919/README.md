# Recent local-encoder and crystallization experiments

**Interim report — captured 2026-09-19T18:22:46.696304+00:00 (UTC).** This consolidates the local-support reset, structural pretraining, expanded MACE parent, onset/context scaling, corrected trainable encoders, and both trajectory-forecast studies. Older H200 causal-state/predictive-memory studies use different targets and are not pooled into these comparisons.

> **UPDATE PENDING:** all 35 trajectory-refinement fits and all 52 corrected encoder screens are complete. The ten promoted 12/24-epoch encoder fits are unfinished; four were running at capture. Update this report when `mace-adaptive-20260919` has all 62 fits complete. Follow the [completion checklist](../../../experiments/crystallization_transfer_20260919/REPORT_UPDATE.md). This is a persistent note, not a scheduled monitor.

## What the evidence supports

- Local structural pretraining learns present geometry and instantaneous topology. The expanded MACE fit supplies the checkpoint used throughout the recent crystallization studies. More data **and** more updates improved selection reconstruction; this does not isolate data volume from compute.
- Broader spatial context and real recent history help frozen-MACE onset prediction. Attention beats weighted averaging. More independent training trajectories help more clearly than denser windows from the same sources.
- The original fine-tuned/scratch models had a normalization/optimization failure. Corrected training restores competitive predictions. The completed six-epoch screens do not establish a clear fine-tuning advantage over the best frozen context model; longer fits remain pending.
- Direct and deterministic autoregressive structural forecasts are the strongest trajectory approaches tested. Improvements over the frozen hazard reference are small point estimates, not yet confirmed by paired source intervals.
- Diffusion is now numerically usable, but still worse at onset likelihood than simpler methods. Gaussian AR remains problematic; event-stratified mixtures have not delivered a consistent advantage. More complexity and longer training do not reliably improve every metric.
- All comparisons use one seed. Previously inspected test sources make this exploratory. Selection metrics choose checkpoints/settings; test metrics below describe outcomes and must not be used to select a new winner.

## 1. Structural pretraining and encoder recipe

The comparable local-support runs are complete. Physical and TDA errors below are standardized block-averaged selection MSE; score is physical + 0.25 × instantaneous TDA. Selection uses 480 observations from 15 native-Al sources. Bond error has a different normalization and is not included in that selection score.

| Run | Dynamic anchors | Total updates | Selected update | Physical ↓ | TDA ↓ | Selection score ↓ | Bond error ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MACE local | 254,520 | 622 | 576 | 0.1405 | 0.0966 | 0.1646 | 1.2217 |
| GATr local + bond | 254,520 | 622 | 576 | 0.1203 | 0.1240 | 0.1513 | 0.8687 |
| MACE expanded; downstream parent | 1,018,080 | 2486 | 2486 | 0.1036 | 0.0746 | 0.1223 | 1.2030 |

At the smaller-data budget GATr has better physical reconstruction and MACE better instantaneous TDA. The expanded MACE model improves both over smaller MACE, but sees four times the anchors and approximately four times the updates. Target normalization is refitted on each training release, so the exact errors are not a fully controlled cross-release comparison. The expanded model's training-mean baseline score is 0.51039. Its bond error 1.20305 is only modestly below the zero-output baseline 1.24868; do not interpret the bond auxiliary as solved. Selection is Al-only, so these results do not establish held-out accuracy on Mg/Ti/Ta. No matched no-VICReg ablation here establishes VICReg's causal contribution.

### Architecture and observation

The deployed parent is `StructuralMACE`, two spatial interaction/product blocks, 32 channels with scalar/vector/rank-2 features, cuEquivariance backend, and one 128-dimensional invariant snapshot output. Multiscale pooling uses model-unit supports (0,3), (3,5), and (6,8), followed by a hidden-104 compression head and output LayerNorm. Inputs are positions, species and material scale; no velocity or temporal frames enter an individual encoder call. Spatial messages use cutoff 5 model units. The encoder is shared across materials.

Coordinates are center-relative with consistent periodic handling and rescaled as `x_normalized = x × 9.192189 / material_scale`. Scale is fitted on training observations and fixed at inference. The actual graph is the sphere of radius 8, with a smooth support taper from 6 to 8, **no halo and no fixed atom-count cap**. For downstream Al, scale 9.121389 gives physical support about 7.94 Å. This local sphere is different from the additional 12–25 Å context-center radius used downstream.

### Data, pairs and targets

Only dynamic training rows enter this fit; inherited static data and Zr are excluded. Training strata are Al native 250,000; Al shooting 300,000; other Al including million-atom MEAM/EAM 79,520; Mg 119,280; Ti 150,000; Ta 119,280. These are 1,018,080 observations, not independent trajectories. New rows exclude held-out ancestry. The generating-potential groups and exact per-update quotas are:

| Material/potential registry label | Training rows | Pairs per update |
| --- | ---: | ---: |
| Al / al-lee2003-meam | 569,774 | 1,142 |
| Al / al-mendelev2008-eam | 59,746 | 128 |
| Mg / mg-wilson2016-eam | 119,280 | 239 |
| Ta / ta-zhong2014-eam | 119,280 | 239 |
| Ti / ti-kavousi2019-meam | 150,000 | 300 |

An update randomly chooses spatial or temporal views with equal probability. Spatial pairs use the anchor and a different atom sampled within normalized radius 4.25 in the same frame. Temporal pairs use the same tracked atom in current and next cached frames; the previous frame is also independently encoded for backtracking. Frame intervals come from trajectory timestamps, not one universal material-independent lag. Static rows cannot supply these triples.

Current and paired views receive physical85 (radial/pair/angular/moment blocks), instantaneous TDA144 (H0/H1/H2), and equivariant q4m/q6m supervision. TDA uses the nearest 80 points in the local support. The previous temporal view is curvature context only. **No relaxed TDA** is used. Physical85 is a geometry target; it is not the downstream physical128 packet that includes motion. The physical decoder is an auxiliary MLP from exported z; it preserves accessible information and is discarded when exporting the encoder. Bond prediction instead reads learned equivariant features: MACE's center l=2 channels generate l=4/l=6 tensors through tensor products. It does not decode bond orientation from invariant z.

### Objective and training schedule

For each material/potential group, VICReg compares a 64-dimensional projected pair. Training heads use group-specific normalization so between-material offsets cannot alone satisfy within-group variance. The shared exported encoder has no dependence on other batch observations. Target standardization uses training endpoints only.

The actual objective is:

`L = L_physical + 0.25 L_TDA + 0.1 (25 I + 25 V + C)/51 + 0.1 L_correlation + 0.69 L_backtracking + 0.1 L_bond`.

VICReg/correlation terms are calculated within material/potential groups and weighted by group sample count. The logged raw VICReg total is `25 I + 25 V + C`; its coefficient in the overall objective is **0.1/51**, not 0.1. Physical and TDA blocks are averaged equally within their respective targets. Full statistical batches, not encoder microbatches, determine variance/covariance and head normalization.

Backtracking is **temporal-only**. With unequal gaps h0 and h1 it uses the mean squared Euclidean norm of `2 × [h0(z_next−z_current) + h1(z_previous−z_current)] / (h0+h1)`. Equal gaps recover `z_next−2z_current+z_previous`. It is not divided by squared time gaps. The fixed weight 0.69 was calibrated in a separate training-only preflight to keep its initial influence small; its fraction need not remain constant during learning. Every frame remains snapshot-encoded at deployment.

| Setting | Actual expanded-MACE recipe |
| --- | --- |
| Seed / duration | 20260919; five sampled epoch equivalents; 2,486 updates |
| Batch | 2,048 pairs; microbatch 512; minimum 128 per material/potential group |
| Optimizer | AdamW, weight decay 1e-4 |
| Peak LR | Heads 0.002; encoder 0.0002 (0.1 multiplier) |
| Schedule | 10% linear warmup; cosine to 1% of peak |
| Precision | Compiled selective BF16; geometry/tensor-sensitive operations and objective in FP32; cuEquivariance |
| Execution | Two encoder replicas; one global grouped objective; gradient replay and summed replica gradients |
| Selection | Every 64 updates; physical + 0.25 TDA; final selected update 2,486 |

Sampling is independent per update, without replacement within each group/update; five epoch equivalents do **not** guarantee five complete passes through every anchor. Group-head inference moments are refreshed using training-only observations. The execution-only continuation retained optimizer/scheduler/RNG and changed CPU preparation to four processes with six-batch lookahead, leaving the GPU executor unchanged. Its numerical gradient reproducibility limitation is documented in the [execution audit](../../../docs/shared_pretraining_mace_dual_optimization_20260919.md); do not claim bitwise reproducibility across GPU runs.

### Exact checkpoint and reproduction references

- Parent training recipe: [mace_expanded_dual/mace.json](../../../configs/shared_pretraining/mace_expanded_dual/mace.json).
- Execution continuation: [mace_optimized_dual/mace.json](../../../configs/shared_pretraining/mace_optimized_dual/mace.json). This contains an existing-run transition receipt; it is not a fresh-run recipe.
- Selected training checkpoint: [best.pt](../../shared_pretraining/mace-expanded-dual-20260919/technical/best.pt); standalone [encoder.pt](../../shared_pretraining/mace-expanded-dual-20260919/technical/encoder.pt).
- Selected checkpoint SHA256: `ccca9087cd6974023dc0c5c9631337c542efd96fa28245bea883e771530a4722`.
- Frozen producer code: `output/shared_pretraining/mace-dual-optimized-campaign-20260919/technical/code`.
- Smaller GATr counterpart: [five-epoch bond-supervised recipe](../../../docs/shared_pretraining_local_structure_20260918.md#gatr-bond-order-update).

Historical launch used conda `pointnet-torch214` and:

```bash
python -m src.training_methods.shared_pretraining.queue submit \
  --plan configs/shared_pretraining/mace_expanded_dual/campaign.json
```

That campaign is already submitted and must not be submitted as a new run. For a fresh reproduction, copy the scientific recipe with a new output/W&B identity and valid allocation through the existing launcher. Preserve the pinned release, support, group quotas and objective. The immutable producer and checkpoint hashes identify this result more precisely than current working-tree code.

## 2. Common crystallization protocol

All recent downstream methods use 150 independent Al trajectories (70,304 atoms, 400–520 K): **90 train / 15 selection / 15 alarm calibration / 30 test**, with 16 outcome-independent centers per source. At-risk origins are every 3 ps from 48 through 498 ps. There are **109,838 eligible training windows and 44,385 test windows**, not that many independent trajectories. Pretraining ancestry is checked against downstream calibration/test. No new simulations were required.

The target is first local sustained crystallization: the start of three consecutive crystalline PTM observations after a liquid history. Confirmation frames define labels only. Forecast horizons are 0.75, 3, 9, 24, 48 and 96 ps. Al conditions and time since quench are available to heads. Up to seven spatial centers are used: the anchor and geometry-selected representatives in inner/outer annuli. They are sparse local embeddings, not all atoms within the context sphere; representative identities may change across frames.

History offsets are [0], [-3,0], [-12,-3,0], or [-48,-12,-3,0] ps. Thresholds target 5% FPR using separate calibration sources. Test FPR can differ. Metrics include event NLL, AP/AUROC, calibration, recall/precision, timing with missed-window counts, repeated-alarm episodes and sparse spatial transformation-fraction/pair diagnostics. Detected-only timing MAE never measures performance on missed events.

## 3. Completed initial and scaling studies

**102 fits plus the no-transition baseline completed.** The initial 2,048-update sampler and subsequent full shuffled epochs are distinct protocols. See the [original complete report](../summary-20260919/README.md) for all results and 5,000-draw paired temperature-stratified source bootstrap intervals.

| Frozen scalar comparison | Test event NLL ↓ |
| --- | ---: |
| Radius 0 / 6 / 12 / 18 / 25 Å, three epochs | 1.1118 / 1.1045 / 1.0675 / 1.0384 / 0.9928 |
| One / three / six epochs, radius 25 Å | 0.9883 / 0.9928 / 0.9637 |
| 30 / 60 / 90 training sources, fixed 5,151 updates | 1.0480 / 1.0206 / 0.9928 |
| Snapshot / real 12 ps / real 48 ps, initial budget | 1.0121 / 0.9778 / 0.9776 |
| Repeated-current 48 ps control, initial budget | 1.0120 |

At three epochs, 25 Å beats center-only NLL by 10.7%; attention beats weighted mean by 7.5%. Six versus three epochs and 90 versus 30 independent sources have paired source intervals favoring the larger setting. Three epochs does not clearly beat one. Using all windows rather than 25% within each source did not show improvement. Tensor context did not clearly improve scalar context (six-epoch NLL 0.9659 versus 0.9637). Forty-eight ps did not clearly improve on 12 ps in the initial matched comparison.

The validation-selected frozen reference has test NLL 0.9637. At 9 / 24 / 96 ps its AP is 0.527 / 0.614 / 0.633, source-weighted window recall 72.6% / 62.2% / 31.6%, and detected-window timing MAE 1.87 / 4.49 / 15.52 ps. At 96 ps it misses **8,336 of 12,127** positive windows. Repeated monitoring detects 395/409 observable event centers with mean lead 44 ps, but this is not accuracy at a fixed 96 ps lead. Spatial fraction MAE is 0.0414 / 0.0909 / 0.2722 at these horizons; sparse centers cannot establish dense front-localization accuracy.

Original three-epoch fine-tuning/scratch NLL was 1.1349/1.1346. A training-only audit found stale fixed normalization, encoder mean drift of 45–84 normalization units, heavy clipping and nearly geometry-independent logits. This is an optimization failure, not decisive evidence against trainable encoders.

## 4. Corrected fine-tuning, scratch and attention study

All **52 six-epoch screens** are complete; ten promoted fits are pending completion. The repair uses differentiable full-training-batch moments, training-only refreshed inference moments, separate encoder/head clipping, a smaller fine-tuning LR, and head-only warmup. The auxiliary structural/VICReg objectives are **not** used in these task-specific copies. The pretrained snapshot checkpoint stays unchanged.

The best completed six-epoch screen **within each mode, ranked on selection NLL**, is:

| Setting | Selection NLL ↓ | Test NLL ↓ |
| --- | --- | --- |
| frozen-joint-E6 | 0.9133 | 0.9605 |
| finetune-norm-eps1e-6-E6 | 0.9120 | 0.9615 |
| scratch-history48-E6 | 0.9172 | 0.9716 |

This substantially repairs the old trainable-model failure, but the best fine-tuned and frozen test values are close; no paired superiority claim is justified yet. Scratch is also much more competitive than the failed initial implementation. Top fine-tuning settings are normalization epsilon 1e-6 and 48 ps history; top scratch settings are 48 ps history and encoder LR 1e-5. Frozen joint attention is promoted. Each selected recipe restarts under 12- and 24-epoch schedules. [Protocol](../../../experiments/crystallization_transfer_20260919/ADAPTIVE.md).

## 5. First complete trajectory-forecast comparison

Ten fits cover five methods × 12/24 epochs. The backbone is frozen. Every method predicts 32 structural states at 3 ps intervals through 96 ps: 128 MACE channels + physical128 + eight bond descriptors + instantaneous crystallinity. Dense onset indicators use 0.75 ps resolution. This is a descriptor trajectory, **not atom-coordinate molecular dynamics**, and no TDA reconstruction target is present in this forecast packet.

Direct predicts the full future jointly; deterministic/Gaussian AR feed previous predicted states; the mixture samples whole component trajectories; diffusion generates whole noisy structural/event paths. Forecasts receive no future observations. These runs retain the same source-held-out population. Each row below chooses its budget by original selection Brier, not test NLL.

| Model | Selection Brier ↓ | Test event NLL ↓ | Physical path MSE ↓ | 9 ps timing MAE | 96 ps timing MAE |
| --- | --- | --- | --- | --- | --- |
| direct-E12 | 0.1047 | 0.9572 | 0.8514 | 1.93 | 16.71 |
| ar_mse-E12 | 0.1018 | 0.9539 | 0.8630 | 1.92 | 16.16 |
| ar_gaussian-E24 | 0.1346 | 1.0617 | 0.9650 | 1.87 | 16.09 |
| mixture-E24 | 0.1064 | 0.9793 | 0.8824 | 1.99 | 16.76 |
| diffusion-E24 | 0.7098 | 4.3064 | 8451.0750 | 2.82 | 45.11 |

Physical MSE is standardized on training targets and averaged over forecast times. Timing here uses dense 0.75 ps onset mass, whereas the original hazard reference uses six bins; their timing estimators are not identical. Deterministic AR's 9/96 ps miss counts are **295/974 and 8,172/12,127**. The modest event-likelihood gain over the frozen hazard reference does not establish improved timing or a statistically significant generalization gain.

Direct selected around epoch 3 and then worsened. Gaussian sampled feedback was worse than mean feedback on a selection-only replay (Brier 0.1463 versus 0.1353). Mixture diversity was mostly within-component noise, not complete component collapse. Old diffusion compressed 269 noisy channels into 128 and amplified terminal denoising errors by about 1,299; its bad scores are implementation evidence, not a verdict on diffusion generally. [Diagnosis](../../../experiments/crystallization_transfer_20260919/PATH_REFINEMENT.md).

## 6. Completed targeted trajectory refinements

**All 30 screens and five longer promotions completed.** Screens had 12-epoch caps; promotions had 36-epoch caps with early stopping. Promotions minimize selection Brier among models within 10% of their family's best selection physical error (equal average of physical packet, bond and crystallinity blocks). This gate is not the physical128-only column below. Selection expanded from 64 to 128 windows/source; raw selection scores across the two studies are not paired comparisons.

The five predeclared longer runs are shown without choosing the best test result:

| Family | Promoted screen | Actual epochs | Selection Brier ↓ | Test NLL ↓ | Physical path MSE ↓ |
| --- | --- | --- | --- | --- | --- |
| ar_gaussian | ar_gaussian-likelihood-E12 | 15 | 0.1642 | 1.2825 | 0.9009 |
| ar_mse | ar_mse-history48-E12 | 15 | 0.1017 | 0.9508 | 0.8637 |
| diffusion | diffusion-steps32-E12 | 18 | 0.1089 | 1.1172 | 0.8781 |
| direct | direct-history48-E12 | 11 | 0.1022 | 0.9580 | 0.8503 |
| mixture | mixture-stratified4-E12 | 15 | 0.1047 | 0.9649 | 0.8805 |

Findings by family:

- **Direct:** matched control/history48 test NLL is 0.95562/0.95523, a tiny point difference. Lower LR, dropout, motion and reduced structural weighting do not give a compelling joint benefit. The longer promoted run is worse at 0.95801. Keep the simple reference until differences are supported by paired uncertainty.
- **Deterministic AR:** H48 selected slightly better than control, but their test NLLs are 0.95421/0.95386. The long run reaches 0.95083, yet its selection Brier is worse than the short H48 checkpoint. It is therefore not automatically the selected model merely because its test result is lower. Earlier teacher-forcing removal and added motion do not clearly help.
- **Gaussian AR:** exact-likelihood training and low-rank covariance do not solve the problem. Rank-8 has better selection event score but selection physical error 1.9781, versus 0.8826 for the diagonal model, so it fails the physical gate. Its physical-only test MSE is 2.5365. Promoted diagonal likelihood improves with longer training but remains poor at event NLL 1.28255. Do not promote risk quality by discarding physical quality.
- **Mixture:** four onset/survival components improve selection Brier over the matched control (0.10432 versus 0.10480), but test NLL is worse (0.97079 versus 0.96420). The longer mixture reaches 0.96493 without a clear gain over the simpler control. Stronger structural weighting is worse here.
- **Diffusion:** stable v/x0 prediction, a full-dimensional noisy skip and absorbing-event projection remove the numerical failure. The promoted model's physical MSE is 0.8781 instead of the old 8451.1. However, its event NLL 1.11721 still trails direct/AR; even the 32-step screen's 1.10301 does. Larger width and EMA do not give a clear advantage. Improved Brier can coexist with worse event NLL because those scores weight calibration/tails differently.

No new paired source intervals have yet been computed for these refinement contrasts. Avoid interpreting tiny differences or the lowest observed test result as a confirmed winner. The [complete 35-fit appendix](REFINEMENT_FITS.md) includes every completed setting, not just favorable outcomes. The [52-screen appendix](ADAPTIVE_SCREENS.md) preserves the corrected encoder comparison.

## 7. What remains to finish

Complete the ten longer native encoder fits, then update the validation-selected frozen/fine-tuned/scratch comparison and timing-with-misses. Compute paired whole-source uncertainty for predeclared contrasts, retaining the existing split and noting one-seed/exploratory limits. Compare frozen context versus direct/AR in physical event outcomes rather than raw latent MSE. Distinguish scalar/tensor context results and 12/48 ps history controls. Do not replace failed historical runs or mix their metric definitions with the revised implementations.

All values above are traced to a [captured result snapshot](technical/snapshot.json) with per-input SHA256 hashes. Existing campaign `tables/METRICS.md` and `technical/metric-contract.json` remain the authoritative frozen definitions. The disposable [capture helper](technical/capture.py) reads current artifacts; it does not train or change jobs. No monitoring or new training was started for this report.
