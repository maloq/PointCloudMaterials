# Atomic predictive state and memory: consolidated results

**Evidence captured 17 September 2026, 11:54 CEST (09:54 UTC).** This report covers the native causal-state and partial-observation memory experiments undertaken in this task, including the supplied H200 results. It includes completed comparisons and explicitly identifies ongoing work. Earlier pretrained-encoder studies remain in the [research index](../../../experiments/README.md).

## Main findings

1. **History can help, but the benefit depends on the objective and comparison.** The older H200 study found a modest benefit from 2.25 ps history at width 32 on the predeclared low-order subset. The newer 3,000-update study did not establish a consistent benefit from 12 or 48 ps history across two seeds and both controls.
2. **Increasing width is not supported at the tested budgets.** Width 32 was inconclusive in the older study and worse than width 16 in all eight matched point estimates in the newer H200 comparison. This does not establish an optimum after convergence.
3. **The strongest new signal comes from preserving present information.** With present-reconstruction weight increased from 0.05 to 1.0, the first completed H100 seed favors 48 ps history over both controls on the joint predictive likelihood. Removing its sample-specific embedding also hurts prediction more. The second seed is pending.
4. **The native prediction heads still leave substantial physical predictability unused.** A linear predictor of the future physical packet from the current packet has test MSE 0.7420, versus 0.8443 for the stronger-loss 48 ps model. This is a matched information reference for position-and-velocity inputs.
5. **No result yet establishes a sufficient local state, a memory cutoff, reliable onset prediction, or the old 0.10 smoothness target.** The new precision-controlled simulations will support a separate test of storage precision and temporal resolution; they have not contributed to these fitted results.

## Evidence and completed work

| Cohort | Encoder training runs/stages | Frozen learned probes | Budget | Status at capture |
| --- | ---: | ---: | --- | --- |
| Initial H100 causal pilot, three seeds | 21 | 72 | 1,000 updates; E adds another stage | Complete, including three Gaussian D fits |
| Longer packed H100 causal cohort, one seed | 3 | 8 | 5,000 updates; probes 2,000 | Complete |
| H200 causal width comparison, three seeds | 18 | 48 | Declared 5,000 updates; probes 2,000 | Complete, user-reported |
| H100 partial-observation pilot and replicate | 12 | — | 3,000 updates | Complete, with state-use/ridge diagnostics |
| H200 partial-observation width comparison | 8 | — | 3,000 updates | Complete, user-reported |
| H100 optimization follow-up | 9 of 16 | — | 12,000 updates | Both seed-17 quartets complete; seed-18 original H0 complete |

The counts sum to **71 completed encoder training runs/stages and 128 learned probes**, not 71 independent scientific replications. E continues a selected D model; budget follow-ups reuse data and seeds. Diagnostic ridge fits are not included in the learned-probe count. The H200 predictive-memory study reportedly took 95 minutes including comparisons.

Local values were read from completed run artifacts; incomplete fits are excluded from matched means. H200 means and qualitative confidence-interval findings come from the supplied summaries. **Raw H200 per-source predictions and numerical interval endpoints have not been received here**, so they are not independently recomputed or combined with local results. The reported H100 reference means in the H200 width table agree with the local two-seed results to their supplied precision.

## Two distinct scientific protocols

| Property | Older causal-state study | New partial-observation memory study |
| --- | --- | --- |
| Main question | Does early geometry–velocity–history interaction improve an exported state? | Does observed history improve the predicted distribution of a fixed physical future under limited spatial observation? |
| Data | 150 sources: 90 train / 30 validation / 30 test | Same exploratory source split |
| Samples | Four centers × three anchors: 1,800 windows | One inherited center × three anchors: 450 windows |
| History | 2.25 ps; oldest observation has zero age-envelope weight | 0, 12, 48 ps, at 0.75 ps cadence |
| Future offsets | 0.75, 3, 9 ps | 0.75, 3, 12, 48, 96 ps |
| Targets | 169 structure, topology and motion coordinates; additional change/path/event supervision | 128 continuous geometry/motion coordinates per future; no topology or crystallization labels |
| Main physical error | Equal average of six standardized target blocks | Equal average of standardized coordinates and lags |
| Primary reported predictor | Matched nonlinear readout fitted after freezing each encoder | Jointly trained four-component path-density head |
| Spatial observation | Exact space–time ancestors provide computational halo | Total radius 17 Å, including all computation; smooth taper from 15 Å |
| Representation | Trainable atom-level scalar, vector and rank-two channels; one final pooling stage | Same early-interaction principle, with explicit partial-observation contract |

The new target packet combines radial and pair-distance bases, angular correlations, relative-speed and signed radial-motion bases, and geometric/motion moments. Its target region has full weight through 5 Å and tapers to zero at 7 Å. Each mixture component describes the **whole five-lag, 640-coordinate future path**, with diagonal-plus-rank-two covariance. Joint negative log-likelihood (NLL) is divided by 640 and reported in nats per coordinate per lag; lower is better. It scores the predicted distribution, while MSE scores its mean. Neither score is a measured mutual information.

All inputs are [causal observed histories](../../../docs/research_glossary.md#observed-history-and-causal-input): only present and past frames reach the encoder. A repeated-current-frame control is trained separately with current observations occupying the historical offsets. See [predictive-memory terminology](../../../docs/research_glossary.md#predictive-memory-in-partial-observations) and the [frozen metric definitions](tables/METRICS.md). Scores from the two protocol columns must not be ranked against one another.

## 1. Original causal-state pilot: history helps a weak velocity baseline

Three training seeds; width 16; 1,000 updates per primary fit and 500 per probe. The table uses matched frozen nonlinear readouts and the **17-source low-order test subset**, defined by current group qbar6 < 0.30. This is an observable threshold, not an independent phase assignment. J uses eligible adjacent pairs from 16 sources.

| Model | Present physical MSE ↓ | 9 ps future physical MSE ↓ | J at 0.75 ps ↓ |
| --- | ---: | ---: | ---: |
| A: position snapshot, present supervision | 0.4729 | 0.4607 | 0.8969 |
| B: A + future supervision | 0.4777 | 0.4608 | 0.8698 |
| C: B + velocities in messages | 0.5922 | 0.5734 | 0.9880 |
| D: C + real atom history | 0.5472 | 0.5312 | 0.8805 |
| Repeated-current-frame control | 0.5855 | 0.5667 | 1.0068 |
| E: D + additional constrained-slowness stage | 0.4904 | 0.4730 | 0.9000 |

Paired 9 ps error differences, first model minus second; negative favors the first:

| Comparison | Difference | 95% source-bootstrap interval |
| --- | ---: | --- |
| B − A | +0.0001 | [−0.0092, +0.0096] |
| C − B | +0.1126 | [+0.0727, +0.1577] |
| D − C | −0.0422 | [−0.0735, −0.0137] |
| D − repeated frame | −0.0355 | [−0.0719, −0.0082] |
| E − D | −0.0583 | [−0.0701, −0.0468] |

Real history improves C, but **the position-only A/B models remain better than D** in this short pilot. E has additional optimization, so its improvement does not isolate a slowness effect. The broad ordering points to optimization and information-retention problems as well as architectural questions.

The [normalized RMS jump](../../../docs/research_glossary.md#normalized-rms-jump) compares embedding increments with within-source covariance: J = sqrt(mean squared increment / (2 × covariance trace)), followed by source averaging. None approaches the declared 0.10 criterion. Raw latent jump values do not establish physical forecasting skill.

Artifacts: [full comparison](../../mace_causal/pilot-comparison-20260916/README.md), [collected scores](tables/causal-pilot_1000-summary.csv), [paired differences](tables/causal-pilot_1000-paired-differences.csv). The tables retain linear and nonlinear readouts, both populations, and all three future horizons.

### Longer H100 causal comparison

With width 16, one seed, 5,000 updates, batch eight and 2,000-update probes:

| Model | Present MSE ↓ | 9 ps MSE ↓ | J at 0.75 ps ↓ |
| --- | ---: | ---: | ---: |
| C: current positions and velocities | 0.3715 | 0.3987 | 1.0015 |
| D: real 2.25 ps history | 0.3636 | 0.4043 | 0.9870 |
| Repeated current frame | 0.3680 | 0.3963 | 0.9973 |

D − C is +0.0056 [−0.0048, +0.0159]; D − repeated frame is +0.0080 [−0.0024, +0.0187]. Thus the shorter pilot's history advantage does not carry over clearly to this longer single-seed cohort. Both encoder and probe budgets and the statistical batch changed, so improvement over the original pilot cannot be attributed to one change. [Completed comparison](../../mace_causal/h100-packed-width16-comparison/README.md).

### H200 causal width comparison

The supplied study completed 18 encoders and 48 probes, across three seeds. At 9 ps on the same declared 17-source low-order subset:

| Input | Width 16 MSE ↓ | Width 32 MSE ↓ |
| --- | ---: | ---: |
| Current positions and velocities | 0.4009 | 0.4028 |
| Real 2.25 ps history | 0.3990 | **0.3946** |
| Repeated current frame | **0.3963** | 0.4056 |
| Persistence | 0.7209 | 0.7209 |

At width 32, history reduces error by about **2.0% against snapshot** and **2.7% against repeated frames**; both paired source intervals reportedly exclude zero. Width-32 history improves only about 1.1% over width-16 history, with an interval spanning zero. History beats snapshot in two of three seeds and is less clearly advantageous across all 30 test sources. At width 16, the repeated-frame mean is lower than real history. These support a small, conditional benefit, not a universal history or width advantage. [Reported numbers](tables/causal-h200-reported.csv).

### Other causal-pilot diagnostics

The Gaussian-head ablation did not improve mean prediction at the tested budget. These are **jointly trained D heads**, distinct from the frozen nonlinear probes above:

| Seed | Deterministic 9 ps MSE, low-order ↓ | Gaussian 9 ps MSE, low-order ↓ | Gaussian NLL, all sources | One-sigma coverage, all sources |
| --- | ---: | ---: | ---: | ---: |
| 20260916 | 0.5411 | 1.3634 | −1.1136 | 76.3% |
| 20260917 | 0.6404 | 1.2488 | −1.0516 | 74.0% |
| 20260918 | 0.5170 | 1.2195 | −1.1765 | 83.0% |

The coverage reference is 68.27% for a calibrated univariate Gaussian, but aggregate coverage alone does not establish calibration. Continuous-density NLL can be negative. This older marginal Gaussian score is not comparable to the newer joint-path NLL. [Per-seed table](tables/causal-gaussian-ablation.csv).

The [state-sufficiency diagnostic](../../../docs/research_glossary.md#state-sufficiency-diagnostic) gives a matched predictor access to original history alongside frozen z. Its low-order 9 ps error changes are:

| Frozen encoder, original pilot | History-access minus constant-history MSE | 95% source interval |
| --- | ---: | --- |
| A | −0.0052 | [−0.0068, −0.0038] |
| B | −0.0079 | [−0.0175, +0.0019] |
| C | +0.0119 | [+0.0063, +0.0177] |
| D | +0.0014 | [−0.0068, +0.0096] |
| Repeated frame | −0.0025 | [−0.0040, −0.0010] |
| E | −0.0024 | [−0.0110, +0.0061] |

Raw history improves the fitted diagnostic for A and repeated frames, indicating accessible predictive information absent from those states. The C diagnostic worsens, illustrating that extra input does not guarantee successful fitting. D's result is unresolved, as is its longer-H100 counterpart, −0.0050 [−0.0132, +0.0031]. Null results do not prove sufficiency. [All sufficiency scores](tables/causal-pilot_1000-sufficiency.csv).

Local onset evaluation has only **one distinct test source/center event**, represented by two overlapping positive windows. The archived event tables are available, but cannot establish general detection, calibration or timing skill. This is [local finite-horizon transition risk](../../../docs/research_glossary.md#finite-horizon-local-transition-risk), not whole-system nucleation or a committor.

## 2. New partial-observation study: no replicated short-budget history gain

All 12 H100 pilot/replicate fits finished. Each used 3,000 updates, width 16, exported dimension 128 and loss NLL + 0.05 × present MSE. Checkpoints were selected by validation NLL. Scores average three neighboring anchors within each test source, then the 30 sources equally.

| Input | Seed 20260917 NLL ↓ | Seed 20260917 future MSE ↓ | Seed 20260918 NLL ↓ |
| --- | ---: | ---: | ---: |
| Positions, H=0 | 0.92851 | 0.93576 | — |
| Positions, H=12 ps | 0.92673 | 0.88809 | — |
| Positions, H=48 ps | 0.93231 | 0.86501 | — |
| Positions, repeated frame | 0.92823 | 0.93542 | — |
| Positions + velocities, H=0 | 0.92531 | 0.93632 | 0.92405 |
| Positions + velocities, H=12 ps | 0.92226 | 0.93548 | 0.92847 |
| Positions + velocities, H=48 ps | 0.92256 | 0.93494 | 0.92867 |
| Positions + velocities, repeated frame | 0.92354 | 0.93606 | 0.92363 |

For position-and-velocity inputs, define positive gain as control NLL minus real-history NLL:

| H=48 comparison | Seed 20260917 gain [95% source interval] | Seed 20260918 gain [95% source interval] |
| --- | --- | --- |
| Against snapshot | +0.00275 [+0.00060, +0.00475] | −0.00462 [−0.01930, +0.01223] |
| Against repeated frame | +0.00098 [−0.00165, +0.00351] | −0.00504 [−0.00954, −0.00117] |

The first-seed snapshot comparison is positive, but does not replicate and does not establish an advantage over both controls. Positions-only history improves mixture-mean MSE while failing to improve the primary density score clearly; these are different aspects of prediction. [All fit scores](tables/memory-fits.csv), [paired gains](tables/memory-paired-gains.csv).

### H200 width comparison under the new protocol

Two-seed mean joint-path NLL, all 30 test sources:

| Input | Width 16, H100 ↓ | Width 32, H200 ↓ |
| --- | ---: | ---: |
| Current positions and velocities | **0.9247** | 0.9354 |
| 12 ps history | **0.9254** | 0.9369 |
| 48 ps history | **0.9256** | 0.9302 |
| Repeated current frame | **0.9236** | 0.9405 |

Width 32 is reportedly worse in all eight per-seed comparisons; six source intervals favor width 16. Its 48 ps history improves mean prediction MSE by only 0.54% against snapshot, a separate statement from this NLL table. There is no clear reported width benefit at the 96 ps horizon. The evidence supports retaining width 16 for current experiments; additional training could change the capacity comparison. [Width table with provenance](tables/memory-width-comparison.csv).

![Width comparisons under the two distinct protocols](plots/width-comparisons.png)

## 3. Why the objective was changed

Replacing each embedding by its training mean while retaining the trained prediction head and actual temperature condition does not hurt the original xv models' validation NLL. For seed-17 H=48, the change is −0.00083; for seed 18 it is −0.00105. The fitted likelihood head makes little effective use of sample-specific state under this intervention. This does **not** prove that the embedding is constant or contains no predictive information.

A separate linear ridge diagnostic, trained only on training sources and selected on validation sources, finds:

| Future predictor | Validation MSE ↓ | Test MSE ↓ |
| --- | ---: | ---: |
| Temperature only | 0.8249 | 0.9277 |
| Current physical packet + temperature | **0.6900** | **0.7420** |
| Original H=48 exported state + temperature, seed 17 | 0.8049 | 0.8565 |

The current packet includes velocities, so it is observation-matched to xv inputs and only an information reference for positions-only inputs. Its stronger performance shows accessible predictive signal in the current observations that the native fitted system has not fully captured. These readouts diagnose the exported state; they do not replace it with a separate smoothing model. [State-use and readout table](tables/state-use.csv).

## 4. Ongoing optimization follow-up: promising first seed

The follow-up gives all xv variants 12,000 updates and compares present-reconstruction weights 0.05 and 1.0, keeping the physical data, radius, model width, target and checkpoint-selection rule fixed. At capture, both seed-17 quartets and the original-loss seed-18 snapshot are complete: **9 of 16 fits**. The remaining comparisons are pending.

| Input, seed 20260917 | Weight 0.05 NLL ↓ | Weight 1.0 NLL ↓ | Weight 0.05 future MSE ↓ | Weight 1.0 future MSE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Snapshot | 0.92531 | 0.94213 | 0.93632 | 0.85148 |
| 12 ps history | 0.92226 | 0.91071 | 0.93548 | 0.87939 |
| 48 ps history | 0.92256 | **0.90483** | 0.93494 | **0.84429** |
| Repeated frame | 0.91780 | 0.92719 | 0.93127 | 0.93349 |

Under the original loss, snapshot/H12/H48 still select update 2,250: the larger permitted budget alone does not improve their selected seed-17 checkpoints. Repeated frames select update 6,000. Under stronger reconstruction, the selected updates are 6,250 / 3,500 / 4,000 / 2,250 for snapshot/H12/H48/repeat respectively. These are validation choices, not test-selected stopping points.

For the stronger objective, paired source intervals favor history:

| Comparison | NLL gain, positive favors history | 95% source interval |
| --- | ---: | --- |
| H12 against snapshot | +0.03142 | [+0.01352, +0.05218] |
| H48 against snapshot | +0.03729 | [+0.01930, +0.05762] |
| H48 against repeated frame | +0.02236 | [+0.01173, +0.03409] |

H48's future MSE improves by 9.7% and present MSE by 22.0% relative to its original-loss counterpart. However, its future MSE is only about 0.84% lower than the **stronger-loss snapshot**. That snapshot improves MSE while worsening NLL. Thus the larger NLL separation cannot be described as the same-sized improvement in mean forecasts; it reflects the full predicted distribution.

The stronger H48 head also becomes more dependent on state: replacing z by its training mean increases validation NLL by **0.00992** and exploratory test NLL by **0.03653**. Its frozen linear state readout achieves test future MSE **0.78595**, better than its native mixture mean's 0.84429 but still above the current-packet reference's 0.74204. This narrows the gap while leaving both encoder retention and head fitting open questions.

**This is a first-seed result.** The source intervals condition on that fitted seed and do not establish replication. The partial seed-18 cohort is listed in the machine table but does not enter a two-seed comparison.

![Present-information objective follow-up](plots/present-loss-followup.png)

![Dependence of fitted predictions on the exported state](plots/state-use.png)

## 5. Numerical precision and new data

Legacy inputs are stored as full-box float16 positions. No matched full-precision model comparison is available yet, so physical memory and denoising of quantization effects are not separated by these results.

The new campaign plans 12 independent 70,304-atom Al sources at 500/520 K, with six training, two validation and four sealed-test sources. Each has a fresh 300 ps melt, 15 ps equilibration and 192 ps measurement at 0.075 ps cadence. Paired float32 reference and float16 canonical exports retain 2,561 frames per source.

At capture, **two training sources are complete and verified in durable storage**, two further sources are running, and eight have not started. The completed sources have componentwise position quantization RMS 0.01263–0.01264 Å and maximum absolute error 0.03125 Å. These are storage-error measurements, not errors in relative geometry, physical targets or forecasts. Quantifying their effect on the research result is still pending. Sealed-test physical outcomes have not been inspected.

The new 192 ps measurement segment samples a different time regime from the old near-300 ps anchors. Its first role is a controlled precision/cadence study, not an interchangeable enlargement of the existing benchmark. [Captured production status](tables/data-production-status.csv); [simulation design and execution record](../../../docs/simulations/predictive_memory_precision_20260917.md).

## 6. Runtime work and its limits

Packed disjoint graphs, resident histories/targets, and batched evaluation are integrated into the main checkout. The completed older-protocol H100 benchmark measured approximately **1.47–2.06× training throughput** improvements across widths 16/32 and batches 2/8/16. Width-16 batch-eight evaluation improved **4.10×**. Maximum packed output absolute discrepancy was 1.19e−6; maximum full-gradient relative L2 discrepancy was 1.06e−6 in the declared FP32 checks.

These timings were collected while the original pilot shared the GPU. They are provisional contended measurements, and are not an H100-versus-H200 hardware comparison or a benchmark of the newer long-history implementation. A larger batch or width changes the statistical experiment as well as memory use. [Full benchmark, memory use and validation record](../../mace_causal/runtime-h100-20260916/README.md).

## What the evidence supports doing next

1. Finish the already-running two-seed objective comparison before judging the stronger-loss history result. Preserve separate NLL, mean-error and state-use conclusions.
2. Keep width 16 as the working baseline. Use additional GPU time first for matched replication and objective/head diagnostics rather than a larger-width sweep.
3. Compare the native head with matched readouts on frozen states and current observations. A state can retain information that its jointly trained head fails to use; conversely, a good head cannot recover information discarded by the encoder.
4. On the new train/validation sources, audit paired precision and cadence before making a memory-length or denoising claim. Keep the declared sealed-test sources for the later frozen comparison.
5. Add wider spatial context, radius/history sweeps, explicit full-history sufficiency tests and calibration/kinetic tests only as separate controlled studies. They are not completed findings in this report.

All local test results here remain exploratory: the same sources have been revisited during method development. Adjacent anchors are correlated; source bootstraps resample sources, not windows. Older intervals use 2,000 draws after within-source seed averaging; newer intervals use 500 draws separately per seed. Neither procedure supplies training-seed confidence from two or three fits. No multiple-comparison correction is claimed. H=48 results at 96 ps address an endpoint beyond the observation duration, but do not alone demonstrate state closure or a converged memory length.

## Audit and reproduction

The [evidence snapshot](technical/evidence.json) contains the captured values and source hashes. [Input checksums](technical/inputs.json) identify each consumed local artifact. [Metric definitions](tables/METRICS.md) and [implementation hashes](technical/metric-contract.json) accompany every summary table. Plot PNGs have matching PDF files in `plots/`.

To collect a **new dated snapshot** from the configured completed results, without retraining:

```bash
conda run -n pointnet python -m src.research.memory_report \
  --config configs/analysis/memory_research_report.json \
  --output output/predictive_memory/research-summary-NEW-DATE
```

The collector refuses to overwrite an existing snapshot. It regenerates evidence tables and figures; this dated interpretation is a manually reviewed narrative. Underlying scientific recipes and protocol records are in [causal-state experiments](../../../experiments/mace_causal_20260916/README.md) and [partial-observation experiments](../../../experiments/predictive_memory_20260917/README.md). No running training or simulation job was restarted to assemble this report.
