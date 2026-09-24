# BCR follow-up: better denoising, weaker structural readouts

Analysis date: 22 September 2026. **The current BCR recipe should not be scaled
unchanged as a general structural encoder.** It provides a denoising advantage across evaluation roots for this fitted seed,
including with a fresh decoder, while making radial
and some multiscale structural measurements harder to recover. Stronger probes and
the archived relaxed-data audit preserve that conclusion. This is evidence about
the tested readouts, not proof that arbitrary nonlinear decoders cannot recover the
information.

All five stages completed in **83.1 minutes**: four checkpoint exports, 60
ridge/residual probe pairs (120 readouts), two conditioning-intervention banks and
three fresh-decoder fits of 10,000 updates each. This analysis fitted no new models.
The original failed G1 engineering gate remains unchanged.

![Four-panel summary](plots/overview.png)

Figure: A, tuning-selected residual radial probes at each checkpoint; B, one frozen
trained decoder with different codes at the strongest noise (separate unconditional
reference shown in gray); C, matched fresh-decoder fits; D, radial transfer changes.
Panels C/D show paired 95% root-bootstrap intervals. One trained seed throughout.
[Vector PDF](plots/overview.pdf).

## 1. The sample-specific contribution is small

At encoder/decoder update 10,000 and σ/d₀=0.12, on the same 384 anchors and corruptions:

| Decoder and supplied code | Noise MSE ↓ |
| --- | ---: |
| BCR decoder, correct environment code | 0.62834 |
| Same decoder, training mean code | 0.62944 |
| Same decoder, one training-fitted constant | 0.62939 |
| Same decoder, unrestricted wrong code | 0.63079 |
| Separately trained unconditional decoder | 0.65028 |

Correct code improves over the fitted constant by **0.166%** (original paired
root interval **0.128–0.217%**), versus **3.373%** over the separately trained
unconditional decoder. The constant retains **95.2% of the numerical performance
gap** against that unconditional reference. This last percentage is descriptive
arithmetic on matched evaluation populations, not a causal decomposition: the
unconditional decoder followed a different training trajectory.

The training mean alone works almost as well as the optimized constant. Its
true-code advantage is 0.174%, so the result is not dependent on extensive constant
optimization. The constant was selected at update 700 using training-root tuning
observations; no development observation fitted it.

Strict matched swaps give 0.311% true-code advantage at 89.84% coverage. Unrestricted
swaps give 0.388% at full coverage. Strict-swap error uses a different subset and
must not be subtracted from full-population values. The amplitude intervention is
consistent with a small useful correction: true amplitude 1 has error 0.62834,
half amplitude 0.62855, zero amplitude 0.62944 and double amplitude 0.62951. Among
these specified amplitudes, 1 works best at this noise level, but its advantage is
small. These are diagnostics, not permission to tune amplitude on development data.

At the earlier 1,000-update checkpoint, the fitted constant actually performs
slightly better than the correct code at 0.12 (0.66726 versus 0.66778; 0.079% lower
error). The true-code gain interval is negative, −0.128% to −0.032%. At 0.08 the
constant/true difference is unresolved. The small sample-specific advantage in the
original decoder therefore appears later in training; the fresh-decoder experiment
below assesses a different, fully retrained decoder.

**Interpretation:** the code is used, but detailed environment-specific variation
adds much less than the model-to-model comparison alone suggests. A global decoder
adjustment or code-distribution effect remains plausible.

## 2. Stronger probes do not resolve the structural deterioration

These are standardized RMSEs from the tuning-selected **ridge + zero-initialized
residual MLP**, with ridge itself eligible as step zero. Normalization and fitting
use ten training roots; two other training roots choose ridge penalty and residual
duration. Evaluation uses six separate development roots, 64 anchors each.
Positive changes below mean deterioration.

| Features | Target | Initial | 10,000 updates | RMSE change | 95% root interval |
| --- | --- | --- | --- | --- | --- |
| exported | angular | 0.9698 | 0.9656 | -0.44% | [-1.66%, +0.70%] |
| exported | radial | 0.3327 | 0.4376 | +31.53% | [+25.31%, +36.61%] |
| exported | rich | 0.7021 | 0.7163 | +2.02% | [+1.43%, +2.72%] |
| pooled | angular | 0.9596 | 0.9567 | -0.31% | [-0.92%, +0.24%] |
| pooled | radial | 0.3261 | 0.3626 | +11.21% | [+7.97%, +14.21%] |
| pooled | rich | 0.6977 | 0.7005 | +0.39% | [-0.18%, +1.00%] |

Exported radial error rises **31.5%**, with all six roots worsening. Pooled radial
error rises **11.2%**, also on all six roots. At the final checkpoint the exported
radial error is 20.7% worse than the pooled error. The final readout therefore
amplifies the problem, but the underlying pooled features have changed adversely
as well. Replacing the final readout with pooled features would only partially
address it, and is not uniformly better for every target: final nearest-distance
error is lower from the export than from pooled features.

The richer target's exported RMSE worsens 2.02%, again on every root. Approximately
**77% of its mean-squared-error increase comes from the l=0 block**, whose RMSE
rises 0.2488→0.3519. Higher angular degrees change much less. The small improvement
in the four-target angular average is not established across melt roots: its
interval includes zero. Moreover, exported q4 and q6 individually worsen while w4
and w6 improve; the family mean hides that distinction.

The radial loss is not confined to nearly empty bins: exported weighted-count
error rises 0.1984→0.3683 (+85.6%), weighted-radius error 0.3218→0.5498 (+70.8%),
and nearest-distance error 0.2522→0.3103 (+23.0%). Count and density are constant
multiples and are not independent evidence. Small relative changes in tiny radial
bins should be interpreted with the 1e-6 normalization floor in mind.

Across all 60 readout pairs, 13 select the unmodified ridge solution, only one
selects the 5,000-update limit, and median tuning MSE improvement is 0.191%.
The stronger protocol removes the original undertrained-500-update-MLP concern;
it does not establish an upper bound on every possible nonlinear readout.

## 3. A fresh decoder still benefits from the trained encoder

Every decoder here starts from the same initial parameter tensors and receives
10,000 updates with matched sample/noise streams and optimizer schedule. Encoders
remain frozen. The initial encoder's decoder has noise MSE 0.69984 at 0.08 and
0.64675 at 0.12.

| Frozen encoder update | σ/d₀ | Fresh decoder noise MSE | Gain over initial encoder | 95% root interval |
| --- | --- | --- | --- | --- |
| 1,000 | 0.08 | 0.69019 | 1.38% | [1.21%, 1.55%] |
| 1,000 | 0.12 | 0.62874 | 2.78% | [2.61%, 2.99%] |
| 10,000 | 0.08 | 0.68871 | 1.59% | [1.41%, 1.77%] |
| 10,000 | 0.12 | 0.62689 | 3.07% | [2.92%, 3.25%] |

Thus the advantage is **not specific to the original co-trained decoder**. At 0.12,
all six roots improve with the final encoder. However, 1,000 encoder updates
already provide a 2.78% advantage versus 3.07% at 10,000. Continuing encoder training
from 1,000 to 10,000 provides only **0.295% further noise-MSE reduction** for the
fresh decoder (95% interval 0.246–0.348%), while radial structural readout deteriorates
substantially. The early radial improvement versus initialization itself is not
established across roots; do not declare an optimal checkpoint from these data.

Fresh-decoder transfer rules out an explanation limited to one coadapted decoder.
It does **not** yet separate detailed code content from a shared code calibration
or optimization effect. The cheap remaining diagnostic is to repeat the constant-
code interventions on these fresh decoders. No such result is claimed here, and no
input-matched VICReg checkpoint was available.

## 4. Relaxed data confirms transfer problems and identifies a useful target

The audit uses **2,880 paired neighborhoods from 45 roots at 400–520 K**: 25 fitting,
five tuning and 15 development roots. The 960 development examples use four frames
and 16 tracked centers per root. Full 8 Å neighborhoods were extracted independently
from existing observed and relaxed full cells; all support, identities and archive
provenance were checked. These are frozen melt-pretrained encoders with newly fitted
structural probes, **not encoders trained on relaxed data**.

The following table uses exported codes and the same stronger readout protocol.
Intervals resample whole roots within each temperature (three roots per temperature).

| Input → target | Target | Initial | 10,000 updates | RMSE change | 95% root interval |
| --- | --- | --- | --- | --- | --- |
| observed → observed | angular | 0.9571 | 0.9497 | -0.77% | [-1.27%, -0.26%] |
| observed → observed | radial | 0.2743 | 0.3311 | +20.73% | [+18.06%, +23.52%] |
| observed → observed | rich | 0.6007 | 0.6123 | +1.92% | [+1.52%, +2.31%] |
| observed → relaxed | angular | 0.9105 | 0.9122 | +0.19% | [-0.35%, +0.73%] |
| observed → relaxed | radial | 0.7273 | 0.7296 | +0.32% | [-0.38%, +1.03%] |
| observed → relaxed | rich | 0.7266 | 0.7327 | +0.84% | [+0.47%, +1.23%] |
| relaxed → relaxed | angular | 0.8533 | 0.8631 | +1.14% | [+0.45%, +1.91%] |
| relaxed → relaxed | radial | 0.2273 | 0.2637 | +16.02% | [+13.28%, +18.57%] |
| relaxed → relaxed | rich | 0.5336 | 0.5456 | +2.26% | [+1.93%, +2.58%] |

Observed→observed radial deterioration occurs on **all 15 development roots** and
within every temperature group (+17.2% to +24.7%). Relaxed→relaxed radial deterioration
also occurs on **all 15 roots**, within every temperature (+13.8% to +19.9%). This
is not explained solely by changing the mixture of temperatures.

Pooled features reduce these radial losses to +5.52% for observed→observed and
+4.65% for relaxed→relaxed. Their rich-target scores are approximately unchanged.
There is one modest favorable result: pooled observed→relaxed radial RMSE improves
0.7250→0.7180 (**0.97% lower**, interval 0.25–1.70% lower), on 11/15 roots. The exported
code does not retain a clear improvement on that task. This is exploratory evidence
among many correlated comparisons, not a standalone proof of a better encoder.

To isolate the value of relaxed **input**, hold relaxed **targets** fixed. For the
final exported code, observed→relaxed radial RMSE is 0.7296 while relaxed→relaxed
is 0.2637 (**63.9% lower**); rich-target error is 25.5% lower. The advantage is also
present before BCR training (68.7% and 26.6%). Relaxed geometry therefore exposes
useful structural information to these readouts, but the advantage does not come
from the tested BCR training. Predicting relaxed structure from an observed local
patch is a materially harder task than reading it from the relaxed patch itself;
full-cell relaxation also uses context outside that local patch.

## 5. Limits that affect the interpretation

- One encoder-training seed; intervals measure evaluation-root uncertainty only.
  Small effects are exploratory and intervals are not multiplicity-adjusted.
- All 384 melt examples **and all 960 transfer examples** satisfy the implemented
  q6<0.35 filter. Observed q6 over the entire transfer cache is only about 0.014–0.070.
  All/liquid rows are duplicate populations. This filter supplies no useful phase
  separation in this assay; these results do not establish crystallization or
  interface discrimination. A future phase-specific audit needs independent local
  structure labels or an appropriately validated order definition.
- The transfer inputs inherit archived full-cell float16 precision. They are
  appropriate to analyze the actual stored structural data, but they are separate
  from the high-precision melt corruption experiment. No precision ablation was run.
- This is a structural/denoising assay, with no forecasting or event-onset evaluation.
  Historical test and calibration roots were not used. Training a different encoder
  on relaxed data remains untested here.
- The historical follow-up metric prose misstated the noise denominator. The actual
  producer divides weighted squared unit-noise error by **3 × sum(weights)**,
  not the sampled noise energy. Numerical results are unchanged; the clarification
  is recorded in this analysis's [metric definitions](tables/METRICS.md).

## Recommended next experiment

Do not spend the next allocation on longer pure-BCR training or a larger exported
code. The 128D export already comes from a 64D pooled representation, and stronger
probes do not recover the missing radial performance.

First apply constant-code interventions to the completed fresh decoders. Then run
a small, explicitly distinct three-arm comparison on one fixed paired dataset:
**BCR**, **code-only geometric reconstruction**, and **their hybrid**. Match encoder,
readout, input support, roots, initialization and update budget. Use the available
relaxed collection; fix the geometric target domain before fitting rather than
changing both data and target definitions opportunistically between arms. Include
a direct pooled-feature baseline to quantify the readout contribution.

For any trained geometric target, reserve other structural measurements for
independent validation. Evaluate temperature/root strata and independently defined
local phases; preserve the original G1 result as a historical mechanism gate.
The selection criterion should require structural retention and useful denoising,
rather than reconstruction loss alone. This recommendation is a new study proposal;
no further training was launched during this analysis.

## Reproduction and artifacts

```bash
conda activate pointnet-torch214
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m src.research.bcr_followup_analysis --config configs/analysis/bcr_followup_20260922.json
```

- [Paired RMSE comparisons and uncertainty](tables/paired_comparisons.csv)
- [Fresh-decoder comparisons](tables/decoder_comparisons.csv)
- [Per-target and angular-degree changes](tables/target_changes.csv)
- [Metric definitions](tables/METRICS.md)
- [Original experiment and tables](../conditioning-audit-20260922/README.md)

Raw-prediction alignment, original reported metrics and completion counts were
verified; three focused tests validate bootstrap scaling, observation weighting,
reproducibility and temperature/root constraints. The frozen input fingerprints
are in `technical/inputs.json`, and implementation hashes in
`technical/metric-contract.json`. Historical result tables remain unchanged.
