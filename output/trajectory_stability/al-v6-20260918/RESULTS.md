# Both encoders show substantial frame-to-frame variation

**GATr has modestly smaller jumps than MACE, but neither traces a smooth path at
the saved 0.75 ps cadence.** On the native training-variance scale, TDA and bond
order have smaller jumps than either encoder. Their fine-scale paths still
backtrack strongly; lower amplitude is not the same as smooth motion.

[Interactive trajectories](explore.html) · [Figure gallery](index.html) ·
[Protocol and reproduction](README.md) · [Metric definitions](tables/METRICS.md)

We tested the latest selected completed Al checkpoints: MACE update **1465** and
GATr update **1216**, frozen at **2026-09-18 16:16 UTC**. Ten sources, two at each
of 400, 450, 500, 510 and 520 K, contribute four tracked atoms each and all 801
frames from 0 to 600 ps. This gives **32,040 matched observations**. The sources
are outside encoder training and selection ancestry. Five separate training
sources supply 420 normalization observations. No encoder was retrained.

| Representation | Normalized RMS jump at 0.75 ps | 95% source interval |
|---|---:|---:|
| MACE | 0.723 | 0.696–0.749 |
| GATr | 0.680 | 0.659–0.701 |
| TDA, nearest 80 atoms | 0.371 | 0.369–0.372 |
| SOAP, nominal 7 Å | 0.587 | 0.571–0.601 |
| Bond order | 0.278 | 0.262–0.293 |
| Radial descriptor | 0.661 | 0.644–0.678 |
| Angular descriptor | 0.513 | 0.506–0.520 |

The [normalized RMS jump](../../../docs/research_glossary.md#normalized-rms-jump)
expresses movement relative to the typical distance between independent
observations from that method's training reference. Values are dimensionless;
they are not percentages of an embedding's vector norm. Intervals resample whole
sources within temperature and hold the training reference fixed.

MACE's jump is **1.063× GATr's** (paired 95% interval **1.053–1.072**). Equivalently,
GATr is about **5.9% lower** on this measure. The difference persists with
coordinate standardization: MACE **0.722**, GATr **0.688**. This compares the two
specific checkpoints, not architecture performance over training seeds.

## What the trajectory shape says

MACE's 0.75 ps jump is already **92.4%** of its 12 ps jump; GATr's is **90.0%**.
Second-difference roughness is **1.480** for MACE and **1.473** for GATr, near the
**1.5** independent-frame reference. TDA is **1.486**, SOAP **1.474**, and bond
order **1.446**. These values indicate strong fast backtracking on top of slower
structural evolution. They do not establish that the complete states at nearby
times are independent.

The unsmoothed examples show changes around the crystallization episodes and
fluctuations within the preceding and following segments. That behavior is
consistent with substantial thermal/local-structure sensitivity at the saved
cadence. It does not isolate thermal motion from coordinate quantization.

![Matched comparison](plots/comparison.png)

## TDA depends on the feature scale

[Instantaneous TDA](../../../docs/research_glossary.md#instantaneous-topology)
summarizes the current neighborhood's persistent connected components, loops and
cavities. Its smaller native jump is present in each block separately:
H0 **0.404**, H1 **0.297**, H2 **0.270**, each using its own training variance.
H0 contributes about **67.4%** of total native TDA variance.

However, separately standardizing all retained coordinates raises the TDA
score to **3.904**. Two tail channels alone contribute **60.5%** of the squared
standardized score; their training variances are only **2.61e−13** and
**1.48e−14**. This sensitivity check is poorly conditioned for those nearly
absent features in the 420-row reference. It prevents a scale-independent claim
that TDA is intrinsically more stable. The headline table explicitly uses
native covariance-trace normalization.

The training-reference effective ranks are MACE **1.87**, GATr **6.28**,
TDA **1.20** and SOAP **2.76**. This participation-ratio measure describes how
concentrated the feature variation is; it does not establish information content
or representational collapse. In particular, a mostly one-directional phase
contrast can make small short-time jumps look favorable on the global scale.

## Numerical checks and limits

- Six scientific-control tests passed: linear drift, tiny alternating motion,
  independent noise, basis/scale invariance, small variance on a large mean,
  and source-stratified aggregation.
- All encoder input tensors exactly matched the existing native producer on
  an actual dynamic training observation and an actual static observation.
- Repeated MACE inference reached at most **1.07e−6** absolute feature difference;
  GATr repeated exactly in the checked batches. Reversing batch order reached
  **1.19e−6** for MACE and **7.15e−7** for GATr.
- Worst normalized RMS batch-order differences were **5.06e−5** and
  **1.36e−4**, respectively, far below the observed trajectory jumps. Those
  checked execution effects do not explain the observed variability.

The saved cadence cannot reveal sub-0.75-ps jitter. Inputs are the existing
float16 full-box coordinates; there is no matched full-precision control here.
Descriptors have different support sizes and scales. Ten previously explored
test sources and one trained seed per encoder make this an exploratory audit.
Smaller jumps are not evidence of better prediction or greater physical-state
sufficiency.

Detailed controls and block calculations are in
[`technical/supplement.json`](technical/supplement.json), reproduced by
[`technical/supplement.py`](technical/supplement.py). The initial static-export
metric snapshot is preserved under `technical/initial-static-export/`; the
final snapshot additionally describes the interactive display. No metric
values changed during that display update.
