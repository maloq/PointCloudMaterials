# Proposed GATr continuation for temporal smoothness

Status: proposal only; no training or job submission. September 18, 2026.

## Why change the objective

The selected Al snapshot GATr, step 1216, has normalized RMS jump 0.680 at
0.75 ps, second-difference roughness 1.473, and training-reference effective
rank 6.28 in the exploratory audit. Repeat/batch-order effects are much smaller.
These observations motivate testing temporal regularization; they do not
establish that all observed fast variation is undesirable.

The current implementation already chooses spatial and same-atom temporal pairs
with equal probability for dynamic updates. VICReg acts on the 64D projector q,
while the audit measures raw encoder z128. The projector can suppress directions
that remain variable in z; smoother q alone would not meet this objective.
Checkpoint selection minimizes physical + 0.25 instantaneous-TDA error without
a temporal score. See `Objective.forward`, `selection_rows` and shared runtime
`evaluate` in the current source.

Start from the verified Al-only step-1216 full checkpoint, preserving its encoder,
heads, species/geometry policy and target normalization. The separate broad
full-TDA continuation stopped at step 256 on its learning-health gate; use the
audited parent for this isolated Al smoothness experiment.

## First experiment: keep snapshot inference

Train on matched same-atom triples at t−0.75, t and t+0.75 ps, encoding each
snapshot independently. No future frame enters the deployed snapshot encoder.
Use existing training-source trajectories/observations; derive missing triple
views from their actual recorded times and atom identities. Keep static/spatial
training and present physical/TDA supervision. Confirm prepared triples and
their ancestry before execution; register any new persistent collection. Do not
change the material mixture, topology target type or input architecture in this
comparison. Training pairs with other cadences cannot silently use the 0.75 ps
formula; initially restrict the extra temporal term to native Al MEAM's verified
uniform 0.75 ps grid.

For V0 equal to the parent z covariance trace on a frozen training-only reference,
define the two additional terms:

```
L_step = mean(||z[t+1] − z[t]||²) / (2 V0)
L_bend = mean(||z[t+1] − 2 z[t] + z[t−1]||²) / (6 V0)
L = existing_objective + lambda_step L_step + lambda_bend L_bend + L_spread
```

The first reduces movement amplitude; the second targets changes in direction
and speed while leaving constant-speed drift unpenalized. These are proposed
regularizers, not established improvements. Low coefficients and retained
physical/TDA reconstruction limit the pressure to flatten real transitions.
Use the global trace scale, not division by tiny individual-channel variances.

`L_spread` is an explicit safeguard on exported z, needed because projector
VICReg alone does not exclude shrinkage of z with compensation by the heads.
Calibrate variance/covariance constraints against the parent using training
observations, avoiding a requirement for 128 equal-variance independent
directions. Its precise coefficient should be calibrated on training gradients
and held fixed across the controlled continuations. Independently reject
apparent wins accompanied by large losses of native spread, effective rank or
decodable information. A fixed denominator alone does not prevent shrinkage.

Run a small matched comparison:

| Arm | Objective change |
|---|---|
| A | Existing objective, matched continuation control |
| B | Existing objective + direct-z step penalty + spread safeguard |
| C | B + three-frame bending penalty |

Every arm encodes the same triples and sees the same anchor/label schedule,
training draws, optimizer budget and seed, even when an extra term is disabled.
This avoids confounding the loss with additional observations or compute.
The frozen parent is also evaluated as a zero-update reference. B/C test the
regularization package; a fourth spread-only arm would isolate its components
if the initial package improves results.

Suggested initial budget: 2–3 Al epoch equivalents, fresh optimizer, encoder
peak LR 2e−5, heads 1e−4, 10% warmup and cosine decay. These are proposed pilot
settings, not measured optima. Calibrate the combined new gradient to roughly
5–10% of the existing encoder-loss gradient on training-only batches; ramp it
over warmup. A small coefficient sweep can use 0.5×, 1×, 2× that calibrated
strength if the first comparison shows a useful tradeoff. Retain head-statistic
calibration on training observations only and the established learning-health
checks. Full-batch penalties must retain the gradient-caching semantics.

## Selection and acceptance

Add a fixed sequence panel from the existing selection sources, distinct from
the ten test trajectories already plotted. Tune strength/checkpoint on selection
sources, then apply the frozen policy to test sources. These already explored
test trajectories remain exploratory; additional unused test sources can give
a stronger confirmation after freezing the proposal.

Choose candidates on the tradeoff between movement, bending and retained
information, with separate physical and TDA error limits. A useful initial
target is 15–25% less normalized jump with improved roughness and no more than
5% relative degradation in either physical or TDA error. These are proposed
acceptance tolerances, not forecasts of achieved improvement. Reject severe
spread/rank reductions; a starting gate is at least 80% of the parent values
on the same training/selection observations. Do not reuse the audit's 6.28 rank
as a universal target on a different sample.

Report both frozen-parent-scale and each-model-own-scale jumps, raw variation,
second differences, full lag curves, source intervals, and fresh frozen linear
readouts. The last checks that a head's adaptation has not hidden lost encoder
information. Check phase-conditioned motion and whether sharp structural changes
remain detectable without extra delay or excess transition width. Retain
unclassified-versus-liquid distinctions; PTM is evaluation context, not a
training label or selector for temporal pairs.

## If stronger smoothing is needed

Consider an explicit slow structural subspace plus a fast residual within the
native encoder, with physical/TDA heads using both. This changes the exported
representation contract and requires separate evaluation of each part. Another
separate protocol can give GATr causal history; compare it with a repeated-frame
control and measure transition lag. These are follow-ups if snapshot smoothing
cannot improve the tradeoff, not additions to the first controlled experiment.

Do not start by increasing temporal pair separation, broadening material coverage,
or simply multiplying every VICReg term. Large-lag agreement can erase meaningful
structural evolution, and changes elsewhere would obscure which mechanism
reduced jitter. Small perturbation/denoising losses are an additional option only
after a controlled sensitivity test identifies the nuisance variation to remove.

## Methodological basis

The slowness principle minimizes temporal variation under nontriviality
constraints; see [Wiskott and Sejnowski's SFA project and publication](https://www.ini.rub.de/PEOPLE/wiskott/Projects/LearningInvariances.html).
[VICReg](https://arxiv.org/abs/2105.04906) supplies a related variance/decorrelation
framework for preventing constant outputs. Neither reference establishes the
best loss coefficients or atomic-state tradeoff for this GATr model. Direct-z
bending, parent-calibrated safeguards and the proposed experiment are research
choices to test here.

See [temporal smoothness](../../docs/research_glossary.md#temporal-smoothness),
[representation collapse](../../docs/research_glossary.md#representation-collapse-and-decoder-saturation-in-shared-pretraining),
and [physical decoder](../../docs/research_glossary.md#physical-decoder).
