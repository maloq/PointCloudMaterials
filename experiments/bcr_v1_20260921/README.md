# Bottleneck-conditioned reconstruction (BCR-v1)

Does reconstruction conditioned on the actual invariant export preserve more
structural information than matched VICReg and ordinary node denoising?
The supplied [protocol](PROTOCOL.md) is the scientific proposal; numerical settings
remain hypotheses. Implementation and testing are separate from a completed study.

## Implemented protocol

MACE scalar/vector/l=2 node pathway, two interactions, correlation 2, six Bessel
functions and 5 A edge support, using the repository's native backbone. The clean
encoder returns a signed 128-vector from center scalars plus smooth weighted scalar
sum divided by a fixed training count. No batch normalization, terminal L2 norm,
per-patch rescaling, temperature, IDs, clean-node or equivariant bypass to decoder.
The readout/support differ from archived JEPA; archived weights are not matched
comparators. New VICReg uses this exact encoder and observation support.

Separate decoder: two blocks, 64x0e+32x1o+16x2e, 16 Gaussian radial channels,
2*d0 edge support, 16-channel log-noise embedding, code/noise FiLM in each block,
scalar-only biases and channel-wise equivariant gates. Permitted channel-wise
triangle/parity tensor products and parameter counts are saved in runtime manifests.
Polynomial solid harmonics avoid singular normalized directions at collisions.
Cutoff uses the declared C2 taper. Only noisy coordinates build decoder edges.

The primary objective is weighted epsilon MSE, equal per environment, with no TDA,
physical, future, SIGReg/VICReg or latent-equality auxiliary term. Center/padding
loss weights are zero. Primary model learns from clean codes and fresh artificial
noise. Fixed membership survives corruption. Padding is dynamic; radius overflow
fails preparation. General skew-cell periodic image enumeration is supported and
checked against explicit lattice enumeration.

Controls: learned constant code; frozen random code; separately fitted matched
VICReg code with verified dataset identity; ordinary node denoising with a noise
level gain and trained pooled scalar export; matched VICReg representation training
with clean/noisy snapshot views. All trainable arms start from identical encoder
initialization tensors when configured with the same seed. The dedicated sampler and
noise generators preserve identical anchor/noise streams across arms. The VICReg
reference uses standard invariance/variance/covariance weights 25/25/1. Its views
are artificial snapshot corruption, not temporal equality.

## Implementation tests versus research results

`configs/bcr/real_overfit.json` is ONLY a diagnostic fixture: eight patches at
radius 8 A, two frames of one full-precision 400 K Lee2003 MEAM trajectory. Four
centers per frame selected without outcomes. Parent is the prepared-liquid SHA256;
other velocity children must inherit that same root. No independent test population
is claimed. Native float32 rounding is bounded conservatively, not called recovered
precision. The training-calibrated d0 and n_ref are in its cache manifest.

FP32 eager is the only released execution mode. BF16/compile requests fail until
separate parity validation. G0 includes replay, gradient, exact-symmetry, padding,
PBC, boundary, loss normalization and matched-shuffle tests. A separate GPU overfit
uses two real patches with fixed noise for 200 updates; this is deliberately not
held-out evidence or a production training recipe.

Reproduce correctness/overfit:
```
python -m src.training_methods.bcr verify --config configs/bcr/real_overfit.json --device cuda
```
Results: `output/bcr/implementation-20260921/`. See the result report for completed
checks. No 10,000-update pilot or 50,000-update confirmation has been launched.

## Before a scientific pilot

Freeze independent-root train/development/test manifests and all exposure history.
The historically consulted crystallization split is developmental, not a fresh
final test. Use higher-precision existing observations where needed; the preparer
rejects a grid entirely below ten times the native rounding uncertainty bound.
Match radius support, noise grid and source exposure across all new controls.
The smoke dataset cannot be reused as a generalization benchmark.

Suggested pilot remains 10,000 updates, effective batch 256, AdamW 3e-4 to 3e-6,
5% warmup/cosine, clip 1; microbatching preserves environment means. Measure the
complete queue budget before submission. Save initial/1%/5%/10%/each10%/terminal
checkpoints. No automatic best-by-reconstruction choice: structural-retention and
robustness assessments must pass before reconstruction breaks ties.

Frozen structural probes, conditional/unconditional gains, matched swaps, source
uncertainty, cross-root retrieval and perturbation tools are implemented. Full
prototype/real-MD responsiveness experiments, G1–G3 claims, final checkpoint
selection and crystallization transfer remain experiments to run on a properly
frozen, sufficiently large independent-root population. Nothing in the tiny
correctness fixture establishes that BCR beats an existing encoder.
