# Fast geometric encoding with controlled rotation error

Implementation update: the protected geometric paths and enlarged snapshot
encoders are now implemented. Ordinary BF16 scalar GATr failed the real-data
rotation gate; compensated BF16 scalar products passed it. See the
[implementation/validation record](../../docs/shared_pretraining_geometry_fp32_2x.md).
The choices below are the original proposal, not claims about measured speed.

The [restart audit](../../output/shared_pretraining/restart-diagnosis-20260918/RESULTS.md)
shows useful information in both encoders, weak trained decoders, and BF16
rotation errors comparable to differences between current GATr states. These
are related but distinct problems. Precision-only changes should first be
compared at identical weights; changes to training objectives require new fits.

## First choice: split precision inside the current GATr

Keep the current two blocks, eight multivector channels and 128 scalar channels.
Use FP32 for operations whose components change under rotations, and BF16 for
operations on invariant scalar channels. Keep atom-level features through the
blocks, followed by the existing single exported state.

| Path | Precision | Reason |
| --- | --- | --- |
| Coordinates, point embedding, smooth supports | FP32 | Preserve geometric input resolution |
| Multivector linear maps, geometric products/joins, norms and residual accumulation | FP32 | Avoid independently rounding orientation-dependent components |
| Geometric query/key contractions and multivector attention values/aggregation | FP32 initially | Rotating components must combine accurately before reducing to invariants |
| Scalar-only linear maps and scalar MLPs | BF16 | Their mathematical inputs/outputs are invariant |
| Scalar-only attention, if separated from geometric attention | BF16 candidate | Retain fast dense kernels without quantizing geometric components |
| Invariant extraction, pooling, exported state, physical/TDA heads and projector | FP32 | Preserve the small differences which the frozen probes recover |
| VICReg statistics, task losses, optimizer/master weights | FP32 | Retain the existing stable arithmetic |

This requires splitting operations within EquiLinear and the attention path,
not simply enabling autocast around every GATr block. The pinned GATr
[geometric attention implementation](https://github.com/Qualcomm-AI-research/geometric-algebra-transformer/blob/6afc26f26b8fcf51136ae8c1d264a36e14b6e497/gatr/primitives/attention.py)
concatenates orientation-dependent multivector query/key components with scalar
components before one SDPA call. Leaving that complete call in BF16 preserves
the numerical failure mechanism. Initial measurements must therefore leave
joint geometric attention in FP32 and quantify the speed retained by scalar work.

This is a mixed-precision version of GATr, not a mathematically different model.
It preserves the intended transformation rules in exact arithmetic. Finite
precision still requires measured rotation and information-retention tolerances;
it does not guarantee bitwise invariance or the full measured 1.59x BF16 speedup.

## Second candidate: local geometry plus global invariant attention

If selective GATr remains too expensive, test an architectural hybrid:

```text
Tracked atom positions/species
  → short FP32 equivariant local stem (MACE or GATr)
  → FP32 invariant atom descriptors and invariant geometric edge features
  → BF16 scalar attention/MLPs for contextual reasoning
  → FP32 smooth pooling, z128 and task heads
```

The stem must construct informative local many-body invariants before the scalar
stage. Scalar attention receives these invariants and invariant distance/angle
features, not raw x/y/z components. Pairwise geometric features are calculated
in FP32. Directional alignment information can be included through contractions
before discarding vector/tensor channels. This is a new hybrid, not unchanged
GATr, and its expressivity must be checked with physical and TDA readouts.

For lower complexity, test smooth local attention and a small set of learned
context tokens (initially 32–64). Token assignment must depend on invariant
features, with smooth support; axis-aligned spatial bins would introduce an
orientation dependence. Local scalar attention followed by global token attention
can replace all-atom dense attention, but changes the receptive field and
information bottleneck. Treat it as a separate scientific ablation.

## Short comparison before another long queue

Use identical source splits, saved data, one seed and batch 1,024. First evaluate
the same checkpoints with (A) full FP32, (B) current BF16, (C) BF16 backbone with
FP32 interfaces, and (D) protected geometric GATr with BF16 scalar computation.
A/B/C have audit results; D remains to be implemented and measured. The hybrid
is a later candidate, not a prerequisite for diagnosing D.

Use at least the same seeded proper rotations plus identity/axis controls, and
cover each material, static/dynamic group and local environment size. Report
rotation error in z, q and physical/TDA outputs, both absolute and relative to
between-observation differences. A proposed pilot acceptance target is rotation
RMS below 1% of FP32 between-observation RMS and no material worsening of matched
frozen readouts (predeclare a 1% relative selection-error tolerance). These are
engineering acceptance criteria, not mathematical guarantees or test-set results.

Only then compare matched short training runs with physical/TDA anchors alone
versus anchors plus VICReg. Include train-centered/scaled diagnostic decoders,
conditional training-mean baselines, state/projector spectra and gradient balance.
The existing trained-head loss and a nonzero-variance check are insufficient.
Keep the original encoder checkpoint export and evaluate information directly.

## Runtime efficiency

The [execution optimization plan](../../docs/shared_pretraining_optimization_plan.md)
prioritizes MACE graph preparation, GPU microbatch sizing, atom-count bucketing,
compilation and attention-backend verification. These remain unmeasured candidates.
