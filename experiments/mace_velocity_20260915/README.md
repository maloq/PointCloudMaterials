# A local coordinate/velocity MACE encoder

Question: can an encoder retain detailed instantaneous structure and smooth
structural geometry while also describing the motion of the same local atom group?

The new representation has 256 structural channels, 32 time-even activity channels,
and 16 time-odd flow channels. It retains the existing smooth-inner MACE geometry
block and adds geometry-conditioned radial messages from relative velocities.
Subtracting the smoothly pooled group velocity removes uniform bulk motion.
Even/odd network constructions enforce the correct velocity-reversal behavior.
This is an experimental extension built here, not a claim of a new published
architecture. Symmetry design follows established geometric learning principles
([E(n)-equivariant networks](https://proceedings.mlr.press/v139/satorras21a.html)).

Structural targets are the current group's 16 bond-order/density/coordination
statistics and its 144-component instantaneous persistence image. Nine instantaneous
velocity observables supervise activity and directed motion. A fixed dual-physics
teacher retains detailed structural channels and limits excess short-time changes.
The whole MACE backbone is fine-tuned. No global time, future label, crystallization
progress, or forecast objective enters the model. Smoothness and information
retention are evaluated independently for structure and motion.

The matched coordinates-only ablation has the same backbone initialization,
structural heads, samples, source weights, optimizer order, and training budget.
It predicts even motion statistics from geometry; its signed motion prediction
is zero, as required for a velocity-reversal-equivariant coordinate-only model.
The test intervention removes/shuffles velocities to establish whether their
measured values matter, beyond geometrically predictable activity.

Recipe: [mace_velocity.json](../../configs/analysis/mace_velocity.json).
Methods: [metric definitions](../../docs/metrics/mace_velocity.md).
Implementation: `src/models/encoders/mace_velocity.py`, `src/research/mace_velocity/`.
The current run is `output/mace_velocity/all-velocity-20260915/`.

Reproduction (pointnet environment, from repository root):

```bash
python -m src.research.mace_velocity inventory
python -m src.research.mace_velocity prepare
python -m src.research.mace_velocity verify --device cuda:0
python -m src.research.mace_velocity teacher --device cuda:0
python -m src.research.mace_velocity train --variant coordinates_velocity --device cuda:0
python -m src.research.mace_velocity train --variant coordinates_only --device cuda:1
```

Preparation is resumable at verified per-source cache boundaries. The run keeps
its immutable discovery inputs under `technical/`. Training retains `best.pt`
and exact completed-epoch `last.pt`; use the same command plus `--resume` to
continue. Epoch/clock limits stop before the current allocation expires. A separately
saved partial checkpoint is diagnostic; exact resumption uses `last.pt`.

Current findings: both variants completed eight epochs, with validation selecting
epoch seven. Real-data symmetry checks and gradient replay passed. Measured
velocities improve the motion readouts, but structural temporal stability remains
close to the coordinates-only control. At 0.75 ps, RMS normalized structural
change is 0.4614 versus 0.4622 for the control and 0.4534 for the original teacher.
The complete representation changes more because its motion channels fluctuate.
See the [completed stability audit](../../output/mace_velocity/all-velocity-20260915/STABILITY.md)
for definitions, within-low-order results, and the paired-sampling limitations.
Improved liquid clustering has not been established.

The [smooth-manifold literature review and proposed experiments](LITERATURE_REVIEW_SMOOTH_MANIFOLD.md)
address the requested RMS jump near 0.10, locally low-dimensional temporal motion,
and preservation of instantaneous structural information. This is a follow-up
proposal; no new training was launched for the review.
