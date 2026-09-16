# Smooth local-state embeddings and low-dimensional temporal motion

Literature review and proposed experiments, 15 September 2026. This is a research
proposal, not a new trained model or a measured improvement. The objective is a
representation of a tracked local atom group, optionally using a short observed
history. Global process progress and forecasting are outside this objective.

## What the present result actually requires us to change

The [completed stability audit](../../output/mace_velocity/all-velocity-20260915/STABILITY.md)
uses 360 same-center pairs from 30 independently melted test sources, separated
by **0.75 ps**. The original smooth-inner representation has RMS normalized jump
0.4534; the new velocity model's structural block has 0.4614. Activity and directed
motion have approximately 0.97, and the full concatenation has 0.5144. These are
finite-lag changes, not measurements of mathematical discontinuities.

The requested 0.10 is interpreted here as **RMS normalized jump at the same
0.75 ps lag**, using the same definition:

\[
R_\tau=\sqrt{\mathbb E\|z_{t+\tau}-z_t\|^2/
                 (2\operatorname{tr}\operatorname{Cov}_{\rm train}(z))}.
\]

Moving from 0.45 to 0.10 means about **20 times less normalized mean squared
motion**. It is substantially more demanding than a small improvement in the
current loss. A uniform rescaling of embeddings does not change this ratio.
Changing feature weights or increasing liquid/crystal separation can change it,
however, without improving resolution inside the liquid.

Indeed, the current structural jump is **0.7420 within the low-order subset**,
using a reference population restricted to low-order environments. The subset
uses group mean qbar6 < 0.30 at both endpoints; it is a liquid-like proxy, not a
PTM phase classification. Both mixed-population and within-liquid results matter.

The current [training loss](../../src/research/mace_velocity/train.py) minimizes
`relu(structural_change_squared - teacher_change_squared)`, weighted by 0.05 and
restricted to lags at most 0.8 ps. Once the student changes no more than the
teacher, this term supplies no further incentive to reduce change. A separate
0.2-weight loss retains the teacher's raw structural features. There is no
curvature loss, local motion-rank constraint, or history encoder. The structure
block also remains a function of coordinates alone. Thus, this run tested adding
motion information while limiting damage to existing stability; it did not train
for 0.10 or for a smooth temporal manifold.

The [previous frozen-feature comparison](../mace_local_state_20260915/RESULTS.md)
already rules out presenting simple compression or averaging as a free solution:

- The learned physical map reduced normalized squared temporal change by 69.83%
  relative to inner PCA16, but instantaneous-TDA error increased 2.554-fold.
- Two-frame averaging reduced squared change by about 73%, but increased the
  PCA control's instantaneous-TDA error by 71.72%.
- The tested linear temporal canonical coordinates did not consistently improve
  the desired balance. Adding more whitened coordinates sometimes made it worse.

Those older changes use a different cohort and normalization and are **squared**
quantities. Their values cannot be compared directly with the current RMS 0.45.

## What the primary literature supports

The sources below address parts of the problem. None establishes that our local
Al representation can reach 0.10 while retaining all present physical information.

| Work | Published evidence | Implication and limitation for this encoder |
| --- | --- | --- |
| Wiskott & Sejnowski, **Slow Feature Analysis**, Neural Computation, 2002 ([author paper page](https://www.ini.rub.de/PEOPLE/wiskott/Abstracts/WisSej2002.html); [author's mathematical notes](https://www.ini.rub.de/PEOPLE/wiskott/Teaching/UnsupervisedLearning/PublicWeb/SlowFeatureAnalysisApplications-L0-LectureNotes-PublicWeb.pdf)) | Learns slowly changing functions with variance and decorrelation constraints that exclude constant outputs. | Directly train temporal slowness with a controlled representation scale. Slowness alone does not ensure preservation of local physics or prevent encoding source identity. |
| Wehmeyer & Noé, **Time-lagged autoencoders**, JCP, 2018 ([paper](https://arxiv.org/abs/1710.11239)) | Uses lagged reconstruction to learn nonlinear slow molecular coordinates. | A useful nonlinear baseline; reconstructing a later observation is a training objective, not inherently a guarantee of an informative current local state. |
| Chen, Sidky & Ferguson, **Capabilities and Limitations of Time-lagged Autoencoders**, JCP, 2019 ([paper](https://arxiv.org/abs/1906.00325)) | Shows that nonlinear time-lagged reconstruction can mix high variance with slowness and fail to recover slow modes. | Avoid making ordinary lagged reconstruction the sole objective, particularly after our negative frozen temporal-map result. |
| Chen, Sidky & Ferguson, **State-Free Reversible VAMPnets**, JCP, 2019 ([paper](https://arxiv.org/abs/1902.03336)) | Learns continuous nonlinear slow molecular coordinates without requiring a discrete state assignment. | Supports continuous state coordinates before clustering. Reversible equilibrium assumptions cannot simply be transferred to our full nonstationary quench cohort. |
| Qiu, Jang, Huang & Yethiraj, **Unsupervised learning of structural relaxation in supercooled liquids from short-term fluctuations**, PNAS, 2025 ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12012455/)) | Short-lag canonical/autoencoder learning identifies structural variation related to relaxation in the Kob–Andersen glass former; medium-range descriptors help. | Particularly relevant evidence for local liquid structure. It does not establish our required topology retention or jump value. Our earlier affine implementation already failed to give a satisfactory encoder. |
| Rifai et al., **Contractive Auto-Encoders**, ICML, 2011 ([paper](https://icml.cc/2011/papers/455_icmlpaper.pdf)) | Penalizes encoder sensitivity while retaining reconstruction; observes representations sensitive to fewer local directions and less sensitive to directions away from the data manifold. | Motivates selective suppression of nuisance fluctuations and measurement of local tangent directions. Evidence is from machine-learning benchmarks, not liquid-metal trajectories. |
| Gropp, Atzmon & Lipman, **Isometric Autoencoders**, NeurIPS, 2020 ([paper](https://arxiv.org/abs/2006.09289)) | Regularizes local distance distortion and the relation between encoder and decoder. | A compact bottleneck should have controlled geometry. Preserving distances of every noisy raw input would conflict with denoising; our physically relevant distance must be specified separately. |
| Wang, Wang, Evans & Tiwary, **From Latent Dynamics to Meaningful Representations**, JCTC, 2024 ([full author manuscript](https://arxiv.org/html/2209.00905v4); [publication](https://doi.org/10.1021/acs.jctc.4c00249)) | DynAE constrains latent transitions with a learned overdamped Langevin model and gives identification results under its assumptions. | A stochastic latent geometry is a relevant extension. Its overdamped/diagonal-diffusion assumptions are not established for our short-lag coordinate/velocity groups. Do not force a deterministic trajectory or claim universal identifiability. |
| Liu et al., **Memory Kernel Minimization Based Neural Networks**, Nature Computational Science, 2025 ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12286716/)) | MEMnets learns slow molecular collective variables while accounting for memory; demonstrations include protein conformational changes. | Warns against assuming a reduced local state is memoryless. Motivates testing bounded observed history and multiple lags, rather than assuming a closed latent evolution equation. |
| Sartore, Nagel, Diez & Stock, **Lost in Projection? Gaussian Filtering Recovers Hidden Conformational States**, JPCL, 2026 ([authors' abstract](https://pubmed.ncbi.nlm.nih.gov/41925212/); [publication](https://doi.org/10.1021/acs.jpclett.6c00341)) | Reports that filtering high-dimensional feature trajectories recovers obscured states in a toy model and HP35 folding. | Strong motivation for a filtering control before state discovery. This review could access the abstract and indexed excerpts, not the complete article; it does not establish a causal filter or retention of instantaneous liquid topology. |
| Hsu et al., **Score-based denoising for atomic structure identification**, npj Computational Materials, 2024 ([paper](https://www.nature.com/articles/s41524-024-01337-z); [author manuscript](https://arxiv.org/abs/2212.02421)) | Equivariant denoising trained on noisy ideal crystals improves atomic structure identification at high thermal noise. | Relevant atomistic control, but an unsuitable unquestioned teacher for liquid motifs: the paper reports changes to melt radial structure and leaves their detailed interpretation open. |

The synthesis below is an engineering proposal for this repository. The combined
architecture and its numerical targets have not been validated by those papers.

## Separate the properties we want

**Slowness** means small changes over a specified physical lag. **Input stability**
means small changes when positions, velocities, ordering, or boundary membership
are perturbed slightly. **Temporal smoothness** additionally concerns irregular
changes of direction or speed. **Low-dimensional motion** means that nearby local
states evolve mainly along a small number of locally shared directions.

These are different properties. A low-dimensional vector can jump wildly. A
constant vector can be perfectly slow and contain no information. Motion on a
curved surface can require many global PCA directions even though each small
neighborhood needs only a few. Conversely, every sufficiently regular individual
trajectory can be described as a curve; fitting one curve per atom track is not
evidence that different groups share a useful state manifold.

We should distinguish the dimension needed to **describe a state** from the number
of directions needed to **describe short-time changes near that state**. Requiring
the entire representation to have four coordinates is stronger than the user's
requirement on motion and may repeat the earlier loss of instantaneous topology.

There is also a physical limit. With the same stationary reference population at
both endpoints, the metric satisfies
`R_tau^2 = 1 - trace(C_tau)/trace(C_0)`. Under those assumptions, 0.10 corresponds
to variance-weighted correlation 0.99 at 0.75 ps. The existing mixed train/test
normalization does not meet those assumptions exactly, so this is an explanation
of the target's strength, not an estimate of the liquid's correlation time.

A geometry constraint also does not make stochastic motion differentiable. Our
operational target should be reduced unwanted fluctuations at measured cadences,
with genuine rearrangements and uncertainty retained.

## Recommended design

### 1. Learn a local state from a bounded local history

Keep the complete message-passing halo, smooth inner pooling, and tracked center.
Apply a shared coordinate/relative-velocity encoder to each observed frame of the
same local group. Compare a snapshot model with a small causal temporal module
that receives a fixed physical-duration history. It ends at the current frame;
it receives no future frames, global timestamp, trajectory identity, whole-system
progress, or whole-system pooling.

Start by comparing zero history with approximately 0.1, 0.3 and 0.75 ps where
recorded cadences support them. These are proposed sweep values, not known Al
timescales. Choose a duration from validation information/smoothness curves and
measured transition delay. Do not substitute interpolated frames for observations.
Neighbor sets are rebuilt around the tracked center with the existing smooth
support; membership changes remain part of the audit.

Velocities may help the module distinguish oscillatory motion from persistent
rearrangement. This is a hypothesis to isolate against coordinates-only history,
velocity shuffling, and a matched simple filtering control. The current encoder
cannot test it because velocities do not enter its structural state block.

### 2. Give the state controlled geometry and constrain local motion

Compare state capacities of 8, 16 and 32 coordinates, with a larger-state control
if topology retention requires it. Treat 16–32 as a reasonable starting range,
not an assertion of intrinsic dimension. If a larger embedding is needed for
compatibility, export `z = g(c)` through a smooth learned map. A regular,
full-rank chart has local image dimension at most `dim(c)`; a bottleneck alone
does not establish meaningful distances, absence of self-intersections, or
temporally smooth coordinates.

Separately compare 4- and 8-direction descriptions of **local temporal motion**.
For example, learn an orthonormal local basis `B(z)` and penalize the component
`(I - B B^T) Delta z` outside it. Fit and assess this basis using collections of
similar physical states from different tracked groups and sources. A basis fitted
to one observed pair can trivially contain that pair's increment and is not a
valid test. Local bases may rotate along a curved manifold; global low-rank PCA
of all increments is not the intended constraint.

Control chart distortion and assess the singular values of its Jacobian. Avoid
solutions that conceal substantial physical variation in tiny coordinates and
recover it with an extremely sensitive readout. Contraction should target tested
nuisance perturbations, not indiscriminately erase real structural changes.

### 3. Optimize the requested behavior directly, under information constraints

Replace the teacher-relative excess-change term with a direct temporal objective.
Use physical-lag bins, source balancing, variance/decorrelation control, and
within-condition training comparisons. Do not pool 0.03 ps and 0.75 ps pairs as
equivalent positives. Evaluate the specified 0.75 ps metric separately from any
derivative-scaled training terms.

A proposed objective combines current physical readout error, normalized temporal
increments, weak temporal curvature regularization, local motion-rank error, and
geometry/variance control. For uneven frame spacing, a second-difference term must
use changes of slopes `(z_next-z)/dt_next - (z-z_prev)/dt_prev`, with its physical
time normalization recorded. Curvature regularization should reduce erratic
changes, not force all local rearrangements onto straight lines or suppress every
physical acceleration. A measured transition should remain visible.

Current bond order, density, coordination, and instantaneous TDA must be readable
from the **main state alone**. Use these losses while also evaluating additional
physical observables not used for training. Raw-teacher feature matching should
be an ablation, not a mandatory equality: retaining the teacher's irrelevant
fluctuations would oppose the requested change.

Separate activity/flow outputs can retain fast motion for downstream use. They
must be reported as additional fast information, with their own stability scores.
Obtaining accurate topology only from a fast residual branch would **not** satisfy
the requested informative smooth main embedding.

For a coordinate-only state, a velocity Jacobian-vector product can constrain its
instantaneous directional derivative. For a state depending on positions and
velocities, its derivative also depends on accelerations. For a history model,
it depends on the history mapping. Therefore the first implementation should use
measured finite differences; do not silently apply `dc/dt = J_position v` to the
full phase-space/history encoder.

### 4. Measure the tradeoff instead of hiding it

For physical targets `y`, readout `h`, errors `epsilon = y - h(z)`, and a readout
with Lipschitz constant `L`, the triangle inequality gives

`||Delta y|| <= ||epsilon_next|| + L ||Delta z|| + ||epsilon_current||`.

Thus, highly variable targets cannot all be recovered accurately from an
arbitrarily stationary representation with a well-conditioned readout. The
experiment must find how much of the present 0.45 is removable nuisance variation
and how much belongs to the requested instantaneous information. If 0.10 requires
unacceptable topology loss, report that infeasibility at the chosen lag and
retention budget rather than quietly redefining the target.

## Staged experiments and acceptance criteria

First run cheap frozen-backbone heads, then fine-tune only the candidates that
improve the validation tradeoff. This tests the design before expensive joint
training. Exact recipes belong in `configs/` when implemented; this review does
not create speculative runnable commands or launch jobs.

| Stage | Controlled comparison | Question |
| --- | --- | --- |
| A | Current model; compact physics-only head; same head with direct slowness and variance control | Can we remove excess variation without a history input, and is the old loss the main limitation? |
| B | A plus local motion-rank and weak curvature terms, individually and together | Do small steps also lie in stable, low-dimensional local directions? |
| C | Best snapshot design versus bounded coordinates-only history, coordinate/velocity history, and matched simple filtering | Does velocity-informed state estimation improve the smoothness/information balance beyond smoothing alone? |
| D | Fine-tune MACE for selected designs with and without raw-feature teacher retention | Was useful stable information inaccessible to a frozen head? |
| E | Optional memory-aware/stochastic latent objective after dense-trajectory diagnostics | Does a stronger dynamical prior help without imposing unsupported Markovian or overdamped assumptions? |

Select dimension, history, loss weights and checkpoint on validation sources.
The present 30 test sources have already informed the design; reuse must be
labelled a development comparison. A confirmatory claim needs independent
untouched sources. Existing dense shooting branches sharing training lineages
are useful for training and diagnostics, not an independent final test cohort.

The existing paired cache alone cannot measure temporal curvature. Obtain
consecutive tracked sequences with at least three observations, preferably much
longer, and sample multiple physical lags. New dense data must have a measured
storage-precision floor; interpolating coarse frames or casting stored values to
a wider type does not recover missing observations or precision.

Proposed success criteria, to freeze before model selection:

- **Jump size:** aim for RMS near 0.10 at 0.75 ps, report median, p95, maximum and
  source-bootstrap uncertainty. Use the same reference identities, weighting and
  normalization procedure for every model. Freeze each model's train-only scale
  for evaluation; report absolute increments and reference spread as well.
- **Liquid relevance:** separately report within-low-order and fixed-temperature
  comparisons, plus crystal and transition cohorts. Improvement based only on
  between-phase separation does not count as resolving liquid structure.
- **Information:** compare main-state-only bond-order and TDA errors at the same
  readout budget. An initial proposed retention allowance is at most 10% relative
  degradation in each physical target group versus the current model; also show
  the full tradeoff curve rather than treating that allowance as a scientific law.
- **Manifold motion:** measure out-of-sample local tangent residual and subspace
  stability using independent groups. A proposed diagnostic is whether 4–8 local
  directions explain at least 90% of short-lag displacement energy. Also measure
  state dimension and decoder conditioning; a small architecture is not itself
  empirical evidence of meaningful low-dimensional physics.
- **Transitions and history:** report detection delay, preserved transition
  amplitudes, and both overlapping and nonoverlapping history-window comparisons.
  Shared observations alone can inflate apparent temporal coherence.
- **Spatial meaning:** evaluate physical-neighbor agreement and spatial structure
  within comparable liquid conditions, including dependence on group overlap.
  Do not smooth spatial labels indiscriminately across physical interfaces.
- **Robustness:** repeat identical-input, rotation, ordering, tracked-center and
  membership-crossing tests; use several seeds for the selected design. Evaluate
  readout generalization and novel physical observables, not just training losses.

Clustering follows these tests. Continuous local-state coordinates and uncertain
assignments remain valid outcomes; forcing separated clusters would recreate the
problem this work is intended to solve.

The recommended first implementation is **joint physical retention and direct
slowness, followed by a local temporal-direction constraint and a bounded
coordinate/velocity history ablation**. It addresses the measured loss limitation,
tests the user's low-dimensional-motion requirement explicitly, and gives a clear
way to reject apparent smoothness obtained by losing local physics.
