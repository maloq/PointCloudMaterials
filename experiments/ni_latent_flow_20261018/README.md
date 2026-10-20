# Ni trajectory distributions: block generation, memory and temporal representations

This experiment asks whether the main failure of the previous local Ni simulator
comes from recursive generation, loss of useful information in its representation,
or insufficient use of observed history. The target is a useful distribution of
local structural and velocity trajectories, including persistent ordering within
liquid. Full-cell-labelled nucleus detection is a separate evaluation.

The frozen recipe is [flow_20261018.json](../../configs/ni_latent_sim/flow_20261018.json).
It declares **17 treatments × three seeds = 51 scientific fits**, using the full
existing native measurement cohort. The queue has two GPU workers and a 12-hour
deadline. That deadline bounds execution; it is not a promise that every declared
fit and readout will finish. This protocol records no launch or scientific result
by itself. Actual execution and incomplete work are reported by the receipts in
the [workflow documentation](../../docs/ni_latent_flow.md).

## Scientific population and time notation

Let i identify a forecast-origin snapshot, with actual physical time t_i, and j
the central atom whose identity remains fixed throughout its local sequence.
Let b identify a physical MD continuation, and let k be an integer observation
offset. With Δt=0.1 ps, the structural/velocity measurement is

\[
Y_{ibj}[k]=Y_j\bigl(X_{ib}(t_i+k\Delta t),V_{ib}(t_i+k\Delta t)\bigr).
\]

Here X and V denote positions and velocities of the physical MD state; Y_j is
the fixed local measurement procedure around atom j. The available native bank
provides one observed physical continuation per origin. Its overlapping windows
do not create additional independent continuations. A neural ensemble sample is
indexed by r, not counted as a new physical b.

The encoder/predictor observes k=-3,-2,-1,0 and predicts k=1,...,60. Four observed
frames span **0.3 ps**. The last target is **6 ps after the origin**. The array
has 64 observations, and its first-to-last span is 6.3 ps. The index i is an
identifier, not a time increment: different sources and central atoms may have
the same physical observation time, and origins may overlap.

The sealed bank contains 14,400 sequences from 60 Ni histories: 3,840 training
windows from 16 ancestors and 10,560 test windows from 44 ancestors. The
original, liquid-liquid and growing-front regimes each have 4,800 windows.
The same center is followed; local nearest-80 membership is reselected at each
observation. All available windows in this bank are retained. No million-row
structural-pretraining collection, new MD, relaxation or derivative teacher is
used in this campaign.

The central Y has 244 components: BO40, TDA113, density74 and velocity17. Regional
arms additionally observe C, an 87-component summary of all atoms within 20 Å
of the same center. The local/r20 new-encoder comparison always has 331 input
components: [Y,0] versus [Y,C]. Its decoder target remains the identical central
244 measurements. The fixed linear-VAMP/VAE and direct-measurement arms use 244
inputs; the inherited regional hybrid uses 331. Actual boxes determine periodic
images in the already prepared data, but are not features.

All prior training ancestry remains training; former selection/evaluation roles
are test with original roles retained as provenance. All transforms and phase
readouts fit on training ancestors only. Temperature, absolute time, simulation
age, species, PTM and birth labels are excluded from learned inputs. Fixed
relative frame positions organize attention; the noise-to-data coordinate used
by flow matching is an algorithmic variable, not physical MD time.

## Implemented matrix

Every row below is repeated with seeds 20261025, 20261026 and 20261027. “Block”
means one jointly generated 60-frame future; “MDN” means recursively sampled
one-step eight-component diagonal Gaussian mixtures.

| Treatment | Representation/input | Predictor change | Decoded auxiliary weight |
| --- | --- | --- | --- |
| `linear_block` | Frozen linear VAMP, local244 →32 | Block flow | 0.25 |
| `vae_block` | Frozen VAE mean, local244 →32 | Block flow | 0.25 |
| `linear_mdn` | Same frozen linear VAMP | Four-frame MDN | 0 |
| `vae_mdn` | Same frozen VAE | Four-frame MDN | 0 |
| `linear_block_noaux` | Same frozen linear VAMP | Block flow without decoded auxiliary | 0 |
| `direct_block` | Identity measurement state244 | Block flow without latent compression | 0.25 |
| `linear_current` | Same frozen linear VAMP | Current-state-only MDN | 0 |
| `linear_gru` | Same frozen linear VAMP | Current state plus GRU of three increments | 0 |
| `linear_filters` | Same frozen linear VAMP | Current state plus three finite-history filters | 0 |
| `linear_mdn_noise` | Same frozen linear VAMP | Four-frame MDN, context noise SD0.05 | 0 |
| `linear_block_noise` | Same frozen linear VAMP | Block flow, context noise SD0.05 | 0.25 |
| `selftica_local` | New SelfTICA, local244+zero87 →32 | Same block flow | 0.25 |
| `fmrc_local` | New bidirectional FMRC, local244+zero87 →32 | Same block flow | 0.25 |
| `selftica_r20` | New SelfTICA, central244+regional87 →32 | Same block flow | 0.25 |
| `fmrc_r20` | New bidirectional FMRC, central244+regional87 →32 | Same block flow | 0.25 |
| `hybrid_r20_block` | Frozen regional hybrid →32 | Block flow | 0.25 |
| `hybrid_r20_mdn` | Same frozen regional hybrid | Four-frame MDN | 0 |

The primary contrasts separately test predictor family, compression, auxiliary
supervision, current versus finite-history information, context corruption,
representation objective and spatial support. The direct arm changes both state
dimension and decoder, so it is an information-loss control, not a parameter-
matched architecture comparison. Matching update budgets does not match FLOPs
or wall time. MDN fitting uses all 60 observed transitions of each selected
sequence; block fitting uses the complete corresponding path.

## Published mathematical cores and Ni adaptations

### Whole-block flow: LaM-SLidE core

The implemented data-prediction interpolant and sampling equations follow
[LaM-SLidE, Appendix C/E.3](https://arxiv.org/html/2502.12128v5) and its
[released implementation](https://github.com/ml-jku/LaM-SLidE/tree/3f7d7fa8f26ba6c4fa5ec4f9c95c85a743ff1ee3).
For a standardized observed/target sequence Z and independent standard Gaussian
noise E, the interpolated state at algorithmic flow coordinate τ is

\[
Z_\tau=\sin(\pi\tau/2)Z+\cos(\pi\tau/2)E.
\]

The network predicts clean Z from this noisy sequence and the supplied observed
history. The objective is clean-data MSE, averaged over all 64 frame positions.
Sampling starts from Gaussian noise and integrates the associated flow with
Euler steps between 0.001 and 0.999. **Ten grid points give nine network
evaluations**, not ten physical MD steps. The predictor uses four width128
attention layers, four heads, rotary relative positions and flow-conditioned
modulation. EMA decay is fixed at 0.999; the final EMA predictor is evaluated.

Our explicit adaptations are one local-state token per time, existing or newly
trained 32-dimensional measurement encoders, a 244-dimensional measurement
decoder, and training-only standardization. This is not the paper's full entity
encoder, atomistic geometry decoder or original molecular benchmark. The direct
arm has one 244-dimensional token instead. It is still local generation.

The auxiliary term decodes the clean prediction at the same noisy training
interpolant and compares it with actual measurements using the established
four-group weights. Decoder weights are frozen while gradients pass through
the decoder to the flow. The term does not compare an average of independent
stochastic trajectories to one target. `linear_block_noaux` isolates its effect.
No physical-coordinate reconstruction encoder pretraining is introduced.

### Temporal representations: SelfTICA and bidirectional FMRC

The SelfTICA arm implements the published contrastive objective with a shared
encoder and a bias-free linear predictor for each physical lag. If z_u and
z_u^+ are the encoded present/future pair for batch item u, A_ℓ is the learned
lag-specific linear predictor, and N is batch size, the loss is

\[
\mathcal L_\ell=
\frac{1}{N(N-1)}\sum_{u\ne v}\bigl[(A_\ell z_u)^\top z_v^+\bigr]^2
-\frac{2}{N}\sum_u(A_\ell z_u)^\top z_u^+.
\]

The implementation does not add target detachment, feature normalization,
spectral gradients or a reconstruction term to this representation objective.
Its Ni adaptation replaces the paper's molecular graph with the fixed 244/331
measurements and the common two-layer width256 SiLU encoder. [SelfTICA methods,
Eq.4](https://arxiv.org/html/2606.15495v3),
[authors' code](https://github.com/Kai-Zhu-2001/SelfTICA).

FMRC uses **both** forward and retrospective conditional flow-regression losses,
following [FMRC Eqs.20–21](https://arxiv.org/html/2408.17139v2) and the
[authors' training notebook](https://github.com/Mingyuan00/Flow_Matching_for_RC/blob/main/1-CLN025/FMRC_training.ipynb).
For each direction, a training-fitted diagonal Gaussian endpoint is paired
independently with the actual target; linear interpolation adds Gaussian path
noise of SD0.001. The conditioned head predicts the endpoint difference. Each
loss sums squared errors over 244 coordinates and averages over examples; the
two directional losses are added. Heads have three width256 ReLU layers and
are separate at lags 0.5/1 ps. The common encoder exports 32 coordinates.

Our FMRC targets remain central244 even with regional331 inputs; the original
optional TICA preprocessing is omitted. Its auxiliary flow heads train the
representation and are discarded as autonomous simulators: a fresh common
block predictor is subsequently fitted. Retrospective conditioning infers an
earlier observation from a later one. It does not reverse velocities or impose
detailed balance. The source-selected, partially observed NPT population does
not establish the equilibrium/reversible assumptions behind spectral claims.

### Memory and noise: controlled adaptations

[PFCG](https://pubs.acs.org/doi/10.1021/acs.jctc.5c02131) motivates separating
current information from observed increments. The implemented `linear_gru`
is a matched one-layer GRU/MDN ablation over three increments. It is **not** the
paper's staged, two-layer-GRU Gaussian model, whose current-state network is
trained first and then frozen. PFCG's five adjacent training terms use observed
windows, not a free five-step rollout; this distinction is visible in its
[source](https://gitlab.com/pak-group/pfcg_m01/-/raw/main/PFCG.py).

The filter arm uses decay factors 0.5,0.8,0.95 over the same three observed
increments, with finite-window weight normalization. It resets that window at
each prediction; it is not a persistent long-memory state. This is a cheap
matched control inspired by
[memory-conditioned diffusion](https://arxiv.org/html/2609.28371v1), not a
reproduction of that paper's diffusion sampler or its stationary-process theory.

The noise arms add independent SD0.05 Gaussian perturbations in internally
standardized history coordinates during training and stochastic inference.
One corrupted context is held fixed within a block sample. This tests the
robustness idea in [STAR-MD](https://arxiv.org/html/2602.02128v2), using our state
and generator; it does not reproduce STAR-MD's protein-specific SE(3) diffusion.
Deterministic controls use clean context. A zero-noise flow is labeled as such,
not called a conditional mean.

## Fixed fitting protocol

All predictor fits have 3,200 updates, batch/microbatch256, AdamW maximum LR0.001,
160-update warmup, cosine decay to 1% of maximum LR, weight decay1e-5 and gradient
norm limit10. BF16 matrix operations retain FP32 parameters, optimizer state and
loss reductions. This is neural arithmetic; no physical integration occurs.
Current-state and history MDNs optimize predictive mixture NLL; block flows
optimize clean-data prediction with the declared optional decoded auxiliary.

Each of the 12 new SelfTICA/FMRC fits first receives 12 representation epochs
over 3,840×51 eligible present/lag pairs, alternating 0.5/1 ps lag. At batch256
this is 9,180 updates. A fresh decoder then receives 12 epochs/9,180 updates on
the same eligible present observations with the encoder frozen. Both stages use
one-epoch warmup and the same maximum LR/cosine schedule. The 3,200-update
predictor follows. Inherited encoders and decoders are reused without these
stages. Checkpoints are the fixed final budgets, with no test-driven selection.

## Evaluation and interpretation

The [metric contract](../../docs/metrics/ni_latent_flow.md) defines connected-path,
increment and increment-correlation energy scores at 1/3/6 ps, endpoint errors,
coverage, temporal/ensemble power, raw ordering persistence, local phase entry
and complete validity accounting. All invalid paths remain counted; finite but
impossible basic measurement values are reported separately. Train-only phase
readouts and actual PTM labels remain evaluation tools, not training losses.

Birth prediction retains all 19,800 original assay rows, 19,652 unique inputs,
the original 32/64-atom nucleus definitions and 1/3/6 ps leads, without additional
neighborhood filtering. **Twenty-one unique encoder fits** receive readouts:
the 12 newly trained representations plus three seeds each of the fixed linear,
VAE and regional hybrid references. Other predictor variants reuse the same
present features and do not repeat that assay. Each representative has four
readout spaces; these are not 21 new MD cohorts. Fixed common measurement
controls provide the direct-information baseline.

The birth encoder receives only the current observed input. Actual four-frame
birth-history controls retain their original fixed-neighbor membership, whereas
native trajectory training reselects neighbors. Consequently no generated path
is fed into the birth assay using that incompatible historical support. A birth
readout gain measures information in the encoder; it does not by itself show a
better generator. AP is diagnostic alongside weighted likelihood, Brier,
calibration and source-paired uncertainty.

The cohort is enriched and nonstationary; a high-order liquid bin is not a proven
phase, and independently generated local paths do not form a connected nucleus.
The strict present-liquid historical forecast panel has very few observed local
entries, so rare-event generation conclusions require additional data. Existing
test sources have informed previous research choices; these are development
comparisons, not a newly untouched external test.

**Deferred fourth stage:** a SupraTITO-inspired collective model would evolve
interacting centers from the same physical state with periodic geometry and
coupled stochasticity. It is not implemented in this queue. Local path validity
and temporal distributions must first become credible; connected nuclei then
require an explicit spatial-consistency/connectivity definition. The
[SupraTITO paper](https://arxiv.org/html/2610.01381v1) is a relevant precedent,
not an implemented whole-cell Ni simulator here.

## Reproduction and findings

From the repository in `pointnet-torch214`, the exact recipe is prepared/launched
with `python -m src.research.ni_latent_flow.queue prepare|launch --config
configs/ni_latent_sim/flow_20261018.json` (choose one stage). Environment variables,
safe resume commands and receipt interpretation are documented in the
[workflow](../../docs/ni_latent_flow.md). Configured results are under
`${storage:training_storage}/ni_latent_flow/native-20261018-v1`, with each fit in
`analyses/<treatment>-s<seed>/` and paired tables in `analyses/comparison/`.

### Completed results, 2026-10-18

The queue completed at 03:38:52 Europe/Paris, after 89.2 minutes on two GPUs.
All 51 fixed-budget fits and native evaluations completed. The birth stage
completed 21 representative encoder assays and recorded 30 intentional
unchanged-feature skips; all 24 common controls completed. There are no failed
execution stages or pending fits. This execution status does not establish
physical validity: five arms have raw inverse-transform overflow, recorded in
their validity diagnostics (three GRU seeds and two regional-hybrid MDN seeds).

The following six-ps values average the three frozen seeds, using the existing
source-weighted exports. Temporal variance is divided by the actual saved-MD
temporal variance; 100% is the observed reference. Constraint failures count
sampled states with at least one necessary-bound violation, not coordinates.

| Treatment | Path energy (lower better) | Temporal variance / MD | Bound-violating states |
| --- | ---: | ---: | ---: |
| VAE + autoregressive MDN | 0.6259 | 80.2% | 34.0% |
| VAE + block flow | 0.6332 | 54.8% | 9.3% |
| Direct244 + block flow | 0.6328 | 65.7% | 45.5% |
| Linear VAMP + autoregressive MDN | 0.6469 | 38.2% | 14.7% |
| Linear VAMP + block flow | 0.6643 | 25.8% | 12.2% |
| Local FMRC + block flow | 0.7386 | 7.3% | 0.08% |
| Local SelfTICA + block flow | 0.8645 | 2.4% | 2.27% |

The VAE block-minus-MDN path-score difference is +0.00725 (source-paired 95%
interval +0.00492 to +0.00922): whole-block generation does not improve this
score in the present comparison. Direct versus VAE block is unresolved
(-0.00044, interval -0.00214 to +0.00105). These scores do not override validity
failures: all direct-block paths and 95.7% of VAE-block paths violate at least
one bound somewhere over six ps. Actual saved-MD controls have zero violations.
FMRC and SelfTICA generate strongly suppressed temporal variation, so their
small invalidity rates are not sufficient evidence of useful dynamics.

For default 64-atom nuclei, mean frozen-readout AP at 1/3/6 ps is:

| Readout input | 1 ps | 3 ps | 6 ps |
| --- | ---: | ---: | ---: |
| Actual local measurement history | 0.814 | 0.823 | 0.709 |
| Local SelfTICA latent | 0.784 | 0.751 | 0.668 |
| Local FMRC latent | 0.816 | 0.791 | 0.701 |
| 20-Angstrom SelfTICA latent | 0.834 | 0.821 | 0.747 |
| 20-Angstrom FMRC latent | 0.825 | 0.795 | 0.718 |
| Previously trained regional-hybrid latent | 0.851 | 0.836 | 0.770 |

These are representation readouts on the existing full case-control assay,
not forecasts made from generated futures. Wider context retains useful birth
information, but the new representations do not surpass the inherited regional
hybrid on these mean AP values. At six ps, latent-readout NLL is 0.301 for
regional SelfTICA, 0.306 for regional FMRC, 0.292 for the inherited regional
hybrid and 0.322 for actual local history; lower is better. Check paired
intervals before interpreting small differences.
Regional-minus-local SelfTICA AP gains are +0.050/+0.070/+0.079, with all three
source-bootstrap intervals excluding zero. Corresponding FMRC gains are
+0.009/+0.004/+0.018, with intervals including zero.

Prestructured-liquid dynamics remain unresolved. Among initially liquid origins,
observed high-order occupancy is 14.8%; linear block gives 13.3%, VAE block
10.9%, direct block 19.7%, and regional SelfTICA 52.9%. Occupancy agreement alone
does not establish correct persistence, temporal variation or phase transitions.
Regional SelfTICA occupancy is particularly seed-sensitive: 6.9%, 59.0% and
92.7%. Its initially-liquid qbar6 reconstruction R-squared is 0.343, compared
with 0.660 for local SelfTICA and 0.827 for the inherited linear VAMP encoder.
The source-weighted observed local-entry rate is only 0.38%, making this native
panel weak evidence about rare generated ordering events. Full connected
nucleation has not been demonstrated.

The immediate research priorities are physically bounded measurement outputs,
preservation of motion information through compression, and a matched direct
autoregressive control. None is silently added to this completed fixed queue.
Detailed scores, paired intervals and definitions remain in
[`analyses/comparison/tables/`](../../output/ni_latent_flow/native-20261018-v1/analyses/comparison/tables/).
