# BCR-v1: Bottleneck-conditioned reconstruction of atomic environments

## Status and scientific question

This is a proposed experimental/implementation specification, not an implemented or validated training result. Numerical settings are initial engineering choices, not established optima for aluminum. Pin the actual repository revision, environment, data manifests and comparator configurations before implementation. Read the repository's agent and experiment-layout instructions before adding files.

**Question:** Does conditioning geometric reconstruction on the exact invariant code exported at inference produce a more informative and robust frozen structural representation than input-matched VICReg and ordinary node denoising?

**Not the question:** Can a decoder obtain low reconstruction error, can codes look Gaussian, or can a model trained on crystallization outcomes predict crystallization?

Pretraining uses snapshots and artificial coordinate corruptions only. No future crystallization labels, force/energy targets, motif classification loss, neighbor-equality loss, temporal alignment, VICReg, SIGReg, EMA teacher, or stop-gradient in the primary BCR arm. Geometry-derived observables may be used for development evaluation, never as hidden auxiliary training losses in this arm.

## 1. Frozen protocol and data splits

Preserve an existing untouched lineage-level test split. If none exists, allocate independent root trajectories approximately 70/15/15 to train/development/test, stratified by known acquisition conditions where possible. All shooting descendants, repeated exports and transformed copies of a configuration inherit the same root split. Do not partition overlapping atom neighborhoods randomly. A historically consulted crystallization test is development evidence, not a fresh final test.

Save a manifest with root/source IDs, frame ID and physical time, center atom ID, species, coordinate units/dtype, periodic cell, generating potential/protocol, temperature metadata when known, neighbor support and preprocessing version. IDs, frame number, temperature, phase and source metadata do not enter the primary BCR network. Species, the distinguished center flag, padding masks, noisy geometry and corruption level do.

Start with Al from one generating potential and a declared temperature range. Use native physical length units. Define one global reference distance d0 from TRAINING data: the median center-to-first-neighbor distance across a source-balanced sample. Save d0. Never normalize every environment/frame to its own length scale. If inputs use length/d0 internally, record the single shared conversion.

Keep the same physical observation support as a newly input-matched VICReg control. An archived control with a different crop, cap, units, or data exposure remains contextual evidence only. Prefer a radius-defined observation and dynamic padding/bucketing without silent truncation. If N_max is used, count every overflow and fail preparation rather than discard neighbors silently.

Use source-balanced then time-block-balanced sampling, with centers drawn independently of future outcomes. Spatial thinning and frame spacing reduce duplication but do not make MD windows statistically independent; uncertainty is calculated over root trajectories. Save total root/source/frame/center counts and anchor exposures.

## 2. Neighborhood and corruption contract

Extract a full, validated center-relative neighborhood X = {x_a}, with x_0 = 0. Handle periodicity in the original simulation cell using a validated neighbor/image implementation. At radii that admit multiple periodic images of one atom, retain explicit image identities and document that observation definition. Do not assume orthogonal cells or apply a naive fractional-coordinate minimum-image rule to arbitrary skew cells.

The corruption uses the same finite atom set for clean and noisy views. Select that set once, then retain membership even when artificial noise moves a point across the original crop boundary. Do not wrap artificially corrupted patch coordinates into the periodic box; the decoder sees a center-relative Euclidean patch. Preserve atom correspondence only for constructing the per-node target; raw IDs never become model features.

For noncentral, nonpadding atoms:

    epsilon_a ~ N(0, I_3)
    x_noisy_a = x_clean_a + sigma * epsilon_a

The center remains at zero, has epsilon_0 = 0, and is excluded from noise loss. Do not subtract a noise center of mass: that changes the corruption law. Apply any random rotation coherently to clean geometry, corrupted geometry and noise targets. Permute both views and labels coherently; do not assign canonical/radius-ranked atom embeddings.

Build the encoder graph from clean geometry. Independently build ALL decoder edges, distances, harmonics and cutoffs from corrupted geometry. The decoder must never consume clean edges, clean radial bins, clean distances, clean per-node features, clean equivariant states, original atom IDs or target-side metadata.

The true padding mask and center/species labels are allowed side information and must be identical across decoder controls. Fixed membership and atom count contain coarse information; conditioning-gain analyses explicitly control for it.

### Boundary policy

For the initial protocol, use a loss weight w(r_clean) equal to 1 for r <= 0.8 R_support and tapered smoothly to 0 at R_support. Use a C2 quintic transition, 1 - 10u^3 + 15u^4 - 6u^5 for u in [0,1]. This weight is used only by the loss/evaluator and is NOT a decoder input. Apply the same weights to every control. Report interior (r <= 0.65 R_support), middle and outer-shell metrics separately, including unweighted diagnostics.

Boundary robustness is evaluated twice: fixed membership and re-extraction from the full perturbed snapshot. Smooth pooling alone is not assumed to guarantee continuity of every cropped message-passing path. If re-extraction causes material discontinuities, repair the observation/support contract in every matched neural arm before claiming robustness.

## 3. Encoder interface

Use the same pinned MACE-style backbone, interaction depth, irreps, radial basis and physical receptive field as the matched VICReg arm. MACE is an example, not a dependency on a new architecture. Do not simultaneously introduce a new high-degree backbone, decoder task and data support.

Return exactly one signed invariant code:

    z = encoder(clean_snapshot)       # [B, 128]

The trained clean pass exports the same code used to condition reconstruction. It is not a diagnostic side projector. Internal sample-local normalization is allowed; avoid batch-dependent inference and do not automatically apply terminal L2 normalization. Use the same export at train/eval/probe time.

A concrete readout when adapting a node backbone is:

    g_center = scalar feature of the distinguished central atom
    g_pool = sum_a w(r_a) * h_a^(0) / n_ref
    z = Linear(SiLU(Linear(concat(g_center, g_pool))))

n_ref is a fixed training-data reference count, not the count of the current patch. The center-aware pooling retains a route for local density and coordination information. Match this readout across retrained invariant-model comparisons where applicable. Keep the pre-head pooled features available for diagnostics, not as an undeclared extra decoder input.

Start at d_z = 128; test 64 and 256 only in later one-factor ablations. Parameter count and absolute code scale are logged. No clean per-atom or equivariant features may cross the bottleneck.

An invariant code does not need to supply a global orientation: the corrupted coordinates are the decoder's geometric reference. At sufficiently strong noise, correspondence/orientation ambiguities impose a genuine limitation; zero reconstruction error is not required. Do not introduce a cubic canonical frame or six Cartesian reconstruction slots.

## 4. Decoder interface and architecture

Primary decoder is separate and unshared with the encoder. This makes the clean-to-decoder information path easy to audit; it is an adaptation rather than an exact reproduction of weight-shared SCD.

    epsilon_hat = decoder(
        noisy_positions, species, center_flags, padding_masks,
        z, log(sigma/d0)
    )                               # [B, N, 3]

Initial decoder:
- Two equivariant message-passing blocks.
- Hidden irreps: 64x0e + 32x1o + 16x2e, with permitted tensor-product paths recorded.
- 16 radial basis functions, smooth radial cutoff at 2*d0, no silent neighbor cap.
- A 16-dimensional log-noise embedding.
- FiLM/adaptive conditioning in each block from concat(z, noise_embedding).
- Scalar nonlinearities and equivariant gates; never apply coordinate-wise nonlinearities to a higher-order irrep.
- Additive conditioning biases are permitted on scalar channels only. Non-scalar gates share one multiplier across all m components of a channel.
- Per-node output is one polar vector (1o under O(3)), predicting unit Gaussian epsilon.

The chosen widths/cutoff are initial controls, not validated optima. Use small nonzero initialization for conditioning so encoder-gradient tests are informative; do not silently initialize all clean-code paths to zero. Prefer FP32 initially and enable BF16/compilation only after parity tests.

The reconstruction is x_hat = x_noisy - sigma*epsilon_hat. This skip from noisy coordinates is allowed. No skip from clean coordinates is allowed.

## 5. Corruption levels and calibration

Initial five equally weighted levels, interpreted as PER-CARTESIAN-COMPONENT standard deviation:

    sigma/d0 = [0.01, 0.02, 0.04, 0.08, 0.12]

Per-atom RMS displacement is sqrt(3)*sigma. These are pilot settings, not measured Al noise scales. Sample one level per anchor; draw new epsilon on every visit. Balance level counts over each effective optimizer batch/window and log actual counts.

Before training, audit a source-balanced training subset for each level:
- actual coordinate quantization uncertainty versus sigma;
- nearest-neighbor identity changes and distance distributions;
- close-pair/collision frequency, finite geometry and decoder connectivity;
- changes in fixed smooth radial/angular observables;
- interior versus boundary behavior.

As an engineering guard, remove levels below ten times the measured coordinate uncertainty. Never treat FP16-to-FP32 casting as precision recovery. If a level produces unsupported or numerically pathological geometry, lower the declared maximum for ALL arms and document the change. Do not selectively reject unfavorable noisy samples based on clean phase or descriptors while still claiming Gaussian corruption.

No noise curriculum in the initial experiment. No encoder corruption in the primary arm. A later robustness augmentation can perturb both the encoder target geometry and the geometry from which decoder noise is added; label that separately. Do not turn it into a hidden latent-equality loss.

## 6. Objective and optimization

For sample i, loss weights w_ia, and n_i_weight = sum_a w_ia:

    L_i = sum_a w_ia * ||epsilon_hat_ia - epsilon_ia||^2 / (3*n_i_weight)
    L_BCR = mean_i L_i

The center and padding weights are zero. Average environments equally, then average source groups according to the declared sampling mixture. Do not let dense environments dominate through point count. At a fixed sigma, per-coordinate reconstruction RMSE equals sigma*sqrt(L) with the same weighting. The vector-displacement RMS is sqrt(3) times this quantity.

Only L_BCR plus ordinary optimizer weight decay trains the primary arm. Do not add Physical85/TDA/order losses, future targets, contrastive learning, SIGReg, VICReg, z-equality, score/force interpretations, or an EMA teacher. Both encoder and decoder receive gradients. Noise/clean coordinates are fixed data targets, but z must NOT be detached.

Proposed optimization start:
- AdamW, lr 3e-4, betas (0.9, 0.95), weight decay 1e-5.
- Effective batch 256 environments; accumulation may change physical batch, not objective weighting.
- Global gradient clipping at 1.0.
- Linear warmup over 5% of the planned run; cosine decay to 3e-6.
- One noise draw per clean code per update.
- Pilot: 10,000 updates, one seed. Confirmation: 50,000 updates, at least three paired seeds.

These are proposed budgets. Measure throughput/memory and obtain the actual compute allocation before launch; do not infer GPU-hour costs here. Match encoder initialization tensors, source streams, noise levels and corruption RNG in applicable arms, not merely an integer seed. Save random states, optimizer/scheduler, data position and all export metadata for resume.

Log raw loss by sigma, source and radial shell; code mean/std/norms; covariance trace/effective rank; code-conditioned versus unconditional diagnostics; and gradient norms entering the encoder through z.

## 7. Mechanism controls

Primary comparison set:
1. BCR: train encoder and shallow decoder jointly.
2. Unconditional decoder: same noisy inputs and decoder class, no informative clean code; use a learned constant code and report active/total parameter count.
3. Frozen VICReg code + the same trainable conditional decoder, using a training-compatible VICReg checkpoint whose source lineage exposure is known.
4. Ordinary node denoising: same backbone, predicts coordinate noise from corrupted input using a per-node equivariant head.
5. Matched VICReg representation reference, retrained if support/data/export conventions differ from the archived comparator.

Additional cheap diagnostics: frozen random encoder + conditional decoder; fixed rich geometric descriptor + decoder; metadata-only conditioning using only side information also available to unconditional reconstruction.

Do not use an untrained final pooling MLP as the sole export of the node-denoising baseline. Evaluate its pooled trained scalar features and record any dimensional difference. For all learned models report both pre-head pooled features and the declared final export as separate diagnostic representations. Do not concatenate richer states into only one arm's primary score.

No-backprop code shuffling, zeroing and perturbation are evaluation interventions, not replacements for independently trained controls. Random codes from an untrained geometry encoder can still carry structural information; do not expect them to be automatically useless.

Stage one-factor ablations after the mechanism pilot: decoder depth 2 versus 4, d_z 64/128/256, and changing the noise grid. Avoid a full factorial. No architecture or corruption selection based on crystallization outcomes.

## 8. Conditioning gain and reconstruction evaluation

Choose fixed development anchors balanced over roots, with per-anchor noise keys. Fast checks: up to 256 anchors, two draws per level. Full checks: up to 1,024 anchors, four draws per level. Report actual root/anchor counts; do not duplicate scarce roots to pretend independence. Keep a second unseen-noise-key bank for final protocol verification.

For each sigma evaluate:

    R_true = mean reconstruction NMSE with the correct code
    R_swap = mean NMSE with matched, deranged clean codes
    R_U = mean NMSE of a separately trained unconditional decoder

Report:

    Delta_swap = R_swap - R_true
    G_swap = (R_swap - R_true) / max(R_swap, tiny)
    G_U = (R_U - R_true) / max(R_U, tiny)

Do not clamp negative gains to zero. Use four independently drawn matched derangements in the full evaluation. Match wrong codes within temperature/condition, atom-count bins, coarse density and current structural-order bins using metadata/geometry computed for evaluation only. Prefer different root trajectories. Save the matching tolerances and bin edges fitted on TRAINING data; when exact matching is impossible, report coverage and progressively relaxed matches separately.

Also report an unrestricted shuffle, but never use it alone: it can measure only broad phase/density differences. Decoder output with wrong codes can be out-of-distribution, so large swap gain is not proof of useful representation. Positive G_U, frozen-VICReg/random-code comparisons and independent structural probes provide the complementary evidence.

Reconstruction metrics: noise NMSE, physical coordinate RMSE, per-shell error and error in fixed descriptors recomputed on x_hat. The target noise is artificial, not a physical force. A small coordinate error at tiny sigma or low normalized error at a very large sigma is not sufficient evidence of good information encoding.

For paired confidence intervals, first average corruptions/shuffles per anchor and anchors per root; resample ROOTS, retaining paired model results. Show seed variability separately. If root count is small, state that uncertainty is poorly resolved rather than counting atoms as independent replicates.

## 9. Structural information retention

Freeze encoder parameters and inference statistics. Fit all feature/target transforms on probe-training data only. Use ridge regression with a fixed log regularization grid and one small two-layer 128-unit SiLU MLP with a fixed training budget. Fit/evaluate both on identical source splits and sample counts. No encoder gradients, no head warm-start from the reconstruction decoder, and no per-frame normalization.

Three primary evaluation families (all derived from observations, not direct BCR loss targets):
- Radial/density: smooth radial bins, local number density, smooth coordination, independent radial length statistics.
- Angular/order: smooth q4/q6 and higher angular quantities, selected third-order/cross-degree contractions, averaged-order measures only where the full required neighborhood is within matched input support.
- Rich geometry: a fixed, well-resolved SOAP/ACE-style target or independent multiscale moment expansion, with documented radial/angular resolution and block scaling.

Use fixed prototypes FCC/HCP/BCC/icosahedral/disordered shells and held-out distortions as a separate structural sensitivity assay, not training motif labels. No finite descriptor is complete physical ground truth. Avoid claiming that a coordinate-derived descriptor is statistically independent of the pretext target: it is an independent *measurement*, not independent information.

Report standardized RMSE and R2 globally, by temperature, and especially within liquid/noncrystalline and interface-like strata defined from fixed current-structure rules. Show the incremental benefit over density/order-only covariates. Retain per-root results. Do not infer within-liquid skill from a high global R2.

Also measure structural nearest-neighbor recall at k=20 against a frozen reference metric, with cross-source candidate restrictions. Define all reference block weights using training data. Report raw and train-standardized Euclidean metrics separately; do not default to spherical normalization.

Code diagnostics: centered covariance spectrum, covariance trace, participation ratio, per-coordinate variances and numerical signal-to-noise. High rank is not an objective, and global rescaling can change raw distances without changing decoded information.

## 10. Robustness without structural blindness

### Exact symmetries and implementation invariance
- Same cloud in repeated inference, different batches, and permuted atom order.
- Arbitrary rotations of clean/noisy/target together: z invariant; predicted epsilon equivariant.
- Translation of raw coordinates before recentering.
- A twelve-neighbor FCC first shell and 24 proper cubic symmetries; no unique frame.
- Include an inversion test if the implementation promises O(3) rather than SO(3).

Start deterministic FP32 tests at atol 1e-6, rtol 1e-5 on normalized fixtures. Calibrate stricter CPU/float64 and looser mixed-precision tolerances against actual numerical results, and save them. Zero tensors require an absolute error criterion, not an unstable relative ratio.

### Small perturbations

For amplitudes a/d0 = 1e-5, 1e-4, 1e-3, 3e-3, 1e-2 (only use real-data amplitudes above validated coordinate resolution), perturb one atom and then all neighbors with independent isotropic displacements. Evaluate fixed membership and full re-extraction separately. Include targeted cutoff and neighbor-order crossings. Encoder sees the perturbed snapshot without reconstruction conditioning during this assay.

Define a fixed train-calibrated embedding-distance scale S_z (for example, median pair distance in a declared matched structural population) and report:

    sensitivity(a) = ||z(X+delta) - z(X)|| / S_z

Also report raw changes and scale S_z, because normalization can hide amplitude collapse. Plot median and 95th percentile versus a and compare with the matched VICReg baseline. Coordinate perturbations are not exact nuisances: continuity is expected, not identical codes.

### Genuine changes

Evaluate hydrostatic strain and volume-preserving shear at signed 1% and 3%, vacancies/controlled neighbor removal, stacking-fault/slip examples, and tracked real MD changes. Recompute reference structural observables after perturbation. Do not demand invariance to these operations. Quantify whether decoded observable changes track true changes, including sign/amplitude where meaningful.

For change detection on held-out trajectories, define events using a current-to-current geometric/rearrangement criterion independent of the encoder, not future crystallization onset. Compare matched noise-only and real-change cases. Report discrimination and descriptor-change fidelity; a large latent jump by itself is not a successful response.

## 11. Gates and checkpoint selection

G0 correctness: clean-to-decoder path is only z; symmetry, gradient, loss-normalization, replay/accumulation and export tests pass. The prediction is unaffected when forbidden clean side data are changed while z/noisy input remain fixed.

G1 mechanism: positive matched-code and trained-unconditional gains on at least two medium/strong levels on held-out roots, with paired uncertainty. As a pilot engineering target, use >=5% relative G_U; freeze or revise this threshold using development uncertainty BEFORE confirmation, never with crystallization test scores. A statistically unresolved result is inconclusive, not proof of no conditioning.

G2 information: compare against input-matched VICReg on the three primary structural families within noncrystalline data. Proposed noninferiority margin is +0.02 in standardized RMSE (2% of a target's training standard deviation), evaluated at family level with per-target results visible. This is a proposed practical tolerance, not a physics constant. Require a declared meaningful improvement in at least one family or retrieval/perturbation assay before claiming a better general encoder.

G3 robustness: no reproducible numerical discontinuities above the agreed tolerance at exact symmetries or negligible perturbations; genuine change fidelity must not deteriorate. Smooth-but-uninformative codes fail G2/G3 jointly.

Save initialization, 1%, 5%, 10%, then every 10% of training and the terminal model. Keep both a general-development-selected checkpoint and the terminal checkpoint. Select only among checkpoints meeting the structural-retention and robustness requirements; use conditional reconstruction performance as a tie-breaker, not swap gain alone. Lower reconstruction loss with worsening frozen probes is evidence against the current setup, not a reason to hide early checkpoints.

Run crystallization probes only after choices are frozen. Use identical observation information, frozen encoders, identical probe capacities and fresh lineages when available. Report archived velocity-rich baselines as richer-input references. A successful general encoder can still fail this transfer task; do not reframe it as guaranteed success.

## 12. Proposed implementation modules and tests

Use the repository's actual layout after inspection. A new isolated training-method namespace may contain:

- data.py: immutable patch records, source-balanced sampler, corruption generator, noisy graph construction;
- model.py: encoder adapter returning z and separate conditional decoder;
- objective.py: normalized weighted epsilon loss only;
- evaluate.py: reconstruction, matched shuffles, trained-control comparisons and frozen feature export;
- probes.py: source-separated frozen structural probes and perturbation suite;
- runtime.py: optimization, evaluation cadence, manifests, resume and checkpoint selection;
- configs/: exact experiment/ablation variants and completed calibration constants.

Required tests:
1. Wrong clean edges/features cannot enter decoder API; fixed z/noisy inputs imply unchanged predictions.
2. Nonzero finite gradient from reconstruction through z into encoder.
3. Same code in condition path and standalone export; reload equality.
4. Permutation/rotation/translation and origin-mask behavior.
5. All decoder graph features recomputed from noisy coordinates.
6. Center/padding excluded; environment means do not depend on padding amount or duplicated batch rows.
7. Corruption variance, physical units, seed reproducibility and no center-of-mass subtraction.
8. Finite gradients near coincident noisy atoms; no silent dropping of hard examples.
9. Fixed and re-extracted support audits, PBC/image identity and overflow errors.
10. Full-batch versus accumulation/compiled-precision parity for deployed settings.
11. Resume reproduces next anchor IDs, noise draws and optimizer state.
12. Paired shuffle coverage/strata, zero/negative gain handling and correct root-bootstrap units.

No large training run should be launched before a tiny real-data overfit test and the G0 suite pass. A synthetic overfit proves only that the optimization path works, not that held-out representation quality is good.

## Primary sources and distinction from this proposal

- Yan, Li and Zhang, GeoRecon: Graph-Level Representation Learning for 3D Molecules via Reconstruction-Based Pretraining, arXiv:2506.13174v1.
- Perez and Gomez-Bombarelli, Self-Conditioned Denoising for Atomistic Representation Learning, arXiv:2603.17196v1.
- Batatia et al., MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields, arXiv:2206.07697.
- Bartok, Kondor and Csanyi, On representing chemical environments, arXiv:1209.3140.
- Lechner and Dellago, Accurate determination of crystal structures based on averaged local bond order parameters, arXiv:0806.3345.
- e3nn official tensor-product documentation; pin the installed implementation/version.

The clean pooled-code conditioning idea comes from the reconstruction literature. Separate encoder/decoder weights, the specific Al-scale corruption grid, frozen-encoder metrics, acceptance margins, and launch budgets above are proposed adaptations. They are not a reproduction of either paper's full training recipe and have not been validated by this protocol document.
