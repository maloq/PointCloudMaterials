# Measurement-aware Ni motion training

This fixed-budget follow-up reuses all630 sealed dense Ni NVE teachers and
all19,800 nucleus assay rows from the history-attention campaign. There are320
train anchors/28 ancestors and310 test anchors/62 ancestors. Original roles,
four exact native history offsets (-0.3/-0.2/-0.1/0ps), fixed central and80 atom
IDs,5fs neural step,100fs physical horizon and1/3/6ps nucleus leads are retained.
No new MD, relaxed structures, filtering or broad spatial replay is introduced.
Neural inputs are only relative Q,V history. Temperature, source age, absolute
time, element and birth labels are not inputs to the dynamics model.

The common model retains the four-frame encoder/decoder attention and adds an
invariant measurement head on the full80-atom vector latent. Per-atom channel
Gram features and cross-atom vector dot products enter two64-wide/four-head
attention layers. A central feature, neighbor mean and learned invariant pool
feed a244-output MLP. No true measurement is supplied to the encoder or this
head. Measurements do not bypass the latent state; its present reconstruction
is explicitly supervised. The original Q,V decoder and evolution remain active.

Four conditions, three seeds20261023/24/25, from scratch at8,000 updates:
present (A) adds present measurement reconstruction; future (B) also predicts
all future measured values and connected5fs increments; observable (C) adds
smooth measurement response; transport (D) instead replaces first-transition
latent response by four transitions throughout the physical teacher. All retain
the preceding atomic value, kinematic, reconstruction, latent-value and physical
response objectives. A/B/C retain the original first-transition latent-response
term. D uses offsets (update+[0,5,10,15]) modulo20, covering each transition
equally without changing data-sampling RNG. Shared history remains fixed under
all current-state and transition tangent queries. Learned latent target power
is detached as before; decoded physical/measurement response is the primary
scientific evaluation, not latent consistency by itself.

Targets are originalBO40/TDA113 plus density74/velocity17. Values and analytic
JVPs of the smooth91 outputs are computed on the sealed float64 physical states
and tangents. Force arithmetic and physical integration are unchanged. BO/TDA
retain the original fixed Ni length convention and receive value supervision
only. No analytic TDA derivative is claimed. Neural training, attention and
neural JVPs use float32. Observable inputs and teacher tangents have native units;
the existing neural Q/5angstrom andV/10angstrom-per-ps scaling is retained.

Measurement mean/std are fitted to all21 times of training teachers only.
Columns with training std<=1e-8 are excluded from normalized measurement losses;
std is floored at1e-5. Each of BO,TDA,density,velocity contributes an equal
group-mean squared error. Present and future weights are0.25 each; connected
increment weight is0.1. Increment loss compares consecutive unanchored decoded
measurement differences to actual differences in the same standardized units.
Observable-response weight is0.25, averaging density and velocity group MSEs
after division by train mean response power per lag/channel/direction, floored
at1e-6 in standardized units. Targets use the full propagated physical tangent;
the model uses exact forward-mode JVP from current Q,V through its latent rollout
and measurement head. Other weights are the original phase/history recipe.
An omitted objective is logged as zero, not evaluated as a scientific zero.

Optimizer/budget: AdamW LR0.001, warmup400, cosine to1% LR, weight decay1e-5,
global grad-norm clip10, effective batch256, microbatch128 with accumulation.
The fixed final8,000-update checkpoint is the only scientific checkpoint.
No test selection, calibration, early stopping, inner split or AP objective.
Twelve scientific models have online W&B IDs; all frozen diagnostic/probe fits
stay local. Their final metrics update the original training IDs.

`learning.csv` retains the original phase losses and adds the four measured
loss fields above. `prediction.csv`, `measurements.csv`, `latent_transport.csv`
and physical paired comparisons retain the ni_phase_history/ni_phase_sim
definitions: source-balanced atomic/response scores, fixed overlap diagnostics,
and BO/TDA/smooth measurements calculated from decoded atomic trajectories.

`measurement_decoder.csv` evaluates the separate244-output head, so it is not
confused with measurements calculated from predicted coordinates. It reports
present and5/20/50/100fs standardized MSE by group, the oracle-current-value
persistence control, connected-increment errors, and smooth response MSE,
target power and error/zero-response error. Response ratios are ratios of
source-balanced means (with denominator floor1e-12); smaller is better and1
means predicting zero tangent. `measurement_paired.csv` compares matched models
within each seed with1,000 whole-source bootstrap draws; differences are left
minus right and uncertainty is conditional on each fitted checkpoint.

Birth readouts retain the same600-tree/depth4/LR0.04 CatBoost protocol and source
roles,81 default64-atom test events/38 ancestors and13532-atom events/42 ancestors,
1.5ps persistence and20% case-control prevalence. Existing present/history
measurement controls are reused with their original metric contracts. Original
phi180/path720/measurement-plus-path spaces remain. New spaces are decoded
present244, decoded present plus20/50/100fs changes976, and actual measured
history976 plus decoded path976. Only neural predicted futures enter features.
New decoder features are in training-standardized units. Proper scores, AP,
AUROC, reliability bins and paired whole-source intervals keep the original
ni_phase_births definitions. AP is an evaluation diagnostic only.

Frozen representation diagnostic: each of three completed parent four-frame
full-response checkpoints is frozen. On the same unique birth-input states,
fit a local pooled180-feature MLP, the atom-resolved invariant measurement head,
and the identical head on raw normalized Q,V padded with zero channels to8.
Fit all244 present targets with group-balanced squared loss (fixed-variance
Gaussian objective), training-only normalization/masks,2,000 AdamW updates,
batch256/microbatch128, LR0.001, warmup100, cosine and final checkpoint. Pooled
features use asinh followed by training mean/std (floor1e-5). No nucleus label
trains these heads. Their reconstructed244 outputs receive the same separate
frozen CatBoost birth probes. `reconstruction.csv` averages error within each
test ancestor then across ancestors. The raw-state control checks readout
competence; a poor latent readout alone does not establish information loss.
Three diagnostic seeds are repetitions, not independent physical sources.

The queue has one absolute12-hour deadline from launch, bounded attempts and
50-update scientific checkpoints. A shorter local allocation may hand off only
within that original deadline. Partial checkpoints are retained and clearly
marked; they are never relabeled as fixed-budget completed experiments.
