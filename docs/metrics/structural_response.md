# Spatial initialization and anchored derivative motion training

## Fast student revision v4

`motion_fast_20261007_v4.json` replaces the unlaunched v3 data recipe for the new
comparison. Both completed spatial checkpoints initialize four1600-update arms:
value_joint, derivative_joint, value_adapter and derivative_adapter. All retain
the same future targets, likelihood selection, short physical banks and batch256.
No spatial replay or broad structural data loader is used.

The neural motion coordinate uses the centered secant
`[E(Q+epsilon*0.1ps*V)-E(Q-epsilon*0.1ps*V)]/(2*epsilon)`, epsilon=.01.
Short input responses use the centered difference of the entire predicted
100fs endpoint at `state +/- .5*basis`, divided by1. Both signs use identical
Brownian noise. These are explicitly approximate STUDENT derivatives. The reused
teacher endpoints and tangents are the sealed v2 analytic targets, bound to the
unchanged spatial geometry implementation, checkpoint and physical bank.
No new MEAM integration occurs. Physical archived states/tangents remainfloat64
and force arithmeticfloat32. All student secants and short predictions usefloat32;
long latent integration and readout useBF16 autocast. Input states for these neural
calculations are explicitly cast tofloat32. Local audits compare the motion
secant with exact AD, halve its step, check the nonlinear adapter JVP, and time
full256-observation optimizer updates without online tracking.

Joint arms train every MACE parameter. Adapter arms freeze MACE and cache its
base coordinates `(s,m)`. A residual128->128->128 SiLU adapter maps `s` to
`H(s)=s+W2*SiLU(W1*s+b1)+b2` and motion to the exact `DH(s)[m]`.
W2 and b2 start at zero, so every arm has the same initial representation and
latent dynamics. Adapter and dynamics learn; the cached MACE backbone does not.
Adapter derivatives are ordinary analytic tensor expressions. Short adapter
responses use the same centered student protocol applied to cached perturbed
base codes. This is a capacity/adaptation comparison, not fully joint MACE.

The long latent objective is evaluated once on the full batch256. Joint encoder
graphs are regenerated in microbatches64 (32 on A100/A40/L40S), then receive the
full-batch latent cotangents. Geometry has no dropout/batch-dependent statistics.
This preserves the objective and gradient chain rule while avoiding32 repeated
latent rollouts. Hardware-dependent floating-point reductions remain recorded.

Birth evaluation uses the pinned `al-nuclei-birth-010` established PTM catalogue
with **no additional neighborhood filtering**, per the October7 clarification.
Use exact1/3/6ps pre-emergence Q,V patches, up to4 founder centers per event and
four same-source random control centers per founder. Controls exclude the
first-member atom IDs of all established births catalogued in that source.
They are **nonbirth controls**, not guaranteed continuously liquid observations;
neither positives nor controls are screened by current crystalline fraction,
component size or distance to crystals. Require all three lead observations,
saved velocities and consistent ancestry. Deduplicate matching whole-cell birth
Q,V/box/membership/time identities and matching positive-center Q,V inputs.
Original selection/calibration/test roles are pooled only for evaluation.
The first4 original-training ancestors under a seeded hash order provide internal
NLL readout selection, followed by refitting all original-training ancestors.
Readouts compare structural128, structural+motion256 and future-law48 coordinates,
using logistic regression and CatBoost predictive likelihood. No birth labels
select the motion encoder. AP/AUROC, NLL and Brier are evaluation diagnostics,
with1000 whole-ancestor bootstrap intervals. Each physical event has equal total
weight across its matched rows; the nominal class prior is.2. This retrospective
event-enriched assay is not a population nucleation rate. The old strict
neighborhood-screened benchmark retains its original definition and scores.
Diagnostics stay local and final metrics update the existing training run.

Historical definitions below retain their stated revision scopes.

The objective is motion utility: improved prediction of actual future structure
relative to matched value-only training. Spatial pretraining is a reusable first
stage, not the final success criterion. One common seed is an exploratory pilot,
not a replicated treatment effect. Nucleus labels, AP, temperature, simulation age
and explicit time covariates never select or train these encoders.

## Spatial source and targets

New explicitly compact recipes bind `descriptors.selection` to training-active
target names. They may prepare and predict only those columns while preserving
the original block weights and target definitions. The current 153-column mask
removes only zero-weight outputs from the 298-column objective below. Existing
checkpoints and frozen recipes retain their original head dimensions.
The subsequently requested CNA-inclusive selection has 198 outputs (40 bond
order, 45 CNA, 113 TDA); its new descriptor objective gives each family one third
of the weight and retains equal weighting across the 18 TDA groups. Its target
selection enters the data identity, preventing reuse of an incompatible panel.

Reuse the immutable packed subset of1,056,768 unrelaxed nearest80 neighborhoods
from117 sources in Al, Mg, Ti and Ta. These are broad spatial observations chosen
without nucleus/phase filters. Preserve existing Al64 training ancestry and the
fixed material-to-Al length normalization; material/potential remain metadata.
The underlying raw pool is larger; this experiment uses the entire stated packed
subset, not all13.4million dynamic patches. Million-atom Al geometry can enter
spatial training without velocities. Static/relaxed collections are not added.

The target uses40 bond-order and258 TDA columns from the checksum-bound442-column
rich-descriptor producer; geometry/CNA columns are excluded. Normalization is the
existing packed-training mean/scale, never held-out moments. Half the descriptor
weight belongs to bond order, equally across active coordinates; half to TDA,
equally across18(n32/n80 x H0/H1/H2 x statistics/spectrum/Betti) blocks and then
active coordinates within each. Inactive coordinates have zero weight. The
reported descriptor_half_mse is half the weighted mean squared standardized
error, averaged over two full views; it equals a fixed-variance Gaussian NLL
up to a constant.

Two methods share initialization, shuffle, data, capacity and schedule:

- vicreg_descriptors: two independent0.01A jitter views of the same neighborhood,
  centered again, predict the unchanged original bond-order/TDA target.
- masked_teacher: the same objective, plus prediction of the complete-view EMA
  embedding from a view removing20% of atoms outside the center+12 nearest
  neighbors. A separate predictor and stop-gradient target avoid directly forcing
  incomplete and complete embeddings to coincide. Teacher parameter EMA rate.01.

VICReg is applied DIRECTLY to the exported128 coordinates. With unbiased sample
covariance C, variance=mean(relu(1-sqrt(diag(C)+1e-4))) and
covariance=sum(offdiag(C)^2)/128. Average each penalty across the two full views.
Invariance=mean((z1-z2)^2). The spatial objective is
25*invariance +25*variance +covariance +descriptor_half_mse +masked_half_mse.
The last term is zero in the first method, otherwise half the squared difference
between per-observation LayerNorm predictor and teacher outputs, averaged over
coordinates and observations. Head outputs and these statistics usefloat32.

Train12 complete epochs, global batch256, with one epoch linear warmup to.001
and11epochs cosine decay to.00001; AdamW weight decay1e-5, gradient clip5.
The final epoch12 checkpoint is exported without held-out checkpoint selection.
A fixed4096-row sample of original structural selection sources is diagnostic
only. All repeated epochs are reuses, not new independent observations.

## Encoder and computation

Geometry-only MACE:128 channels,128-dimensional structural output, two interactions,
correlation3, max_ell3,5A edges and nearest80 support within8A. Center-relative
coordinates retain physical Al-equivalent units. One constant atom channel and
center flag; no species/material/time/temperature features and no coordinate
reconstruction. Delete the previous per-layer saturation and repeated radial
attenuation. Pool center scalars and a weighted neighborhood mean. The single
pooling taper is flat through6A and cosine from6–8A. Pool normalization is a
trainable per-observation LayerNorm, replacing frozen initialization moments.
Export is an orthogonal-initialized linear projection plus an MLP residual.

Mixed precision is explicit: spatial fitting and latent dynamics/readout use BF16
autocast; geometry, radial/harmonic calculations, pooling, projection, losses and
covariances usefloat32. Parameters/optimizer remainfloat32. All neural motion input
JVPs usefloat32 and the same cuEq contraction polynomials evaluated by ordinary
PyTorch operations. Learned weights, polynomial coefficients and the forward
function are shared; only the high-order contraction implementation changes.
The ordinary PyTorch path avoids excessive CPU memory in derivative kernel
generation. Structural fitting and ordinary teacher forwards keep fused cuEq kernels.
The original v1 numerical preparation used microbatch16/8 and failed before any
motion scientific fit. Motion revision v2 reuses the completed epoch-12 exports
by their original campaign identity and checkpoint SHA256. It retains exact reverse-over-reverse AD for the inner velocity JVP and outer
response gradients. A forward-mode candidate failed the checkpoint comparison
and was rejected before scientific fitting. Finite differences are used only
for a local numerical audit, never as training derivatives. Joint motion uses microbatch8 on large GPUs or4 on the other
queues, accumulating to256. Frozen geometry is cached at those microbatches; the latent-only training
uses256 or128. Spatial replay uses256 or128 with full-batch covariance. Physical saved MEAM states and tangents remain their original
float64; forces werefloat32. No new simulation is performed in this campaign.
The native cuEquivariance backend uses ir_mul layout and fused convolution.
Copied teachers are rebuilt from the same architecture and exact state dictionary;
generic deep copying is incompatible with this installed fused-kernel wrapper.

Gradient caching computes full256-observation VICReg cotangents then replays
encoder microbatches. It does not replace full-batch covariance with smaller
covariances. Actual microbatch/backends, memory and measured precision are in
execution and numerical receipts. No batch statistics or dropout affect replay.

## Motion continuation

Each spatial checkpoint supplies four matched continuations: value_frozen,
derivative_frozen, value_joint, derivative_joint. All use s=E(Q),
m=0.1ps DE(Q)[V], the same autonomous kinetic latent SDE and initial dynamics
parameters,4 training particles,0.05ps maximum latent integration step and the
same draws of native future observations and short simulation records.
The frozen arms keep E fixed. Joint arms use encoder learning rate1e-5 and
in historical v1/v2 replay256 broad spatial samples every fourth update at weight.1; the descriptor
and masked-prediction heads are frozen to their pretrained meanings. Dynamics
learning rate1e-4, AdamW decay1e-5, clip10,1600 updates, effective batch256.

The v3 data revision disables broad structural replay entirely (weight0,
interval0), per the October7 user instruction. It does not instantiate the replay
model or open the packed structural training collection during motion fitting.
The spatial checkpoints retain their original full12-epoch provenance. A per-fit
`motion-data.json` records actual observation-role counts, encoder and predictor
inputs, initialization and whether a spatial replay loader was opened. Physical
teacher records supply ancestor/source grouping; cached target job order must
match the bank exactly.

New birth evaluation pins `al-nuclei-birth-010` through its cumulative reader.
Original-training ancestors are used for fitting/inner NLL selection; original
selection/calibration/test are pooled only for final evaluation. The source/event
contract records actual saved velocity availability, exact0.10ps timelines and
exclusions. Catalogue established-event counts are candidate records before
lead/neighborhood/control eligibility and physical event deduplication, not final
case counts or independent trials. Strict pre-birth and present-selected
embryo-establishment targets remain separate. Birth labels never enter motion
encoder fitting or select its checkpoints.

Reuse the pinned88 verified-MEAM source views from84 ancestors and their11,200
exact deduplicated current Q,V observations:6080 fitting,1024 inner selection,
4096 pooled original-held-out rows. Original spatial fitting may include the
original-training ancestors subsequently used for motion inner selection; no
future-label gradients come from that inner set. Original selection/calibration/
test remain pooled for final future evaluation. All64 and legacy16 are explicit.

Long supervision uses exact saved native0.1/0.5/3/12ps futures. The24 fixed
radial/angular statistics retain their previous training-only normalization and
mixture likelihood. Additionally, re-extract each future's nearest80 neighborhood
of the same center and encode it with the FIXED pretrained geometry E0. This
128-D target anchors future structural states beyond the24 statistics. E0 stays
fixed for every motion arm; no moving-target scale contraction is possible.
The long loss is descriptor mixture NLL +.2*(half MSE of mean latent structural
forecast against E0(future)). Training means average4 latent particles.

Reuse176 archived100fs whole-cell FastAl BAOAB teachers with two noise shots,
0.01A RMS position and0.1A/ps RMS velocity directions. The target endpoint is
mean_shot E0(Q_100fs). Its derivative is mean_shot DE0(Q_100fs)[simulator tangent].
The value term is half latent endpoint MSE. Derivative arms add the sum over two
directions of .25*mean((student_JVP-target_JVP)^2)/max(mean(target_JVP^2),1e-6).
Both are weighted.1. The same fixed teacher and scale are used in value-only
controls. Frozen arms precompute current representations and input Jacobians;
joint arms differentiate the live geometry. Short derivatives usefloat32.
No derivative is claimed at3/12ps: those targets are saved MD observations.
Fixed-box BAOAB response targets and saved NPT long futures are distinct laws.
Short response crops follow the same initial80 atom identities; long saved
targets reselect the nearest80 around the same center. These supports are recorded
separately and are not assumed to be identical distributions at100fs.

Checkpoint selection uses only inner-original-training future-descriptor NLL,
every100updates. No structural test, birth AP, clustering score or fitted ensemble
selects the model. Resumption preserves optimizer and sampling state. Scientific
training is online in teshbek/PointCloudMaterials; numerical checks are local.

## Reports

Future evaluation uses8 particles and common query blocks/seeds. NLL is the
joint24-D diagonal-Gaussian mixture negative log likelihood divided by24.
Descriptor MSE compares the particle mean with normalized actual future targets.
Structural MSE compares the mean latent structural state with fixed E0(future).
Coverage90 is the fraction of target coordinates whose mixture marginal CDF lies
between .05 and .95. The persistence baseline keeps the present24 descriptors;
its MSE uses the same future targets. The marginal likelihood baseline fits one
diagonal Gaussian per horizon to training targets only, with variance floor1e-5.
Predictive variance is mean_particle(variance+mean^2)-mixture_mean^2.
Aggregate within whole melt ancestor then equally over ancestors; intervals are
percentiles2.5/97.5 of1000 ancestor-bootstrap mean draws. Compare structural MSE
within a shared spatial teacher; cross-method numerical scales need not coincide.
The common24-descriptor scores support comparisons across spatial methods.

Representation diagnostics report average/minimum coordinate standard deviation
and covariance participation rank, (trace C)^2/trace(C^2). These measure scale and
variance concentration, not mutual information or proof of sufficiency. A smaller
training derivative loss alone is not evidence that motion prediction improved.
Held-out100fs bank records report endpoint half MSE and the same two-direction
relative tangent error for every arm, including value-only controls. The run
summary averages these within ancestor and then equally across ancestors. These
use held-out source states and their archived physical streams; no training
bank observation is relabeled as test data.
All exported scientific CSVs bind these definitions and implementation hashes.

Fixed future embeddings, endpoint/tangent teachers, frozen current codes and
frozen input Jacobians are cached under their checkpoint, data and implementation
hashes. Caches are resumable across allocations and shared by the matched arms;
they do not add observations or change targets. Atomic partial packets and sealed
combined tensors distinguish preparation from scientific optimizer updates.

All motion arms load one checksum-bound latent-dynamics initialization, produced
from the declared seed and retained in the execution receipts. This guarantees
identical actual initial parameters across worker hardware and launch order.

Joint derivative fitting is admitted only on devices with at least64GiB memory;
this resource rule changes scheduling, not the eight scientific comparisons.
Held-out response evaluation computes the same JVP values without retaining
parameter-gradient graphs between records.


## Parent dataset cadence contract (October 7, 2026)

The shared catalogue now restricts `al-nuclei-birth-030` to uniformly saved
native 0.30 ps inputs, checking raw timestep arrays. The `010` source/event
population used here is unchanged. New preparations pin the current parent
fingerprint; frozen runs keep their original pins and metric definitions.
Source-bound 0.10 ps label/history supplements retain their original provenance.
