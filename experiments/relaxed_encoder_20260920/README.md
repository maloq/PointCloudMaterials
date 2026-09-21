# Do relaxed inputs and targets improve a local predictive representation?

One-seed matched MACE pilot, requested September 20–21, 2026. Recipe:
`configs/analysis/relaxed_encoder_pilot.json`. Results:
`output/relaxed_encoder/pilot-20260920/RESULTS.md` (generated after readouts).

Three width64 MACE fits share the development-selected E parent, initialization,
optimizer, batch256, 512 updates, compiled BF16, SIGReg on the raw export, six
neighbor queries, future-center prediction and physical/TDA/order/moment anchors:

| Arm | Current encoder input | Neighbor and next-frame embedding views / physical targets |
|---|---|---|
| instantaneous | observed | observed |
| hot_to_relaxed | observed | relaxed |
| relaxed_to_relaxed | relaxed | relaxed |

The hot→relaxed arm supplies the observed current center to its encoder and uses
relaxed target-view encodings (the existing joint-gradient JEPA objective). No EMA
teacher or changed gradient convention is introduced. Neighbor positions used as
predictor queries come from the current observed frame for hot arms, and the current
relaxed frame for the relaxed-input arm; future coordinates never become queries.
All target families, including fixed equivariant moments, use the target domain.
Velocities are absent from these position-only encoders. Each exported encoder is
still a snapshot encoder; relaxed-input inference additionally requires quenching.

Use 4 training and 2 selection sources per temperature (400/450/500/510/520 K),
chosen before inspecting outcomes. Two origins, frames64/368 (48/276 ps), plus
next-frame targets at +0.75 ps. 64 centers per training cell; 16 per selection cell.
This is 2,560 training anchors and 320 validation anchors, about51.2 sampled epoch
equivalents at512updates, not512 full dataset epochs. The small release is a pilot.

Quench complete periodic70304-atom cells, fixed box, generating Lee2003 Al MEAM,
FIRE infinity force tolerance0.01 eV/Å. Select nearest80 atoms from the original
current frame for each center and retain their identities in its relaxed pair.
Future-frame neighborhoods are selected using that future observed geometry; query
center identities persist across the pair. These80 atoms form a tracked candidate set. Inputs AND all structural targets
are cropped at normalized radius8 in each domain; atoms that leave support after
relaxation are excluded from both. No new atoms enter the candidate set. This
nearest80 cap differs from the parent's uncapped radius crop, identically across all three arms; parent hot/cold controls quantify
input-domain effects. No isolated free-surface cluster quench is called equivalent.

Frozen tests retain all150 historical independent sources and their original
train/selection/calibration/test roles, using the same two origins restricted to
original natural at-risk rows. Original MD onset defines labels at0.75/3/6/9/12ps.
No quenched PTM classification defines the outcome. Include frozen parent hot/cold,
paired hot/cold geometry85+TDA144+order8, historical geometry baseline, and condition
only. Matched padded-input linear/MLP heads share budgets and training-only scaling.
Source-bootstrap intervals do not include seed uncertainty. Sparse origins and
few near-term events limit power. Do not select configurations on these test scores.

Speed experiment: three training cells, same CPU host,32 MPI ranks, FIRE0.01 versus
0.03/0.1 eV/Å from the same observed coordinates; measure target and coordinate
fidelity as well as runtime. Full-cell reuse amortizes relaxation across centers.
Constrained halos and GPU MEAM are possible later comparisons, not tested claims.

Known information-budget caveat: full-cell relaxation uses the surrounding cell,
so any benefit can include information transmitted from beyond the encoder crop,
not only removal of thermal displacement. Deployment must have that cell available.
The historical descriptor control (`original_geometry`) includes the original
packet's motion channels as well as geometry/order. Conditions are temperature
one-hot, observed trajectory age /600 ps and its square.

Prelaunch event-count audit: 3,464 paired at-risk observations. Within12ps there
are38/10/4/8 positive windows in train/selection/calibration/test respectively;
the8 test positives come from5 test sources. Timing and AP will be noisy; this
is a feasibility and effect-direction pilot, not a decisive accuracy ranking.

September 21 recovery: nine of360 cells reached the original10,000-iteration
limit. They are restarted from archived full-precision failed-quench coordinates,
with FIRE internal state reset, up to50,000 additional iterations/250,000 force
evaluations. The generating potential, fixed box and0.01 eV/Angstrom convergence
requirement are unchanged. The other351 completed cells remain unchanged.
Per-cell metadata records which observations required this numerical recovery.
