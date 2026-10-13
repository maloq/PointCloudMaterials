# History-attention Ni phase-space prediction and birth readouts

This protocol reuses the frozen dense Ni phase-space v2 physical teachers and
ninety-source birth-probe v1 rows. Scientific data, ancestry, native force/state
precision, perturbation directions and labels are unchanged. No new MD is run.
All observations follow the same central atom and the same initial eighty atom
IDs, with exact Q,V at offsets -0.3/-0.2/-0.1/0 ps. The periodic box at each
saved frame recovers center-relative images; it is not a model input. No source
age, temperature, species or future physical state is input to prediction.

The matched experiment changes training starts to the physical anchor (0 fs)
because only native 0.1-ps past frames exist. It does not interpolate past data
or substitute dense simulated frames for true history. The old random-start
fits remain historical comparisons; new classic one-frame controls are trained
at the same anchor starts as attention models. Unroll choices/probabilities,
8000 updates, effective batch256, AdamW LR0.001, warmup400, cosine schedule,
loss weights and fixed final checkpoint are otherwise the preceding protocol.
All normalizers reuse the same calculation from all training teacher paths.
Microbatch128 with two accumulation passes retains effective batch256 on all
workers, allowing the same recipe on48GiB GPUs. Attention models have80304
parameters; classic controls have31264. This capacity change is isolated by
the matched one-frame attention controls.

Architectures: classic is the original eight-vector-channel PhaseProcess.
Attention retains the same spatial VectorField and recurrent evolution field,
adding two four-head equivariant temporal-attention blocks to both encoder and
decoder plus a nonlinear invariant-coefficient vector readout. Scalar queries
and keys are computed from vector Gram matrices; weighted values are vectors,
preserving rotation equivariance. Trainable chronological slot biases identify
the four observation slots, with no absolute-time covariate. One-frame and
four-frame attention controls have identical parameter counts. All neural
states, attention, exact JVPs and losses use float32.

The observed past remains fixed during the autonomous future. The encoder
attends to its three stored past states and current state; the decoder uses the
same past memory and the evolving current code. No true future token enters a
free rollout. Predictions decode changes and start at the actual initial Q,V.
The original 5-fs evolution step and soft Q/V kinematic loss are retained.
The observed past is held fixed under perturbations of the current Q,V, matching
the reused physical teacher. This is conditional sensitivity to the present,
not a derivative with respect to the whole observed historical trajectory.
Reconstruction and latent consistency reencode true future states with the same
anchor-conditioned past. Both history encoders and decoders receive gradients
through the shared memory; it is not detached from parameter optimization.

`learning.csv` uses the original phase objective and definitions: normalized
Q,V path MSE, kinematic loss, joint dynamic-state reconstruction, one-step latent
value consistency, physical-response JVP error and latent tangent transport.
The three history arms have values only, physical response, and physical plus
latent response respectively. Classic/one-frame controls have values/full arms.
Training runs are online in the existing W&B project with stable IDs. There is
no static pretraining, broad spatial replay, test selection or AP objective.

`prediction.csv`, `measurements.csv` and `latent_transport.csv` reuse the exact
ni_phase_sim definitions and calculation functions. Test contains the unchanged
310 anchors/62 ancestors. Q,V error, zero-response-normalized sensitivity,
inner16/all79, overlap diagnostics, BO/TDA/density/velocity measurement errors,
and fixed example plots are evaluated at the same horizons. Scores average
within ancestor then across ancestors. `physical_paired.csv` subtracts matched
models within seed with 1000 whole-source bootstrap replicates.

Birth evaluation preserves all 19800 task rows/19652 unique observations and
all 1/3/6 ps exact leads. The default nucleus is64 atoms sustained1.5ps;32 is a
separate threshold. There are81/135 test events from38/42 source ancestors,
respectively. No new neighborhood filter, re-splitting or input exclusion.
The four founders and sixteen controls per event give20% prevalence; this is a
case-control assay rather than natural nucleation incidence. Whole-source
uncertainty keeps correlated founder atoms and root events together.

Frozen readouts are unchanged600-tree depth4 CatBoost, LR0.04, L2=10,64 borders,
binary likelihood fitting, no early stopping, inner split or calibration.
Scalar latent features use the preceding invariant summaries, 180 for initial
phi and720 for initial phi plus changes at20/50/100fs. Probe spaces are phi,
path, original244 measurements plus path, and four-frame976 measurements plus
path. Controls include original227/244 present measurements and their full
four-frame history. All history descriptors use the fixed current80 IDs.
`scores.csv` and `calibration.csv` preserve ni_phase_births calculations; local
readouts update existing training IDs, never create separate online runs.

`birth_summary.csv` averages metrics over checkpoint seeds, reporting seed SD.
`birth_paired.csv` uses matched source draws and averages paired differences
over seeds, conditional on the fitted readouts. Positive AP/AUROC and negative
NLL/Brier differences favor the left side. Parent results are preserved with
their original random-start training and frozen metric hashes.
