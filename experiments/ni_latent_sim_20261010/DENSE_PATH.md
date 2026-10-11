# Dense physical trajectory and response supervision, October 11

Question: does supervising the interior of a physical 100-fs trajectory make
the learned local evolution more accurate than supervising its endpoint?

The six matched fits use the existing 14,400 fixed-central-atom Ni sequences,
original ancestry/roles and training-only feature normalization. They reuse
the frozen unrelaxed Small eSEN and actual saved velocities. Each observation
is the same 280-vector, compressed to 16 dimensions. Four native observations
at -0.3, -0.2, -0.1 and 0 ps initialize the forecast. No new birth filtering,
structural replay, species, temperature, simulation-age or time covariates.

All fits advance by 5 fs, twenty neural steps per native 0.1-ps interval.
At intermediate steps only the current code changes; the preceding three
observation codes remain fixed. At each 100-fs boundary the completed current
code replaces the oldest observation slot. Thus reducing the integration step
does not turn a 0.3-ps observed history into an invented 15-fs history.
Drift scales with the timestep, and stochastic amplitude with its square root,
relative to the previous 100-fs convention. No physical state or teacher output
is inserted into the predicted path after initialization.

The same 300 original full-cell states are continued for 100 physical steps of
1 fs with GPU Ni Mendelev2012 EAM/FS. Retain values and analytic initial-state
tangents every 5 fs. Two shots use the original seeds, thermostat, fixed box,
positions, velocities and local perturbation directions. Float64 integration,
momenta, noise and tangents; float32 forces. The last physical frame is compared
with the archived endpoint to audit unchanged semantics. These short fixed-box
BAOAB teachers remain distinct from the original NPT observations.

Every teacher follows the same initial 80 atom IDs and center for the 100 fs;
the longer saved-data targets recompute nearest neighbors about that center.
Frozen eSEN input JVPs project physical position tangents to 128-dimensional
structural responses at all twenty times. The two directions perturb initial
positions (0.01 Angstrom atom-vector RMS) or velocities (0.1 Angstrom/ps RMS).
Their propagated derivatives refer to the same initial perturbation throughout
the path, rather than new perturbations inserted at intermediate times.

| Treatment | Physical value targets | Physical response targets |
| --- | --- | --- |
| endpoint_response | 100 fs | 100 fs |
| path_values | 5,10,...,100 fs | none |
| path_response | 5,10,...,100 fs | 5,10,...,100 fs |

Each treatment uses two seeds, 3,200 updates, batch/microbatch 256, AdamW,
maximum LR 0.001, 200-step warmup and cosine decay. The long likelihood observes
native futures at 0.1, 0.5 and 1 ps, reached by 20, 100 and 200 autonomous neural
steps respectively. Its four-particle mixture likelihood, current-feature
reconstruction weight 0.2, physical value weight 0.1 and response weight 0.1
match the earlier protocol. Physical losses use 16 training-source anchors per
update. Time losses are averaged, keeping total weights equal across treatments.
Neural response differences use epsilon 0.5 and identical neural random streams
for both signs; past observations stay fixed. Physical targets average two MD
shots, neural short predictions average four latent paths.

Checkpoint selection uses native-future NLL on the same inner-selection sources,
not response error, held-out data or birth AP. The earlier heuristic latent
Jacobian penalty is omitted from all three treatments to isolate dense physical
supervision. eSEN remains frozen. This experiment changes motion learning only.

Evaluation includes all twenty short physical times on held-out sources,
separate Q/V response errors, target response powers, zero-response and
persistence-response controls. Free neural rollouts reach 6 ps (1,200 fine
steps), scored at 0.1/0.5/1/3/6 ps, including the three sequence regimes.
The unchanged 32/64-atom nucleus assays use 1/3/6-ps lead times and all pooled
held-out roles. No new independent nuclei or MD source ancestors are claimed.
Short supervision constrains path means and initial-state sensitivities; it does
not establish a complete physical path distribution or correct NPT closure.

Recipe: `configs/ni_latent_sim/dense_path_response_20261011.json`.
Launch: `python -m src.research.ni_latent_sim.workflow submit --config
configs/ni_latent_sim/dense_path_response_20261011.json`.
Execution details and readiness: [operations](../../docs/ni_latent_sim.md).


Operational recovery policy (October11): cross-device replay of the older100-fs
endpoint is diagnostic only. Preserve independently validated new paths even
when this comparison differs or is unavailable. Checkpoint retries and separate
evaluation stages retain the exact scientific recipe. Results from an unfinished
fit must carry its actual completed and selected updates; they are provisional,
not full-budget evidence. See the operational record for supervision receipts.
