# Float32 batched full-cell response training

This version preserves the geometry, feature/readout calculations, likelihood
objectives, selection rules and held-out error formulas described in
`response_training.md`. Differences below are explicit; the historical float64
bank and its exported definitions retain their original identities.

The simulator profile is `configs/simulation/response_atlas_fast_default.json`:
float32 cuEquivariance MACE-MPA-0, exact GPU periodic cutoff graph, conditional
force derivative graphs, two AD directions, and four independent seed replicas
per call. BAOAB, 450K, 1fs timestep and 20/100fs observations are unchanged.
Initial geometries and zero-translation bases are generated in float64 and
explicitly cast. Initial momentum and thermostat random draws retain their
original per-seed float64 generator streams before casting scaled increments.
Saved query q/box/basis and responses record actual float32 simulation inputs.
Preparation separately retains the float64 generating geometry. Neural inputs
and teachers remain as documented, with no new condition/time/species covariates.

Training parents execute eight response queries, reuse their eight values, acquire
24 remaining ordinary values, and independently audit one shared training seed.
This means33 actual trajectory executions, instead of40. Selection executes32
ordinary trajectories. Test executes32 ordinary and8 disjoint response streams.
The new recipe uses a separate seed namespace and separate paths/identity.
Every batch is persisted to SCRATCH and atomically published to STORE before
continuing. Resume uses identical seeds, batch partition and numerical contract.

`oracle-cost.csv` records parent/role, collection_seconds (sum of measured batch
wall times), screen_seconds, ordinary_seconds, response_seconds, audit_seconds,
executed_trajectories, reused_value_rows, logical_force_calls and logical_hvp_calls.
Logical calls count each simulated replica, not batched energy invocations.
Collection time includes query computation and host transfer, excludes checkpoint
loading and publication I/O, and never assigns imaginary per-shot costs to reused
values. A completed resumed batch retains its original measured acquisition cost.

For this version all three training arms report the SAME actual shared training
and selection bank cost plus screens. Responses and audits are included, even for
value-only fits that share the bank. This is not the cost of a hypothetical
standalone value-only experiment. No arm-specific acquisition cost or cost-matched
superiority is inferred from this accounting. Training/evaluation time and the
predictive metrics retain their original definitions. Checkpoint identities and
W&B metadata explicitly record oracle_precision=float32.

Float32 is accepted against the original float64 AD oracle by full-horizon
relative error <=0.01 AND maximum absolute error <=1e-4 for values and responses.
The independent same-seed value/AD execution audit uses atol1e-6, rtol1e-5.
This numerical budget applies to the declared synthetic20/100fs case; it is not
validation of longer liquid trajectories. Numerical checks stay local; scientific
fits use the existing online W&B project and resumable IDs.
