# Present-state-selected birth-rate calibration, v1

Select before inspecting futures. The original 150-source release is screened
for current largest components of 8–63 PTM-crystalline atoms, <=1% crystalline
atoms in the cell, no earlier confirmed establishment, and a complete next
12 ps observed interval. Use native 0.75 ps cadence only. Rank eligible sources
by SHA256(seed, source ID), take the first 64, and draw one eligible frame per
source uniformly with the recorded seed. The tracked seed is the current largest
component; equal-size ties use the smaller graph node ID. The anchor is the atom
nearest that component's periodic centroid. All eligibility, role, ancestry and
initial hashes are sealed in the plan BEFORE measuring any saved future. No
future establishment or dissolution label selects a case, frame, or replica.
Cases may overlap previous ancestry; exact reused source/frame pairs are recorded
and the new analysis is not pooled as an independent extension of old cohorts.

Each of 64 starting full 70,304-atom Al cells gets one original-settings NPT
continuation and two BAOAB continuations: 192 new 12 ps branches. Both protocols
use the source temperature; no common 450 K temperature substitution. NPT uses
3 fs, 0.3 ps thermostat, 3 ps isotropic zero-pressure barostat, and momentum
removal every 100 steps. It resets missing thermostat/barostat state. BAOAB uses
1 fs, 100 fs friction and fixed initial box. Actual saved simultaneous Q,V are
used without relaxation, velocity redraw, or added displacement. Original Q,V
were stored float16, decoded to float32. BAOAB states/momenta/noise/forces are
float32; NPT integration is native float64 with float32 forces. Scalar energy
reduction remains float64. Seeds have a distinct protocol namespace. There are
no encoder/predictor inputs, training teachers, or W&B training runs.

Existing native branch metrics are archived unchanged. The PRIMARY new result
is measured after verified float16 archival so both saved reference and restart
futures receive the same observation precision, PTM classifier and periodic
connectivity. Recompute all 17 full-cell labels, then apply the unchanged
`nucleus_controls.analysis.measure` to the complete observation window and its
original historical graph prefix. This comparison does not restore initial
precision lost from old source states. It does not alter historical exports.

PTM RMSD<=0.1 and FCC/HCP/BCC; 3.6 Å connectivity; minimum tracked size 4;
strong overlap 4 atoms and 25% of the smaller component. An isolated nucleus
requires >=64 atoms across three adjacent observations (1.5 ps span), born
strictly after restart and confirmed by the horizon (3/6/12 ps). The inherited
lineage and arrival/merge/interface exclusions remain as in `nucleus_controls`.
`seed_established` additionally requires descent from the chosen present embryo.
Crystal fraction and largest component at each horizon are separate continuous
diagnostics. These birth criteria are an operational detector, not a critical
nucleus estimate. Original reference and all replay arms share this detector.

`paired-comparison` averages a case's planned replicas, then averages across
ancestors. Include only states with every planned shot for that arm complete;
the corresponding reference mean uses precisely the same subset. Differences
are restart minus recorded future. Paired ancestry bootstrap uses 2,000 draws,
fixed seed, and retains every shot of a sampled ancestor. Temperature-stratified
and overall means remain separate. These descriptive intervals are not proof
of equivalence, and degenerate all-zero bootstrap intervals are not zero-risk
bounds. Rate ratios are undefined if the reference mean is zero.

`first-shot-birth-rates` additionally takes one predeclared minimum-replicate
shot per state/arm, reporting paired denominators, original/restart event counts,
and discordant directions. Two-sided 95% Wilson score intervals for the restart
rate use z=1.959963984540054; they remain nonzero-width at zero events. Repeated
shots are never counted as independent states. Do not compare the saved count
with a denominator including all noise repetitions. Partial coverage is
provisional and does not trigger outcome-dependent additions or extensions.

A new three-hour shared deadline includes scheduler wait; reserve time checkpoints
unfinished owned work. No earlier expired budget is extended. Completed/queued/
unfinished observations remain explicit. Other Al cadence populations, dense or
shooting descendants, million-atom cells and relaxed states are outside this
named diagnostic. This is a state-selected birth-rate comparison; it cannot
alone isolate bath-state reset from initial quantization or changed dynamics.
