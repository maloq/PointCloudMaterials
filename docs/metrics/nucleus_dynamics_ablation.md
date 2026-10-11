# Box and thermostat continuation controls, v1

Reuse all 64 states from the sealed present-selected calibration, identity
`cb3b5e7305fca9d7a6e322386abc59373d872918beb51cc3e64ecdac15a6e3ec`.
There is no new outcome-dependent state selection. Each state is a full
70,304-atom Al cell with its actual saved simultaneous Q,V and original box.
All conditions use the source temperature and Lee2003 Al MEAM / FastAl.
The previously selected current 8–63-atom embryo remains the tracked seed.
No velocity redraw, relaxation or displacement is applied. Original Q,V were
float16; decoded initial arrays are float32. No encoder or predictor is fitted,
and both input contracts are null. Source roles and ancestry remain unchanged.

Reuse, with hashes, the 64 saved futures, 64 NPT 3 fs and 128 BAOAB 100 fs-damping
continuations from the completed calibration. Measure 384 NEW 12 ps branches:

* 64 native NVT / Nose–Hoover, 3 fs timestep, 0.3 ps thermostat.
* 64 native NVT / Nose–Hoover, 1 fs timestep, 0.3 ps thermostat.
* 128 BAOAB, 1 fs timestep, 300 fs damping time, two shots per state.
* 128 BAOAB, 1 fs timestep, 1000 fs damping time, two shots per state.

The NVT controls use native LAMMPS float64 integration and the SAME existing
FastAl analytic force/normal-virial callback as the NPT reference. The sole
thermostat command change for NPT3->NVT3 removes isotropic pressure control;
NVT holds the initial box fixed. Native momentum removal stays every 0.3 ps:
100 steps at 3 fs and 300 steps at 1 fs, with absolute source step alignment.
Missing thermostat state is reset initially; native restart files preserve it
on a resumed branch. NVT float64 is an explicit native-control precision
deviation, not a change to response simulation precision.

BAOAB reuses the unchanged float32 Q,p,noise,forces integrator, 1 fs step and
fixed box, with no momentum-removal fix. Only the damping time changes.
Scalar energy reduction is float64 for both backends. Each case/replicate
reuses the parent 100 fs BAOAB seed at both damping settings: paired common
random streams, not new independent noise samples. Damping time is tau in
exp(-dt/tau); a larger value means weaker stochastic coupling. Hardware and
PyTorch execution provenance remain separately recorded. NVT-versus-BAOAB
also changes splitting, integration precision and momentum removal; it is not
an isolated thermostat-only contrast.

No interpolation: observe every exact 0.75 ps, 17 full-cell frames, endpoints
3/6/12 ps. All PRIMARY future classifications use verified float16 archived
positions, identical to the reused calibration measurement. Preserve native
metrics and integration checkpoints separately. Reuse the unchanged
`nucleus_controls.analysis.measure` detector: PTM RMSD<=0.1, FCC/HCP/BCC,
3.6 Angstrom periodic connectivity; isolated new establishment requires >=64
atoms across three adjacent observations (1.5 ps), born after restart and
confirmed by the horizon. Lineage, arrival/merge/interface exclusions are
unchanged. `seed_established` requires descent from the present embryo;
`local_new_nucleus` requires birth within 8 Angstrom of the advected anchor.
Crystal fraction and largest cluster at a horizon are separate growth measures.
These thresholds are an operational detector, not an estimated critical size.

`branches` includes original common-precision reused records with renamed arm
labels and `reused=true`, plus new common-precision records with `reused=false`.
Historical plan identities remain true; no historical table is overwritten.
`saved-reference` copies the original 64 measured futures.

`paired-comparison` includes each arm versus saved futures and predeclared
contrasts: NPT3->NVT3 (box dynamics), NVT3->NVT1 (timestep), NVT1->BAOAB100
(thermostat/integration bundle), BAOAB100->BAOAB300 and ->BAOAB1000 (damping).
Only states with EVERY planned shot complete on BOTH sides enter a contrast.
Average shots within each state, then average across states. Difference is
restart minus reference; bootstrap paired ancestors (2,000 draws, recorded
seed), retaining all shots per sampled state. Export overall and source-T
strata separately. Ratios are undefined at zero reference mean. Descriptive
bootstrap intervals, especially constant samples, are not equivalence or
zero-risk guarantees. Comparisons are exploratory, without multiplicity
correction or outcome-based selection/promotion.

`first-shot-birth-rates` uses predeclared replicate zero per state/arm, the
matching saved-reference subset, discordant directions and a two-sided 95%
Wilson interval (z=1.959963984540054). It never treats repeated shots as
independent sources. These outputs distinguish initial embryo growth from
birth elsewhere; headline anywhere-birth rates alone are not same-site replay.

A NEW nine-hour shared deadline includes scheduler wait. Existing completed
runs are reused without extending an expired campaign. Own workers checkpoint
before the absolute/allocation deadline; own supervisors bound their child
process groups. Completion and partial coverage are explicit. The same 64
ancestors do not enlarge the independent-source population. Original 0.75 ps
data only: dense/shooting descendants, other cadences, million-atom and relaxed
data remain outside this named diagnostic. This does not isolate initial
quantization from missing thermostat-state effects or establish all-Al rates.
