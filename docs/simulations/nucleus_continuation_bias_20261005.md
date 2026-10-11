# Continuation bias and birth-rate calibration

The user questioned whether zero births in the extra negative campaign reflected
suppressed crystallization under the continuation protocol. There are two
separate checks; neither rewrites previous campaign outputs or definitions.

## Paired audit of existing continuations

`configs/analysis/nucleus_continuation_bias_20261005.json` audits all 180 matched
and 240 supplemental continuations against their own saved future. Measurements
at exact 0/0.75/1.5/3/6/12 ps include crystalline fraction and largest component,
paired ancestor bootstrap intervals, temperature strata, and thermodynamics.

Historical branch labels were measured before archive rounding, while original
source labels used float16 observations. The audit preserves those historical
metrics and separately reclassifies archived restart positions at common float16
precision. Check original source input commands and potential hashes, actual
observation timelines, and available NPT thermodynamic records. Earlier log
rows lost during a checkpoint resume remain missing; structural observations
are complete. No additional potential-validation gate or MD is needed for this
retrospective audit. See [definitions](../metrics/nucleus_continuation_bias.md).

```bash
python -m src.research.nucleus_controls.bias --config configs/analysis/nucleus_continuation_bias_20261005.json
```

Output resolves to `nucleus_recovery/continuation-bias-20261005/analyses/paired-v1`
under training storage. Per-branch receipts permit resumption. CSVs use the
metric contract system; grouped plots compare original versus restart growth.

## New rate comparison selected without future outcomes

The negative-only sample had zero original births as well as zero replay births.
The matched positive and dissolved cohorts were selected by future outcome.
Neither design establishes unbiased equality of prospective birth probabilities.
The separate `nucleus_rate_calibration_20261005.json` recipe freezes 64 sources
by seeded hashing, one present-embryo frame per source, before reference futures
are measured. Eligibility is current largest size 8–63 atoms, <=1% current
crystalline fraction, no prior confirmed establishment and 12 ps of coverage.
No eventual fate or future event label selects a source or frame. Exact reused
states and overlapping ancestry are recorded rather than counted twice.

Each full saved Q,V cell receives one source-temperature NPT and two independent
source-temperature BAOAB continuations, 12 ps each. This removes the temperature
substitution of the earlier common-450 K response arm. The other dynamical
differences remain explicit: BAOAB uses 1 fs/100 fs friction/fixed box; NPT uses
native 3 fs/0.3 ps thermostat/3 ps barostat and resets missing bath state. The
established FastAl implementation is reused without extra validation. BAOAB
states, momenta, noise and forces remain float32; NPT integration remains native
float64. The backend scalar energy reduction is separately float64.

Primary future labels use identical float16 observation precision for saved and
new futures, remeasuring all 17 observations after verified archival. Native
branch labels remain preserved separately. Whole-window nucleus and seed-fate
measurements use the existing frozen detector. Completed pairs, source means,
paired differences, temperature strata and independent first-shot denominators
are explicit. See [definitions](../metrics/nucleus_rate_calibration.md).

```bash
python -m src.research.nucleus_controls.calibration prepare --config configs/simulation/nucleus_rate_calibration_20261005.json
CUDA_VISIBLE_DEVICES=0,1 python -m src.research.nucleus_controls.calibration launch --config configs/simulation/nucleus_rate_calibration_20261005.json
python -m src.research.nucleus_controls.calibration summarize --config configs/simulation/nucleus_rate_calibration_20261005.json
```

Use conda `pointnet-torch214`. The maintained launcher freezes code, starts two
owned detached GPU workers and four one-GPU Slurm helpers. The newly authorized
check has a separate three-hour deadline including queue wait, no automatic
extension, and recorded process-group ownership. Other jobs and containing
allocations are unchanged. A task lease remains held until both native MD
publication and common-precision analysis complete. GPU helper architecture and
cache provenance are recorded. No encoder or predictor is trained; no W&B run.

Analysis output is `nucleus_recovery/rate-calibration-20261005/analyses/calibration-v1`;
new MD uses SCRATCH and verified completed trajectories/checkpoints use STORE.
The new collection is registered as `al-nucleus-rate-calibration-20261005`.
Original 0.75 ps source roles/ancestry remain intact. Dense and shooting descendants,
million-atom data, relaxed views and other Al cadence protocols are outside this
specific comparison. Finite samples can diagnose a discrepancy; failure to find
one does not prove exact equality of crystallization probabilities.

## Calibration launch

The sealed plan has 64 independent source states and 192 branches, identity
`cb3b5e7305fca9d7a6e322386abc59373d872918beb51cc3e64ecdac15a6e3ec`.
All 150 sources were eligible; none of the 64 selected source/frame pairs exactly
reuses an earlier matched/extra starting state. Temperature counts are
17/12/11/11/13 at 400/450/500/510/520 K. Only after sealing the plan, reference
analysis found 2/7/13 births by 3/6/12 ps. These outcomes did not change sampling.

Launched October 5 at 20:33:11 Paris time; absolute deadline 23:33:11. The
local supervisor PID/PGID is `3592256`, using two GPUs in allocation `1022218`.
Helper array `1022888_[0-3]` was submitted for four additional GPUs. Ownership
and environment are in the configured output's `technical/launch.json`.
The preparation-only process was interrupted after its 64 references finished
to remove repeated expensive path resolution in status checks; no MD was running
and no metric export had been written. That receipt is preserved separately.
