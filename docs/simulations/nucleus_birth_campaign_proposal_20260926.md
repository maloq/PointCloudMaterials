# Producing more spontaneous crystal births

Proposal, 26 September 2026. No simulations submitted by this document.

Follow-up: the user authorized a more uniform grid with fewer replicas.
The [launched first stage](al_birth_uniform_20260926.md) uses 400–500 K in 10 K
steps, two fresh melts each (22 sources), Slurm array 1009492. The original
five-temperature/four-replica design below remains proposal history.

Prioritize fresh independently melted Al replicas, an isothermal temperature
sweep, and early collection of isolated establishments. Optimize usable distinct
births per allocated compute-hour, including preparation and analysis. Keep
fixed-duration, outcome-blind evaluation sources. A faster complete solidification
is not necessarily a more productive birth simulation.

## Evidence from our data

The [completed origin audit](../../experiments/crystallization_origin_20260925/RESULTS.md)
found 348 isolated establishment candidates across the 150 Al sources. The
[training harvest](../nucleus_harvest.md) contains 206 candidates from 90 training
sources; 205 support at least one 12 ps-history candidate window. These are
operational establishments, not independently verified critical nuclei.

This proposal uses only training-source outcomes for choosing new settings.
Grouping their primary isolated establishments by preparation temperature gives:

| Temperature | Independent training sources | Candidate births | Births/source | Confirmed births retained by proposed early stop |
| --- | ---: | ---: | ---: | ---: |
| 400 K | 18 | 70 | 3.89 | 68 |
| 450 K | 18 | 64 | 3.56 | 59 |
| 500 K | 18 | 25 | 1.39 | 24 |
| 510 K | 18 | 29 | 1.61 | 28 |
| 520 K | 18 | 18 | 1.00 | 18 |

Reproduction of this descriptive calculation: read the fixed release `plan.json`
at the registered Al64-v1 cache and the corresponding audit
`technical/sources/<id>/{events.json,graph.npz}`. Restrict release `role` to
`train`, events to `threshold == primary` and `kind == isolated_establishment`.
Count events, not crops. Define `f10` as the first frame where
`crystalline_atoms / atom_count >= 0.10`, and proposed stopping frame as
`min(f10 + 16, 800)`, or 800 if the fraction never crosses. The native interval
is 0.75 ps. Retention requires `confirmation_frame <= stopping_frame`.

Of 206 births, 191 occur at or before the first 10% crossing; 197 are confirmed
by the proposed stop. Across these 90 histories, mean retained measurement
duration plus the existing 315 ps preparation is 62.55% of the original 915 ps.
That is **37.45% fewer integration steps**, not a measured wall-clock speedup;
denser output and online analysis have costs. It also does not establish complete
forecast-label availability for every retained birth or recover rare late births.
The two zero-birth 520 K training sources remain in all denominators.

## Relevant primary literature

| Study | Result relevant to this campaign | Limit of transfer |
| --- | --- | --- |
| Mahata, Zaeem & Baskes, *Understanding homogeneous nucleation in solidification of aluminum by molecular dynamics simulations*, MSMSE 26 (2018), [paper](https://arxiv.org/abs/1706.07307), [DOI](https://doi.org/10.1088/1361-651X/aa9f36) | Million-atom Al with 2NN-MEAM; 1325 K/300 ps melt preparation; an isothermal sweep found its maximum measured nucleation rate at 475 K. | Supports examining 425–475 K, not assuming 475 K is optimal for our exact files, operational labels, finite box or compute budget. |
| Hussain & Haji-Akbari, *How to quantify and avoid finite size effects … homogeneous crystal nucleation*, JCP 156 (2022), [paper](https://arxiv.org/abs/2111.12647), [DOI](https://doi.org/10.1063/5.0079702) | Periodic spanning nuclei diagnose strong size artifacts; measured rates varied non-monotonically with size. | Lennard-Jones evidence motivates geometry/size checks; it does not validate an Al atom-count threshold. |
| Zhang, Zuckerman & Jasnow, *The weighted ensemble path sampling method is statistically exact …*, JCP 132 (2010), [paper](https://arxiv.org/abs/0810.1963), [DOI](https://doi.org/10.1063/1.3306345) | Weighted resampling can concentrate trajectory effort while preserving the represented path distribution. | Weights and the underlying dynamics must be preserved; cloned deterministic states do not create new paths by themselves. |
| Allen, Warren & ten Wolde, *Sampling rare switching events in biochemical networks*, PRL 94 (2005), [paper](https://arxiv.org/abs/q-bio/0406006), [DOI](https://doi.org/10.1103/PhysRevLett.94.018104) | Forward-flux sampling uses interfaces and conditional crossings to sample rare transitions and estimate rates. | Original demonstration is a biochemical switch; applying it to our MD needs a separately specified dynamical and weighting protocol. |

Interpretation: lower temperature is not automatically better; increasing the
driving force can compete with slower rearrangement. The relevant collection
quantity is the number of usable isolated births before growth consumes the
remaining liquid, divided by total cost. A temperature minimizing the time to
mostly crystalline material need not maximize that quantity.

## First stage: 20 independent development sources

Run **400, 425, 450, 475 and 500 K, four new independently melted replicas each**.
These are simulation replicas, not encoder-training seeds. This is a screening
stage with limited uncertainty resolution; retain two promising temperatures
rather than claiming a precisely located optimum from four sources.

| Setting | Proposed value |
| --- | --- |
| Material/potential | Al, existing Lee2003 2NN-MEAM, exact pinned files |
| Cell | Cubic periodic 26 × 26 × 26 conventional FCC cells, 70,304 atoms |
| Preparation | Independent velocity seed, 1325 K melt for 300 ps |
| Target-temperature start | Match existing source protocol: assign fresh target-temperature velocities once, then NPT hold |
| Ensemble | Zero-pressure isotropic Nose–Hoover NPT |
| Integration | 3 fs; thermostat 0.3 ps; barostat 3 ps; COM removal every 0.3 ps |
| Observation | Save from target-temperature initialization, including the first 15 ps |
| Maximum hold | Existing 15 ps preparation segment plus up to 600 ps measurement |
| Development stopping | After the first 10% full-cell PTM-crystalline crossing, continue at least 12 ps; otherwise run to cap |
| Saved dynamics | Positions, velocities, boxes, atom IDs and exact timeline every 0.15 ps (50 steps) |

The actual existing producer is
[`independent_meam_source.py`](../../src/simulation/campaigns/independent_meam_source.py).
It assigns new velocities at the target temperature, runs 15 ps, resets the
step counter, then starts the measurement dump. This is an abrupt initialization,
not a finite-rate cooling ramp. Preserve its physical setup for the initial
comparison, but remove the observational blind interval. Maintain a continuous
timeline and explicit segment boundaries in the new data.

Full 12 ps histories for the main post-preparation hold analysis must lie after the
declared 15 ps preparation segment and must never cross the velocity reset.
Earlier births remain an explicit preparation/short-history stratum rather
than being discarded or called ordinary long-history examples. A finite-rate
cooling protocol would be a separate future ablation.

Potential checksums already pinned by the producer:

```text
Lee2003_Al.library.meam  f72f19b5185e6da9c4e4c26029346b9210296b289ba791178dee1e923281835e
Lee2003_Al.meam          b1ba33a29d8884692aeb4a1f0c78df51146f6f68d281121135dfca3207506e6a
```

Validate that the prepared high-temperature melt is liquid using existing
thermodynamic/PTM checks, complemented by cluster persistence, RDF and diffusion.
A low total crystal fraction alone could conceal a surviving seed. Keep
preparation failures and their native states explicitly recorded; do not reject
valid target-temperature trajectories for having inconvenient outcomes.

Parallelize independent replicas on CPU allocations using the maintained MEAM
workflow. Record allocated core-hours, wall time and analysis/I/O cost. Do not
assume a GPU accelerator supports this exact MEAM kernel or replace its potential
to exploit H100/H200 hardware. A finish-time estimate requires measured source
throughput on the allocated nodes.

## Second stage: matched size comparison and production

At the two retained temperatures, add **four fresh 256,000-atom cubic replicas
per temperature** (40 cubed conventional FCC cells). Compare with the matching
70,304-atom development sources using identical dynamics, labels and stopping.
Inspect periodic-image connections, nucleus extents, competing-front distances,
usable history, and cost-normalized event yield. Equal run counts alone are not
an equal-resource comparison; extrapolate only using measured resource costs.

The existing million-atom Al audit supplies 42 candidates, useful for harvesting
now. It is not a matched size ablation: its elongated cell and 1 fs integration
with different damping differ from the 70k family. If new moderate-size results
justify scaling, use several cubic 1,048,576-atom replicas (64 cubed FCC cells)
with the same physical kernel, rather than relying on one large trajectory.

An illustrative production expansion is **80 fresh melt roots**, split equally
between the two selected temperatures and using the selected cell size. Preassign
48 train / 8 selection / 8 calibration / 16 test, balanced across temperature,
before observing any outcomes. Pilot and size-screen sources are development
data only. All descendants inherit the melt-root role. Four-replica screening
does not eliminate uncertainty; production is staged, with no guaranteed birth
count or allocation duration claimed here.

Use early stopping for the declared early-transformation training population.
Selection/calibration/test production sources run the full predetermined
600 ps measurement regardless of crystal fraction or event count. Freeze the
outcome-blind region/origin sampling and target population before evaluation.
Early-stopped data do not represent late-time risk. Preserve original Al64-v1
and its roles unchanged; register this as a separate versioned birth release.

## What makes a generated birth useful

Keep the existing primary 64-atom/1.5 ps establishment criterion and sensitivity
variants, with persistence expressed in physical time at the denser cadence.
Continue causal lineage tracking to distinguish new establishment, arrival,
merging/interface cases and unresolved origin. This threshold is not a physical
critical-nucleus claim. Record transient and dissolving embryos as well as
successful establishments; use original observed MD for labels, not minimized
structures. Dense and 0.75 ps-downsampled labels need an explicit comparison.

Report distinct accepted births, independent melt roots, unresolved fraction,
usable 3/6 ps forecast origins with 0/3/6/12 ps histories, competing arrivals,
and allocated cost. Count multiple crops, frames and branches separately from
independent birth lineages. A full 6 ps negative requires 6 ps follow-up plus
the 1.5 ps confirmation allowance; the trajectory tail is censored when that
requirement fails. The proposed 12 ps stop extension does not make every late
origin complete automatically.

Harvest the whole cell, then retrieve ordinary atom-centered observations using
the [existing causal-risk protocol](../../experiments/crystallization_origin_20260925/HARVEST_PROPOSAL.md).
Do not locate predictor inputs using future nucleus centroids or memberships.
Keep an outcome-blind control sample and exact inclusion probabilities for event
enrichment; neither enriched AP nor raw positive fraction estimates natural
prevalence. Temperature, simulation age and absolute time stay audit metadata,
never encoder, predictor or probe inputs. Selection remains based on predictive
likelihood for models; this simulation screen measures data yield and quality.

## Optional approaches after direct MD

**Conditional shooting:** start with approximately 20 training-parent precursor
states across size/order strata, including plausible failures, and eight 12–24 ps
branches each. This studies whether a precursor establishes or dissolves. With
our deterministic thermostat, identical full-state continuations are identical;
fresh Maxwell–Boltzmann velocities define a position-conditioned experiment,
not independent continuations of the same observed position–velocity history.
Record that intervention and its conditional ensemble. Keep branches grouped
by parent, and do not count 160 branches as 160 new independent nuclei. A true
committor additionally needs explicit competing basins and stopping rules.

**Weighted ensemble / forward-flux sampling:** useful if direct trajectories
become too event-poor at conditions of scientific interest. Preserve path weights,
ancestry and basin definitions; define how paths diversify under the chosen
dynamics. These are separate weighted trajectory studies, not an automatic
replacement for the natural-frequency evaluation set. No claimed speedup for
our Al system is established yet.

**Inserted crystal seeds:** useful for growth, interfaces or nucleus-stability
studies. They do not supply spontaneous precursor histories for the birth task.
Similarly, removing an existing grain or simulating isolated liquid pockets
changes the physical preparation and cannot silently populate this dataset.

Only after validating the Al pipeline should it expand to Mg/Ta, using each
potential's own undercooling scan and liquid checks rather than copying Al
temperatures. Resolve the current Ti ancestry ambiguity before increasing its
production. Material normalization and geometry-only model inputs stay unchanged.

## Precision, storage and implementation boundary

At the full 615 ps cap, 0.15 ps cadence gives 4,101 frames: about 3.46 GB per
70,304-atom source for float16 positions and velocities alone. Full-cell analysis,
native restarts, temporary output and any float32 references add overhead.
Save uniformly at this cadence during the first stage; event-triggered-only
recording would complicate representative controls and must not be introduced
without an explicit sampling design.

Keep full-precision integration restart files; never restart from float16
analysis arrays. Use `scripts/convert_trajectory.py`, verified float16 positions,
float32 boxes and exact identity/timeline arrays, with quantization errors and
checksums before deleting temporary dumps. Stage new simulations on SCRATCH and
publish complete runs and stopped failures/restarts to STORE; record locations,
potential provenance and ancestry in `configs/datasets.json` when implemented.

This proposal needs a maintained recipe/runner extension for continuous early
recording, causal chunk-boundary stopping, observation conversion and the fresh
manifest. Existing launchers do not implement the entire proposal. Do not resume
the explicitly stopped September 17 precision campaign. No new executable
configuration, simulation job or training run is created here.
