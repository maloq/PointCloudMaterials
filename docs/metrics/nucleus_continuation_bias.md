# Paired continuation-bias audit, v1

No new MD or model is fitted. Audit all 180 matched-control and 240 supplemental
branches separately; a previously used embryo is not a new independent state.
Every restart is paired with its own original saved future at exact native
0/0.75/1.5/3/6/12 ps times. Record actual input manifests, completion hashes,
original LAMMPS settings/potential identities, and complete follow-up.

Two measurement protocols preserve historical metric meanings. `native` reuses
the saved branch graph's crystal fraction and largest tracked component, which
were measured before archive rounding. `common_float16` reclassifies the verified
archived float16 Q using the same PTM RMSD 0.1 and periodic 3.6 Å connectivity
as the original reference catalogue. Both use float64 decoded geometry, native
boxes and >=4-atom tracked components. Reference metrics are read from the pinned
original graph. This controls the observation precision mismatch; it does not
undo the rounding of the state used to start the dynamics. Components are an
instantaneous structural diagnostic, not newly defined nuclei or birth events.
Historical birth outcomes remain under their original frozen metric definitions.

`paired-branches` records reference/restart crystal fractions and largest sizes
per arm/shot/time/measurement. Group by campaign, cohort, arm, time, measurement
and either all sources or an exact source-temperature stratum. Average repeated
shots within each ancestor, then average across ancestors. `paired-structure`
reports these two means, restart minus reference, ratio-of-means percentage
change, and a paired ancestry bootstrap (10,000 draws, fixed seed) of the mean
difference. A single ancestor has no interval. These are descriptive intervals
without multiplicity corrections, not a formal population-wide equivalence test.
Original future selected-positive and selected-dissolved cohorts remain separately
labeled. Their saved binary outcome is conditioned by selection and cannot
provide an unbiased spontaneous birth-rate baseline. The supplemental random
liquid cohort was selected using present structure only, but its 12 ps reference
birth incidence is zero and thus has little power to establish rate equivalence.

Thermodynamic records use original source `measurement.lammps.log`, exact saved
steps, and NPT continuation `lammps.log`. Repeated log blocks use their last
record for a step. Earlier log rows missing after a checkpoint resume remain
undefined, with missing steps recorded in each branch receipt; no thermodynamic
values are imputed. Structural observations remain complete. Source inputs are audited against the configured timestep,
momentum-removal interval, NPT temperatures/damping/pressure, and MEAM file hashes.
Volumes are products of archived box lengths. Potential energy is total energy
per atom (eV), pressure is bar, temperature is K. BAOAB has only final kinetic
T in this audit: sum(p²)/mass/(3N kB), using saved float32 momenta promoted for
reduction. Its initial/intermediate kinetic temperatures and all pressures and
energies remain undefined here. Native LAMMPS temperature uses its original
COM-adjusted degrees of freedom; the O(1/N) convention difference is recorded,
not interpreted as a thermal drift. `paired-thermodynamics` applies the same
ancestor-level difference/bootstrap procedure.

These checks distinguish a protocol discrepancy, a structural growth discrepancy,
and a measurement discrepancy. They cannot by themselves assign causation to
bath-state loss, initial quantization, thermostat choice, or force precision.
Original source NPT used a continuous thermostat/barostat state. Its reconstruction
resets missing bath state. The response arm instead changes ensemble, damping,
timestep and, for non-450 K sources, bath temperature. No equivalence of its
kinetics to the original NPT process is assumed. Other Al cadences and populations
are outside this named paired audit.
