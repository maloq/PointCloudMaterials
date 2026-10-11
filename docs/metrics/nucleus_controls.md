# Dissolved-embryo and random-liquid continuation controls, v1

No encoder, predictor or probe is trained. This is a new simulation diagnostic,
separate from the previous 15-birth recovery assay. All new starts are complete
70,304-atom periodic Al cells with their actual simultaneous saved velocities.
Temperature is a physical thermostat setting and matching/audit metadata only.

## Cohorts fixed before new dynamics

The original 150-source, native 0.75 ps Al catalogue is screened. Strict
historical failed-embryo candidates must be isolated, single-root, strongly
connected episodes that reached at least eight atoms at two observations and
then have verified disappearance. These are small embryos, not previously
established critical nuclei. Select the first peak-size observation. Require
no previously confirmed establishment, no current tracked component >=64,
initial PTM crystal fraction <=1%, and complete recorded dissolution and
reference observations within the next 12 ps. Choose the earliest eligible
peak per source. Exclusions, including additional episodes of the same source,
remain explicit. The frozen campaign contains 12 dissolved cases.

An establishing control has a small precursor on the strongest-overlap backward
path of a later isolated establishment. Ties select the lower parent node ID.
The same initial-cell restrictions apply, and establishment must be confirmed
within 12 ps. Match the source thermostat temperature exactly and initial
connected embryo size within two atoms. Unique positive sources are assigned
globally to minimize size mismatch (100 cost per atom), then prefer the same
ancestor (5 cost for another source), with a fixed hash tie-break below one.
A positive source cannot be another block's negative source. Two cases can
share ancestry only within their own matched block; this is checked explicitly.
This is a declared event-enriched comparison, not an unbiased birth-rate sample.

The random-liquid control uses the dissolved case's ancestral source. Uniformly
sample among available frames with a largest current tracked cluster <=7 atoms
and <=1% current PTM crystal fraction; sample its anchor atom uniformly.
Only present structure and availability of the fixed 12 ps observation interval
determine eligibility. Its future crystal labels are not inspected for selection.
These are random eligible liquid states of the matched ancestors, not a random
sample of every Al trajectory in the pool. The native 0.75 ps cadence is retained.

For embryo cohorts the anchor is the atom nearest the selected embryo's periodic
centroid. All future local-site scores use the advected position of this fixed
atom, never a future-selected center. Original roles, ancestry and raw precision
are preserved. References are remeasured with the same analysis, requiring every
positive seed to establish and every negative seed to dissolve in the saved data.

## Continuations and endpoints

Each of the 36 states receives one original-settings NPT continuation and four
450 K response-style BAOAB continuations: 180 new branches, each 12 ps long.
The NPT control has native float64 integration, float32 GPU force arithmetic,
source temperature, 3 fs, 0.3 ps thermostat, 3 ps zero-pressure isotropic
barostat, and momentum removal every 100 steps; missing bath state is reset.
BAOAB uses float32 Q, momenta, noise and forces, 1 fs and 100 fs friction;
the shared backend's scalar energy reduction is float64. No initial velocity
redraw, coordinate perturbation or relaxation occurs. Actual input Q,V originally
stored at float16 retain that provenance. Scientific label generation uses
native generated PTM labels and pre-archive float32 geometric observations.

The complete historical graph strictly before the restart is inherited.
PTM RMSD<=0.1 FCC/HCP/BCC atoms are connected within 3.6 Å, with minimum tracked
size four. Strong overlap requires four shared atoms and 25% of the smaller
component, with the original one-missing-observation handling. Establishment
requires >=64 atoms in three adjacent observations, a 1.5 ps persistence span.
The established ancestry/merge/interface/periodic-ambiguity exclusions are reused.

At 3, 6 and 12 ps, `any_new_nucleus` means an isolated establishment born
strictly after the restart and confirmed by that horizon. `local_new_nucleus`
additionally requires its birth centroid within 8 Å of the advected anchor.
`seed_established` instead requires its birth node to descend from the starting
embryo through the graph links; it is undefined for random-liquid anchors.
Nuclei first reaching 64 too close to an endpoint to confirm persistence are
not yet established at that endpoint. A zero is a finite-window outcome.

`crystal_fraction` is instantaneous full-cell PTM crystal fraction and
`largest_cluster_atoms` the largest tracked component at the endpoint.
`reached_10pct_crystal` asks whether the observed fraction ever reached 10%
up to that endpoint. Maximum fraction and component size use the full 12 ps.
These bulk observables include any growth; they do not redefine nucleus birth.

For seed fate, select the starting component sharing at least four atoms and
25% of the smaller seed/component with the original embryo. Follow every directed
descendant, including weak contacts, conservatively through interactions.
`seed_dissolved` requires this entire tracked descendant graph to disappear
and all terminal-leaf atom identities to be PTM liquid for the next three saved
observations (2.25 ps after the last tracked component). Incomplete tails and
residual crystal remain unresolved/right-censored; they are not relabeled as
dissolution. `seed_dissolved=0` therefore means no confirmed dissolution within
this window, not necessarily sustained growth. An unresolved initial seed has
undefined fate. The terminal-status field and `seed_established` disambiguate it.

`first_new_nucleus_ps` is the earliest qualifying birth time, blank if absent.
RMS positional divergence uses fractional-coordinate minimum images multiplied
by original box lengths; it removes homogeneous box deformation, not translation.

## Aggregation

Cohort rates average per-case fractions after every planned repetition of that
case has finished: one for NPT/saved reference, four for response450K. Case,
ancestor and branch denominators are explicit. Partial coverage is provisional.
Each cohort has one case per independent ancestor, so 2,000 case-bootstrap
draws give the displayed percentile intervals; intervals remain undefined with
one case. Four noise branches are not four independent ancestors.

Pairwise cohort differences use only completed matched blocks. Compute the
within-block difference of case means, average across blocks, and bootstrap
whole blocks 2,000 times. Blocks are ancestrally disjoint. Differences report
both their direction and number of matched blocks. Random-liquid seed metrics
are undefined and never included in seed-fate comparisons.

Saved reference outcomes are observational checks, not new stochastic branches.
They remain a separate arm. A common 16-hour deadline includes scheduler wait;
incomplete branches at the limit are checkpointed and excluded from completed
rates, with unfinished coverage explicit. No outcome-dependent extension or
sampling occurs. Force backend, source graphs, raw manifests, plans and metric
implementation hashes accompany exports. Other Al cadences and cohorts remain
outside this named campaign.
