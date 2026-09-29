# External crystallization-origin counts

This is an exploratory extension of [Al64 ancestry](crystallization_origin.md),
not a new benchmark split or a pooled cross-material performance comparison.
The shared producers `ancestry.components`, `establish`, `onset_roots` and
`classify` retain the same operational taxonomy and strong/possible ancestry
distinction. `external_data.py` supplies explicit trajectories, normalized
length criteria and verified time grids; `external_audit.py` computes counts.

## Population and chronology

The sealed structural-source catalogue selects 26 dynamic records: seven Al,
six Mg, seven Ti and six Ta. The Al records include one continuous million-atom
melt/measurement history. Their shared endpoint is verified by atom IDs,
periodic boxes and minimum-image coordinate discrepancy, and retained once.
Full-cell ancestry spans both phases. Al forecast risk and first local onset
begin at the shared measurement-start frame: the deliberately crystalline
initial state before melting must not permanently exclude later recrystallization
outcomes. The event catalogue labels preparation versus measurement births;
full-cell counts include both, while prediction windows cover measurement only.
Other archived branches keep their recorded shared preparation ancestry.
Branch-local cluster counts are not independent-event counts across branches;
both contributing branches and distinct recorded ancestry groups are exported.
Zr static configurations have no temporal labels and are explicitly excluded.

All selected raw histories have 0.1 ps saved cadence. The audit covers their
entire duration at **0.5 ps**, by selecting every fifth observation. No positions,
boxes or labels are interpolated. The grid is finer than the Al64 0.75 ps grid,
but it does not resolve excursions between retained observations.

The primary size criterion is 64 atoms for **1.5 ps**, requiring four consecutive
0.5 ps observations. Sensitivities use 32 and 128 atoms for 1.5 ps, or 64 atoms
for 3 ps (seven observations). Local first onset and preceding liquid history
also span 1.5 ps. One missing observation can be bridged in ancestry, but never
extends a persistence streak. This represents a different physical gap duration
from Al64 and is recorded rather than silently equated.

## Geometry, centers and controls

PTM uses original observed coordinates, periodic boxes, RMSD cutoff 0.1 and
FCC/HCP/BCC types. Actual coordinate quantization metadata is retained per
source. The 3.6 Å connectivity and 8 Å birth/interface criteria are expressed
in Al-equivalent units, using the existing training-calibrated material scales:
`radius_A(material) = radius_A(Al) * scale_material / scale_Al`. This is fixed
preprocessing, not fitting a cutoff from these outcomes. Effective native-Å
cutoffs are recorded per source. Grain orientations are not part of connectivity.

The audit reuses the structural release's outcome-blind 1% atom-identity samples:
10,000 centers for a million atoms; 10,486 for 1,048,576 atoms; 1,000 for Ti's
100,000 atoms; 100,004 for the 10,000,422-atom Ta branches. A reproducible random
64-center subset provides a nested sparse-coverage control. For the continuous
Al history, measurement-phase centers are tracked back through the melt.
Sampling never targets known birth locations. Material, potential and times
remain audit/label metadata; no encoder, predictor or training input is produced.

## Counts and denominators

`distinct_clusters` counts operational full-cell establishments by kind.
`distinct_clusters_covered` counts each establishment once if its birth centroid
is within the declared radius of any sampled center. Sampled1pct and baseline64
are separate coverage tracks. Left-censored existing crystals, interface-
associated candidates and unresolved establishment geometry/merges remain separate.
These are candidate establishments, not physically validated critical nuclei.

Forecast origins occur at every retained observation with sufficient preceding
liquid history and full 6 ps follow-up plus onset confirmation. The same origin
population serves 3 and 6 ps. A center is at risk only before its first sustained
onset and while its recent observations are liquid. `onset_windows` partitions
this population into the shared origin labels and `no_onset_within_horizon`.
`distinct_onsets_represented` deduplicates center identities within each label
and horizon; origin-relative labels can differ, so do not add these across labels.
`regional_birth_windows` counts windows with an isolated birth in the center's
local region, independently of whether that center itself transforms. It still
uses the historical-style center-liquid risk set and can omit subcritical ordered
precursors. Full-cell birth counts therefore provide a different availability
measure from the sampled forecast windows.

The plan's source identity, center ID and retained origin index determine each
row. `positive_windows.npz` stores all rows positive within 6 ps, their two
horizon labels, all center onset frames, the full origin grid and regional-birth
positive indices. The graph retains every sampled center's PTM label, allowing
the full risk population and negative denominator to be reconstructed. Sparse
exports do not remove negatives from any reported metric count.

Tables group by material **and potential**, threshold, sampling track, horizon
and unit. They never combine the native fixed Al64 windows with this denser new
population. `branches_with_label` and `ancestry_groups_with_label` describe
provenance coverage, not a claim that the latter equals a validated number of
independent nucleation experiments. Partial reports include completed records
only; pending records are not treated as zeros.
