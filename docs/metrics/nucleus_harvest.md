# Nucleus harvest: training-source availability and precursor diagnostics

Producer: src/research/crystallization_origin/harvest.py, using the completed
Al64 ancestry/PTM producers and periodic bonds passed to the existing
geoframe_evolution.reference.bond_descriptors. No model performance is measured.

## Population

Only the fixed 90 training sources are opened. Selection/calibration/test are
not harvested. Original release labels and samples remain unchanged. Outputs
are an exploratory analysis release, **not a finalized training or evaluation
population**. No temperature, age or absolute-time model input exists.

For each original establishment criterion, enumerate all atoms within 8 Å of
an isolated birth centroid at birth at the eight preceding 0.75–6 ps origins.
The target region follows the atom. A prospective input uses its actual position
at the origin, never future membership or a future centroid. Deduplicate
source/criterion/atom/origin rows; preserve per-event associations.

Require origin + 6 ps + **3 ps confirmation**, the longest sensitivity, so all
four criteria share complete follow-up. Retain excluded late rows as censored,
not negatives. History tables for 0/3/6/12 ps are nested availability subsets;
early births remain in shorter-history tables. Negative origins are omitted.

Uniform controls draw 24 distinct valid origin frames and 128 distinct atom
IDs per frame, independent of outcomes. Exact row inclusion probability is
(24 / valid_frames) * (128 / atoms). These are training-source diagnostics,
not a test sample, and may have positive or competing-event outcomes.

## Causal eligibility and first-event screen

At the origin exclude centers within 8 Å of an atom in a current component at
the size threshold, or a lineage already confirmed by that origin. Possible
ancestry counts conservatively as prior contact. Allow small ordered components
and PTM-crystalline centers. Large unconfirmed transients are excluded by the
size rule; retain both overlapping exclusion flags.

Gate retrospective roots by **confirmation time**, never backfilled birth time.
Recompute ancestry on actual trajectory prefixes at zero, the last control
origin, and (when available) just before the first non-left-censored birth and
at confirmation. Current-frame confirmed membership must match the gated
full-trajectory result. Disagreement fails with context. This is a scientific
causal-invariance check on production data.

For eligible rows compare the first future local establishment to the first
future observed contact with an already confirmed lineage:

- isolated_birth_first: isolated local birth occurs first;
- confirmed_lineage_contact_first: confirmed contact occurs first;
- other_establishment_first: non-isolated local establishment occurs first;
- simultaneous_or_ambiguous: tied contact/birth, multiple equal-frame births,
  or an unresolved establishment;
- none_within_horizon: none within the fully observed 3/6 ps horizon.

This screen is not a validated grain-front label. Contact uses observed
confirmation, so an external new cluster can exist before contact qualifies.
Frame-level contact and near_large arrays preserve this distinction. Later
contact does not erase an earlier birth. Subthreshold embryos that never
establish can remain no-event controls. Ineligible rows have code **-1**, not zero.

## Units and capped examples

availability.csv groups criterion/history/horizon/track/label. Count means
unique atom-origin rows; distinct_source_births counts source/event IDs for
isolated-birth rows; sources_with_rows counts contributing sources. Never sum
overlapping histories, horizons, criteria or tracks. Pending sources are absent.

events.csv retains candidate centers, enumerated/complete/eligible rows,
overlapping exclusions, earlier competing events, single-lineage peak size and
merging, and availability for every history/horizon. Keep zero-yield events.

Select up to four centers per birth among eligible first-birth windows with
12 ps history, retaining all their eligible enumerated origins. Save the
conditional inclusion chance min(4, eligible_centers) / eligible_centers.
Ownership is the first event; ties are excluded. This is conditional on the
event-centered pool, **not** a population training weight. Uniform controls and
cases overlap; a future release must account for union probabilities and
finalize control strata. Capped crops do not multiply independent nuclei.

## Structural quality and visual review

For all selected primary examples and up to 256 uniformly sampled eligible
controls per source, compute existing q4, q6, normalized w4/w6, averaged q6,
mean q6 coherence and nearest-12 density. Neighbors of neighbors use periodic
minimum-image bonds. Keep coherent-bond counts at .65/.70/.75. These descriptors
introduce no acceptance threshold.

Five predeclared training sources supply review crops at -6/-3/-.75/0/+1.5/+6 ps
relative to their first isolated primary birth. These are **future-birth-centered
visualizations only**, never predictor inputs. Figures show a central 10 Å slab;
NPZs retain the 24 Å sphere, atom IDs, PTM types and component sizes.

Original float16 MD, float64 periodic geometry, source manifests, graph/catalogue
hashes and verified PTM chunks are preserved. New exports contain references
and audit descriptors, not species channels, relaxed labels or simulations.
Implementation hashes accompany each CSV export.


## Mechanism queue extension (2026-09-26)

Version2 applies the sealed training definition to explicit selection/calibration/test roles and records8/32 Å established/PTM-clear masks. Historical training artifacts remain unchanged. See encoder_mechanisms.md for the new uniform-support table.
