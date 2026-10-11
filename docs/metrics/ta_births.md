# Ta full-cell births and nucleus fates (v1)

Label-side search over **30 complete 24 ps Ta trajectories**: six earlier runs
and four additional velocity branches from each of the same six archived parents.
`model_1m` contains 1,024,000 atoms; each of the five other parents contains
10,000,422 atoms. All descendants use the registered Zhong2014 EAM potential.
The archived positions' generating potential and common preparation are unknown;
all 30 futures retain one conservative preparation-ancestry group. There is no
new simulation, relaxation, predictor fitting, train/test split or W&B run.

## Observations and reuse

Verify the actual 2 fs integration steps and native 0.10 ps saved timestamps.
Select **exact 0.50 ps** observations for every source (49 frames, 0–24 ps).
This preserves the established September26 Ta search grid and permits reuse of
its six verified full-cell PTM/ancestry artifacts. It is a coarser protocol than
the Al shooting search at 0.30 ps and can miss briefer episodes. No interpolation,
mixed-cadence pooling or fabricated dense history is used. Static Ta structures
alone cannot supply event trajectories; the six parents enter through their MD.

Bind original manifests, actual array schemas, parent hashes, distinct velocity
seeds, outcome receipts and potential hashes. Workers verify consumed raw array
checksums. Reused PTM chunks and graphs retain their original hashes/identities
and source paths; reuse requires exactly matching timelines, manifest identities
and classifier settings. New PTM is checkpointed one full frame at a time.
Threading changes execution only. Raw float16 position quantization and float32
boxes are preserved; available quantization receipts remain in the source plan.

## Definitions

Reuse the [Al shooting fate rules](shooting_births.md), including the conservative
all-overlap exclusion of establishment-connected failed episodes and directed
ancestry back to the starting frame. Numerical spatial support is the **existing
Ta** contract: connectivity **3.70496348320465 Å**, local birth/interface distance
**8.233252184899222 Å**, derived from the frozen Ta/Al length-scale ratio. These
are engineering criteria from the previous audit, not validated critical sizes.

PTM FCC/HCP/BCC with RMSD cutoff0.1 defines crystal. This assay therefore does
not detect every possible ordered Ta structure; unclassified/non-FCC/HCP/BCC
atoms are operationally noncrystalline here. Four atoms is the tracking minimum;
strong temporal overlap needs four shared atoms and 25% of the smaller component.
One missing observation can bridge ancestry but cannot satisfy a streak.

- Establishment: ≥64 atoms in four adjacent strong-linked observations, **1.5 ps**.
- Brief failed candidate: peak≥8, isolated single temporal root, connected strong
  lineage, no graph connection to any establishment, verified disappearance.
- Sustained failed candidate: the above plus≥8 atoms in three adjacent observations,
  **1.0 ps**, the minimum grid span meeting the declared≥0.75 ps requirement.
- Disappearance: no outgoing descendant; every terminal-branch core atom must
  be PTM-noncrystalline at five successive observations after the final tracked
  frame, **2.5 ps** of follow-up (minimum grid duration≥2.25 ps).
- Growth: reach `max(128, ceil(2*size_at_confirmation))` for four adjacent
  observations strictly after establishment confirmation and before merger,
  **1.5 ps**. Growth and terminal status are separate columns.

The sustained failed group is a subset of the brief-allowed group, never an
additional set. Peaks4–7 remain inventoried without disappearance verification.
Missing follow-up is censored; residual crystal and uncertain origins are explicit.
Parent-present subcritical embryos that establish later are
`parent_present_embryo_growth`, not new births. `establishment_kind` preserves
the underlying classifier output; `initial_cluster_ancestry` records the extra
parent-ancestry check. The check cannot reconstruct pre-observation history or
sub-four-atom ancestry. Candidate classifier histories still require the strict
crystal-free input eligibility pass.

## Tables and interpretation

`branches` reports source/campaign/parent identity, atom/frame counts, reuse and
actual physical persistence/follow-up durations. `established-fates` and
`transient-episodes` use the same per-event fields as the Al shooting audit, with
`material=Ta` and the actual generating potential. Times are relative to the
start of each MD future. `counts` groups by parent configuration, population and
label and reports distinct events, futures, parent hashes and ancestry groups.
Per-future tables distinguish the earlier run and all four new shots.

Counts are correlated finite-observation availability, not independent nuclei,
nucleation rates or evidence of a universal critical size. Temperature and time
are label/audit metadata, never model inputs. Original Al and historical Ta
results retain their original definitions and are not rewritten.
