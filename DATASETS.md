# Datasets: start here

**Atomistic response-training collection:** [workflow](docs/response_atlas.md#atomistic-response-training)
collects fresh fixed-MACE values/responses for56 full-cell Al256 synthetic
configurations, with32/8/16 train/selection/test roles and shared FCC ancestry.

**Value-only predictive baseline:** [workflow](docs/predictive_baseline.md) derives a fixed
274-coordinate target from the sealed Al480 local assay; all historical rows and
source roles are retained. No new MD or Al64 resplit.

**Response-atlas numerical pilot:** [workflow](docs/response_atlas.md) reuses Al480 outcomes and adds a separately registered fixed-MACE Al256 perturbed-FCC response collection.


**Shooting future-law assay:** [workflow](docs/shooting_laws.md) reuses the four
compatible 48 ps Al campaigns (40 parents, 480 futures), with separate CSLD,
nested-Al and Ta diagnostic collections. No new simulations or Al64 resplit.

**Pre-appearance birth classification:** [workflow](docs/birth_prediction.md)
derives matched crystal-free histories and continuously liquid controls from
the completed original Al64 birth catalogue, preserving all source roles.
This event-enriched track is distinct from the historical all64 window benchmark.

**Spatial VICReg mechanism cache:** [workflow](docs/spatial_vicreg_bias.md) derives
128-atom parents, exact 80-atom views and physical/TDA references at fixed Al64
anchors. No resplit, future labels or crystal-based encoder-training filter.

**Multimaterial rich-descriptor training (prepared cache complete):** [workflow](docs/rich_multimaterial_encoder.md) derives 442 local targets from the raw Al/Mg/Ti/Ta pool; a fixed 1,056,768-patch subset gets 60 epochs with correlation/angular order 3 and batch8192.


**RD-MACE256 rich-descriptor training:** [workflow](docs/rich_descriptor_encoder.md) reuses all 183,596 unrelaxed training contexts and the exact existing 3,536-feature bank; no new data or source split.

**Liquid sensitivity and paired archived relaxation:** [workflow](docs/liquid_controls.md) freezes the available raw/relaxed intersection, with both original and recomputed relaxed crystal labels. Synthetic labels are separately identified.


**Rich liquid descriptors:** [CPU derivation](docs/liquid_descriptors.md) reuses the sealed
Al64 crystal-free observations for TDA, CNA, bond-order and geometric feature banks.
No new trajectories or source split.


**Liquid predictability descriptors and clearance:** [study workflow](docs/liquid_predictability.md)
derives geometry-only summaries and label-side observation clearance from the sealed
dense Al64 cache. No new trajectories or source split; CPU preparation is resumable.

**Strict liquid-only distance view:** [LCD-MACE128-VC](docs/crystal_liquid_distance.md)
reuses the dense Al64 context cache, retaining only liquid queries with no
established crystal in any observed patch for fitting. A crystal must exist
elsewhere for localization; crystal-absent cases are a separate evaluation.

**Dense interface-invisible contexts:** [256 centers × 64 frames × 150 existing Al sources](docs/crystal_interface_unseen.md)
proposed 2,457,600 extra contexts, retaining new geometry only when all input
patches exclude the interface. The sealed cache contains 1,133,196 contexts; original rows and
source/ancestor roles remain intact. Added centers are not independent sources.

**Crystal-interface targets:** [CIV-MACE128](docs/crystal_interface.md) retains fixed
Al64 rows and adds uniform held-out centers for interior evaluation; derived from
existing trajectories and PTM lineages, with no new simulation or source split.

**New Al 0.1 ps production:** [150 prepared-liquid reruns with velocities](docs/simulations/al_main_010ps_20260927/README.md)
retain the original melt ancestry and frozen roles; 2 fs integration, 600 ps measurement.
These are new descendants, not a replacement of the frozen 0.75 ps benchmark.
The [October 1 continuation](docs/simulations/al_main_010ps_20260927/README.md#october-1-continuation)
keeps the 20 completed descendants, recovers three native restarts and queues the
127 unstarted sources with the same ancestor roles.
**Additional 0.01-ps Al observations:** [two full 600-ps descendants](docs/simulations/al_main_001ps_20261001/README.md)
reuse source 886 (400 K) and 1004 (520 K), both train ancestors, with velocities.
They do not add independent melt lineages or alter the 0.1-ps source contract.

**Material-specific distance adaptation:** the [Al/Ta fine-tuning protocol](docs/distance_encoder_material_finetune.md)
retains fixed Al64 roles and uses newly simulated Ta velocity branches from a known
parent for a separately labeled conditional-generalization assay.

**Six-frame MD distance training:** [nominal 0.75-ps and exact 0.10-ps experiments](docs/distance_encoder_history.md)
record actual spacing and separate fixed-Al versus external-ancestry evaluation contracts.

**Regional emergence:** [training-source harvest audit](docs/nucleus_harvest.md)
reuses existing Al trajectories for precursor eligibility, event windows and
control diagnostics. This is an analysis collection, not a released benchmark.

**New Al comparisons:** [fixed 64-center benchmark and large structural-pretraining
sets](docs/datasets/fixed_al64.md), with frozen 90/15/15/30 source roles and an exact
historical 16-center comparison track.

**Completed structural expansion:** [256 native centers and proportional
multi-material sampling](docs/datasets/structural_multimaterial_256.md), preserving
the fixed prediction benchmark.

**[Open the searchable dataset registry](docs/datasets/index.html)** ·
[Browse the Markdown cards](docs/datasets/README.md) ·
[Download CSV](docs/datasets/datasets.csv) · [Registry JSON](docs/datasets/registry.json)

**[Potential files, hashes and provenance](docs/datasets/potentials.md)**

The registry covers **Al, Mg, Ti, Ta, Zr and Al–Ni structures**: raw dynamics,
static structures, synthetic geometry, training caches, physical targets and
potential files. Filter by material, potential or use classification. Each card
links to producer records, array schemas, hashes and known provenance.

Read the classification before using a dataset. A complete binary is not
necessarily an independent trajectory, an accepted protocol or a training input.
Duplicate exports, incomplete preparations, removed Zr dynamics, shared ancestry,
metadata gaps and reported H200 copies remain visible.

Refresh from the repository root in `pointnet`:

```bash
python scripts/project.py datasets --refresh
```

This reads metadata, potential files, array headers and filesystem sizes. It does
not alter datasets, load full trajectories, submit jobs or fit models. Pages show
an observation at the displayed timestamp, not live job status. Unregistered
directories are listed for review.

**Register new data in [configs/datasets.json](configs/datasets.json)**, including
storage role, material, generating potential, provenance and ancestry when known.
Keep unknown values explicit. See [the registry guide](docs/datasets/GUIDE.md).

Response-atlas follow-up (2026-10-01): stronger toy controls and fixed-parent shot precision; see `experiments/response_atlas_20261001/FOLLOWUP.md` and `docs/response_atlas.md`.
