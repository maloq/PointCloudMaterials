# Research retained on September 13, 2026

Review window: **September 11–13 inclusive**, based on completed scientific work
and active research, not modification time. Folder dates mark when work began.
Older records/outputs are recoverable through [the STORE archive](archived_research.md).
Cleanup branch: `codex/research-retention-20260913`.

| Research retained | Evidence in this window | Necessary state |
| --- | --- | --- |
| [Embedding forecasts](../experiments/embedding_forecast_20260911/README.md) | Pilot/autoregressive comparisons and two completed enlarged 32-epoch fits | Pilot fits, large AR/direct best and last checkpoints, original scaler/config/source, paired errors; recovery/handoff evidence |
| [History-length comparison](../experiments/forecast_context_20260913/RESULTS.md) | 24 completed direct/AR fits, six contexts × two seeds × two methods | All 24 selected fits and their paired test errors; shared window/normalization identity |
| [Local crystallization](../experiments/forecast_crystallization_20260913/RESULTS.md) | Completed local transition/timing assay | Frozen AR/direct checkpoint copies, selected atom IDs, PTM labels, ridge probe, validation thresholds, score arrays |
| [Spatial and probabilistic forecasts](../experiments/forecast_spatial_mixture_20260913/README.md) | Active 12-fit ablation and initial matched physical assays | Entire active fit/analysis trees, best/last state, neighbor index, source snapshots, local assay and normalization reference |
| [Short-history extension](../experiments/forecast_spatial_mixture_20260913/SHORT_HISTORY.md) | Queued 14 new fits; six existing fits supply references, 20 models in the comparison | Entire active queue and future fit paths, original six reference fits/assays; no completed short-history result claimed |
| [Relaxed MACE/VICReg](../experiments/mace_vicreg_relaxed_20260910/RESULTS_20260911.md) | September 11 review of 24 fits; topology/history and checkpoint-selection conclusions | 24 selected encoders, original analyses/probes, training logs and metric definitions; the seed-20260910 single-frame encoder defines all forecast targets |
| [Full-data MACE](../experiments/mace_vicreg_full_20260910/README.md) | September 11 analysis/gallery, reference in the MACE comparison | Selected epoch-9 encoder, training/config/source evidence and full-data view cache |
| [Al denoising](../experiments/mace_al_denoising_20260910/README.md) | Older prerequisite, not newly claimed September 11–13 training | Independent-source selection, FIRE relaxed targets and pretrained MACE weights consumed by retained research; denoising comparison evidence |

## Checkpoints and dependencies

The [checkpoint table](research_checkpoints.csv) lists observed selected model paths;
[technical hashes](technical/research_checkpoints_20260913.json) verify their identity.
Live fits are protected as directories, so new or changing checkpoints are retained
without treating a transient file observation as a completed result. The
[active fit path table](active_research_fits.csv) names all 26 new/current fits
and the six reused references; paths for queued fits are reserved, not claims
that checkpoints already exist.

- Frozen target encoder: `anchor_vicreg__rep01`, seed **20260910**, epoch **2**,
  under `output/mace_vicreg_relaxed_20260910/runs/`. This is essential even though
  its original fit predates the review window.
- Enlarged forecasters: `output/embedding_forecast_20260911/scale/runs/`,
  `path_ar_large_aug-seed20260911` (selected epoch **5**) and
  `path_direct_large_aug-seed20260911` (selected epoch **11**). Both producer status
  files report complete. Keep best/last plus scalers and the exact source transition.
- The 24 MACE comparison fits span local retained output and
  `/store/PERSO/vmorozov/experiments/mace/`. The full-data reference is the 25th
  gallery checkpoint. These external checkpoints and analysis artifacts are retained.
- `output/pretrained_mace_spatiotemporal_20260906/mace_mp_0b2_small.model` remains:
  it is the pretrained model read by the current denoising producer. Other artifacts
  in that older directory can be archived.
- Keep IDS caches `embedding-forecast-full-20260911`, `embedding-forecast-20260911`
  when present, `mace-meam`, `mace-full`, and the pilot cache; they encode the retained
  producer/source/split identities. This cleanup does not prune external caches.
- Keep WORK simulation sources, original datasets and restarts. Legacy simulations
  under `output/synthetic_data` are archived byte-for-byte with a compatibility
  link; they are not treated as disposable analysis output.

The enlarged fits each have about **4.52 GiB of paired test errors**. These are the
largest retained local files and support matched source-level comparisons; removing
them would lose analysis inputs. The history sweep similarly retains about 4 GiB
of paired scores across its 24 fits. No metric was recomputed or relabeled by cleanup.

## Active execution and removal boundary

At review, Slurm allocations 990987 (nodesumo01), 991149 (node53) and 991141
(node51) were running. Their processes were inspected. The first two contain the
spatial/mixture controllers and the short-history waiter. No simulation controller
was active. Active plans, exact-resume configurations, frozen source and external
batch files remain untouched.

Retained output roots are `embedding_forecast`, `embedding_forecast_20260911`,
`mace`, `mace_vicreg_relaxed_20260910`, `mace_vicreg_full_20260910` and
`mace_al_denoising_20260910`, plus the required pretrained model and dataset aliases.
`registry` and `maintenance` hold generated indexes and cleanup receipts.
Within the forecast family, the superseded failed global-onset assay and disposable
scale preflight are archived; the completed local assay is retained.

The other older local output trees and the legacy `outputs/` root are retired
only after comparing their exact file hashes/link targets with STORE. Changed files
receive a new verified copy. Research source stays versioned in `src/`; archived
records/configurations and the old checkout remain available for historical replay.
Operational and simulation records moved to `docs/`, rather than remaining as
experiment folders. [The move map](technical/research_retention_moves_20260913.json)
records their destinations.

Verification and applied-removal receipts are linked in [the archive note](archived_research.md).

## Completed cleanup and validation

Local removal completed after source/archive verification: **39.91 GiB allocated**
(**47.42 GiB logical**) across 79 selected output items, including the legacy
`outputs/` root. Seven research directories remain. The original dataset alias
`output/synthetic_data` resolves to its verified STORE copy. No external training
cache, WORK simulation input or active forecast tree was deleted.

Verified 68 retained selected model/checkpoint copies and preserved all 49 forecast
JSON configurations byte-for-byte. The active-fit table protects 26 fit directories
plus six reused references, including future checkpoints from queued training.
All current documentation links resolve. The focused portability, inventory,
registry, simulation, workflow and research-layout checks passed (37 tests across
the initial and corrected runs); all seven metric documentation contracts passed.
Moved simulation entry points and archived configuration lookups were checked.

New files are operational documentation/indexes, the maintained simulation-inventory
implementation and a regression fixture. Cleanup plans/receipts are generated
operational artifacts under `output/maintenance/retention-20260913/technical` and
STORE. No new scientific experiment was created by this cleanup.
