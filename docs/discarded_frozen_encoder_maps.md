# Discarded frozen-encoder representation maps

On 16 September 2026, the user discarded training replacement local-state
embeddings on top of a frozen encoder. The active implementation, entry point,
five recipes and tests specific to that approach were removed. This is a scoped
retirement, not a repository reset.

**Embedding forecasting remains active and unchanged.** Native MACE and
coordinate/velocity encoder training, physical diagnostic readouts and their
checkpoints also remain. Auxiliary physical heads and motion directions used to
train MACE itself are part of native encoder training and are retained.

## Retained scientific records

| Discarded protocol | Evidence retained |
| --- | --- |
| [Frozen local-group states](../experiments/mace_local_state_20260915/README.md) | Affine temporal coordinates, learned physical distance and uncertain state discovery; completed comparison |
| [Frozen-state smoothness](../experiments/mace_local_smooth_20260915/README.md) | First 32-fit comparison completed with no accepted candidate; 22 capacity fits saved, final capacity evaluation not established |
| [Consecutive local motion](../experiments/mace_local_motion_20260916/README.md) | All 44 fits and evaluation completed; no candidate passed the information gate or joint 0.10 jump requirement |

These results remain evidence about the tested frozen-map protocols. Discarding
the implementation does not prove that every possible version of the idea fails.
Exact recipes now live in each dated experiment's `configs/` directory; commands
in those records are historical commands for the archived source.

All result directories, checkpoints, optimizer states, cached data, plots, tables
and exported metric definitions remain in their original locations. No experiment
output or cache was removed. In particular, the consecutive-sequence cache remains
a label/provenance dependency of native encoder data preparation.

## Exact source and verification

The pre-retirement snapshot at Git revision `8faa165` is
[source-before-retirement.tar.gz](/store/PERSO/vmorozov/projects/PointCloudMaterials-frozen-map-retirement-20260916/source-before-retirement.tar.gz).
It preserves `src/`, `configs/`, `scripts/`, `tests/`, `docs/`, `experiments/`,
`AGENTS.md` and `requirements.txt`, including original recipe paths.
[Source verification](/store/PERSO/vmorozov/projects/PointCloudMaterials-frozen-map-retirement-20260916/source-verification.json)
records archive and file hashes. Restore it into a separate checkout for historical
reproduction; do not overwrite the maintained checkout or original run artifacts.

[Original result hashes](/store/PERSO/vmorozov/projects/PointCloudMaterials-frozen-map-retirement-20260916/results-before.json)
cover the three discarded protocols, their capacity/smoke runs and the completed
native data-amount study.
[Protected forecasting source hashes](/store/PERSO/vmorozov/projects/PointCloudMaterials-frozen-map-retirement-20260916/forecast-protected.json)
record the forecasting files preserved during this change.
The discarded metric families' exact contracts are saved in
[discarded-metric-contracts.json](/store/PERSO/vmorozov/projects/PointCloudMaterials-frozen-map-retirement-20260916/discarded-metric-contracts.json).
Their historical metric documents remain readable; they are no longer registered
as active implementations. Original run exports are immutable.

## Shared native dependencies

Only shared physical observables remain in `src/research/mace_local_state/`:
`physics.py` is byte unchanged. Pure finite-time differences and subspace
operations moved to `src/research/mace_velocity/motion.py`; verified trajectory
reading and atomic NPZ writing moved to `sequence_data.py` in the same package.
Their function bodies are unchanged. Native encoder imports and implementation
hashes follow these moves; model architecture, losses and checkpoint format do not
change. Prepared native caches can still be loaded. Exact preparation/resume that
checks the old implementation hash requires the archived source; new preparation
must use a new cache identity rather than rewriting historical manifests.

## Completed checks

[Final verification receipt](/store/PERSO/vmorozov/projects/PointCloudMaterials-frozen-map-retirement-20260916/retirement-verification.json):
18 focused tests passed, all 14 active metric families verified, 203 protected
forecasting files unchanged, and all 1,198 result files/links across the six checked
run directories unchanged. Five shared helper function bodies and all five
historical recipes match the archived originals exactly.

The native CLI remains available. A completed native checkpoint loaded on the
allocated H100 and replayed eight atom groups against stored 304-channel embeddings;
maximum absolute difference was 1.43e-5, within the existing 2e-5 absolute/relative
comparison tolerance. The check wrote only to the retirement archive.
