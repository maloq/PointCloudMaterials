# Thirty additional Ni crystallization runs on GPU

Requested and launched October 12, 2026. This campaign adds 30 independent melts
to the existing 60 Ni sources, with five new sources at each of 950, 1000, 1050,
1100, 1125 and 1150 K. The temperature grid follows the latest low-temperature
campaign. Seeds, temperatures and roles were fixed before observing outcomes;
no liquid configurations, velocities or nuclei are inherited from prior sources.

**Current status:** all 30 simulations completed by 04:37 Europe/Paris on
October 12, with 73,780 saved frames and 7.375 ns total production history.
Twenty-four sources reached the half-crystalline-plus-10-ps stop; six reached
the 420-ps cooling-plus-hold limit. These are whole-cell stopping outcomes,
not counts of established nucleus births. Birth analysis is requeued after the
execution-bundle repair described below.

## Physical protocol

The [recipe](../../configs/simulation/ni_births_lowT_010_20261012.json) uses
32,000 Ni atoms and the unchanged Mendelev 2012 Ni1 EAM/FS potential. FastNi
CUDA supplies float32 forces; native LAMMPS integrates isotropic zero-pressure
NPT with float64 states and a 1-fs step. Thermostat/barostat damping remains
0.1/1 ps. Each source independently melts at 2300 K for 40 ps, passes the existing
liquid-quality gates, cools over 20 ps, and holds at its target for up to 400 ps.
The recorded production history is at most 420 ps including cooling. The
existing 50%-crystalline stop retains 10 ps follow-up, subject to that cap.

Native positions and velocities are saved every 0.10 ps from production
initialization. Verified archives retain float16 Q/V, float32 boxes, exact
timesteps and atom IDs, and conversion/checksum receipts. Native float64 NPT
restarts are written every 5 ps. Simulation staging is on SCRATCH; checkpoints
and verified completed trajectories publish to STORE. The maximum uncompressed
Q/V payload for 30 full-length histories is about 48.4 GB, excluding native
checkpoints and analysis caches. Space was checked before launch.

Source IDs use replicates 12–16 at every temperature. Roles are train,
selection, test, train, test, giving 12/6/12 sources. IDs and preparation seeds
are disjoint from all 60 prior sources. Existing cohort roles, evaluation
contracts and trained models remain unchanged; including this new collection
in a future experiment requires its own explicit data contract.

## Execution and ownership

Six detached GPU workers each receive five sources, with a ten-hour allocation
ceiling and checkpoint/publication reserves. GPU partitions are alternatives
in one request per worker, not duplicate reservations for different types.
All six workers obtained A40s. Historical source timings suggest approximately
3–6 hours for the simulation queue, depending on crystallinity stops and host
throughput; the ten-hour allocation is not a promise of ten hours of MD.

| Lane | Temperature K | GPU job | Initial node |
|---|---:|---:|---|
| 0 | 950 | 1034766 | node28 |
| 1 | 1000 | 1034767 | node30 |
| 2 | 1050 | 1034768 | node30 |
| 3 | 1100 | 1034769 | node38 |
| 4 | 1125 | 1034770 | node24 |
| 5 | 1150 | 1034771 | node25 |

The original dependent CPU search job 1034772, created in this launch, was
replaced while still pending by GPU extraction **1034773**, then CPU lineage
tracking **1034774**. No simulation worker or job from another chat was stopped
or changed. GPU extraction waits for all six MD workers; lineage waits for
successful extraction. This does not use or alter the user's node60 allocation.

Campaign identity:
`5d4a689eb39d938a4d171c81ea5a3592ebf6c9f36ba7c3c7b16bbab64e5d1353`.

- [Frozen source plan](../../output/ni_nucleation/campaign-lowT-20261012/technical/plan.json)
- [MD launch receipt](../../output/ni_nucleation/campaign-lowT-20261012/technical/launch.json)
- [Current dispatch](../../output/ni_nucleation/campaign-lowT-20261012/technical/dispatch.json)
- [Source independence and space preflight](../../output/ni_nucleation/campaign-lowT-20261012/technical/preflight.json)
- [GPU search launch](../../output/ni_nucleation/campaign-lowT-20261012/technical/gpu-search/launch.json)

Per-source `progress.json`, `outcome.json`, checkpoints and completion receipts
under the archive provide actual progress. `status-00.json` through
`status-05.json` summarize worker outcomes as sources finish. A no-birth source
is a valid observation; failed preparation and wall-time partial histories are
reported separately and are not counted as completed full-duration controls.

## Birth analysis and registered collections

The [analysis recipe](../../configs/analysis/ni_lowT_gpu_search_20261012.json)
uses GPU resident/voronoi PTM and bond-order extraction, followed by the existing
CPU chronological lineage tracker. It processes every verified saved frame.
The primary established-nucleus definition remains **64 connected atoms
sustained for 1.5 ps**. The 8/32/128-atom thresholds, detector controls, cooling
versus isothermal onset and zero-birth censoring remain explicit. A request for
30 runs does not imply 30 realized crystallization events.

- Raw dataset: `ni-births-lowT-010-20261012`.
- Derived search: `ni-lowT-20261012-gpu-full-v1`.
- Archive: `${storage:archive}/simulations/ni-births-lowT-010-20261012`.
- Search: `${storage:training_storage}/nuclei_search/ni-lowT-20261012-gpu-full-v1`.
- Analysis cache: `${storage:cache}/nuclei-search/ni-lowT-20261012-gpu-full-v1`.

Both collections are registered in `configs/datasets.json`. They add new
independent ancestry; detector records and later local windows are not extra
independent melts.

## Reused numerical evidence

The force and NPT producer are unchanged. A repository refactor simplified two
trajectory readers by removing checks guaranteed by the converters. The existing
two-frame 32,000-atom publication assay gives identical arrays, checksums and
periodic frame geometry with old/current readers. All trajectory writers and
composition functions are AST-identical; GPU descriptor kernels and ordering
retain their reviewed hashes. No new force-engine numerical campaign was run.

The [MD-reader review](../metrics/ni_validation_reader_review_20261012.json) and
[descriptor-reader review](../postprocessing_gpu_ni_reader_review_20261012.json)
bind the exact source changes and inherited validation. New execution bundles
freeze these receipts; historical exported metrics and definitions are intact.

Inspect current state with conda `pointnet-torch214`:

```bash
python -m src.simulation.campaigns.ni_births.workflow status --config configs/simulation/ni_births_lowT_010_20261012.json
```

This campaign is already submitted. Do not submit a duplicate queue or alter
its frozen recipe while sources are running.

## Analysis bundle recovery on October 12

The original GPU extraction job 1034773 failed before descriptor extraction:
its frozen bundle omitted `docs/postprocessing_gpu_ptm_20261008.json`, which
was still required by the inherited Ni metric contract. Its dependent lineage
job 1034774 consequently could not start. This was a packaging error; all
simulation archives and their completion receipts are intact.

At 11:43 Europe/Paris, extraction **1035586** and lineage **1035587** were
submitted from a separate recovery bundle. The missing audit was copied at
the exact hash in the original contract. Every scientific source and the
original metric contract remain unchanged, and the failed bundle/log are
preserved. Only the receipt-owned blocked lineage job was canceled. No MD was
rerun. The maintained launcher now copies every contract dependency into new
bundles so inherited audit files outside `docs/metrics/` are included.

- [Recovery and failure evidence](../../output/ni_nucleation/campaign-lowT-20261012/technical/gpu-search-recovery-01/recovery.json)
- [Frozen-bundle preflight](../../output/ni_nucleation/campaign-lowT-20261012/technical/gpu-search-recovery-01/preflight.json)
- [Current analysis jobs](../../output/ni_nucleation/campaign-lowT-20261012/technical/gpu-search-recovery-01/launch.json)

At requeue verification the GPU stage was waiting for resources; lineage
depends on successful extraction. Prior identical GPU extraction took about
1,774 seconds for 88,736 new frames, suggesting roughly 25–40 minutes for this
73,780-frame cohort once a GPU starts, excluding queue time and lineage work.
