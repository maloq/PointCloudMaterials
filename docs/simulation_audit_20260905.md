# Live simulation audit — 2026-09-05, 14:44 CEST

**Later recovery:** see [simulation_recovery_20260905.md](simulation_recovery_20260905.md).
The low-temperature chain has since been resubmitted, source 28 prepared for
retry, and the failed-extension acceptance index corrected. Do not repeat the
historical submission instructions below without inspecting the current queue.

Completed artifacts checked cleanly, but the source submission chain is not fully
healthy. This report supersedes the counts in the September 4 handoff. No
simulation jobs were cancelled or newly submitted during this audit.

## Slurm and the cleanup regression

- `981369_9` on nodecpu07 and `981369_10` on nodecpu06 are running 510 K
  independent-source histories. Both completed melting and are advancing in the
  600 ps measurement stage. Logs are updating, sampled temperatures are near
  510 K, and neither measurement log contains the checked LAMMPS error/NaN
  signatures. Both request 48 MPI ranks with six-hour limits.
- High-temperature controller `981370` is waiting on that array's `afterany`
  dependency; this is expected.
- Low-temperature array `981344` completed source indices 51–53, but controller
  `981345` failed to open the old launcher path removed by the script cleanup.
  Its stderr establishes the cause. The 400/450/500 K chain is stopped at 54/90;
  its stale root status still says `submitted`.
- Restored temporary forwarding entry points at both original source-launcher
  paths. Their `--help` checks pass. This also protects the already-queued high-
  temperature controller. No external Slurm batch file was rewritten.
- Source 28 at 510 K remains deliberately interrupted by the local overnight
  cutoff. Its partial artifacts are preserved and require explicit recovery;
  the normal source runner will refuse to overwrite them.
- Slurm accounting (`sacct`) is unavailable: connection to slurmdbd was refused.
  Queue inspection and filesystem artifacts supplied the evidence.
- Login-node SSH to `gpu-gw` was denied. The low-temperature chain was therefore
  not resubmitted from the current H100 allocation. Accelerator work was untouched.

Recover the stopped low-temperature chain **from an authorized login-node shell**
after confirming no new matching array/controller has appeared:

```bash
cd /home/infres/vmorozov/PointCloudMaterials
squeue -r -u vmorozov
conda run -n pointnet python scripts/run_lammps_campaign.py independent-meam-source \
  submit-next-wave \
  --campaign-root /home/ids/vmorozov/simulations/al_meam_independent_sources_70304_400-500K_30perT_float16_20260902 \
  --start-index 54
```

The existing submission function checks active conflicts and the ten-entry QOS
limit and preserves the configured three-task concurrency. It starts from the
first missing source, not from a completed run.

## Data inventory

| Independent-source temperature | Complete / planned |
| --- | ---: |
| 400 K | 30 / 30 |
| 450 K | 24 / 30 |
| 500 K | 0 / 30 |
| 510 K | 10 / 30 |
| 520 K | 3 / 30 |
| Total | 67 / 150 |

These are 600 ps measurement histories: 40.2 ns total, 53,667 stored frames,
70,304 atoms per frame, positions and velocities in float16 at 0.75 ps cadence.
Their binary directories total 45.33 GB. Melt and equilibration time are additional.
Completed-run mean temperatures differ from their targets by less than 0.25 K;
thermodynamic arrays are finite. Of these histories, 66/67 nucleated within the
measurement period. Candidate-band coverage is not proof of a balanced production
parent set, especially at 510/520 K where the broad 1–99-atom band is diagnostic.

The main old-parent shooting dataset now consists of:

- 480 existing fixed-48 ps paths;
- 160/160 new fixed-15 ps top-up paths, completed at 08:33 CEST on September 5;
- 640 unique futures across 40 parents and 20 root source lineages;
- exactly 16 futures per parent at the requested horizons up to 12 ps;
- 85,440 stored frames and 25.44 ns total branch integration;
- approximately 144.71 GB binary trajectory storage for these 640 paths.

The top-up alone has 8,160 frames and 14.90 GB of binary trajectories plus
restarts. Its 15 ps paths have 51 frames at 0.3 ps cadence, 70,304 atoms,
float32 positions/velocities, first-passage progress, and verified 24-rank
restart artifacts. Outcomes: 102 crystallization, 6 dissolution, 52 censored.
Its index contains 960 correlated 12 ps windows at 0.6 ps start stride.

Other existing data are separate: the 144-path fixed-24 ps compatible nested
pilot is auxiliary; its original variable-stop representation is not another
144 independent futures. The 40-path historical 48 ps partial top-up and smoke
campaigns are excluded from the main 640-path total.

The new 160-parent / 2,560-branch production root does not exist yet: 0/2,560
production futures are available. The completed top-up covers Part A of the
request; it does not satisfy Part B's new-parent diversity requirement.

All simulation directories together occupy about 290.8 GB allocated storage
(270.8 GiB), with about 305 GB apparent file sizes. This includes partial runs,
archives and historical products. The current user quota could not be queried;
the handoff's 512 GB quota is historical, not newly verified.

## Scientific limitations and next work

1. Exact 15-to-24 ps continuation remains blocked by the failed comparator.
   Short paths remain valid; no exact-extension claim is justified. The explicit
   short-path exception in `extension_gate.json` remains in force.
2. Finish the source pool and calibrate high-temperature basin/transition
   interfaces before selecting the requested split-safe liquid, crystal and
   transition parents. Keep final-validation selection independent of futures.
3. Source histories use their configured 0.75 ps cadence. The final parent-history
   requirement is 0.3 ps with at least 12 ps prehistory; that finer history still
   needs generation under the production parent protocol.
4. Confirm storage quota before main production. Do not merge the historical
   partial top-up, smoke paths or duplicate pilot representations into the main
   conditional-law index.

## Verification scope

The read-only diagnostic under `output/simulation_audit_20260905/` contains its
code and JSON report. It checked every currently complete independent-source
binary (67) and every new top-up binary (160), including all stored array
checksums, dtype/atom/frame/timestep contracts, source input hashes, progress and
thermodynamic artifact hashes, and required restart files. Top-up restart hashes
were also recomputed. High-temperature and top-up manifest hashes matched their
sidecars; the original low-temperature producer has no manifest SHA sidecar.

A separate merged-index check verified 640 distinct velocity/thermostat seed
pairs, shot indices 0–15 for every parent, complete referenced outcomes, and no
root source lineage spanning train/validation splits. Legacy 48 ps trajectory
arrays were not all rehashed in this audit. The two running partial trajectories
were assessed through their current logs; they are not counted as complete data.
