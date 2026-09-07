# Simulation recovery — 2026-09-05

This supersedes the operational recovery instructions in the 14:44 CEST
[audit](simulation_audit_20260905.md). The prior audit remains the detailed data
inventory. Recovery was authorized by the user and submitted from `gpu-gw`.

## Source production resumed

- Restored the two legacy source launchers required by already-generated Slurm
  controllers. Keep them until those chains finish.
- Resubmitted the low-temperature campaign from first missing index 54 through
  the maintained `run_lammps_campaign.py independent-meam-source submit-next-wave`
  command. Array **981561**, indices **54–56**, concurrency 3; successor
  controller **981562**. Source 54 is running on nodecpu03; 55–56 await resources.
- High-temperature sources **981369_9** and **981369_10** remain running on
  nodecpu07/nodecpu06; controller **981370** awaits their array. Concurrency 2.
- Confirmed interrupted source 28's old PID is absent on lamedell11. Archived
  its partial attempt (3,946,859,820 bytes) by same-filesystem moves under its
  run directory, in `interrupted_attempt_local_cutoff_retry_20260905T133035Z`.
  Original inputs and metadata are retained, with copies in the archive.
  The high-temperature root's `source_028_recovery.json` records the recovery.
  Its normal sequential chain will rerun index 28 from the original seeds;
  no duplicate job was submitted.
- No completed trajectory was overwritten. GPU allocation 981505 is untouched.

These are active multi-hour/multi-day chains, not completed campaigns. At the
post-recovery queue check, the completed-data inventory remains 67 independent
sources (54 low-temperature, 13 high-temperature) and 160/160 completed 15 ps
top-up branches. The merged shooting collection contains 640 futures, including
the older 480 fixed-48 ps branches. Login quota reported **295.4 GiB / 512 GiB**
for the entire IDS home, before the additional diagnostic outputs; this is not
the size of just accepted simulation data.

## Invalid extension acceptance fixed

The extension code previously published a complete extension index before the
smoke comparison passed. Ordinary extension calls now require a passing exact
continuation proof and, for non-smoke campaigns, explicit permission in the
campaign's extension gate. Smoke candidates enter the accepted index only after
the comparison passes. Accepted records include the proof path and SHA-256.

The existing failed smoke's misleading one-branch accepted index was backed up
under `restart_acceptance_correction_20260905T133908Z` in its campaign root.
`rejected_restart_extensions.json` preserves the rejected record and reason;
`extended_24ps_branches.json` now reports blocked with zero accepted branches.
All trajectory files were preserved. Completed 15 ps top-up data are unaffected.

Validation in `pointnet`: `tests/test_exact_continuation_gate.py` and
`tests/test_independent_meam_source_campaign.py` passed (5 tests total).
`git diff --check` passed. The final live log check showed source 54 advancing
through melting and high-temperature sources 9/10 at measurement steps
171250/197000 of 200000, respectively.

## Remaining scientific blocker

CPU diagnostic **981566** completed successfully in about six minutes.
[Experiment and recipe](../experiments/restart_audit_20260905/README.md).
It reproduced the original 15 ps checkpoint byte for byte but observed velocity
divergence on the first restarted step, with unchanged rank ownership and local
atom order. Disabling sorting also failed. The exact-extension gate remains
closed; no 24 ps extension or new 2560-future production campaign was launched.

LAMMPS's Gaussian cache serialization is a likely underlying defect, as detailed
in the experiment. Repairing the engine and proving a complete 24 ps comparison
is still outstanding; merely resubmitting the same extension cannot fix it.
An uninterrupted rerun from a parent's original initial conditions is a possible
alternative, but must verify its existing 15 ps prefix before replacing data.
Do not reinterpret failed continuations as accepted exact extensions.

The existing source histories use their intended 0.75 ps cadence. The requested
0.3 ps parent histories and broader production campaign still require the
remaining source/calibration/parent-selection and storage gates described in
the audit; they are not missing jobs that can safely be submitted immediately.
