# Predictive memory: velocity-input seed replication

Queued after the primary pilot on the same H100 allocation. Uses training seed
20260918 for four matched 3,000-update fits: snapshot, 12 ps history, 48 ps history,
and a separately trained 48 ps repeated-anchor control. All receive positions and
measured velocities. Only seed and output path differ from the primary recipe.
The same 150-source release is reused; no new simulations or data preparation.

Purpose: test whether history beyond measured current velocities improves fixed
physical future prediction consistently across training runs. The seed was chosen
before the primary history comparisons completed. A separate paired source-level
comparison runs automatically after all four equal-budget fits finish.

- [Waiting / dependency status](technical/detached/queue_status.json)
- [Training queue status, created when the primary completes](technical/allocation-status.json)
- [Launch plan](technical/allocation-plan.json)
- [Paired comparison, after completion](comparison/README.md)
- [Scientific protocol](../../../experiments/predictive_memory_20260917/README.md)

The controller starts this follow-up only after the primary tracked command
succeeds. At least 55 minutes must remain before the 03:10 Paris cutoff to start
the suite; per-fit budget checks and exact-resume checkpoints remain active.
This is exploratory replication on previously examined, float16 sources.
