# Causal MACE three-seed GPU pilot

Running the controlled pilot on node53, H100 NVL, existing Slurm allocation
991900. Data preparation verifies 150 independent Al sources and produces
1,800 causal examples with 0.75/3/9 ps physical targets.

- [Scientific protocol](../../../experiments/mace_causal_20260916/README.md)
- [Preparation log](technical/prepare.log)
- [Three concurrent seed queues](technical/launch-plan.json)
- [Launch progress](technical/launch-status.json)
- [Seed 20260916 progress](../pilot-20260916-lane20260916/technical/allocation-status.json)
- [Seed 20260917 progress](../pilot-20260916-lane20260917/technical/allocation-status.json)
- [Seed 20260918 progress](../pilot-20260916-lane20260918/technical/allocation-status.json)
- [Tracked seed controller](technical/controller-seed20260916/execution/run_record.json)
- [Correctness tests](technical/tests.log)
- [Paired comparison recipe](../../../configs/mace_causal/comparison.json)

The queue contains 21 encoder fits, 72 diagnostic readout fits, and a paired
collector. A–D and repeated-anchor fits have matched budgets; E is a second
training phase initialized from D. Selection uses validation sources only.
The full source snapshot, environment, launch arguments and outcome are retained
by the existing tracked execution workflow. Best and exact-resume checkpoints
are retained per encoder variant. No external jobs are submitted or changed.

Three additional D fits use the diagonal-Gaussian future head. Their individual
physical-error, NLL and coverage tables are retained as a separate uncertainty
comparison; they do not alter the primary matched deterministic suite.


The completed data audit contains 1,080/360/360 train/validation/test windows.
The low-order subset contains 373/105/190 windows from 35/10/17 sources.
Sustained future onset labels cover 10/6/1 distinct centers from 10/5/1 sources;
the two positive test windows overlap around the same local event. Consequently,
this pilot cannot establish general onset-detection or timing skill. Event scores
are descriptive only; a broader fixed-time/transition-aware anchor protocol is
needed for that separate question. The data population was not changed after
this audit. See [audit counts](technical/data-audit.json).
