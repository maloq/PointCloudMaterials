# BCR independent-root pilot operations

Use conda pointnet-torch214. Scientific protocol is in
`experiments/bcr_g1_20260921/README.md`. The independent-source inventory is
`docs/datasets/bcr_independent_roots.csv`; its exact selected-root receipt is
`output/bcr/g1-independent-20260921/technical/inventory.json`.

`python -m src.research.bcr_pilot.queue prepare --config configs/bcr/pilot_20260921/study.json`
checks the existing snapshots and produces a small immutable full-radius cache,
then writes matched training recipes. No LAMMPS simulation is run. Existing melt
integration snapshots retain their original bytes; no precision is recovered from
float16. A superseded preliminary box-float32 cache is retained only as a diagnostic
under the cache suffix `-superseded-box32`; it is not a training input.

Run `preflight --device cuda` on an allocated GPU. This reuses the BCR verification
and records a bounded batch256 throughput profile separately from training. Then
`submit` freezes code/tests/recipes and starts exactly three matched GPU jobs on
the profiled GPU class. With `submit --allocation 1001497` from that allocation,
BCR runs detached on the current H100 and only the two controls request new RTX
jobs. This requires passing cross-GPU FP32 parity and enough allocation time for
the measured full workflow plus a 30-minute reserve. Each new allocation has at least12h, covering training plus
checkpoint reconstruction/structural readouts; the measured fitting estimate gets
35% overhead plus2h evaluation allowance before choosing the wall limit. A required
budget above23h fails before submission.

The BCR lane evaluates all three checkpoints together at1,000/3,000/10,000 updates
(including initialization at the first comparison). The other lanes fit their full
budgets. Metrics/plots update after each completed comparison. Fits save exact
stream/optimizer state and stop with120s reserve; completed evaluations are skipped
on resume. If allocation exhaustion occurs, rerun `arm --arm NAME --device cuda`
from the frozen technical/code directory in a continuation allocation. All three
arms use dedicated RNG streams and identical exposure, independent of elapsed time.

Check `technical/launches.json`, `technical/budget.json`, per-arm training logs,
`technical/evaluations/*/complete.json` and `RESULTS.md`. A running training loss or
a successful overfit is not G1 evidence. Existing relaxed-data jobs are separate.
