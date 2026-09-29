# Running the nucleus-emergence harvest analysis

This CPU-only pass uses the 90 original Al training sources and completed PTM/
ancestry caches. Other roles remain for a later frozen release.
[Proposal](../experiments/crystallization_origin_20260925/HARVEST_PROPOSAL.md)
and [definitions](metrics/nucleus_harvest.md) describe the scope.

Use conda pointnet-torch214 inside a CPU Slurm allocation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OVITO_THREAD_COUNT=1 \
python -m src.research.crystallization_origin.harvest \
  --config configs/analysis/nucleus_harvest_20260926.json --workers 24
```

--sources 860 processes one real training source under the same contract; the
full run resumes its receipt. --report-only rebuilds tables without trajectory
processing. Worker count does not change scientific identity. No training,
simulation, W&B run or automated test suite is launched.

Outputs use external analysis storage: nucleus_harvest/train-audit-20260926/.
RESULTS.md, tables/ and plots/ hold readable results; references, descriptors,
causal checks, source receipts and progress are in technical/. The state.json
there distinguishes partial/complete/failed runs. Resume submitted jobs using
their frozen snapshot in technical/code/.

This pass exports exploratory references and diagnostics, not a training-ready
probability dataset.

The real-source pilot (860) completed as job 1009410, with all four criteria's
causal-prefix checks passing. The full 90-source run resumes it in detached
Slurm job **1009412**, with 24 CPU workers and 96 GiB requested on nodecpu05.
The three-hour wall limit is a limit, not a completion estimate.

- [Live report](/work/PERSO/vmorozov/analysis/nucleus_harvest/train-audit-20260926/RESULTS.md)
- [Progress](/work/PERSO/vmorozov/analysis/nucleus_harvest/train-audit-20260926/technical/state.json)
- [First spatial review](/work/PERSO/vmorozov/analysis/nucleus_harvest/train-audit-20260926/plots/source-860-birth-review.png)
