# CPU crystallization-origin campaign

Requested deadline: 26 September 2026, approximately 10:00 Europe/Paris.
All work runs in detached Slurm CPU allocations using `pointnet-torch214`.
No simulation, training or W&B run is created.

Completed 26 September: Al64 at 03:07 Paris time (job 1009235), external
extraction/ancestry by 03:56 (array 1009244 and job 1009262). All CPU jobs have
finished. [Scientific results](../experiments/crystallization_origin_20260925/RESULTS.md)
cover all 150 + 26 records; result-file and producer hashes were checked again
at completion review.

The existing Al64 audit resumes unchanged from its verified chunks and receipts,
using 24 extraction workers and four ancestry workers on 28 allocated CPUs.
Scientific configurations and frozen producer hashes are unchanged; runtime
parallelism is separate from the labeling contract. Only its old audit steps on
the GPU allocation were stopped.

The [external recipe](../configs/analysis/crystallization_origin_external_20260926.json)
adds the continuous million-atom Al history and available Al EAM, Mg, Ti and Ta
branches. [Definitions](metrics/crystallization_origin_external.md) explain
physical-time persistence, material length normalization, shared ancestry and
the nested 64-center/1% coverage comparison. Static Zr is ineligible for a
temporal audit.

External full-cell PTM is split into 1,674 independently verified four-frame
chunks, balanced by atom-frame count across 12 Slurm array lanes. Four processes
per lane provide 48 concurrent extraction workers. Heavy ten-million-atom Ta
chunks run first to expose their runtime/memory needs early. An independent
four-worker ancestry allocation consumes each complete trajectory as soon as
its chunks are available; it does not wait for the whole extraction array.

Maintained commands (execute the workers inside a CPU Slurm allocation):

```bash
python -m src.research.crystallization_origin.cpu fixed \
  --config configs/analysis/crystallization_origin_20260925.json \
  --workers 24 --analysis-workers 4

python -m src.research.crystallization_origin.external_data plan \
  --config configs/analysis/crystallization_origin_external_20260926.json

python -m src.research.crystallization_origin.external_data extract-lane \
  --config configs/analysis/crystallization_origin_external_20260926.json

python -m src.research.crystallization_origin.external_audit \
  --config configs/analysis/crystallization_origin_external_20260926.json
```

`extract-lane` uses `SLURM_ARRAY_TASK_ID` (0–11). Use one thread per process:
`OVITO_THREAD_COUNT=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
Submitted scripts, immutable code snapshots, job IDs and logs are under each
run's external `technical/` directory. Resume those snapshots, not changed code.
The initial CPU wall limits are nine hours, ending before the requested deadline
when started promptly. Completion estimates depend on measured rates and queue
starts; a wall limit itself is not a completion guarantee.

- [Fixed Al64 live report](/work/PERSO/vmorozov/analysis/crystallization_origin/al64-20260925/RESULTS.md)
- [External materials live report](/work/PERSO/vmorozov/analysis/crystallization_origin/multimaterial-20260926/RESULTS.md)

Progress files are `extraction-state.json`/`audit-state.json` for Al64, and
`lane-XX.json`/`audit-state.json` for the external campaign. Extraction chunk
receipts survive interruptions. Finished graphs, event catalogues and labels
are retained. No cache or result arrays are written inside the repository.
