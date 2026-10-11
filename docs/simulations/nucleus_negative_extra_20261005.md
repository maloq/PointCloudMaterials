# Additional negative controls, October 5

The user requested more negative controls with a three-hour queue. The preceding
matched 180-branch campaign is complete. This separate recipe adds 64 original
ancestors absent from every previous matched group, one liquid state per source,
and four new stochastic shots for each of the existing 12 failed-embryo starts.
There are 240 new 12 ps branches: 64 NPT, 128 liquid BAOAB, 48 embryo BAOAB.
The full-cell Q,V inputs, original source roles and native 0.75 ps observations
remain explicit; no encoder/predictor is trained and no W&B run is created.

The existing FastAl implementation and maintained recovery integrators are reused
without new potential validation. BAOAB states/momenta/noise/forces are float32;
NPT retains native float64 integration and float32 GPU force arithmetic. Scalar
energy reduction is separately float64. NPT resets missing bath state. Recorded
future checks use the same frozen birth metrics. The supplement does not rerun
deterministic NPT from identical embryo states.

Selection is frozen before new dynamics: exclude all prior matched ancestors,
then rank eligible sources by seeded hash, sample a present-liquid frame and
anchor uniformly. Do not select by its saved future outcome. All source exclusions,
eligible frames, temperatures, roles, ancestry, initial hashes and previous-release
links are in `technical/plan.json`. Only 12 independent strict failed-embryo sources
were available in the preceding protocol; extra noise shots cannot increase that
state count. Earlier positives remain a separate reference, not a new matched
comparison with this supplemental liquid population. See [metric definitions](../metrics/nucleus_negative_extension.md).

Use `pointnet-torch214`:

```bash
python -m src.research.nucleus_controls.negative prepare --config configs/simulation/nucleus_negative_extra_20261005.json
CUDA_VISIBLE_DEVICES=0,1 python -m src.research.nucleus_controls.negative launch --config configs/simulation/nucleus_negative_extra_20261005.json
python -m src.research.nucleus_controls.negative summarize --config configs/simulation/nucleus_negative_extra_20261005.json
```

`launch` requires the explicitly selected free GPU pair in an existing allocation.
It freezes code, starts two detached supervised workers there, and submits four
one-GPU helper jobs to A100/A40/L40S. Node52 is excluded after the preceding
campaign's recorded CUDA initialization failure. Helpers request two hours to
fit scheduler backfill; all workers share a three-hour absolute deadline including
queue wait. A 15-minute reserve checkpoints work, with supervisor termination of
only its recorded child process groups before the deadline. No automatic extension,
allocation cancellation, or changes to unrelated jobs occur. Worker leases prevent
duplicate branch execution. `UCX_TLS=tcp,self` avoids the previously diagnosed
MPI/InfiniBand memlock failure. Architectures and compiler caches are per worker.

Outputs resolve from the recipe: analysis under
`nucleus_recovery/negative-extra-20261005/analyses/negative-v1`, new integration
files in SCRATCH, verified converted trajectories and restart states in STORE.
`technical/launch.json` owns the local PIDs and helper array; `budget.json` records
the absolute deadline; `progress.json` separates planned and completed branches.
Report both the independent first-shot Wilson intervals and all-shot state means;
unfinished branches are not failures to nucleate. Scientific CSVs have frozen
metric definitions and implementation hashes. Other Al cadences and ancestry
families remain deferred, as listed in the plan.

## Launch receipt

Launched October 5 at 17:16:22 Paris time; absolute deadline 20:16:22.
Node60 allocation `1022218` hosts the two new owned workers; helper array
`1022663_[0-3]` started on four A100 GPUs (node02, node07, two distinct devices
on nodeaudible01). The local supervisor is PID/PGID `3574566`; the allocation
itself is not owned by this campaign. The complete launch receipt is in the
configured analysis output at `technical/launch.json`.

Prepared plan identity:
`aa3341d8ebcdc9adfbffd4e08bbcf452e1a819a8691aacb1f4c3f5a0b8ea9709`.
The 64 new liquid states comprise 12/11/14/14/13 sources at
400/450/500/510/520 K, respectively. Input audit verified all 76 hashes, finite
float32 Q,V arrays, 70,304 unique atom IDs, exact 16-observation-step spans,
ancestry separation and disjoint stochastic seeds. All recorded reference
windows were measured before launch. Completion remains live in `progress.json`.
