# Ni birth-window GPU relaxation, October 8

The user selected every available native 0.10 ps frame from 6 ps before to
6 ps after each established Ni birth. This run uses the original twelve-melt
catalogue, its twelve-melt extension and the completed 36-melt low-temperature
catalogue. Together they contain 70 established PTM liquid-origin roots.
Other detector views and established composites are not extra instances of
these births. Original event definitions, roles and independent melt ancestry
remain unchanged. Sources without such births remain explicit in plan coverage.

The window anchor is the catalogue's `first_time_ps`: tracked family appearance
for a root that subsequently establishes, not its later confirmation time.
Every requested offset is matched to actual saved integer timesteps; missing
boundary observations are reported, never interpolated. Overlapping windows
share one relaxation per source frame. The plan retains the event-to-frame map,
potential checksum, source manifest and selected-frame checksums, catalogue
hashes, and exact execution recipe. These retrospective event windows are
not an outcome-independent training cohort or a population-rate sample.

## Physical protocol

Each task independently minimizes the entire 32,000-atom periodic cell at its
original box. It starts from that frame's archived positions, not a preceding
relaxed frame. The potential is the unchanged Mendelev 2012 Ni EAM/FS file used
to generate the source. No isolated neighborhood is relaxed, and no thermostat,
barostat, forward MD or encoder training is part of this run.

The existing FastNi CUDA callback supplies forces to native LAMMPS FIRE.
Minimizer coordinates/states are float64, force arithmetic float32, and scalar
energy accumulation float64. The initial FIRE timestep is 0.001 ps (an algorithm
parameter, not a physical observation interval), with a maximum force-component
tolerance of 0.01 eV/Angstrom, 20,000 iterations and 100,000 force evaluations.
Unconverged frames fail explicitly and their working state is archived.

A first production quench checks that the new FIRE wiring is active and that
the converged endpoint meets native CPU EAM force/energy agreement. Existing
FastNi force validation is reused; no force-engine benchmark campaign is repeated.
Two GPU workers depend on this check, using one combined compatible-partition
list per worker. Only this run's launch receipt establishes job ownership.

## Reuse and output

The shared relaxation index is consulted before collection and on every task.
The GPU callback, kernel/table code, native LAMMPS library, potential, minimizer
and precision settings are bound to the recipe. This is a distinct GPU recipe;
CPU minima and results from another recipe are not silently substituted.

Converged native restart states are published to the shared index. The standard
trajectory converter also produces verified float16 full-cell coordinates with
float32 boxes and exact IDs/timesteps. New local neighborhoods can be extracted
from the native restart without another minimization; the global float16 copy
does not stand in for a pre-quantization float32 neighborhood. Raw source Q/V
and birth labels are unchanged. There are no newly sampled velocities here.

- Registry: `ni-birth-relaxed-windows-20261008`.
- Recipe: [ni_birth_relaxation_20261008.json](../../configs/simulation/ni_birth_relaxation_20261008.json).
- Working cells: `${storage:simulation_runs}/ni-birth-relaxed-windows-20261008`.
- Published cells/failures: `${storage:archive}/simulations/ni-birth-relaxed-windows-20261008`.
- Plan, launch, progress and per-cell receipts:
  `${storage:training_storage}/ni_relaxation/birth-windows-20261008/technical`.

```bash
python -m src.simulation.campaigns.ni_birth_relaxation prepare --config configs/simulation/ni_birth_relaxation_20261008.json
python -m src.simulation.campaigns.ni_birth_relaxation submit --config configs/simulation/ni_birth_relaxation_20261008.json
python -m src.simulation.campaigns.ni_birth_relaxation status --config configs/simulation/ni_birth_relaxation_20261008.json
```

Use conda `pointnet-torch214`. Submission freezes implementation and config;
the same launch refuses duplicate submission. Worker and audit receipts, rather
than registration, determine actual completion. Simulation/preparation jobs
do not create W&B training runs.

## Submitted population

Preparation bound 70 events from 33 independent sources: 11 training, 12
selection and 10 test ancestors. All 8,470 event-offset requests are available;
overlap reduces these to **6,571 unique full cells**. The refreshed shared index
has no existing exact-recipe result for these cells. The population identity is
`53c649624ce569db7c024d105483415d467035575900e4f0cb5d5a885db3c436`.

At 17:58 Europe/Paris, the submitted jobs were first-quench check **1029100**,
dependent GPU workers **1029101 / 1029102**, and final collection status
**1029103**. Each worker requests one GPU from A40/A100/L40S/H100/RTX6000PRO
alternatives with a 24-hour allocation limit. This limit is not an ETA.
The launch receipt is `technical/launch.json` under the output root above.

## Four-GPU shared queue, October 8 evening

At the user's request to speed up this run, the new dispatch module
`src.simulation.campaigns.ni_birth_relaxation_queue` submits four workers over
the **same frozen producer, population and numerical recipe**. It does not
alter the original two-worker configuration or overwrite its launch receipt.
The separate receipt is `technical/speedup-01/launch.json`.

The first two replacement workers obtain GPUs before retiring their respective
original relaxation jobs. Ownership is checked against the original receipt
and actual Slurm command. Interrupted working files are retained on STORE;
completed native minima remain reusable through the shared index. Two further
workers become eligible after the corresponding old jobs end. Each GPU is an
alternative A40/A100/L40S/H100/RTX6000PRO request, with a 16-hour worker limit.

Until an original lane has stopped and its interruption is preserved, its
source frames are excluded from the shared queue. Released tasks use the
existing exclusive per-cell locks; already completed receipts are skipped.
The four workers can take work from one another. The original status job's
dependencies include all original and replacement jobs. Unrelated simulation
and training allocations are untouched.

Recent measured throughput before this change was approximately 186 cells/hour
on the A40 and 227 cells/hour on the A100, including publication overhead.
About 3 seconds per cell lies outside the timed GPU adapter body, including
interpreter startup/teardown. The current speedup changes resource use and
dispatch only; persistent-process or GPU-resident FIRE would require a separate
execution audit. Four workers at comparable rates imply approximately seven
hours for the remaining work after all four are active, excluding queue delay.

Submission (already recorded; do not duplicate):

```bash
python -m src.simulation.campaigns.ni_birth_relaxation_queue submit \
  --config configs/simulation/ni_birth_relaxation_20261008.json \
  --folder /store/PERSO/vmorozov/experiments/ni_relaxation/birth-windows-20261008/technical/speedup-01
```

## Software optimization, October 8 evening

The four pending scaling-only jobs (1029430–1029433) were superseded after the
user requested implementation speedups. Their original receipt is preserved,
with `technical/speedup-01/superseded.json` recording cancellation. Two optimized
replacement workers, **1029448 / 1029449**, are recorded separately in
`technical/speedup-03/launch.json`. Each acquires its GPU before retiring the
corresponding receipt-owned original worker. Thus the continuing campaign uses
two production GPUs after both handoffs, with unchanged coverage.

`src/simulation/eam_gpu/relaxation_fast.py` implements a fixed-box callback:

- One CUDA graph includes the conservative displacement bound, conditional
  neighbor rebuild, unchanged EAM kernels and float64 scalar reductions.
- Pinned reusable input/output buffers combine coordinate and force/scalar
  transfers, with one host synchronization per force evaluation.
- Atom-tag changes force neighbor rebuilding; displacement checks still run
  every evaluation. Box-changing commands are rejected.
- `relaxation_service.py` keeps one CUDA subprocess alive across cells, while
  creating a fresh native LAMMPS FIRE minimizer for each cell. The parent enforces
  the same 600-second per-cell timeout and retains explicit errors.

The potential, FIRE settings, independent per-frame initial positions, force
tolerance (0.01 eV/Angstrom), float64 minimizer states and float32 force kernels
are unchanged. The optimized callback has a **distinct execution recipe** in the
shared relaxation index; old minima and completed receipts retain their actual
producer. The population plan identity continues to bind frames and ancestry;
each archived `metadata.json` binds its actual settings and index recipe.

Local engineering audits under `technical/optimization-02/` used the original
worker's A40, temporarily pausing only that receipt-owned worker and recording
its resumption. They do not create W&B training runs.

| Same A40, three matched input cells | Original | Optimized |
| --- | ---: | ---: |
| Whole quench, including input writing | 14.61 / 12.54 / 14.53 s | 9.25 / 6.75 / 6.56 s |
| Mean quench time | 13.89 s | 7.52 s |
| Warm fixed-geometry force callback | 2.74 ms | 1.71 ms |

The first optimized cell includes initial CUDA setup. Conversion, index
publication and final archive verification are outside these quench timings.
The measured mean quench speedup is **1.85×**; warm callback speedup is **1.60×**.
These are software gains on the same GPU, not GPU-count scaling estimates.

All six paired endpoints passed an independent native CPU `eam/fs` force/energy
check; optimized native maximum forces were 0.00893–0.00992 eV/Angstrom. Six
additional force comparisons included neighbor rebuilds, large displacements
and tag permutation; RMS differences from the original callback were
1.1e−7–7.6e−6 eV/Angstrom. The largest perturbation was 0.35 Angstrom per-axis
Gaussian noise and included much larger forces than ordinary quench endpoints.

Individual minima are **not bitwise reproducible**. CUDA neighbor-bin ordering
changes float32 summation roundoff, and FIRE can finish in a nearby minimum.
For the checked first cell, original-versus-original coordinate RMS difference
was 0.105 Angstrom, versus 0.130 Angstrom for original-versus-optimized; energy
differences were 0.279 and 0.207 meV/atom respectively. Three matched quenches
establish endpoint convergence and speed, not statistical basin equivalence.
The paired audit and force/repeatability audit are bound by SHA-256 in
`technical/optimization-02/deployment-audit.json`.

The final submission additionally fixes metadata for cached-state restoration:
zero force calls now report zero neighbor rebuilds without reading an unused
output buffer. The numerical code is exactly unchanged; the one-line difference
and a successful native restart restoration audit are bound in
`technical/optimization-02/deployment-audit-v2.json`. The intermediate pending
jobs 1029442/1029443 were superseded before running, with their receipt preserved
under `technical/speedup-02/`. The two production cells used for the pipeline
audit retain that earlier optimized recipe and are not recomputed.

The frozen optimized producer also completed two actual remaining production
cells through conversion, checksum verification, native-index publication and
final STORE publication (`technical/optimization-02/pipeline-audit.json`). The
first took 93.8 seconds including compilation for the newly frozen CUDA source
path; this is startup overhead. The second took 24.9 seconds including 21.3
seconds for 6,790 force calls, a harder cell than the approximately 2,000-call
paired benchmarks. Its remaining 3.6 seconds were preparation/publication
overhead. The unchanged tolerance means per-cell times still depend strongly
on FIRE convergence; the three-cell benchmark is not a universal wall-time bound.

Reproduction command (already submitted; do not duplicate):

```bash
python -m src.simulation.campaigns.ni_birth_relaxation_queue submit \
  --config configs/simulation/ni_birth_relaxation_20261008.json \
  --folder /store/PERSO/vmorozov/experiments/ni_relaxation/birth-windows-20261008/technical/speedup-03 \
  --optimized-audit /store/PERSO/vmorozov/experiments/ni_relaxation/birth-windows-20261008/technical/optimization-02/deployment-audit-v2.json \
  --workers 2
```

## Precision feasibility checks, October 8 late evening

The user's request for float32 was clarified to **FP16/FP32 on the GPU**.
Local assays are recorded under `technical/precision-01/`, with script/result
hashes in `assay-receipt.json`. These checks do not change the production recipe.

The existing force kernels already calculate interactions in float32, while
native CPU FIRE states and scalar energy reductions use float64. The GPU
implementation dispatches float32/float64 explicitly; neural-network AMP does
not turn these custom neighbor-loop kernels into half-precision kernels.

On the same A40, narrowing every transfer buffer to float32 increased the median
callback time from 1.73 to 2.31 ms in six alternating measurements. Raw transfer
time fell from 0.139 to 0.078 ms, but conversion/preprocessing overhead outweighed
it. A more selective variant kept float64 coordinate preprocessing and energy
totals, transferring only forces in float32. Including the copy into native
LAMMPS's float64 force array, its median was 1.77 versus 1.86 ms (about 5% faster).
These are callback assays, not measured whole-campaign speedups.

The FP16 feasibility assay evaluated three saved liquid frames and their three
previously computed full-precision relaxed endpoints:

- Direct FP16 storage of the potential tables overflowed some coefficients.
  A second, explicitly mixed variant retained overflowing coefficients in
  float32 and quantized the others through float16 before unchanged float32
  force arithmetic. At relaxed endpoints, force RMS differences were about
  0.00040 eV/Angstrom and maximum component differences 0.00172–0.00186 eV/Angstrom.
  Two of three endpoints then exceeded the 0.01 eV/Angstrom stopping tolerance
  slightly. This does not establish failure of a coarse pass followed by an
  original-potential refinement, which has not been run.
- Quantizing relaxed positions to float16 produced about 0.0088 Angstrom
  coordinate RMS error and changed maximum forces from 0.00892–0.00992 to
  **0.484–0.533 eV/Angstrom**. Float16 states therefore cannot represent these
  converged endpoints at the declared tolerance. The observed source positions
  are already archived in float16; their small additional quantization error
  is not evidence that float16 minimizer states are adequate.

The half-table assay used float16-to-float32 round trips to isolate numerical
error. It did **not** implement or time an FP16 CUDA kernel, so it establishes no
FP16 speed gain. A possible future variant would retain positions, distances and
force accumulation in float32, use selected lower-precision storage only where
validated, and finish against the original float32 potential. Moving FIRE onto
the GPU targets CPU work and transfers directly; FP16 alone does not remove them.
