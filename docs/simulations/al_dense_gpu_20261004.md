# Dense Al sources on differentiable GPU MEAM

## Completed batch and CPU comparison

All five descendants completed, converted and archived on October 4 at
16:23 CEST. The [read-only comparison](../../output/meam_gpu/dense-cpu-comparison-20261004/index.html)
includes all 150 historical CPU parents, 34 completed main dense CPU descendants
and these five GPU descendants. The two supplemental 0.01 ps trajectories are
excluded from this comparison. Parent/daughter paths share ancestry.

| Source | K | GPU first confirmed half observation | Same-parent historical CPU |
| --- | ---: | ---: | ---: |
| 894 | 400 | Not observed through 600 ps | 315 ps |
| 914 | 450 | 195 ps | 180 ps |
| 944 | 500 | 270 ps | 300 ps |
| 974 | 510 | 180 ps | 135 ps |
| 1005 | 520 | 330 ps | 150 ps |

Half crossing requires two consecutive samples exactly 15 ps apart above 50%;
the table reports the first sample. These are bulk transitions, not nucleus births.
The historical parent CPU used 3 fs integration; both dense engines use 2 fs.
The same native prepared liquids, velocity seeds and potential file hashes were
verified. Long same-parent differences therefore do not isolate GPU arithmetic.

At 400 K, three of ten complete dense CPU runs also lacked confirmed halfway
crossings. At 520 K, 22 of 24 dense CPU runs crossed, spanning 120–405 ps;
the GPU's 330 ps falls within that range. All four GPU crossing times fall
within their historical same-temperature CPU ranges. CPU 2 fs versus 3 fs
paired reruns at 520 K already shift by −330 to +105 ps among the 20 pairs
with observed crossings in both protocols (median absolute shift 105 ps).
These conditional comparisons do not discard the separately reported censored runs.

Energy/volume versus observed crystal fraction overlap closely. For the first
15 ps, GPU potential energy differs from the same-parent historical CPU by
at most 0.382 meV/atom, volume by 0.0216%, and mean temperature by 0.186 K.
The dense CPU comparisons at 400/520 K are tighter in energy: GPU early energy
is within 0.035 meV/atom of the same-temperature 2 fs CPU cohort mean, and volume
within 0.006%. Source 884/885 thermodynamic prefixes are unavailable in their
canonical restart publications, so these early CPU cohorts contain 8/24 runs;
their crystallization curves remain included. Full GPU logged means stay within
0.404 K of target, with pressure between −0.386 and +1.255 bar.

This is reassuring for the potential/NPT implementation, not proof of equal
nucleation rates, equilibrium phase diagrams or long-time thermodynamic equivalence.
One GPU trajectory per temperature and different historical timesteps cannot
provide that proof. The decisive next comparison would use matched 2 fs CPU/GPU
replicates with the same initial states and equal observation/stopping rules.
No simulations or jobs were changed by this audit; the remaining dense CPU queue
continues separately, and nucleus-birth analysis is outside this audit.

Reproduce with conda `pointnet-torch214`, using a new output revision for a
changed availability capture:

```bash
python -m src.research.meam_gpu.dense_comparison --config configs/simulation/al_dense_gpu_comparison_20261004.json
```

[Metric definitions](../metrics/meam_dense_comparison.md), hashes, native log
coverage and reused PTM provenance accompany the CSV tables and plots. Earlier
launch records below are preserved as historical operational context.

## Original transfer

User-authorized transfer of five **unstarted** trajectories from the existing
[dense Al queue](al_main_half_stop_20261001/README.md) to two single-GPU workers.
At the selection audit, 34 of 152 trajectories were complete, two CPU recovery
trajectories were running, and 116 were unstarted. The five transferred sources
share existing melt ancestors and retain their original training roles. Existing
completed data, running CPU trajectories and fixed Al64 releases are unchanged.

| Source | Temperature | Maximum measurement duration |
| --- | --- | --- |
| 894 | 400 K | 600 ps |
| 914 | 450 K | 291 ps |
| 944 | 500 K | 411 ps |
| 974 | 510 K | 411 ps |
| 1005 | 520 K | 561 ps |

Every source contains 70,304 atoms. Reuse the original full-precision prepared
liquid, velocity seed, 15 ps equilibration, 2 fs timestep, native LAMMPS NPT fix
(0.3 ps thermostat, 3 ps isotropic barostat, zero target pressure), momentum
removal every 0.3 ps, exact 0.1 ps positions/velocities and native restart files.
The existing PTM monitor confirms >=50% crystal at two observations exactly
15 ps apart, then retains the existing 6 ps prediction tail, subject to the cap.
Caps without confirmation remain right-censored.

## GPU coupling and numerical gate

[Adapter](../../src/research/meam_gpu/npt.py) supplies analytic forces, energy and
three normal strain derivatives through [LAMMPS fix external](https://docs.lammps.org/fix_external.html).
The same pinned Lee2003 MEAM energy uses float32 force arithmetic and float64
native integration. The unchanged isotropic barostat consumes the virial trace.
Shear stress is not computed: tilted cells and anisotropic pressure control are
outside this adapter's contract. CUDA neighbor support is rebuilt each step to
cover box deformation and native atom reordering. One LAMMPS rank runs per GPU.
This numerical execution change is explicitly recorded in each source and binary
trajectory manifest; exact long-time trajectories are not expected to match CPU.
The original differentiable force/HVP API remains available for response work.

Full-cell source 894 audit, 100 native NPT steps (0.2 ps), passed before submission:

- Initial force RMS difference: 1.01775e-5 eV/angstrom.
- Initial energy difference: 9.833e-8 eV/atom.
- Largest initial normal-pressure difference: 0.07912 bar.
- Final position RMS difference: 1.05975e-5 angstrom.
- Final relative volume difference: 1.14134e-7.
- GPU stepping: 0.7498 seconds; single-rank native CPU: 47.08 seconds.

Timing excludes GPU compilation, dense trajectory output and PTM callbacks.
It is not a comparison against the campaign's 48-rank CPU workers. This short
gate establishes numerical coupling, not equilibrium NPT or nucleation-rate
equivalence. Validation snapshots are diagnostic and excluded from sample counts.

Submitted October 4, 2026 at 14:38 CEST: array **1020872**, two GPU tasks on
node60/node61; reconciliation **1020873**. CPU successor **1020296** waits for
CPU recovery array **1020820** and reconciliation. Initial GPU lane assignments
are source894 then974, and source1005 then944 then914.

## October 4 hardware transfer

The user subsequently requested that active RTX6000 trajectories finish, with
the remaining sources moved to A100/L40S/A40. At capture, source1005 was complete,
source894 was approaching its 600 ps cap, source944 was running, and sources914
(450 K) and974 (510 K) had not started. Their native scientific inputs and the
original frozen physics, converter and monitoring code are unchanged.

The [transfer recipe](../../configs/simulation/al_dense_gpu_transfer_20261004.json)
submits one source per job, eligible only for `A100,L40S,A40`. Replacement array
**1020942** received two separate A100-PCIE-40GB GPUs on node05. Each job records
its actual architecture, builds the extension for that architecture and runs a
100-step full-cell native-versus-GPU NPT check before production. The separately
frozen [transfer driver](../../src/simulation/campaigns/dense_al_gpu_transfer.py)
executes the original campaign's frozen scientific code.

Both A100 gates passed, and both native production runs started at15:43 CEST.
Force RMS errors were1.040e-5 and1.028e-5 eV/angstrom, maximum normal-pressure
errors0.092 and0.123 bar, and final relative volume errors below1.23e-7.
Each source advanced through its native equilibration with actual GPU force
calls recorded in `technical/gpu-progress.json`. Job1020942_0 owns source914;
job1020942_1 owns source974.

The first hardware startup (array1020936) was stopped before either transferred
production trajectory started: the A100 nodes defaulted to CUDA11.8, and MPI's
default UCX transport failed memory registration. The revised launcher uses a
user-scoped copy of the working CUDA13.4 compiler and `UCX_TLS=tcp,self` for the
single-rank LAMMPS process. No system toolkit or running simulation was modified.
The failed startup and both software/launch revisions are retained under
`hardware-transfer-20261004/`; active receipts are in its `attempt-v2/` directory.

Reservations for914/974 are marked transferred. The existing frozen RTX workers
finish current MD, conversion and publication, then exit at their next-source
reservation guard. This planned boundary exit can appear as a failed Slurm job;
it does not mark the completed trajectory as failed and starts no duplicate MD.
No signal is sent to the active RTX workers. Replacement collector **1020943**
waits for both arrays, then reconciles all five original source IDs. CPU successor
**1020296** now waits for CPU recovery1020820 and collector1020943. Superseded
pending collectors1020873/1020937 were canceled; scientific workers were preserved.

## Queue and preservation

[Recipe](../../configs/simulation/al_dense_gpu_20261004.json) and
[queue implementation](../../src/simulation/campaigns/dense_al_gpu.py) freeze the
code, original input checksums, source identities and assignment. GPU workers
write new simulations on SCRATCH. The established conversion command verifies
float16 positions/velocities, float32 boxes and exact IDs/timesteps before deleting
text dumps. Native restart precision is retained; completed and failed attempts
publish to STORE. GPU restarts require reinstalling the external force callback;
the native file retains atom and NPT extended state, not that Python callback.

Only the pending CPU successor is briefly held to reserve these inputs. Both
running CPU recovery jobs continue. The successor waits for CPU recovery and GPU
reconciliation; completed GPU records become its canonical completed sources.
Failed GPU attempts are archived and their unchanged unstarted CPU inputs are
released, with an explicit receipt. This prevents duplicate execution of the same
source. All other unstarted sources remain in the original CPU queue.

Using conda `pointnet-torch214`:

```bash
python -m src.simulation.campaigns.dense_al_gpu prepare --config configs/simulation/al_dense_gpu_20261004.json
python -m src.simulation.campaigns.dense_al_gpu submit --launch /store/PERSO/vmorozov/simulation-launches/al-dense-gpu-20261004
```

Launch receipts, local validation, frozen code, per-source completion and failed
attempt records live in that launch directory. `completion.json` is written by
the reconciliation job only after both GPU lanes finish. The rest of the Al
pool, including historical and not-yet-analyzed trajectories, remains tracked in
the [shared overview](../al_research_pool.md).
