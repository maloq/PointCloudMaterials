# Ti crystallization source and Ta branches — 2026-09-07

**Current launch:** Ta runs first on all 48 physical cores; Ti then starts with exactly
100,000 atoms on the same cores. This supersedes the initial concurrent million-atom
Ti launch. Partial runs are preserved under `interrupted_attempts/initial_24rank_campaigns/`.

Research question: replace the rejected reconstructed-potential Zr trajectories
with a pure-Ti crystallization source, then sample futures from different stages
of that **one source**. Add the five unused Ta snapshots to the existing Ta branch.
The user selected six Ti branches of 240 ps and six Ta branches total, and then
explicitly changed Ti from independent melts to source-then-branch sampling.

## Scientific protocol and sources

Ti uses the **unmodified author-supplied Kavousi et al. (2019) Ni/Ti 2NN-MEAM**
files in `potential/`, downloaded from the
[NIST implementation](https://www.ctcms.nist.gov/potentials/entry/2019--Kavousi-S-Novak-B-R-Baskes-M-I-et-al--Ni-Ti/).
The paper is [DOI 10.1088/1361-651X/ab580c](https://doi.org/10.1088/1361-651X/ab580c).
It explicitly fits/tests the pure species as well as the binary alloy for
high-temperature solid/liquid properties. Download URLs and SHA-256 hashes are
in `ti.json`. Do not substitute the generic LAMMPS library or omit the parameter
file: that would change the potential, including its second-neighbor terms.

The exact mapping is:

```lammps
pair_style meam
pair_coeff * * Kavousi2019_NiTi.library.meam Ni Ti Kavousi2019_NiTi.meam Ti
```

There is **one atom type, entirely Ti**. `Ni Ti` before the parameter filename
preserves the file's numerical indices (Ni=1, Ti=2); the final `Ti` maps the only
LAMMPS atom type. This follows the
[LAMMPS MEAM indexing rules](https://docs.lammps.org/pair_meam.html).
Retain `nn2(2,2)=1`, `augt1=0`, `ialloy=2`, `erose_form=2`, the 5 Å cutoff,
and all supplied Ti screening parameters. The potential's mass is 47.880 g/mol.

One periodic 50×40×25 BCC cell (exactly 100,000 Ti atoms, initial lattice constant
3.29089653438 Å) melts for 50 ps at 3000 K and zero pressure. The starting
lattice is only a way to make atoms: require <1% FCC/HCP/BCC classifications
and >10 Å² MSD growth over the last half of melting before releasing the liquid.
The source is quenched to 1250 K with velocity scaling and a fresh NPT fix.
It then evolves continuously, without restarting the integrator between checks.
No crystalline seeds, restraints, impurities, or free surfaces are introduced.

1250 K is an exploratory undercooling, **not a measured optimum nucleation
temperature**. The author's
[2019 conference report](https://www.lsu.edu/eng/mie/graduate/conference/2019docsandphotos/sepideh_kavousi_.pdf)
reports a pure-Ti melting point near 1941 K for this development. Homogeneous
nucleation remains stochastic; this setup does not guarantee a particular
induction time or equilibrium phase diagram.

The Ti timestep is 1 fs; isotropic NPT damping is 0.1 ps for temperature and 1 ps
for pressure. These correspond to the approximate 100/1000-step guidance in
the [LAMMPS NPT documentation](https://docs.lammps.org/fix_nh.html).
Frames are saved every 0.1 ps. Every 4 ps, save a position snapshot and restart
and measure full-system FCC/HCP/BCC fractions using OVITO PTM with RMSD cutoff
0.17. This cutoff tolerates thermal displacement; the preflight checks it against
both a melt and a hot BCC crystal (0.58% versus 97.1% classified crystalline). The
[OVITO documentation](https://www.ovito.org/manual/reference/pipelines/modifiers/polyhedral_template_matching.html)
explains the thermal robustness and cutoff tradeoff. Counts for all three
structures are retained; BCC is not imposed as the outcome.

Operationally, complete crystallization means ≥95% FCC+HCP+BCC at two consecutive
4 ps checks, with at least 24 ps of source dynamics. Grain boundaries and defects
can remain. A 2 ns source limit prevents unbounded work: failure to meet the
criterion sets campaign state to `failed`, preserves all outputs, and **does not
launch the branches or claim complete crystallization**. Inspect the resulting
progress before authorizing any different temperature/protocol or extension.

After source completion, select six distinct chronological snapshots nearest
crystal fractions 0, 0.1, 0.3, 0.5, 0.7, and 0.9. Record actual fractions and times
in `selected_branches.json`; finite sampling may skip an exact target fraction.
Each starts a 240 ps, 2401-frame NPT trajectory at 1250 K with its own recorded
Maxwell–Boltzmann velocity seed. These are **position-conditioned branches**, not
exact phase-space continuations. All six share one root lineage; selection uses
the completed source and is intentionally conditioned on crystallization stage.
They must not be represented as six independent nucleation experiments or split
across train/test as independent sources.

Ta retains the existing Zhong et al. (2014) EAM potential and archived-coordinate
protocol: 1900 K, zero pressure, 2 fs timestep, NPT damping 0.2/2 ps, momentum
removal every 100 steps, 24 ps and 241 frames per branch. The unused `2.7ns`,
`2.8ns`, `2.9ns`, `3.0ns`, and `3.60ns` snapshots each contain **10,000,422 atoms**;
they are not cropped to match the earlier 1,024,000-atom `model_1m` branch.
Source positions are preserved and fresh velocities are generated. Exact input
hashes and seeds are in `ta.json`. The existing completed branch stays at
`datasets/ta_initial_1x24ps`.

## Reproduction and operation

From the repository root, using `pointnet`:

```bash
conda run -n pointnet python scripts/run_lammps_campaign.py elemental sequence \
  --ta-config experiments/ti_ta_crystallization_20260907/ta.json \
  --ti-config experiments/ti_ta_crystallization_20260907/ti.json
```

This command refuses to overwrite a previous sequence or campaign. It finishes all
Ta simulations and verified conversions before starting Ti; a Ta failure stops the
sequence and is reported in `sequence_status.json`. For a new experiment, change
the output root in a new explicit configuration. Do not run this command again
over the active output directories.

Output root: `/home/ids/vmorozov/simulations/ti_ta_crystallization_20260907/`.
The repository link `datasets/ti_ta_crystallization_20260907` points there.
`Ti/` contains `melt/`, `source/`, and `branches/`; `Ta/branches/` contains the
five new Ta paths. Each campaign has `status.json` and `runner.log`. Ti additionally
has `source_progress.json`, `liquid_validation.json`, and selected parent records.
Complete branch artifacts are published only after conversion checks succeed.

One worker runs as `ta-then-ti-resume-20260907.service` on `lamedell11`, with logout
persistence enabled. Each campaign uses all physical CPU IDs 0–47 (48 MPI ranks
across both sockets; the host exposes 96 logical threads). The campaigns never
overlap. The existing separate Ti and Ta units have been stopped. GPU/Slurm
campaigns are not modified. Service definitions and receipts are generated
diagnostics in the output directory.

```bash
systemctl --user status ta-then-ti-resume-20260907.service
cat datasets/ti_ta_crystallization_20260907/sequence_status.json

# Ti files appear after Ta completes:
cat datasets/ti_ta_crystallization_20260907/Ti/status.json
cat datasets/ti_ta_crystallization_20260907/Ti/source_progress.json
cat datasets/ti_ta_crystallization_20260907/Ta/status.json
tail -f datasets/ti_ta_crystallization_20260907/Ti/melt/stdout.log
```

Final restarts, source snapshots, and verified float32 binaries remain on the large
filesystem. On recovery, both configs enable `delete_verified_source_text`: raw
trajectory text is deleted only after binary verification. The maintained converter
`scripts/convert_trajectory.py elemental BRANCH` is called automatically after
dynamics. It verifies exact cadence, frame/atom counts, IDs/types, source SHA-256,
binary array checksums, and a semantic coordinate hash. Float32 coordinates are
wrapped relative to the per-frame lower box corner, matching the existing reader.
Conversion scratch arrays use disk-backed memory maps and are removed only after
verification. No quantized float16 coordinates are used for new results.

## Validation and findings

- A 3456-atom Ti preflight melted at 3000 K for 20 ps, then ran 2 ps NVE and
  2 ps of quenching. It completed with zero dangerous neighbor builds. PTM at
  cutoff 0.15 found only 0.116% crystalline environments in the melt. NVE total
  energy variation over the stored samples was about 0.003 eV in the whole box
  (about 9×10⁻⁷ eV/atom).
- The full 10,000,422-atom Ta preflight completed 100 steps with pressure from
  −0.36 to −8.97 bar and no dangerous neighbor builds. Dynamics took 77.25 s on
  24 ranks; full runs will also incur substantial trajectory I/O.
- A real 128-atom LAMMPS workflow smoke test exercised the melt/source shell
  callbacks, continuous timeline, selection of six distinct parents, all six
  branch runs, and all seven binary conversions. Its permissive phase threshold
  is strictly a control-flow fixture, **not evidence of crystallization**.
  The first diagnostic fixture had only one MSD sample and failed loudly; the
  corrected fixture has multiple samples. Both diagnostic attempts are retained.
- Nine tests passed: conversion corruption/cadence checks and a sequence test
  verifying that failed Ta work prevents Ti from starting.
- The full-size 1,024,000-atom Ti preflight passed 100 steps in 80.07 s on
  24 ranks with no dangerous builds. The hot-crystal check motivated increasing
  the production PTM cutoff from 0.15 to 0.17: crystal recognition increased
  from 94.76% to 97.11%, while melt classification remained below 1%.
- Full-size Ti and hot-crystal preflight results and launch status are recorded
  in the output root's `preflight/` and `launch.json`.
- Production crystallization and branch outcomes are pending; do not treat
  submitted/running jobs as completed data.

## Zr removal and file classification

`zr_removal.json` records the user-authorized deletion of six result directories
(8,958,644,641 apparent bytes). Small historical metadata was preserved under
the output root's `zr_removal_audit/`. Al/Mg results, original `datasets/Zr`
configurations, and the historical external raw archive remain untouched.
The old dataset manifest now lists only the 12 retained Al/Mg branches; old
aggregate conversion statistics are explicitly labeled historical.

New files under this experiment are **versioned experiment records and potential
inputs**. `src/simulation/campaigns/elemental.py` and
`src/data_utils/conversion/elemental.py`, exposed through the existing family
commands, are **maintained implementation**; the conversion tests are maintained
checks. Generated LAMMPS inputs, smoke configs, logs, service definitions,
preflight reports and results under the output root are **disposable diagnostics
and simulation outputs**, not new scripts under `scripts/`.

Historical launch, superseded by the 00:41 CEST sequential launch: both systemd services were active
with 24 actual LAMMPS ranks each on disjoint physical CPU sets, and both logs
showed the expected full-system atom counts. Ti started melt preparation; Ta
started the `2.7ns` branch. Logout persistence is confirmed by `Linger=yes`.

At 00:41 CEST, the replacement sequential worker was launched with Ta first,
48 MPI ranks per campaign, and the revised 100,000-atom Ti configuration.
Both previous partial runs were explicitly stopped, labeled interrupted by user
reconfiguration, and preserved. No existing Ta snapshot or finished branch was
deleted. The full-size million-atom Ti preflight above is historical potential
validation; the reduced-size Ti production has not started while Ta is running.

## Storage failure observed 2026-09-07 at 11:09 CEST

**Subsequent recovery, 11:47 CEST:** the authorized IDS cleanup reclaimed
99.05 GiB net. The existing `Ta/branches/2.7ns` dynamics were converted and fully
verified; its raw text and conversion scratch were removed, its final restart
retained, and its complete outcome published. No dynamics were rerun and the
remaining sequence was not resubmitted. See
[the cleanup record](../../docs/ids_storage_cleanup_20260907.md) and
`output/ids_storage_cleanup_20260907/ta_recovery.json`. The failure below is
historical; budget remaining branch output and conversion peaks before resuming.

The service exited at 02:21:26 CEST during binary conversion of the first Ta
branch. Its full 24 ps dynamics and final restart completed; no later Ta branch
or Ti source started. A write probe confirmed EDQUOT on `/home/ids`. The
filesystem-wide space check missed the user quota. The incomplete duplicate
binary was removed, preserving raw data, decoded positions, restarts and logs.
Quota errors still prevent new files. Existing status files were corrected in
place because quota prevented atomic replacements. See
`storage_failure_20260907.json`. More quota or another output location is needed
before resuming; completed dynamics must not be rerun.

## Resume after user storage cleanup — 2026-09-07

The user reclaimed space and finalized `Ta/2.7ns` as a verified 241-frame float32
binary, with raw text removed and the final restart retained. The resumed worker
verifies its binary checksums, restart checksum, source identity, and protocol,
then skips its dynamics. Four Ta branches remain, followed by the 100,000-atom
Ti source and six 240 ps branches, always using 48 physical cores sequentially.

```bash
conda run -n pointnet python scripts/run_lammps_campaign.py elemental sequence \
  --ta-config experiments/ti_ta_crystallization_20260907/ta.json \
  --ti-config experiments/ti_ta_crystallization_20260907/ti.json --resume-ta
```

This is the recovery command, not a second launch command while the service is
active. Failed status records are preserved in `resume_history/`; completed
branch artifacts remain untouched. A changed or incomplete branch fails
verification rather than being silently skipped or overwritten. The Ti source
is not automatically restarted by `--resume-ta`.

The converter consumes its own validated scratch positions file instead of
copying it to a second full-size file, saving about 27 GiB of peak disk use per
Ta branch. Both production configs delete raw dumps only after source, binary,
and semantic-coordinate verification. Final restarts and provenance are kept.
The first resume encountered another EDQUOT while appending the old runner log;
a subsequent 16 MiB write/fsync and write to that log passed. The retry then
verified the completed branch and advanced to `2.8ns`. Errors are also sent to
the systemd journal so exhausted simulation storage cannot hide the sequence's
failure traceback. The NFS quota RPC endpoint is unavailable; successful write
probes do not establish the total remaining quota.

Validation: 14 conversion/resume tests and three temporal binary reader tests
passed. Recovery code updates are maintained implementation/tests; this README
and the configuration changes are experiment records. New launch receipts,
service snapshots and history files in the output root are disposable operational
diagnostics. See `resume_launch.json` and `resume_verification.json` there.

## Float16 storage and second quota recovery — 2026-09-07

User requested all Ta trajectory positions and future simulations use float16.
Both production configs now select it explicitly; the elemental converter defaults
to it, and AGENTS.md records the preference for subsequent simulation launches.
MD arithmetic and restart files remain unchanged. Float16 positions can round over
a periodic boundary, so temporal readers decode to float32 and wrap again.

Ta `2.7ns`, `2.8ns`, `2.9ns` had completed conversion; `3.0ns` completed dynamics
but exhausted quota in conversion at 17:10 CEST. Its incomplete 17.4 GB allocated
float32 scratch was removed with an audit in `float16_recovery.json`; the raw
trajectory and final restart were retained. No dynamics are repeated.

The experiment recipe `recover_float16.py` invokes maintained converters to
compress baseline `model_1m`, baseline preflight, and the three completed new Ta
branches, verifies the 3.0ns restart and finalizes its trajectory as float16, then
resumes the final Ta branch followed by Ti. Reproduction (only with no active worker):

```bash
PYTHONPATH=. conda run -n pointnet python experiments/ti_ta_crystallization_20260907/recover_float16.py
```

Detached service: `ta-float16-recovery-v2-20260907.service`, with 48-core affinity.
Log: `datasets/ti_ta_crystallization_20260907/float16_recovery.log`.
The first compression launch verified baseline data but failed removing its old
NFS directory because the converter still held a frame view. The view is now
released before deletion; baseline target checksums were reverified and its old
path restored as a compatibility symlink before restarting the queue.

Each binary has `float16_conversion.json`, original manifest provenance, exact
checksums for unchanged arrays, and measured periodic-coordinate quantization
error. Baseline max error is 0.125 angstrom per coordinate; RMS is 0.031197 angstrom.
Original float32 paths remain compatibility symlinks. Recovery publishes
`float16_recovery_complete.json` before starting the remaining simulations.
The scientific settings and exact original starting configurations are preserved.

New `position_storage.py` is maintained conversion implementation exposed through
`convert_trajectory.py temporal-storage`; `recover_float16.py` and this audit are
experiment records. Logs, service receipts and temporary arrays are disposable
operational files in the output directory.
