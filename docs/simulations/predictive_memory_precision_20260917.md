# Predictive-memory precision sources — 17 September 2026

**Stopped at the user's request on 17 September, about 13:45 CEST.** Three
500 K training sources (000–002) completed and were verified in STORE. Source
003 stopped during melt preparation and source 004 during measurement; both
partial runs and available native restart files were copied and hash-verified
under their `-failed` STORE records. Seven sources never started. Worker status
uses `failed` for the interrupted LAMMPS subprocesses (exit 143); the explicit
user-requested stop is recorded in `user-stop.json` beside the durable launch
records. Both CPU workers have exited. Do not restart this campaign automatically.
The [stop update](../../output/predictive_memory/research-summary-20260917-stopped/RESULTS.md)
records the new research results and preservation state. The description below
retains the original campaign plan.

The user authorized new simulations on 17 September, superseding the earlier
existing-data-only restriction. The active H100 optimization comparison continues
on its original immutable data; none of these new sources enter that comparison.

The [recipe](../../configs/simulation/predictive_memory_precision.json) prepares
12 independent melt lineages: six each at 500 and 520 K. Per temperature, three
lineages are assigned train, one validation and two sealed test before simulation.
All new melt/quench seeds are checked against both original 150-source manifests.
Shared FCC preparation is retained, but every lineage has its own 300 ps melt
at 1325 K and its own quench velocities. Split membership follows the whole lineage.
The four sealed test sources are reserved for a future locked evaluation protocol;
their future observables, target distributions and model scores must not inform
development. Numerical integrity and predeclared preparation QC remain permitted.

The physical kernel matches the existing source family: 70,304 Al atoms,
Lee2003 2NN-MEAM with the same pinned potential hashes, 3 fs integration,
isotropic Nose–Hoover NPT at zero pressure, 0.3 ps thermostat and 3 ps barostat,
and center-of-mass momentum removal every 0.3 ps. After the 300 ps melt and
15 ps equilibration, measure 192 ps without velocity resets, outcome-dependent
stopping or measurement PTM screening. Each source has 2,561 frames at 0.075 ps
cadence, including both endpoints. The predeclared melt QC uses the existing
less-than-1% crystalline-fraction check only on the 1325 K prepared melt.

This shorter measurement covers an earlier regime than the old pilot's 300 ps
anchor. It is intended for precision and temporal-resolution studies, and is
not an automatically matched confirmation of that old anchor protocol. Its
duration permits 96 ps history and 96 ps future at a 96 ps anchor; shorter
histories allow more matched anchors. Dense input must be declared explicitly
in future experiments, not silently substituted into the old coarse-cadence fits.

Each output keeps full-resolution native restart files and a paired float32
observation reference alongside canonical float16 positions/velocities. Box
bounds remain float32 and identity/timeline arrays exact. Both variants are
verified, with every rounding value checked and maximum/RMS coordinate and
velocity errors recorded. Only then is the new temporary ASCII trajectory
removed. The float32 reference is retained for full-box versus local-centered
rounding audits; those scientific audits and matched encoder fits remain to be
performed. Casting old data to float32 cannot provide that reference.

New runs stage at `${storage:simulation_runs}/memory-al-precision-20260917/`.
Each source is self-contained, retaining exact input/FCC/potential files, campaign
manifest, split, input hashes, melt validation, source stdout, conversion receipt,
native restarts and completion state. Completed sources publish with hash-verified
copies to STORE and receive dataset IDs `memory-al-precision-20260917-sourceNNN-TTTT`.
Stopped failures are archived with their restart state; they are never silently
rerun or counted as fresh sources. The immutable source/config snapshot and CPU
submission records live in the sibling `memory-al-precision-20260917-launch/`.
That sibling is a link to durable `${storage:archive}/simulation-launches/`
storage. The submitted Slurm array is **995981**, with workers **995981_0** on
nodecpu03 and **995981_1** on nodecpu06 at launch. Each tracked worker executes
the frozen source copy, checks its SHA-256 inventory, and retains an environment
record, original Git revision/patch, immutable manifest, source archive and logs.

Two detached CPU workers each process six sources serially with 48 MPI ranks.
They request 64 GiB and 36 hours per worker, and use no GPU. Earlier 48-rank source
timings suggest roughly 12–20 hours after scheduling; denser text output,
conversion, publication and different CPUs can extend this estimate. Retained
paired arrays total about 78 GB across the 12 sources, plus native/input artifacts.
The 256-atom, 50-step local smoke fixture is infrastructure validation only and
is excluded from the campaign and research counts.

Preparation:

```bash
python scripts/run_lammps_campaign.py memory-sources prepare \
  --config configs/simulation/predictive_memory_precision.json \
  --run-name memory-al-precision-20260917
```

Execution inside each allocated CPU worker:

```bash
python -m src.simulation.campaigns.memory_sources run-worker \
  --campaign-root CAMPAIGN_ROOT --worker-index 0 --workers 2
```

Use worker indices 0 and 1. Input hashes and immutable manifest are checked before
execution. Any existing run status causes a loud error requiring inspection;
retrying never overwrites partial trajectories. Worker progress lives in
`workers/worker-INDEX.json`; individual simulation progress appears in native
LAMMPS logs. The conversion can be explicitly finalized from retained completed
dynamics with `convert_trajectory.py memory-pair RUN_DIR` after inspecting a
conversion failure; rerunning dynamics is unnecessary.
