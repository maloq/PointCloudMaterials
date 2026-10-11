# Million-atom Al: 0.10 ps velocity recovery

The user selected the **400 ps production trajectory first**. The preceding
300 ps melt is deferred. Original coordinates, checkpoints and nucleus labels
remain unchanged. This collection adds no independent melt ancestry.

Dataset: `al-million-velocity-replay-20261007`.
Parent: `al_meam_1m_450K_400ps_with_melt_20260913T205405Z`.
Recipe: [al_million_velocity_replay_20261007.json](../../configs/simulation/al_million_velocity_replay_20261007.json).
Implementation: [al_million_replay.py](../../src/simulation/campaigns/al_million_replay.py).

Submitted October 7 at 14:52 Paris: audit **1027047**, dependent production
**1027048**. At submission, the audit is pending CPU resources and production
is pending the audit. No replay section is completed yet. Syntax/input inspection
and a read-only native checkpoint/NPT-state restoration check passed; numerical
trajectory agreement remains the job of the queued audit.

## Recovery protocol

Replay 100 sections of 4 ps, each anchored to the original native restart.
The first section reads the final melt restart, resets the production clock and
applies the original one-time velocity scaling to 450 K. Later sections restore
the saved velocities and the matching `ensemble` NPT fix state without rescaling.

The settings follow the original executed input: one million Al atoms,
Lee2003 2NN-MEAM, native CPU float64 forces and integration, 1 fs timestep,
450 K isotropic NPT at zero pressure, 0.1 ps thermostat damping, 1 ps barostat
damping, 96 MPI ranks and a 3×4×8 decomposition. No momentum-removal fix is added.
Potential, restart, input and current executable/library hashes are recorded.
The old run's log records the same LAMMPS version; current binary identity alone
does not establish identity with an unrecorded historical binary hash.

CPU replay is used to maximize historical agreement. The separate V100 shooting
campaign continues with its existing one-GPU limit. This recovery does not use
the float32 FastAl force backend or borrow its 70,304-atom/2-fs/NPT settings.
[LAMMPS restart documentation](https://docs.lammps.org/read_restart.html)
explains why executable, decomposition and matching fix identity matter.

## Gates and interpretation

The first job runs sections 0–4 and 4–8 ps. The remaining production job depends
on successful completion; an audit failure invalidates the dependency rather
than leaving a permanently pending job. Every subsequent section repeats the
same gates and stops on failure:

| Check | Required agreement |
| --- | --- |
| Native endpoint coordinates, minimum image | maximum component error ≤ 1e-5 Å |
| Native endpoint velocities | maximum component error ≤ 1e-4 Å/ps |
| Native endpoint box bounds | maximum error ≤ 1e-7 Å |
| Every atom at every 0.10 ps saved frame | historical float16 coordinate bin, allowing an additional 1e-4 Å for original text/float32 transforms and replay drift |
| Every saved box | bounds agree within 1e-4 Å of the historical float32 box |
| Identities/timeline | exactly 1,000,000 Al IDs and the declared integer steps |

These are declared numerical recovery tolerances, not a proof of bitwise
trajectory identity or independently measured historical intermediate velocities.
The original position quantization RMS is about 0.035 Å. Full-precision endpoint
checks are therefore essential; agreement with coarse stored positions alone
would be insufficient.

If a gate fails, the generated Q/V and native endpoint are preserved as a failed
recovery attempt. They are not attached to the old coordinates or assigned old
birth outcomes. Continuing as a new trajectory would require its own scientific
labeling. The queue never silently switches to that interpretation.

## Outputs and execution

Preparation and submission:

```bash
conda activate pointnet-torch214
python -m src.simulation.campaigns.al_million_replay prepare \
  --config configs/simulation/al_million_velocity_replay_20261007.json
python -m src.simulation.campaigns.al_million_replay submit --launch LAUNCH
```

`LAUNCH` is `${storage:archive}/simulation-launches/al-million-velocity-replay-20261007`.
`submission.json` records the exact jobs owned by this launch. No other jobs are
stopped or altered. One 96-core CPU worker runs at a time: a three-hour audit cap
followed by a 48-hour production cap. The original median was approximately
18 minutes per 4 ps, or 30 hours of integration for 400 ps. Current queue delay,
hardware performance and conversion/publication overhead are additional.
Live `status.json` estimates remaining work from completed replay sections.

Scratch holds only current replay work. Each completed or failed section is
published with checksum verification to
`${storage:archive}/simulations/al-million-velocity-replay-20261007`.
The standard `scripts/convert_trajectory.py offline-dump` verifies float16 Q/V,
float32 boxes, exact identity/timeline arrays and quantization receipts before
deleting temporary text. Native endpoint restarts retain full precision.

After all 100 sections pass, `trajectory_binary_float16/` contains one usable
4001-frame Q/V trajectory. Exact starting checkpoints win at shared 4-ps
boundaries; `frame_segment_index.npy` records those boundaries. Redundant
per-section arrays are removed only after the consolidated export is verified;
their manifests, fidelity/conversion receipts, logs and native endpoints remain.
The trajectory occupies approximately 48 GB, with about 8.8 GB of native
endpoints plus small metadata; temporary sections require additional space.

Consumers must check `status.json` for `complete` and retain the replay provenance.
This does not automatically change the consolidated birth catalogue or count
replayed sections as independent sources. The original 0.75 ps collection and
the other dense/shooting Al campaigns retain their own data and progress.
