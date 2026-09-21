# Short-horizon frozen information diagnostic

Use conda pointnet-torch214. Scientific protocol:
[experiment](../experiments/crystallization_information_20260920/README.md).

`python -m src.research.crystallization_information --config configs/analysis/crystallization_information_short.json`
runs prepare, matched probes/decoders and report. Stages `prepare`, `fit`, `report`
are separately available. Fit resumes selected and last head checkpoints. Lane
count is fixed by config; `--stage fit --lane N` runs one lane. No encoder is
modified. Inputs are checksummed existing observations and frozen feature exports.

Production runs use an immutable source snapshot and a standalone CPU Slurm job,
8 cores,32GiB, four probe lanes with two Torch threads each. No GPU allocation is
cancelled or shared. Logs, feature identities, task states and checkpoints live
under the run's technical directory. Tables retain metric definitions and hashes.
Run `--stage report` to regenerate the partial report from completed fits.
