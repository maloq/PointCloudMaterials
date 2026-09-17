# H200 task: shared pretraining with batch size 1,024

Run the following on **one H200**, independently of the local batch-512 campaign.
Use the supplied source archive and nine configs in
[`configs/shared_pretraining/h200_batch1024/`](../configs/shared_pretraining/h200_batch1024/).
Do not launch the local Slurm `submit` recipe on this server: its allocation IDs
belong to the other cluster. The new `queue serial` command needs no Slurm.

## Task for the H200 agent

Implement no new scientific method for this handoff. Verify the transferred code
and two prepared releases, check the three proposed microbatches on the H200,
then run the supplied queue detached. Start fresh structural fits with **12 epoch
equivalents, batch size 1,024, seed 20260919, peak LR 0.02, 10% linear warmup and
cosine decay to 0.0002**, followed by causal training and frozen evaluations.
Use online W&B throughout. Keep the supplied run IDs and output directories.

The queue finishes one variant's structural → causal → analysis pipeline before
starting the next, in this order:

| Variant | Encoder input | Initial GPU microbatch | Statistical batch |
| --- | --- | ---: | ---: |
| MACE + VICReg | Current positions/species | 352 | 1,024 pairs |
| GATr + VICReg | Current positions/species | 832 | 1,024 pairs |
| GATr + temporal JEPA/SIGReg | Three observed position frames | 96 | 1,024 pairs |

The microbatches are **starting values, unmeasured on H200**: twice the local
values, with an 80 GiB allocator limit. If a memory check fails, reduce only the
microbatch in that variant's structural and causal configs and record the value.
The full-batch objective must still see all 1,024 pairs through gradient caching.
Do not silently reduce statistical batch size, add mixed precision, change model
width, rescale SIGReg, increase LR, alter targets, or reuse a batch-512 optimizer.
Use MACE/cuEquivariance and the existing GATr implementation from this archive.

Keep geometry and **instantaneous TDA** anchors on the exported z128 state.
Keep the fixed material radius calibration and within-material/potential
regularization batches. Dynamic VICReg uses spatial or temporal neighbors;
dynamic JEPA predicts the next separately encoded snapshot, with actual time
offsets. Static samples have only spatial views and no next-time prediction.
Inputs do not include velocities. See the repository meanings of
[VICReg](research_glossary.md#vicreg-in-the-earlier-encoder-protocols),
[temporal JEPA/SIGReg](research_glossary.md#lejepa-and-sigreg-in-the-structural-pretraining-proposal)
and [instantaneous topology](research_glossary.md#instantaneous-topology).

Initialize each causal fit from **its own H200 structural best checkpoint**.
Retain present/representation objectives and add fixed physical/TDA targets at
**0.75, 3 and 9 ps**. Every fourth update replays the broad structural release.
Preserve all 90/15/15/30 train/selection/calibration/test source roles. Do not
generate data or recalculate the prepared labels. No relaxed TDA is used.

Run the existing frozen analysis after each completed causal fit. Evaluate both
its structural parent and causal encoder with identical ridge/nonlinear probes,
source-balanced physical/TDA errors and persistence comparisons. **Probe batch
size remains 1,024**, matching the local evaluation; only encoder training
batches double. Preserve predictions, source IDs, metric definitions and source
bootstrap intervals. Do not rank different encoders by raw latent MSE.

Use a **16-hour first slot**, or stop before the actual allocation end if earlier.
The queue checkpoints unfinished work and resumes from the same command with a
new deadline. Do not shorten the 12-epoch budgets to fit this slot. Completion
time is unknown until the H200 runs; doubling batch size does not imply twice
the throughput. Verify advancing finite-loss updates, a saved checkpoint and an
online W&B URL, then leave it detached. Return the PID/job, URLs, progress,
measured memory and remaining queue. Wait for the user to request interpretation.

## Files to transfer

Send the [prepared code archive](../output/shared_pretraining/h200-handoff-20260918/technical/h200-shared-b1024-20260918.tar.gz), including its `handoff/`
checksums, and these **exact prepared directories**. The older H200 raw data and
previous causal-memory caches do not replace these releases.

| Release | Destination beneath `roots.cache` | Logical size |
| --- | --- | ---: |
| Five-metal structural observations, 250,000 training records + 480 selection | `structural_pretraining/broad-250k-v2-20260917/` | 25.74 GiB |
| Native Al causal/evaluation observations, 38,400 windows | `shared-causal-38400-20260918/` | 3.53 GiB |

Total prepared inputs: **29.27 GiB** before transfer compression. Source locations
are in the archive's `handoff/transfer.json`; preserve every manifest, plan,
receipt and shard. The checksum list covers all files, using producer-recorded
SHA-256 values for arrays. There is no need to transfer the 171 GiB raw archive,
old results, local machine settings, credentials or local batch-512 checkpoints
to execute these fresh fits. Absolute original source paths inside the manifests
are provenance; runtime reads the prepared shards under `roots.cache`.

Extract code into a **new, stable directory**, not over a running checkout.
Verify `sha256sum -c handoff/code.sha256` from that directory. Copy both releases
beneath the H200 cache root, then run `sha256sum -c /ABSOLUTE/CHECKOUT/handoff/prepared-data.sha256`
from that cache root; save the verification log. Do not edit manifests to change
paths. Set only the H200 checkout's ignored `machine.local.yaml`, for example:

```yaml
roots:
  cache: /ABSOLUTE/H200/CACHE
  output: output
execution:
  backend: local
  device: cuda:0
```

Confirm manifest identities:

```text
structural: 61d5ada797bfb77cf56d92dc17b08c3f7954f37e212e2acdbfcae8c68009332e
causal:     1f0322dc51a99ec84cdd7e135183bdab20a665838c2f7927f4689608d9663aa3
```

The archive includes the exact local scientific source, dependency requirements,
metric contracts, and separate H200 configs. Runtime/encoder/objective/analysis
code matches the frozen local campaign. Only queue orchestration, the explicit
preflight memory limit, configs and descriptive documentation differ. Start
H200 fits from these bytes and keep the checkout path stable for exact resume.

## Environment and short preflight

Use conda `pointnet` and the supplied pinned environment files. The tested stack
uses Python 3.12, torch 2.11.0+cu128, mace-torch 0.3.16, cuEquivariance 0.10.0 and
W&B 0.26.0. Preserve the GATr and LeJEPA commits in
`environments/requirements-backbone-v2.txt` and
`environments/requirements-structural-pretraining.txt`; the latter two requirement
files are installed with `--no-deps` alongside the GPU/core environment. Check
the existing server environment before changing it. Use that server's own W&B
authentication for `teshbek/PointCloudMaterials`; never copy credentials.

From the extracted checkout, with the selected H200 visible as CUDA device 0:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PCM_PROJECT_ROOT="$PWD"
python -m pytest -q tests/test_shared_pretraining.py tests/test_shared_pretraining_queue.py
python -c 'from src.experiment_runner.metric_docs import check_metric_docs; check_metric_docs(family="shared_pretraining")'
python -m src.training_methods.shared_pretraining.profile --config configs/shared_pretraining/h200_batch1024/mace_vicreg_structural.json --microbatch 352 --memory-limit-gib 80 --output output/shared_pretraining/h200-b1024-preflight-20260918/technical/mace.json
python -m src.training_methods.shared_pretraining.profile --config configs/shared_pretraining/h200_batch1024/gatr_vicreg_structural.json --microbatch 832 --memory-limit-gib 80 --output output/shared_pretraining/h200-b1024-preflight-20260918/technical/gatr-vicreg.json
python -m src.training_methods.shared_pretraining.profile --config configs/shared_pretraining/h200_batch1024/gatr_lejepa_structural.json --microbatch 96 --memory-limit-gib 80 --output output/shared_pretraining/h200-b1024-preflight-20260918/technical/gatr-jepa.json
python -m src.training_methods.shared_pretraining.profile --config configs/shared_pretraining/h200_batch1024/gatr_lejepa_structural.json --microbatch 96 --memory-limit-gib 80 --material Al --output output/shared_pretraining/h200-b1024-preflight-20260918/technical/gatr-jepa-al.json
```

These are separate disposable one-update memory checks on large-support training
observations. Their weights never initialize a scientific fit, and their cold
timings are not a throughput comparison. The explicit Al history check covers
three-frame inputs even if the largest generic support belongs to static data.
Inspect finite loss/gradients and peak memory, then stop profiling. Save any
microbatch correction before launching. During causal startup, check memory on
the native release too; resume with a smaller microbatch if needed, preserving
statistical batch and saved state. Do not repeatedly benchmark inside training.

## Detached launch and continuation

After verification, use the existing serial queue. This example budgets 16 hours
from launch; replace the deadline with the actual allocation end minus a safety
margin when earlier. On a scheduled server, run it inside the allocated job.

```bash
mkdir -p output/shared_pretraining/h200-b1024-campaign-20260918/technical
h200_deadline=$(date -u -d '+16 hours' +%Y-%m-%dT%H:%M:%S+00:00)
nohup python -u -m src.training_methods.shared_pretraining.queue serial \
  --plan configs/shared_pretraining/h200_batch1024/campaign.json \
  --deadline-utc "$h200_deadline" \
  > output/shared_pretraining/h200-b1024-campaign-20260918/technical/queue.log 2>&1 < /dev/null &
```

Record `$!` immediately in the handoff receipt. The controller locks its output,
keeps `technical/queue-state.json`, and stops when a stage is incomplete. Later
phases require completed parents. Fatal errors stop the queue and require
inspection. Resume with the same plan, outputs, code, seed and W&B IDs, supplying
a new deadline; completed stages exit without retraining. Allocation changes
restore optimizer, schedule, RNG and SIGReg state. Do not move a live checkpoint
between different checkout paths or overwrite the local campaign.

W&B group: **`shared-12ep-b1024-h200-20260918`**. Each stage records its URL in
`technical/wandb_run.json`. Return all nine run folders plus preflight/controller
receipts: selected `best.pt`, exported `encoder.pt`, `last.pt` until confirmed
complete, identities, configs, logs, probe weights/predictions, tables and plots.

## Interpretation of the batch comparison

Structural training uses **2,930 updates / 3,000,320 anchor draws** (12.00128
epoch equivalents), versus 5,860 updates locally. Warmup is 293 updates. Causal
training uses **270 native + 90 replay updates**, versus 540 + 180 locally;
native/replay draws stay 276,480/92,160, and warmup is 36 updates. Validation and
checkpoint intervals halve to 64 and 16, respectively, matching draw intervals.
Sampling remains stochastic; matching the seed does not make individual batches
or selected checkpoints identical across batch sizes.

This is a **batch-size experiment at equal data exposure**, with changed
optimizer-update counts and different hardware. Full-batch VICReg/SIGReg
statistics also change with batch size. In particular, upstream Epps–Pulley
SIGReg includes a sample-count factor, so a fixed non-Gaussian distribution can
receive a different penalty at 1,024. Keep the declared formula and coefficient
and report its measured magnitude. Quality comparisons use the same held-out
physical/TDA metrics; this experiment alone cannot isolate a hardware speedup.
