# Mixed MACE with equivariant bond-order supervision

The node58 recipe is `configs/shared_pretraining/mace_mixed_bond_order/mace.json`.
It initializes a new MACE from scratch for five epoch equivalents (622 updates)
on the existing 254,520 dynamic Al/Mg/Ti/Ta anchors. Static data is excluded.
It preserves the GATr run's 2,048-anchor statistical batch, grouped VICReg,
physical85 + instantaneous TDA144 heads, and temporal-only backtracking.
Spatial updates encode two independent snapshots; temporal updates encode
current/next/previous independently. Past frames are curvature context only.

Head LR warms up to 0.002, encoder LR to 0.0002, then cosine-decays across the
five-epoch budget. AdamW, clipping, seed, group quotas, selection sources and
learning-health checks match the mixed GATr setup. cuEquivariance and full-graph
compilation are enabled, using BF16 scalar work with FP32 geometry/tensors.

The added head consumes the tracked center's learned MACE l=2 tensor channels,
before invariant pooling, and forms l=4 and l=6 with equivariant tensor products.
It predicts 22 real q4m/q6m components, never from the exported invariant state.
Targets use 12 nearest supported neighbors. The head has no coordinate or target
input. Bond-order loss has weight 0.1 and a fixed isotropic random-bond scale;
see [all formulas and limitations](metrics/shared_pretraining.md).
This auxiliary encourages tensor information but is not proof that the invariant
state retains it. Physical/TDA decoders still constrain that exported state.
A high-order tensor formed from l=2 features may have limited expressiveness in
highly symmetric environments; the new component and magnitude metrics expose
that limitation rather than claiming it solved in advance.

The export `technical/encoder.pt` retains the normal 128-state StructuralMACE
interface and state dictionary. Load it with `StructuralMACE(backend='cueq')`,
then `load_state_dict(checkpoint['encoder'])`. Full `best.pt` / `last.pt` also
contain the auxiliary head and grouped physical/TDA readouts. The new protocol
identity is `shared_pretraining_mixed_mace_bond_v9`; it is not a GATr continuation.

## Preflight and loading

Disposable checks live in
`output/shared_pretraining/mace-mixed-bond-order-checks-20260918/technical/`.
They use only training observations and never supply weights to the fresh fit.
`calibration.json` records three full temporal batches used to choose one fixed
backtracking weight with the same 2%-loss / 10%-encoder-gradient policy as the
GATr transition. The MACE coefficient is calibrated independently.
`throughput.json` separates exposed input-wait from completed GPU update time;
the first calibration batch includes compilation warmup.

Training prepares the next complete statistical batch in a background thread
while the current batch executes. Four CPU workers build independent observation
graphs and targets; ordered collection preserves the sampler's exact row order.
The shared LRU has locked mutations, including concurrent validation requests.
Microbatches are pinned in host memory and
copied asynchronously; the current full batch remains resident on the GPU for
both gradient-cache passes. Timings and peak memory stay in local JSONL only.
W&B uses built-in GPU monitoring and the compact loss/validation dashboard,
adding just weighted bond-order loss and its validation error. Production uses
a 64-snapshot physical microbatch under the 40 GiB allocator limit, while the
statistical VICReg batch remains 2,048 anchors.

The preflight selects a fixed curvature weight of **2.7**, giving 0.65–0.69%
of initial temporal loss and 9.4–9.8% of its encoder gradient norm on the three
calibration batches. The scalar target was limited by the gradient criterion.
Initial serial graph preparation exposed an 11.2% wait fraction on one measured
spatial update; the parallel loader is tested separately in `feeding.json`.

Tests cover q4/q6 agreement with the existing complex-harmonic producer, known
FCC values, rotations/reflections, FP32 tensor operations under BF16, atom
permutation, packed centers, auxiliary gradients through caching, and exclusion
of past labels. Actual node58 CUDA tests exercise the cuEquivariance path.

## Launch

Use conda `pointnet-torch214` and the existing queue command:

```bash
python -m src.training_methods.shared_pretraining.queue submit \
  --plan configs/shared_pretraining/mace_mixed_bond_order/campaign.json
```

The queue freezes source/configs and starts a detached worker inside the recorded
node58 allocation. Launch details are in the campaign's `technical/submissions.json`.
Monitor the run's `technical/status.json`, `updates.jsonl` and W&B link. Checkpoint
selection remains physical + 0.25*TDA for comparison with GATr. The native-Al
selection population does not establish held-out other-metal generalization.

With four preparation workers, the six measured steady-state updates at microbatch
96 exposed only 2.1–2.9 ms of input waiting (less than 0.02% of total time),
including spatial/temporal transitions. Peak allocation was 33.1 GiB. However,
spatial updates took 15.1–15.4 s versus about 9.7 s in the original microbatch-64
profile. Production therefore retains microbatch 64 and uses the parallel loader;
its actual data-wait times are checked again after launch. These are short
operational measurements, not a scientific learning comparison.

The detached worker was submitted at 21:32 UTC on September 18 into allocation
**999600**, node58 (allocation end: September 19, 01:13 UTC). The run is
[`mace-mixed-bond-0918`](https://wandb.ai/teshbek/PointCloudMaterials/runs/mace-mixed-bond-0918).
Artifacts: `output/shared_pretraining/mace-mixed-bond-order-20260918/`.
Campaign: `output/shared_pretraining/mace-mixed-bond-order-campaign-20260918/`.
All 31 CPU checks passed (one CUDA-only test skipped there); all seven bond-order
checks passed on node58, including the CUDA-only test. Metric contracts pass.

Production verification: the first three completed temporal updates at microbatch
64 took 14.72–14.74 s each, allocated about 23.0 GiB, and exposed 2.3–2.6 ms
of data waiting (under 0.02%). A GPU observation during these updates reported
100% utilization. Initial validation and checkpoint export passed; three compiled
graphs and no graph breaks were recorded. The exact startup rows are preserved
in the checks directory as `startup-verification.json`.
