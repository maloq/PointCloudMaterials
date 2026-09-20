# MACE on both node61 GPUs with four times the dynamic data

Recipe: [`configs/shared_pretraining/mace_expanded_dual/`](../configs/shared_pretraining/mace_expanded_dual/).
This is a fresh five-epoch-equivalent MACE fit on both RTX PRO 6000 Blackwell
GPUs in allocation 999611. The completed node58 fit remains a reference;
GATr's H100 job is independent and is not moved or stopped.

## Data preparation

| Dynamic training stratum | Previous anchors | Expanded anchors |
| --- | ---: | ---: |
| Al native | 62,500 | 250,000 |
| Al shooting | 75,000 | 300,000 |
| Other Al (million-atom MEAM/EAM trajectories) | 19,880 | 79,520 |
| Mg | 29,820 | 119,280 |
| Ti | 37,500 | 150,000 |
| Ta | 29,820 | 119,280 |
| Total | 254,520 | 1,018,080 |

The 763,560 added anchors use source/frame combinations absent from the parent's
anchor tasks. Within each stratum, new counts balance ancestral lineages and
then trajectories. If a short trajectory lacks enough unused frames, its excess
quota is redistributed within the same lineage. Every eligible trajectory must
have unused frames; exhaustion fails with source context. Selection and test
ancestries never enter new training tasks. Counts describe observations, not
independent trajectories or guaranteed nonoverlapping neighborhoods.

All parent shards are inherited immutably, including 480 unchanged native-Al
selection anchors. The cache retains inherited static shards for provenance,
but the trainer excludes every static row, including Zr. New shards contain
only the local radius-8 support, with the same fixed material scaling, smooth
6–8 taper and radius-4.25 spatial partner selection. Temporal frames retain
tracked identities and measured physical intervals. Every supervised current,
spatial and next-frame view has instantaneous TDA144. No relaxed TDA, velocities
or new simulations are added. Generating potentials follow the parent registry.

Registered dataset: `structural-neighbors-dynamic-1018080-20260919`.
Preparation uses 12 CPU workers in the existing allocation and frozen source at
`output/shared_pretraining/mace-expanded-data-20260919/technical/code`.
Status lives in the release's `status.json`; the detached launch and log are in
that preparation campaign's `technical/` directory. Interrupted preparation can
resume with that same frozen source and configuration; complete shard receipts
are checked against the immutable release identity.

## Training and two-GPU execution

One model, seed 20260919, global batch 2,048, encoder microbatch 512 on each GPU,
compiled selective BF16 with FP32 geometry, 40 GiB allocator cap per GPU.
Five epoch equivalents give 2,486 updates. Each update independently samples
anchors; an epoch equivalent is sampled anchors divided by training-set size,
not a guarantee that every observation has been visited.

Retains within-material/potential VICReg, physical85 and instantaneous-TDA144
anchors, equivariant q4m/q6m supervision at weight 0.1, and temporal-only
backtracking at weight 0.69. Heads peak at LR 0.002, encoder at 0.0002, with
10% warmup and cosine decay to 1%. Validation remains the same 15 native-Al
sources, so it does not establish held-out generalization for other metals.
This larger-data fit also has four times as many updates as the reference;
it is not a compute-matched data ablation.

Whole packed graph microbatches are dispatched to two independent encoder
replicas. Ordered features return to GPU 0 for one global grouped objective.
Its exact feature derivatives are replayed on each GPU. Encoder gradients are
summed (not averaged), clipped together with the head gradients, and one optimizer
step updates the primary model. Replica weights are then synchronized. Heads
and the statistical VICReg batch are never divided into per-GPU losses.
Validation runs on GPU 0. Checkpoints/export retain the original single-encoder
state schema; secondary replicas are runtime objects, not checkpoint parameters.
The implementation specifically supports deterministic StructuralMACE.

Eight preparation threads and next-update prefetch overlap CPU packing with GPU
computation. Profiling is separate from the production fit; no hardware benchmark
is inserted into training. Per-update operational timings stay in technical logs;
W&B retains the compact scientific dashboard and its native GPU monitoring.

## Launch

```bash
conda run -n pointnet-torch214 python -m src.training_methods.shared_pretraining.queue submit \
  --plan configs/shared_pretraining/mace_expanded_dual/campaign.json
```

This one-shot submission freezes the tested trainer and starts a detached worker
in allocation 999611 with both GPUs. It waits for the separately running data
build to finish successfully, checks full TDA coverage, then starts training.
A failed build blocks training loudly; an incomplete cache is never consumed.
The allocation deadline reserves time for checkpointing. No continuation
allocation is requested by this recipe.

Campaign receipt/logs: `output/shared_pretraining/mace-expanded-dual-campaign-20260919/technical/`.
Training: `output/shared_pretraining/mace-expanded-dual-20260919/technical/`.
W&B ID: `teshbek/PointCloudMaterials/mace-4x-dual-0919`.

Correctness checks include actual dual-GPU FP32/BF16 parity with single-GPU
updates, spatial pairs and temporal triplets, uneven graph/microbatch sizes,
global domain statistics, bond-order gradients and exact replica synchronization.
Data tests cover held-out/static exclusion, unused source/frame pairs, source
coverage, exact TDA reproduction and parent immutability. Full-batch compiled
preflight is retained separately in `mace-expanded-dual-checks-20260919`.

Launch receipt: detached Slurm step submitted at 23:57 UTC on 18 September
(01:57 local on 19 September), controller PID 379522 in allocation 999611.
The full cache was complete before training submission: 4,891 shards,
1,018,080 dynamic training anchors and 480 unchanged selection anchors.

Validation before launch: 3 actual dual-GPU parity cases and 11 data/queue tests
passed. Six full-batch compiled updates plus training-only head calibration and
selection inference passed, with no graph breaks. Warm update computation took
0.65 s for spatial batches and 0.96–0.99 s for temporal batches, approximately
13.9 GiB peak allocated per GPU. These exclude input preparation and are not
end-to-end throughput claims. The separate input preflight confirmed loader cost
remains material; training overlaps next-batch preparation but full GPU saturation
is not guaranteed.
