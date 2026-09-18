# Mixed-material snapshot GATr with temporal curvature

Initial recipe: `configs/shared_pretraining/gatr_mixed_triplets/`. This is a
fresh twelve-epoch-equivalent fit. The exported encoder still accepts one
snapshot and emits 128 channels.

This initial stage was checkpointed at update 250 and continues under the
[temporal-only backtracking recipe](shared_pretraining_temporal_backtracking_20260918.md).
Its original configuration and frozen implementation remain historical records.

## Normalization fix

The failed broad continuation trained heads with homogeneous-domain moments
but calibrated evaluation with pooled material moments. At identical step-zero
weights, restoring the parent's Al moments changed the Al selection score from
0.736723 to 0.121529 in the new target units. This isolates a normalization
mismatch; it does not establish that later diverged weights are useful. See
the failed run's `technical/normalization-diagnosis.json`.

New batches mix all material/potential groups. Heads normalize separately within
each group over the complete statistical batch. Before evaluation, fixed
training-only references at current encoder weights fit group-specific moments.
Both paths use population variance. Learned group affine parameters let
physical/TDA heads express material-specific means. VICReg and physical
correlation are computed within groups and averaged with their batch fractions.
Material separation cannot satisfy the variance term. Encoder inference uses
no batch statistics or group-dependent head parameters.

## Data and objective

The existing full-instantaneous-TDA release is reused. Static shards are excluded
before opening arrays, constructing rows, or fitting target moments. No new
simulation, TDA preparation, or relaxed target is required.

| Dynamic training group | Anchors | Pairs per update |
| --- | ---: | ---: |
| Al / Lee 2003 MEAM | 142,364 | 1,141 |
| Al / Mendelev 2008 EAM | 15,016 | 128 |
| Mg / Wilson 2016 EAM | 29,820 | 239 |
| Ta / Zhong 2014 EAM | 29,820 | 239 |
| Ti / Kavousi 2019 MEAM | 37,500 | 301 |

Sampling is proportional with a minimum of 128 pairs per group and
largest-remainder rounding. Draws are without replacement within an update,
with possible repeats across updates. Twelve epoch equivalents over 254,520
anchors give 1,492 updates at 2,048 anchors, not twelve shuffled passes.
Zr is absent because this release contains only static Zr.

Updates use spatial or temporal VICReg pairs with probability one half. Every
anchor also supplies its tracked center at previous/current/next times, each
encoded independently. Only the original two pair endpoints receive
physical/TDA supervision. Past-frame placeholders never enter a decoder loss.
For spatial pairs, the next temporal frame is encoded only for curvature.

The new term is `0.001 * mean(sum(residual**2, channels))`. With equal time gaps,
`residual = z_next - 2*z_current + z_previous`, exactly the requested formula.
For unequal positive gaps `h_previous, h_next`, it is
`2*(h_previous*(z_next-z_current) + h_next*(z_previous-z_current)) /
(h_previous+h_next)`. Constant motion therefore has zero penalty where shooting
cadence changes. There is no inverse-time-squared amplification.
The physical, 0.25 TDA, 0.1 normalized VICReg and 0.1 physical-correlation
terms remain. See [metric definitions](metrics/shared_pretraining.md).

## Execution

Batch 2,048 doubles the previous 1,024. Encoder microbatches are 256, with
gradient caching across the statistical batch. Compiled selective BF16 retains
FP32 geometry and FP32 heads/statistics. Allocator cap: 40 GiB. Head/encoder peak
LRs are 0.002/0.0002, with 150-update warmup and cosine decay to 1% of peak.
One seed, online W&B, checkpoints every 16 updates, selection every 64 updates.
Existing learning-health gates remain.

Future submissions use the [compact W&B layout](shared_pretraining_logging.md).
The already submitted September 18 fit uses its original frozen logger.

```bash
conda run -n pointnet-torch214 python -m src.training_methods.shared_pretraining.queue submit \
  --plan configs/shared_pretraining/gatr_mixed_triplets/campaign.json
```

The queue freezes code/configuration and starts detached in H100 allocation
997799. No additional allocation is reserved: measured throughput fits its
remaining time. Checkpoints still permit an exact continuation if needed.
Submit once; consult the receipt before retrying.

- Fit: `output/shared_pretraining/gatr-mixed-triplets-20260918/technical/`
- Queue: `output/shared_pretraining/gatr-mixed-triplets-campaign-20260918/technical/`
- Preflight: `output/shared_pretraining/gatr-mixed-triplets-checks-20260918/technical/`
- W&B: `teshbek/PointCloudMaterials/gatr-mixed-triplets-0918`

Selection remains 480 native-Al observations from 15 held-out sources. Group
training curves do not establish held-out generalization to Mg/Ti/Ta. This run
bundles normalization, fresh training, filtering, batch doubling and curvature;
it cannot isolate curvature's contribution without a matched zero-weight run.

## Preflight evidence

28 relevant tests passed, including grouped train/eval normalization, training-only
sampling, irregular timestamps, gradients through all three states, and cached/full
batch update agreement. A real-data compiled BF16 check at batch 2,048 completed
both pair types with finite loss/gradients, peak allocation 25.6 GiB, two compiled
graphs and no graph breaks. Temporal updates took 16.2–16.3 seconds after compile;
the spatial update included compilation. Selection moved from 0.80307 to 0.77836
after four updates, still worse than the 0.50976 mean baseline as expected this
early. This validates startup only, not eventual training quality. The production
run restarts from the same seed with a fresh optimizer and no pilot checkpoint.
