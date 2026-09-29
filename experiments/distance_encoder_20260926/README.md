# End-to-end MACE for crystal distance and early spatial detection

[Separate Al/Ta fine-tuning of the completed dense-history model](MATERIAL_FINETUNE.md)
records the material-adaptation comparison begun 27 September.

This encoder is named **CD-MACE128**. See the [local-only evaluation and
CD-MACE128-VC comparison](VCREG.md) for the new regularized treatment.

Train the geometry-only, 128-channel MACE and a head on its exported 128-vector
jointly, using the large existing dynamic multi-material collection. Earlier
distance experiments froze the encoder; this experiment allows distance and
early-warning likelihood gradients to update all its trainable parameters.

Training data after the fixed initial-history exclusion:

| Material | Training neighborhoods |
|---|---:|
| Al | 7,441,328 |
| Mg | 503,328 |
| Ti | 721,000 |
| Ta | 4,082,080 |
| **Total** | **12,747,736** |

Selection contains 192,000 Al neighborhoods from the existing 15 selection
sources. Whole-source benchmark splits are unchanged. Repeated frames and
nearby centers are correlated. The external material families remain train-only,
so this experiment does not establish unseen-material generalization. Existing
static Zr/Mg/Ta inputs are excluded because they lack the past history required
by the confirmed-crystal distance definition. No new simulation is generated.

The encoder starts from the previous observed Al likelihood-trained checkpoint;
the head is initialized fresh. Input is one nearest-80 patch within normalized
radius 8 Å, with 5 Å edges and two MACE interactions. Material normalization is
the established fixed geometric preprocessing. Material IDs, species features,
temperature, absolute time, velocities, histories and relaxed geometry do not
enter the model. Crystal history is used only to construct the target mask.

Objective: censored zero-inflated lognormal distance NLL plus twice a weighted
CDF Bernoulli likelihood at 8/12/20/32 Å. The weights .05/.15/.40/.40 prioritize
larger warning distances without rewarding false alarms or using AP-based
optimization. Distances are Al-equivalent Å, identical to native Å for Al.

One seed, 12 full epochs, global batch **4096**, per-GPU batch **2048** on two
GPUs, cuEquivariance, compiled tensor operations, mixed bfloat16 computation
with float32 likelihoods/master weights. Coordinates remain float32 in VRAM;
learned features are recomputed on every update. Selection is the declared
predictive likelihood, never AP. Mandatory online W&B.

After a complete fit, evaluate the joint head on the original fixed Al samples
and approach/away scans. Then freeze the newly trained encoder and fit the two
selected context predictors—vector messages and harmonic hierarchy—on the exact
previous distance-readout population. These probes use distance NLL. Their
inputs include surroundings; the first encoder-training stage uses local
patches. This distinction matters because earlier warning was mostly explained
by an existing crystal entering the wider context.

[Recipe](../../configs/distance_encoder/multimaterial_early_20260926.json) ·
[metric definitions](../../docs/metrics/distance_encoder.md) ·
[operations and resumption](../../docs/distance_encoder.md).

```bash
python -m src.research.distance_encoder.prepare plan --config configs/distance_encoder/multimaterial_early_20260926.json
python -m src.research.distance_encoder.prepare lane --plan LABEL_ROOT/plan.json --lane 0 --workers 4
# Repeat the other declared lanes, then seal the full release.
python -m src.research.distance_encoder.prepare seal --plan LABEL_ROOT/plan.json
python -m torch.distributed.run --standalone --nproc_per_node=2 -m src.research.distance_encoder.train --config configs/distance_encoder/multimaterial_early_20260926.json
python -m src.research.distance_encoder.context --config configs/distance_encoder/multimaterial_early_20260926.json
```


## Relaxed static Al structure analysis

The completed original CD-MACE128 encoder (epoch 12, 37,356 updates) now has a
[six-snapshot gallery](../../output/structural_static/cd-mace128-epoch12-al-20260926/index.html)
and [linked structure explorer](../../output/structural_static/cd-mace128-epoch12-al-20260926/analyses/exploration-v1/data/explorer.html).
This analyzes 684,723 interior neighborhoods and the native 128-D export before
the distance head. It is descriptive transfer from observed training geometry to
relaxed static Al, not held-out prediction evaluation. The separate clustering
has its own cluster IDs. At PTM RMSD <=0.10, C1 is 97.4% crystalline and C7 has
6.0% ICO matches; the atlas and radial profiles expose within-cluster variation.
[Reproduction protocol](../../docs/structural_static_analysis.md#cd-mace128-distance-supervised-encoder).
