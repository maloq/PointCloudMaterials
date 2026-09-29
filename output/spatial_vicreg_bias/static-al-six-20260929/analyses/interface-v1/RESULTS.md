# Six relaxed Al snapshots: interface cluster comparison

[Interactive gallery](index.html) · [Exact metric definitions](tables/METRICS.md) · [All metrics](tables/metrics.csv)

Completed frozen inference on 684,723 grid centers across 166, 170, 174, 175, 177 and 240 ps. Each neighborhood uses the full 1,048,576-atom source as context. Sixteen views compare two GeoFormer checkpoints, their projectors, TDA, bond order, CNA and the balanced descriptor combination; every page includes two dense MD panels and a snapshot selector. No neural training or cluster refitting.

The table below compares each **raw encoder** with the frozen joint descriptor partition, restricted to centers within 12 Å of the accepted crystal-side interface. ARI is adjusted for chance; numeric cluster IDs need not match. These scores describe correspondence, not prediction.

| Snapshot | Centers ≤12 Å | S1 seed17 epoch24 ARI | S0 seed17 epoch4 ARI |
| --- | ---: | ---: | ---: |
| 166ps | 1,103 | 0.404 | 0.380 |
| 170ps | 7,474 | 0.470 | 0.419 |
| 174ps | 23,438 | 0.488 | 0.427 |
| 175ps | 31,049 | 0.507 | 0.428 |
| 177ps | 50,206 | 0.530 | 0.434 |
| 240ps | 57,351 | 0.463 | 0.323 |

Both encoders agree substantially with joint descriptors around established interfaces. This does not establish distinctive liquid structure or a precursor. In connected PTM-unclassified matter within **20 Å**, with **zero PTM-crystalline atoms in the exact 80-atom input**, mutual information is only 0.0018–0.0043 nats for S1 epoch24 and 0.0035–0.0067 nats for S0 epoch4 across 166–177 ps. At 240 ps this population has zero rows and no score is defined.

For example, at 177 ps this restricted population contains 11,600 centers. S1 assigns 10,786 to one neural cluster; the largest descriptor cluster contains 9,864. Low correspondence therefore accompanies coarse, imbalanced partitions. It does not prove that continuous embeddings lack physical information. The 20 Å crystal-free population also differs in distance distribution from the 12 Å interface population, so their difference is not a controlled causal effect of removing crystal from the input.

Interpretation limits: these are relaxed static structures with unknown generating potential and incompletely recorded ancestry; no periodic cell or persistent atom correspondence is inferred. All cluster models were fitted on the earlier MD cohort and transferred unchanged. The two neural examples differ in epoch and alignment strength, so this comparison cannot isolate the augmentation effect. Colors identify cluster membership within each space, not equivalent physical states across independently fitted spaces.

Slurm: preparation 1014065; A40 inference 1014066 on node22/node25; descriptor projections 1014067; numerical publication 1014068. The publication destination was corrected by copy-only job 1014130; the original numerical outputs and metric definitions were preserved. Browser review 1014131 checks the copied gallery.
