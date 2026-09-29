# Al and Ta specialization: results at the 27 September stop

Material-specific fine-tuning offers little benefit in this single-seed comparison.
Al completed twelve epochs and all evaluations. Ta was stopped at the user's
request around epoch 11.17; its latest durable checkpoint is epoch 11.1304,
update 33,280. That checkpoint was evaluated without further optimization in an
explicit interim bundle. It is not the predeclared twelve-epoch selected model.

Each parent/child comparison uses identical evaluation rows within its material.
All distances below are Al-equivalent Angstrom (ordinary Angstrom for Al), capped
at 64 for point errors. Lower values are better.

| Evaluation | Model | NLL | Mean-prediction RMSE | Median-prediction MAE | Brier, within 20 A |
| --- | --- | ---: | ---: | ---: | ---: |
| Al fixed test | Shared parent | 2.477755 | 19.331762 | 14.533287 | 0.128645 |
| Al fixed test | Al fine-tune, epoch 12 | 2.473545 | 19.253690 | 14.324350 | 0.127736 |
| Ta shooting test | Shared parent | 3.650930 | 23.054929 | 20.228704 | 0.149511 |
| Ta shooting test | Ta fine-tune, interim epoch 11.1304 | 3.657438 | 23.207326 | 20.214431 | 0.150687 |

Al improves the point estimates by 0.17% in NLL, 0.40% in RMSE and 1.44% in MAE.
Ta's NLL and RMSE worsen slightly while its MAE barely changes. No source- or
seed-uncertainty interval is claimed. These results do not establish a useful
general benefit from material specialization over the shared parent.

## Front warning on Al

For P(current distance <=20 A) >0.5 at two consecutive spatial positions,
recall while still at least 12 A away changes from 2.83% to 2.42%; far-control
false alarms decrease from 4.79% to 3.77%. These are 495 approach paths and 292
controls. The conditional median warning distance remains zero at thresholds
0.5, 0.75 and 0.95. Every one of the fine-tuned model's 158 detections at 0.5 has
PTM-crystalline atoms somewhere in its actual input observations. Specialization
does not establish advance detection beyond visible crystalline structure.

Controlled-scan distance NLL worsens from 4.6048 to 4.6525 and RMSE from 21.3231
to 21.9617 A. The small fixed-test improvement is not uniform across populations.
There is no corresponding Ta spatial-scan assay in this experiment.

## Can we rank the metals by predictability?

The current model has lower raw error on the current Al benchmark. An intrinsic
metal-difficulty ranking is not supported by these experiments:

- Al has 45,291 liquid-at-risk test observations over 30 sources. Ta has 184,320
  uniformly sampled observations from three new velocity branches of ONE known
  preparation. Its parent configuration was already represented in training.
- The Al test contains no crystalline centers; about 19.1% of Ta test centers
  have zero distance. Distances are censored at 64 A for 44.7% of Al versus 7.9%
  of Ta. These different target distributions affect both NLL and point errors.
- Native Al uses 0.75-ps observations, Ta 0.70 ps. Al fine-tuning uses 4.56 million
  windows, Ta 3.06 million, with different effective numbers of independent
  configurations. The stopped checkpoints have different training durations.
- This task estimates current distance to an existing confirmed crystal. It does
  not measure future nucleation predictability or compare crystal-front warning
  skill across metals.

A stronger comparison would match liquid-only observation rules, distance bands,
physical history, training exposure and independent-preparation splits, then
measure gains over the same declared simple baselines. Keep the shared parent
as the reference; these runs provide no compelling reason to replace it with a
material-specific child.

## Evidence

- [Al paired distance table](/work/PERSO/vmorozov/analysis/distance_encoder/material-al-20260927-v2/analyses/material-comparison-v1/tables/distance.csv)
- [Al paired alarms](/work/PERSO/vmorozov/analysis/distance_encoder/material-al-20260927-v2/analyses/material-comparison-v1/tables/alarms.csv)
- [Ta interim paired distance table](/work/PERSO/vmorozov/analysis/distance_encoder/material-ta-20260927-v2/distance-interim-stop-20260927/analyses/material-v1/tables/distance.csv)
- [Ta checkpoint and stop receipt](/work/PERSO/vmorozov/analysis/distance_encoder/material-ta-20260927-v2/technical/stopped-by-user.json)
- [Interim metric definitions](../../docs/metrics/distance_encoder_material_interim.md)

Ta evaluated checkpoint SHA256:
`5055db09684d19a4d5c01ec0a4e0610bd255e2d9692ab912d3d79a9a5bb240dc`.
Parent and adapted Ta predictions retain exact source/atom/frame identities and
checksums. Interim metrics update the original training W&B run separately from
final metrics; no evaluation run or additional training was created.
