# Overnight context and representation study

One seed; historical source-held-out cohort. Controls and treatments use identical capacity, warm parents and budgets. Selections use development data only. The new-encoder bridge adds a current snapshot export to the same fixed original-encoder trajectory context and targets.

| Task | Status |
|---|---|
| encoder: continued-control | complete |
| encoder: strong-order | complete |
| encoder: linear-information | complete |
| encoder: linear-no-reg | complete |
| path: direct-control-E18 | complete |
| path: direct-shells-E18 | complete |
| path: direct-history-E18 | complete |
| path: direct-both-E18 | complete |
| path: direct-descriptors-only-E18 | complete |
| path: ar_mse-control-E18 | complete |
| path: ar_mse-shells-E18 | complete |
| path: ar_mse-history-E18 | complete |
| path: ar_mse-both-E18 | complete |
| path: ar_mse-descriptors-only-E18 | complete |
| path: mixture-control-E18 | complete |
| path: mixture-both-E18 | complete |
| path: diffusion-control-E18 | complete |
| path: diffusion-both-E18 | complete |
| encoder_long: selected-long | complete |
| probe: selected-long | complete |
| probe: continued-control | complete |
| probe: strong-order | complete |
| probe: linear-information | complete |
| probe: linear-no-reg | complete |
| path_long: direct-selected-E36 | complete |
| path_long: ar_mse-selected-E36 | complete |
| path_long: mixture-selected-E36 | complete |
| path_long: diffusion-selected-E36 | complete |
| bridge: direct-new-encoder-E18 | complete |
| bridge: ar_mse-new-encoder-E18 | complete |

## Trajectory prediction

| Run | 96ps integrated Brier ↓ | 12ps AP ↑ | 12ps timing MAE, detected only ↓ | Missed/event windows | Physical path MSE ↓ |
|---|---:|---:|---:|---|---:|
| ar_mse-both-E18 | 0.10964 | 0.565776689181739 | 2.3857295585971556 | 386/1349 | 0.85912 |
| ar_mse-control-E18 | 0.11094 | 0.5637212218801371 | 2.4652415214306536 | 377/1349 | 0.87960 |
| ar_mse-descriptors-only-E18 | 0.11987 | 0.5410191205130701 | 2.500446511669515 | 432/1349 | 0.86967 |
| ar_mse-history-E18 | 0.11077 | 0.5720052465913028 | 2.4103871167029722 | 357/1349 | 0.87728 |
| ar_mse-new-encoder-E18 | 0.11057 | 0.5701503852082089 | 2.422081414620833 | 353/1349 | 0.87788 |
| ar_mse-selected-E36 | 0.11006 | 0.5624651543608047 | 2.379194020062966 | 385/1349 | 0.85944 |
| ar_mse-shells-E18 | 0.10823 | 0.5702815509147209 | 2.4250202546967854 | 390/1349 | 0.86032 |
| diffusion-both-E18 | 0.11605 | 0.5353041674242063 | 2.589486738188572 | 411/1349 | 0.87087 |
| diffusion-control-E18 | 0.11681 | 0.5132716968224272 | 2.5701700898181916 | 440/1349 | 0.87206 |
| diffusion-selected-E36 | 0.11475 | 0.5320485985820692 | 2.535268965542735 | 424/1349 | 0.86718 |
| direct-both-E18 | 0.11044 | 0.5916275640289239 | 2.421188933789196 | 385/1349 | 0.84771 |
| direct-control-E18 | 0.11073 | 0.5634049308767695 | 2.4753886120811814 | 401/1349 | 0.84895 |
| direct-descriptors-only-E18 | 0.11993 | 0.5250008798039305 | 2.496998724039294 | 504/1349 | 0.86057 |
| direct-history-E18 | 0.11136 | 0.5761409512754757 | 2.4477528650969522 | 404/1349 | 0.84834 |
| direct-new-encoder-E18 | 0.11083 | 0.5871227369046274 | 2.433685164180904 | 382/1349 | 0.84877 |
| direct-selected-E36 | 0.10831 | 0.5857628569706659 | 2.4192657620592297 | 408/1349 | 0.84754 |
| direct-shells-E18 | 0.10810 | 0.5861030897691333 | 2.4267126295456007 | 407/1349 | 0.84730 |
| mixture-both-E18 | 0.11248 | 0.5736364189461615 | 2.44083012353849 | 430/1349 | 0.88114 |
| mixture-control-E18 | 0.11298 | 0.5582486895048735 | 2.450097307822473 | 416/1349 | 0.88448 |
| mixture-selected-E36 | 0.11271 | 0.5770531239901334 | 2.4745717925018296 | 399/1349 | 0.88052 |

All trajectory forecasts are evaluated without future target feedback. Timing MAE excludes misses; inspect both. Future physical trajectories and original-encoder latent targets are fixed across treatments. Reports distinguish a better local snapshot encoder from gains due to additional observed spatial/history context.

Detailed encoder probes: encoders/CRYSTALLIZATION.md and short_readouts/technical/fits/. Scientific protocol: experiments/context_night_20260921/README.md. Update after all queued tasks finish.
