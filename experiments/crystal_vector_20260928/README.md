# CDV-MACE128 with independent random batches

User-directed revision of the [27 September comparison](../crystal_vector_20260927/README.md):
remove fixed distance-bin quotas and draw each batch independently from the
declared source-weighted training population. Keep batch/microbatch 256.

The population remains half fixed-at-risk and half uniformly sampled centers,
with equal source mass inside each half. Sampling consumes only these row
probabilities, not distance labels. All per-example loss weights are one.
No minimum easy count, batch rejection, or inverse-probability correction.

| Target region | Probability | Expected count in 256 |
| --- | ---: | ---: |
| Inside confirmed crystal, d=0 | 0.19832919 | 50.77 |
| Liquid center, 0<d<=8 A | 0.12304892 | 31.50 |
| Liquid center, 8<d<=20 A | 0.08978334 | 22.98 |
| d>20 A, including censored | 0.58883856 | 150.74 |

There are about 54.49 liquid centers within 20 A per batch in expectation.
For the nearest 0–8 A group, probability of an empty independent batch is
(1-0.12304892)^256, about 2.52e-15. The law of large numbers motivates the
approach; the finite-batch probability follows the binomial distribution.

All three treatments restart from identical original CD-MACE128 spatial
weights with fresh optimizers and the same seed. Former partial checkpoints
remain under their historical identities; their continuation jobs are canceled.
Do not mix the two sampling regimes inside a run or compare their incomplete
states as completed sixteen-epoch models. The encoder, objectives, 16 nominal
epochs, epoch-12 minimum selection and unsampled evaluation are unchanged.

The existing sealed geometry release is reused: 63,251 training contexts,
29,667 selection contexts, and unchanged Al64 calibration/test/scan rows.
No new data and no additional simulations. The geometry release's producer is
the captured 27 September code; changes to the training sampler do not relabel
its provenance or trigger geometry regeneration.

Interior target is exactly zero distance to the established crystal set. There
is no signed distance or interior depth label. Direction is undefined and
excluded from directional likelihood for these samples; distance supervision
uses the mixture's zero-probability mass.

[Recipe](../../configs/crystal_vector/al64_20260928.json) ·
[Metrics](../../docs/metrics/crystal_vector.md) · [Execution](../../docs/crystal_vector.md).
