# Does order-preserving regularization improve local structure representations?

The large neighborhood-JEPA encoders remain low rank and have not clearly beaten
fixed descriptors on frozen crystallization prediction. This study tests whether
regularizer/projector choice and explicit preservation of orientational order help.
It uses only MACE and existing native Al Lee-MEAM observations; one training seed.

All models use the same width64 encoder, 128D invariant export and typed 120D
angular export, instantaneous TDA/physical anchors and current/future neighbor
prediction. Add supervised q4, q6, w4, w6, averaged q6, q6 coherence, neighbor
density and smooth coordination, decoded from the exported invariant features.
Targets use only the observed crop. No PTM crystallinity label trains the encoder.

| Factor | Treatments |
|---|---|
| Regularizer | none, sample-normalized SIGReg, VICReg variance/covariance, EpiJEPA-inspired geometric reservoir |
| Projector | MLP with/without LayerNorm, linear, identity |
| Export | per-observation LayerNorm, raw export for declared direct-regularization arms |
| Order anchor | weight .25; matched SIGReg control with weight zero |
| Initialization | common development-selected width64 parent; two explicit scratch controls |

The exact 16 combinations are in `regularization/specs.py`. All use B512,
768 updates (12 sampled epoch equivalents), encoder/head peak LR .0001/.001,
10% warmup, cosine decay to 1%, AdamW decay 1e-4, gradient clips 1/5.
Only current independent center observations enter regularization. VICReg supplies
variance/covariance while conditional JEPA supplies alignment; it does not force
moving local states to remain identical. Epi uses fixed random geometric features,
FP64 ridge/logdet and a scale-controlled learned state; see the metric document.

Warm initialization copies the encoder and conditional prediction/physical/TDA
heads from large node53 step2560, chosen by its development physical/TDA score.
All arms reset projector and new order head, and reset optimizer. Scratch controls
quantify whether adapting a low-rank pretrained model impedes learning.

The active encoder already uses per-observation LayerNorm. New tests verify
train-mode batch independence, rotation invariance and exact gradient replay;
there is no BatchNorm correction. Raw-export arms remove only the outermost
LayerNorm. Historical runs remain references; the new no-order control shares
the common initialization, budget and refreshed projector with the order arm.

Three longer runs add 1,536 updates to the best warm order-anchored model in each
regularizer family. Promotion uses development physical+.25TDA+.25order+.25future
physical; no crystallization test metrics. The within-run checkpoint rule remains
physical+.25TDA for every arm. Export rank is a diagnostic, never the selection rule.

Every selected checkpoint gets frozen linear and MLP onset-hazard readouts, with
the same historical 150-source assay as earlier runs. Include condition-only,
93-feature geometry/order-only, and historical136-feature geometry/motion/order
baselines. The historical baseline has velocity information the snapshot encoder
cannot observe, so use the positions-only comparator for the input-matched question.
Its order neighborhoods can still extend beyond the encoder crop. Report event
NLL, AP/AUROC, calibration, timing MAE together with misses, and sampled-center
spatial scores. One seed and a reused historical test cohort limit claims.

Reproduction and detached queue operation: [operations](../../docs/neighborhood_jepa_regularization.md).
Definitions: [training metrics](../../docs/metrics/neighborhood_jepa_regularization.md),
[frozen crystallization](../../docs/metrics/neighborhood_crystallization_v2.md).
Results update under `output/neighborhood_jepa/regularization-20260920/`.

[Direct Epi-inspired static Al analysis](STATIC_AL.md) applies the selected invariant embedding to the matched six-snapshot clustering protocol.
