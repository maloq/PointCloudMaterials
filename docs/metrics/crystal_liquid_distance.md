# Liquid-only localization of an external established crystal

LCD-MACE128-VC is a supervised, jointly trained geometry encoder and vector-message
spatial predictor. Reuse the frozen 150-source Al64 release and dense 256-centre,
64-frame expansion (`0720998c...`). No new simulations, source splits, species,
material identities, temperature, time, velocity or MD history inputs. Encoder:
shared MACE128 for all 25 patches, nearest 80 atoms, radius 8 Å, cutoff 5 Å, two
interactions; scalar 128 and equivariant vector 16-channel exports. Predictor:
two vector-message blocks over actual relative patch positions, shared prediction
heads and a learned mixture. Maximum observation reach is 32 Å.

The observation condition is `~inside_crystal & ~crystal_visible_context`, using
**established past-confirmed crystal** masks on actual consumed radius-8 patches.
Do not exclude atoms solely because they show instantaneous/subcritical local order:
partially ordered liquid is the intended source of signal. The query must be outside
crystal and none of its input patches may contain established crystal.

Training, validation, calibration and primary test localization also require an
established crystal elsewhere: finite `crystal_distance`. These are label-side
population conditions, never predictor inputs. This is conditional localization;
the model is not trained here to infer whether a crystal exists somewhere in the
simulation. Cases with no crystal remain a separately labelled absence challenge.

Reuse sealed interface-derived geometry without relabelling the cache. Before any
fitting, assert on **every eligible row** that an interface exists, no interface
atom is visible, and cached `distance` equals `crystal_distance` exactly. Fail if
these conditions do not hold. The inherited nearest-interface vector therefore
points to a nearest crystal atom on this domain. Retain the inherited interface
tie/zero/censor direction mask; do not claim a newly recomputed crystal-set tie
definition. Empty-cell and crystal-interior samples never enter fitting or selection.

The original pre-exclusion reference population is half fixed at-risk and half
uniform, with equal source mass within each half. Uniform denominators include all
proposed extra candidates before the earlier interface-visibility rejection.
Condition these probabilities on the complete liquid/external-crystal predicate
and renormalize once. Draw independently with replacement, with unit loss weights
and no distance quotas. Record eligibility counts, source counts, distances ≤20/32 Å,
censored distances, exclusions and expected batch coverage. Every actual training
batch is checked against the eligibility predicate.

Same snapshot parent and fresh predictor/optimizer as the superseded interface-
only exclusion recipe. The parent historically saw crystal-containing training
inputs; this is restricted adaptation, not visibility-naive initialization. Use
one seed, global batch/microbatch 512 (256 per GPU), cuEquivariance, BF16, compiled
patch operations and differentiable globally synchronized VCReg moments. Training
has 16 nominal blocks ×512 updates =8192 updates/4,194,304 replacement draws; these
are not exhaustive dataset epochs. Select blocks 12–16 by full conditional
validation predictive likelihood, excluding VCReg.

Distance/direction/proximity likelihoods and VCReg are unchanged from
[crystal_vector](crystal_vector.md). The target is distance to an existing external
crystal. Distances ≥64 Å are right-censored for likelihood; use capped means/medians
only for point errors. No AP loss, ranking loss or AP selector.

Primary `distance/direction/reliability.csv` contain eligible liquid contexts only,
with the original per-population macro-source definitions. `absence-*` tables
contain crystal-free observations with no established crystal anywhere. Never mix
them into the localization score or selector; probabilities there are an absence
challenge, not a calibrated presence classifier. All cached original rows retain
predictions and row identities; crystal-containing rows are not part of the primary
numerical tables. Saved historical-interface likelihood fields outside the declared
domain are not interpreted as nearest-crystal metrics.

`paired-original-liquid-*` uses exact source-local parent indices to compare the
previous frozen interface-VCReg checkpoint on every original eligible row. Verify
prediction/checkpoint hashes and matching identity, role, frame, atom, distance,
crystal distance and phase/visibility fields. Expanded uniform rows are additional
evidence, not a replacement of the original fixed benchmark.

`liquid-baselines.csv` compares trained capped-mean RMSE and proximity Brier with
a constant fitted on the **conditional training population only**: its weighted
mean capped distance and weighted empirical CDF at 4/8/12/20/32 Å. Evaluate the same
rows/weights for both predictors, separately on original and expanded-plus-original
tracks. The constant has no NLL entry and does not claim a continuous density.

Scan alarms require two consecutive original queries before the first established
crystal becomes visible in any patch or the query enters crystal. Keep all original
paths in denominators; never join disconnected invisible segments. The historical
interface-alarm calculation is reused with crystal visibility substituted and
explicitly renamed output fields. Rank/readout/noise/stability diagnostics use
eligible liquid anchors/training rows. Earlier or noise-perturbed observations can
change visibility; report these as response diagnostics, not filtered trajectories.
The physical-information readout never trains the encoder. Noise RMS is relative to
the recorded nearest-neighbour scale and MD diagnostic lag is 0.75 ps.

Record encoder and predictor contexts separately, cohort identity, prior exposure,
exact population and per-source counts, parent/selected checkpoint hashes, sampling
and synchronized two-GPU settings. One online W&B scientific run; diagnostics and
associated evaluation update that run, with no additional online probe runs.
