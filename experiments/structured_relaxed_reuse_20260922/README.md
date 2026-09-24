# Archived relaxed inputs for symmetric-context crystallization forecasts

Test whether the development-selected relaxed MACE representation improves
crystallization forecasts relative to the original frozen MACE when both heads
use the same available origins, histories and future targets.

Reuse 3,062 archived full-cell quenches across all 150 independent Al sources.
The frozen availability inventory gives 2,762 source/origin combinations before
filtering originally at-risk tracked centers. Retain the original 90 train,
15 selection, 15 calibration and 30 test sources. The three latest **real**
observations within 72 ps define each history; attention receives their actual
time offsets. No missing frames are synthesized or interpolated. Availability
is frozen before fitting; later completed cells cannot change this run.

The four heads (direct, autoregressive MSE, mixture, diffusion) are trained from
scratch for each of two frozen input encoders, eight fits total, one seed. Keep
reference width128, two attention blocks, LR1e-4, batch128, warmup/cosine schedule,
36-epoch ceiling and validation-Brier early stopping. Both arms retain center plus
12 cuboctahedral queries at10 A and12 at20 A. Assign atom identities in observed
geometry; retain them through relaxation and supply actual offsets. The relaxed
encoder's nearest80 observed candidates are retained across quenching and cropped
to its trained normalized radius8. Descriptor histories use geometry from each
arm's input domain. Two historical differences are present; the unused third
reference difference block is zero in both arms.

**Targets:** shared original MACE latent states plus original MD physical,
bond-order and crystallinity trajectories every3 ps through96 ps; original MD
sustained-onset labels every0.75 ps. Both arms learn their initial state in that
shared target space. Relaxed future embeddings are **not** imputed. This avoids
new relaxation while providing comparable trajectory errors. Neither encoder is
fine-tuned. `cold-vic-temp01` was selected by cold-domain development Physical +
0.25 TDA, not test forecasting performance; ancestry excludes test/calibration.

**Precision:** archived full-cell coordinates are float16, decoded before local
geometry. A four-cell exact-quench comparison across liquid, transition and
post-onset cases gave median feature cosine0.999105 and relative RMS0.063568
with fixed observed query IDs, passing the declared0.99/0.10 engineering gate.
The initial cold-geometry query assignment exceeded that RMS gate (0.100839);
its receipt is retained. Fixed IDs remove query changes from the precision test
and align both experimental arms' atom choices. Passing is not proof of equal
prediction quality. The result remains an archived-precision experiment.

Eight unit tests and eight one-epoch, four-source GPU pipeline fits passed.
Those smoke metrics are execution checks, not experimental results. Real checks
confirmed identical targets/normalizers/population between arms and that future
states cannot alter observed inputs.

Report classification, calibration, timing including misses, physical/latent path
errors and sampled-center spatial scores using the existing metric exporters.
For irregular origins report false alarm episodes per1000 observed origins,
not a fictitious continuous-time exposure rate. Historical dense-history results
are context, not matched controls. Update this record after the eight full fits.

Config: `configs/crystallization_transfer/symmetric_relaxed_reuse_20260921.json`.
Output: `output/crystallization_transfer/symmetric-relaxed-reuse-20260921/`.
Reproduction and detached workers: [operations](../../docs/structured_relaxed_context.md).

Frozen at-risk population: 9,396 train / 3,625 selection / 2,721 calibration / 7,654 test windows. The test set contains226 positive12 ps windows (overlapping windows are not independent events). All90 training sources retain examples,18 at each temperature.

## Figure release

The14 PNG figures and separate captions are in
`output/crystallization_transfer/relaxed-reuse-analysis-20260922/index.html`.
Reproduce using `python -m src.research.structured_context.reuse_figures --config
configs/analysis/relaxed_reuse_figures.json` in a GPU allocation. For presentation
changes only, add `--render-only` (CPU). No simulations or model training occur.

The fixed-event offset analysis retains66 local onset/control pairs from18 test
sources across lead bins[12,24),[24,36),[36,48),[48,60) ps. Actual origins are used,
not interpolated. These balanced-cohort AP values differ in interpretation from
natural test-population AP. The real-space illustration uses source881, atoms155
(frame32),7176 and26461 (frame512); each panel pair uses identical80 atom IDs.
All coordinate panels and displacement arrows use actual Angstrom scale.
