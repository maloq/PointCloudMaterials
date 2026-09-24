# Symmetric context with relaxed MACE

Does a validation-selected relaxed-input MACE encoder improve open-loop local
crystallization forecasts when its spatial observations are also relaxed?

Use `cold-vic-temp01`, selected by minimum cold-domain development Physical +
0.25 TDA (0.2984237894); no test-score checkpoint selection. Keep the original
150-source 90/15/15/30 split, 16 tracked centers, 3 ps origin grid, four observed
times (-48,-12,-3,0 ps), and 96 ps forecast horizon. Four frozen-encoder heads:
direct, autoregressive, mixture, diffusion. Preserve the reference head settings,
36-epoch ceiling, early stopping, physical targets and original-MD onset labels.
No GATr is included.

Each observation undergoes generating-potential Lee2003 MEAM full-periodic,
fixed-box FIRE relaxation to <=0.01 eV/Angstrom. Choose 25 symmetric query slots
(center, 12 at 10 A, 12 at 20 A) in relaxed geometry with unique actual atoms and
<=4 A query error. For each assigned atom, retain its 80 nearest **observed** atom
identities across relaxation and apply the trained normalized radius-8 support.
Save centered float32 observations before reducing full-cell precision. Context
geometry and the velocity-independent descriptor auxiliaries are evaluated on
relaxed structures. Future latent targets use the same relaxed encoder; physical,
bond-order and crystallinity forecast labels remain original MD. Compare physical
and event metrics, not raw latent errors against the original encoder.

Full preparation requires 150 x 199 = 29,850 full-cell quenches; benchmark range
17–79 s/cell implies roughly 140–650 GPU-hours before extraction/IO/failed attempts.
The full scientific queue is frozen, but its large-scale submission is awaiting
the user's pilot-versus-full budget choice. Execution preflight is not a pilot
scientific result.

Config: `configs/crystallization_transfer/symmetric_relaxed_mace_20260921.json`.
Output: `output/crystallization_transfer/symmetric-relaxed-mace-20260921/`.
See `docs/structured_relaxed_context.md` for operations.
