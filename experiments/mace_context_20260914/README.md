# Complete context and tracked-center MACE readouts

Question: can complete surrounding message context plus a smooth inner readout,
or the tracked center node, remove nearest-neighbor membership jumps without
losing topology information and sensitivity to physical evolution?

The matched pilot starts from the exact epoch-02 encoder used to produce forecast
embeddings. Compare four frozen representations: original 80-node mean,
complete-context hard-80 mean (isolates graph truncation), complete-context smooth
inner pooling, and complete-context center feature. Continue training the original
mean and the two requested alternatives for eight matched VICReg epochs. This is
a warm-start screening experiment with one initialization, not a full reproduction
of the original 24-epoch training campaign.

Pooling has full weight to 5 Angstrom, then a quintic taper to zero at 7 Angstrom.
The median training patch's 80th radius is 6.994 Angstrom (5th–95th percentiles
6.816–7.126). Two 5 Angstrom layers require a conservative context to 17 Angstrom;
18 Angstrom candidates provide a checked augmentation margin. Graph pruning
evaluates the exact two-hop ancestors, verified against the unpruned computation.
The center's feature retains both scalar blocks, giving the same 256 dimensions.

Use the original 64 diagnostic centers per context, 18/6/6 whole-source splits,
hot and relaxed TDA labels, and 144 tracked temporal series. Frozen and trained
readouts use validation-selected float64 SVD ridge penalties. Controlled physical
80th/81st radial crossings distinguish a jump from smooth geometric sensitivity.
TDA still uses the old hard-80 support; smoothing the embedding cannot be judged
only by reproducing discontinuities in those labels. See the
[metric definitions](../../docs/metrics/mace_context.md).

Recipe: [mace_context.json](../../configs/analysis/mace_context.json).
With conda `pointnet`, invoke `python -m src.research.mace_context.run --config
configs/analysis/mace_context.json --stage prepare`, then `--stage verify`,
`--stage frozen --mode MODE`, `--stage train --mode MODE`, and `--stage summarize`.
Use a new configured output/cache for a fresh experiment; completed inputs and
training attempts are preserved. Operational launch records live under the run's
`technical/` directory. No downstream forecaster is retrained by this protocol.

Results: [pilot output](../../output/mace_context/forecast-seed20260910-pilot-20260914/README.md).
The [completed findings](RESULTS.md) include all three matched training runs.
The [frozen findings](FROZEN_RESULTS.md) retain the comparison before continuation.
The [smoothness follow-up](SMOOTHNESS.md) separates temporal variation from
membership discontinuity. Reproduce it with the same config and `--stage smoothness`;
this CPU analysis writes a separate output suffixed `-smoothness`.
See [checkpoint/API usage](../../docs/mace_context_encoder.md) for operational details.
