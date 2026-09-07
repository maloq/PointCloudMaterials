# Liquid short-range-order benchmark: completed comparison

The new benchmark measures future topology, continuous bond order and atomic
rearrangements within initially noncoherent environments. PTM/HCP classes are
not training targets or ground truth. Reference MACE, SchNet-style and a
smooth-density MLP were each trained on Al/Mg/Ta for 60 epochs with three seeds,
using same-center jitter views and a separate VICReg projector.

The primary dynamical test uses 3,054 Al centers from six held-out source runs,
eight shooting futures per parent, and 12/24/48 ps horizons. Entire source runs
are separated between training, validation and testing. Each representation
must add predictive information beyond current bond order, density and
temperature. Positive scores below mean percentage reduction in forecast MSE
over that baseline. They are not classification accuracies.

| Representation | Future topology | Future bond order | Future mobility |
|---|---:|---:|---:|
| GeoFrame v2, earlier checkpoint | +0.44% | -0.04% | +0.23% |
| Reference MACE, untrained | +0.76% | +0.83% | +4.64% |
| Reference MACE + VICReg | +0.81% | +1.05% | +4.68% |
| SchNet-style + VICReg | +0.59% | +1.10% | +1.24% |
| Smooth-density MLP + VICReg | **+7.16%** | **+6.67%** | +10.37% |
| Smooth-density PCA | +6.13% | +6.26% | **+11.95%** |
| SOAP + PCA | +4.79% | +5.58% | +9.93% |
| TDA, 16 PCs, exploratory | +3.81% | +3.97% | -6.22% |
| TDA, 128 PCs | -74.44% | -46.00% | -21.05% |
| Shuffled SOAP control | -1.68% | -2.25% | -2.48% |

The density MLP's topology gain has a 95% source-bootstrap interval of
[+2.82%, +10.54%]. It also improves future agreement among neighbors matched on
temperature and coarse bond order by 3.50% [2.40%, 4.46%]. This supports useful
information beyond coarse order, without establishing a new metastable phase.

The accelerated MACE implementation initially failed the final rotation
audit: fused convolution requires `ir_mul`, while the first configuration used
`mul_ir`. The affected models were archived as invalid and all three MACE seeds
were retrained from scratch. Corrected trained models pass the GPU rotation
controls. Five scientific tests pass, including native and fused-GPU rotation,
cutoff, radial-gradient and complete-receptive-field controls.

After this correction, MACE's forecast scores remain close to the untrained
reference. The paired source intervals for the trained-versus-untrained gains
include zero for all three forecast targets. Covariance-conditioned probes do
not remove the gap. This points to limited benefit from this jitter-only
VICReg pretraining objective, rather than proving that MACE is unsuitable for
atomic structure. No energy, force or temporal-prediction loss was trained here.

TDA is useful as a complementary geometric assay. Its 128-PC probe suffers
extreme prediction errors; the exploratory 16-PC variant is more stable but
does not outperform the density/SOAP baselines. A 0.02 Å perturbation changes
the 64-neighbor TDA window in 27% of the sampled environments. TDA therefore
should not become a new unquestioned label source.

The primary future benchmark is Al. Mg/Ta have auxiliary structural checks;
there is no independent-source Mg/Ta shooting benchmark here. In particular,
the current supplemental Al/Mg test snapshots contain too few noncoherent
centers to establish liquid-only generalization, and Ta has only one trajectory.
The rare persistent-order assay has approximately 1% prevalence at 48 ps, so it
remains secondary. Eight futures also leave substantial uncertainty in topology
propensities.

The strongest current candidate is the smooth-density encoder. The next
architecture/objective comparison should add predictive training on the
training trajectories, retain these source-held-out assays, and obtain
independent liquid Mg/Ta shooting ensembles. The current results are a
representation benchmark, not discovery or validation of a new phase.

All generated artifacts are physically inside this repository:

- [Full report, methods, uncertainty and plots](../output/liquid_sro_benchmark_20260905/RESULTS.md)
- [Comparison CSV](../output/liquid_sro_benchmark_20260905/evaluation/comparison.csv)
- [Pairwise source-bootstrap comparisons](../output/liquid_sro_benchmark_20260905/evaluation/pairwise_advantages.csv)
- [Selected checkpoints](../output/liquid_sro_benchmark_20260905/checkpoint_selection.json)
- [Code and reproduction commands](../experiments/liquid_sro_benchmark_20260905/README.md)
