# Ta potential choice for crystallization and position shooting

Literature checked 2026-09-26. Decision: retain the Zhong/Sheng 2014 EAM as the
primary shooting baseline. It is well motivated for this question, but the
review does **not** establish a best potential or quantitative nucleation accuracy.
An independent force-field comparison remains necessary before interpreting an
encoder's success as transferable physics of real Ta.

## Evidence and alternatives

| Potential | Relevant evidence | Limitation for our question |
| --- | --- | --- |
| Zhong/Sheng EAM, 2014 | The author's distribution explicitly links this Ta model to the monatomic-glass paper. The page supplies liquid-density, pair-correlation and structure-factor comparisons and a BCC/liquid coexistence example at 3285 K. [Distribution and validation](https://sites.google.com/site/eampotentials/ta), [original paper](https://www.nature.com/articles/nature13617). | These checks do not establish accurate nucleation barriers or branching probabilities. |
| Lin–Purja Pun–Mishin PINN, published 2022 | Combines a bond-order model with neural corrections. DFT comparisons include liquid radial and bond-angle distributions, competing solid structures and defects. Reported melting point: 3000 ± 6 K; experimental reference: 3293 K. Liquid training states include 2600, 2900, 3500 and 5000 K. [Paper/preprint](https://arxiv.org/html/2101.06540), [published DOI](https://doi.org/10.1016/j.commatsci.2021.111180). | Promising independent check, but the reviewed validation does not establish 1900 K crystallization kinetics. Its reported liquid surface tension is a liquid–vapor quantity, not solid–liquid interfacial free energy. |
| Thompson et al. SNAP, 2015 | DFT liquid pair correlations and solid structures were assessed. Reported melting point: 2790 K. [Paper](https://arxiv.org/html/1409.3880), [published DOI](https://doi.org/10.1016/j.jcp.2014.12.018). | Developed with substantial emphasis on defect/plasticity behavior; no demonstrated superiority for deeply supercooled Ta nucleation. |
| Ravelo et al. EAM, 2013 | A widely used alternative developed for shock-induced plasticity. [Paper](https://doi.org/10.1103/PhysRevB.88.134101). | Its target application alone gives no reason to replace the glass-focused baseline for ambient-pressure liquid ordering. |

The most directly relevant follow-up is Hu et al. (2025),
[Monatomic glass formation through competing order balance](https://www.nature.com/articles/s41467-025-63221-8).
Its Ta EAM citation is Zhong et al. (reference 6). It studies competing BCC,
icosahedral and quasi-crystalline ordering in Ta/Zr; its Ta BCC melting point is
3255 ± 5 K. The paper's comparison with PINN includes diatomic energy curves;
that is not a replication of its nucleation results under PINN. We verified
the literature family, not byte identity between our file and the 2025 authors'
simulation files. The difference from the author's 3285 K coexistence example
should be resolved by our own coexistence calculation rather than silently
choosing one value as an exact property of our bytes.

The [NIST Ta repository](https://www.ctcms.nist.gov/potentials/system/Ta/) is a
useful implementation index, not a ranking of fitness for this application.
PINN has an [available implementation](https://github.com/ymishin-gmu/LAMMPS-USER-PINN)
requiring its corresponding LAMMPS pair style; installation, throughput and
stability in our workflow have not been checked.

## Consequences for the comparison

Our calculation from the published melting temperatures gives, at 1900 K,
`T/Tm = 0.584` for the 2025 EAM estimate, `0.633` for PINN and `0.681` for SNAP.
These are different thermodynamic conditions. A later comparison should report
both common absolute temperatures and matched reduced undercooling, plus measured
diffusion/relaxation. Matching `T/Tm` alone does not match driving forces or kinetics.

Our proposed validation sequence, not a completed literature result:

1. **Phase competition:** BCC, A15 and sigma/beta-Ta energy–volume curves,
   finite-temperature stability, and representative liquid/interface energies and
   forces against independent DFT. Do not select a force field because it produces
   the clusters we hoped to see.
2. **Liquid order:** density, pair and angular correlations, joint bond-order and
   Voronoi descriptors, connected icosahedral domains, and motif lifetimes in
   independently equilibrated cells. Repeat descriptor thresholds; avoid reducing
   Ta to one liquid-versus-BCC label.
3. **Dynamics:** diffusion, structural relaxation and planar crystal growth,
   followed by independently prepared nucleation realizations. Evaluate finite-size
   sensitivity before scaling to the million-atom shooting workload.
4. **Representation robustness:** within each potential, use the same declared
   frozen likelihood probes and information controls. Then evaluate transfer to
   the other potential with parent-lineage separation. Report proper predictive
   scores, calibration and uncertainty; AP remains diagnostic. No temperature,
   time or potential-ID input is added to predictors by this comparison.

Force-field sensitivity is a separate uncertainty from velocity-shooting variation.
Repeated velocities under one Hamiltonian cannot validate that Hamiltonian.
Good outcomes under two models would provide stronger evidence, not proof against
all model error. Small-cell DFT checks and the larger dynamical comparison answer
different questions and both matter.

## Exact local baseline and scope

The registered `potential-ta-zhong2014` file is `Ta_Zhong2014.lammps.eam`,
SHA256 `8908993117f2502ed48bd31b737556719c3f7f11d1e7b7213eb257cd1ca42386`.
Its header names H. W. Sheng and dates generation to 2014-02-08. This is the
recorded potential of the previous six Ta dynamical branches.

The potential that generated the archived starting positions is unknown. Fresh
shots therefore mean positions conditioned on those archived configurations,
with newly sampled velocities and a new NPT state. They are not exact restart
continuations. Four outcomes per parent are a pilot, not a precise committor
estimate; no absorbing basins are defined. Preserve this distinction in labels
and analysis. See the [shooting record](ta_shooting_20260926.md).
