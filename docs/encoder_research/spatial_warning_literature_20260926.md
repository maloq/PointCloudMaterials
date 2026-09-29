# Prior work on spatial crystal proximity and nucleation precursors

Literature search: 26 September 2026. Related local evidence:
[fixed-snapshot protocol](../../experiments/spatial_approach_20260926/README.md)
and [completed analysis](../../experiments/spatial_approach_20260926/RESULTS.md).

## Answer and scope

There is substantial prior work on learned atomic order, crystal–liquid
interfaces and ordering in the surrounding liquid. The closest methodological
precedent is SODAS, including a graph-neural-network analysis of **aluminum**
interfaces. A separate Al study plots an SVM structural score across interfaces.
The broad proposition that atomic structure can reveal proximity to a crystal
should therefore not be presented as new. The sources below distinguish those
results from nucleation prediction.

I did not find a publication combining every feature of our particular
evaluation: a probe traversing fixed atomistic snapshots, limited local or
multi-patch observations, a first-alarm distance, separately calibrated path
false alarms and an audit of whether crystal atoms were already in the input.
This is a bounded search result, not proof that no such paper exists. A distinct
evaluation protocol alone does not establish a strong scientific novelty claim.

| Primary study | System | Relation to our task |
| --- | --- | --- |
| [Chapman et al., 2023](https://www.nature.com/articles/s41467-023-39755-0) | Al | Closest GNN/interface characterization precedent |
| [Men, 2024](https://www.nature.com/articles/s41467-024-50182-7) | Al | Structural ML score plotted across crystal–liquid interfaces |
| [Freitas and Reed, 2020](https://www.nature.com/articles/s41467-020-16892-4) | Si, Cu | Liquid environment, interface ordering and growth kinetics |
| [Hu and Tanaka, 2022](https://www.nature.com/articles/s41467-022-32241-z) | Primarily NiAl | Intervention on preordering changes crystallization kinetics |
| [Xie et al., 2019](https://www.nature.com/articles/s41467-019-10663-6) | Si–Au interface | Unsupervised dynamical representations distinguish interfacial states |
| [Díaz Leines and Rogal, 2018](https://doi.org/10.1021/acs.jpcb.8b08718) | Ni | Prestructured surroundings improve a nucleation reaction coordinate |
| [de Jager et al., 2023](https://doi.org/10.1063/5.0161356) | Hard/charged colloids | Explicit precursor search with a negative result |

## Closest studies

### Chapman et al. (2023): SODAS

**Quantifying disorder one atom at a time using an interpretable graph neural
network paradigm**, Nature Communications 14, 4030.

A GNN produces a continuous per-atom ordering score in Al. Figure 4 and
Supplementary Figure S5 examine the transition across solid–liquid interfaces
and compare conventional structural classifiers. Calibration uses thermal
ensembles; the scalar is a disorder measure, not an event probability. The
reported 3.5 Å graph edge cutoff is not automatically its entire receptive field.
This is a strong precedent for our continuous spatial readout, although it does
not report our first-alarm experiment.
[Paper](https://www.nature.com/articles/s41467-023-39755-0);
[author PDF](https://sites.bu.edu/mil/files/2023/07/s41467-023-39755-0.pdf).

The authors provide [Graphite code](https://github.com/LLNL/graphite) and a
[training/inference notebook](https://github.com/LLNL/graphite/blob/main/notebooks/sodas/training-and-inference.ipynb).
Those links are reported by the paper; this review did not execute or audit that
implementation.

### Men (2024): Al interface profiles and attachment kinetics

**A joint diffusion/collision model for crystal growth in pure liquid metals**,
Nature Communications 15, 5749.

The study uses SVM/neural classifiers with 23 radial features to distinguish
liquid and solid environments in Al. Figure 3 plots the signed SVM score against
position through an Al(111) interface. Other analyses resolve orientation and
attachment mechanisms for (111), (110) and (100). Labels come from bond-order
analysis. The principal prediction concerns growth kinetics; the plotted
structural margin is not a calibrated warning probability. This is a particularly
close material-specific precedent and motivates a simple radial-feature control.
[Paper](https://www.nature.com/articles/s41467-024-50182-7).

### Freitas and Reed (2020): structure near a growing crystal

**Uncovering the effects of interface-induced ordering of liquid on crystal
growth using machine learning**, Nature Communications 11, 3260.

An SVM uses 21 radial structural features to characterize crystallizing versus
liquid environments; the reported descriptor cutoff is 10.8 Å. Studies of Si and
Cu connect interface-related liquid ordering with mobility and growth kinetics.
The reported 96% classification accuracy is not comparable to our distance AP
or calibrated path recall. A descriptor centered on liquid can include solid
neighbors: the paper does not establish performance with every crystalline
atom excluded from the full observation. That distinction is crucial to our
precursor question.
[Paper](https://www.nature.com/articles/s41467-020-16892-4).

### Hu and Tanaka (2022): perturb the proposed precursor

**Revealing the role of liquid preordering in crystallisation of supercooled
liquids**, Nature Communications 13, 4519.

Their principal NiAl simulations identify bond-orientational preordering around
growing crystals. A biasing procedure suppresses this order and greatly reduces
crystallization rates. This intervention supplies evidence beyond an attractive
spatial correlation plot, within that simulated system. It neither supplies a
warning-distance detector nor establishes the effect for our Al potential.
[Paper](https://www.nature.com/articles/s41467-022-32241-z).

## Related representation and nucleation studies

### Xie et al. (2019): GDyNet

**Graph dynamical networks for unsupervised learning of atomic scale dynamics in
materials**, Nature Communications 10, 2667.

Graph representations trained with VAMP/Koopman objectives identify four Si
states in a Si–Au system: bulk liquid, bulk solid and their respective interfacial
states. Figure 3 includes spatial state profiles and dynamical validation. This
shows that interface information can emerge from an objective without crystal
class labels. It is a useful precedent for our separate self-supervised branch,
with a different objective from spatial first-warning detection.
[Paper](https://www.nature.com/articles/s41467-019-10663-6);
[author code](https://github.com/txie-93/gdynet).

### Díaz Leines and Rogal (2018): Ni nucleus and its surroundings

**Maximum Likelihood Analysis of Reaction Coordinates during Solidification in
Ni**, Journal of Physical Chemistry B 122, 10934–10942.

Transition-path sampling and likelihood analysis identify the prestructured
liquid surrounding a crystalline cluster as useful additional information for
the nucleation reaction coordinate. This motivates observing the surroundings
of candidate nuclei. It addresses nucleus formation and transition mechanisms,
not a spatial approach to an existing crystal. The assessment here is based on
the authors' abstract, not a full reproduction of their sampling protocol.
[Paper](https://doi.org/10.1021/acs.jpcb.8b08718);
[author abstract](https://pubmed.ncbi.nlm.nih.gov/30362758/).

### de Jager, Smallenburg and Filion (2023): negative precursor result

**In search of a precursor for crystal nucleation of hard and charged colloids**,
Journal of Chemical Physics 159, 134902.

They follow spontaneous nucleation using conventional structure measures and
unsupervised learning. In the systems examined, structural signs emerge with
nucleation rather than as a separately detectable precursor. This is a useful
counterexample to assuming that a richer encoder must expose advance warning.
It is not evidence that Al lacks precursors or that interfaces cannot order
nearby liquid.
[Published paper](https://doi.org/10.1063/5.0161356);
[author preprint](https://arxiv.org/abs/2306.05886).

## Interpretation for our results

The following are our inferences from the comparison, not claims made by those
papers.

1. **Our strongest current result overlaps established interface recognition.**
   In our exploratory 20 Å alarm analysis, 97–98% of detected approaches already
   include reference crystal in the contextual observations. This is consistent
   with the kinds of spatial discrimination studied above. It does not yet
   demonstrate the stronger crystal-free warning capability.
2. **Observation radius and physical influence length must be separated.**
   A patch centered 13 Å from crystal can contain crystal if its contextual
   support extends farther. Its first alarm is not a measurement of a 13 Å
   structural precursor or interfacial ordering length. Graph edge cutoff,
   complete receptive field, profile width and our nearest-crystal-atom
   distance are four different quantities.
3. **A smooth order field is not a calibrated predictor.** Interfacial profiles
   motivate representations and baselines, but our warning task still requires
   independent calibration and proper predictive scores. Published classification
   accuracy, growth velocity and reaction-coordinate quality cannot be compared
   numerically to our path recall or temporal AP.
4. **The physical question remains worthwhile.** A useful contribution would
   quantify how much information about proximity and later transformation is
   available from naturally crystal-free observations, how that information
   changes with observation radius, and whether simple order measures already
   explain it. A reproducible failure to find additional information would also
   constrain our encoder hypotheses.

## Literature-informed controls for the next experiment

These are proposed adaptations, not experiments run as part of this review.

- **Match spatial support before comparing representations.** Give radial
  descriptors and continuous bond-order features the same 25-patch observations
  and predictor family as MACE. Our existing local-only geometry baseline cannot
  isolate the contribution of learned features from the benefit of wider input.
- **Add established structural controls.** Neighbor-averaged harmonic order
  features follow [Lechner and Dellago (2008)](https://arxiv.org/abs/0806.3345).
  A local [PTM detector (Larsen et al., 2016)](https://arxiv.org/abs/1603.05143)
  supplies a recognition baseline. Because our reference labels already depend
  on PTM, its performance is partly circular; report that explicitly and keep
  it separate from a claim of independent physical validation. Neither control
  may access the full-cell reference-component distance as an input.
- **Use continuous order and strict visibility audits together.** Binary crystal
  labels can miss partial order. Retain the existing input-atom audit and also
  inspect continuous order profiles. Evaluate naturally clear complete receptive
  fields; simply deleting crystalline atoms would change the observed geometry
  and introduce an artificial cue.
- **Keep spatial and temporal tasks separate.** A locked 20 Å proximity alarm
  can evaluate navigation toward a crystal. A different experiment should ask
  whether liquid-only features predict later ordering or nucleus establishment.
  For the latter, match current proximity and simple local order before claiming
  that the embedding contains additional precursor information.
- **Retain our research constraints.** Train and select with declared proper
  likelihoods, use AP only diagnostically, preserve source splits and keep
  temperature/age/time out of predictor inputs. An unmodified literature model
  with thermal-ensemble supervision is a different treatment whose actual
  training targets must be documented; it is not silently interchangeable with
  our likelihood-trained encoders. No new physical-reconstruction pretraining is
  proposed.

## Search coverage and limitations

Queries covered combinations of crystal proximity/distance, interface-induced
ordering, Al solid–liquid interfaces, structural ML scores, GNN interface
classification, spatial early warning and positive/negative nucleation-precursor
studies. Exact titles and citation links were then followed. Technical claims
above rely on primary papers, author abstracts or author repositories; review
papers were used only as discovery aids.

Relevant methods/results passages were checked for the main interface papers
and GDyNet; the Ni reaction-coordinate entry is explicitly abstract-level. Some
publisher/PMC page openings failed, so indexed primary-paper text and author
PDF links were used where available. The search is targeted rather than a
systematic bibliographic review, and does not justify a priority claim for the
specific stopping-rule protocol. Later work may use different terminology.

No training, simulation, environment installation or remote job submission was
performed for this literature review.
