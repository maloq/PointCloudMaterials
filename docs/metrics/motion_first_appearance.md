# First observed local crystalline appearance

This is a separate prospective panel assay. It does not relabel or filter the
previous established-nucleus founder/control benchmark. The original consolidated
`al-nuclei-birth-010` fingerprint and its full velocity-bearing source/ancestor
contract are pinned. Catalogue birth events and eventual fates do not select
locations, observation times or outcomes. Eight unavailable source views retain
their original no-velocity/outside-fixed-ancestry exclusions.

At elapsed 10, 40, 70 and 100 ps from each recorded source view's first frame,
128 atom IDs are drawn uniformly without replacement using a source/window hash
seed. Selection is saved before any PTM label is read. Present inputs are the
original unrelaxed nearest-80 coordinates and actual same-frame velocities.
Coordinates are centered with periodic minimum images; encoder support is 8 A.
The consumed atom IDs (nearest80 intersect radius8) are fixed throughout the
future. This is a material patch, not a sphere whose membership changes in time.
Physical labels use original **full-cell periodic** PTM, RMSD cutoff 0.1, with
FCC/HCP/BCC types 1/2/3. Neither truncated-patch PTM nor minimum component size is
used. The same classifier is applied at every native exact 0.1 ps observation.

The risk set contains sampled patches with no crystalline atom **at the present**.
This restriction is inherent to predicting an appearance from a currently clear
observation; it is not an external crystal-distance screen. No requirement is
placed on nearby crystals outside the consumed atoms. No future labels determine
eligibility. Labels at the past 0.5 ps are retained for a separately reported
history-clear subgroup, never used to exclude primary rows or supplied to models.

Let k be the first future saved frame at which any consumed present atom is
PTM-crystalline. At horizon H, y(H)=1 when 0 < 0.1 k <= H, otherwise zero with
complete follow-up. Horizons 0.1/0.5/1/3/6 ps use the **same** present inputs and
rows. A positive can dissolve immediately; neither 8/64-atom growth nor successful
establishment is required. This is the next first observed appearance after the
present, not a claim about first-ever crystallinity in an atom's entire history.
Onset is interval-censored to the 0.1 ps sampling grid; intermediate flickers may
be missed. A separate persistence sensitivity requires the **same atom** to be
classified crystalline on three consecutive frames (0.2 ps between first and
last). Its onset is the first of those frames. Follow-up extends by 0.2 ps to
confirm every possible onset through the 6 ps maximum horizon. The persistence
readouts are fitted at the declared 3 ps horizon only.

Exact present Q,V duplicates are reduced in fixed source/window order, independent
of label; alias records disclose differing future onsets if present. Identical
Q,V across independent declared ancestors, identical Q across train/test, or
ancestral train/test overlap is a hard error. The original selection, calibration
and test roles are pooled for evaluation. Their labels never select fits.
Eight original training ancestors chosen by a fixed hash provide internal NLL
selection; final probes refit on all original training ancestors. Unsupported
single-class partitions are recorded as unsupported, not assigned artificial
metrics. No temperature, absolute time, simulation age, source identity or label
metadata is passed to any predictor. Observation times organize extraction only.

All eight completed motion models and both 12-epoch spatial starts retain their
original weights and scientific selection. Structural code has 128 components;
structural+motion has 256; motion-only has 128. The spatial-start motion control
is the same finite-difference directional encoding of its pretrained geometry,
without motion training. No stochastic future-law branch is used in this assay.
Four present-feature controls use the previous exact input descriptors: qbar6,
all bond order, TDA, and nine instantaneous velocity summaries. A training-prior
predictor supplies a likelihood reference. CatBoost depth4, lr0.04, maximum600
iterations and patience75 uses training-only NLL iteration selection. No AP
selection, class balancing or outcome-dependent sampling is performed.

Each independent ancestor has total weight one. AP uses the existing
sklearn-equivalent weighted precision-recall calculation; AUROC, binary NLL and
Brier use the same weights. All reported primary confidence intervals use 300
whole-ancestor bootstrap draws conditional on fitted models. The historical
bootstrap implementation excludes draws missing a class and reports the valid
count. The history-clear subgroup has point estimates only and is explicitly
marked as such. Ten fixed-width probability bins report weighted observed and
predicted rates. Raw row prevalence and ancestor-weighted prevalence are both
exported. AP's chance reference is this assay's event prevalence, not the old
forced 20%. Horizon classification differs from lead-before-a-selected-birth
classification; their AP values must not be compared as a matched improvement.

The data were previously explored under another target, so this remains an
exploratory evaluation. Random local onsets can include transient order or an
arriving crystalline front; they are not all independent homogeneous-nucleation
events. Multiple sites, overlapping patches, source views and horizons do not
increase the independent ancestor count.


## Parent dataset cadence contract (October 7, 2026)

The shared catalogue now restricts `al-nuclei-birth-030` to uniformly saved
native 0.30 ps inputs, checking raw timestep arrays. The `010` source/event
population used here is unchanged. New preparations pin the current parent
fingerprint; frozen runs keep their original pins and metric definitions.
Source-bound 0.10 ps label/history supplements retain their original provenance.
