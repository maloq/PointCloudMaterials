# How useful is the original 0.75 ps Al campaign for nucleus-birth detection?

**Historical audit:** completed before the user's October 7 decision to delete
the original 150 production trajectories. The tables and figures remain valid
as saved audit results; raw inputs are now unavailable, so the reproduction
command below requires data that is no longer retained. The campaign is retired
from new experiments. See the
[removal record](../../../../../docs/storage/al075_removal_20261007/README.md).

**Assessment: useful for sustained liquid-origin nucleus establishment and broad
precursor studies; incomplete for the earliest appearance, short-lived embryos,
and detailed formation-versus-dissolution kinetics.** This is an audit of saved
search evidence, not a measurement of detector precision or recall against ground
truth. No simulation, model training or new structural extraction was launched.

## The actual original campaign

The original campaign comprises **150 trajectories from 150 melt ancestors**,
30 each at 400, 450, 500, 510 and 520 K. Each records 600 ps at 0.75 ps spacing:
801 observations per source, **90 ns total**. All 150 have saved velocity arrays
at that cadence; those are not missing 0.10-ps velocities reconstructed by
finite differences. Input decks use native LAMMPS MEAM, 3 fs integration, and
15 ps of initial equilibration before observation time is reset to zero.

The broader saved search has **179 exact 0.75 ps views**. Its additional 29 views
are other campaigns, including finer-cadence histories and related partial
records. Its aggregate 412 PTM established liquid-origin lineages must not be
reported as the yield of the original 150-source campaign.

## Detected established lineages

Using the completed v3 lineage search, the original campaign contains **336 PTM
liquid-origin independent establishments in 145/150 sources**. A record first
appears at eight connected atoms; establishment requires a connected descendant
of at least 64 atoms across a sampled 1.5 ps span. At 0.75 ps this means three
saved observations. These are operational definitions, not a measured critical
nucleus or proof of continuous persistence between frames.

| Temperature | Sources with a PTM established liquid-origin lineage | Established lineages | Liquid-origin embryo appearances |
|---|---:|---:|---:|
| 400 K | 30/30 | 98 | 9,104 |
| 450 K | 30/30 | 114 | 2,986 |
| 500 K | 29/30 | 47 | 846 |
| 510 K | 30/30 | 47 | 710 |
| 520 K | 26/30 | 30 | 624 |
| Total | 145/150 | 336 | 14,270 |

These lineages can share one simulation and are not 336 independent experiments.
The median is two established lineages per trajectory, with a maximum of seven.
The temperature table describes observed yields; it is not a nucleation-rate
estimate, since the time spent as unaffected liquid and subsequent transformation
differ across trajectories.

Another **33 composite families establish after subcritical families coalesce**.
They are a separate pathway, not 33 extra independent liquid-origin births to add
to the 336. One source with no established liquid-origin root (old-860) forms a
large crystal through this route. The other four zero-root-establishment sources
have maximum PTM connected sizes of 14–37 atoms in the saved audit.

## How sensitive is this result to the structural detector?

| Detector | Established liquid-origin lineages | Liquid-origin embryo appearances |
|---|---:|---:|
| PTM | 336 | 14,270 |
| Reference coherent bond order | 335 | 66,510 |
| Strict coherent bond order | 338 | 11,818 |
| Loose coherent bond order | 418 | 559,118 |

The close PTM/reference/strict established totals support stability of the
aggregate yield. They do **not** show that all three identify the same physical
events. Small-embryo appearance is much more dependent on the detector.

The saved PTM/reference comparison matches 6,011 of 15,291 PTM root appearance
records to 71,369 bond-order root records, including initial and uncertain origins.
Among those matches, origins agree in 94.9% and terminal fates in 59.4%; the median
absolute appearance-time difference is 2.25 ps. These diagnostics use onset-member
and lifetime overlap, include many tiny fluctuations, and are **not established-only
precision/recall**. A definition of first appearance can introduce uncertainty
larger than the 0.75 ps sampling interval itself.

## What does coarse sampling lose on the same trajectories?

The strongest existing count comparison uses **22 other Al birth-campaign
trajectories saved natively at 0.15 ps**, inspected at exact 0.30 and 0.75 ps.
Both views describe the same MD; they are not interpolated, although their
observation grids are not nested.

| PTM records on the same 22 trajectories | 0.30 ps view | 0.75 ps view |
|---|---:|---:|
| Liquid-origin embryo appearances | 4,550 | 2,560 |
| Established liquid-origin lineages | 64 | 67 |

The 0.75 ps view has **43.7% fewer detected embryo appearances**, while the
established count is 4.7% higher. This shows that coarse sampling changes which
short-lived fluctuations are seen and how families are linked; it does not
prove that exactly 43.7% of physical embryos were missed. A higher established
count is not automatically better: coarse frames can hide temporary dissolution
or interactions and alter persistence/lineage assignments.

A small second check uses two completed native-0.01-ps descendants, excluding
historical partial duplicates. Their 0.10 and 0.75 ps views contain respectively
1,061 and 314 PTM liquid-origin appearances, with three established lineages at
either cadence. Equal totals do not establish event identity, and two trajectories
are insufficient to quantify a general detection probability. These descendants
share ancestry with the original campaign and add no independent melt ancestors.

![Campaign yield and matched-cadence counts](plots/campaign-and-cadence.png)

## Limits for detecting genuinely new births

1. **The first 15 ps are absent.** All 150 source input decks delay trajectory
   output until after equilibration. At the first saved frame, **29/150 sources
   already contain an >=8-atom PTM embryo**, comprising 32 initial embryo records.
   None starts with a >=64-atom PTM component in that saved frame. Growth of an
   initial embryo must remain separate from an embryo first observed forming
   later. The v3 catalogue keeps the initial records separate, but cannot recover
   how they formed during the missing initial history.
2. **A persistent crossing is bracketed only to a 0.75 ps observation interval.**
   A cluster that forms and disappears between observations may be invisible.
   Three apparently persistent observations can also conceal subframe loss and
   reappearance. Relative to a 3 ps prediction horizon, a 0.75 ps onset interval
   is substantial.
3. **Origins require atom-identity tracking across frames.** Coarse sampling can
   obscure exchange, fragmentation and mergers. Low local crystallinity or a
   newly crystalline tracked atom alone does not demonstrate new nucleation;
   full-cell lineage and existing-front proximity remain necessary.
4. **Detection and prediction eligibility are separate.** A historical strict
   preappearance experiment extracted 95 distinct birth events from 77 sources,
   with 295 positive center histories and 1,180 matched controls. That used an
   earlier birth definition and zero-PTM input histories in moving 8 Å regions.
   It demonstrates that useful precursor examples exist; it is not the complete
   eligible subset of these 336 v3 lineages. Original train/selection/calibration/
   test roles remain recorded. Initial-history gaps and retrospective site/control
   selection need separate treatment in any new predictive assay.

## Scientific interpretation before retirement

The original campaign offered a broad independent-ancestor benchmark for
sustained establishment and 3/6 ps precursor information, subject to event-time
intervals and initial/ambiguous origins. It is now retired. Use dense continuous 0.10 ps data for
first crystal appearance, brief embryos, competing fates and time-resolved
formation mechanisms. Never merge different cadences as interchangeable histories.

Before claiming a numerical detection accuracy, perform an event-by-event
matched-cadence audit on native0.01/0.075/0.15 ps trajectories that support exact
0.75 ps views, using several sampling offsets. A native0.10 ps history cannot
produce an exact0.75 ps grid by integer subsampling. Compare establishment,
origin and fate of the same families, not only totals. The finer catalogue is
still an operational reference rather than physical ground truth.

## Evidence, completeness and reproduction

All 150 original sources have completed four-detector v3 source summaries.
The source tables and input receipts here retain their exact provenance.
The cleaned v3 search no longer retains the original150 per-event0.75-ps JSON
shards. At audit time, original trajectories, completion receipts and exported
source/paired summaries were available; the trajectories were subsequently deleted.
This audit therefore does not claim a new established-only
match or re-evaluate every event's precursor eligibility. Those require label
regeneration. Earlier prediction cohorts retain their own original artifacts.

The new V100 birth-focused queue, CPU dense descendants, shooting, million-atom
replay and relaxed views are separate holdings. This assessment does not declare
all Al searches or simulations complete and does not change any running jobs.

```bash
conda run -n pointnet-torch214 python -m src.research.crystallization_origin.cadence_audit \
  --config configs/analysis/al075_birth_quality_20261007.json
```

[Definitions and frozen implementation](tables/METRICS.md) ·
[Source inventory](tables/sources.csv) ·
[Temperature results](tables/temperatures.csv) ·
[Detector sensitivity](tables/detectors.csv) ·
[Matched-cadence comparisons](tables/matched-cadences.csv) ·
[Input checksums](technical/input-receipts.json)
