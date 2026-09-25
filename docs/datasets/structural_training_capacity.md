# Existing structural-training capacity — 25 September 2026

This is an availability calculation, not a new dataset release or launch. The
current runs and fixed Al64 prediction rows are unchanged. More structural centers
must come only from training ancestors; all64 and legacy16 evaluation stay fixed.

## Native Al training sources

The sealed `fixed-cohort/al64-v1-20260925/plan.json` contains 90 training
trajectories, each with 70,304 atoms and 801 saved frames spanning 600 ps at
0.75 ps cadence. Current physical pretraining extracts only 64 centers and 201
frames at 3 ps spacing: 1,157,760 neighborhoods.

Across the pilot, expanded and large-test relaxed caches and the expanded-TDA
completion records, deduplication by original source-manifest hash and frame
finds **1,710 completed native training cells**, 19 per training source. Their
archived full-cell folders are present. Current paired pretraining uses 15 frames
per source (86,400 pairs at 64 centers). The older expanded cache supplies more
completed training frames, including adjacent frames. These are not independent
observations and do not increase the independent-source count.

| Centers per frame | Observed neighborhoods at 3 ps spacing | Pairs from all 1,710 completed training quenches |
| ---: | ---: | ---: |
| 64 | 1,157,760 | 109,440 |
| 512 | 9,262,080 | 875,520 |
| 1,024 | 18,524,160 | 1,751,040 |
| 4,096 | 74,096,640 | 7,004,160 |
| All 70,304 | 1,271,799,360 | 120,219,840 |

Including every stored frame and every atom in these same 90 sources gives
**5,068,215,360 possible observed center/frame neighborhoods**. These are
extractable candidates, not an existing patch cache or independent samples.
Spatial neighborhoods overlap and nearby frames are correlated. The 64-center
benchmark contract does not limit structural sampling from its training sources.

At float32, 80 atoms × 3 coordinates costs 960 bytes per observed neighborhood,
before graph edges, targets and model activations. The 1,024-center observed
option is about 16.56 GiB of coordinate payload; the exhaustive observed pool is
about 4.43 TiB. Paired coordinates double the per-row payload. The current loader
concatenates whole source arrays on the host and loads the complete pool onto a
GPU; scaling should use bounded shards or batches extracted from raw frames,
rather than materializing the exhaustive pool or allocating it all in VRAM.

## Further existing collections

Three completed precision-source trajectories are separately registered on STORE
(`memory-al-precision-20260917-source000/001/002-T500`). Each is preassigned train,
has a distinct melt lineage outside the fixed 150-source cohort, and has
2,561 × 70,304 observed positions at 0.075 ps cadence. Float32 arrays are present.
Together these offer another 540,145,632 possible center/frame observations.
Their float16 exports are copies, not additional trajectories. Adding them
requires a new structural-training release while preserving the fixed evaluation.

The completed million-atom Al campaign
`al_meam_1m_450K_400ps_with_melt_20260913T205405Z` retains 3,001 melt frames and
4,001 measurement frames, each with 1,000,000 atoms (array headers inspected).
That is 7.002 billion stored atom/frame observations from one preparation lineage.
It is a separate protocol and is not already assigned to the current structural
release; train eligibility and ancestry must be declared before use. It should
not overwhelm the independently replicated source distribution merely because it
contains more atoms.

The mixed-material dynamic structural cache is also complete and records
1,018,080 dynamic training anchors across Al/Mg/Ti/Ta, with separate inherited
static material. These are derived views of existing trajectories, not additive
independent data. Their material, potential, support and coordinate normalization
contracts differ from the present native-Al encoder; reusing them needs an
explicit material-aware protocol and ancestry filtering.

The expanded-TDA campaign is only partially complete: 1,667 task references,
106,688 center references and 780 unique relaxed cells across all roles were
recorded during this inspection. Its planned 1,280,960 training references must
not be advertised as completed pairs. Its native completed frames are already
contained in the 1,710-cell union above.

## Practical next scale

A useful first expansion is 1,024 outcome-independent centers per training source:
18.52 million observed neighborhoods and 1.75 million existing relaxed pairs.
This increases spatial coverage before densely repeating every adjacent frame.
Sample sources equally, preserve a documented liquid/crystal population, and
record both optimizer updates and examples processed. Twelve full epochs on the
larger sets costs much more than twelve epochs in the current study. Different
pretraining populations/budgets remain pipeline comparisons, not isolated loss
comparisons. No new simulation or quench is needed for these sizes.
