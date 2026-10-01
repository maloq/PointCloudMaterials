# Spatial VICReg mechanism study and coordinate diagnostics

## Six relaxed static Al snapshots

The [static recipe](../configs/analysis/static_al_interface.json) extends the
interface comparison to **166, 170, 174, 175, 177 and 240 ps**. It reuses the
established static grid: **684,723 centers**, with all 1,048,576 source atoms per
snapshot available to nearest-80 neighborhoods. The inherited grid contributes
only coordinates and source-row identities, not the previous model's labels.
The frozen GeoFormers (S1-seed17 epoch24 and S0-seed17 epoch4), their encoder and
projector centroids, and the earlier TDA/bond-order/CNA/joint cluster models are
unchanged. No neural training or cluster fitting is performed.

The static structures are already relaxed and lack recorded periodic-box
metadata. This workflow therefore uses complete nonperiodic neighborhoods,
full-source PTM, and an 8 Å outer-boundary guard for interface atoms. Coordinate
bounds are display bounds, not an invented periodic cell. Snapshot time is
display metadata only; source-row indices are not asserted to be persistent
atom IDs. This is descriptive transfer, not held-out crystallization prediction.
[Exact definitions](metrics/static_interface.md).

```bash
python -m src.research.spatial_vicreg_bias.static_md submit \
  --config configs/analysis/static_al_interface.json
```

Submission: CPU preparation **1014065** (six tasks, three concurrent); frozen
inference **1014066** (two checkpoints); descriptor PaCMAP **1014067**;
publication **1014068**, dependent on both. GPUs may be A40, RTX3090 or L40S;
there is no required node. Sampled neural vectors remain in RAM through PaCMAP,
so this workflow creates no persistent encoder feature bank. All calculations
run on Slurm, using `pointnet-torch214`.

[Static gallery](../output/spatial_vicreg_bias/static-al-six-20260929/analyses/interface-v1/index.html)
contains sixteen completed views (eight spaces × all-static/interface20), each with
2D and 3D PaCMAP plus two independent dense MD panels and a six-snapshot selector.
The 24,000 projection observations are identical across spaces; MD displays every
grid center. Per-snapshot interface/shell correspondence tables retain counts,
ARI, AMI, MI, conditional entropies and contingency matrices. The durable source
is `WORK/analysis/spatial_vicreg_bias/static-al-six-20260929/analyses/interface-v1`;
the gallery, figures, tables and provenance are real copies in repo output.

All six preparation tasks, both inference tasks and all projections completed.
Inference used A40 node22/node25 concurrently after a recorded throttle override.
The interface20 projection population has 8,834 of the 24,000 sampled centers.
Numerical publication **1014068** initially rebound the resolved repo path to
its frozen code root. Copy-only recovery **1014130** published the completed
bundle to the real repository; `technical/publication.json` records both paths
and copied artifact hashes. Future submissions preserve resolved absolute paths.
Browser review **1014131** completed: all sixteen pages, six snapshot assets and
four neural spaces passed output checks, and the 166 ps full view and 177 ps
interior slab were visually inspected. PNG previews are copied into `plots/`.

The requested paired 3D display uses `paired_pacmap` to republish the sixteen
interactive pages from saved coordinates: neural PaCMAP on the left and rich
descriptor PaCMAP on the right, with independent feature-space selectors.
Both layouts must have identical sample/frame/atom/grid-row identities and
color fields. Each panel defaults to its own frozen cluster assignments; other
shared coloring options remain available. Dense MD and unlinked atom behavior
are preserved. The 2D PNG figures remain in the gallery. This is rendering only;
the scientific projections, metrics and frozen definitions are unchanged.
`technical/rendering/paired-pacmap.json` records the new rendering and hashes.

The current display adds a one-to-one optimal color assignment for each
neural/descriptor pair, fitted descriptively to the pooled 684,723 dense-grid
memberships. It remains fixed across snapshots and filters; original neural IDs
remain visible as `N → D` in legends. The heatmap, matched-pair IoU bars, matched
fraction and ARI describe the currently displayed PaCMAP subset. The same
model/family selectors now govern both PaCMAP and MD, with no atom linkage.
Long explanations are collapsed under Methods. New pooled/per-snapshot overlap
exports live separately in `analyses/cluster-color-matching-v1`; existing metrics
and cluster assignments are preserved. [Definitions](metrics/cluster_color_matching.md).
CPU job **1014192** published all sixteen updated pages; browser review
**1014193** verified the default 24,000-center comparison and a filtered
1,836-center comparison after changing model and descriptor family. The
PaCMAP/MD neural legend colors agree in both views. The default sampled view
has 73.8% matched overlap and ARI 0.575; this is distinct from its 73.6% pooled
dense-grid matching reference. Low or zero per-pair IoU remains visible even
when the global one-to-one assignment must allocate that pair a shared color.
[Findings](../output/spatial_vicreg_bias/static-al-six-20260929/analyses/interface-v1/RESULTS.md)
separate interface correspondence from the weak correspondence in nearby
crystal-free inputs; neither establishes predictive liquid states.

## Dense MD comparison

The user requested full density, two MD panels, consistent cluster colors, and
removal of interactive atom correspondence. The [dense recipe](../configs/analysis/interface_dense_md.json)
uses all **70,304 atoms** in source **908**, frame **768**, selected by the largest
accepted interface layer among the 390 recorded held-out snapshots. This is a
visualization choice; the fixed evaluation cohort and fitted clusterers are unchanged.

`dense_md prepare` evaluates the exact periodic nearest-80 descriptors on Slurm
CPUs and assigns the frozen train-fitted TDA/bond-order/CNA/joint centroids.
Previously observed patches and their saved descriptors are preserved after a
producer-agreement check. Job **1014002** completed full classical coloring;
job **1014026** verified the exact original float32 assignment path with zero
differences in all four families on the 64 original observations.
`dense_md infer` uses the original frozen model-code snapshot, CUDA float32,
eval/inference mode, and saved K=7 encoder/projector centroids. No optimizer,
training, new cluster fitting, future inputs or W&B runs. Job **1014010** completed
in 44 seconds on **A40 node22** after removing the node58 restriction, for
S1-seed17 epoch24 and S0-seed17 epoch4. All four encoder/projector comparisons
against saved assignments had zero discrepancies on the original 64 observations.
The original submitted configuration is preserved; the scheduling override has
its own receipt in `technical/gpu-scheduling-override.json`.

`md_space --config RESOLVED_PACMAP_CONFIG --dense-config RESOLVED_DENSE_CONFIG`
publishes two independent dense MD panels: neural clusters and joint classical
clusters (with individual descriptor-family options). Each cluster keeps the
same palette entry in PaCMAP and MD. Independent neural/classical numeric IDs
do not imply semantic alignment. The same full snapshot and optional z slab
are displayed in both panels. Point size, camera and legend controls remain;
atom selection/hover linkage and automatic cross-panel redraw are removed.
Shared snapshot/label assets load on demand instead of embedding every full
snapshot in each page. Unavailable neural colors are explicitly marked pending.

Twelve existing pages were republished by job **1014020**. Render jobs **1014017**,
**1014018** and **1014019** refresh after dense inference, the GPU projection
preview and the full projection sweep. Old linked-render jobs 1013990/1013991
were cancelled so they cannot overwrite the new viewer. Browser rendering was
inspected with Chromium in Slurm job **1014022**. Receipts and the resolved
recipe/scripts are in the WORK `analyses/dense-md-v1/technical` directory.
The final dense-panel Chromium preview completed in job **1014039**; the full
classical atom cloud was visually inspected. Job **1014074** subsequently rendered
and visually verified both dense NN and descriptor panels with all 70,304 atoms.
[Assignment definitions](metrics/dense_md.md).

Historical display: job 1013978 initially published a linked panel containing
only 64 sampled centers per snapshot. The original numerical PaCMAP artifacts
and their frozen definitions remain unchanged.

## PaCMAP views of frozen features

The [PaCMAP recipe](../configs/analysis/interface_pacmap.json) adds 2D PNG panels and offline interactive 2D/3D views for epochs 4/12/24, all nine models, encoder/projector and TDA/bond-order/CNA/joint features. It reuses frozen cluster assignments and may regenerate evicted vectors with frozen CPU inference only: no optimizer or backward pass. Initial production preview job **1013965** is followed by full sweep **1013966**, dependent on both the preview and correspondence job **1013932**. All calculations run on Slurm CPUs; [definitions](metrics/interface_pacmap.md). Install the optional pinned [PaCMAP dependencies](../environments/requirements-pacmap.txt) in pointnet-torch214. Completed views publish incrementally under the same repo run in `analyses/interface-pacmap-v1/index.html`.

## Interface cluster correspondence

The user clarified that the intended diagnostic is independent clustering of rich TDA, bond order and CNA, with the interface and surrounding layers as the primary population. [Scientific protocol](../experiments/spatial_vicreg_bias_20260929/INTERFACE_CORRESPONDENCE.md) and [metric definition](metrics/interface_cluster_correspondence.md). Submitted CPU jobs **1013931** (physical reference) and **1013932** (dependent correspondence). The new `correspondence submit` workflow reuses frozen neural assignments and rich descriptors, adds instantaneous periodic interface-layer references, and submits CPU preparation followed by correspondence analysis. It creates no scientific W&B run. Outputs are under the original WORK run in `analyses/interface-correspondence-v1`; completed plots/tables are copied into the matching repo output folder. Previous descriptor-readout results remain unchanged.

The [scientific protocol](../experiments/spatial_vicreg_bias_20260929/README.md)
is complete. Nine matched GeoFormer fits finished: alignment strengths
0/0.5/1 × seeds 17/29/43, each with 24 complete unfiltered training passes.
[Recipe](../configs/spatial_vicreg_bias/al64_20260929.json) ·
[Metric definitions](metrics/spatial_vicreg_bias.md).

All nine fits completed 24 passes/108,552 updates, with all 63 checkpoint assays,
six reference controls and 18 associated W&B endpoint updates complete.
[Scientific results](../experiments/spatial_vicreg_bias_20260929/RESULTS.md) ·
[Copied summary figures and tables](../output/spatial_vicreg_bias/matched-al64-20260929/analyses/completed-review-v1/README.md).
The saved-output review ran through CPU Slurm job 1013898; it did not refit models.

## Execution and saved work

Use conda `pointnet-torch214`. All preparation/training/analysis runs through
Slurm. Submission receipt:
`/work/PERSO/vmorozov/analysis/spatial_vicreg_bias/matched-al64-20260929/technical/queue/launch.json`.
CPU preparation uses the original `queue/code/` snapshot. The GPU workers use
`technical/gpu-execution-v2/code/`, recorded in that receipt. The pending GPU jobs
were replaced before training to let the first scheduled lane perform preflight,
avoiding a dependency on lane 0 obtaining a GPU first. Edits to the working
repository do not change submitted calculations.

- CPU preparation/sealing: **1013509**, six workers in one allocation.
- RTX6000PRO worker lanes: **1013521/1013522/1013523**, one GPU each, excluding node58.
  Each lane runs one alignment strength across three seeds, with per-checkpoint
  assays. Lane 2 also computes six classical/null controls. Fits authenticate
  online W&B before optimization; evaluation stays local and updates existing IDs.
- The initial 12-task-array submission was rejected by the account's Slurm submit
  quota before launching anything. Its frozen attempt is preserved under
  `technical/submission-attempt-1/`. The replacement uses only four Slurm jobs.
- Deadline handling saves optimization state and requeues the same lane job;
  completed epochs/assays are reused. It does not add passes or new online runs.
  Failures are saved under `technical/failures/` and fail loudly.

Numerical results/checkpoints:
`/work/PERSO/vmorozov/analysis/spatial_vicreg_bias/matched-al64-20260929/`.
Each `S0-seed17`-style component contains `checkpoints/`, `technical/`, and named
`analyses/epoch-00`, `epoch-01`, … `epoch-24` bundles with grouped plots, tables,
scientific arrays and frozen metric definitions. `nulls/analyses/` contains the
four crystal-field controls and local/averaged q6 controls. `comparison/` shows
paired seed trajectories and explicit coverage. A missing `complete.json` does
not establish completion; use receipts/logs, not disappearance from `squeue`.

Training parents and descriptor references are reusable IDS data under
`/home/ids/vmorozov/training-cache/spatial-vicreg-bias/al64-v1-20260929`.
They inherit fixed Al64 source/sample roles. Disposable generated embeddings use
`spatial-vicreg-bias/features`, globally limited to six entries across these GPU
lanes with active leases protected. Checkpoints, assignments, physical readout
coefficients, source error arrays and metrics are retained.

```bash
python -m src.research.spatial_vicreg_bias.queue submit \
  --config configs/spatial_vicreg_bias/al64_20260929.json
```

Submission refuses an existing receipt. Do not resubmit this completed submission
command to create duplicates. New scientific variations need a distinct recipe
and output identity. The maintained module also exposes `data source|seal`,
`train smoke|train`, `evaluate encoder|nulls`, and `report` for explicit Slurm
recovery using the frozen producer/config, preserving the recorded definitions.

Local preflight receipts are `technical/preparation`, `technical/preflight` and
`technical/preflight-compiled`. Source 860 prepared successfully (12,864 training
parents, 1,472 assay patches, all 252 original benchmark rows). Both eager and
compiled checks had finite losses/gradients, exact replay of view tensors and
bitwise identical same-batch evaluation. Compiled steady forward/backward updates
were about 0.011 s after compilation in the eight-step diagnostic; this is not an
end-to-end throughput guarantee. The final producer repeats the gate before any
scientific optimization. No W&B runs were created by these checks.

## Why annotate crystalline atoms in inputs?

These are evaluation annotations. Training includes all phases and has no PTM,
TDA or future-label targets. The main interface experiment keeps mixed inputs.
The annotations distinguish an encoder observing crystal inside a nominally
liquid center's crop from an encoder distinguishing structure with no detected
crystal in its actual input. Pair diagnostics cover both views' union. Crystal
fraction and at-most-one/three-detection sensitivities accompany the strict-zero
subset. PTM-unclassified does not prove unstructured liquid. Conditional readouts
ask what cluster membership adds beyond stated physical controls; they do not
establish a precursor, committor, or future predictive skill.

## Completed archived-coordinate assay

The original job **1013457** failed a bitwise comparison between different batch
sizes. It published no valid metrics. Revision 2, job **1013483**, completed:
[results and plots](../output/spatial_vicreg_bias/archived-geof34-coordinates-20260929/analyses/coordinate-profiles-v2/README.md).
Matched 256-example repeats were exactly equal; 64-versus-256 batch maximum
errors were 2.4e−6–5.2e−6, within the explicitly recorded numerical tolerance.
The failed revision and its frozen definition remain intact.

[Recipe](../configs/analysis/spatial_vicreg_coordinates.json) ·
[metric contract](metrics/spatial_vicreg_coordinates.md) ·
[scientific interpretation](../experiments/spatial_vicreg_bias_20260929/COORDINATE_FINDINGS.md).

```bash
python -m src.research.spatial_vicreg_bias.coordinates \
  --config configs/analysis/spatial_vicreg_coordinates.json
```

The completed revision refuses overwrite. Figures (PNG/PDF), atom-centered
coordinates, paths and all native-coordinate metrics are retained under
`output/spatial_vicreg_bias/archived-geof34-coordinates-20260929/analyses/coordinate-profiles-v2/`.
This archived checkpoint actually has neighbor shifting disabled and FactorVAE
enabled. Its snapshots were in its training data. The assay describes its field;
it does not establish the causal effect of spatial-neighbor VICReg.

## Readout pilot

The full K=7 readout/profile/export path completed on 31 currently prepared
sources (55,744 rows) through a step inside Slurm allocation 1013509. The exact
subset and outputs are under `technical/readout-pilot/`; this is a local pipeline
check, not a full-cohort result or a criterion for changing the scientific recipe.
A separate pilot-job submission hit the same account quota; using the existing
CPU allocation required no additional batch job or login-node computation.
Conditional ridge improvement can reflect nonlinear feature expansion of known
physical variables. Scalar-field clusters receive identical readouts to expose
this limitation; no conditional-information or precursor claim follows from
positive regression gain alone.

### GPU replay recovery

CPU PaCMAP sweep 1013966 stopped at the unchanged saved-assignment gate. Audit 1013983 found six disagreements with both the original K-means kernel and float64 distances, so the difference precedes centroid assignment. Original incomplete artifacts are preserved. New recipe `configs/analysis/interface_pacmap_cuda.json` queues frozen inference on node58 using the exact original A+B batch layout: preview **1013986**, full sweep **1013987**. It inherits only completed descriptor projections unchanged, and keeps the same rows and verification threshold. No neural training is introduced. The MD renderer republishes after preview and completion to the existing gallery.

On the user's subsequent request, the node58 requirement was removed from those
pending jobs. Preview **1013986** completed on A40 node24. Sweep **1013987** ran
there until NFS cache eviction encountered a memory map that outlived its lease.
Its scientific outputs remain preserved. Recovery **1014072** uses the copied
`code-memory-r2` snapshot: explicitly close writable maps and load the small
sampled bank into RAM before projection. No numerical protocol or gate changed.
The proven empty failed-eviction directory was removed under both cache locks;
the six-entry retention limit and active-lease protection remain intact.
`technical/queue/memory-recovery.json` records the code hashes and job IDs.
Dependent dense renderer **1014019** now follows the recovery job. Live launch
defaults accept A40/3090/L40S and respect the four-CPU limit of the 3090 partition.

### Detached descriptor-island audit

Slurm CPU jobs **1013992** and **1013994** completed the screenshot-specific audit
of the saved joint/all_test view. The final run checks exact original-space
20-neighbor membership, separate families and ±5 training-z clipping, reusing
all saved rows and descriptors. [Findings and reproduction](../experiments/spatial_vicreg_bias_20260929/PACMAP_ISLANDS.md).
The result bundle `analyses/pacmap-islands-v1` is copied from WORK into the same
repository output run; its `technical/audit.json` includes definitions and
input/implementation hashes. No inference, training or PaCMAP refit is involved.

## Checkpoint explorer

The visible **Data** selector switches between held-out Al MD and the six static
Al snapshots (166, 170, 174, 175, 177 and 240 ps). Both open the comparison directly.
Open the top-level experiment pages: [GeoFormer held-out Al MD](../output/spatial_vicreg_bias/matched-al64-20260929/heldout-al.html)
and [GeoFormer static Al](../output/spatial_vicreg_bias/static-al-six-20260929/al-static.html).
`comparison_layout`, MACE publication and the checkpoint explorer maintain full
HTML copies at the experiment root, with asset paths pointing to the retained
analysis bundles. Publishing these entry pages does not recalculate results.
PaCMAP and MD occupy a two-column grid with square canvases filling each
column. The later request for larger square panels supersedes the previous
one-viewport height limit. Each panel can be expanded independently. Cluster examples and then correspondence follow below.
`comparison_layout --publication REPO_BUNDLE --dataset matched|static` refreshes
this display from saved page payloads without changing scientific arrays or
re-exporting frozen metrics. Its receipt is `technical/rendering/compact-layout.json`.
The frame slider below the MD panels synchronizes the PaCMAP subset, full MD
snapshot and local cluster examples. `All PaCMAP frames` restores pooled PaCMAP.
PaCMAP axes retain the full saved-layout ranges while filtering frames; MD
camera state is retained when changing snapshots. Slider positions index saved
snapshots, whose actual ps values are displayed; no intermediate time is synthesized.
Static Al has six saved times. The held-out study includes all 13 assay frames
of source 908 (0, 64, …, 768), with all 70,304 atoms in each MD view.
Slurm preparation **1014297**, frozen inference **1014304** and publication
**1014306** completed. All 351 checkpoint–frame pairs (27 checkpoints × 13 frames)
reproduce original sampled encoder/projector assignments with zero disagreements.
Full MD is source 908;
the separate PaCMAP source filter can retain all held-out sources.

The interactive local samples show up to five actual 80-atom neighborhoods per
cluster, drawn uniformly without replacement with deterministic seeds from the
selected full snapshot. They are examples, not fitted centroids or selected
typical structures. Rendering reuses `src.analysis.representative_style`:
PCA orientation and sparse geometric connections (these are display connections,
not inferred chemical bonds). The current viewer uses a uniform saturated
cluster color and a slightly larger central atom; colors do not label individual
neighbors. Earlier radial-color assets remain preserved. Both panels share an angstrom scale. Selecting a neural cluster
initially selects its assigned descriptor
partner; either side's example can then be changed. These examples are independent
of region filters and MD z slabs. `sample_environments` reads saved patches and
labels; hashes, selected atom identities and selection rules accompany the assets
under `sample-data/` and `technical/rendering/environment-samples.json`.

The timeline bundle is in repository output at
`output/spatial_vicreg_bias/matched-al64-20260929/analyses/dense-md-timeline-v1`.
WORK exhausted its quota during preparation; only this request's unfinished
outputs were moved to this bundle. Existing inputs and checkpoints remain at
their recorded locations. `technical/timeline/storage.json` records that
operational deviation, and `launch.json` records the Slurm dependency chain.
Per-snapshot sample export uses four CPU workers; publication waits for all
snapshot manifests and validates checkpoint and asset hashes before extending
the page's frame list. This changes neither training nor existing projections,
cluster fits, matching reference populations or frozen metric exports.

The matched-study [main comparison](../output/spatial_vicreg_bias/matched-al64-20260929/analyses/interface-pacmap-v1/index.html)
now opens a single GeoFormer-versus-descriptor view. Choose epochs 4, 12 or 24
without navigating the former 116-row gallery. The default is spatial-neighbor
VICReg, raw encoder, repeat 1 (seed 17), epoch 24; it is a fixed browsing default,
not a claim that this checkpoint is best. Training pairing variants, repeat seeds
and the VICReg loss projector are in advanced controls. Layout selection keeps
the original all-test and interface20 projections distinct.

S0, S0.5 and S1 are the neighbor-alignment coefficient: same-center augmented
views only, equal same-center/spatial-neighbor alignment, or neighbor alignment
only. The architecture and variance/covariance terms are unchanged. Seeds 17,
29 and 43 are independent training repeats; treatment comparisons within each
seed used matched initialization, order and augmentation. These labels are not
different neural architectures or simulation timestamps.

The explorer reuses all 116 saved projections and preserves their 2D figures.
Old page links select the corresponding checkpoint in the new explorer. Dense
MD uses that selected checkpoint's frozen encoder/projector and original K=7
centroids; no alternate model is substituted. The expanded inference queue uses
any available A40/3090/L40S GPU with no node pin and does not retrain networks.

Slurm inference **1014209** completed on A40 node22 in 2m59s: all 27 checkpoints
have 70,304-atom assignments for both encoder and projector, with zero replay
disagreements on the original saved atoms. CPU publication **1014213** and browser
review **1014216** completed. The browser review covered the default, epoch 12
with source/frame/region filters, and an alternative pairing/repeat/projector.
PaCMAP and MD colors agreed in each case; the default browser correspondence
matched the saved numerical table. Review receipts are in the publication's
`technical/rendering/checkpoint-browser.json`.

Resolved inference configuration and execution receipts live under the existing
`dense-md-v1/technical/checkpoint-explorer-*` paths. Run `dense_md infer` with that
config, then `checkpoint_explorer --dense-config` in a Slurm CPU job. The new
`checkpoint-cluster-matching-v1` bundle defines optimal colors on the fixed
24,960 held-out display rows and exports overlap/IoU/ARI with frozen definitions.
It is distinct from the six-static-snapshot color reference. See the
[metric definitions](metrics/checkpoint_cluster_matching.md). The renderer checks
identity and cluster consistency before publishing; a selected MD checkpoint
must have an exact matching receipt. No atom selection is linked between panels.


## Expanded viewer, lattice comparisons and travel

The viewer now uses two full-width square panels per row. Custom fixed-size
legend buttons retain visibility by panel, feature space and original cluster
ID, including frames with zero members. Marker-size sliders do not resize the
legend. Samples use larger, uniform saturated cluster colors; sparse connections
remain. The optional local-shell, ideal-lattice and displacement overlays use
central-atom PTM with explicit candidate quality and full-sample mismatch.

At the bottom, spatial-neighbor and embedding-neighbor paths show actual frozen
vectors, their coordinate profiles and full-vector changes. Both use one sampled
snapshot and retain explicit atom identities; these are not MD time paths.
See [definitions and limitations](metrics/interactive_structure_paths.md).
Producers are `sample_lattice --publication BUNDLE --workers 4` and
`embedding_travel infer|publish`; run computation on Slurm and finish with
`comparison_layout` for each dataset. Existing scientific arrays stay frozen.

### Responsive rendering

Visible sections render independently. Off-screen MD, examples, correspondence
and travel defer their asset loading and drawing until they approach the viewport;
frame and checkpoint changes leave only the latest pending state. Unchanged
panels are reused. Point-size controls restyle markers without rebuilding samples,
paths or correspondence. Lattice assets load only when an overlay is enabled.
Plotly WebGL now uses two rendering pixels per CSS pixel on each axis for
sharper rendering (four times the pixel count of the earlier responsive preset).
Lazy rendering, partial updates and bounded caches remain enabled; all requested
MD centers and original coordinate values remain present. Ideal-lattice overlays
show only the 80 assigned reference sites and their connecting edges, with faint
lines and smaller markers; fitted spacing and scientific measurements are unchanged.

Path search runs in a dedicated browser worker. Its cached undirected graphs and
binary-heap Dijkstra preserve full-vector Euclidean weights and deterministic
node-index tie ordering. The position slider updates its cursor and guide lines;
it does not rebuild the graph, coordinate heatmap or MD context. Browser caches
retain a bounded set of recent assets per type and protect the current selection.
This is separate from the scientific feature-cache retention policy.

`compact_view_assets --publication REPO_BUNDLE` creates additional display assets
on Slurm CPU. It strips unused historical radial-color tables from sample assets
and packs float32 embeddings losslessly. Packing refuses any value that cannot
round-trip exactly through float32; browser decoding restores ordinary JS number
arrays so distance arithmetic remains unchanged. Original assets remain preserved.
Follow with `comparison_layout` to activate the compact manifests.

Manual browser profiling at 1400×900 measured initial data-asset requests falling
from 13,997,928 to 2,994,183 bytes and initial plot draws from 11 to 2. Ten rapid
path-cursor inputs changed from 30 full draw requests to one cursor restyle and
two guide-line relayouts. These are data/routing measurements, **not GPU frame-rate
or wall-clock speedup claims**. The observed spatial and embedding paths and their
128-coordinate profiles matched exactly before/after. Real Plotly browser reviews
on both datasets preserved square panels, independent legends, hidden red clusters
across frames, sample overlays, and both path modes. Large-panel review rendering
used a reduced draw for software WebGL; original full trace counts were retained
for inspection. Evidence is in `technical/performance/` and
`technical/rendering/responsive-browser.json` of the published bundles.

Travel plots place headings and the profile legend outside the plotting area,
with separate space for axis labels and the coordinate heatmap. Path markers and
background atoms share original MD coordinates; periodic crossing segments are
not drawn across the box. The physical length retains minimum-image distances.
Embedding-neighbor paths can visit distant sites within a single snapshot.

## Current rich-MACE checkpoint in the interface viewer

The active command is:

```bash
python -m src.research.spatial_vicreg_bias.mace_checkpoint infer --config configs/analysis/mace_rich_current.json
python -m src.research.spatial_vicreg_bias.mace_checkpoint publish --config configs/analysis/mace_rich_current.json
```

The current selected state is the RH2 normalized residual-head checkpoint,
**epoch 14/update 1806**, using the recorded Al selection descriptor Gaussian NLL.
See [replacement and execution](#current-mace-replacement-rh2-best-checkpoint-2026-09-30).
The original publication used the older SiLU-head step-1088 checkpoint; that
analysis copy and its generated data are removed after the replacement review.
Its historical metric definition is retained in `docs/metrics/rich_mace_interface.md`.
The original scientific training run is preserved independently.

## General descriptor comparison (2026-09-30)

The active GeoFormer and rich-MACE viewers now use the all-training descriptor
fits, replacing the earlier interface-adjacent fits. This is a versioned analysis
change, not neural retraining. The recipe is
`configs/analysis/general_descriptors_20260930.json`; implementation is
`src.research.spatial_vicreg_bias.general_descriptors`.

The all-phase fitting population contains 74,880 uniform observations from 90
training sources. It excludes held-out and static observations. The existing
primary-seed-17 K7 models and train-fitted scaling are reused for TDA, bond order,
CNA and the joint vector. Assignments, descriptor PaCMAP, dense MD coloring,
examples, ideal-lattice overlays and color correspondence are regenerated.
Frozen neural embeddings, neural clusters and neural PaCMAP remain unchanged.

Active controls no longer select interface-only populations or physical regions.
The sole interface-specific control is **Highlight interface layers**, which
emphasizes finite distances <=12 Å in both PaCMAP panels with fully opaque,
40% larger markers and a thin black outline (0.6 px). Other markers keep their
normal size and opacity (85%). All points and correspondence statistics remain.
The point-size slider preserves this size ratio. The toggle does not redraw MD. Cluster
visibility remains attached to original IDs when this toggle or the frame changes.
Old interface-only interactive links redirect to the full comparison.

New assets, metrics and provenance are in
`output/spatial_vicreg_bias/general-descriptors-20260930/analyses/{matched,static}`.
Each active publication records `technical/general-descriptors.json`; layout
refresh reapplies it after loading saved sample manifests. This prevents a layout
refresh from restoring interface-trained descriptor labels. Historical fit
models, projections and frozen metric exports retain their original definitions.
Travel keeps its existing real sampled centers and neural vectors; its stored
descriptor labels are updated. Historical cluster-based supplemental sampling
is retained and explicitly recorded.

The dated general-descriptor recipe records the initial publication, including the now-removed step-1088 MACE component. Current MACE updates use `mace_rich_current.json` and reuse these shared descriptor assets. The following stages describe reproduction of that original revision.

Run the following stages on Slurm CPU using `pointnet-torch214`, with
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` and two BLAS/OpenMP/Numba threads. For each
`DATASET` (`matched` and `static`):

```bash
python -m src.research.spatial_vicreg_bias.general_descriptors --config configs/analysis/general_descriptors_20260930.json --dataset DATASET --stage build
python -m src.research.spatial_vicreg_bias.general_descriptors --config configs/analysis/general_descriptors_20260930.json --dataset DATASET --stage publish
```

The publisher requires explicit JSON asset sidecars. Legacy assets were migrated
once, without changing their JS or numerical data; the operational receipt is
`output/spatial_vicreg_bias/viewer-cleanup-20260930/technical/migrated-assets.json`.
[Metric definitions](metrics/general_descriptor_comparison.md) specify the two
color-matching reference populations and the browser's displayed-row diagnostics.


## Current MACE replacement: RH2 best checkpoint (2026-09-30)

Interactive pages are directly in the experiment folder:
[Held-out Al MD](../output/spatial_vicreg_bias/mace-rh2-best-20260930/heldout-al.html) ·
[Six static Al snapshots](../output/spatial_vicreg_bias/mace-rh2-best-20260930/al-static.html).

The active MACE comparison now uses the normalized residual-head RH2 run's
**best validation checkpoint**, epoch 14/update 1806, selection descriptor Gaussian
NLL **1.0348901381502593**. The latest saved training cursor was epoch 42/update
5486 at the time of selection; it is not the selected checkpoint. Selection
uses the existing Al validation descriptor likelihood, never AP or viewer
correspondence. The immutable copy and selector record are in
`output/spatial_vicreg_bias/mace-rh2-best-20260930/technical/`.

The current recipe is `configs/analysis/mace_rich_current.json`; frozen inference
runs within the exact source tree recorded by the checkpoint, with the current
analysis entry point overlaid. This matters because the live repository has
refactored its typed trunk since that checkpoint was written. No training or
new descriptor calculation is launched. Current GPU inference uses this node's
Slurm allocation; publication and browser review use Slurm CPU.

Both analysis datasets are under
`output/spatial_vicreg_bias/mace-rh2-best-20260930/analyses/{matched,static}`.
They retain all-training descriptor assignments/projections, exact observed atom
identities, all 13 held-out MD frames and all six relaxed static frames. New neural
K7 centroids fit the original 74,880 uniform training observations only. Neural
PaCMAP, dense labels, color correspondence, samples, PTM overlays and travel
vectors are produced for this checkpoint. The only interface control remains
the opacity highlight. [Metric contract](metrics/rich_mace_comparison.md).

The former step-1088 MACE analysis directory and its copied checkpoint are removed
after the replacement browser review succeeds, as explicitly requested. Its
training-run source artifacts are outside this analysis replacement. Deletion
provenance is recorded in the new analysis's `technical/previous-analysis-removal.json`.
