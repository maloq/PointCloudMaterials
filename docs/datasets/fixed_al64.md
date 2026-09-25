# Fixed Al64 comparison datasets

The default data contract for **new matched Al encoder comparisons** is
[`al64_v1.json`](../../configs/fixed_cohort/al64_v1.json). The release lives at
`${storage:cache}/fixed-cohort/al64-v1-20260925`. Running and historical experiments
retain their original data and input records.

Release status: **complete**, identity
`e148b7ec215ba5e6d86fc57d21dac266bbd501f1e91320968266b5dbaeb8f44d`.
The exact [source roles and center IDs](../../configs/fixed_cohort/al64_v1.splits.json)
are also retained in the repository, independently of the cache.

## Fixed independent sources

| Role | Trajectories | Tracked centers | At-risk windows | Positive by 3 / 6 ps | Permitted use |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 90 | 5,760 | 43,523 | 314 / 694 | Fit encoder, predictor, probes and normalizers |
| Selection | 15 | 960 | 20,883 | 141 / 298 | Validation and checkpoint selection using the declared objective |
| Calibration | 15 | 960 | 16,848 | 147 / 347 | Fit calibration or operating thresholds after selection |
| Test | 30 | 1,920 | 45,291 | 267 / 592 | Final matched reporting |

Total: **126,545 prediction windows**. Positive-window counts use onset delay
less than or equal to the stated horizon; they are not independent event counts.

Source roles and independent melt ancestries are inherited from the existing
assay. Every source uses the **same 64 tracked atom identities across time**, from
its previously sampled outcome-independent pool. The historical 16 centers are a
subset. Sources, frames and atoms are never re-split for a model or training seed.
Windows and centers within a source are correlated; resample whole sources for
uncertainty. These are **historical test trajectories, not a new untouched test**.
More centers improve coverage without creating additional independent runs.
Shooting branches, duplicate exports and other descendants are not included.

## Crystallization benchmark: `benchmark/`

Original MD defines crystallization: PTM FCC/HCP/BCC for three consecutive saved
frames, onset at the first of those frames. At-risk origins precede the first
sustained onset and have no crystalline label in the current or preceding two
frames. Every origin has follow-up through 12 ps plus confirmation. Saved cadence
is 0.75 ps; event bins end at 0.75, 3, 6, 9, 12 ps, plus no onset by 12 ps.

Training origins are inherited frames 64, 80, 128, 176, …, 656; other roles use
frames 64, 80, 96, …, 656. Different sampling densities are retained explicitly from
the earlier protocol. All new models use identical role-specific rows. A model
cannot omit difficult rows or choose its own relaxed-frame availability.

Patches contain the center and 79 observed nearest neighbors. Center-relative,
minimum-image coordinates are float32 **Angstroms**, decoded from verified
float16 full-cell histories. A radius 8 Å mask is returned separately; models must
respect it. This is nearest 80 support cropped at 8 Å, **not a complete radius 8 ball
or message-passing halo**. Cutoff and message-passing depth remain model choices.

Relaxed coordinates retain the observed atom identities. Existing fixed-box
full-cell FIRE quenches use force tolerance 0.01 eV/Å; source, frame, potential and
binary hashes are verified. Relaxation is an explicit input or training-only
teacher; MD still defines the outcome. Current relative MD velocities are an
optional observed-input view. Full source/frame references remain available for
declared causal history and spatial context. Context embeddings are not cached
in this model-independent release.

`population.npz` stores sample IDs, source, role, frame, center, patch index, event
bin and delay. IDs include the full source-manifest hash. `legacy_order.npy`
selects the exact earlier 31,609 observations in their original order; labels and
roles are checked exactly against that population. The **legacy16 track** supports
historical comparisons. Report it separately from the new **all64 track**.
The sample/label match does not assert bitwise equality with every historical
coordinate cache: this release consistently re-extracts the stored full cells.

## Structural pretraining: `structural/`

| Role | Sources | Frames per source | Centers | Raw neighborhoods |
| --- | ---: | ---: | ---: | ---: |
| Train | 90 | 201 | 64 | 1,157,760 |
| Selection | 15 | 201 | 64 | 192,960 |

Frames 0, 4, …, 800 span 0–600 ps at 3 ps sampling. Both liquid and crystalline states are
included without PTM filtering, event sampling or crystallization labels. Inputs
are current geometry and Al species. The loader provides no temperature, age,
absolute time, explicit time covariates, velocity or future targets. Any fitted
normalization must use train sources only; coordinates remain in physical units.

A paired view uses all existing candidate quench frames, including training
frame 32, **without an onset risk mask**: 86,400 observed/relaxed training pairs and
36,480 validation pairs. Neither pretraining view can request calibration/test
roles. Selection is validation only, not gradient updates or normalization.

Only native Al with the assay's Lee2003 second-nearest-neighbor MEAM potential is included.
Static/mixed-material releases retain their own protocols. This release contains
geometry rather than precomputed TDA/physical targets. Objectives needing those
targets must add a versioned, row-matched target cache.

## Build, verify and use

Run in `pointnet-torch214` from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OVITO_THREAD_COUNT=1 \
  python -m src.data.fixed_cohort.prepare \
  --config configs/fixed_cohort/al64_v1.json --workers 4
```

This is CPU data preparation: no new simulations, fitting or W&B run. It resumes
verified source shards. A lock excludes concurrent writers. Completion is
published only after every source and historical-population check passes.
Configuration or producer changes require a new release.

```python
from src.data.fixed_cohort.dataset import (
    OnsetDataset, StructuralDataset, verify_release,
)
root = '${storage:cache}/fixed-cohort/al64-v1-20260925'
verify_release(root)  # Offline contract and observation checksum audit.
train = OnsetDataset(root, 'train', domain='hot')
test = OnsetDataset(root, 'test', domain='hot')
pretrain = StructuralDataset(root, 'train')
validation = StructuralDataset(root, 'selection')
paired = StructuralDataset(root, 'train', paired=True)
legacy_test = OnsetDataset(root, 'test', track='legacy16')

item = train[0]
encoder_inputs = item['inputs']  # Positions, species, radius mask only.
event_target = item['event']     # Kept outside encoder inputs.
test.validate_predictions(exported_sample_ids)  # Exact ordered full coverage.
```

The loaders work with PyTorch DataLoader and memory-map source arrays. They return
fixed 80-candidate tensors and masks; graph construction should compact valid nodes
and apply its declared physical cutoff. `paired=True` returns `inputs` and
`teacher`, never events. Scientific training keeps online W&B; preparation does not.

## Comparison contract

Every run/export must record release identity, all64/legacy16 track, input view,
encoder and predictor context separately, objective, selection rule, fitted data
roles, sample IDs and source-weighting convention. Exact counts and identity are
in `manifest.json`; source roles and centers are in `splits.json`. Source and array
SHA-256 receipts support offline integrity verification.

Supervised models/probes use predictive likelihood objectives/selectors.
Self-supervised encoders use label-free objectives/selectors; supervised probes
are a separate fit. AP at 3/6 ps remain diagnostic, alongside NLL, Brier/calibration,
information controls, trajectory stability and normalized noise response.
Fixed samples do not equate different supports, budgets or readout capacities.

For expansion beyond the current pretraining sample, see
[existing structural-training capacity](structural_training_capacity.md). Its
counts distinguish already stored data, extractable neighborhoods and completed
relaxed cells while retaining the fixed evaluation roles.
