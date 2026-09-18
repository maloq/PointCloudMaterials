# Archived research

The [restored FactorVAE analysis galleries](../output/factor_vae_archive/index.html)
contain four historical static-Al analysis sets for three FactorVAE-trained
GeoFrame models, including a later reanalysis of the epoch-034 checkpoint.
Each restored run retains its original analysis files, saved training config,
cache metadata and a SHA-256 copy manifest. These are copies of archived results;
the archived originals and historical metric values remain unchanged.

Older experiment records, source code and results are in the
[STORE repository copy](/store/PERSO/vmorozov/projects/PointCloudMaterials-20260913T174741Z/).
Browse its `experiments/`, `output/` and `outputs/`; `.git/` preserves repository history.
External dataset symlinks are references, not duplicate backups of all simulation data.

The full copy encountered concurrent writes, so it is not an atomic snapshot of
active runs. **Every output selected for this cleanup has since been verified
separately against STORE**, including exact file hashes and symlink targets:
[verification receipt](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/retirement-verification.json).
No changed or unverified item is eligible for local removal.

The [September 13 supplement](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/)
contains a verified copy of all current experiment records, including uncommitted
short-history notes, a Git bundle and the pre-cleanup working-tree patch.
[Experiment retirement receipt](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/experiment-retirement.json)
and [output retirement receipt](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/output-retirement.json)
record what was removed locally. Copy an archived run to a new working directory
before resuming or changing it; retain original provenance.

[The September 11–13 retention review](research_retention.md) identifies the current
research, checkpoints and older dependencies kept in the working checkout.

The configuration cleanup also has a complete, verified
[pre-cleanup `configs/` copy](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/configs/)
and [file-hash receipt](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/configs.verification.json).
Restore old Hydra recipes with their defaults and loader dependencies together;
see the [current configuration index](../configs/README.md).

Archived WandB log links that formerly pointed into the live checkout now point
to their retained archive files; [the link audit](/store/PERSO/vmorozov/projects/PointCloudMaterials-retention-20260913/archive-log-links.json)
preserves original targets. No trajectory, checkpoint or frozen source bytes changed.

The [September 16 frozen-map retirement](discarded_frozen_encoder_maps.md) removes
replacement-embedding training over frozen encoders while preserving all results,
exact source/recipes, native encoder training and embedding forecasting.
