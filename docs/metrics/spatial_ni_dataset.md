# Ni extension of the spatial fitting set

The parent is the fixed 1,056,768-row Al/Mg/Ti/Ta packed spatial collection.
All parent fitting rows are retained, in their existing order, with their source,
ancestry, role and original-pool mapping. Raw historical trajectories are not
opened. The 198 named bond-order/CNA/TDA values are copied from existing targets.
All parent selection/calibration/test observations remain held out and retain
their original role-specific row IDs.

Ni sources are verified completed production trajectories from the three declared
campaigns, frozen at planning time. Original train/selection/test roles and melt
ancestry are retained; incomplete sources are explicitly excluded. No PTM, birth,
phase, temperature or outcome filter chooses observations. Availability at freeze
and each simulation's original stopping rule delimit the population.

Each Ni source has deterministic tracked centers sampled from 1% of atoms (320
for a 32,000-atom cell). Frames are spaced at least 3 ps apart, using their actual
native 0.10-ps timestamps. Training rows are a uniform draw without replacement
from the resulting training-only frame/center candidates. Source weights therefore
follow available durations, not equal source or temperature weights. Timestamps,
temperature, species, scale, velocities and outcomes are never model inputs.

`Ni_rows = round(parent_training_rows * fraction / (1 - fraction))`; the specified
fraction is Ni's share of the **expanded** fitting set. At 0.20 this gives 264,192
Ni rows and 1,320,960 combined training rows. Separate uniform samples of 4,096
selection and 4,096 test Ni rows use only their original held-out ancestors.

One constant Ni length scale follows the established material normalization:
the 0.995 quantile of nearest-160 distances from 4,000 points across at most eight
training sources, multiplied by 1.02. Calibration uses saved frame 2. Ni positions
are multiplied by the unchanged Al reference scale divided by the Ni scale.
All previous material scales remain fixed. Nearest-80 periodic neighborhoods
include the center; descriptor support is radius <8 in normalized coordinates.

The objective has 40 bond-order, 45 CNA and 113 TDA columns from the pinned target
contract. It does not select targets using held-out errors. All old target values
are retained; Ni targets use the audited safe native producer. Standardization
is refitted using **combined training rows only**: population mean and variance,
scale floored at 1e-4, active mask from SD >=1e-4. Spatial objectives weight the
three families equally and split TDA equally over its 18 groups, then active
coordinates. Held-out targets receive the training transform.

`population.csv` reports physical observation counts by original role and material.
Rows sharing source/ancestor or tracked atom are not independent replicates.
Checksums bind inputs, row maps, sampling, scale evidence, code and outputs.
This workflow prepares data only: no encoder fitting or online W&B run.
