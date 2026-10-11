# Ta appearance candidate temporal cutouts

This export selects the eight ambiguous appearance candidates from the completed
20 Angstrom Ta crop collection. It verifies the original plan digest and each
source crop's metadata, XYZ and NPZ hashes. No raw MD, event classification,
spatial membership or centering is recomputed. Original exports remain unchanged.

The requested time interval is12ps before to12ps after the start of each
confirmed64-atom establishment streak. This anchor is the existing audit event,
not a new determination of first crystalline-atom appearance. Only observations
inside the requested interval and the available source crop are kept. There is
no padding, interpolation, branch stitching or simulated extension. Native
observations remain0.1ps apart. The eight source events are weak-link-dependent
parent-connected establishments; they are not relabeled independent nuclei.

The20 Angstrom moving sphere, original atom identities, variable atom membership,
float32 centered coordinates, original full-cell boxes, absolute centers and
integer source timesteps are copied exactly. XYZ frames are copied as text blocks.
The NPZ flattened atom arrays are sliced using the original frame offsets, then
offsets are rebased to zero. Full-cell PTM labels remain available only at0.5ps
audit frames; intermediate values remain−1. No relaxed coordinates are used.

`available_before_ps` is minus the first retained relative time and
`available_after_ps` the last retained relative time. Missing time is the
requested duration minus the available duration, floored at zero.
`full_window` requires both available sides to reach12ps, tolerance1e-8ps.
The24ps source branches cannot supply both sides for these eight anchor times;
none has a full window. The index retains exact coverage, labels and file paths.

Every derived XYZ frame is independently read through ASE and compared with
the corresponding NPZ coordinates, IDs, partial PTM labels and nonperiodic flags
using the existing production validator. Completion receipts bind all files.
A ZIP package contains the eight separate event folders, readme and coverage
tables. This is a visual-review export, not a causal training dataset; future
event centering and fate labels are not predictor inputs.
