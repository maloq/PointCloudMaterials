# Ta nucleus event crops

This is a geometry export for visual review, not a new prediction dataset or
event classifier. It reads the completed Ta audit and verifies consumed graph,
event and PTM records against their recorded hashes. Raw trajectory schemas and
manifests are verified by the existing source loader; the earlier audit's full
array checksums are retained as provenance. Original inputs are never modified.

The declared population is the eight shooting establishments whose connection
to frame0 requires weak graph links, plus all28 shooting episodes with verified
dissolution. The existing labels remain unchanged. The narrower supported
`strict_isolated` selection retains only the three strong-lineage isolated
dissolutions when explicitly configured. Historical trajectories are excluded.

An appearance is anchored at the first observation in its confirmed≥64-atom
establishment streak; this is not the first crystalline atom. A disappearance
is anchored at the first0.5ps audit observation after the last tracked cluster:
the original verification confirms all terminal-core atoms noncrystalline at
that observation and the subsequent follow-up observations. Its time bracket
and later confirmation timestamp remain in metadata. The0.1ps export does not
refine or relabel these0.5ps event times.

The requested window is [event−18ps,event+18ps], intersected with the original
0–24ps trajectory. Keep every native0.1ps saved observation in that intersection.
No extrapolation, padding, interpolation, sibling-branch stitching or new MD.
`available_before_ps` and `available_after_ps` measure the actual available
durations from the anchor. Missing durations are18 minus the corresponding
available duration, floored at zero. `full_window` requires both sides≥18ps.
Every crop is incomplete because the requested36ps exceeds the source24ps.

At the recorded centroid reference frame, choose a tracked anchor atom: use the
existing dissolution anchor ID, or the nearest periodic atom to the appearance
centroid. Preserve the minimum-image fractional displacement from that atom to
the centroid. At each frame the center follows that atom plus this offset
scaled by the current box. Thus the crop matches the recorded centroid at its
reference frame and follows local motion and box deformation afterwards.
This is a specified local reference, not a re-estimated nucleus center of mass.

Keep every atom at minimum-image distance≤20 Angstrom from the moving center;
membership can change between frames. Original atom IDs and native integer
timesteps are exact. NPZ coordinates are float32 centered displacements derived
from the existing float16 observations; promoting precision cannot recover
quantized information. Frame boundaries use the `offsets` array. Preserve
original per-frame boxes and absolute crop centers. This derived local export
does not replace the original periodic trajectory.

The extended XYZ movie translates local coordinates by+20 Angstrom into a40
Angstrom box with all periodic flags false. Artificial periodic interactions
must not be introduced at crop boundaries. Properties include original atom ID,
tracked-center marker and `ptm_type`. PTM types are copied from full-cell audit
observations every0.5ps, with −1 meaning unavailable at intermediate0.1ps frames.
No temporal interpolation of labels; zero retains its original unclassified
meaning. A viewer can recompute local PTM, but boundary atoms lack full context.

Production validation checks radius, unique IDs, finite coordinates and presence
of the tracked atom. Every exported XYZ frame is independently read by ASE and
compared with the NPZ geometry, IDs, labels and nonperiodic cell flags. Checksums
bind completed files. The crop index reports population, labels, radius, frame
count, physical coverage and file locations; it does not report a predictive
score, rate or independent nucleus count. Future-selected centers and these
labels must not enter causal predictor inputs.
