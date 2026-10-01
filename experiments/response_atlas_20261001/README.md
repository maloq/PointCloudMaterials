# Do simulator responses improve conditional-future representations?

Next stage: [matched atomistic value/response training](ATOMISTIC_TRAINING.md).
The historical feasibility release below remains distinct from the new nine-fit
full-cell student experiment.

This first release implements the three feasibility experiments preceding a
five-arm active-learning comparison:

1. Existing Al480 independent-shot precision and cross-fit future retrieval,
   with prior and descriptor-ridge conditional-mean controls.
2. Gaussian mean/variance and cancellation mechanisms, a slow double-well
   diagnostic, and six matched value/response toy likelihood fits.
3. Fixed-MACE full-cell Al256 numerical response gate and development-only
   horizon/noise/cost pilot on 16 controlled perturbed-FCC configurations.

The source is the user-supplied `response_atlas.py`, preserved verbatim as
`src/research/response_atlas/reference.py` (SHA256
`cf7dff85b5ef92953a92142ed4f8887bd3d6d20314a0497c8623d4d5468f25a6`).
Its historical test claims are not imported as evidence; actual reference and
physical consumers are verified through local numerical gates. No test suite
or new automated tests are introduced.

The hypothesis for the eventual comparison is improved independent-parent
future-law prediction per total simulation/training cost. This release does
not yet test that full hypothesis. Toy paired fits isolate supervision under
matched data, and the physical stage measures feasibility without training an
atomistic student. Responses under MACE never label the MEAM future law.

No temperature, age or absolute-time inputs enter learned predictors. No
physical-reconstruction encoder pretraining is introduced. AP is not an
objective or selector. The historical Al benchmark retains its existing roles;
the FCC pilot is a separate synthetic-parent collection, not an Al64 resplit.

Next-stage gates: corrected acquisition must distinguish known errors from
branch noise; physical AD and coupled FD must agree; response variance and cost
must support a useful horizon. Only then specify the five-arm random/active,
value/response comparison and later original-MEAM bridge. Neither later stage
is silently launched by this release.

[Recipe](../../configs/response_atlas/feasibility_20261001.json) ·
[Metrics](../../docs/metrics/response_atlas.md) ·
[Operations](../../docs/response_atlas.md).
