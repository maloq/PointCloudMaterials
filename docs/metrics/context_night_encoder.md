# Overnight local-encoder information preservation

Existing `neighborhood_jepa_regularization.md` defines physical85, TDA144, fixed
equivariant moments, future/neighbor JEPA, nonlinear order8 and SIGReg losses.
These calculations remain unchanged. The snapshot encoder still exports128
invariants and120 equivariant channels; extra readout heads are training-only.

`linear_order`: training-normalized order8 decoded by a single Linear(128,8).
`linear_angular`: training-normalized angular-bin components64:80 of Physical85,
decoded by Linear(128,16). Training terms are component/sample means multiplied
by the declared spec weights. Validation reports their unweighted source-equal
MSE. All four arms instantiate the same heads; zero weights disable their training
contribution. Normalization is inherited from training-only original releases.

Common checkpoint selection for all arms: source-equal development Physical85
block MSE +0.25 TDA144 block MSE +0.25 nonlinear order8 MSE. Linear-head errors,
crystallization outcomes and all test metrics do not select checkpoints or the
longer continuation. Separate frozen linear/MLP readouts use the unchanged
`neighborhood_crystallization_v2.md` and `crystallization_information.md` metrics.
