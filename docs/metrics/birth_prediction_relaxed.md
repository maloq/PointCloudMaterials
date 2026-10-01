# Paired full-cell relaxed birth inputs (v1)

This study retains the original 1,475 histories, 295 cases, 1,180 controls,
source roles, event weights and five frozen readout folds. Every original
sampled `(source, frame)` full periodic cell is quenched independently with
the original Lee2003 Al MEAM files, fixed simulation box, FIRE timestep
0.001 ps, and infinity-norm force convergence at 0.01 eV/A. Minima are static
inherent configurations; minimizer steps are not future MD time.

Use exactly the original nearest-80 atom IDs at every observed frame, centered
on the original tracked atom's relaxed position with minimum-image handling.
Save local float32 offsets from the full-precision converged coordinates before
verified global float16 archival conversion. The full cell is computational
context for minimization; encoders and descriptors receive only the resulting
80-atom patch. Original MD eligibility and outcomes are unchanged. Do not
exclude or relabel a patch when minimization changes its apparent ordering.
No future frames, temperature, age, absolute time, material IDs or velocities
enter the model inputs. Actual timelines and source manifests are verified by
the original trajectory producer. Failed cells remain archived and block fitting;
no model-specific or outcome-specific exclusions are permitted.

Rich descriptors and both original frozen MACE checkpoints are exported again
on quenched coordinates, with the same inference source and preprocessing.
Their weights are not retrained. This is an input-domain transfer experiment,
including possible distribution shift for the original encoders.

Repeat the six readout arms and thirteen observation treatments defined in
[birth_prediction_temporal.md](birth_prediction_temporal.md). Linear regularization
and CatBoost stopping are selected by selection-source binary NLL. Separate
calibration sources fit the monotone probability map. One fit seed is retained.
Fixed test and readout CV are separate. CV sources were exposed to frozen encoder
pretraining. Held-out sources have already been inspected in earlier studies;
this is exploratory follow-up. AP is diagnostic and never a fitting selector.

`train-test-errors.csv` reads the existing verified per-fit score receipts for
both domains. Train means resubstitution on original fitting rows, not held-out
skill. Test is the fixed 200 histories from 15 births in 11 observed sources.
Report raw and calibrated event-weighted NLL, Brier, AP and AUROC separately.
Generalization gap is test minus train NLL or Brier. Calibration is fitted on
separate calibration rows, so calibrated train error is not a training objective.
For error formulas and probability clipping, see
[birth_prediction_extension.md](birth_prediction_extension.md).

`paired-domain-differences.csv` compares predictions on identical fixed-test
or pooled out-of-fold rows. For each readout, treatment and score type, report
relaxed minus unrelaxed NLL, Brier, AP, AUROC and matched AUC. Matched AUC uses
the same within-set comparisons defined in `birth_prediction_temporal.md`.
95% intervals are percentiles of 2,000 shared whole-source draws, conditional
on fitted readouts and the frozen encoders. Negative NLL/Brier differences
favor relaxation; positive AP/AUROC/matched-AUC differences favor it. This does
not quantify fitting-seed uncertainty or blind prospective nucleation incidence.
No multiple-comparison correction or selection on these test results is claimed.

Historical metric tables and their frozen contracts are read without alteration.
New tables include `METRICS.md` and hashes of their actual implementation.
