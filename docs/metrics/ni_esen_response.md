# Ni unrelaxed eSEN simulator response and prediction

The completed 12-epoch unrelaxed Small eSEN is the common geometry-only start.
Its fixed Ni-to-Al-reference length factor multiplies coordinates, velocities,
and physical response directions before encoding. No species, material, age,
temperature or explicit time covariates enter a neural layer. Thermostat
temperature is used only by the physical simulator. No broad spatial replay.

The autonomous latent state is (s,m), with s=E(Q) and m approximating
0.1 ps DE(Q)[V] by symmetric float32 neural secants with epsilon=0.01.
Physical teacher endpoints use 100 steps of 1 fs fixed-box full-cell BAOAB,
Ni Mendelev 2012 EAM/FS, mass58.71, friction1000fs, two independent noise shots.
All states, momenta, tangents, noise and stored physical observables are float64;
force arithmetic is float32 and scalar energy reduction float64. Only the
selected local80 initial positions or velocities are perturbed: RMS0.01A or
0.1A/ps. The rest of the periodic32000-atom cell evolves. Endpoint geometry
values are averaged over shots; their derivatives use exact eSEN input JVPs
of the simulator's analytic propagated tangents. Neural response secants use
epsilon0.5 and identical stochastic latent paths on both signs. This does not
claim differentiation of arbitrary MD source files or native NPT restarts.

Four matched arms: joint encoder or frozen encoder plus residual adapter,
each value-only or value-and-response. Effective batch256, microbatch32
(explicit memory deviation), 1600 AdamW updates, lr1e-4 with encoder multiplier
0.1, weight decay1e-5 and gradient clipping10. The objective is standardized
24-descriptor mixture NLL plus0.2 frozen future embedding half-MSE, plus0.1
short endpoint half-MSE, and for derivative arms0.1 relative response loss.
The response loss is sum over two directions of0.25 times mean squared error
divided by teacher mean square (floor1e-6). Neural secants are float32;
long latent training uses BF16 autocast. Physical response teachers remain exact
analytic tangents, independent of the secant approximation used by the student.

Saved NPT targets at0.1,0.5,3,12ps use exact native frames; no interpolation.
Five outcome-blind times per source span the isothermal hold after20ps with a
complete12ps future;64 random fixed atom centers per source. This gives19200
rows across60 trajectories, not19200 independent trajectories. Variable saved
trajectory endpoints affect phase occupancy and are disclosed. Four original
training ancestors provide label-free NLL checkpoint selection; all40 original
selection/test ancestors are pooled for evaluation. Standardization uses only
inner training rows. The spatial checkpoint ancestry is checked using metadata,
without loading spatial training arrays.

Evaluation has distinct populations and time meanings:

- Appearance: no FCC/HCP/BCC atom presently within the consumed nearest80/radius8
  neighborhood. Follow those exact present atom identities and report any first
  observed crystalline label within0.1,0.5,1,3,6ps. This is first observed after
  the input, not necessarily first ever, and transient labels count. No external
  neighborhood screening. Missing historical atomic PTM arrays are recomputed
  with the same full-cell OVITO PTM RMSD cutoff0.1; existing atomic arrays reused.
- Nucleus32 and nucleus64: catalogue liquid-origin root events with the indicated
  size sustained1.5ps. Input lead times1,3,6ps are measured before the START of
  the first qualifying persistent interval, not its confirmation1.5ps later,
  nor the catalogue's first8-atom embryo onset. The latter remains metadata.
  Four founder centers per event and four matched random controls per founder,
  excluding the selected event-founder IDs from control sampling. Controls are
  not independently verified never-crystallizing liquid. No neighborhood,
  component or distance filtering. Exact saved Q,V/offset availability only.
  Duplicate physical rows within a task/lead are removed, and event weights
  sum to one. This is an event-matched localization assay, not an unconditional
  nucleation-rate estimate. The default nucleus threshold is64 atoms/1.5ps.

Frozen structural128, motion128 and joint256 readouts compare the spatial start
and each response fit. Linear regularization and CatBoost iterations are chosen
by NLL within original training ancestors, then refit on all original training
ancestors. AP never selects training or readouts. A constant training prior is
reported. AP, AUROC, binary NLL and Brier use declared row/event weights. Confidence
intervals are2.5/97.5 percentiles of1000 whole-ancestor bootstrap replicates; class-
degenerate replicates are omitted. Cohorts lacking both classes in fit, selection
or evaluation are explicitly unavailable, not assigned zero. APs between tasks
have different prevalence and are not directly comparable. Scientific training
is online W&B; probes remain local and update the existing run by API only.

Future descriptor NLL is a four-particle diagonal-Gaussian mixture during fitting
and eight particles at evaluation, summed over24 coordinates then divided by24.
Descriptor MSE is in training-standardized units; structural MSE is in the fixed
teacher embedding. Coverage90 is the fraction of coordinate-wise mixture PITs
between0.05 and0.95. Held-out future means are row-weighted over outcome-blind
observations, with uncertainty driven by sources rather than row independence.
MD labels derive from quantized archived coordinates; short teacher observables
are retained before quantization. A correct short response is not evidence of
rare-event or long-horizon NPT distribution equivalence.
