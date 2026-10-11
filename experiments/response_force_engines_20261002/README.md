# Can cheaper force-only teachers accelerate local response supervision?

Compare UNEP-v1 and the parent Lee2003 MEAM potential against the existing MACE
20/100fs local-response teacher. Separate three questions: complete response
cost; numerical stability of common-noise finite differences and finite halos;
and changes in the signed future-response field caused by changing potential.

Four existing fixed-Al64 training-source queries × two noise seeds ×18/24/36Å
moving environments are shared across teachers. Each input perturbation spans
the same nearest80 atoms and two unit-norm, zero-COM directions. A complete
response uses five ordinary force-only paths, retaining the exact BAOAB mass,
timestep, thermostat and observable definitions. CPU arithmetic is explicitly
float64; the frozen MACE GPU reference uses float32. No temperature, time,
history, motion, species or material input is added to any encoder/predictor.
No encoder is fitted at this engineering stage.

Evaluate signed256×2 responses at epsilon0.03/0.01/0.003Å; report convergence,
prefix-wise differences/cosines against MACE, complete per-shot costs and
environment convergence. Smaller halos are eligible only after their own
all-parent, both-prefix gate. Precision, force/energy and vacuum-box checks
precede collection. Timing ratios explicitly distinguish one CPU thread from
the original GPU. This is not a held-out crystallization prediction result.

Implementation and reproduction: [workflow](../../docs/simulations/response_force_engines_20261002.md).
Recipe: [configuration](../../configs/simulation/response_force_engines_20261002.json).
Output: [comparison](../../output/response_atlas/force-engines-20261002/).

Scientific interpretation: MEAM responses correspond to the original generating
potential, while UNEP and MACE define different surrogate futures. Agreement
with MACE is a diagnostic, not a requirement that MEAM reproduce another
potential. No alternative silently replaces the running MACE experiment.
Only an independently validated distilled potential could support a claim of
MACE-like physics at lower cost; ACE fitting is outside this first comparison.

Initial matched-parent findings (2 October,17:12 Paris): at24Å/3,245 simulator
atoms, cached MACE GPU AD cost14.37s per response; serial MEAM averaged9.17s,
and three persistent MEAM CPU workers averaged4.56s over two seeds and three
hardware repeats. Serial UNEP averaged49.63s and its three-worker version25.77s.
Both parallel engines reproduced all saved serial values/responses exactly in
this check. These timings exclude process/potential startup; CPU resources and
the simulated potential differ from MACE. The complete four-parent halo and
epsilon campaign remains running; no generalization result or completed
alternative-encoder training is claimed. The result page updates after both
detached workers complete.
