# Dense deterministic Ni response observations

Collection: `ni-phase-nve-20261013`, registered in `configs/datasets.json`.
Generating potential: unchanged Mendelev2012 Ni EAM/FS,
SHA256 `acc8bd87f61dd67ecd8f72a6f181d6aafe52ca45b29fe6c15cec7c39c0bf76df`.
Backend: [validated FastNi](ni_eam_gpu_20261007.md).

The coordinate/velocity learning campaign starts630 full-cell continuations
from the pinned diverse320/pooled310 anchor selection:320 train states from28
ancestors,310 test states from62 different ancestors. Actual saved Q,V and
the instantaneous periodic box initialize deterministic fixed-box NVE, with
1fs velocity Verlet through100fs. No thermostat noise, barostat, relaxation,
resampling of velocities or exact historical NPT replay is claimed.

Each path exports the same initial80 atom IDs, center-relative Q,V, relative
acceleration and two propagated Q/V perturbation tangents at0,5,...,100fs.
Initial directions perturb only those80 atoms within the full32k-atom cell;
initial RMS is0.01Å for Q and0.1Å/ps for V before center subtraction.
States, integration, tangent propagation and exported observables are float64;
force arithmetic is float32. Per-path full-cell energy changes are audit fields.

SCRATCH collection is checksummed into STORE before completion is declared.
Local response observables retain prequantization precision; no full trajectory
archive is written. Shared source readers verify manifests and identities.
The dataset packets and scientific queue are bound to the same frozen plan.

The two receipt-owned workers1036592(node29/A40) and1036593(node32/A40) collect
these observations and train the fixed experiment queue within the shared
12-hour deadline. [Training workflow and active receipts](../ni_phase_sim.md).
