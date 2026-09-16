# Using the MACE context pilot

The [scientific protocol](../experiments/mace_context_20260914/README.md) tests
complete message context and two raw 256-dimensional readouts of the forecast
MACE checkpoint. It uses a separate graph API in
`src/models/encoders/mace_context.py`. Original 80-point encoder checkpoints and
forecast caches retain their existing API and meaning.

Input clouds are variable-length arrays of physical Angstrom offsets. Index zero
must identify the tracked center. The candidate sphere includes every atom out
to 18 Angstrom, with periodic minimum-image coordinates from the actual source.
`make_context_graph` retains the exact two-hop ancestors of the selected readout
atoms. It does not infer missing surrounding atoms from an 80-point patch.

`halo_inner` uses a unit-weight core through 5 Angstrom and a nonnegative quintic
taper to zero at 7 Angstrom. `halo_center` returns both scalar blocks at the
tracked center. Both concatenate the same two 128-channel blocks as the original
encoder. `halo_mean80` is a diagnostic control whose first 80 input atoms must
be the selected nearest 80. The preparation module preserves the producer's
exact nearest-neighbor tie ordering for that comparison.

Training artifacts are explicit `mace_context_vicreg_warm_start_v1` payloads,
containing the mode, recipe, original checkpoint hash, full model and optimizer
states, epoch history and random states. They are not ordinary Lightning
checkpoints. Restore their weights into the original model with a strict state
load, then call the context `encode` function with the saved mode:

```python
from pathlib import Path
import torch
from src.project_runtime.paths import load_json
from src.research.mace_context.engine import load_model, encode

config = load_json("configs/analysis/mace_context.json")
config["device"] = "cuda:0"
mode = "halo_inner"
payload = torch.load(
    Path(config["output"]) / "technical" / f"train-{mode}" / "best.pt",
    map_location="cpu", weights_only=False,
)
assert payload["protocol"] == "mace_context_vicreg_warm_start_v1"
assert payload["mode"] == mode
model, _ = load_model(config)
model.load_state_dict(payload["model_state"], strict=True)
features = encode(config, model.eval(), clouds, mode)
```

`clouds` must come from the complete-context producer. Calling the original
`model.encoder(points)` still invokes the original 80-point mean API and would
not implement the saved context readout. Forecasting with a selected new readout
requires a separately identified embedding cache and a newly fitted forecaster.

The node57 pilot uses existing allocation 992489, its assigned GPUs 0 and 1,
and the [allocation recipes](../configs/mace-context/). Launch logs, status files,
source snapshots and explicit attempt records live under the output's
`technical/` directory and its `lanes/` execution records. The first CPU controller
(992617) exited when one training command encountered the temporarily absent
verification artifact during the taper precision correction. No center training
updates occurred in that attempt. GPU1's Slurm step continued independently;
GPU0 was restarted by CPU controller 992619 using the explicit recovery recipe.
The recovery controller waits for both lanes before producing the final report.

The first expanded-polynomial inner readout and its verification are preserved
under `technical/preserved-attempts/`. The active factored polynomial eliminates
small negative roundoff weights near the outer radius. Its frozen extraction and
GPU verification were rerun before the active comparison. No old outputs or
checkpoints were deleted.
