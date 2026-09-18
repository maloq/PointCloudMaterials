# PyTorch 2.14 GPU environment

New GPU work uses `conda activate pointnet-torch214` on this cluster. Its Python is
`/home/infres/vmorozov/miniconda3/envs/pointnet-torch214/bin/python`.
The original `pointnet` environment remains at PyTorch 2.11.0+cu128 for existing
jobs, frozen code and exact resumes. Environment changes create a new training
implementation identity; do not force an old checkpoint through an exact-resume
version check.

The upgraded stack is PyTorch 2.14.0+cu130, TorchVision 0.29.0+cu130, Triton 3.8.0,
MACE 0.3.16, cuEquivariance 0.10.0 (CUDA 13 operators), GATr 1.4.2 at the existing
pinned source revision, and xformers 0.0.35 from the CUDA 13.0 wheel index.
The CUDA 13.0 build supports both the H100 and RTX PRO 6000 Blackwell. The latter
is compute capability 12.0, included as `sm_120` in this PyTorch build.
Observed drivers are 595.84 on H100 and 615.71.09 on Blackwell.

[PyTorch 2.14 release](https://pytorch.org/blog/pytorch-2-14-release-blog/) and
[NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
describe the release and driver requirements. Actual local compilation checks,
rather than version numbers alone, establish usability on these nodes.

## Installation

The existing scientific environment was cloned before changing packages:

```bash
conda create -y --name pointnet-torch214 --clone pointnet
conda activate pointnet-torch214
python -m pip uninstall -y torchaudio cuequivariance-ops-torch-cu12 cuequivariance-ops-cu12
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 \
  torch==2.14.0+cu130 torchvision==0.29.0+cu130 cuequivariance-ops-torch-cu13==0.10.0
python -m pip install --no-deps --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu130 xformers==0.0.35
```

The old TorchAudio binary was removed from the clone: this repository does not
use it, and no matching 2.14 distribution is available. The original environment
retains it. GPU recipes and CUDA operator provenance now name the CUDA 13
distribution explicitly. The historical CPU lock remains unchanged.

`pip check` reports the same five pre-existing metadata conflicts as the source
environment: GATr's old NumPy upper bound; Zarr and Rasterio requiring NumPy 2;
and Wheel and Xarray requiring newer Packaging. These are recorded in the
before/after logs, not claimed to be resolved by the PyTorch upgrade. The actual
MACE/GATr training path is tested independently. NumPy remains at 1.26.4 for the
atomistic stack.

## Validation records

Install reports, frozen package lists, compiler checks and test logs are in
[`output/environment/torch214-20260918/technical`](../output/environment/torch214-20260918/technical/).
The standalone validation compiles a scalar network with `fullgraph=True`, then
checks each actual encoder with `fullgraph=False` in FP32 and BF16, including
backward execution and agreement with eager outputs. It allows existing upstream
graph breaks; it does not claim a single compiled graph for either backbone.
GATr explicitly disables compilation around its SDPA wrapper.

On Blackwell, 30 existing training, symmetry, gradient-caching and cuEquivariance
tests passed. Another 39 simulation/provenance tests passed on CPU. Both backbones
passed compiled FP32/BF16 forward and backward checks on both H100 and Blackwell
with `options={"emulate_precision_casts": True}` and a Dynamo recompile
limit of 32, needed for GATr's many EquiLinear channel shapes and precision modes.
The full-graph scalar test passed as well. Default compiled BF16 GATr failed the
eager-agreement check on both GPUs; those failed records are retained.

Blackwell maximum absolute encoder differences against eager execution:

| Encoder | FP32, casts preserved | BF16, default compilation | BF16, casts preserved |
| --- | ---: | ---: | ---: |
| MACE | 7.15e-7 | 0.01196 | 0.00275 |
| GATr | 7.15e-7 | 0.03136 | 2.38e-7 |

With casts preserved on H100, maximum FP32 differences were 7.15e-7 (MACE) and
8.64e-7 (GATr); BF16 differences were 0 and 2.38e-7 respectively. These are small
synthetic checks, not a bound over all observations. The original environment's
package freeze is byte-identical before and after the upgrade. The CUDA 13
xformers extension loads successfully; current GATr tensor masks use PyTorch SDPA.

New simulation provenance records CUDA 13 operators. Reading existing source
manifests checks the operator field against their recorded CUDA version, so
CUDA 12 source records remain valid and unchanged.

Compiler fusion normally removes intermediate downcast/upcast pairs. Preserving
them improved agreement with eager BF16 in this check. Eager agreement is not a
rotation-invariance test or evidence of improved physical prediction; default
fusion is not intrinsically less accurate against a higher-precision reference.
Compilation options must be explicit and validated on the intended training
workload before enabling them in a scientific campaign.

This upgrade does not itself enable compilation in scientific training, change
the mixed-precision policy, fix BF16 rotational errors, or establish a speedup.
The compiler checks use synthetic inputs and do not update research checkpoints.

## Current structural model sizes

Counts sum unique `nn.Parameter` elements, excluding buffers. Both encoders export
128-dimensional states. MACE uses 16 channels and two spatial layers; GATr uses
8 multivector and 128 scalar channels with two spatial and two temporal blocks.

| Parameters | MACE | GATr |
| --- | ---: | ---: |
| Stored encoder | 73,491 | 763,618 |
| Encoder snapshot modules | 73,491 | 399,504 |
| Snapshot modules + VICReg, physical and TDA heads | 185,528 | 511,541 |
| Complete stored structural model | 202,232 | 892,359 |

Snapshot GATr skips 364,114 parameters in temporal blocks, time embeddings and
history gates. The complete model also stores a 16,704-parameter JEPA predictor
unused by VICReg. Module counts describe architecture capacity, not a claim that
every parameter receives a nonzero gradient on every sample. Activations and
attention dominate training VRAM; these model weights occupy only a few MiB.
Exact counts and the model source hash are in `technical/parameters.json`.
