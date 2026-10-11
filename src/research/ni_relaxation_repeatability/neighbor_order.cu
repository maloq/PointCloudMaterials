// Diagnostic canonicalization only; the physical force kernels are unchanged.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>

__global__ void canonical(int* neighbors, const int* counts,
                          const int64_t* tags, const int* rebuild) {
  if (!*rebuild) return;
  const int atom = blockIdx.x, slot = threadIdx.x;
  __shared__ int indices[256];
  __shared__ int64_t keys[256];
  const int index = slot < counts[atom] ? neighbors[atom*256+slot] : -1;
  indices[slot] = index;
  keys[slot] = index >= 0 ? tags[index] : INT64_MAX;
  __syncthreads();
  for (int size=2; size<=256; size*=2) {
    for (int stride=size/2; stride>0; stride/=2) {
      const int other = slot ^ stride;
      if (other > slot) {
        const bool ascending = (slot & size) == 0;
        if ((keys[slot] > keys[other]) == ascending) {
          const auto key=keys[slot]; keys[slot]=keys[other]; keys[other]=key;
          const int idx=indices[slot]; indices[slot]=indices[other]; indices[other]=idx;
        }
      }
      __syncthreads();
    }
  }
  if (slot < counts[atom]) neighbors[atom*256+slot] = indices[slot];
}

void sort_neighbors(torch::Tensor neighbors, torch::Tensor counts,
                    torch::Tensor tags, torch::Tensor rebuild) {
  c10::cuda::CUDAGuard guard(neighbors.device());
  canonical<<<counts.numel(),256,0,at::cuda::getCurrentCUDAStream()>>>(
    neighbors.data_ptr<int>(), counts.data_ptr<int>(),
    tags.data_ptr<int64_t>(), rebuild.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("sort_neighbors", &sort_neighbors); }
