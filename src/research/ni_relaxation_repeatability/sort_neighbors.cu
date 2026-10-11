// Diagnostic extension: canonical neighbor order, without changing membership.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

__global__ void canonical(int* neighbors, const int* count, const int* tags,
                          const int* rebuild, const int* error) {
  if (!*rebuild || *error) return;
  int atom = blockIdx.x, lane = threadIdx.x;
  __shared__ unsigned long long keys[256];
  int n = count[atom];
  int j = lane < n ? neighbors[atom*256+lane] : 0;
  keys[lane] = lane < n ? (static_cast<unsigned long long>(tags[j]) << 32) |
                          static_cast<unsigned>(j) : ~0ULL;
  __syncthreads();
  for (int width=2; width<=256; width<<=1) {
    for (int stride=width>>1; stride; stride>>=1) {
      int other = lane ^ stride;
      if (other > lane) {
        auto a=keys[lane], b=keys[other];
        bool ascending = (lane & width) == 0;
        if ((ascending && a>b) || (!ascending && a<b)) {
          keys[lane]=b; keys[other]=a;
        }
      }
      __syncthreads();
    }
  }
  if (lane<n) neighbors[atom*256+lane]=static_cast<int>(keys[lane] & 0xffffffffULL);
}

void sort_neighbors(torch::Tensor neighbors, torch::Tensor count, torch::Tensor tags,
                    torch::Tensor rebuild, torch::Tensor error) {
  TORCH_CHECK(neighbors.is_cuda() && neighbors.scalar_type()==torch::kInt32 &&
              neighbors.is_contiguous() && neighbors.dim()==2 && neighbors.size(1)==256,
              "Expected contiguous CUDA int32 [N,256] neighbor array");
  c10::cuda::CUDAGuard guard(neighbors.device());
  canonical<<<neighbors.size(0),256,0,at::cuda::getCurrentCUDAStream()>>>(
      neighbors.data_ptr<int>(), count.data_ptr<int>(), tags.data_ptr<int>(),
      rebuild.data_ptr<int>(), error.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("sort_neighbors", &sort_neighbors); }
