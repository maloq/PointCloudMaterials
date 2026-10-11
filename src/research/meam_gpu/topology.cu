// GPU topology selection for the pinned pure-Al MEAM energy.
// GPL-2.0-or-later. Screening equations derive from LAMMPS/Sandia MEAM;
// see ../response_meam/NOTICE.md. Energy and all continuous derivatives remain
// in PyTorch. This kernel only selects the exact active discrete support.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int MAX_NEIGH = 128;
constexpr int BIN_CAPACITY = 128;
constexpr unsigned WARP_MASK = 0xffffffffu;

template<typename T> __device__ inline T delta(T d,T length,bool periodic) {
  return periodic ? d-length*nearbyint(d/length) : d;
}

template<typename T> __global__ void fill_bins(
    const T* x,const T* origin,const T* length,int nx,int ny,int nz,int atoms,int total,
    T cutoff,bool periodic,int* counts,int* bins,int* atom_cells,int* error) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=total)return;
  int c[3];int dims[3]={nx,ny,nz};
  for(int d=0;d<3;d++) {
    // Bin addressing is infrequent and uses double even for float force
    // coordinates. Float wrapping of a tiny negative x can round to L exactly.
    double v=double(x[3*i+d])-double(origin[d]);
    if(periodic) {
      const double L=double(length[d]);v-=L*floor(v/L);
      c[d]=min(dims[d]-1,int(floor(v*double(dims[d])/L)));
    } else c[d]=int(floor(v/double(cutoff)));
    if(c[d]<0 || c[d]>=dims[d]) {atomicExch(error,4);return;}
    atom_cells[3*i+d]=c[d];
  }
  int cell=(((i/atoms)*nx+c[0])*ny+c[1])*nz+c[2];
  int slot=atomicAdd(counts+cell,1);
  if(slot>=BIN_CAPACITY)atomicExch(error,1);
  else bins[cell*BIN_CAPACITY+slot]=i;
}

template<typename T> __global__ void make_neighbors(
    const T* x,const T* length,int nx,int ny,int nz,int atoms,int total,T cutoff2,
    bool periodic,const int* counts,const int* bins,const int* atom_cells,
    int* neighbors,int* nneigh,int* pair_i,int* pair_j,int* pair_count,int pair_capacity,int* error) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=total)return;
  if(*error)return; // fill_bins completed on this stream; never read invalid cells
  int n=0;int batch=i/atoms;
  for(int ox=-1;ox<=1;ox++)for(int oy=-1;oy<=1;oy++)for(int oz=-1;oz<=1;oz++) {
    int cx=atom_cells[3*i]+ox,cy=atom_cells[3*i+1]+oy,cz=atom_cells[3*i+2]+oz;
    if(periodic){cx=(cx+nx)%nx;cy=(cy+ny)%ny;cz=(cz+nz)%nz;}
    else if(cx<0||cy<0||cz<0||cx>=nx||cy>=ny||cz>=nz)continue;
    int cell=((batch*nx+cx)*ny+cy)*nz+cz;
    int count=min(counts[cell],BIN_CAPACITY);
    for(int slot=0;slot<count;slot++) {
      int j=bins[cell*BIN_CAPACITY+slot];
      if(j==i)continue;
      T dx=delta(x[3*j]-x[3*i],length[0],periodic);
      T dy=delta(x[3*j+1]-x[3*i+1],length[1],periodic);
      T dz=delta(x[3*j+2]-x[3*i+2],length[2],periodic);
      T r2=dx*dx+dy*dy+dz*dz;
      if(r2>=cutoff2 || r2<=0)continue;
      if(n<MAX_NEIGH)neighbors[i*MAX_NEIGH+n]=j;
      else atomicExch(error,2);
      n++;
      if(i<j) {
        int pair=atomicAdd(pair_count,1);
        if(pair<pair_capacity){pair_i[pair]=i;pair_j[pair]=j;}
        else atomicExch(error,3);
      }
    }
  }
  nneigh[i]=min(n,MAX_NEIGH);
}

template<typename T> __global__ void select_screening(
    const T* x,const T* length,bool periodic,const int* neighbors,const int* nneigh,
    const int* pair_i,const int* pair_j,int npairs,int64_t* out_i,int64_t* out_j,
    int64_t* out_pair,int64_t* out_k,int* counts,int pair_capacity,int triplet_capacity) {
  int lane=threadIdx.x%32;
  int p=(blockIdx.x*blockDim.x+threadIdx.x)/32;
  if(p>=npairs)return;
  int i=pair_i[p],j=pair_j[p];
  T vx=delta(x[3*j]-x[3*i],length[0],periodic);
  T vy=delta(x[3*j+1]-x[3*i+1],length[1],periodic);
  T vz=delta(x[3*j+2]-x[3*i+2],length[2],periodic);
  T r2=vx*vx+vy*vy+vz*vz;
  // Conservative support around floating-point branch boundaries. The energy
  // evaluates the original cutoff exactly; extra candidates contribute zero.
  const T guard=sizeof(T)==4 ? T(2.e-5) : T(1.e-12);
  if(r2>=T(4.5*4.5)+guard)return;
  constexpr double ebound=2.8*2.8/(4*(2.8-1));
  int chosen[4]={-1,-1,-1,-1};unsigned masks[4]={0,0,0,0};int n=0;
  #pragma unroll
  for(int chunk=0;chunk<4;chunk++) {
    if(chunk*32>=nneigh[i])break;
    int slot=chunk*32+lane;
    int k=slot<nneigh[i] ? neighbors[i*MAX_NEIGH+slot] : -1;
    bool eligible=k>=0&&k!=j;T c=T(3.);
    if(eligible) {
      T ix=delta(x[3*k]-x[3*i],length[0],periodic);
      T iy=delta(x[3*k+1]-x[3*i+1],length[1],periodic);
      T iz=delta(x[3*k+2]-x[3*i+2],length[2],periodic);
      T jx=delta(x[3*k]-x[3*j],length[0],periodic);
      T jy=delta(x[3*k+1]-x[3*j+1],length[1],periodic);
      T jz=delta(x[3*k+2]-x[3*j+2],length[2],periodic);
      T ik2=ix*ix+iy*iy+iz*iz,jk2=jx*jx+jy*jy+jz*jz;
      T ratio=(ik2-jk2)/r2,a=T(1.)-ratio*ratio;
      eligible=a>T(0.)&&ik2<=T(ebound)*r2&&jk2<=T(ebound)*r2;
      if(eligible)c=(T(2.)*(ik2+jk2)/r2+a-T(2.))/a;
    }
    if(__any_sync(WARP_MASK,eligible&&c<=T(.49)-guard))return;
    bool partial=eligible&&c<T(2.8)+guard;
    masks[chunk]=__ballot_sync(WARP_MASK,partial);
    chosen[chunk]=partial ? k : -1;
    n+=__popc(masks[chunk]);
  }
  int edge=0,start=0;
  if(lane==0){edge=atomicAdd(counts,1);start=atomicAdd(counts+1,n);}
  edge=__shfl_sync(WARP_MASK,edge,0);start=__shfl_sync(WARP_MASK,start,0);
  if(edge>=pair_capacity||start+n>triplet_capacity){if(lane==0)atomicExch(counts+2,1);return;}
  if(lane==0){out_i[edge]=i;out_j[edge]=j;}
  int offset=0;
  #pragma unroll
  for(int chunk=0;chunk<4;chunk++) {
    if(chosen[chunk]>=0) {
      unsigned preceding=masks[chunk]&((1u<<lane)-1u);
      int t=start+offset+__popc(preceding);
      out_pair[t]=edge;out_k[t]=chosen[chunk];
    }
    offset+=__popc(masks[chunk]);
  }
}

void build_neighbors(torch::Tensor x,torch::Tensor origin,torch::Tensor length,
    int64_t nx,int64_t ny,int64_t nz,int64_t atoms,double cutoff,bool periodic,
    torch::Tensor bin_counts,torch::Tensor bins,torch::Tensor atom_cells,
    torch::Tensor neighbors,torch::Tensor nneigh,torch::Tensor pair_i,torch::Tensor pair_j,
    torch::Tensor pair_count,torch::Tensor error) {
  c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream();
  int total=x.numel()/3;
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(),"meam_bins",[&]{
    fill_bins<scalar_t><<<(total+127)/128,128,0,stream>>>(x.data_ptr<scalar_t>(),origin.data_ptr<scalar_t>(),
      length.data_ptr<scalar_t>(),nx,ny,nz,atoms,total,scalar_t(cutoff),periodic,
      bin_counts.data_ptr<int>(),bins.data_ptr<int>(),atom_cells.data_ptr<int>(),error.data_ptr<int>());
    make_neighbors<scalar_t><<<(total+127)/128,128,0,stream>>>(x.data_ptr<scalar_t>(),length.data_ptr<scalar_t>(),
      nx,ny,nz,atoms,total,scalar_t(cutoff*cutoff),periodic,bin_counts.data_ptr<int>(),bins.data_ptr<int>(),
      atom_cells.data_ptr<int>(),neighbors.data_ptr<int>(),nneigh.data_ptr<int>(),pair_i.data_ptr<int>(),
      pair_j.data_ptr<int>(),pair_count.data_ptr<int>(),pair_i.numel(),error.data_ptr<int>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void screening(torch::Tensor x,torch::Tensor length,bool periodic,torch::Tensor neighbors,
    torch::Tensor nneigh,torch::Tensor pair_i,torch::Tensor pair_j,int64_t npairs,
    torch::Tensor out_i,torch::Tensor out_j,torch::Tensor out_pair,torch::Tensor out_k,
    torch::Tensor counts) {
  c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(),"meam_screening",[&]{
    select_screening<scalar_t><<<(npairs+7)/8,256,0,stream>>>(x.data_ptr<scalar_t>(),length.data_ptr<scalar_t>(),
      periodic,neighbors.data_ptr<int>(),nneigh.data_ptr<int>(),pair_i.data_ptr<int>(),pair_j.data_ptr<int>(),
      npairs,out_i.data_ptr<int64_t>(),out_j.data_ptr<int64_t>(),out_pair.data_ptr<int64_t>(),
      out_k.data_ptr<int64_t>(),counts.data_ptr<int>(),out_i.numel(),out_pair.numel());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("build_neighbors",&build_neighbors);
  m.def("screening",&screening);
}
