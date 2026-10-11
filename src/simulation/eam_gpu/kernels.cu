// GPL-2.0-or-later. See NOTICE.md. Single-element EAM, no fast-math.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int MAX_NEIGH = 256;
constexpr int BIN_CAPACITY = 128;
constexpr unsigned WARP_MASK = 0xffffffffu;

template<typename T> __device__ inline T delta(T d,T length,bool periodic) {
  return periodic ? d-length*nearbyint(d/length) : d;
}

template<typename T> __global__ void fill_bins(
    const T* x,const T* origin,const T* length,int nx,int ny,int nz,int atoms,int total,
    T cutoff,bool periodic,int* counts,int* bins,int* atom_cells,int* error,const int* rebuild=nullptr) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=total || (rebuild && !*rebuild))return;
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
    int* neighbors,int* nneigh,int* error,const int* rebuild=nullptr,T* reference=nullptr) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=total || (rebuild && !*rebuild))return;
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
      if(r2<=0){atomicExch(error,5);continue;}
      if(r2>=cutoff2)continue;
      if(n<MAX_NEIGH)neighbors[i*MAX_NEIGH+n]=j;
      else atomicExch(error,2);
      n++;

    }
  }
  nneigh[i]=min(n,MAX_NEIGH);
  if(reference)for(int a=0;a<3;a++)reference[3*i+a]=x[3*i+a];
}

// Each warp owns one atom. Directed neighbors avoid force atomics entirely.
template<typename T> __device__ inline T warp_sum(T v) {
  for(int o=16;o;o/=2)v+=__shfl_down_sync(WARP_MASK,v,o);
  return v;
}
template<typename T> struct Value {T y,d,dd;};
template<typename T> __device__ inline Value<T> eval(const T* table,int n,T inv,T r) {
  T coordinate=r*inv;
  int k=max(0,min(int(coordinate),n-1));T t=min(coordinate-T(k),T(1));
  const T* c=table+4*k;
  return {((c[3]*t+c[2])*t+c[1])*t+c[0],((T(3)*c[3]*t+T(2)*c[2])*t+c[1])*inv,
          (T(6)*c[3]*t+T(2)*c[2])*inv*inv};
}
template<typename T> __global__ void density_kernel(const T* x,const T* dq,const T* box,bool periodic,
    const int* neighbors,const int* count,int atoms,int total,int dirs,const T* rho_table,int nr,T invr,
    const T* embed,int nrho,T invrho,T cutoff2,T* fp,T* dfp,double* energy,int* error) {
  int i=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
  if(i>=total || *error)return;
  T rho=0,drho[4]={0,0,0,0};
  for(int slot=lane;slot<count[i];slot+=32) {
    int j=neighbors[i*MAX_NEIGH+slot];T d[3],r2=0;
    for(int a=0;a<3;a++){d[a]=delta(x[3*j+a]-x[3*i+a],box[a],periodic);r2+=d[a]*d[a];}
    if(r2>=cutoff2)continue;
    if(r2<=T(0)){atomicExch(error,5);continue;}
    T r=sqrt(r2);auto rho_r=eval(rho_table,nr,invr,r);rho+=rho_r.y;
    for(int k=0;k<dirs;k++) {
      T dr=0;for(int a=0;a<3;a++)dr+=d[a]*(dq[(3*j+a)*dirs+k]-dq[(3*i+a)*dirs+k]);
      drho[k]+=rho_r.d*dr/r;
    }
  }
  rho=warp_sum(rho);for(int k=0;k<dirs;k++)drho[k]=warp_sum(drho[k]);
  if(lane==0){
    if(rho<T(0)||rho>T(nrho)/invrho||!isfinite(rho)){atomicExch(error,6);}
    auto f=eval(embed,nrho,invrho,rho);fp[i]=f.d;energy[i]=double(f.y);
    for(int k=0;k<dirs;k++)dfp[i*dirs+k]=f.dd*drho[k];
  }
}
template<typename T> __global__ void force_kernel(const T* x,const T* dq,const T* box,bool periodic,
    const int* neighbors,const int* count,int total,int dirs,const T* rho_table,const T* pair_table,int nr,T invr,
    T cutoff2,const T* fp,const T* dfp,double* energy,double* virial,T* forces,T* tangent,const int* error) {
  int i=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;if(i>=total || *error)return;
  T f[3]={0,0,0},h[12]={0},e=0,w=0;
  for(int slot=lane;slot<count[i];slot+=32){
    int j=neighbors[i*MAX_NEIGH+slot];T d[3],r2=0;
    for(int a=0;a<3;a++){d[a]=delta(x[3*j+a]-x[3*i+a],box[a],periodic);r2+=d[a]*d[a];}
    if(r2>=cutoff2||r2<=0)continue;
    T r=sqrt(r2),ir=T(1)/r;
    auto rho=eval(rho_table,nr,invr,r);auto z=eval(pair_table,nr,invr,r);
    T phi=z.y*ir,phip=(z.d-phi)*ir,phipp=(z.dd-T(2)*phip)*ir;
    T emb=fp[i]+fp[j],g=emb*rho.d+phip,gr=g*ir;
    e+=T(.5)*phi;w-=T(.5)*g*r;
    for(int a=0;a<3;a++)f[a]+=gr*d[a];
    for(int k=0;k<dirs;k++){
      T dd[3],dr=0;
      for(int a=0;a<3;a++){dd[a]=dq[(3*j+a)*dirs+k]-dq[(3*i+a)*dirs+k];dr+=d[a]*dd[a];}
      dr*=ir;
      T dg=(dfp[i*dirs+k]+dfp[j*dirs+k])*rho.d+(emb*rho.dd+phipp)*dr;
      T dgr=(dg-gr*dr)*ir;
      for(int a=0;a<3;a++)h[a*dirs+k]+=dgr*d[a]+gr*dd[a];
    }
  }
  e=warp_sum(e);w=warp_sum(w);
  for(int a=0;a<3;a++)f[a]=warp_sum(f[a]);
  for(int k=0;k<3*dirs;k++)h[k]=warp_sum(h[k]);
  if(lane==0){energy[i]+=double(e);virial[i]=double(w);
    for(int a=0;a<3;a++)forces[3*i+a]=f[a];
    for(int k=0;k<3*dirs;k++)tangent[3*i*dirs+k]=h[k];}
}
void build_neighbors(torch::Tensor x,torch::Tensor origin,torch::Tensor length,int64_t nx,int64_t ny,int64_t nz,
    int64_t atoms,double cutoff,bool periodic,torch::Tensor bin_counts,torch::Tensor bins,torch::Tensor atom_cells,
    torch::Tensor neighbors,torch::Tensor nneigh,torch::Tensor error){
  c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream();int total=x.numel()/3;
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(),"eam_neighbors",[&]{
    fill_bins<scalar_t><<<(total+127)/128,128,0,stream>>>(x.data_ptr<scalar_t>(),origin.data_ptr<scalar_t>(),length.data_ptr<scalar_t>(),nx,ny,nz,atoms,total,scalar_t(cutoff),periodic,bin_counts.data_ptr<int>(),bins.data_ptr<int>(),atom_cells.data_ptr<int>(),error.data_ptr<int>());
    make_neighbors<scalar_t><<<(total+127)/128,128,0,stream>>>(x.data_ptr<scalar_t>(),length.data_ptr<scalar_t>(),nx,ny,nz,atoms,total,scalar_t(cutoff*cutoff),periodic,bin_counts.data_ptr<int>(),bins.data_ptr<int>(),atom_cells.data_ptr<int>(),neighbors.data_ptr<int>(),nneigh.data_ptr<int>(),error.data_ptr<int>());
  });C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void evaluate(torch::Tensor x,torch::Tensor dq,torch::Tensor box,bool periodic,torch::Tensor neighbors,
    torch::Tensor count,torch::Tensor rho,torch::Tensor embed,torch::Tensor pair,double dr,double drho,double cutoff,
    torch::Tensor fp,torch::Tensor dfp,torch::Tensor energy,torch::Tensor virial,torch::Tensor force,torch::Tensor tangent,torch::Tensor error){
  c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream();int total=x.numel()/3;
  int dirs=dq.numel()==0?0:dq.numel()/x.numel();TORCH_CHECK(dirs<=4,"At most four simultaneous tangent directions");
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(),"eam_evaluate",[&]{
    density_kernel<scalar_t><<<(total+7)/8,256,0,stream>>>(x.data_ptr<scalar_t>(),dq.data_ptr<scalar_t>(),box.data_ptr<scalar_t>(),periodic,neighbors.data_ptr<int>(),count.data_ptr<int>(),total,total,dirs,rho.data_ptr<scalar_t>(),rho.size(0),scalar_t(1/dr),embed.data_ptr<scalar_t>(),embed.size(0),scalar_t(1/drho),scalar_t(cutoff*cutoff),fp.data_ptr<scalar_t>(),dfp.data_ptr<scalar_t>(),energy.data_ptr<double>(),error.data_ptr<int>());
    force_kernel<scalar_t><<<(total+7)/8,256,0,stream>>>(x.data_ptr<scalar_t>(),dq.data_ptr<scalar_t>(),box.data_ptr<scalar_t>(),periodic,neighbors.data_ptr<int>(),count.data_ptr<int>(),total,dirs,rho.data_ptr<scalar_t>(),pair.data_ptr<scalar_t>(),rho.size(0),scalar_t(1/dr),scalar_t(cutoff*cutoff),fp.data_ptr<scalar_t>(),dfp.data_ptr<scalar_t>(),energy.data_ptr<double>(),virial.data_ptr<double>(),force.data_ptr<scalar_t>(),tangent.data_ptr<scalar_t>(),error.data_ptr<int>());
  });C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<typename T> __global__ void needs_rebuild(const T* x,const T* ref,const T* box,int n,T skin2,int* rebuild,int* error){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;T r2=0;
  for(int a=0;a<3;a++){T d=delta(x[3*i+a]-ref[3*i+a],box[a],true);r2+=d*d;}
  if(!isfinite(r2))atomicExch(error,4);
  if(r2>=skin2)atomicExch(rebuild,1);
}
__global__ void reset_bins(int* counts,int n,const int* rebuild,int* counter){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(!*rebuild)return;
  if(i<n)counts[i]=0;if(i==0)atomicAdd(counter,1);
}
void dynamic_neighbors(torch::Tensor x,torch::Tensor reference,torch::Tensor length,torch::Tensor origin,
    int64_t nx,int64_t ny,int64_t nz,int64_t atoms,double cutoff,double skin,
    torch::Tensor bin_counts,torch::Tensor bins,torch::Tensor atom_cells,torch::Tensor neighbors,
    torch::Tensor nneigh,torch::Tensor rebuild,torch::Tensor counter,torch::Tensor error){
  c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream();int total=x.numel()/3;
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(),"eam_dynamic_neighbors",[&]{
    needs_rebuild<scalar_t><<<(total+127)/128,128,0,stream>>>(x.data_ptr<scalar_t>(),reference.data_ptr<scalar_t>(),length.data_ptr<scalar_t>(),total,scalar_t(skin*skin/4),rebuild.data_ptr<int>(),error.data_ptr<int>());
    reset_bins<<<(bin_counts.numel()+127)/128,128,0,stream>>>(bin_counts.data_ptr<int>(),bin_counts.numel(),rebuild.data_ptr<int>(),counter.data_ptr<int>());
    fill_bins<scalar_t><<<(total+127)/128,128,0,stream>>>(x.data_ptr<scalar_t>(),origin.data_ptr<scalar_t>(),length.data_ptr<scalar_t>(),nx,ny,nz,atoms,total,scalar_t(cutoff),true,bin_counts.data_ptr<int>(),bins.data_ptr<int>(),atom_cells.data_ptr<int>(),error.data_ptr<int>(),rebuild.data_ptr<int>());
    make_neighbors<scalar_t><<<(total+127)/128,128,0,stream>>>(x.data_ptr<scalar_t>(),length.data_ptr<scalar_t>(),nx,ny,nz,atoms,total,scalar_t(cutoff*cutoff),true,bin_counts.data_ptr<int>(),bins.data_ptr<int>(),atom_cells.data_ptr<int>(),neighbors.data_ptr<int>(),nneigh.data_ptr<int>(),error.data_ptr<int>(),rebuild.data_ptr<int>(),reference.data_ptr<scalar_t>());
  });C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<typename T> __global__ void kick_drift(double* q,double* v,const T* f,T* x,int n,double dt,double alpha){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  v[i]+=.5*dt*alpha*double(f[i]);q[i]+=dt*v[i];x[i]=T(q[i]);
}
template<typename T> __global__ void finish_kick(double* v,const T* f,int n,double dt,double alpha){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)v[i]+=.5*dt*alpha*double(f[i]);
}
void kick(torch::Tensor q,torch::Tensor v,torch::Tensor f,torch::Tensor x,double dt,double alpha,bool first){
  c10::cuda::CUDAGuard guard(q.device());auto stream=at::cuda::getCurrentCUDAStream();int n=q.numel();
  AT_DISPATCH_FLOATING_TYPES(f.scalar_type(),"eam_kick",[&]{
    if(first)kick_drift<scalar_t><<<(n+127)/128,128,0,stream>>>(q.data_ptr<double>(),v.data_ptr<double>(),f.data_ptr<scalar_t>(),x.data_ptr<scalar_t>(),n,dt,alpha);
    else finish_kick<scalar_t><<<(n+127)/128,128,0,stream>>>(v.data_ptr<double>(),f.data_ptr<scalar_t>(),n,dt,alpha);
  });C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("build_neighbors",&build_neighbors);m.def("evaluate",&evaluate);m.def("dynamic_neighbors",&dynamic_neighbors);m.def("kick",&kick);}
