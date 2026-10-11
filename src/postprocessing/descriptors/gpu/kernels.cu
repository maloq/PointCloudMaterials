// Experimental structural analysis kernels. No fast-math or fused reductions.
// freud/fsph arithmetic adaptation: see NOTICE.md for upstreams and licenses.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

constexpr int CAP=64, BIN=128;
__device__ float wrap(float x,float length) {
  // Match freud Box::wrap, including its two float32 modulus operations.
  float f=(x+length*.5f)/length;
  f=fmodf(fmodf(f,1.f)+1.f,1.f);
  return -length*.5f+f*length;
}
__device__ float image_delta(float xj,float xi,float length) {
  float d=xj-xi;
  // AABBQuery translates the query position before taking the difference.
  if(d>=length*.5f)xi+=length;
  if(d< -length*.5f)xi-=length;
  return xj-xi;
}
__global__ void bins_kernel(const float* x,const float* box,int n,int nx,int ny,int nz,
                            int* counts,int* bins,int* cells,int* error) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  int dim[3]={nx,ny,nz};int c[3];
  for(int d=0;d<3;d++) {
    double v=double(x[3*i+d])/double(box[d])+.5;
    v-=floor(v);c[d]=min(dim[d]-1,int(v*dim[d]));cells[3*i+d]=c[d];
  }
  int cell=(c[0]*ny+c[1])*nz+c[2];int slot=atomicAdd(counts+cell,1);
  if(slot>=BIN)atomicExch(error,1);else bins[cell*BIN+slot]=i;
}
__global__ void neighbors_kernel(const float* x,const float* box,int n,int nx,int ny,int nz,
    float r2,const int* counts,const int* bins,const int* cells,int* neighbors,int* degree,int* error) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||*error)return;
  int found[CAP],size=0;
  for(int a=-1;a<=1;a++)for(int b=-1;b<=1;b++)for(int c=-1;c<=1;c++) {
    int cx=(cells[3*i]+a+nx)%nx,cy=(cells[3*i+1]+b+ny)%ny,cz=(cells[3*i+2]+c+nz)%nz;
    int cell=(cx*ny+cy)*nz+cz;
    for(int k=0;k<min(counts[cell],BIN);k++) {
      int j=bins[cell*BIN+k];if(i==j)continue;
      float dx=image_delta(x[3*j],x[3*i],box[0]);
      float dy=image_delta(x[3*j+1],x[3*i+1],box[1]);
      float dz=image_delta(x[3*j+2],x[3*i+2],box[2]);
      if((dx*dx+dy*dy)+dz*dz>=r2)continue;
      if(size>=CAP){atomicExch(error,2);continue;}
      // freud's NeighborList is sorted by atom index, not distance.
      int at=size++;while(at>0&&found[at-1]>j){found[at]=found[at-1];at--;}
      found[at]=j;
    }
  }
  degree[i]=size;
  for(int k=0;k<CAP;k++)neighbors[i*CAP+k]=k<size?found[k]:-1;
}

// fsph's normalized Jacobi recurrence, specialized to l=6. Match float32
// neighbor accumulation and reference wrap/distance paths separately.
__global__ void harmonics_kernel(const float* x,const float* box,const int* neighbors,
    const int* degree,int n,float* harmonic,float* ql) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  float real[13]={0},imag[13]={0};
  constexpr double pi=3.14159265358979323846;
  for(int k=0;k<degree[i];k++) {
    int j=neighbors[i*CAP+k];
    float dx=wrap(x[3*j]-x[3*i],box[0]),dy=wrap(x[3*j+1]-x[3*i+1],box[1]);
    float dz=wrap(x[3*j+2]-x[3*i+2],box[2]);
    float ax=image_delta(x[3*j],x[3*i],box[0]),ay=image_delta(x[3*j+1],x[3*i+1],box[1]);
    float az=image_delta(x[3*j+2],x[3*i+2],box[2]);
    float r=sqrtf((ax*ax+ay*ay)+az*az);
    float phi=atan2f(dy,dx),theta=r?acosf(fminf(1.f,fmaxf(-1.f,dz/r))):0.f;
    float z=cosf(theta),s= sinf(theta),spow=1.f,j0=float(1./sqrt(2.));
    #pragma unroll
    for(int m=0;m<=6;m++) {
      if(m){spow*=s;j0=float(double(j0)*sqrt(1.+1./2./m));}
      float prev=0.f,p=j0;
      #pragma unroll
      for(int l=1;l<=6-m;l++) {
        float f1=float(2.*sqrt(1.+(m-.5)/l)*sqrt(1.-(m-.5)/(l+2*m)));
        float f2=l==1?0.f:float(-sqrt(1.+4./(2*l+2*m-3))*sqrt(1.-1./l)*sqrt(1.-1./(l+2*m)));
        float next=z*f1*p+f2*prev;prev=p;p=next;
      }
      float value=float(double(spow*p)/sqrt(2.*pi));
      float angle=m*phi,re=value*cosf(angle),im=value*sinf(angle),phase=m%2?-1.f:1.f;
      real[m]+=phase*re;imag[m]+=phase*im;
      if(m){real[6+m]+=re;imag[6+m]-=im;}
    }
  }
  float norm=0;
  for(int m=0;m<13;m++) {
    float re=real[m]/degree[i],im=imag[m]/degree[i];
    harmonic[(i*13+m)*2]=re;harmonic[(i*13+m)*2+1]=im;norm+=re*re+im*im;
  }
  ql[i]=sqrtf(norm*float(4.*pi/13.));
}

__global__ void correlations_kernel(const int* neighbors,const float* h,const float* q,
    int n,float* correlations) {
  int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=n*CAP)return;
  int i=e/CAP,j=neighbors[e];if(j<0){correlations[e]=-2.f;return;}
  float sum=0;
  for(int m=0;m<13;m++)sum+=h[(i*13+m)*2]*h[(j*13+m)*2]+h[(i*13+m)*2+1]*h[(j*13+m)*2+1];
  correlations[e]=sum*(float(4.*3.14159265358979323846/13.)/(q[i]*q[j]));
}
__global__ void solid_kernel(const int* neighbors,const float* corr,int n,float threshold,int required,
    bool* solid) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int degree=0;
  for(int k=0;k<CAP;k++)if(neighbors[i*CAP+k]>=0&&corr[i*CAP+k]>threshold)degree++;
  solid[i]=degree>=required;
}

__device__ int root(int* parent,int x) {
  int p=atomicAdd(parent+x,0);
  while(p!=x){x=p;p=atomicAdd(parent+x,0);}return x;
}
__global__ void union_kernel(const int* neighbors,const bool* solid,const float* corr,
    float threshold,int n,int* parent) {
  int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=n*CAP)return;
  int i=e/CAP,j=neighbors[e];
  if(j<0||!solid[i]||!solid[j]||(corr&&!(corr[e]>threshold)))return;
  int a=root(parent,i),b=root(parent,j);
  while(a!=b) {
    int hi=max(a,b),lo=min(a,b);int old=atomicCAS(parent+hi,hi,lo);
    if(old==hi)return;
    a=root(parent,old);b=root(parent,lo);
  }
}
__global__ void roots_kernel(int* parent,const bool* solid,int n,int* sizes,int* roots) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  // Union has completed. parent is only read here; write compressed roots
  // separately to avoid mixing atomic reads and non-atomic concurrent writes.
  int r=solid[i]?root(parent,i):i;roots[i]=r;
  if(solid[i]) {
    // Aggregate equal roots within a warp: a large crystal no longer issues
    // one contended global atomic increment for every atom.
    unsigned peers=__match_any_sync(__activemask(),r);
    if((threadIdx.x&31)==__ffs(peers)-1)atomicAdd(sizes+r,__popc(peers));
  }
}
__global__ void labels_kernel(int* parent,const int* sizes,const int* names,const bool* solid,
    int n,int minimum,int* labels) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  int r=parent[i];labels[i]=solid[i]&&sizes[r]>=minimum?names[r]:0;
}

// Combined correlation/histogram pass avoids dynamic boolean indexing, float64
// copies of all edges, bucketize, and their device/host size synchronizations.
__global__ void correlation_histogram_kernel(const int* neighbors,const float* h,const float* q,
    int n,const double* edges,float* corr,int64_t* histogram,int* error) {
  __shared__ int hist[40];if(threadIdx.x<40)hist[threadIdx.x]=0;__syncthreads();
  int e=blockIdx.x*blockDim.x+threadIdx.x;
  if(e<n*CAP) {
    int i=e/CAP,j=neighbors[e];float value=-2.f;
    if(j>=0) {
      float sum=0;
      for(int m=0;m<13;m++)sum+=h[(i*13+m)*2]*h[(j*13+m)*2]+h[(i*13+m)*2+1]*h[(j*13+m)*2+1];
      value=sum*(float(4.*3.14159265358979323846/13.)/(q[i]*q[j]));
      if(!isfinite(value))atomicExch(error,1);
      if(value>=-1.f&&value<=1.f) {
        int lo=0,hi=41;while(lo<hi){int mid=(lo+hi)/2;if(double(value)<edges[mid])hi=mid;else lo=mid+1;}
        atomicAdd(hist+min(39,lo-1),1);
      }
    }
    corr[e]=value;
  }
  __syncthreads();if(threadIdx.x<40)atomicAdd((unsigned long long*)(histogram+threadIdx.x),(unsigned long long)hist[threadIdx.x]);
}
__global__ void masks_kernel(const int* neighbors,const float* corr,const uint8_t* ptm,int n,
    const float* thresholds,const int* required,bool* masks,int64_t* counts) {
  __shared__ int totals[4];if(threadIdx.x<4)totals[threadIdx.x]=0;__syncthreads();
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) {
    int found[3]={};for(int k=0;k<CAP;k++)if(neighbors[i*CAP+k]>=0) {
      float c=corr[i*CAP+k];for(int d=0;d<3;d++)found[d]+=c>thresholds[d];
    }
    bool crystalline=ptm[i]>=1&&ptm[i]<=3;masks[i]=crystalline;
    if(crystalline)atomicAdd(totals,1);
    for(int d=0;d<3;d++) {bool solid=found[d]>=required[d];masks[(d+1)*n+i]=solid;if(solid)atomicAdd(totals+d+1,1);}
  }
  __syncthreads();if(threadIdx.x<4)atomicAdd((unsigned long long*)(counts+threadIdx.x),(unsigned long long)totals[threadIdx.x]);
}
void correlation_histogram(torch::Tensor neighbors,torch::Tensor h,torch::Tensor q,torch::Tensor edges,
    torch::Tensor corr,torch::Tensor histogram,torch::Tensor error) {
  c10::cuda::CUDAGuard guard(h.device());int n=h.size(0);
  correlation_histogram_kernel<<<(n*CAP+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(neighbors.data_ptr<int>(),h.data_ptr<float>(),q.data_ptr<float>(),n,edges.data_ptr<double>(),corr.data_ptr<float>(),histogram.data_ptr<int64_t>(),error.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void masks(torch::Tensor neighbors,torch::Tensor corr,torch::Tensor ptm,torch::Tensor thresholds,
    torch::Tensor required,torch::Tensor masks,torch::Tensor counts) {
  c10::cuda::CUDAGuard guard(neighbors.device());int n=neighbors.size(0);
  masks_kernel<<<(n+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(neighbors.data_ptr<int>(),corr.data_ptr<float>(),ptm.data_ptr<uint8_t>(),n,thresholds.data_ptr<float>(),required.data_ptr<int>(),masks.data_ptr<bool>(),counts.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void neighbors(torch::Tensor x,torch::Tensor box,int nx,int ny,int nz,double cutoff,
    torch::Tensor neighbors,torch::Tensor degree,torch::Tensor counts,torch::Tensor bins,
    torch::Tensor cells,torch::Tensor error) {
  c10::cuda::CUDAGuard guard(x.device());auto s=at::cuda::getCurrentCUDAStream();int n=x.size(0);
  bins_kernel<<<(n+255)/256,256,0,s>>>(x.data_ptr<float>(),box.data_ptr<float>(),n,nx,ny,nz,
    counts.data_ptr<int>(),bins.data_ptr<int>(),cells.data_ptr<int>(),error.data_ptr<int>());
  neighbors_kernel<<<(n+127)/128,128,0,s>>>(x.data_ptr<float>(),box.data_ptr<float>(),n,nx,ny,nz,float(cutoff)*float(cutoff),
    counts.data_ptr<int>(),bins.data_ptr<int>(),cells.data_ptr<int>(),neighbors.data_ptr<int>(),degree.data_ptr<int>(),error.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void harmonics(torch::Tensor x,torch::Tensor box,torch::Tensor neighbors,torch::Tensor degree,
    torch::Tensor h,torch::Tensor q) {
  c10::cuda::CUDAGuard guard(x.device());auto s=at::cuda::getCurrentCUDAStream();int n=x.size(0);
  harmonics_kernel<<<(n+127)/128,128,0,s>>>(x.data_ptr<float>(),box.data_ptr<float>(),neighbors.data_ptr<int>(),
    degree.data_ptr<int>(),n,h.data_ptr<float>(),q.data_ptr<float>());C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void correlations(torch::Tensor neighbors,torch::Tensor h,torch::Tensor q,torch::Tensor corr) {
  c10::cuda::CUDAGuard guard(h.device());auto s=at::cuda::getCurrentCUDAStream();int n=h.size(0);
  correlations_kernel<<<(n*CAP+255)/256,256,0,s>>>(neighbors.data_ptr<int>(),h.data_ptr<float>(),q.data_ptr<float>(),n,corr.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void solid(torch::Tensor neighbors,torch::Tensor corr,double threshold,int required,torch::Tensor mask) {
  c10::cuda::CUDAGuard guard(corr.device());auto s=at::cuda::getCurrentCUDAStream();int n=neighbors.size(0);
  solid_kernel<<<(n+255)/256,256,0,s>>>(neighbors.data_ptr<int>(),corr.data_ptr<float>(),n,float(threshold),required,mask.data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
torch::Tensor components(torch::Tensor neighbors,torch::Tensor solid,c10::optional<torch::Tensor> corr,
    double threshold,int minimum) {
  c10::cuda::CUDAGuard guard(neighbors.device());auto s=at::cuda::getCurrentCUDAStream();int n=neighbors.size(0);
  auto opts=neighbors.options();auto parent=torch::arange(n,opts);auto sizes=torch::zeros({n},opts);auto roots=torch::empty({n},opts);
  union_kernel<<<(n*CAP+255)/256,256,0,s>>>(neighbors.data_ptr<int>(),solid.data_ptr<bool>(),
    corr.has_value()?corr.value().data_ptr<float>():nullptr,float(threshold),n,parent.data_ptr<int>());
  roots_kernel<<<(n+255)/256,256,0,s>>>(parent.data_ptr<int>(),solid.data_ptr<bool>(),n,sizes.data_ptr<int>(),roots.data_ptr<int>());
  auto names=torch::cumsum(sizes.ge(minimum),0,torch::kInt32);auto labels=torch::empty({n},opts);
  labels_kernel<<<(n+255)/256,256,0,s>>>(roots.data_ptr<int>(),sizes.data_ptr<int>(),names.data_ptr<int>(),solid.data_ptr<bool>(),n,minimum,labels.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return labels;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("neighbors",&neighbors);m.def("harmonics",&harmonics);m.def("correlations",&correlations);
  m.def("solid",&solid);m.def("components",&components);
  m.def("correlation_histogram",&correlation_histogram);m.def("masks",&masks);
}
