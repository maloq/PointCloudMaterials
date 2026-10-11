// Experimental CUDA FCC/HCP/BCC PTM. Upstream matching core: ptm/PROVENANCE.json.
// Neighbor search/face clipping are native CUDA; all geometry/matching float64.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <cassert>
template<class T> __device__ T ptm_max(T a,T b){return a>b?a:b;}
#include "ptm/ptm_convex_hull_incremental.cuh"
#include "ptm/ptm_graph_tools.cuh"
#include "ptm/ptm_canonical_coloured.cuh"
#include "ptm/ptm_normalize_vertices.cuh"
#include "ptm/ptm_graph_data.cuh"
#include "ptm/ptm_polar.cuh"
#include "ptm/ptm_structure_matcher.cuh"

namespace {
constexpr int NN=18, BIN=128, POLYGON=24;
__global__ void initialize_tables(int* error) {
  const ptm::refdata_t* all[3]={&ptm::structure_fcc,&ptm::structure_hcp,&ptm::structure_bcc};
  for(int a=0;a<3;a++) {
    const auto* s=all[a];
    for(int i=0;i<s->num_graphs;i++) {
      auto& g=s->graphs[i];int8_t degree[PTM_MAX_NBRS];
      int md=ptm::graph_degree(s->num_facets,g.facets,s->num_nbrs,degree);
      if(md>s->max_degree){*error=10;return;}
      double origin[3]={},normal[3];
      for(int j=0;j<s->num_facets;j++)ptm::add_facet(&s->points[0][1],g.facets[j][0],g.facets[j][1],g.facets[j][2],g.facets[j],normal,origin,0,nullptr);
      int8_t code[2*PTM_MAX_EDGES],colours[PTM_MAX_POINTS]={};
      int ret=ptm::canonical_form_coloured(s->num_facets,g.facets,s->num_nbrs,degree,colours,g.canonical_labelling,code,&g.hash);
      if(ret){*error=11;return;}
    }
  }
}
__device__ double delta(double a,double b,double length) {
  double d=a-b;return d-nearbyint(d/length)*length;
}
__global__ void bins_kernel(const double* x,const double* box,int n,int nx,int ny,int nz,
                           int* count,int* bins,int* cell,int* error) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  int dims[3]={nx,ny,nz};
  for(int d=0;d<3;d++)cell[3*i+d]=min(dims[d]-1,int(x[3*i+d]/box[d]*dims[d]));
  int c=(cell[3*i]*ny+cell[3*i+1])*nz+cell[3*i+2],slot=atomicAdd(count+c,1);
  if(slot>=BIN)atomicExch(error,1);else bins[c*BIN+slot]=i;
}
template<bool joint> __global__ void nearest_kernel(const double* x,const double* box,int n,int nx,int ny,int nz,double r2,
     const int* count,const int* bins,const int* cell,double* vectors,int* indices,int* error, const float* qx,const float* qb,float q2,int* neighbors,int* degree) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||*error)return;
  int close[64],nclose=0;
  double dist[NN];int ids[NN];for(int k=0;k<NN;k++){dist[k]=r2;ids[k]=-1;}
  for(int a=-1;a<=1;a++)for(int b=-1;b<=1;b++)for(int c=-1;c<=1;c++) {
    int cx=(cell[3*i]+a+nx)%nx,cy=(cell[3*i+1]+b+ny)%ny,cz=(cell[3*i+2]+c+nz)%nz;
    int bin=(cx*ny+cy)*nz+cz;
    for(int k=0;k<min(count[bin],BIN);k++) {
      int j=bins[bin*BIN+k];if(j==i)continue;
      double dx=delta(x[3*j],x[3*i],box[0]),dy=delta(x[3*j+1],x[3*i+1],box[1]),dz=delta(x[3*j+2],x[3*i+2],box[2]);
      if constexpr(joint) {
        float d[3];
        for(int v=0;v<3;v++) {
          float xi=qx[3*i+v],diff=qx[3*j+v]-xi;
          if(diff>=qb[v]*.5f)xi+=qb[v];
          if(diff< -qb[v]*.5f)xi-=qb[v];
          d[v]=qx[3*j+v]-xi;
        }
        if((d[0]*d[0]+d[1]*d[1])+d[2]*d[2]<q2) {
          if(nclose==64){atomicExch(error,6);return;}
          int at=nclose++;while(at>0&&close[at-1]>j){close[at]=close[at-1];at--;}close[at]=j;
        }
      }
      double d=(dx*dx+dy*dy)+dz*dz;if(d>=r2||d>dist[NN-1])continue;
      int at=NN-1;
      if(d==dist[at]&&j>=ids[at])continue;
      while(at>0&&(d<dist[at-1]||(d==dist[at-1]&&j<ids[at-1]))) {dist[at]=dist[at-1];ids[at]=ids[at-1];at--;}
      dist[at]=d;ids[at]=j;
    }
  }
  if(ids[NN-1]<0){atomicExch(error,2);return;}
  if constexpr(joint) {
    if(nclose==0){atomicExch(error,7);return;}
    degree[i]=nclose;for(int k=0;k<64;k++)neighbors[i*64+k]=k<nclose?close[k]:-1;
  }
  for(int k=0;k<NN;k++) {
    indices[i*NN+k]=ids[k];
    for(int d=0;d<3;d++)vectors[(i*NN+k)*3+d]=delta(x[3*ids[k]+d],x[3*i+d],box[d]);
  }
}
__device__ double dot(const double* a,const double* b){return (a[0]*b[0]+a[1]*b[1])+a[2]*b[2];}
__device__ void cross(const double* a,const double* b,double* c){c[0]=a[1]*b[2]-a[2]*b[1];c[1]=a[2]*b[0]-a[0]*b[2];c[2]=a[0]*b[1]-a[1]*b[0];}
// Each face is its bisector plane clipped by the other 17 neighbor halfspaces.
// Unlike Voro++ this uses bounded convex polygons; dense, periodic full cells only.
__global__ void angles_kernel(const double* vectors,int n,double* angles,int* error) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=n*NN||*error)return;
  int i=t/NN,k=t%NN;const double* p=vectors+i*NN*3;const double* r=p+k*3;
  double norm=sqrt(dot(r,r));if(norm==0){atomicExch(error,3);return;}
  double normal[3]={r[0]/norm,r[1]/norm,r[2]/norm},u[3]={},v[3];
  int axis=fabs(normal[0])<fabs(normal[1])?0:1;if(fabs(normal[2])<fabs(normal[axis]))axis=2;
  double unit[3]={};unit[axis]=1;cross(normal,unit,u);double un=sqrt(dot(u,u));for(int d=0;d<3;d++)u[d]/=un;cross(normal,u,v);
  double bound=20*sqrt(dot(p+(NN-1)*3,p+(NN-1)*3));
  double px[POLYGON]={-bound,bound,bound,-bound},py[POLYGON]={-bound,-bound,bound,bound};int nv=4;
  for(int j=0;j<NN&&nv;j++)if(j!=k) {
    const double* q=p+j*3;double a=dot(u,q),b=dot(v,q),c=.5*(dot(q,q)-dot(r,q));
    double ox[POLYGON],oy[POLYGON];int count=0;
    for(int e=0;e<nv;e++) {
      int f=(e+1)%nv;double ds=a*px[e]+b*py[e]-c,dt=a*px[f]+b*py[f]-c;
      bool s=ds<=0,t=dt<=0;
      if(s!=t) {
        double frac=ds/(ds-dt);if(count>=POLYGON){atomicExch(error,4);return;}
        ox[count]=px[e]+frac*(px[f]-px[e]);oy[count++]=py[e]+frac*(py[f]-py[e]);
      }
      if(t){if(count>=POLYGON){atomicExch(error,4);return;}ox[count]=px[f];oy[count++]=py[f];}
    }
    nv=count;for(int e=0;e<nv;e++){px[e]=ox[e];py[e]=oy[e];}
  }
  double xyz[POLYGON][3];
  for(int e=0;e<nv;e++) {
    if(fabs(px[e])>=bound*(1-1e-10)||fabs(py[e])>=bound*(1-1e-10)){atomicExch(error,5);return;}
    for(int d=0;d<3;d++)xyz[e][d]=.5*r[d]+u[d]*px[e]+v[d]*py[e];
    double length=sqrt(dot(xyz[e],xyz[e]));for(int d=0;d<3;d++)xyz[e][d]/=length;
  }
  double angle=0;
  for(int e=2;e<nv;e++) {
    double c[3];cross(xyz[e-1],xyz[e],c);
    angle+=fabs(2*atan2(dot(xyz[0],c),1+dot(xyz[0],xyz[e-1])+dot(xyz[e],xyz[0])+dot(xyz[e-1],xyz[e])));
  }
  angles[t]=angle;
}
__global__ void match_kernel(const double* vectors,const double* angles,int n,double cutoff,
                             uint8_t* labels,double* rmsd,int* error) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||*error)return;
  int order[NN];for(int k=0;k<NN;k++)order[k]=k;
  if(angles)for(int k=1;k<NN;k++) {
    int at=k,id=order[k];double value=angles[i*NN+id];
    while(at>0&&value>angles[i*NN+order[at-1]]){order[at]=order[at-1];at--;}
    order[at]=id;
  }
  double points[PTM_MAX_INPUT_POINTS][3]={},normalized[PTM_MAX_INPUT_POINTS][3];
  for(int k=0;k<NN;k++)for(int d=0;d<3;d++)points[k+1][d]=vectors[(i*NN+order[k])*3+d];
  ptm::normalize_vertices(PTM_MAX_INPUT_POINTS,points,normalized);
  ptm::convexhull_t hull={};ptm::result_t result={};result.rmsd=INFINITY;
  for(int k=0;k<PTM_MAX_POINTS;k++)result.mapping[k]=k;
  ptm::match_fcc_hcp_ico(normalized,points,PTM_CHECK_FCC|PTM_CHECK_HCP,&hull,&result);
  ptm::match_general(&ptm::structure_bcc,normalized,points,&hull,&result);
  rmsd[i]=result.rmsd;
  labels[i]=result.ref_struct&&result.rmsd<=cutoff?result.ref_struct->type:0;
}
}

void initialize(torch::Tensor error) {
  c10::cuda::CUDAGuard guard(error.device());
  initialize_tables<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>(error.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void nearest(torch::Tensor x,torch::Tensor box,int nx,int ny,int nz,double radius,
    torch::Tensor counts,torch::Tensor bins,torch::Tensor cells,torch::Tensor vectors,
    torch::Tensor indices,torch::Tensor error,c10::optional<torch::Tensor> qx,
    c10::optional<torch::Tensor> qb,double cutoff,c10::optional<torch::Tensor> neighbors,c10::optional<torch::Tensor> degree) {
  c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream();int n=x.size(0);
  bins_kernel<<<(n+127)/128,128,0,stream>>>(x.data_ptr<double>(),box.data_ptr<double>(),n,nx,ny,nz,counts.data_ptr<int>(),bins.data_ptr<int>(),cells.data_ptr<int>(),error.data_ptr<int>());
  if(qx)nearest_kernel<true><<<(n+127)/128,128,0,stream>>>(x.data_ptr<double>(),box.data_ptr<double>(),n,nx,ny,nz,radius*radius,counts.data_ptr<int>(),bins.data_ptr<int>(),cells.data_ptr<int>(),vectors.data_ptr<double>(),indices.data_ptr<int>(),error.data_ptr<int>(),qx->data_ptr<float>(),qb->data_ptr<float>(),float(cutoff)*float(cutoff),neighbors->data_ptr<int>(),degree->data_ptr<int>());
  else nearest_kernel<false><<<(n+127)/128,128,0,stream>>>(x.data_ptr<double>(),box.data_ptr<double>(),n,nx,ny,nz,radius*radius,counts.data_ptr<int>(),bins.data_ptr<int>(),cells.data_ptr<int>(),vectors.data_ptr<double>(),indices.data_ptr<int>(),error.data_ptr<int>(),nullptr,nullptr,0,nullptr,nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void angles(torch::Tensor vectors,torch::Tensor angles,torch::Tensor error) {
  c10::cuda::CUDAGuard guard(vectors.device());int n=vectors.size(0);
  angles_kernel<<<(n*NN+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(vectors.data_ptr<double>(),n,angles.data_ptr<double>(),error.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void match(torch::Tensor vectors,c10::optional<torch::Tensor> angles,double cutoff,torch::Tensor labels,torch::Tensor rmsd,torch::Tensor error) {
  c10::cuda::CUDAGuard guard(vectors.device());int n=vectors.size(0);
  match_kernel<<<(n+63)/64,64,0,at::cuda::getCurrentCUDAStream()>>>(vectors.data_ptr<double>(),angles?angles->data_ptr<double>():nullptr,n,cutoff,labels.data_ptr<uint8_t>(),rmsd.data_ptr<double>(),error.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("initialize",&initialize);m.def("nearest",&nearest);m.def("angles",&angles);m.def("match",&match);}
