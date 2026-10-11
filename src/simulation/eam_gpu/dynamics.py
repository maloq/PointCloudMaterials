"""GPU-resident fixed-box NVE with exact per-step Verlet-list safety checks."""
import math
import torch


class CapturedNVE:
    """A block of velocity Verlet steps with float64 positions and velocities.

    Neighbor displacement checks and conditional rebuilds execute on the GPU on
    EVERY step, inside the graph. The error flag is sticky throughout each block;
    run() checks it before returning. No unsafe fixed rebuild cadence is used.
    This is fixed-box NVE only, with no temperature/pressure control.
    """
    def __init__(self,model,q,v,*,dt_ps=.001,block_steps=100):
        if model.box is None or q.shape!=(model.atoms,3) or v.shape!=q.shape:
            raise ValueError('CapturedNVE requires one periodic cell, shape (atoms,3)')
        if q.dtype!=torch.float64 or v.dtype!=torch.float64:raise ValueError('NVE states must be float64')
        if dt_ps<=0 or block_steps<1:raise ValueError('Positive timestep and block length required')
        if v.device!=q.device or not bool(torch.isfinite(v).all()):raise ValueError('Finite velocities on the position device required')
        self.model=model;self.q=q.contiguous().clone();self.v=v.contiguous().clone();self.dt=dt_ps;self.block_steps=block_steps
        self.box=model.box.clone();self.alpha=9648.533215665327/model.data.mass
        x=model.support(q);n=len(x);self.x=x.clone()
        # support() can reuse a list built at an earlier configuration. Its
        # displacement origin must remain that actual build configuration.
        self.initial_reference=model.reference.clone();self.reference=self.initial_reference.clone()
        self.neighbors=model.neighbors.clone();self.count=model.count.clone()
        grid=(self.box/model.cutoff).floor().long().tolist();self.grid=grid
        integer=dict(device=q.device,dtype=torch.int32);cells=math.prod(grid)
        self.bin_counts=torch.empty(cells,**integer);self.bins=torch.empty(cells,128,**integer)
        self.atom_cells=torch.empty(n,3,**integer);self.error=torch.zeros(1,**integer)
        self.rebuild=torch.zeros(1,**integer);self.rebuild_count=torch.zeros(1,**integer)
        self.origin=torch.zeros(3,device=q.device,dtype=x.dtype)
        self.fp=torch.empty(n,device=q.device,dtype=x.dtype);self.empty=x.new_empty(0)
        self.e=torch.empty(n,device=q.device,dtype=torch.float64);self.w=torch.empty_like(self.e)
        self.f=torch.empty_like(x)
        self._evaluate()
        if int(self.error):raise RuntimeError('Invalid initial EAM state')
        self.graph=torch.cuda.CUDAGraph();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):self._step()
        torch.cuda.current_stream().wait_stream(stream)
        self._restore(q,v)
        with torch.cuda.graph(self.graph):
            for _ in range(block_steps):self._step()
        self._restore(q,v)

    def _evaluate(self):
        m=self.model
        m.native.evaluate(self.x,self.empty,self.box,True,self.neighbors,self.count,m.rho,m.embed,m.pair,
                          m.data.dr,m.data.drho,m.data.cutoff,self.fp,self.empty,self.e,self.w,self.f,self.empty,self.error)

    def _step(self):
        m=self.model
        m.native.kick(self.q,self.v,self.f,self.x,self.dt,self.alpha,True)
        self.rebuild.zero_()
        m.native.dynamic_neighbors(self.x,self.reference,self.box,self.origin,*self.grid,m.atoms,m.cutoff,m.skin,
            self.bin_counts,self.bins,self.atom_cells,self.neighbors,self.count,self.rebuild,self.rebuild_count,self.error)
        self._evaluate()
        m.native.kick(self.q,self.v,self.f,self.x,self.dt,self.alpha,False)

    def _restore(self,q,v):
        self.q.copy_(q);self.v.copy_(v);self.x.copy_(q);self.reference.copy_(self.initial_reference)
        self.neighbors.copy_(self.model.neighbors);self.count.copy_(self.model.count)
        self.error.zero_();self.rebuild_count.zero_();self._evaluate()

    def run(self,steps):
        if steps<0 or steps%self.block_steps:raise ValueError('Nonnegative steps divisible by the captured block length required')
        if not torch.equal(self.box,self.model.box):raise ValueError('CapturedNVE box was changed; construct a new integrator')
        for _ in range(steps//self.block_steps):
            self.graph.replay()
            code=int(self.error)
            if code:raise RuntimeError(f'Captured Ni NVE failed with error {code}; state rejected')
        return self.q.clone(),self.v.clone()

    def total_energy(self):
        return self.e.sum()+.5*self.model.data.mass/9648.533215665327*self.v.square().sum()
