"""Explicit small-velocity-perturbation Ni NPT ensemble, separate from model fitting."""
import argparse,fcntl,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from src.research.continuous_motion.data import Source,read,root,cache,resolve_path,sha,digest,write_json,atomic_save
from src.experiment_runner.execution import ExecutionBundle,SlurmQueue,allocation_deadline
from src.research.ni_phase_sim.campaign import environment
from src.experiment_runner.stages import run_stage
from src.research.meam_md.dynamics import archive_file
from src.project_runtime.paths import machine
MODULE='src.simulation.campaigns.ni_ordering_branches'
REPO=Path(__file__).resolve().parents[3]


def runtime_environment(bundle):
    """Use the shared Ni CUDA toolkit on every allocated GPU node."""
    cuda=machine()['execution']['gpu_meam_portable_cuda_root']
    return dict(environment(bundle),CUDA_HOME=cuda,
        PATH=cuda+'/bin:'+str(Path(sys.executable).parent)+':/usr/bin:/bin',
        UCX_TLS='tcp,self',MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0;12.0')


def prepare(c):
    path=root(c)/'technical/plan.json'
    if path.exists():return read(path)
    parent=read(resolve_path(c['ordering_run'])/'technical/plan.json')
    bank=torch.load(cache(parent['config'])/'base.pt',map_location='cpu',weights_only=False)
    labels=torch.load(parent['labels_path'],map_location='cpu',weights_only=False)
    column=bank['columns'].index('bond_order/l6_qbar')
    order=bank['x'][:,3,column].numpy();liquid=(~np.asarray(labels['crystal'])[:,3])&(np.asarray(labels['fraction'])[:,3]<=.05)
    threshold=np.quantile(order[liquid & np.array([r['role']=='train' for r in bank['rows']])],[1/3,2/3])
    bins=np.searchsorted(threshold,order);chosen=[];used=set()
    for group in range(3):
        for role in ('train','test'):
            candidates=[i for i,r in enumerate(bank['rows']) if liquid[i] and bins[i]==group and r['role']==role]
            candidates.sort(key=lambda i:digest([c['seed'],bank['rows'][i]['id']]))
            count=0
            for i in candidates:
                r=bank['rows'][i]
                if r['ancestor'] in used:continue
                chosen.append(dict(row_index=i,row=r,group=group));used.add(r['ancestor']);count+=1
                if count==4:break
            if count!=4:raise ValueError('Insufficient distinct ancestors in present-order stratum')
    jobs=[]
    for index,origin in enumerate(chosen):
        for branch in range(17):
            jobs.append(dict(name=f'origin-{index:02d}-branch-{branch:02d}',origin=index,branch=branch,
                seed=int(digest([c['seed'],index,branch])[:7],16),**origin))
    p=dict(config=c,origins=chosen,jobs=jobs,sources=parent['sources'],columns=bank['columns'],
        parent_identity=parent['identity'],ancestry='12 train and12 test ancestors inherited; no model fitting from this pilot',
        distribution='branch0 unperturbed; branches1..16 independent Gaussian velocity directions,COM-free, RMS vector amplitude1% of initial thermal RMS, then rescale peculiar velocities to preserve initial kinetic energy; positions fixed',
        protocol='Reconstructed saved float16 Q/V/box; new NPT thermostat/barostat chains identically initialized, not native restart replay or exact conditional ensemble calibration',
        integration='LAMMPS isotropic NPT1fs float64; FastNi Mendelev2012 EAM/FS float32 force;0bar; damping0.1/1ps',
        targets='nearest80 central Q,V and244measurements; full-cell PTM central label, central connected-component size, total fraction and largest component every0.1ps',
        thresholds_train_only=threshold.tolist())
    p['identity']=digest(p);write_json(path,p);return p


def observe(native,ids,center,c):
    from src.simulation.campaigns.ni_births.md import snapshot
    from src.postprocessing.descriptors.ptm import full_labels
    q,v,lo,hi=snapshot(native);box=hi-lo;ci=int(np.searchsorted(ids,center))
    _,near=cKDTree(q,boxsize=box).query(q[ci],k=80)
    local=q[near]-q[ci];local-=box*np.rint(local/box)
    state=np.concatenate((local,v[near]),-1).astype(np.float32)
    phase=full_labels(q,box,.1);solid=np.isin(phase,[1,2,3]);selected=np.flatnonzero(solid);largest=0;central_size=0
    if len(selected):
        edges=cKDTree(q[selected],boxsize=box).query_pairs(3.2,output_type='ndarray')
        graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(selected),len(selected))).tocsr()
        _,components=connected_components(graph,directed=False);counts=np.bincount(components);largest=int(counts.max())
        if solid[ci]:central_size=int(counts[components[np.searchsorted(selected,ci)]])
    return dict(state=state,center_ptm=int(phase[ci]),center_component=central_size,
        crystal_fraction=float(solid.mean()),largest_component=largest,box=box.astype(np.float32))


def run(c,job,deadline,max_frames=None):
    from lammps import lammps
    from src.simulation.eam_gpu.npt import install
    from src.research.measurement_sim.features import structural,SmoothMeasurements
    p=read(root(c)/'technical/plan.json');out=resolve_path(c['scratch'])/job['name'];out.mkdir(parents=True,exist_ok=True)
    published=resolve_path(c['archive'])/job['name'];published.mkdir(parents=True,exist_ok=True)
    marker=root(c)/'analyses'/job['name']/'technical/complete.json'
    if marker.exists():return True
    identity=digest([p['identity'],job]);item=p['sources'][job['row']['source']];reader=Source(item['source'])
    q,v,box=reader.frame(job['row']['frame']);temperature=item['source']['temperature_K'];ids=reader.ids.astype(np.int32)
    native=lammps(cmdargs=['-screen','none','-log',str(out/'lammps.log')]);partial=published/'partial.pt'
    observations=[];start=0;begin=time.monotonic()
    if partial.exists():
        packet=torch.load(partial,map_location='cpu',weights_only=False)
        if packet['identity']!=identity:raise ValueError('Changed physical branch identity')
        observations=packet['observations'];start=packet['frame'];native.command('read_restart '+str(published/f'state-{start:03d}.restart'))
    else:
        mean=v.mean(0);peculiar=v-mean
        if job['branch']:
            rng=np.random.default_rng(job['seed']);noise=rng.normal(size=v.shape);noise-=noise.mean(0)
            noise*=.01*np.sqrt(np.mean(np.sum(peculiar**2,axis=1)))/np.sqrt(np.mean(np.sum(noise**2,axis=1)))
            changed=peculiar+noise;changed*=np.sqrt(np.sum(peculiar**2)/np.sum(changed**2));v=mean+changed
        native.commands_string(f'''units metal
atom_style atomic
boundary p p p
atom_modify map array sort 0 0.0
region cell block 0 {box[0]:.17g} 0 {box[1]:.17g} 0 {box[2]:.17g} units box
create_box 1 cell
mass 1 58.71
''')
        n=native.create_atoms(len(q),atomid=ids,atype=np.ones(len(q),np.int32),x=np.remainder(q,box).ravel(),v=v.ravel())
        if n!=len(q):raise ValueError('Native full-cell atom creation failed')
        write_json(published/'initial.json',dict(identity=identity,source_input_identity=item['source']['input_identity'],
            source_frame=job['row']['frame'],center=job['row']['center'],role=job['row']['role'],ancestor=job['row']['ancestor'],
            branch=job['branch'],seed=job['seed'],temperature_K=temperature,atoms=len(q),
            perturbation_rms_A_ps=float(np.sqrt(np.mean(np.sum((v-reader.frame(job['row']['frame'])[1])**2,axis=1)))),
            semantics=p['protocol'],distribution=p['distribution']))
    if sha(resolve_path(c['potential']))!=c['potential_sha256']:raise ValueError('Changed Ni potential')
    callback=install(native,resolve_path(c['potential']),out)
    native.commands_string(f'''neighbor 0.6 bin
neigh_modify every 1 delay 0 check yes
timestep 0.001
fix integrate all npt temp {temperature} {temperature} 0.1 iso 0 0 1.0
thermo 100
thermo_style custom step time temp press vol pe ke etotal
thermo_modify flush yes lost error
run 0
''')
    def checkpoint(frame):
        path=out/f'state-{frame:03d}.restart';native.command('write_restart '+str(path));archive_file(path,published/path.name)
        atomic_save(partial,dict(identity=identity,frame=frame,observations=observations))
        write_json(marker.parent/'progress.json',dict(frame=frame,total=60,seconds=time.monotonic()-begin,updated=time.time()))
    try:
        if not observations:observations.append(observe(native,ids,job['row']['center'],c));checkpoint(0)
        for frame in range(start+1,61):
            if time.time()>deadline-180:checkpoint(frame-1);return False
            native.command('run 100 pre no post no');observations.append(observe(native,ids,job['row']['center'],c))
            if frame%10==0:checkpoint(frame)
            if max_frames and frame>=max_frames:checkpoint(frame);return False
        checkpoint(60)
    finally:native.close()
    state=torch.from_numpy(np.stack([r['state'] for r in observations]));smooth=SmoothMeasurements().cuda().eval()
    with torch.no_grad():values=smooth(state.cuda()).cpu()
    geometry=torch.from_numpy(structural(state[...,:3].numpy(),p['columns'][:153],c['length_factor']))
    atomic_save(published/'measurements.pt',dict(identity=identity,value=torch.cat((geometry,values),-1),state=state,
        observations=[{k:v for k,v in r.items() if k!='state'} for r in observations],times_ps=np.arange(61)*.1))
    write_json(marker,dict(state='complete',identity=identity,sha256=sha(published/'measurements.pt'),archive=str(published),seconds=time.monotonic()-begin))
    return True


def launch(c):
    p=prepare(c);tech=root(c)/'technical'
    if (tech/'submission.json').exists():raise FileExistsError('Physical pilot already launched')
    bundle=ExecutionBundle.freeze(REPO,tech/'source',c,directories=['src','configs','docs/metrics'],files=[(REPO/'machine.local.yaml','machine.local.yaml')])
    receipt=dict(started=time.time(),deadline=time.time()+9*3600,jobs={},origins=24,perturbed_branches=384,unperturbed_controls=24)
    queue=SlurmQueue(tech,bundle,MODULE,runtime_environment(bundle),tech/'submission.json',receipt,'ni-branch')
    with queue.submission():
        for lane in range(c['runtime']['workers']):
            queue.submit(f'worker-{lane}',['--gres=gpu:1','--cpus-per-task=4','--mem=32G','--time=09:00:00'],partition=c['runtime']['partitions'],command_stage='worker',arguments=['--lane',str(lane)])
    print(receipt,flush=True)


def worker(c,bundle,lane):
    p=read(root(c)/'technical/plan.json');global_deadline=read(root(c)/'technical/submission.json')['deadline']
    deadline=min(global_deadline,allocation_deadline(reserve_seconds=180))
    # A shared compiler failure must not consume every independent branch retry.
    try:
        from src.simulation.eam_gpu.potential import extension
        compiler=subprocess.check_output([str(Path(os.environ['CUDA_HOME'])/'bin/nvcc'),'--version'],text=True)
        extension()
        write_json(root(c)/'technical'/f'worker-runtime-{lane}.json',dict(cuda_home=os.environ['CUDA_HOME'],
            compiler=compiler,device=torch.cuda.get_device_name(),checked_at=time.time()))
    except Exception:
        write_json(root(c)/'technical'/f'worker-failed-{lane}.json',dict(stage='runtime',traceback=traceback.format_exc(),finished=time.time()))
        raise
    while time.time()<deadline-300:
        progress=False
        # Interleave origins: even a partial night covers all present-order bins.
        for job in sorted(p['jobs'],key=lambda j:(j['branch'],j['origin'])):
            result = run_stage(
                root(c), bundle, lane, deadline, job['name'], 'run', 'complete.json',
                module=MODULE, hard_deadline=True, max_attempts=c['runtime']['max_attempts'])
            if result=='paused':return
            if result is not None:progress=True;break
        if not progress:break
    write_json(root(c)/'technical'/f'worker-exit-{lane}.json',dict(finished=time.time(),completed=sum((root(c)/'analyses'/j['name']/'technical/complete.json').exists() for j in p['jobs']),planned=len(p['jobs'])))


def main():
    a=argparse.ArgumentParser(__doc__);a.add_argument('stage',choices=['prepare','launch','worker','run']);a.add_argument('--config',required=True);a.add_argument('--name');a.add_argument('--lane',type=int,default=0);a.add_argument('--deadline',type=float,default=float('inf'));a.add_argument('--max-frames',type=int)
    args=a.parse_args();c=read(args.config);torch.set_num_threads(1)
    if args.stage=='prepare':print(prepare(c)['identity'])
    elif args.stage=='launch':launch(c)
    elif args.stage=='worker':worker(c,ExecutionBundle(Path(args.config).resolve().parent),args.lane)
    else:
        p=read(root(c)/'technical/plan.json');job=next(j for j in p['jobs'] if j['name']==args.name)
        if not run(c,job,args.deadline,args.max_frames):sys.exit(75)

if __name__=='__main__':main()
