"""Audit ancestry/precision; extract existing full-precision melt snapshots."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from src.project_runtime.paths import resolve_path,dataset_path
from src.data.temporal import TemporalLAMMPSDumpDataset
from src.training_methods.bcr.data import allowed_levels,audit_corruptions,balanced_subset
from src.training_methods.bcr.runtime import identity


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_melt(path):
    with Path(path).open() as f:
        h=TemporalLAMMPSDumpDataset._read_frame_header(f,source_path=Path(path))
        if h['atom_columns']!=('id','type','x','y','z'):raise ValueError('Unexpected melt-validation columns')
        table=np.loadtxt(f,max_rows=h['num_atoms'])
        if f.read().strip():raise ValueError('Expected one end-of-melt validation snapshot')
    # The shared header reader stores boxes as float32; reread these exact
    # producer fields in float64 before periodic extraction.
    with Path(path).open() as f:lines=[next(f) for _ in range(9)]
    bounds=np.array([[float(v) for v in line.split()] for line in lines[5:8]],dtype=np.float64)
    if bounds.shape!=(3,2):raise ValueError('Orthorhombic validation dump required')
    h['box_low']=bounds[:,0];h['box_high']=bounds[:,1]
    np.testing.assert_array_equal(table[:,0],np.arange(1,len(table)+1));np.testing.assert_array_equal(table[:,1],1)
    if h['timestep']!=100000:raise ValueError('Not the declared 300 ps independent melt endpoint')
    return h,table


def inventory(config):
    plan=json.loads(resolve_path(config['source_plan']).read_text());rows=[];selected=[]
    for source in plan['sources']:
        directory=dataset_path(source['dataset'])/Path(source['relative_trajectory_path']).parent
        validation=json.loads((directory/'source_validation.json').read_text());seed=int(validation['preparation_seed'])
        if source['lineage']!=f'independent_melt_{seed}':raise ValueError('Lineage does not match the melt producer')
        binary=json.loads((directory/Path(source['relative_trajectory_path']).name/'manifest.json').read_text())
        melt=directory/'melt_validation.lammpstrj';input_path=directory/'melt.in.lammps'
        text=input_path.read_text()
        # Validate the actual acquisition recipe; undercooling setpoint is NOT
        # the thermodynamic condition of the preceding high-temperature melt.
        if 'npt temp 1325 1325' not in text or 'run 100000' not in text:raise ValueError(f'Melt recipe differs: {input_path}')
        role=source.get('validation_role',source['split'])
        row=dict(trajectory_id=source['id'],independent_parent_id=source['lineage'],potential='Lee2003 Al 2NN-MEAM',
            undercooling_temperature_K=source['temperature_K'],melt_temperature_K=1325.,protocol='300 ps independently seeded full-box melt',
            history_dtype=binary['arrays']['positions']['dtype'],history_time_range_ps=f"0..{(binary['last_timestep']-binary['first_timestep'])*.003:g}",
            melt_precision='9 significant decimal digits from integration state; not float16 recovery',melt_time_ps=300.,melt_exists=melt.is_file(),
            historical_role=role,previous_model_selection_exposure='historical trajectory/representation studies; not a new final test',
            eligible_split='pilot_candidate' if role=='train' and source['temperature_K']==520 else 'locked_out',
            path=str(directory),melt_sha256=sha(melt) if melt.is_file() else None,validation_sha256=sha(directory/'source_validation.json'),melt_input_sha256=sha(input_path))
        rows.append(row)
        if row['eligible_split']=='pilot_candidate' and row['melt_exists']:selected.append(row)
    if len({r['independent_parent_id'] for r in rows})!=len(rows):raise ValueError('Duplicate independent roots in inventory')
    selected.sort(key=lambda r:r['trajectory_id']);rng=np.random.default_rng(config['seed']);selected=[selected[i] for i in rng.permutation(len(selected))]
    if len(selected)<18:raise ValueError('Need 18 eligible independent high-precision roots')
    selected=selected[:18]
    for i,r in enumerate(selected):r['pilot_split']='train' if i<12 else 'development'
    root=resolve_path(config['output'])/'technical';root.mkdir(parents=True,exist_ok=True)
    report=dict(rows=rows,selected=selected,source_plan_sha256=sha(resolve_path(config['source_plan'])),config=config,
        final_test='No final-test coordinates read; historical test/calibration/selection roots excluded')
    path=root/'inventory.json'
    if path.exists() and json.loads(path.read_text())!=report:raise ValueError('Frozen inventory changed')
    path.write_text(json.dumps(report,indent=2)+'\n')
    docs=Path('docs/datasets');docs.mkdir(parents=True,exist_ok=True)
    with (docs/'bcr_independent_roots.csv').open('w') as f:
        fields=list(rows[0]);fields+=['pilot_split'] if 'pilot_split' not in fields else []
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return report


def patches_from_snapshot(x,cell,atom_ids,centers,radius,max_atoms):
    """Exact orthorhombic fast extraction; explicit image offsets, no truncation."""
    cell=np.asarray(cell);length=cell.diagonal()
    if not np.allclose(cell,np.diag(length),atol=0,rtol=0) or radius>=length.min()/2:raise ValueError('Fast melt extraction requires orthorhombic cell and unique periodic image')
    x=np.mod(x,length);tree=cKDTree(x,boxsize=length);patches=[];identities=[];images=[];overflow=[]
    for center in centers:
        idx=np.array(sorted(tree.query_ball_point(x[center],radius)),dtype=int);idx=np.r_[center,idx[idx!=center]]
        raw=x[idx]-x[center];image=-np.rint(raw/length).astype(np.int64);relative=raw+image*length
        keep=np.linalg.norm(relative,axis=1)<radius;idx=idx[keep];relative=relative[keep];image=image[keep]
        if len(idx)>max_atoms:overflow.append((int(center),len(idx)))
        patches.append(relative.astype(np.float32));identities.append(atom_ids[idx]);images.append(image)
    if overflow:raise ValueError(f'All support overflows ({len(overflow)}): {overflow}')
    return patches,identities,images


def prepare(config):
    report=inventory(config);root=resolve_path(config['data']);root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():
        manifest=json.loads((root/'manifest.json').read_text())
        if manifest['config']!=config or sha(root/'patches.npz')!=manifest['patches_sha256']:raise ValueError('Pilot cache/config changed')
        return manifest
    patches=[];ids=[];images=[];records=[];uncertainty=0.;potential=config['potential_sha256']
    for source in report['selected']:
        path=Path(source['path'])/'melt_validation.lammpstrj'
        if sha(path)!=source['melt_sha256']:raise ValueError('Melt input changed')
        header,table=read_melt(path);x=table[:,2:]-header['box_low'];cell=np.diag(np.array(header['box_high'])-header['box_low'])
        # Producer used %.9g: worst half-last-digit rounding, doubled for
        # neighbor-center subtraction; include float32 local-offset storage.
        bound=float(10**(np.floor(np.log10(np.abs(table[:,2:]).max()))-8)+2*np.finfo(np.float32).eps*config['radius_A'])
        uncertainty=max(uncertainty,bound)
        rng=np.random.default_rng(np.random.SeedSequence([config['seed'],source['trajectory_id']]))
        # Spatial thinning is independent of order/outcomes. Minimum center
        # spacing 8 A reduces overlap; it does not imply independent patches.
        tree=cKDTree(np.mod(x,cell.diagonal()),boxsize=cell.diagonal());blocked=np.zeros(len(x),bool);centers=[]
        for idx in rng.permutation(len(x)):
            if blocked[idx]:continue
            centers.append(idx);blocked[tree.query_ball_point(np.mod(x[idx],cell.diagonal()),config['center_spacing_A'])]=True
            if len(centers)==config['centers_per_root']:break
        if len(centers)!=config['centers_per_root']:raise ValueError('Not enough spatially thinned centers')
        pp,ii,im=patches_from_snapshot(x,cell,table[:,0].astype(np.int64),centers,config['radius_A'],config['max_atoms'])
        patches.extend(pp);ids.extend(ii);images.extend(im)
        for center in centers:
            records.append(dict(root=source['independent_parent_id'],source=source['trajectory_id'],split=source['pilot_split'],block=0,frame=0,time_ps=300.,
                center_atom_id=int(table[center,0]),temperature_K=1325.,species='Al',cell_A=cell.tolist(),native_dtype='text %.9g integration snapshot',
                source_manifest_sha256=source['validation_sha256'],potential_sha256=potential,path=str(path),snapshot_sha256=source['melt_sha256']))
    train=[i for i,r in enumerate(records) if r['split']=='train'];d0=float(np.median([np.linalg.norm(patches[i][1:],axis=-1).min() for i in train]));n_ref=float(np.mean([len(patches[i]) for i in train]))
    levels=allowed_levels(config['noise_levels'],d0,uncertainty)
    if levels!=config['noise_levels']:raise ValueError('The fixed five-level pilot grid failed its precision guard')
    offsets=np.r_[0,np.cumsum([len(p) for p in patches])];np.savez(root/'patches.npz',positions=np.concatenate(patches),offsets=offsets,atom_ids=np.concatenate(ids),images=np.concatenate(images))
    manifest=dict(config=config,records=records,d0=d0,n_ref=n_ref,radius_A=config['radius_A'],coordinate_units='angstrom',noise_levels=levels,
        uncertainty_A=uncertainty,uncertainty_method='9 significant-digit text rounding plus local float32 storage, center-relative conservative bound',
        roots=18,training_roots=12,development_roots=6,anchors=len(records),training_anchors=len(train),development_anchors=len(records)-len(train),
        test_status='development_only_previously_studied_roots',patches_sha256=sha(root/'patches.npz'),inventory_sha256=sha(resolve_path(config['output'])/'technical/inventory.json'),
        preprocessing='bcr-complete-radius-pinned-v1',overflow_count=0,
        corruption_audit=audit_corruptions([patches[i] for i in balanced_subset(records,train,48)],levels,d0,config['radius_A']))
    manifest['identity']=identity(manifest);(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
