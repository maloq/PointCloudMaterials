import os,json,csv,hashlib,shutil,time
from pathlib import Path
base=Path('/home/infres/vmorozov/PointCloudMaterials')
audit=base/'docs/storage/experiment_removal_20261002'; audit.mkdir(parents=True,exist_ok=True)
roots=[Path(p) for p in '''/store/PERSO/vmorozov/experiments/embedding_forecast
/work/PERSO/vmorozov/analysis/encoder_mechanisms
/store/PERSO/vmorozov/projects/PointCloudMaterials-20260913T174741Z/output/embedding_forecast_20260911
/work/PERSO/vmorozov/analysis/spatial_vicreg_bias
/home/infres/vmorozov/PointCloudMaterials/output/crystallization_transfer
/store/PERSO/vmorozov/experiments/liquid_predictability
/store/PERSO/vmorozov/experiments/pretrained_mace_spatiotemporal_20260906
/home/infres/vmorozov/PointCloudMaterials/output/encoder_research
/work/PERSO/vmorozov/analysis/liquid_predictability
/work/PERSO/vmorozov/analysis/encoder_context
/store/PERSO/vmorozov/projects/PointCloudMaterials-20260913T174741Z/output/embedding_forecast
/home/infres/vmorozov/PointCloudMaterials/output/spatial_vicreg_bias
/work/PERSO/vmorozov/analysis/structural_static'''.splitlines()]
(audit/'targets.json').write_text(json.dumps([str(p) for p in roots],indent=2))
keep={'.md','.csv','.tsv','.json','.jsonl','.yaml','.yml','.txt','.html','.svg','.png','.pdf','.jpg','.jpeg','.log','.toml'}
rows=[]; preserved=[]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
for i,root in enumerate(roots):
 if root.is_symlink(): raise RuntimeError(f'Root is symlink: {root}')
 if not root.is_dir(): raise RuntimeError(f'Missing root: {root}')
 for directory,dirs,files in os.walk(root,followlinks=False):
  for name in dirs+files:
   p=Path(directory)/name; st=p.lstat()
   if p.is_symlink(): rows.append([str(p),st.st_dev,st.st_ino,st.st_nlink,st.st_size,st.st_blocks*512,st.st_mtime_ns,'symlink',os.readlink(p)]); continue
   if p.is_dir(): continue
   rows.append([str(p),st.st_dev,st.st_ino,st.st_nlink,st.st_size,st.st_blocks*512,st.st_mtime_ns,'file',''])
   if p.suffix.lower() in keep:
    dst=audit/'results'/f'{i:02d}-{root.name}'/p.relative_to(root); dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,dst); digest=sha(p)
    if sha(dst)!=digest or p.stat().st_mtime_ns!=st.st_mtime_ns: raise RuntimeError(f'Changed: {p}')
    preserved.append({'source':str(p),'saved':str(dst.relative_to(base)),'sha256':digest,'bytes':st.st_size})
 print('Preserved',root,flush=True)
with open(audit/'inventory.csv','w') as f:
 w=csv.writer(f);w.writerow(['path','device','inode','links','bytes','allocated_bytes','mtime_ns','type','target']);w.writerows(rows)
(audit/'preserved.json').write_text(json.dumps(preserved,indent=2))
print(json.dumps({'files':len(rows),'preserved':len(preserved),'preserved_bytes':sum(p['bytes'] for p in preserved)}),flush=True)
