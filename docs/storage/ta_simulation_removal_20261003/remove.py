import json,csv,hashlib,shutil,os,collections,concurrent.futures,datetime
from pathlib import Path
base=Path('/home/infres/vmorozov/PointCloudMaterials');audit=base/'docs/storage/ta_simulation_removal_20261003'
roots=[Path(p) for p in json.load(open(audit/'targets.json'))];protected=[Path(p) for p in json.load(open(audit/'protected.json'))]
def files(root):
 for directory,dirs,items in os.walk(root,followlinks=False):
  for name in dirs+items:
   p=Path(directory)/name
   if p.is_symlink() or p.is_file():yield p
protected_stat={}
for root in protected:
 for p in files(root):
  s=p.lstat();protected_stat[str(p)]=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
(audit/'protected_files.json').write_text(json.dumps(protected_stat,indent=2))
rows=[];saved=[]
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for i,root in enumerate(roots):
 if root.is_symlink() or not root.is_dir():raise ValueError('Physical deletion root required: '+str(root))
 if any(root==p or root in p.parents or p in root.parents for p in protected):raise ValueError('Protected overlap')
 for p in files(root):
  s=p.lstat();kind='link' if p.is_symlink() else 'file'
  rows.append(dict(path=str(p),type=kind,device=s.st_dev,inode=s.st_ino,links=s.st_nlink,bytes=s.st_size,allocated_bytes=s.st_blocks*512,mtime_ns=s.st_mtime_ns))
  if kind=='file' and (p.suffix.lower() in ['.md','.csv','.tsv','.json','.txt','.log','.yaml','.yml'] or p.name.endswith('.in.lammps') or p.name in ['in.lammps','log.lammps','lammps.log']):
   target=audit/'results'/f'{i:02d}-{root.name}'/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
   digest=sha(p)
   if sha(target)!=digest:raise RuntimeError('Preservation checksum differs '+str(p))
   saved.append(dict(source=str(p),saved=str(target),sha256=digest,bytes=s.st_size))
with open(audit/'inventory.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(audit/'preserved.json').write_text(json.dumps(saved,indent=2))
for r in rows:
 s=Path(r['path']).lstat()
 if (s.st_ino,s.st_size,s.st_mtime_ns)!=(r['inode'],r['bytes'],r['mtime_ns']):raise RuntimeError('Changed during preservation '+r['path'])
counts=collections.Counter((r['device'],r['inode']) for r in rows if r['type']=='file');unique={(r['device'],r['inode']):r for r in rows if r['type']=='file'}
released=sum(r['allocated_bytes'] for key,r in unique.items() if r['links']==counts[key])
aliases=[]
for i in range(6):
 p=Path(f'/scratch/PERSO/vmorozov/PointCloudMaterials/simulations/ta-shooting-20260926-parent{i:02d}')
 if p.is_symlink():
  if p.resolve()!=roots[i+1]:raise RuntimeError('Unexpected alias '+str(p))
  aliases.append(str(p))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
 for r in ex.map(lambda p:(shutil.rmtree(p),str(p))[1],roots):print('Deleted',r,flush=True)
for p in aliases:Path(p).unlink()
assert all(not p.exists() for p in roots)
for p,expected in protected_stat.items():
 s=Path(p).lstat()
 if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]!=expected:raise RuntimeError('Protected file changed: '+p)
held=sum(Path(r['saved']).stat().st_blocks*512 for r in saved)
receipt=dict(state='complete',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 deleted_roots=[str(p) for p in roots],deleted_aliases=aliases,deleted_production_trajectories=29,
 deleted_files_and_symlinks=len(rows),preserved_result_files=len(saved),preserved_result_allocated_bytes=held,
 released_file_allocated_bytes=released,net_released_file_allocated_bytes=released-held,
 protected_roots=[str(p) for p in protected],protected_file_count=len(protected_stat),protected_stat_verification='passed',
 accounting='Unique file inodes whose final links were deleted; preserved report/provenance allocations subtracted, directory blocks excluded.')
(audit/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
