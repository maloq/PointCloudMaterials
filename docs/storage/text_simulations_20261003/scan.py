from pathlib import Path
import os,csv,json,datetime
roots=['/work/PERSO/vmorozov/simulations','/store/PERSO/vmorozov/simulations','/scratch/PERSO/vmorozov/PointCloudMaterials/simulations','/work/PERSO/vmorozov/datasets','/store/PERSO/vmorozov/projects/PointCloudMaterials-20260913T174741Z/output/synthetic_data','/home/infres/vmorozov/PointCloudMaterials/output','/store/PERSO/vmorozov/experiments','/store/PERSO/vmorozov/relaxed_encoder','/store/PERSO/vmorozov/relaxed_tda','/store/PERSO/vmorozov/birth_prediction','/store/PERSO/vmorozov/liquid-predictability','/home/ids/vmorozov/training-cache','/store/PERSO/vmorozov/simulation-launches']
out=Path('/home/infres/vmorozov/PointCloudMaterials/docs/storage/text_simulations_20261003');out.mkdir(parents=True,exist_ok=True)
rows=[]
for root in roots:
 for d,dirs,files in os.walk(root,followlinks=False):
  dirs[:]=[v for v in dirs if v not in ['code','.git','__pycache__']]
  for name in files:
   p=Path(d)/name
   if p.is_symlink() or p.suffix.lower() not in ['.lammpstrj','.dump','.xyz','.extxyz','.pos']:continue
   st=p.stat()
   with p.open('rb') as f: header=f.read(1024).decode(errors='replace')
   rows.append({'path':str(p),'bytes':st.st_size,'allocated_bytes':st.st_blocks*512,'type':'LAMMPS dump' if header.startswith('ITEM: TIMESTEP') else p.suffix,'columns':next((l for l in header.splitlines() if l.startswith('ITEM: ATOMS')),''),'classification':'archived snapshot' if '/datasets/Ta/' in str(p) or '/initial_configurations/' in str(p) or '/inherent_configurations/' in str(p) else 'needs review','mtime_utc':datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat()})
 print('scanned',root,flush=True)
rows.sort(key=lambda x:x['allocated_bytes'],reverse=True)
with open(out/'files.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'summary.json').write_text(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':len(rows),'allocated_bytes':sum(r['allocated_bytes'] for r in rows),'roots':roots,'largest':rows[:25]},indent=2))
print(json.dumps({'files':len(rows),'GB':sum(r['allocated_bytes'] for r in rows)/1e9,'largest':rows[:8]}),flush=True)
