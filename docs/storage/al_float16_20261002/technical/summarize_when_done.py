from pathlib import Path
import json,time,datetime,os
ROOT=Path('/home/infres/vmorozov/PointCloudMaterials')
BASE=ROOT/'docs/storage/al_float16_20261002'
plans=[BASE,BASE/'additional']
manual=[Path('/work/PERSO/vmorozov/simulations/restart_boundary_audit_20260905/boundary_001/uninterrupted/trajectory.lammpstrj.float16'),Path('/home/ids/vmorozov/training-cache/mace-velocity-all-20260915/converted/91ac83a15d9b-selected-source.lammpstrj.float16')]
def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
 except FileNotFoundError:return False
while True:
 rows=[r for base in plans for r in json.loads((base/'plan.json').read_text())]
 rows += [dict(path=str(p).removesuffix('.float16'),output=str(p)) for p in manual]
 complete=[];remaining=[];seen=set();stored=0;by_partition={}
 for r in rows:
  source=Path(r['path']);target=Path(r['output']);receipt=target/'conversion.json'
  if not source.exists() and receipt.exists():
   proof=json.loads(receipt.read_text())
   if proof['state']!='verified' or not proof['source_deleted']:remaining.append(r);continue
   complete.append(proof)
   partition='IDS' if str(source).startswith('/home/ids/') else 'INFRES' if str(source).startswith('/home/infres/') else str(source).split('/')[1].upper()
   by_partition[partition]=by_partition.get(partition,0)+proof['source_allocated_bytes']
   for p in target.iterdir():
    if p.is_file():
     s=p.stat();key=(s.st_dev,s.st_ino)
     if key not in seen:stored+=s.st_blocks*512;seen.add(key)
  else:remaining.append(r)
 running=any(alive(json.loads((b/'launch.json').read_text())['pid']) for b in plans)
 summary=dict(state='running' if running else ('complete' if not remaining else 'finished_with_exceptions'),planned_files=len(rows),verified_and_deleted=len(complete),remaining_files=len(remaining),original_allocated_bytes_removed=sum(x['source_allocated_bytes'] for x in complete),replacement_allocated_bytes_unique=stored,net_allocated_bytes_reduction=sum(x['source_allocated_bytes'] for x in complete)-stored,original_allocated_bytes_removed_by_partition=by_partition,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),note='Logical source bytes and allocated storage differ. Filesystem-wide free space can change independently. Incomplete terminal bytes are preserved; native restarts are untouched.')
 tmp=BASE/'summary.tmp';tmp.write_text(json.dumps(summary,indent=2)+'\n');tmp.replace(BASE/'summary.json')
 if not running:
  (BASE/'remaining.json').write_text(json.dumps(remaining,indent=2)+'\n')
  break
 time.sleep(30)
