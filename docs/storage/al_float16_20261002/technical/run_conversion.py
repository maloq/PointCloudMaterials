"""One-off execution receipt driver for the reviewed Al archival plan."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import datetime,json,os,subprocess,traceback,time
ROOT=Path('/home/infres/vmorozov/PointCloudMaterials')
BASE=ROOT/'docs/storage/al_float16_20261002'
PYTHON='/home/infres/vmorozov/miniconda3/envs/pointnet-torch214/bin/python'
plan=json.loads((BASE/'plan.json').read_text())

def convert(item):
 source=Path(item['path']);target=Path(item['output'])
 tag=__import__('hashlib').sha256(str(source).encode()).hexdigest()[:16]
 log=BASE/'technical'/f'{tag}.log'
 result=dict(item)
 try:
  in_flight={'/store/PERSO/vmorozov/simulations/al-main-010ps-20260927-source0876-failed/trajectory.lammpstrj':3062650,'/store/PERSO/vmorozov/simulations/al-main-010ps-20260927-source0882-failed/trajectory.lammpstrj':3063677}
  if str(source) in in_flight:
   while Path('/proc') .joinpath(str(in_flight[str(source)])).exists():time.sleep(5)
  status=source.parent/'status.json'
  if status.exists():
   s=json.loads(status.read_text())
   if s.get('state')=='running' and s.get('slurm_job_id'):
    probe=subprocess.run(['scontrol','show','job',str(s['slurm_job_id'])],capture_output=True,text=True)
    result['job_status_check']=probe.stdout+probe.stderr
    if any('JobState='+state in probe.stdout for state in ['RUNNING','PENDING','COMPLETING','CONFIGURING','SUSPENDED']):
     result['state']='deferred_active';return result
    if probe.returncode and 'Invalid job id' not in probe.stderr:
     raise RuntimeError('Cannot establish whether source job is inactive: '+probe.stderr)
  with log.open('w') as f:
   run=subprocess.run([PYTHON,'scripts/convert_trajectory.py','offline-dump',str(source),'--output',str(target),'--delete-source'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
  if run.returncode:raise RuntimeError(f'Converter exit {run.returncode}; see {log}')
  receipt=json.loads((target/'conversion.json').read_text())
  if source.exists() or not receipt['source_deleted']:raise RuntimeError('Original still present')
  result.update(state='complete',receipt=str(target/'conversion.json'),source_allocated_bytes=receipt['source_allocated_bytes'],binary_allocated_bytes=sum(f.stat().st_blocks*512 for f in target.iterdir() if f.is_file()),frames=receipt['frame_count'],incomplete_tail=receipt['incomplete_tail'])
 except Exception as e:result.update(state='failed_original_preserved',error=str(e))
 return result

results=[]
with ThreadPoolExecutor(max_workers=2) as pool,(BASE/'actions.jsonl').open('a',buffering=1) as log:
 futures=[pool.submit(convert,p) for p in plan]
 for f in as_completed(futures):
  r=f.result();results.append(r);log.write(json.dumps(r)+'\n')
  summary=dict(state='running',total=len(plan),finished=len(results),completed=sum(x['state']=='complete' for x in results),failed=sum(x['state']=='failed_original_preserved' for x in results),deferred=sum(x['state']=='deferred_active' for x in results),deleted_allocated_bytes=sum(x.get('source_allocated_bytes',0) for x in results),binary_allocated_bytes=sum(x.get('binary_allocated_bytes',0) for x in results),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  temp=BASE/'progress.tmp';temp.write_text(json.dumps(summary,indent=2)+'\n');temp.replace(BASE/'progress.json')
summary['state']='complete' if not summary['failed'] and not summary['deferred'] else 'finished_with_exceptions'
(BASE/'progress.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
