import os,json,subprocess,concurrent.futures,re
from pathlib import Path
out=Path('/home/infres/vmorozov/PointCloudMaterials/docs/storage/ta_simulation_removal_20261003')
script='''import os,json
roots=json.load(open('/home/infres/vmorozov/PointCloudMaterials/docs/storage/ta_simulation_removal_20261003/targets.json'))
hits=[]
for pid in os.listdir('/proc'):
 if not pid.isdigit():continue
 p='/proc/'+pid
 try:
  if os.stat(p).st_uid!=os.getuid():continue
  cmd=open(p+'/cmdline','rb').read().replace(b'\\0',b' ').decode(errors='replace')
  refs=[]
  for sub in ['cwd']+['fd/'+f for f in os.listdir(p+'/fd')]:
   try:refs.append(os.readlink(p+'/'+sub))
   except OSError:pass
  try:refs.extend(open(p+'/maps').read().splitlines())
  except OSError:pass
  if any(r in cmd or any(r in v for v in refs) for r in roots):hits.append(dict(pid=pid,cmd=cmd,refs=[v for v in refs if any(r in v for r in roots)]))
 except OSError:pass
print(json.dumps(hits))
'''
jobtext=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-t','R','-o','%i %N'],text=True)
jobs={}
for line in jobtext.splitlines():
 job,node=line.split();jobs.setdefault(node,job)
def check(job):
 actual=re.search(r'JobId=(\d+)',subprocess.check_output(['scontrol','show','job',job],text=True)).group(1)
 r=subprocess.run(['srun','--jobid='+actual,'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','python','-'],input=script,text=True,capture_output=True,timeout=45)
 (out/('processes-'+job+'.json')).write_text(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)))
 if r.returncode:raise RuntimeError((job,r.stderr))
 hits=json.loads(r.stdout)
 if hits:raise RuntimeError((job,hits))
 return job,'no readers/writers'
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
 for r in ex.map(check,jobs.values()):print(r)
