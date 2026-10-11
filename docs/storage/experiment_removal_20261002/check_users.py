import os,json
roots=json.load(open('/home/infres/vmorozov/PointCloudMaterials/docs/storage/experiment_removal_20261002/targets.json'))
hits=[]
for pid in os.listdir('/proc'):
 if not pid.isdigit():continue
 p='/proc/'+pid
 try:
  if os.stat(p).st_uid!=os.getuid():continue
  cmd=open(p+'/cmdline','rb').read().replace(b'\0',b' ').decode(errors='replace')
  refs=[]
  for sub in ['cwd']+['fd/'+f for f in os.listdir(p+'/fd')]:
   try: refs.append(os.readlink(p+'/'+sub))
   except OSError:pass
  try: refs.extend(open(p+'/maps').read().splitlines())
  except OSError:pass
  if any(r in cmd or any(r in v for v in refs) for r in roots): hits.append({'pid':pid,'cmd':cmd,'refs':[v for v in refs if any(r in v for r in roots)]})
 except (OSError,PermissionError):pass
print(json.dumps(hits))
