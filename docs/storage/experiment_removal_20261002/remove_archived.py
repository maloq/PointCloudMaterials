import os,json,csv,shutil,collections,concurrent.futures,datetime
from pathlib import Path
base=Path('/home/infres/vmorozov/PointCloudMaterials'); audit=base/'docs/storage/experiment_removal_20261002'
roots=[Path(p) for p in json.load(open(audit/'targets.json'))]
rows=list(csv.DictReader(open(audit/'inventory.csv')))
# Fail before any deletion if a source changed after preservation.
for r in rows:
 st=Path(r['path']).lstat()
 if (st.st_ino,st.st_size,st.st_mtime_ns)!=(int(r['inode']),int(r['bytes']),int(r['mtime_ns'])):raise RuntimeError('Changed: '+r['path'])
counts=collections.Counter((r['device'],r['inode']) for r in rows if r['type']=='file')
unique={(r['device'],r['inode']):r for r in rows if r['type']=='file'}
released=sum(int(r['allocated_bytes']) for k,r in unique.items() if counts[k]==int(r['links']))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for root in pool.map(lambda p:(shutil.rmtree(p),str(p))[1],roots): print('Removed',root,flush=True)
assert all(not p.exists() and not p.is_symlink() for p in roots)
saved=json.load(open(audit/'preserved.json'))
allocated=sum((base/p['saved']).stat().st_blocks*512 for p in saved)
receipt={'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'roots':[str(p) for p in roots],'removed_files_and_symlinks':len(rows),'preserved_files':len(saved),'preserved_allocated_bytes':allocated,'released_file_allocated_bytes':released,'net_released_file_allocated_bytes':released-allocated,'accounting':'Unique file inodes whose last link was removed; directory blocks excluded. Preserved result file allocation subtracted; audit metadata excluded.'}
(audit/'receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
