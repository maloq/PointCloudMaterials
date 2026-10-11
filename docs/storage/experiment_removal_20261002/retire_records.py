import json,shutil
from pathlib import Path
base=Path('/home/infres/vmorozov/PointCloudMaterials'); audit=base/'docs/storage/experiment_removal_20261002'
assert (audit/'receipt.json').exists()
roots=[Path(p) for p in json.load(open(audit/'targets.json'))]
path=base/'configs/datasets.json'; shutil.copy2(path,audit/'registry_before.json'); data=json.loads(path.read_text())
storage={'output':base/'output','analysis':Path('/work/PERSO/vmorozov/analysis'),'training_storage':Path('/store/PERSO/vmorozov/experiments'),'archive':Path('/store/PERSO/vmorozov'),'repo':base}
changed=[]
for key,v in data['datasets'].items():
 root=storage.get(v.get('root'))
 if root is None:continue
 p=root/v.get('path','')
 if not any(p==r or r in p.parents for r in roots):continue
 v['state']='payload_deleted'; m=v.setdefault('metadata',{});m['classification']='retired';m['role']='provenance_only';m['retirement_record']='docs/storage/experiment_removal_20261002/receipt.json';m['retirement_note']='Experiment folder deleted at user request on 2026-10-02. Reports, tables, plots and text provenance preserved under docs/storage/experiment_removal_20261002/results; mapping and checksums in preserved.json.';changed.append(key)
path.write_text(json.dumps(data,indent=2)+'\n');(audit/'retired_datasets.json').write_text(json.dumps(changed,indent=2));print(changed)
r=json.load(open(audit/'receipt.json'))
doc=base/'docs/data_storage.md';text=doc.read_text();note=f"\nThe user-authorized experiment cleanup on 2026-10-02 removed the 13 reviewed\nresult folders, including the remaining STORE embedding-forecast collections.\nSaved {r['preserved_files']:,} report/table/plot/provenance files under\n[the results archive](storage/experiment_removal_20261002/results/), with original\npaths and SHA-256 checksums in [the preservation index](storage/experiment_removal_20261002/preserved.json).\nNet file allocation released after preserving results: {r['net_released_file_allocated_bytes']/1e9:.3f} GB.\nCheckpoints and binary arrays in those folders were deleted; historical references\nto them no longer imply availability. See [the receipt](storage/experiment_removal_20261002/receipt.json).\n\n"
doc.write_text(text.replace('# Data and result storage\n','# Data and result storage\n'+note,1))
