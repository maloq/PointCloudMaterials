"""Separate frozen-representation quality from MD-to-static readout transfer."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
import torch
from experiments.smooth_temporal_encoder_20260905.evaluate import structure_fit
from experiments.smooth_temporal_encoder_20260905.prepare import write_json


def metrics(y, predicted):
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, labels=[0,1,2,3], zero_division=0)
    return dict(macro_f1=float(f1.mean()), common_012_macro_f1=float(f1[:3].mean()),
                accuracy=float((y==predicted).mean()), precision=precision.tolist(), recall=recall.tolist(),
                f1=f1.tolist(), counts=support.tolist(), predicted_counts=np.bincount(predicted,minlength=4).tolist(),
                confusion=confusion_matrix(y,predicted,labels=[0,1,2,3]).tolist())


@torch.inference_mode()
def predict(probe, values):
    probe = {k:v.cuda() for k,v in probe.items()}
    results=[]
    for batch in values.split(8192):
        x=torch.cat(((batch-probe['mean'])/probe['std'],batch.new_ones(len(batch),1)),1).double()
        results.append(probe['classes'][(x@probe['coefficients']).argmax(1)].cpu().numpy())
    return np.concatenate(results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    pilot,out=ROOT/cfg['pilot'],ROOT/cfg['output']
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    md={s:dict(np.load(pilot/'embeddings'/f'{s}_metadata.npz')) for s in ('train','val','test')}
    static=dict(np.load(pilot/'full_static_Al/metadata.npz'))
    masks={s:np.isin(static['source_ids'],cfg[f'static_probe_{s}_sources']) for s in ('train','val','test')}
    result={'protocol':cfg,'models':{}}
    for name in ('density_pca','power_mlp','mace_product','geoframe_vicreg'):
        z={s:torch.tensor(np.load(pilot/'embeddings'/f'{name}_{s}.npy'),device='cuda') for s in md}
        full=torch.tensor(np.load(pilot/'full_static_Al'/f'{name}.npy'),device='cuda')
        original=torch.load(pilot/'embeddings'/f'{name}_structure_probe.pt',weights_only=True)
        pm=predict(original,z['test'])
        ps=np.load(pilot/'full_static_Al'/f'{name}.predicted_ptm.npy')
        row={'original_md':metrics(md['test']['labels'],pm),
             'original_md_Al':metrics(md['test']['labels'][md['test']['material']==0],pm[md['test']['material']==0]),
             'original_full_static':metrics(static['ptm_labels'],ps),
             'original_static_test':metrics(static['ptm_labels'][masks['test']],ps[masks['test']]),
             'original_by_source':{str(i):metrics(static['ptm_labels'][static['source_ids']==i],ps[static['source_ids']==i]) for i in range(6)}}
        selection,probe,p=structure_fit(z['train'][md['train']['material']==0],md['train']['labels'][md['train']['material']==0],
            z['val'][md['val']['material']==0],md['val']['labels'][md['val']['material']==0],full,static['ptm_labels'])
        row['Al_only_MD_probe']={'selection':selection,'full_static':metrics(static['ptm_labels'],p),
            'md_Al_test':metrics(md['test']['labels'][md['test']['material']==0],predict(probe,z['test'][md['test']['material']==0]))}
        torch.save(probe,out/f'{name}.Al_only_MD_probe.pt')
        selection,probe,p=structure_fit(full[masks['train']],static['ptm_labels'][masks['train']],
            full[masks['val']],static['ptm_labels'][masks['val']],full[masks['test']],static['ptm_labels'][masks['test']])
        row['static_adapted_probe']={'selection':selection,'static_test':metrics(static['ptm_labels'][masks['test']],p)}
        torch.save(probe,out/f'{name}.static_adapted_probe.pt')
        result['models'][name]=row
        write_json(out/'readouts.json',result)
        print(name,'transferred',row['original_static_test']['macro_f1'],'adapted',row['static_adapted_probe']['static_test']['macro_f1'],
              'Al-only',row['Al_only_MD_probe']['full_static']['macro_f1'],flush=True)
        del z,full


if __name__=='__main__':
    main()
