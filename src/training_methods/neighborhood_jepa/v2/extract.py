"""Execute via run_path under the checkpoint's frozen producer, never a shape adapter."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.data.structural_pretraining.batches import collate,move
from src.data.structural_pretraining.support import REFERENCE_RADIUS,support_weights
from src.data.structural_pretraining.prepare import file_hash,save_json
from src.training_methods.shared_pretraining.compilation import compile_encoder


@torch.no_grad()
def run(record_path):
    record = json.loads(Path(record_path).read_text())
    root = Path(record['directory'])
    torch.set_num_threads(1)
    if file_hash(Path(record['checkpoint'])) != record['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after crystallization assay freeze')
    saved = torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
    kind = record['kind']
    if kind == 'regularization':
        from src.training_methods.neighborhood_jepa.regularization.model import Encoder
        model = Encoder(saved['manifest']['config']['encoder_channels'],saved['spec']['export_norm'])
    elif kind == 'v2_large':
        from src.training_methods.neighborhood_jepa.v2.model import Encoder
        model = Encoder(channels=saved['manifest']['config']['encoder_channels'])
    elif kind == 'v2':
        from src.training_methods.neighborhood_jepa.v2.model import Encoder
        model = Encoder()
    elif kind == 'v1':
        from src.training_methods.neighborhood_jepa.model import NeighborhoodEncoder
        model = NeighborhoodEncoder('mace')
    elif kind in ('mace','gatr'):
        from src.models.encoders.structural import StructuralMACE,StructuralGATr
        model = StructuralMACE() if kind=='mace' else StructuralGATr()
    else:
        raise ValueError(kind)
    model.load_state_dict({k.removeprefix('encoder.'):v for k,v in saved['model'].items() if k.startswith('encoder.')},strict=True)
    model = model.cuda().eval()
    architecture = 'gatr' if kind=='gatr' else 'mace'
    plan = json.loads(Path(record['assay_plan']).read_text())
    cache = Path(record['assay_cache'])
    population = np.load(record['population'])
    source_ids,graph_ids = population['source'],population['graph']
    initialized = False
    start_time = time.monotonic()
    for sid in np.unique(source_ids):
        out = root/'features'/f'{sid}.npy'
        receipt = out.with_suffix('.json')
        if receipt.exists():
            info = json.loads(receipt.read_text())
            if info['checkpoint_sha256']!=record['checkpoint_sha256'] or file_hash(out)!=info['sha256']:
                raise ValueError(f'Feature receipt mismatch: {sid}')
            continue
        ids = np.flatnonzero(source_ids==sid)
        graphs = graph_ids[ids]
        folder = cache/str(sid)
        original = json.loads((folder/'complete.json').read_text())
        if original['identity']!=plan.get('cache_identity',plan['identity']):
            raise ValueError('Crystallization graph release changed')
        names = ('positions','offsets','edges','edge_offsets')
        a = {name:np.load(folder/f'{name}.npy',mmap_mode='r') for name in names}
        def sample(g):
            lo,hi = a['offsets'][g:g+2]
            elo,ehi = a['edge_offsets'][g:g+2]
            x = np.array(a['positions'][lo:hi])
            return dict(positions=x[None],weights=support_weights(x)[None],center=0,
                times=np.array([0.],np.float32),species=1,log_scale=np.log(plan['scale']/REFERENCE_RADIUS),
                physical=np.zeros(85,np.float32),tda=np.zeros(144,np.float32),tda_valid=False,
                edges=np.array(a['edges'][:,elo:ehi],dtype=np.int64))
        def pack(start):
            return collate([sample(g) for g in graphs[start:start+128]],architecture)
        values = []
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(pack,0)
            for start in range(0,len(graphs),128):
                batch = move(pending.result(),'cuda')
                if start+128<len(graphs): pending=pool.submit(pack,start+128)
                if not initialized:
                    compile_encoder(model,batch,'bf16')
                    initialized = True
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    z = model(batch).float()[:,:128]
                if not torch.isfinite(z).all():
                    raise FloatingPointError('Nonfinite frozen encoder export')
                values.append(z.cpu().numpy())
        out.parent.mkdir(parents=True,exist_ok=True)
        temp = out.with_suffix('.building.npy')
        np.save(temp,np.concatenate(values))
        temp.replace(out)
        save_json(receipt,dict(checkpoint_sha256=record['checkpoint_sha256'],sha256=file_hash(out),
            population_sha256=record['population_sha256'],rows=len(graphs)))
        print(json.dumps(dict(source=int(sid),rows=len(graphs),seconds=time.monotonic()-start_time)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--record',required=True)
    run(parser.parse_args().record)
