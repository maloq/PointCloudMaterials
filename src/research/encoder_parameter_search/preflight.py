"""Disposable real-trainer and native-gradient checks before freezing the queue."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from src.research.encoder_screen.common import sha,write
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule
from src.training_methods.trainer import train_model
from src.research.structural_state.common import Study
from src.research.structural_state.runtime import train
from .geoframe_fit import Record
from .queue import read


def run(path):
    c=read(path);root=Path(c['output'])/'technical/preflight';root.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-m','pytest','tests/test_encoder_parameter_search.py',
        'tests/test_structural_state.py','tests/test_structural_state_dynamics.py','-q','--disable-warnings'],check=True)
    # Match architecture initialization explicitly across every loss/head arm.
    signatures={}
    for item in [i for i in c['fits'] if i['family']=='geoframe']:
        cfg=OmegaConf.load(item['training_config']);torch.manual_seed(item['seed']);model=VICRegModule(cfg)
        signature={k:v.detach().cpu().clone() for k,v in model.encoder.state_dict().items()}
        seed=item['seed']
        if seed in signatures:
            for key,value in signature.items():torch.testing.assert_close(value,signatures[seed][key],rtol=0,atol=0)
        else:signatures[seed]=signature
        del model
    for name in ['gf-mlp-cov1-factor1-s123','gf-identity-cov5-factor0-s123','gf-visreg-identity-factor1-s123']:
        item=next(i for i in c['fits'] if i['name']==name);cfg=OmegaConf.load(item['training_config'])
        cfg.experiment_name='disposable-'+name;cfg.wandb_mode='disabled';cfg.check_val_every_n_epoch=160
        folder=root/name;folder.mkdir(exist_ok=True)
        record=Record(folder,1,123,time.time()+3600)
        checkpoint=ModelCheckpoint(dirpath=folder,monitor=None,save_top_k=-1,filename='epoch-{epoch:03d}',
            auto_insert_metric_name=False,every_n_epochs=1,save_on_train_epoch_end=True)
        trainer,model,dm,_=train_model(cfg,VICRegModule,run_dir=str(folder),checkpoint_callbacks=[record,checkpoint],run_test=False)
        assert (folder/'epoch-000.ckpt').exists()
        del trainer,model,dm;torch.cuda.empty_cache()
    s=Study('configs/encoder_parameter_search/mace-s20260923.json');s.identity='disposable-parameter-search-preflight'
    for arm in s.config['arms']:
        folder=root/arm['name'];train(s,arm['name'],stop_after=3,directory=folder)
        saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        assert saved['step']==3 and all(torch.isfinite(v).all() for v in saved['model'].values())
        rates=[g['lr'] for g in saved['optimizer']['param_groups']]
        assert abs(rates[0]/rates[1]-arm['encoder_lr']/s.config['training']['head_lr'])<1e-8
        del saved;torch.cuda.empty_cache()
    files=list(Path('src/research/encoder_parameter_search').glob('*.py'))+[
        Path('src/research/structural_state/common.py'),Path('src/research/structural_state/runtime.py'),Path('src/research/structural_state/queue.py'),
        Path('src/training_methods/contrastive_learning/vicreg_module.py')]
    files+=list(Path('configs/encoder_parameter_search').glob('*'))
    write(root.parent/'preflight.json',dict(passed=True,campaign_sha256=sha(path),
        files={str(p):sha(p) for p in files},geoframe_full_passes_disposable=3,mace_updates_per_arm_disposable=3,
        initial_encoder_weights_matched_within_seed=True,automatic_and_manual_optimization_checked=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);run(p.parse_args().config)
