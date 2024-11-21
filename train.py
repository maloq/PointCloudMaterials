import sys,os
sys.path.append(os.getcwd())
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from src.cls.train_classification import train_classification
from datetime import datetime
    
def get_rundir_name() -> str:
    now = datetime.now()
    return str(f'output/{now:%Y-%m-%d}/{now:%H-%M-%S}')

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="train_cls")
def main(cfg: DictConfig):    
    train_classification(cfg)
    logging.info('Train finished!')


if __name__ == "__main__":
    experiment_name = 'latest'
    rundir_name = get_rundir_name()
    sys.argv.append(f'hydra.run.dir={rundir_name}')
    main()