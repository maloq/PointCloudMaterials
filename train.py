import sys,os
sys.path.append(os.getcwd())
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from src.cls.train_classification import get_rundir_name
from src.cls.train_classification import train_classification


print(os.environ.get('WANDB_API_KEY'))
@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="Al_classification")
def main(cfg: DictConfig):    
    train_classification(cfg)
    logger.print
('Train finished!')


if __name__ == "__main__":
    experiment_name = 'latest'
    rundir_name = get_rundir_name()
    sys.argv.append(f'hydra.run.dir={rundir_name}')
    main()