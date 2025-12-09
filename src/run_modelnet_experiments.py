import argparse
import os
import sys
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

# Add src to path
sys.path.append(os.getcwd())

# from src.data_utils.modelnet_loader import ModelNetDataModule
from src.training_methods.spd.spd_experiments_module import SPDExperimentsModule
torch.set_float32_matmul_precision('high')
from src.utils.visualization import visualize_reconstructions
from src.data_utils.modelnet_fast_loader import ModelNetFastDataModule

def update_config_for_fast_data(cfg):
    """Update config to use fast data path if not already set."""
    if not hasattr(cfg.data, 'fast_data_path'):
        cfg.data.fast_data_path = 'datasets/ModelNet40_fast'
    return cfg

def load_config(experiment_name, n_classes=None):
    base_cfg = OmegaConf.load("configs/experiments/modelnet/base.yaml")
    
    if experiment_name == "baseline":
        exp_cfg = OmegaConf.load("configs/experiments/modelnet/baseline.yaml")
        exp_name = "exp0_baseline"
    elif experiment_name == "axis_conflict":
        exp_cfg = OmegaConf.load("configs/experiments/modelnet/axis_conflict.yaml")
        exp_name = "exp1_axis_conflict"
    elif experiment_name == "symmetry":
        exp_cfg = OmegaConf.load("configs/experiments/modelnet/symmetry.yaml")
        exp_name = "exp2_symmetry"
    elif experiment_name == "capacity":
        if n_classes not in [3, 5, 10, 20]:
             raise ValueError(f"Unsupported n_classes: {n_classes}")
        exp_cfg = OmegaConf.load(f"configs/experiments/modelnet/capacity_{n_classes}cls.yaml")
        exp_name = f"exp3_capacity_{n_classes}cls"
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")
        
    cfg = OmegaConf.merge(base_cfg, exp_cfg)
    return cfg, exp_name

def run_experiment(args):
    cfg, exp_name = load_config(args.experiment, args.n_classes)

    # Dry run overrides
    if args.dry_run:
        cfg.max_epochs = 1
        cfg.batch_size = 4
        cfg.data.num_points = 128 # Faster
        cfg.decoder.kwargs.num_points = 128
        # Limit batches
        limit_train_batches = 2
        limit_val_batches = 2
        exp_name += "_dryrun"
    else:
        limit_train_batches = None
        limit_val_batches = None

    print(f"Running Experiment: {exp_name}")
    print(f"Classes: {cfg.data.classes}")


    cfg = update_config_for_fast_data(cfg)


    print("Using Fast Data Loader")
    dm = ModelNetFastDataModule(cfg)
    
    # Initialize Model
    model = SPDExperimentsModule(cfg)
    
    # Logger
    logger = WandbLogger(project="spd_modelnet_experiments", name=exp_name, offline=not cfg.wandb_online)
    
    # Trainer
    num_devices = 1
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_devices,
        strategy=DDPStrategy(find_unused_parameters=True) if num_devices > 1 else "auto",
        logger=logger,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_val_batches,
        check_val_every_n_epoch=5,
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(monitor="val/loss", mode="min", save_last=True)
        ],
        precision="32-true",
    )
    
    
    trainer.fit(model, datamodule=dm)
    
    # Test
    trainer.test(model, datamodule=dm)

    # Visualize
    print("Generating visualizations...")
    vis_dir = os.path.join("output", "modelnet_visualizations", exp_name)
    try:
        visualize_reconstructions(model, dm, vis_dir, num_instances=10)
    except Exception as e:
        print(f"Failed to generate visualizations: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, 
                        choices=["baseline", "axis_conflict", "symmetry", "capacity"],
                        help="Experiment to run")
    parser.add_argument("--n_classes", type=int, default=3,
                        help="Number of classes for capacity experiment (3, 5, 10, 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run a short dry run for verification")
    
    args = parser.parse_args()
    run_experiment(args)
