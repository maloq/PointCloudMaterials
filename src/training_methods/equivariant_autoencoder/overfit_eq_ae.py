import os
import sys
from datetime import datetime

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import wandb

sys.path.append(os.getcwd())
try:
    from src.utils.logging_config import setup_logging
    from src.training_methods.equivariant_autoencoder.eq_ae_module import EquivariantAutoencoder
    from src.data_utils.data_module import (
        RealPointCloudDataModule,
        SyntheticPointCloudDataModule,
    )
except ImportError:
    # If running from src/training_methods/equivariant_autoencoder/ substitute path
    sys.path.append(os.path.join(os.getcwd(), '..', '..', '..'))
    from src.utils.logging_config import setup_logging
    from src.training_methods.equivariant_autoencoder.eq_ae_module import EquivariantAutoencoder
    from src.data_utils.data_module import (
        RealPointCloudDataModule,
        SyntheticPointCloudDataModule,
    )

torch.set_float32_matmul_precision('high')

logger = setup_logging()


@rank_zero_only
def get_rundir_name() -> str:
    now = datetime.now()
    return str(f"output/overfit/{now:%Y-%m-%d}/{now:%H-%M-%S}")


@rank_zero_only
def init_wandb(cfg: DictConfig, run_dir: str):
    os.environ['WANDB_MODE'] = cfg.wandb_mode
    os.environ['WANDB_DIR'] = 'output/wandb'
    os.environ['WANDB_CONFIG_DIR'] = 'output/wandb'
    os.environ['WANDB_CACHE_DIR'] = 'output/wandb'
    wandb.init(project='PointCloudMaterials', name=f"OVERFIT_{cfg.experiment_name}")
    return WandbLogger(
        save_dir=os.path.join(os.getcwd(), run_dir),
        project=cfg.project_name,
        name=f"OVERFIT_{cfg.experiment_name}",
        log_model=False,
    )


def apply_overfit_overrides(cfg: DictConfig) -> None:
    try:
        OmegaConf.set_readonly(cfg, False)
        OmegaConf.set_struct(cfg, False)
    except Exception:
        pass

    overrides = []

    cfg.precision = "32-true"
    overrides.append("precision=32-true")

    if hasattr(cfg, "max_samples"):
        cfg.max_samples = 3
        overrides.append("max_samples=1")
    if hasattr(cfg, "batch_size"):
        cfg.batch_size = 3
        overrides.append("batch_size=1")
    if hasattr(cfg, "num_workers"):
        cfg.num_workers = 1
        overrides.append("num_workers=1")
    if hasattr(cfg, "kl_latent_loss_scale"):
        cfg.kl_latent_loss_scale = 0.0
        overrides.append("kl_latent_loss_scale=0.0")
    if hasattr(cfg, "decay_rate"):
        cfg.decay_rate = 0.0
        overrides.append("decay_rate=0.0")
    if hasattr(cfg, "gradient_clip_val"):
        cfg.gradient_clip_val = 0.0
        overrides.append("gradient_clip_val=0.0")

    enc_cfg = getattr(cfg, "encoder", None)
    enc_kwargs = getattr(enc_cfg, "kwargs", None) if enc_cfg is not None else None
    if enc_kwargs is not None:
        if hasattr(enc_kwargs, "use_batchnorm"):
            enc_kwargs.use_batchnorm = False
            overrides.append("encoder.kwargs.use_batchnorm=False")
        if hasattr(enc_kwargs, "dropout_rate"):
            enc_kwargs.dropout_rate = 0.0
            overrides.append("encoder.kwargs.dropout_rate=0.0")

    # Increase learning rate for faster overfitting
    if hasattr(cfg, "learning_rate"):
        cfg.learning_rate = 0.005
        overrides.append("learning_rate=0.005")

    data_cfg = getattr(cfg, "data", None)
    aug_cfg = getattr(data_cfg, "augmentation", None) if data_cfg is not None else None
    if aug_cfg is not None:
        for name in ("rotation_scale", "noise_scale", "jitter_scale", "scaling_range"):
            if hasattr(aug_cfg, name):
                setattr(aug_cfg, name, 0.0)
                overrides.append(f"data.augmentation.{name}=0.0")
    
    # Increase displacement scale for better reconstruction
    dec_cfg = getattr(cfg, "decoder", None)
    dec_kwargs = getattr(dec_cfg, "kwargs", None) if dec_cfg is not None else None
    if dec_kwargs is not None:
        if hasattr(dec_kwargs, "use_batchnorm"):
            dec_kwargs.use_batchnorm = False
            overrides.append("decoder.kwargs.use_batchnorm=False")
        if hasattr(dec_kwargs, "disp_scale"):
            dec_kwargs.disp_scale = 0.5
            overrides.append("decoder.kwargs.disp_scale=0.5")
        if hasattr(dec_kwargs, "final_disp_scale"):
            dec_kwargs.final_disp_scale = 0.5
            overrides.append("decoder.kwargs.final_disp_scale=0.5")
        if hasattr(dec_kwargs, "feature_clamp_max"):
            dec_kwargs.feature_clamp_max = 500.0
            overrides.append("decoder.kwargs.feature_clamp_max=500.0")

    loss_params = getattr(cfg, "loss_params", None)
    chamfer_cfg = getattr(loss_params, "chamfer", None) if loss_params is not None else None

    if overrides:
        logger.print("Overfit overrides: " + ", ".join(overrides))


def freeze_batchnorm_for_overfit(model: torch.nn.Module) -> int:
    """
    Set all BatchNorm layers to eval mode to use running statistics.
    This prevents issues with tiny batch sizes during overfitting.
    Returns the number of frozen BatchNorm layers.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.eval()
            # Freeze the parameters too
            module.weight.requires_grad = False
            module.bias.requires_grad = False
            count += 1
    return count


class BatchNormEvalCallback(pl.Callback):
    """
    Callback that ensures all BatchNorm layers stay in eval mode during training.
    This is necessary for overfitting with tiny batch sizes.
    """
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        for module in pl_module.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                module.eval()


def train_model(cfg: DictConfig, model_class, run_dir: str | None = None, devices=None):
    logger.print(f"Starting OVERFIT run in {os.getcwd()}")

    if run_dir is None:
        run_dir = get_rundir_name()

    apply_overfit_overrides(cfg)

    wandb_logger = init_wandb(cfg, run_dir)

    if cfg.data.kind == "synthetic":
        dm = SyntheticPointCloudDataModule(cfg)
    else:
        dm = RealPointCloudDataModule(cfg)
    model = model_class(cfg)
    
    # Freeze BatchNorm layers to avoid issues with tiny batch sizes
    num_frozen = freeze_batchnorm_for_overfit(model)
    logger.print(f"Frozen {num_frozen} BatchNorm layers for overfitting")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    bn_callback = BatchNormEvalCallback()
    callbacks = [lr_monitor, bn_callback]
    logger.print("Curriculum learning DISABLED for overfitting")

    if devices is None:
        if isinstance(cfg.devices, ListConfig):
            devices = list(cfg.devices)
        elif isinstance(cfg.devices, (list, tuple)):
            devices = list(cfg.devices) if len(cfg.devices) > 0 else [0]
        else:
            devices = [0]

    ddp_strategy = None
    if cfg.gpu and len(devices) > 1 and getattr(cfg, 'ddp_find_unused_parameters', False):
        ddp_strategy = DDPStrategy(find_unused_parameters=True)

    overfit_epochs = 500  # More epochs for proper overfitting

    trainer_kwargs = dict(
        default_root_dir=run_dir,
        max_epochs=overfit_epochs,
        accelerator='gpu' if cfg.gpu else 'cpu',
        devices=devices,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        precision=cfg.precision,
        benchmark=True,
        check_val_every_n_epoch=1,
        overfit_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )

    if hasattr(cfg, 'gradient_clip_val'):
        trainer_kwargs['gradient_clip_val'] = cfg.gradient_clip_val
        trainer_kwargs['gradient_clip_algorithm'] = 'norm'

    if ddp_strategy is not None:
        trainer_kwargs["strategy"] = ddp_strategy

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, dm)

    return trainer, model, dm


def train(cfg: DictConfig):
    return train_model(cfg, EquivariantAutoencoder)


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), 'configs'), config_name='spd_vn_equivariant.yaml')
def main(cfg: DictConfig):
    train(cfg)


if __name__ == '__main__':
    sys.argv.append('hydra.run.dir=output/overfit/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
