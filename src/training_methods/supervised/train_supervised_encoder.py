import os
import sys
import hydra
import torch
from datetime import datetime
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only

sys.path.append(os.getcwd())
from src.utils.logging_config import setup_logging
from src.training_methods.spd.supervised_encoder_module import SupervisedEncoder
from src.training_methods.spd.train_spd import train_model

torch.set_float32_matmul_precision('high')

logger = setup_logging()


@rank_zero_only
def get_rundir_name() -> str:
    now = datetime.now()
    return str(f"output/supervised_encoder/{now:%Y-%m-%d}/{now:%H-%M-%S}")



def get_checkpoint_filename(cfg: DictConfig) -> str:
    """Generate checkpoint filename with hyperparameters."""
    encoder_name = cfg.encoder.name
    latent_size = cfg.latent_size
    batch_size = cfg.batch_size
    lr = cfg.learning_rate
    rotation_mode = cfg.rotation_mode

    filename = (f"supervised_encoder_{encoder_name}_"
               f"l{latent_size}_bs{batch_size}_lr{lr:.4f}_"
               f"rot{rotation_mode}_epoch{{epoch:02d}}")
    return filename


def train(cfg: DictConfig):
    logger.print(f"Starting supervised encoder pretraining in {os.getcwd()}")
    run_dir = get_rundir_name()

    # Checkpoint callback with hyperparameters in filename
    checkpoint_filename = get_checkpoint_filename(cfg)
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        monitor='val/accuracy',
        filename=checkpoint_filename,
        save_top_k=3,
        mode='max',  # Maximize validation accuracy
    )

    # Also save best loss checkpoint
    checkpoint_callback_loss = ModelCheckpoint(
        dirpath=run_dir,
        monitor='val/loss',
        filename=checkpoint_filename.replace('epoch', 'loss_epoch'),
        save_top_k=1,
        mode='min',
    )

    # Train using shared training logic
    trainer, model, dm, checkpoint_cbs = train_model(
        cfg,
        SupervisedEncoder,
        run_dir=run_dir,
        checkpoint_callbacks=[checkpoint_callback, checkpoint_callback_loss],
        devices=[0],
        run_test=False  # Supervised encoder doesn't run test phase
    )

    logger.print(f"Training completed. Checkpoints saved to {run_dir}")
    logger.print(f"Best checkpoint (accuracy): {checkpoint_callback.best_model_path}")
    logger.print(f"Best checkpoint (loss): {checkpoint_callback_loss.best_model_path}")


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), 'configs'), config_name='supervised_encoder_synth')
def main(cfg: DictConfig):
    train(cfg)


if __name__ == '__main__':
    sys.argv.append('hydra.run.dir=output/supervised_encoder/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
