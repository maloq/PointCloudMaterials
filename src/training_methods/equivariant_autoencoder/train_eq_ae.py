import os
import sys
import traceback
import hydra
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.rank_zero import rank_zero_only

sys.path.append(os.getcwd())
from src.utils.logging_config import setup_logging
from src.training_methods.equivariant_autoencoder.eq_ae_module import EquivariantAutoencoder
from src.training_methods.spd.train_spd import train_model
from src.training_methods.equivariant_autoencoder.predict_and_visualize import (
    run_post_training_analysis,
)

torch.set_float32_matmul_precision('high')
logger = setup_logging()


@rank_zero_only
def run_post_training_analysis_safe(checkpoint_path: str, output_dir: str, cuda_device: int = 0, cfg: DictConfig | None = None):
    """Run post-training analysis with error handling."""
    try:
        logger.print("\n" + "=" * 60)
        logger.print("Starting Equivariant AE analysis...")
        logger.print("=" * 60)

        run_post_training_analysis(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            cuda_device=cuda_device,
            cfg=cfg,
        )

        logger.print("Post-training analysis completed successfully!")
    except Exception as e:
        logger.print(f"\nWarning: Post-training analysis failed with error: {e}")
        logger.print("Training completed successfully, but analysis could not be run.")
        traceback.print_exc()


def train(cfg: DictConfig, run_analysis: bool = True):
    """Equivariant Autoencoder-specific training function."""
    trainer, model, dm, checkpoint_callbacks = train_model(cfg, EquivariantAutoencoder)

    if run_analysis:
        best_ckpt = checkpoint_callbacks[0].best_model_path if checkpoint_callbacks else ""
        if best_ckpt and os.path.exists(best_ckpt):
            output_dir = os.path.join(os.path.dirname(best_ckpt), "analysis")

            if isinstance(cfg.devices, ListConfig):
                cuda_device = list(cfg.devices)[0] if cfg.devices else 0
            elif isinstance(cfg.devices, (list, tuple)):
                cuda_device = cfg.devices[0] if cfg.devices else 0
            else:
                cuda_device = 0

            run_post_training_analysis_safe(best_ckpt, output_dir, cuda_device, cfg)
        else:
            logger.print("Warning: No best checkpoint found, skipping post-training analysis")

    return trainer, model, dm, checkpoint_callbacks


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), 'configs'), config_name='eq_ae_vn.yaml')
def main(cfg: DictConfig):
    train(cfg)

if __name__ == '__main__':
    if not any(arg.startswith("hydra.run.dir=") for arg in sys.argv):
        sys.argv.append('hydra.run.dir=output/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
