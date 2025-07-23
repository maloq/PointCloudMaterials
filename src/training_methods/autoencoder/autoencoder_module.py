import os
import sys
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F  # Import F for functional operations like mse_loss
from src.loss.reconstruction_loss import *
import src.models.autoencoders.encoders
import src.models.autoencoders.decoders
from src.models.autoencoders.factory import build_model
from src.utils.logging_config import setup_logging  
from src.utils.optimizer_utils import get_optimizers_and_scheduler
from src.data_utils.data_transforms import random_rotate_point_cloud_batch
import wandb
from pytorch_lightning.loggers import WandbLogger
logger = setup_logging()


class PointNetAutoencoder(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        # ------------------------------------------------------------------
        # Build encoder & decoder **separately**
        # ------------------------------------------------------------------
        self.encoder, self.decoder = build_model(cfg)

        # Optional compilation of the individual components
        if cfg.torch_compile:
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)

        self.num_points = cfg.data.num_points
        self.feature_transform_loss_scale = cfg.get('feature_transform_loss_scale', 0.0)
        self.rotation_loss_scale = cfg.get('rotation_loss_scale', 0.0)
        self.l1_latent_loss_scale = cfg.get('l1_latent_loss_scale', 0.0)
        self.repulsion_h = cfg.get('repulsion_h', 0.05)
        self.repulsion_scale = cfg.get('repulsion_scale', 0.1)
        self.kl_latent_loss_scale = cfg.get('kl_latent_loss_scale', 0.0)
        self.latent_noise_std = cfg.get('latent_noise_std', 0.0)
        if self.latent_noise_std > 0:
            logger.print(f"Latent noise enabled with std: {self.latent_noise_std}")

        if cfg.loss == 'CD':
             self.criterion = chamfer_loss
             logger.print(f"Using Chamfer loss")
        elif cfg.loss == 'CD_FTreg':
             self.criterion = chamfer_regularized_encoder_loss
             logger.print(f"Using Chamfer loss with feature transform regularization")
        elif cfg.loss == 'CD_FTreg_Rep':
             self.criterion = chamfer_regularized_encoder_loss_repulsion
             logger.print(f"Using Chamfer loss with feature transform regularization and repulsion loss")
        elif cfg.loss == 'Sinkhorn':
             self.criterion = sinkhorn_loss
             logger.print(f"Using Sinkhorn loss")
        else:
            logger.warning(f"Unknown or basic loss type '{cfg.loss}' specified. Defaulting to basic chamfer_loss.")
            self.criterion = chamfer_loss
            
        logger.print(f"Loss: {self.criterion.__name__}")
        if self.rotation_loss_scale > 0:
             logger.print(f"Rotational Consistency Loss enabled with scale: {self.rotation_loss_scale}")
        if self.l1_latent_loss_scale > 0:
            logger.print(f"L1 Latent Loss enabled with scale: {self.l1_latent_loss_scale}")
        if self.kl_latent_loss_scale > 0:
            logger.print(f"β-VIB (KL) loss enabled with scale: {self.kl_latent_loss_scale}")


    def forward(self, x):
        """
        Performs the forward pass of the autoencoder.

        Args:
            x (torch.Tensor): Input point cloud tensor (B, C, N).

        Returns:
            tuple: A tuple containing:
                - reconstructed_points (torch.Tensor): The reconstructed point cloud (B, N, 3).
                - latent_vector (torch.Tensor): The latent representation (B, latent_dim).
                - trans_feat_list (list): A list of feature transformation matrices.
        """
        # Encode & decode
        latent_vector, _, trans_feat = self.encoder(x)
        reconstructed_points = self.decoder(latent_vector)
        trans_feat_list = [trans_feat] if trans_feat is not None else []
        
        # TODO: check if this is needed
        if reconstructed_points.ndim == 3: 
            if reconstructed_points.shape[1] == 3 and reconstructed_points.shape[2] == self.num_points:
                 reconstructed_points = reconstructed_points.permute(0, 2, 1) # (B, 3, N) -> (B, N, 3)
            elif reconstructed_points.shape[2] == 3 and reconstructed_points.shape[1] == self.num_points:
                 pass # Already (B, N, 3)

        return reconstructed_points, latent_vector, trans_feat_list

    def _prepare_loss_args(self, target_points, predicted_points, trans_feat_list, latent=None):
        """Prepares the arguments dictionary for the loss function."""
        loss_args = {
            'pred': predicted_points.float(),
            'target': target_points.float(),
            'trans_feat_list': trans_feat_list,
            'feature_transform_loss_scale': self.hparams.get('feature_transform_loss_scale', 0.0),
            'repulsion_h': self.hparams.get('repulsion_h', 0.05),
            'repulsion_scale': self.hparams.get('repulsion_scale', 0.1),
            'rotation_loss_scale': self.hparams.get('rotation_loss_scale', 0.0),
            'l1_latent_loss_scale': self.hparams.get('l1_latent_loss_scale', 0.0),
            'kl_latent_loss_scale': self.hparams.get('kl_latent_loss_scale', 0.0)
        }
        
        # Add latent vector for L1 regularization
        if latent is not None:
            loss_args['latent'] = latent
            
        return loss_args

    def _log_losses(self, main_loss, aux_loss_dict, step_type, rotation_loss=None):
        """Logs the computed losses, assuming aux_loss_dict structure."""
        log_prefix = f"{step_type}_"
        total_loss = main_loss

        # Log main loss first
        self.log(f'{log_prefix}main_loss', main_loss, prog_bar=False, sync_dist=True)

        # Handle and log rotation loss if applicable
        if rotation_loss is not None and self.rotation_loss_scale > 0:
            total_loss += self.rotation_loss_scale * rotation_loss
            self.log(f'{log_prefix}rotation_loss', rotation_loss, prog_bar=(step_type == 'train'), sync_dist=True)

        # Log auxiliary losses from the dictionary
        if aux_loss_dict: # Check if dictionary is not empty
             for loss_name, loss_value in aux_loss_dict.items():
                  # Ensure loss_value is a tensor before checking equality
                  if isinstance(loss_value, torch.Tensor) and not torch.equal(loss_value, torch.tensor(0.0, device=self.device)):
                      # Decide prog_bar based on loss_name if needed, default False
                      self.log(f'{log_prefix}{loss_name}', loss_value, prog_bar=False, sync_dist=True)
                  elif isinstance(loss_value, float) and loss_value != 0.0: # Handle potential float losses
                      self.log(f'{log_prefix}{loss_name}', loss_value, prog_bar=False, sync_dist=True)


        # Log the final total loss
        self.log(f'{log_prefix}loss', total_loss, prog_bar=True, sync_dist=True)

        return total_loss # Return total_loss needed for training_step return

    def training_step(self, batch, batch_idx):
        points = batch                                  # (B, N, 3)
        points_permuted = points.permute(0, 2, 1)       # (B, 3, N)

        # ------------------------------------------------------------------
        # Encode only once to avoid double-updating BatchNorm statistics
        # ------------------------------------------------------------------
        # encoder returns: (latent_code, trans, trans_feat)
        latent, _, trans_feat = self.encoder(points_permuted)
        trans_feat_list = [trans_feat] if trans_feat is not None else []

        # ------------------------------------------------------------------
        # Optional Gaussian noise  (applied **only** during training)
        # ------------------------------------------------------------------
        if self.latent_noise_std > 0:
            latent_input = latent + torch.randn_like(latent) * self.latent_noise_std
        else:
            latent_input = latent

        # Single pass through the decoder
        pred = self.decoder(latent_input)

        # Prepare loss arguments (regularisers use clean latent)
        loss_args = self._prepare_loss_args(points, pred, trans_feat_list, latent=latent)
         
        # Compute criterion loss
        main_loss, aux_loss_dict = self.criterion(**loss_args)
        
        # Compute rotation loss if applicable (uses clean latent)
        computed_rotation_loss = None
        if self.rotation_loss_scale > 0:
            points_rotated = random_rotate_point_cloud_batch(points)
            points_rotated_permuted = points_rotated.permute(0, 2, 1)

            latent_rotated, _, _ = self.encoder(points_rotated_permuted)
            computed_rotation_loss = F.mse_loss(latent, latent_rotated)

        # Log all losses and get total loss
        total_loss = self._log_losses(main_loss, aux_loss_dict, 'train', rotation_loss=computed_rotation_loss)

        if isinstance(self.logger, WandbLogger):
            if batch_idx % 100 == 0:
                # Monitor encoder statistics (clean latent)
                latent_mean = latent.detach().mean(dim=0).cpu().numpy()
                latent_std  = latent.detach().std(dim=0).cpu().numpy()

                dims = np.arange(latent_mean.shape[0])
                mean_table = wandb.Table(data=list(zip(dims, latent_mean)),
                                        columns=["latent_id", "mean"])
                std_table  = wandb.Table(data=list(zip(dims, latent_std)),
                                        columns=["latent_id", "std"])

                self.logger.experiment.log({
                    "train/latent_mean_bar": wandb.plot.bar(mean_table, "latent_id", "mean", title="Mean per latent dimension"),
                    "train/latent_std_bar":  wandb.plot.bar(std_table,  "latent_id", "std",  title="Std per latent dimension"),
                }, step=self.global_step)

        return {'loss': total_loss}


    def validation_step(self, batch, batch_idx):
        points = batch
        points_permuted = points.permute(0, 2, 1)

        latent, _, trans_feat = self.encoder(points_permuted)
        pred = self.decoder(latent)
        trans_feat_list = [trans_feat] if trans_feat is not None else []
        
        loss_args = self._prepare_loss_args(points, pred, trans_feat_list, latent=latent)
        
        main_loss, aux_loss_dict = self.criterion(**loss_args)
        
        # In validation, we don't compute rotation loss. You can add it if needed for monitoring.
        total_loss = self._log_losses(main_loss, aux_loss_dict, "val")

        if isinstance(self.logger, WandbLogger) and batch_idx % 100 == 0:
            # Compute mean and std across the batch for each latent dimension
            latent_mean = latent.detach().mean(dim=0).cpu().numpy()
            latent_std  = latent.detach().std(dim=0).cpu().numpy()

            # Prepare one row per latent component  →  [["dim", value], …]
            dims = np.arange(latent_mean.shape[0])
            mean_table = wandb.Table(data=list(zip(dims, latent_mean)),
                                     columns=["latent_id", "mean"])
            std_table  = wandb.Table(data=list(zip(dims, latent_std)),
                                     columns=["latent_id", "std"])

            # One bar = one latent component
            self.logger.experiment.log({
                "val/latent_mean_bar": wandb.plot.bar(mean_table,
                                                        "latent_id", "mean",
                                                        title="Mean per latent dimension"),
                "val/latent_std_bar":  wandb.plot.bar(std_table,
                                                        "latent_id", "std",
                                                        title="Std per latent dimension"),
            }, step=self.global_step)

        return total_loss

    def configure_optimizers(self):
        return get_optimizers_and_scheduler(self.hparams, self.parameters())
