import os
import sys
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F  # Import F for functional operations like mse_loss
from src.loss.reconstruction_loss import *
from src.models.autoencoders_nn.pointnet_autoencoder import build_model
from src.utils.logging_config import setup_logging  
from src.utils.optimizer_utils import get_optimizers_and_scheduler
import math
logger = setup_logging()


class PointNetAutoencoder(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = build_model(cfg)
        self.sphere_radius = cfg.data.radius
        self.num_points = cfg.data.num_points
        self.dr = getattr(cfg.data, 'dr', None)
        self.reconstruction_loss_scale = cfg.get('reconstruction_loss_scale', 0.0)
        self.feature_transform_loss_scale = cfg.get('feature_transform_loss_scale', 0.0)
        self.rotation_loss_scale = cfg.get('rotation_loss_scale', 0.0)
        self.l1_latent_loss_scale = cfg.get('l1_latent_loss_scale', 0.0)
        self.repulsion_h = cfg.get('repulsion_h', 0.05)
        self.repulsion_scale = cfg.get('repulsion_scale', 0.1)

        if cfg.torch_compile:
            self.model = torch.compile(self.model)
            
        if cfg.loss == 'chamfer_loss':
             self.criterion = chamfer_loss
             logger.print(f"Using Chamfer loss")
        elif cfg.loss == 'chamfer_regularized_encoder_loss':
             self.criterion = chamfer_regularized_encoder_loss
             logger.print(f"Using Chamfer loss with feature transform regularization")
        elif cfg.loss == 'chamfer_regularized_encoder_loss_repulsion':
             self.criterion = chamfer_regularized_encoder_loss_repulsion
             logger.print(f"Using Chamfer loss with feature transform regularization and repulsion loss")
        else:
            logger.warning(f"Unknown or basic loss type '{cfg.loss}' specified. Defaulting to basic chamfer_loss.")
            self.criterion = chamfer_loss
            
        if 'density' in self.criterion.__code__.co_varnames or 'dr' in self.criterion.__code__.co_varnames:
             if self.dr is None:
                  raise ValueError("`cfg.data.dr` must be specified for RDF-based losses.")
             self.density = self.compute_density()
             logger.print(f"Density computed: {self.density}")
        else:
             self.density = None

        logger.print(f"Loss: {self.criterion.__name__}")
        if self.rotation_loss_scale > 0:
             logger.print(f"Rotational Consistency Loss enabled with scale: {self.rotation_loss_scale}")
        if self.l1_latent_loss_scale > 0:
            logger.print(f"L1 Latent Loss enabled with scale: {self.l1_latent_loss_scale}")


    def compute_density(self):
        if self.sphere_radius <= 0 or self.num_points <= 0:
             raise ValueError("Sphere radius and num_points must be positive for density calculation.")
        volume = (4/3) * torch.pi * (self.sphere_radius**3)
        density = self.num_points / volume
        return density
    
    def random_rotate_point_cloud_batch(self, batch_pc):
        """Applies a random rotation to each point cloud in the batch."""
        B, N, _ = batch_pc.shape
        device = batch_pc.device

        angles = torch.rand(B, 3, device=device) * 2 * torch.pi

        cos_x, sin_x = torch.cos(angles[:, 2]), torch.sin(angles[:, 2])
        cos_y, sin_y = torch.cos(angles[:, 1]), torch.sin(angles[:, 1])
        cos_z, sin_z = torch.cos(angles[:, 0]), torch.sin(angles[:, 0])

        rot_x = torch.zeros(B, 3, 3, device=device)
        rot_x[:, 0, 0] = 1
        rot_x[:, 1, 1] = cos_x
        rot_x[:, 1, 2] = -sin_x
        rot_x[:, 2, 1] = sin_x
        rot_x[:, 2, 2] = cos_x

        rot_y = torch.zeros(B, 3, 3, device=device)
        rot_y[:, 0, 0] = cos_y
        rot_y[:, 0, 2] = sin_y
        rot_y[:, 1, 1] = 1
        rot_y[:, 2, 0] = -sin_y
        rot_y[:, 2, 2] = cos_y

        rot_z = torch.zeros(B, 3, 3, device=device)
        rot_z[:, 0, 0] = cos_z
        rot_z[:, 0, 1] = -sin_z
        rot_z[:, 1, 0] = sin_z
        rot_z[:, 1, 1] = cos_z
        rot_z[:, 2, 2] = 1

        rotation_matrix = rot_z @ rot_y @ rot_x

        rotated_batch_pc = torch.bmm(batch_pc, rotation_matrix)
        return rotated_batch_pc

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
        # self.model(x) returns: (reconstructed_x, latent_code, _, aux_outputs)
        # The third element is unused.
        reconstructed_points, latent_vector, _, trans_feat_list = self.model(x)

        # Ensure reconstructed points have the shape (B, N, 3) if necessary
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
            'l1_latent_loss_scale': self.hparams.get('l1_latent_loss_scale', 0.0)
        }
        if self.density is not None:
            loss_args.update({
                'density': self.density,
                'dr': self.hparams.data.dr,
                'sphere_radius': self.hparams.data.radius,
                'reconstruction_loss_scale': self.hparams.get('reconstruction_loss_scale', 0.0)
            })
        
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
        points = batch
        points_permuted = points.permute(0, 2, 1)

        # model returns: reconstructed_points, latent_vector, _, transformation_features
        model_outputs = self.model(points_permuted)
        pred, latent, _, trans_feat_list = model_outputs

        # Prepare loss arguments
        loss_args = self._prepare_loss_args(points, pred, trans_feat_list, latent=latent)

        # Compute criterion loss
        main_loss, aux_loss_dict = self.criterion(**loss_args)

        # Compute rotation loss if applicable
        computed_rotation_loss = None
        if self.rotation_loss_scale > 0:
            points_rotated = self.random_rotate_point_cloud_batch(points)
            points_rotated_permuted = points_rotated.permute(0, 2, 1)
            
            # self.model.encoder returns: (latent_code, trans, trans_feat)
            encoder_outputs_rotated = self.model.encoder(points_rotated_permuted)
            latent_rotated = encoder_outputs_rotated[0]
            
            # latent is the primary latent from the original points.
            computed_rotation_loss = F.mse_loss(latent, latent_rotated)

        # Log all losses and get total loss
        total_loss = self._log_losses(main_loss, aux_loss_dict, 'train', rotation_loss=computed_rotation_loss)

        return {'loss': total_loss}


    def validation_step(self, batch, batch_idx):
        points = batch
        points_permuted = points.permute(0, 2, 1)

        model_outputs = self.model(points_permuted)
        pred, latent, _, trans_feat_list = model_outputs
        
        loss_args = self._prepare_loss_args(points, pred, trans_feat_list, latent=latent)
        
        main_loss, aux_loss_dict = self.criterion(**loss_args)
        
        # In validation, we don't compute rotation loss. You can add it if needed for monitoring.
        total_loss = self._log_losses(main_loss, aux_loss_dict, "val")

        return total_loss

    def configure_optimizers(self):
        return get_optimizers_and_scheduler(self.hparams, self.parameters())
