from pytorch3d.loss import chamfer_distance
import torch
import torch.nn as nn
from omegaconf import DictConfig
import logging
from src.utils.logging_config import setup_logging
import math
logger = setup_logging()



def build_model(cfg: DictConfig):
    """
    Factory method to create the model based on OmegaConf configuration.

    Args:
        cfg (DictConfig): The configuration containing model parameters.

    Returns:
        nn.Module: Instantiated model.
    """
    model_type = cfg.model.type
    num_points = cfg.data.num_points
    latent_size = cfg.model.latent_size
    num_refinement_steps = cfg.model.num_refinement_steps

    if model_type == "MLP_AE_seq2seq":
        logger.print("MLP_AE_seq2seq")
        return MLP_AE_seq2seq(input_dim=3, n_points=num_points, latent_dim=latent_size)
    elif model_type == "ConvUNetSequenceAutoencoderWithRefinement":
        logger.print("ConvUNetSequenceAutoencoderWithRefinement")
        return ConvUNetSequenceAutoencoderWithRefinement(seq_length=num_points,
                                                        input_dim=3,
                                                        base_channels=128,
                                                        latent_dim=latent_size,
                                                        num_refinement_steps=num_refinement_steps,
                                                        num_levels=4)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 



class MLP_Encoder(nn.Module):
    def __init__(self, input_dim=3, n_points=100, latent_dim=128):
        """
        Encoder module that compresses a sequence of point clouds into a latent representation.

        Args:
            input_dim (int): Dimensionality of input points (e.g., 3 for (r, theta, phi)).
            n_points (int): Number of points in each point cloud sample.
            latent_dim (int): Dimensionality of the latent space.
        """
        super(MLP_Encoder, self).__init__()
        self.n_points = n_points
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(n_points * input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass for the encoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, n_points, input_dim).

        Returns:
            z (Tensor): Latent representation of shape (batch_size, latent_dim).
        """
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        z = self.fc3(x)
        return z, None, None

class MLP_Decoder(nn.Module):
    def __init__(self, input_dim=3, n_points=100, latent_dim=128):
        """
        Decoder module that reconstructs the point cloud sequence from the latent representation.

        Args:
            input_dim (int): Dimensionality of output points.
            n_points (int): Number of points to reconstruct.
            latent_dim (int): Dimensionality of the latent space.
        """
        super(MLP_Decoder, self).__init__()
        self.n_points = n_points
        self.input_dim = input_dim

        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, n_points * input_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        """
        Forward pass for the decoder.

        Args:
            z (Tensor): Latent representation of shape (batch_size, latent_dim).

        Returns:
            reconstruction (Tensor): Reconstructed point cloud tensor of shape 
                                     (batch_size, n_points, input_dim).
        """
        batch_size = z.size(0)
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # Reshape to (batch_size, n_points, input_dim)
        reconstruction = x.view(batch_size, self.n_points, self.input_dim)
        return reconstruction


class MLP_AE_seq2seq(nn.Module):
    def __init__(self, input_dim=3, n_points=100, latent_dim=128):
        """
        Autoencoder that learns a latent representation for atomic sequence point clouds.

        Args:
            input_dim (int): Dimensionality of input points.
            n_points (int): Number of points in each sequence sample.
            latent_dim (int): Dimensionality of the latent space.
        """
        super(MLP_AE_seq2seq, self).__init__()
        self.encoder = MLP_Encoder(input_dim=input_dim, n_points=n_points, latent_dim=latent_dim)
        self.decoder = MLP_Decoder(input_dim=input_dim, n_points=n_points, latent_dim=latent_dim)

    def forward(self, x):
        """
        Forward pass for the autoencoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, n_points, input_dim).

        Returns:
            reconstruction (Tensor): Reconstructed point cloud tensor.
            latent (Tensor): Latent representation of the input.
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent



class ConvUNetSequenceAutoencoderWithRefinement(nn.Module):
    def __init__(
        self,
        seq_length,
        input_dim=3,
        base_channels=64,
        latent_dim=32,
        num_refinement_steps=5,
        num_levels=4,  # Number of down/upsampling levels in U-Net
    ):
        """
        Convolutional U-Net Autoencoder with iterative refinement for sequence data.
        Modified so encoder and decoder are connected only through latent vector,
        with skip connections only inside the encoder and decoder modules.
        
        Args:
            seq_length: Length of the input sequences
            input_dim: Dimension of each time step in the sequence (default 3)
            base_channels: Base number of channels for convolutions
            latent_dim: Dimension of the latent representation
            num_refinement_steps: Number of refinement iterations
            num_levels: Number of down/upsampling levels in the U-Net
        """
        super().__init__()
        
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.num_refinement_steps = num_refinement_steps
        self.num_levels = num_levels
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        self.encoder_skip_ups = nn.ModuleList()  # For encoder's internal skip connections
        self.encoder_skip_blocks = nn.ModuleList()  # For encoder's internal connections
        
        # First encoder block
        self.encoder_blocks.append(
            nn.Sequential(
                nn.Conv1d(input_dim, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(inplace=True)
            )
        )
        
        # Remaining encoder blocks
        for i in range(1, num_levels):
            in_channels = base_channels * (2**(i-1))
            out_channels = base_channels * (2**i)
            
            self.encoder_pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Encoder's upsampling path with internal skip connections
        for i in range(num_levels-1, 0, -1):
            in_channels = base_channels * (2**i)
            out_channels = base_channels * (2**(i-1))
            
            self.encoder_skip_ups.append(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
            )
            self.encoder_skip_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final encoder output layer (reducing to input_dim for internal reconstruction)
        self.encoder_output_conv = nn.Conv1d(base_channels, input_dim, kernel_size=1)
        
        # Bottleneck to latent space
        self.bottleneck_size = seq_length // (2**(num_levels-1)) * base_channels * (2**(num_levels-1))
        self.to_latent = nn.Linear(self.bottleneck_size, latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.bottleneck_size)
        
        # Decoder - simplified to just use upsampling path
        self.decoder_ups = nn.ModuleList()
        self.decoder_up_blocks = nn.ModuleList()
        
        # Upsampling blocks
        for i in range(num_levels-1, 0, -1):
            in_channels = base_channels * (2**i)
            out_channels = base_channels * (2**(i-1))
            
            self.decoder_ups.append(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
            )
            self.decoder_up_blocks.append(
                nn.Sequential(
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Output layer
        self.output_conv = nn.Conv1d(base_channels, input_dim, kernel_size=1)
        
        # Refinement network (another U-Net with residual connection)
        self.refinement_encoder_blocks = nn.ModuleList()
        self.refinement_encoder_pools = nn.ModuleList()
        self.refinement_decoder_ups = nn.ModuleList()
        self.refinement_decoder_blocks = nn.ModuleList()
        
        # First refinement encoder block
        self.refinement_encoder_blocks.append(
            nn.Sequential(
                nn.Conv1d(input_dim, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(inplace=True)
            )
        )
        
        # Remaining refinement encoder blocks
        for i in range(1, num_levels):
            in_channels = base_channels * (2**(i-1))
            out_channels = base_channels * (2**i)
            
            self.refinement_encoder_pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            self.refinement_encoder_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Refinement upsampling blocks
        for i in range(num_levels-1, 0, -1):
            in_channels = base_channels * (2**i)
            out_channels = base_channels * (2**(i-1))
            
            self.refinement_decoder_ups.append(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
            )
            self.refinement_decoder_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Refinement output layer (residual)
        self.refinement_output = nn.Conv1d(base_channels, input_dim, kernel_size=1)
        
    def encoder(self, x):
        """Encode input to latent representation using 1D U-Net encoder with internal skip connections."""
        # x shape: [batch, seq_length, input_dim]
        batch_size, seq_len, feat_dim = x.shape
        
        # Reshape for 1D convolution: [batch, input_dim, seq_length]
        x = x.transpose(1, 2)
        
        # Store skip connections for encoder's internal U-Net
        encoder_skips = []
        
        # Down path
        for i in range(self.num_levels):
            x = self.encoder_blocks[i](x)
            if i < self.num_levels - 1:
                encoder_skips.append(x)
                x = self.encoder_pools[i](x)
        
        # Store bottleneck features for latent space
        bottleneck = x
        
        # Encoder's internal up path with skip connections
        for i in range(self.num_levels - 1):
            x = self.encoder_skip_ups[i](x)
            
            # Handle potential size mismatch
            diff = encoder_skips[-(i+1)].size(-1) - x.size(-1)
            if diff > 0:
                x = torch.nn.functional.pad(x, (diff // 2, diff - diff // 2))
                
            x = torch.cat([x, encoder_skips[-(i+1)]], dim=1)
            x = self.encoder_skip_blocks[i](x)
        
        # Flatten bottleneck for latent space projection
        bottleneck_flat = bottleneck.reshape(batch_size, -1)
        z = self.to_latent(bottleneck_flat)
        
        return z, None, None
    
    def decoder(self, z):
        """
        Decode latent representation to output sequence using a U-Net structure 
        with its own internal skip connections.
        """
        batch_size = z.size(0)
        
        # Project from latent space and reshape to match the bottleneck size
        x = self.from_latent(z)
        
        # Calculate the feature dimension based on the bottleneck size and sequence length
        feature_dim = self.base_channels * (2**(self.num_levels-1))
        
        # Reshape to [batch, channels, seq_length_at_bottleneck]
        x = x.view(batch_size, feature_dim, self.seq_length // (2**(self.num_levels-1)))
        
        # Apply the first convolution at the bottleneck level
        # No need to go through decoder_blocks[num_levels-1] since we're starting from bottleneck
        
        # Store features for decoder's internal skip connections
        decoder_skips = []
        
        # Start upsampling path directly from the bottleneck
        for i in range(self.num_levels - 1):
            # Upsample
            x = self.decoder_ups[i](x)
            
            # Apply corresponding decoder block after upsampling
            # (we use decoder_up_blocks for the upsampling path)
            x = self.decoder_up_blocks[i](x)
            
            # Save for potential skip connections if we need them
            if i < self.num_levels - 2:  # Save all but the last level
                decoder_skips.append(x)
        
        # Final output layer
        x = self.output_conv(x)
        
        # Reshape back to [batch, seq_length, input_dim]
        x = x.transpose(1, 2)
        
        return x
        
    def refine(self, x_prev):
        """Refine the previous reconstruction using a similar U-Net structure."""
        # x_prev shape: [batch, seq_length, input_dim]
        batch_size, seq_len, feat_dim = x_prev.shape
        
        # Reshape for 1D convolution: [batch, input_dim, seq_length]
        x = x_prev.transpose(1, 2)
        
        # Store skip connections for refinement U-Net
        skip_connections = []
        
        # Down path
        for i in range(self.num_levels):
            if i == 0:
                x = self.refinement_encoder_blocks[i](x)
            else:
                x = self.refinement_encoder_pools[i-1](x)
                x = self.refinement_encoder_blocks[i](x)
            
            if i < self.num_levels - 1:
                skip_connections.append(x)
        
        # Up path with skip connections
        for i in range(self.num_levels - 1):
            x = self.refinement_decoder_ups[i](x)
            # Handle potential size mismatch during upsampling
            diff = skip_connections[-(i+1)].size(-1) - x.size(-1)
            if diff > 0:
                x = torch.nn.functional.pad(x, (diff // 2, diff - diff // 2))
            x = torch.cat([x, skip_connections[-(i+1)]], dim=1)
            x = self.refinement_decoder_blocks[i](x)
        
        # Final refinement layer (outputs residual)
        residual = self.refinement_output(x)
        
        # Add residual and reshape back to [batch, seq_length, input_dim]
        residual = residual.transpose(1, 2)
        
        return x_prev + residual
    
    def forward(self, x, return_all_steps=False):
        """
        Forward pass through the convolutional U-Net autoencoder with iterative refinement.
        Encoder and decoder are connected only through the latent vector.
        
        Args:
            x: Input tensor of shape [batch, seq_length, input_dim]
            return_all_steps: If True, return all refinement steps
            
        Returns:
            If return_all_steps is True, returns a list of reconstructions.
            Otherwise, returns only the final reconstruction.
        """
        # Encode input to latent representation
        z, _, _ = self.encoder(x)
        
        # Decode latent representation to initial reconstruction
        # Note: No skip connections passed between encoder and decoder
        x_recon = self.decoder(z)
        
        refinements = [x_recon]
        
        # Apply refinement steps
        for _ in range(self.num_refinement_steps):
            x_recon = self.refine(x_recon)
            if return_all_steps:
                refinements.append(x_recon)
        
        if return_all_steps:
            return refinements, z
        else:
            return x_recon, z
        

