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
    model_type = cfg.type
    num_points = cfg.data.num_points
    latent_size = cfg.latent_size
    num_refinement_steps = cfg.get("num_refinement_steps", 5)
    diffusion_steps = cfg.get("diffusion_steps", 1000)
    diffusion_beta_schedule = cfg.get("diffusion_beta_schedule", "linear")
    blocks_per_level = cfg.get("blocks_per_level", 2)

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
    elif model_type == "ResNetSequenceAutoencoderWithRefinement":
        logger.print("ResNetSequenceAutoencoderWithRefinement")
        return ResNetSequenceAutoencoderWithRefinement(seq_length=num_points,
                                                      input_dim=3,
                                                      base_channels=64,
                                                      latent_dim=latent_size,
                                                      num_refinement_steps=num_refinement_steps,
                                                      num_levels=4,
                                                      blocks_per_level=blocks_per_level)
    elif model_type == "LatentDiffusionSequenceModel":
        logger.print("LatentDiffusionSequenceModel")
        return LatentDiffusionSequenceModel(seq_length=num_points,
                                           input_dim=3,
                                           base_channels=128,
                                           latent_dim=latent_size,
                                           diffusion_steps=diffusion_steps,
                                           diffusion_beta_schedule=diffusion_beta_schedule,
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
        


class LatentDiffusionSequenceModel(nn.Module):
    def __init__(
        self,
        seq_length,
        input_dim=3,
        base_channels=64,
        latent_dim=32,
        diffusion_steps=1000,
        diffusion_beta_schedule="linear",
        diffusion_beta_start=1e-4,
        diffusion_beta_end=0.02,
        encoder_type="conv_unet",
        num_levels=4,
    ):
        """
        Latent Diffusion Model for sequence data with a fixed-size latent space.
        
        Args:
            seq_length: Length of the input sequences
            input_dim: Dimension of each time step in the sequence (default 3)
            base_channels: Base number of channels for convolutions
            latent_dim: Dimension of the latent representation (fixed-size)
            diffusion_steps: Number of diffusion timesteps
            diffusion_beta_schedule: Schedule for noise variance ("linear" or "cosine")
            diffusion_beta_start: Starting value for beta schedule
            diffusion_beta_end: Ending value for beta schedule
            encoder_type: Type of encoder ("conv_unet" or "mlp")
            num_levels: Number of down/upsampling levels in the encoder/decoder
        """
        super().__init__()
        
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.diffusion_steps = diffusion_steps
        
        # Instantiate encoder-decoder architecture
        if encoder_type == "conv_unet":
            # Use UNet encoder/decoder structure similar to ConvUNetSequenceAutoencoderWithRefinement
            self.encoder = nn.ModuleList()
            self.decoder = nn.ModuleList()
            self._build_conv_unet_encoder_decoder(base_channels, num_levels)
        else:
            # Use MLP encoder/decoder similar to MLP_AE_seq2seq
            self.encoder = MLP_Encoder(input_dim=input_dim, n_points=seq_length, latent_dim=latent_dim)
            self.decoder = MLP_Decoder(input_dim=input_dim, n_points=seq_length, latent_dim=latent_dim)
        
        # Diffusion model components
        # - Time embedding for the diffusion process
        self.time_embedding = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        
        # - Noise prediction network (predicts noise from noisy latent and timestep)
        self.noise_predictor = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        
        # Set up diffusion schedule
        self.beta_schedule = diffusion_beta_schedule
        self.beta_start = diffusion_beta_start
        self.beta_end = diffusion_beta_end
        self._setup_diffusion_schedule()
    
    def _build_conv_unet_encoder_decoder(self, base_channels, num_levels):
        """Build convolutional U-Net encoder and decoder similar to ConvUNetSequenceAutoencoder"""
        
        # Encoder (downsampling path)
        encoder_blocks = nn.ModuleList()
        encoder_pools = nn.ModuleList()
        
        # First encoder block
        encoder_blocks.append(
            nn.Sequential(
                nn.Conv1d(self.input_dim, base_channels, kernel_size=3, padding=1),
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
            
            encoder_pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            encoder_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Bottleneck to latent space
        bottleneck_size = self.seq_length // (2**(num_levels-1)) * base_channels * (2**(num_levels-1))
        to_latent = nn.Linear(bottleneck_size, self.latent_dim)
        from_latent = nn.Linear(self.latent_dim, bottleneck_size)
        
        # Decoder - upsampling path
        decoder_ups = nn.ModuleList()
        decoder_blocks = nn.ModuleList()
        
        # Upsampling blocks
        for i in range(num_levels-1, 0, -1):
            in_channels = base_channels * (2**i)
            out_channels = base_channels * (2**(i-1))
            
            decoder_ups.append(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
            )
            decoder_blocks.append(
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
        output_conv = nn.Conv1d(base_channels, self.input_dim, kernel_size=1)
        
        # Store encoder components
        self.encoder_blocks = encoder_blocks
        self.encoder_pools = encoder_pools
        self.to_latent = to_latent
        
        # Store decoder components
        self.from_latent = from_latent
        self.decoder_ups = decoder_ups
        self.decoder_blocks = decoder_blocks
        self.output_conv = output_conv
        
        # Store architecture parameters
        self.bottleneck_size = bottleneck_size
        self.num_levels = num_levels
        self.base_channels = base_channels
    
    def _setup_diffusion_schedule(self):
        """Set up the beta schedule and precompute values needed for diffusion process"""
        
        if self.beta_schedule == "linear":
            # Linear beta schedule
            betas = torch.linspace(
                self.beta_start, self.beta_end, self.diffusion_steps
            )
        elif self.beta_schedule == "cosine":
            # Cosine beta schedule (less aggressive noise at the beginning)
            steps = self.diffusion_steps + 1
            s = 0.008
            x = torch.linspace(0, self.diffusion_steps, steps)
            alphas_cumprod = torch.cos(((x / self.diffusion_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        
        # Precompute diffusion values
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        sqrt_recip_alphas = torch.sqrt(1. / alphas)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        
        # Register buffers for efficient access during training
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        self.register_buffer('posterior_variance', posterior_variance)
    
    def encode(self, x):
        """Encode input to latent representation"""
        if isinstance(self.encoder, MLP_Encoder):
            z, _, _ = self.encoder(x)
            return z
        else:
            # Using convolutional encoder
            batch_size, seq_len, feat_dim = x.shape
            
            # Reshape for 1D convolution: [batch, input_dim, seq_length]
            x = x.transpose(1, 2)
            
            # Down path
            for i in range(self.num_levels):
                x = self.encoder_blocks[i](x)
                if i < self.num_levels - 1:
                    x = self.encoder_pools[i](x)
            
            # Flatten bottleneck for latent space projection
            bottleneck_flat = x.reshape(batch_size, -1)
            z = self.to_latent(bottleneck_flat)
            
            return z
    
    def decode(self, z):
        """Decode latent representation to output sequence"""
        if isinstance(self.decoder, MLP_Decoder):
            return self.decoder(z)
        else:
            # Using convolutional decoder
            batch_size = z.size(0)
            
            # Project from latent space and reshape to match the bottleneck size
            x = self.from_latent(z)
            
            # Calculate the feature dimension based on the bottleneck size and sequence length
            feature_dim = self.base_channels * (2**(self.num_levels-1))
            
            # Reshape to [batch, channels, seq_length_at_bottleneck]
            x = x.view(batch_size, feature_dim, self.seq_length // (2**(self.num_levels-1)))
            
            # Upsampling path
            for i in range(self.num_levels - 1):
                # Upsample
                x = self.decoder_ups[i](x)
                
                # Apply corresponding decoder block after upsampling
                x = self.decoder_blocks[i](x)
            
            # Final output layer
            x = self.output_conv(x)
            
            # Reshape back to [batch, seq_length, input_dim]
            x = x.transpose(1, 2)
            
            return x
    
    def _denoise_latent_step(self, latent_noisy, t, noise=None):
        """
        Predict noise in the latent representation at timestep t,
        then use this to get predicted x_0 (clean latent).
        """
        batch_size = latent_noisy.shape[0]
        
        # Time embedding
        t_emb = self.time_embedding(t.reshape(-1, 1).float())
        
        # Predict noise by conditioning on timestep and noisy latent
        noise_pred = self.noise_predictor(torch.cat([latent_noisy, t_emb], dim=1))
        
        # If noise is provided, return MSE loss for training
        if noise is not None:
            return torch.nn.functional.mse_loss(noise_pred, noise)
        
        # Compute predicted original sample from noisy latent using predicted noise
        pred_original_sample = (
            (latent_noisy - self.sqrt_one_minus_alphas_cumprod[t] * noise_pred) / 
            self.sqrt_alphas_cumprod[t]
        )
        
        return pred_original_sample
    
    def diffuse_latent(self, latent, t):
        """Add noise to latent according to diffusion schedule at timestep t"""
        batch_size = latent.shape[0]
        
        # Generate random noise of same shape as latent
        noise = torch.randn_like(latent)
        
        # Sample from q(z_t | z_0) as per the diffusion process
        noisy_latent = (
            self.sqrt_alphas_cumprod[t, None] * latent + 
            self.sqrt_one_minus_alphas_cumprod[t, None] * noise
        )
        
        return noisy_latent, noise
    
    def sample_latent(self, batch_size, device):
        """Sample a latent from the diffusion model by denoising pure noise"""
        # Start from pure noise
        latent = torch.randn(batch_size, self.latent_dim, device=device)
        
        # Iteratively denoise
        for t in range(self.diffusion_steps - 1, -1, -1):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # No noise needed during inference (t > 0)
            with torch.no_grad():
                latent = self._denoise_latent_step(latent, t_tensor)
                
                # Add noise if t > 0 (not the final step)
                if t > 0:
                    noise = torch.randn_like(latent)
                    sigma_t = torch.sqrt(self.posterior_variance[t])
                    latent = latent + sigma_t * noise
        
        return latent
    
    def forward(self, x, t=None):
        """
        Forward pass through the Latent Diffusion Model.
        
        During training:
            - Encodes input to latent space
            - Adds noise to latent according to timestep t
            - Predicts noise and computes loss
            
        During inference (if t is None):
            - Samples a latent from pure noise through the diffusion process
            - Decodes the denoised latent to generate a sequence
            
        Args:
            x: Input tensor of shape [batch, seq_length, input_dim] (can be None during inference)
            t: Timestep tensor of shape [batch] for diffusion process (None during inference)
            
        Returns:
            During training: Loss for noise prediction
            During inference: Reconstructed sequence and sampled latent
        """
        if t is not None:
            # Training mode:
            # 1. Encode input to latent space
            z = self.encode(x)
            
            # 2. Add noise to latent according to timestep t
            noisy_z, noise = self.diffuse_latent(z, t)
            
            # 3. Predict noise and compute loss
            loss = self._denoise_latent_step(noisy_z, t, noise)
            
            return loss, z
        else:
            # Inference mode:
            if x is not None:
                # Reconstruction mode (encode-decode)
                z = self.encode(x)
                x_recon = self.decode(z)
                return x_recon, z
            else:
                # Generation mode (sample from noise)
                batch_size = 1  # Default for pure generation
                device = next(self.parameters()).device
                
                # Sample latent through diffusion process
                z = self.sample_latent(batch_size, device)
                
                # Decode to get sequence
                x_gen = self.decode(z)
                
                return x_gen, z

class ResNetBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        """
        Basic ResNet block for 1D sequence data.
        
        Args:
            channels: Number of input/output channels
            kernel_size: Size of the convolutional kernel
            padding: Padding for the convolution
        """
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class ResNetSequenceAutoencoderWithRefinement(nn.Module):
    def __init__(
        self,
        seq_length,
        input_dim=3,
        base_channels=64,
        latent_dim=32,
        num_refinement_steps=5,
        num_levels=4,
        blocks_per_level=2
    ):
        """
        ResNet-based Autoencoder with iterative refinement for sequence data.
        
        Args:
            seq_length: Length of the input sequences
            input_dim: Dimension of each time step in the sequence (default 3 for 3D coordinates)
            base_channels: Base number of channels for convolutions
            latent_dim: Dimension of the latent representation
            num_refinement_steps: Number of refinement iterations
            num_levels: Number of down/upsampling levels
            blocks_per_level: Number of ResNet blocks per level
        """
        super().__init__()
        
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.num_refinement_steps = num_refinement_steps
        self.num_levels = num_levels
        self.blocks_per_level = blocks_per_level
        
        # Initial convolution to increase channels
        self.initial_conv = nn.Conv1d(input_dim, base_channels, kernel_size=3, padding=1)
        
        # Encoder (downsampling path with ResNet blocks)
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsample = nn.ModuleList()
        
        # Create encoder blocks
        for level in range(num_levels):
            level_blocks = nn.ModuleList()
            channels = base_channels * (2**level)
            
            # Add ResNet blocks for this level
            for _ in range(blocks_per_level):
                level_blocks.append(ResNetBlock(channels))
            
            self.encoder_blocks.append(level_blocks)
            
            # Add downsampling except for the last level
            if level < num_levels - 1:
                next_channels = base_channels * (2**(level+1))
                self.encoder_downsample.append(
                    nn.Sequential(
                        nn.Conv1d(channels, next_channels, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm1d(next_channels),
                        nn.ReLU(inplace=True)
                    )
                )
        
        # Bottleneck to latent space
        self.bottleneck_size = seq_length // (2**(num_levels-1)) * base_channels * (2**(num_levels-1))
        self.to_latent = nn.Linear(self.bottleneck_size, latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.bottleneck_size)
        
        # Decoder (upsampling path with ResNet blocks)
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()
        
        # Create decoder blocks
        for level in range(num_levels-1, -1, -1):
            level_blocks = nn.ModuleList()
            channels = base_channels * (2**level)
            
            # Add ResNet blocks for this level
            for _ in range(blocks_per_level):
                level_blocks.append(ResNetBlock(channels))
            
            self.decoder_blocks.append(level_blocks)
            
            # Add upsampling except for the last level
            if level > 0:
                prev_channels = base_channels * (2**(level-1))
                self.decoder_upsample.append(
                    nn.Sequential(
                        nn.ConvTranspose1d(channels, prev_channels, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm1d(prev_channels),
                        nn.ReLU(inplace=True)
                    )
                )
        
        # Output layer
        self.output_conv = nn.Conv1d(base_channels, input_dim, kernel_size=1)
        
        # Refinement network (ResNet-based)
        self.refinement_initial = nn.Conv1d(input_dim, base_channels, kernel_size=3, padding=1)
        self.refinement_blocks = nn.ModuleList()
        
        # Create a series of ResNet blocks for refinement
        for _ in range(4):  # Using 4 ResNet blocks for refinement
            self.refinement_blocks.append(ResNetBlock(base_channels))
        
        # Refinement output layer
        self.refinement_output = nn.Conv1d(base_channels, input_dim, kernel_size=1)
        
    def encode(self, x):
        """Encode input to latent representation using ResNet encoder."""
        # x shape: [batch, seq_length, input_dim]
        batch_size, seq_len, feat_dim = x.shape
        
        # Reshape for 1D convolution: [batch, input_dim, seq_length]
        x = x.transpose(1, 2)
        
        # Initial convolution
        x = self.initial_conv(x)
        
        # Down path through encoder
        for level in range(self.num_levels):
            # Apply ResNet blocks at this level
            for block in self.encoder_blocks[level]:
                x = block(x)
            
            # Apply downsampling if not the last level
            if level < self.num_levels - 1:
                x = self.encoder_downsample[level](x)
        
        # Flatten bottleneck for latent space projection
        bottleneck_flat = x.reshape(batch_size, -1)
        z = self.to_latent(bottleneck_flat)
        
        return z, None, None
    
    def decode(self, z):
        """Decode latent representation to output sequence using ResNet decoder."""
        batch_size = z.size(0)
        
        # Project from latent space and reshape to match the bottleneck size
        x = self.from_latent(z)
        
        # Calculate the feature dimension based on the bottleneck size and sequence length
        feature_dim = self.base_channels * (2**(self.num_levels-1))
        
        # Reshape to [batch, channels, seq_length_at_bottleneck]
        x = x.view(batch_size, feature_dim, self.seq_length // (2**(self.num_levels-1)))
        
        # Up path through decoder
        for level, level_blocks in enumerate(self.decoder_blocks):
            # Apply ResNet blocks at this level
            for block in level_blocks:
                x = block(x)
            
            # Apply upsampling if not the last level
            if level < len(self.decoder_upsample):
                x = self.decoder_upsample[level](x)
        
        # Final output layer
        x = self.output_conv(x)
        
        # Reshape back to [batch, seq_length, input_dim]
        x = x.transpose(1, 2)
        
        return x
        
    def refine(self, x_prev):
        """Refine the previous reconstruction using ResNet blocks."""
        # x_prev shape: [batch, seq_length, input_dim]
        # Reshape for 1D convolution: [batch, input_dim, seq_length]
        x = x_prev.transpose(1, 2)
        
        # Initial convolution
        x = self.refinement_initial(x)
        
        # Apply ResNet blocks for refinement
        for block in self.refinement_blocks:
            x = block(x)
        
        # Final refinement layer (outputs residual)
        residual = self.refinement_output(x)
        
        # Add residual and reshape back to [batch, seq_length, input_dim]
        residual = residual.transpose(1, 2)
        
        return x_prev + residual
    
    def forward(self, x, return_all_steps=False):
        """
        Forward pass through the ResNet autoencoder with iterative refinement.
        
        Args:
            x: Input tensor of shape [batch, seq_length, input_dim]
            return_all_steps: If True, return all refinement steps
            
        Returns:
            If return_all_steps is True, returns a list of reconstructions.
            Otherwise, returns only the final reconstruction.
        """
        # Encode input to latent representation
        z, _, _ = self.encode(x)
        
        # Decode latent representation to initial reconstruction
        x_recon = self.decode(z)
        
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




