from pytorch3d.loss import chamfer_distance
import torch
import torch.nn as nn
from omegaconf import DictConfig
import logging
from src.utils.logging_config import setup_logging
import math
import torch.nn.functional as F
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
    elif model_type == "ResNet1D_AE":
        logger.print("ResNet1D_AE")
        return ResNet1D_AE(input_dim=3,
                           n_points=num_points,
                           latent_dim=latent_size,
                           base_filters=64,
                           num_blocks=[2, 2, 2, 2, 2])
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



class ResNetBlock1D(nn.Module):
    """
    Standard 1D ResNet block with optional downsampling.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResNetBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Second conv always has stride 1 to preserve dimensions within the block after potential downsampling in conv1
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection to handle dimension changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity) # Add shortcut
        out = self.relu(out) # Final ReLU
        return out

# --- Residual Block for Decoder (using Upsampling) ---
class ResNetBlockUpsample1D(nn.Module):
    """
    1D ResNet block with upsampling using nn.Upsample + Conv1d.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResNetBlockUpsample1D, self).__init__()
        self.upsample = nn.Identity()
        self.conv1_stride = 1 # Conv1 doesn't handle stride here

        # If stride > 1, we upsample before the first convolution
        if stride > 1:
            self.upsample = nn.Upsample(scale_factor=stride, mode='linear', align_corners=False)

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, self.conv1_stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection to handle dimension changes (upsampling and channel adjustment)
        self.shortcut = nn.Sequential()
        if stride > 1 or in_channels != out_channels:
            modules = []
            # Apply upsampling first if needed
            if stride > 1:
                modules.append(nn.Upsample(scale_factor=stride, mode='linear', align_corners=False))
            # 1x1 convolution to adjust channels (applied after potential upsampling)
            modules.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False))
            modules.append(nn.BatchNorm1d(out_channels))
            self.shortcut = nn.Sequential(*modules)

    def forward(self, x):
        identity = x

        # Apply main path transformations
        out = self.upsample(x) # Upsample if stride > 1
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))

        # Apply shortcut transformations
        shortcut_out = self.shortcut(identity)

        # Ensure spatial dimensions match before adding (robustness check)
        if shortcut_out.size(-1) != out.size(-1):
             shortcut_out = F.interpolate(shortcut_out, size=out.size(-1), mode='linear', align_corners=False)

        out += shortcut_out # Add shortcut
        out = self.relu(out) # Final ReLU
        return out


class ResNet1D_Encoder(nn.Module):
    def __init__(self, input_dim=3, n_points=100, latent_dim=128, base_filters=64, num_blocks=[2, 2, 2, 2]):
        """
        1D ResNet Encoder.

        Args:
            input_dim (int): Number of input channels (e.g., 3).
            n_points (int): Length of the input sequence.
            latent_dim (int): Dimension of the output latent vector.
            base_filters (int): Number of filters in the initial conv layer.
            num_blocks (list): Number of ResNet blocks in each stage.
        """
        super(ResNet1D_Encoder, self).__init__()
        self.input_dim = input_dim
        self.n_points = n_points
        self.latent_dim = latent_dim
        self.current_filters = base_filters

        # Initial convolution block
        self.conv1 = nn.Conv1d(input_dim, self.current_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.current_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layers = nn.ModuleList()
        in_filters = self.current_filters
        for i, num_block in enumerate(num_blocks):
            # Downsample (stride=2) in the first block of each stage, except the first stage
            stride = 1 if i == 0 else 2
            self.layers.append(self._make_layer(in_filters, self.current_filters, num_block, stride=stride))
            in_filters = self.current_filters # Input filters for the next stage
            # Double the number of filters for the next stage
            if i < len(num_blocks) -1 : # Don't double after the last stage
                 self.current_filters *= 2


        # Final pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool1d(1) # Pool features across the sequence length
        self.fc = nn.Linear(in_filters, latent_dim) # Project to latent space

    def _make_layer(self, in_filters, out_filters, num_blocks, stride):
        """Helper function to create a ResNet stage."""
        layers = []
        # First block handles stride and filter changes
        layers.append(ResNetBlock1D(in_filters, out_filters, stride=stride))
        # Subsequent blocks in the stage
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock1D(out_filters, out_filters, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim, n_points).
        Returns:
            z (Tensor): Latent representation of shape (batch_size, latent_dim).
        """
        # Initial convolutions
        # print(f'encoder input shape: {x.shape}')
        x = x.permute(0, 2, 1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # ResNet stages
        for layer in self.layers:
            x = layer(x)

        # Global average pooling
        x = self.avgpool(x) # Shape: (batch_size, last_layer_filters, 1)
        x = torch.flatten(x, 1) # Shape: (batch_size, last_layer_filters)

        # Final projection
        z = self.fc(x) # Shape: (batch_size, latent_dim)
        return z, []


class ResNet1D_Decoder(nn.Module):
    def __init__(self, latent_dim=128, output_dim=3, n_points=100, base_filters=64, num_blocks=[2, 2, 2, 2]):
        """
        1D ResNet Decoder.

        Args:
            latent_dim (int): Dimension of the input latent vector.
            output_dim (int): Number of output channels (e.g., 3).
            n_points (int): Target length of the output sequence.
            base_filters (int): Number of filters used (should match encoder).
            num_blocks (list): Number of ResNet blocks in each stage (should match encoder).
        """
        super(ResNet1D_Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_points = n_points

        # Calculate filter sizes corresponding to encoder stages
        encoder_filter_sizes = [base_filters]
        for i in range(len(num_blocks) - 1):
            encoder_filter_sizes.append(base_filters * (2**(i+1)))

        # Starting filters for the decoder is the last filter size of the encoder
        self.decoder_initial_filters = encoder_filter_sizes[-1]

        # Estimate the spatial dimension before the AdaptiveAvgPool1d in the encoder
        # This depends on n_points and the downsampling factors (conv s=2, maxpool s=2, layer strides)
        # Let's start with a small fixed spatial dimension, e.g., 7, as it's hard to calculate exactly.
        # Alternatively, calculate based on expected downsampling:
        downsampling_factor = 2 * 2 # Initial conv and pool
        for i in range(1, len(num_blocks)): # Strides in subsequent layers
             downsampling_factor *= 2
        self.initial_length = max(1, n_points // downsampling_factor) # Estimate
        # Using a fixed small size can also work well:
        # self.initial_length = 7

        # Initial linear layer and reshape to start the sequence generation
        self.fc = nn.Linear(latent_dim, self.decoder_initial_filters * self.initial_length)

        # Decoder ResNet layers (reverse order of filters and strides)
        self.layers = nn.ModuleList()
        current_filters = self.decoder_initial_filters
        # Reverse encoder structure: filter sizes decrease, strides are applied for upsampling
        decoder_num_blocks = list(reversed(num_blocks))
        decoder_filter_sizes = list(reversed(encoder_filter_sizes))

        for i, num_block in enumerate(decoder_num_blocks):
            in_filters = current_filters
            # Determine output filters for this stage (decreasing)
            out_filters = decoder_filter_sizes[i+1] if i < len(decoder_num_blocks) - 1 else base_filters
            # Determine stride for upsampling (mirroring encoder's downsampling)
            # Encoder strides: [s=1, s=2, s=2, s=2] -> Decoder strides: [s=2, s=2, s=2, s=1]
            stride = 1 if i == (len(decoder_num_blocks) - 1) else 2 # Upsample except in the last stage

            self.layers.append(self._make_layer(in_filters, out_filters, num_block, stride=stride))
            current_filters = out_filters # Update for next layer's input

        # Final upsampling and convolution layers to reconstruct the sequence
        # Need to reverse the initial conv(s=2) and maxpool(s=2) of the encoder
        self.upsample1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False) # Reverse MaxPool
        # Conv layer after first upsample - should roughly match filters before maxpool (base_filters)
        self.final_conv1 = nn.Conv1d(base_filters, base_filters, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn_final1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False) # Reverse Conv stride=2
        # Final conv layer to map to output_dim
        self.final_conv2 = nn.Conv1d(base_filters, output_dim, kernel_size=7, stride=1, padding=3)


    def _make_layer(self, in_filters, out_filters, num_blocks, stride):
        """Helper function to create a Decoder ResNet stage using Upsample blocks."""
        layers = []
        # First block handles stride (upsampling) and filter changes
        layers.append(ResNetBlockUpsample1D(in_filters, out_filters, stride=stride))
        # Subsequent blocks in the stage
        for _ in range(1, num_blocks):
            # Stride is 1, channels remain out_filters
            layers.append(ResNetBlockUpsample1D(out_filters, out_filters, stride=1))
        return nn.Sequential(*layers)

    def forward(self, z):
        """
        Args:
            z (Tensor): Latent vector of shape (batch_size, latent_dim).
        Returns:
            x (Tensor): Reconstructed sequence of shape (batch_size, output_dim, n_points).
        """
        # Initial projection and reshape
        x = self.fc(z)
        x = x.view(z.size(0), self.decoder_initial_filters, self.initial_length)

        # Decoder ResNet stages
        for layer in self.layers:
            x = layer(x)

        # Final upsampling and convolutions
        x = self.upsample1(x)
        x = self.relu(self.bn_final1(self.final_conv1(x)))
        x = self.upsample2(x)
        x = self.final_conv2(x)

        # Ensure output size matches n_points using interpolation
        if x.size(-1) != self.n_points:
            x = F.interpolate(x, size=self.n_points, mode='linear', align_corners=False)

        return x


class ResNet1D_AE(nn.Module):
    def __init__(self, input_dim=3, n_points=100, latent_dim=128, base_filters=64, num_blocks=[2, 2, 2, 2]):
        """
        1D ResNet Autoencoder.

        Args:
            input_dim (int): Dimensionality of input points (e.g., 3).
            n_points (int): Number of points in each sequence sample.
            latent_dim (int): Dimensionality of the latent space.
            base_filters (int): Number of filters in the first convolutional layer.
            num_blocks (list): List containing the number of ResNet blocks in each layer group for encoder/decoder.
                               Length determines the depth. Example: [2, 2, 2, 2] for ResNet18-like depth.
        """
        super(ResNet1D_AE, self).__init__()
        self.encoder = ResNet1D_Encoder(input_dim, n_points, latent_dim, base_filters, num_blocks)
        self.decoder = ResNet1D_Decoder(latent_dim, input_dim, n_points, base_filters, num_blocks) 

    def forward(self, x):
        """
        Forward pass for the autoencoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim, n_points).

        Returns:
            reconstructed_x (Tensor): Reconstructed sequence of shape (batch_size, input_dim, n_points).
            z (Tensor): Latent representation of shape (batch_size, latent_dim).
        """
        z = self.encoder(x)[0]
        reconstructed_x = self.decoder(z)
        reconstructed_x = reconstructed_x.permute(0, 2, 1)
        return reconstructed_x, z