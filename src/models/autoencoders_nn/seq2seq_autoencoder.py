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

    if model_type == "MLP_AE_seq2seq":
        logger.print("MLP_AE_seq2seq")
        return MLP_AE_seq2seq(input_dim=3, n_points=num_points, latent_dim=latent_size)
    elif model_type == "TransformerAutoencoder":
        logger.print("TransformerAutoencoder")
        return TransformerAutoencoder(input_dim=3, model_dim=latent_size, num_heads=8, encoder_layers=4, decoder_layers=4, dropout=0.1, seq_length=num_points)
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
        return z

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



class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding used in the Transformer.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Args:
            d_model (int): Dimension of the model.
            dropout (float): Dropout value.
            max_len (int): Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create constant 'pe' matrix with values dependent on 
        # position and i (dimension).
        pe = torch.zeros(max_len, d_model)  # shape: (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape \((B, T, d_{\text{model}})\).
        Returns:
            Tensor: The input tensor with positional encodings added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerAutoencoder(nn.Module):
    """
    Transformer-based Autoencoder for sequence reconstruction.
    
    This autoencoder compresses an input sequence (e.g. from AtomicSequenceDataset)
    into a latent representation using a Transformer encoder and then reconstructs
    the sequence using a Transformer decoder.
    
    The encoder first projects the input points (e.g. 3D coordinates in spherical space)
    into a higher-dimensional space and adds positional encodings before passing them 
    through a stack of Transformer encoder layers. The token embeddings are aggregated 
    (mean-pooled) to yield a latent vector. The decoder then uses a set of learnable 
    query tokens (added to the latent vector) and a stack of Transformer decoder layers 
    to generate reconstructed tokens that are finally mapped back to the input dimension.
    """
    def __init__(self, input_dim=3, model_dim=128, num_heads=8, 
                 encoder_layers=6, decoder_layers=6, dropout=0.1, seq_length=100):
        """
        Initialize the Transformer-based autoencoder.
        
        Args:
            input_dim (int): Dimension of input points (e.g. 3 for (x,y,z) or spherical coordinates).
            model_dim (int): Dimension of the Transformer model.
            num_heads (int): Number of attention heads.
            encoder_layers (int): Number of encoder layers.
            decoder_layers (int): Number of decoder layers.
            dropout (float): Dropout rate.
            seq_length (int): Length of the input sequence (number of points per sample).
        """
        super(TransformerAutoencoder, self).__init__()
        self.seq_length = seq_length
        
        # Encoder: project input to model_dim and add positional encoding.
        self.encoder_input_proj = nn.Linear(input_dim, model_dim)
        self.encoder_positional = PositionalEncoding(model_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        
        # Bottleneck: aggregate encoder tokens into a latent representation,
        # here we use mean-pooling.
        self.latent_projection = nn.Linear(model_dim, model_dim)
        
        # Decoder: Use a set of learnable query tokens (of length seq_length) which,
        # when added to the latent vector (broadcasted), serve as initial decoder tokens.
        self.decoder_query = nn.Parameter(torch.randn(seq_length, model_dim))
        self.decoder_positional = PositionalEncoding(model_dim, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        
        # Project the decoder output back to the input dimension.
        self.output_proj = nn.Linear(model_dim, input_dim)
        
    def forward(self, x):
        """
        Forward pass of the autoencoder.
        
        Args:
            x (Tensor): Input tensor of shape \((B, T, d_{\text{in}})\).
            
        Returns:
            Tensor: Reconstructed tensor of shape \((B, T, d_{\text{in}})\).
        """
        batch_size, seq_len, _ = x.size()
        
        # ---------- Encoder ----------
        # Project input points and add positional encodings.
        x_enc = self.encoder_input_proj(x)             # (B, T, model_dim)
        x_enc = self.encoder_positional(x_enc)           # (B, T, model_dim)
        # Transformer encoder expects (T, B, model_dim)
        x_enc = x_enc.transpose(0, 1)                    
        encoder_output = self.encoder(x_enc)             # (T, B, model_dim)
        
        # Aggregate tokens into a single latent vector via mean-pooling.
        latent = encoder_output.mean(dim=0)              # (B, model_dim)
        latent = self.latent_projection(latent)          # (B, model_dim)
        
        # ---------- Decoder ----------
        # For decoding, we prepare a set of learnable query tokens.
        # We first expand the latent vector to match the decoder sequence length.
        latent_expanded = latent.unsqueeze(1).expand(-1, self.seq_length, -1)  # (B, T, model_dim)
        # Add the learnable query tokens (broadcast for each sample).
        dec_input = latent_expanded + self.decoder_query.unsqueeze(0)          # (B, T, model_dim)
        dec_input = self.decoder_positional(dec_input)                         # (B, T, model_dim)
        # Permute to (T, B, model_dim) as required.
        dec_input = dec_input.transpose(0, 1)
        
        # As memory for the decoder, we use the latent representation.
        # Here we use a single latent token per sample.
        memory = latent.unsqueeze(0)                       # (1, B, model_dim)
        decoder_output = self.decoder(dec_input, memory)   # (T, B, model_dim)
        decoder_output = decoder_output.transpose(0, 1)    # (B, T, model_dim)
        
        # Project decoder outputs back to the original input space.
        reconstruction = self.output_proj(decoder_output)  # (B, T, input_dim)
        return reconstruction, latent




