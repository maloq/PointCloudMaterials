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
    elif model_type == "TransformerAutoencoderS2S":
        logger.print("TransformerAutoencoderS2S")
        return TransformerAutoencoderS2S(latent_dim=latent_size, 
                                         hidden_dim=512, 
                                         num_encoder_layers=4,
                                         num_decoder_layers=4,
                                         nhead=8,)
    elif model_type == "SequenceAutoencoderWithRefinement":
        logger.print("SequenceAutoencoderWithRefinement")
        return SequenceAutoencoderWithRefinement(seq_length=num_points,
                                               input_dim=3,
                                               hidden_dim=64,
                                               latent_dim=latent_size,
                                               num_refinement_steps=3)
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
    def __init__(self, d_model, max_seq_length=2000):
        super().__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter but should be saved and loaded with the model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch, seq_len, embedding_dim]
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerAutoencoderS2S(nn.Module):
    def __init__(self, 
                 latent_dim=256, 
                 hidden_dim=512, 
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 nhead=8,
                 dropout=0.1):
        super().__init__()
        
        # Input embedding layer (from 3D point to hidden dimension)
        self.input_embedding = nn.Linear(3, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Project to latent space
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        
        # Latent to sequence
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, 3)
        
        # Learnable query embeddings for the decoder
        self.query_embeddings = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Latent space parameters
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
    def encode(self, x):
        # x shape: [batch, num_points, 3]
        batch_size, num_points, _ = x.shape
        
        # Embed points
        x = self.input_embedding(x)  # [batch, num_points, hidden_dim]
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer encoder
        memory = self.transformer_encoder(x)  # [batch, num_points, hidden_dim]
        
        # Average pooling over sequence dimension to get a fixed-size representation
        pooled = memory.mean(dim=1)  # [batch, hidden_dim]
        
        # Project to latent space
        latent = self.to_latent(pooled)  # [batch, latent_dim]
        
        return latent
    
    def decode(self, latent, num_points):
        # latent shape: [batch, latent_dim]
        batch_size = latent.shape[0]
        
        # Convert latent to hidden dim
        hidden = self.from_latent(latent)  # [batch, hidden_dim]
        
        # Expand latent to create memory for the decoder
        memory = hidden.unsqueeze(1).expand(-1, num_points, -1)  # [batch, num_points, hidden_dim]
        
        # Create query embeddings for each point
        query = self.query_embeddings.expand(batch_size, num_points, -1)  # [batch, num_points, hidden_dim]
        
        # Add positional encoding to the query
        query = self.positional_encoding(query)
        
        # Pass through transformer decoder
        # The transformer decoder will use the queries to attend to the memory
        decoded = self.transformer_decoder(query, memory)  # [batch, num_points, hidden_dim]
        
        # Project back to 3D space
        output = self.output_projection(decoded)  # [batch, num_points, 3]
        
        return output
    
    def forward(self, x):
        # x shape: [batch, num_points, 3]
        batch_size, num_points, _ = x.shape
        
        # Encode input sequence to latent representation
        latent = self.encode(x)
        
        # Decode latent representation back to sequence
        reconstructed = self.decode(latent, num_points)
        
        return reconstructed, latent



class SequenceAutoencoderWithRefinement(nn.Module):
    def __init__(
        self, 
        seq_length, 
        input_dim=3, 
        hidden_dim=64, 
        latent_dim=32, 
        num_refinement_steps=3
    ):
        """
        Autoencoder with iterative refinement for sequence data.
        
        Args:
            seq_length: Length of the input sequences
            input_dim: Dimension of each time step in the sequence (default 3)
            hidden_dim: Hidden dimension for the encoder and decoder
            latent_dim: Dimension of the latent representation
            num_refinement_steps: Number of refinement iterations
        """
        super().__init__()
        
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_refinement_steps = num_refinement_steps
        
        # Encoder
        self.encoder_rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.encoder_fc = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.decoder_out = nn.Linear(hidden_dim, input_dim)
        
        # Refinement network
        self.refinement_rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.refinement_out = nn.Linear(hidden_dim * 2, input_dim)
        
    def encoder(self, x):
        """Encode input to latent representation."""
        _, hidden = self.encoder_rnn(x)
        # Concatenate the final hidden states from both directions
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.encoder_fc(hidden)
    
    def decoder(self, z):
        """Decode latent representation to output sequence."""
        batch_size = z.size(0)
        
        # Transform latent vector to initial hidden state for 2-layer GRU
        h_0 = self.decoder_fc(z).unsqueeze(0).repeat(2, 1, 1)
        
        # Create a sequence of zeros as initial input to the RNN
        decoder_input = torch.zeros(
            batch_size, self.seq_length, self.hidden_dim, 
            device=z.device
        )
        
        output, _ = self.decoder_rnn(decoder_input, h_0)
        return self.decoder_out(output)
    
    def refine(self, x_prev):
        """Refine the previous reconstruction."""
        output, _ = self.refinement_rnn(x_prev)
        residual = self.refinement_out(output)
        return x_prev + residual
    
    def forward(self, x, return_all_steps=False):
        """
        Forward pass through the autoencoder with iterative refinement.
        
        Args:
            x: Input tensor of shape [batch, seq_length, input_dim]
            return_all_steps: If True, return all refinement steps
            
        Returns:
            If return_all_steps is True, returns a list of reconstructions.
            Otherwise, returns only the final reconstruction.
        """
        # Encode input to latent representation
        z = self.encoder(x)
        
        # Decode latent representation to initial reconstruction
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


