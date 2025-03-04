import sys, os
sys.path.append(os.getcwd())

from datetime import datetime 
import pytorch_lightning as pl
import torch
from torchsummary import summary
import hydra
from omegaconf import DictConfig
from src.autoencoder_seq2seq.autoencoder_s2s_module import AutoencoderSeq2Seq
import numpy as np


def get_rundir_name() -> str:
    now = datetime.now()
    return str(f'output/{now:%Y-%m-%d}/{now:%H-%M-%S}')
    

def summarize_autoencoder(cfg: DictConfig) -> None: 
    """Generate a summary of the AutoencoderSeq2Seq model"""
    run_dir = get_rundir_name()  
    
    # Create model instance
    model = AutoencoderSeq2Seq(cfg)
    
    # Determine input shape based on config
    # Assuming input is point cloud with shape [num_points, point_dims]
    num_points = cfg.data.num_points
    point_dims = 3  # Typically x, y, z coordinates
    
    # Print model architecture summary
    print(f"\n{'='*50}")
    print(f"AutoencoderSeq2Seq Model Summary")
    print(f"{'='*50}")
    
    # Display model summary
    # Note: input shape should match what the model expects
    input_shape = (num_points, point_dims)
    try:
        summary(model, input_shape)
    except Exception as e:
        print(f"Could not generate summary with torchsummary: {e}")
        # Create a sample input tensor and use forward pass as fallback
        dummy_input = torch.randn(1, num_points, point_dims)
        print("\nModel structure:")
        print(model)
        
        print("\nForward pass with sample data:")
        with torch.no_grad():
            output, latent = model(dummy_input)
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Latent vector shape: {latent.shape}")
    
    print(f"{'='*50}\n")


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="s2s_autoencoder")
def main(cfg):    
    summarize_autoencoder(cfg)


if __name__ == "__main__":    
    sys.argv.append('hydra.run.dir=output/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main() 