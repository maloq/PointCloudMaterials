import sys
import os
import numpy as np
sys.path.append(os.getcwd())
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from src.clustering.cluster_ae_latent import predict_and_save_latent
from src.autoencoder.autoencoder_module import PointNetAutoencoder
from src.utils.model_utils import load_model_from_checkpoint
from hydra import compose, initialize

import random
import torch
import warnings
warnings.filterwarnings("ignore")
print(f'Running from {os.getcwd()}')

def get_config_path_from_checkpoint(checkpoint_path):
    config_path_chekpoint = os.path.join(*checkpoint_path.split('/')[:-1], '.hydra')
    if not os.path.exists(config_path_chekpoint):
        config_path = '../../configs' if 'src' in os.getcwd() else 'configs' 
        print(f"Config in {config_path_chekpoint} not found, using default location {config_path}")
        config_name = 's2s_autoencoder'
    else:
        config_path =  config_path_chekpoint
        config_name = 'config'

    return config_path, config_name


# Encapsulate the main logic into a function
def run_clustering_pipeline(checkpoint_path: str,
                            save_folder: str,
                            liquid_file_path: str,
                            crystal_file_path: str,
                            model_class: str,
                            cuda_device: int = 0,
                            max_samples: int = None,
                            add_parent_dir=False):
    """
    Runs the clustering pipeline: loads model, generates/saves latents.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        save_path (str): Path to save the latent-label pairs (.npy).
        liquid_file_path (str): Path to the liquid dataset file.
        crystal_file_path (str): Path to the crystal dataset file.
        model_class (str): Type of model ('Autoencoder' or 'Seq2Seq').
        cuda_device (int): GPU device index to use.
        max_samples (int, optional): Maximum number of samples to process. Defaults to None.
    """
    config_path, config_name = get_config_path_from_checkpoint(checkpoint_path)

    with initialize(version_base=None, config_path='../../' + config_path ):
        print(f"Loading config from {config_path}/{config_name}")
        cfg = compose(config_name=config_name)

    device = f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = None
    if model_class == 'Autoencoder':
        model = load_model_from_checkpoint(checkpoint_path, cfg, device=device, module=PointNetAutoencoder)
    else:
        raise ValueError(f"Unknown model_class: {model_class}. Use 'Autoencoder' or ...")

    if model is None:
        raise RuntimeError("Failed to load the model.")

    # Predict and save latent vectors
    predict_and_save_latent(cfg=cfg,  # Pass the loaded hydra config
                            model=model,
                            liquid_file_path=liquid_file_path,
                            crystal_file_path=crystal_file_path,
                            device=device,
                            save_folder=save_folder,
                            model_class=model_class, # Pass model_class to predict_and_save_latent
                            max_samples=max_samples) # Pass max_samples


# Example usage (optional, can be removed or put under if __name__ == '__main__')
if __name__ == '__main__':
    checkpoint_path = 'output/2025-04-15/05-09-11/pointnet-epoch=7999-val_loss=0.04.ckpt'
    save_folder = 'output'
    liquid_file_path = 'datasets/Al/inherent_configurations_off/166ps.off'
    crystal_file_path = 'datasets/Al/inherent_configurations_off/240ps.off'
    model_class = 'Seq2Seq' # Or 'Autoencoder'

    run_clustering_pipeline(checkpoint_path=checkpoint_path,
                            save_folder=save_folder,
                            liquid_file_path=liquid_file_path,
                            crystal_file_path=crystal_file_path,
                            model_class=model_class,
                            cuda_device=0,
                            max_samples=1000,
                            add_parent_dir=True) 