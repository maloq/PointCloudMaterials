import argparse
import os
import sys
import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.append(os.getcwd())

from src.training_methods.spd.spd_experiments_module import SPDExperimentsModule
from src.data_utils.modelnet_loader import ModelNetDataModule
from src.utils.visualization import visualize_reconstructions
from src.run_modelnet_experiments import load_config

def main():
    parser = argparse.ArgumentParser(description="Visualize reconstructions from a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file (.ckpt)")
    parser.add_argument("--output_dir", type=str, default="output/visualizations_manual", help="Directory to save visualizations")
    parser.add_argument("--num_instances", type=int, default=10, help="Number of instances per class to visualize")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name (optional, for fallback config loading)")
    parser.add_argument("--n_classes", type=int, default=3, help="Number of classes (optional, for fallback config loading)")
    
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    print(f"Loading model from {args.checkpoint}...")
    
    # Try loading model
    model = None
    cfg = None
    
    try:
        # First try loading directly
        # This works if hparams were saved correctly
        model = SPDExperimentsModule.load_from_checkpoint(args.checkpoint)
        print("Model loaded successfully from checkpoint.")
        
        # Try to retrieve config from hparams
        if hasattr(model, "hparams") and model.hparams:
            # Check if hparams is the config itself or contains it
            # Usually save_hyperparameters(cfg) makes hparams a dict-like object with cfg content
            # We convert it to DictConfig if it's a dict for easier access
            if isinstance(model.hparams, (dict, argparse.Namespace)):
                cfg = OmegaConf.create(model.hparams)
            else:
                cfg = model.hparams
                
            # Check if it looks valid (has data section)
            if not hasattr(cfg, 'data') and 'data' not in cfg:
                print("Warning: Checkpoint hparams do not contain 'data' section.")
                cfg = None
        
    except Exception as e:
        print(f"Direct load failed: {e}")
        
    # Fallback if model load failed or config is missing
    if model is None or cfg is None:
        if args.experiment:
            print(f"Loading config for experiment: {args.experiment}")
            cfg, _ = load_config(args.experiment, args.n_classes)
            
            if model is None:
                print("Loading model with explicit config...")
                model = SPDExperimentsModule.load_from_checkpoint(args.checkpoint, cfg=cfg)
        else:
            if model is None:
                print("Error: Could not load model and no experiment name provided.")
                sys.exit(1)
            else:
                print("Error: Could not retrieve config from checkpoint and no experiment name provided.")
                sys.exit(1)

    model.eval()
    
    # Setup DataModule
    print("Initializing DataModule...")
    try:
        dm = ModelNetDataModule(cfg)
        dm.setup(stage='test')
    except Exception as e:
        print(f"Error initializing DataModule: {e}")
        sys.exit(1)
    
    print(f"Visualizing to {args.output_dir}...")
    visualize_reconstructions(model, dm, args.output_dir, num_instances=args.num_instances)
    print("Done!")

if __name__ == "__main__":
    main()
