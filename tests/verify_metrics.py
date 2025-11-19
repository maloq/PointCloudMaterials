
import torch
import numpy as np
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.getcwd())

# We assume we are in the correct environment now, so we don't mock pl or omegaconf
from src.training_methods.spd.spd_module import ShapePoseDisentanglement
from omegaconf import OmegaConf

def test_metrics_integration():
    # 1. Create a dummy config
    cfg = OmegaConf.create({
        "rotation_mode": "sixd_head",
        "latent_size": 128,
        "load_supervised_checkpoint": False,
        "loss": "chamfer",
        "ortho_scale": 1.0,
        "kl_latent_loss_scale": 0.0,
        "sinkhorn_blur_schedule": {"start": 0.05, "end": 0.001, "start_epoch": 0, "duration_epochs": 10, "enable": True},
        "encoder": {"kwargs": {"latent_size": 128}},
        "decoder": {"kwargs": {}},
        "rot_head": {"kwargs": {}},
        "model": "pointnet", # Dummy model name
        "max_supervised_samples": 100,
        "max_test_samples": 100,
        "enable_expensive_metrics": False
    })

    # 2. Instantiate the module
    # We still mock build_model and build_rot_head to avoid loading actual weights or needing model definitions
    
    with torch.no_grad():
        # Mock build_model and build_rot_head
        import src.training_methods.spd.spd_module as module
        module.build_model = MagicMock(return_value=(MagicMock(), MagicMock()))
        module.build_rot_head = MagicMock(return_value=MagicMock())
        
        model = ShapePoseDisentanglement(cfg)
        
        # Mock self.log to capture calls
        model.log = MagicMock()
        
        # 3. Populate the cache with dummy data
        batch_size = 20
        n_phases = 3
        
        # Create dummy rotations (identity + noise)
        pred_rots = []
        gt_rots = []
        labels = []
        
        for i in range(batch_size):
            # Random rotation
            r_pred = torch.eye(3)
            r_gt = torch.eye(3)
            
            pred_rots.append(r_pred)
            gt_rots.append(r_gt)
            labels.append(i % n_phases)
            
        pred_rots = torch.stack(pred_rots)
        gt_rots = torch.stack(gt_rots)
        labels = torch.tensor(labels)
        
        # Fill cache
        cache = model._supervised_cache["test"]
        cache["rotations"].append(pred_rots)
        cache["gt_rotations"].append(gt_rots)
        cache["phase"].append(labels)
        # Add dummy latents/originals to satisfy other checks if any
        cache["latents"].append(torch.randn(batch_size, 128))
        cache["originals"].append(torch.randn(batch_size, 100, 3)) 
        cache["reconstructions"].append(torch.randn(batch_size, 100, 3))
        
        # 4. Run _log_supervised_metrics
        print("Running _log_supervised_metrics('test')...")
        model._log_supervised_metrics("test")
        
        # 5. Check if metrics were logged
        logged_keys = [call.args[0] for call in model.log.call_args_list]
        print("Logged metrics:", logged_keys)
        
        # Verify specific new metrics
        expected_metrics = [
            "test/rot_global/rot_aligned_error_phase_0",
            "test/rot_sym/rot_sym_error_phase_0"
        ]
        
        all_passed = True
        for metric in expected_metrics:
            if metric in logged_keys:
                print(f"[PASS] Found {metric}")
            else:
                print(f"[FAIL] Missing {metric}")
                all_passed = False
                
        if all_passed:
            print("Verification SUCCESS!")
        else:
            print("Verification FAILED!")
            sys.exit(1)

if __name__ == "__main__":
    test_metrics_integration()
