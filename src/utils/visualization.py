import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def visualize_reconstructions(model, datamodule, save_dir, num_instances=5):
    """
    Visualizes reconstructions for a few instances per class.
    Generates an image with 3 columns: Input, Canonical, Reconstruction.
    """
    print(f"Generating visualizations in {save_dir}...")
    model.eval()
    device = model.device
    
    # Get test dataset
    # We access the dataset directly to pick specific indices
    if hasattr(datamodule, 'test_dataset'):
        dataset = datamodule.test_dataset
    else:
        # Fallback if setup() wasn't called or different structure
        datamodule.setup(stage='test')
        dataset = datamodule.test_dataset
    
    # Group indices by class
    class_indices = {}
    # We iterate through the dataset to find indices for each class
    # This might be slow if dataset is huge and not preloaded, but for ModelNet test set (2k samples) it's fine.
    # If preloaded, it's fast.
    
    print("Grouping samples by class...")
    for idx in range(len(dataset)):
        # dataset[idx] returns (pc, label, class_name)
        # We avoid calling __getitem__ repeatedly if possible, but here we need class names.
        # If dataset has metadata, use it directly to be faster.
        if hasattr(dataset, 'metadata'):
            class_name = dataset.metadata.iloc[idx]['class']
        else:
            _, _, class_name = dataset[idx]
            
        if class_name not in class_indices:
            class_indices[class_name] = []
        class_indices[class_name].append(idx)
        
    os.makedirs(save_dir, exist_ok=True)
    
    for class_name, indices in class_indices.items():
        # Select random instances
        n_samples = min(len(indices), num_instances)
        selected_indices = np.random.choice(indices, n_samples, replace=False)
        
        # Create figure: Rows = instances, Cols = 3
        fig = plt.figure(figsize=(15, 5 * n_samples))
        
        for i, idx in enumerate(selected_indices):
            batch = dataset[idx]
            pc = batch[0] # (N, 3)
            
            # Prepare batch for model
            pc_batch = pc.unsqueeze(0).to(device) # (1, N, 3)
            
            with torch.no_grad():
                # Forward pass
                # Output: inv_z, recon, cano, rot, vq_loss
                inv_z, recon, cano, rot, _ = model(pc_batch)
                
            # Convert to numpy
            pc_np = pc.numpy()
            cano_np = cano[0].cpu().numpy()
            recon_np = recon[0].cpu().numpy()
            
            # Plot Input (Rotated)
            ax = fig.add_subplot(n_samples, 3, i * 3 + 1, projection='3d')
            ax.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2], s=2, c='b', alpha=0.5)
            ax.set_title("Input (Rotated)")
            ax.axis('off')
            # Set equal aspect ratio for 3D
            set_axes_equal(ax)
            
            # Plot Canonical
            ax = fig.add_subplot(n_samples, 3, i * 3 + 2, projection='3d')
            ax.scatter(cano_np[:, 0], cano_np[:, 1], cano_np[:, 2], s=2, c='g', alpha=0.5)
            ax.set_title("Canonical Reconstruction")
            ax.axis('off')
            set_axes_equal(ax)
            
            # Plot Final Recon (Rotated back)
            ax = fig.add_subplot(n_samples, 3, i * 3 + 3, projection='3d')
            ax.scatter(recon_np[:, 0], recon_np[:, 1], recon_np[:, 2], s=2, c='r', alpha=0.5)
            ax.set_title("Rotated Reconstruction")
            ax.axis('off')
            set_axes_equal(ax)
            
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{class_name}_visualization.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved visualization for {class_name} to {save_path}")

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
