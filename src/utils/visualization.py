import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from src.loss.reconstruction_loss import chamfer_distance

def visualize_reconstructions(model, datamodule, save_dir, num_instances=10):
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
        # Optimization: Use class_names list directly if available (ModelNetFastDataset)
        if hasattr(dataset, 'class_names'):
            class_name = dataset.class_names[idx]
        elif hasattr(dataset, 'metadata'):
            class_name = dataset.metadata.iloc[idx]['class']
        else:
            _, _, class_name = dataset[idx]
            
        if class_name not in class_indices:
            class_indices[class_name] = []
        class_indices[class_name].append(idx)
        
    print(f"Found {len(class_indices)} classes with {len(dataset)} samples total.")
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
            
            # Normalize inputs to match training (centering + unit sphere)
            centroid = torch.mean(pc_batch, dim=1, keepdim=True)
            pc_batch = pc_batch - centroid
            m = torch.max(torch.sqrt(torch.sum(pc_batch**2, dim=2, keepdim=True)), dim=1, keepdim=True)[0]
            m = torch.maximum(m, torch.tensor(1e-6, device=device))
            pc_batch = pc_batch / m
            
            with torch.no_grad():
                # Forward pass
                # Output: inv_z, recon, cano, rot, vq_loss
                inv_z, recon, cano, rot, _ = model(pc_batch)
                
                
            # Calculate Chamfer Distances
            # Input vs Canonical (Input is rotated, Canonical is aligned)
            # This distance might be large if rotation is significant.
            cd_cano, _ = chamfer_distance(pc_batch, cano, squared=False, point_reduction='mean')
            
            # Input vs Rotated Reconstruction (should match well)
            cd_recon, _ = chamfer_distance(pc_batch, recon, squared=False, point_reduction='mean')
            
            # Convert to numpy (use normalized input for visualization consistency)
            pc_np = pc_batch[0].cpu().numpy()
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
            ax.set_title(f"Canonical Reconstruction\nCD: {cd_cano.item():.4f}")
            ax.axis('off')
            set_axes_equal(ax)
            
            # Plot Final Recon (Rotated back)
            ax = fig.add_subplot(n_samples, 3, i * 3 + 3, projection='3d')
            ax.scatter(recon_np[:, 0], recon_np[:, 1], recon_np[:, 2], s=2, c='r', alpha=0.5)
            ax.set_title(f"Rotated Reconstruction\nCD: {cd_recon.item():.4f}")
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

def save_point_cloud_ply(points, file_path):
    """
    Save point cloud to PLY file.
    points: (N, 3) numpy array
    """
    with open(file_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
