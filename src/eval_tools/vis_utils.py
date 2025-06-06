from scipy.spatial import KDTree
import numpy as np
import plotly.graph_objects as go
import torch



def visualize_predictions(predictions, title: str = "Phase Predictions"):
    """Visualize 3D predictions using plotly.
    
    Args:
        predictions: Nx4 array where each row is (x, y, z, prediction)
        title: Plot title
    """

    fig = go.Figure()
    
    # Create scatter plot for each unique prediction class
    for phase in np.unique(predictions[:, 3]):
        mask = predictions[:, 3] == phase
        points = predictions[mask]
        
        if phase == 0:
            marker_size = 1
        else:
            marker_size = 3
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(size=marker_size),
            name=f'Phase {int(phase)}'
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    
    fig.show()


def visualize_probabilities(points_and_probs, title: str = "Phase Probabilities"):
    """Visualize 3D points with probability values using plotly.
    
    Args:
        points_and_probs: Nx4 array where each row is (x, y, z, probability)
        title: Plot title
    """
    fig = go.Figure()
    # color=points_and_probs[:, 3],

    fig.add_trace(go.Scatter3d(
        x=points_and_probs[:, 0],
        y=points_and_probs[:, 1],
        z=points_and_probs[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            colorscale='Viridis',
            colorbar=dict(title='Probability'),
            opacity=0.8
        ),
        name='Points'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=800,
    )
    
    fig.show()



def find_n_nearest(points, n_neighbors):
    """Find N nearest neighbors for each point using KDTree"""
    tree = KDTree(points)
    # Get indices of n+1 nearest points (first one is the point itself)
    distances, indices = tree.query(points, k=n_neighbors + 1)
    # Return all except first index (which is the point itself)
    return indices[:, 1:]


def plot_point_cloud_3d(points, n_connections=3, title='Point Cloud',
                       num_points=5, color=None):
    """
    Create interactive 3D visualization of point cloud with connections to nearest neighbors.
    
    Args:
        points: np.array of shape (N, 3) containing XYZ coordinates
        n_connections: int, number of nearest neighbors to connect
        title: str, plot title
        num_points: int, size of points
        color: optional array of values for color mapping
    
    Returns:
        plotly.graph_objects.Figure
    """
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    
    nearest_indices = find_n_nearest(points, n_connections)
    
    # Create line segments
    lines_x, lines_y, lines_z = [], [], []
    for i, neighbors in enumerate(nearest_indices):
        for neighbor_idx in neighbors:
            lines_x.extend([points[i, 0], points[neighbor_idx, 0], None])
            lines_y.extend([points[i, 1], points[neighbor_idx, 1], None])
            lines_z.extend([points[i, 2], points[neighbor_idx, 2], None])
    
    fig = go.Figure(data=[
        # Plot points
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=num_points,
                color=color,
                colorscale='Viridis' if color is not None else None,
                opacity=0.8
            ),
            name='Points'
        ),
        go.Scatter3d(
            x=lines_x,
            y=lines_y,
            z=lines_z,
            mode='lines',
            line=dict(color='gray', width=1),
            opacity=0.5,
            name='Connections'
        )
    ])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y', 
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=800,
        height=800,
    )

    return fig



def visualize_original_and_reconstructed(original_points, reconstructed_points, random_sample=False):
    """
    Visualize original and reconstructed point clouds side by side.
    
    Args:
        original_points: Original point cloud array of shape (N, 3)
        reconstructed_points: Reconstructed point cloud array of shape (N, 3)
    """
    # Visualize original points
    if random_sample:
       random_sample_idx = np.random.randint(0, original_points.shape[0])

    fig_original = plot_point_cloud_3d(
        original_points[random_sample_idx][:, :3],  # Take only XYZ coordinates
        n_connections=3,
        title='Original',
        num_points=3, 
    )
    fig_original.update_layout(
        width=600,  
        height=400 
    )
    fig_original.show()

    # Visualize reconstructed points
    fig_reconstructed = plot_point_cloud_3d(
        reconstructed_points[random_sample_idx][:, :3],  # Take only XYZ coordinates
        n_connections=3,
        title='Reconstructed',
        num_points=3,
        color='red'
    )
    fig_reconstructed.update_layout(
        width=600,  
        height=400
    )
    fig_reconstructed.show()
    


def perform_tsne_clustering(latents, labels, chosen_label='liquid', n_clusters=3, 
                            perplexity=20, n_iter=2000, random_state=42,
                            include_other_labels=False, cluster_colors=None):
    """
    Perform t-SNE dimensionality reduction and K-means clustering on latent representations.
    
    Parameters:
    -----------
    latents : numpy.ndarray
        Latent representations of the data
    labels : numpy.ndarray
        Labels for each latent representation
    chosen_label : str, default='liquid'
        Label to perform clustering on
    n_clusters : int, default=3
        Number of clusters for K-means
    perplexity : int, default=20
        Perplexity parameter for t-SNE
    n_iter : int, default=2000
        Maximum number of iterations for t-SNE
    random_state : int, default=42
        Random seed for reproducibility
    include_other_labels : bool, default=False
        Whether to include other labels in the t-SNE visualization
    cluster_colors : list, default=None
        Colors for the clusters. If None, default colors will be used
        
    Returns:
    --------
    tuple
        (tsne_results, cluster_labels, kmeans)
    """
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Default cluster colors if not provided
    if cluster_colors is None:
        cluster_colors = ['coral', 'gold', 'olive']
    
    # Get unique labels
    unique_labels = np.unique(labels)
    
    # Filter latents for the chosen label
    mask = labels == chosen_label
    filtered_latents = latents[mask]
    
    print(f"Performing t-SNE on {'all data' if include_other_labels else 'filtered data'}...")
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=random_state)
    
    if include_other_labels:
        # Perform t-SNE on all data
        tsne_results = tsne.fit_transform(latents)
        filtered_indices = np.where(mask)[0]
    else:
        # Perform t-SNE only on filtered data
        tsne_results = tsne.fit_transform(filtered_latents)
    
    # Perform K-means clustering on filtered latents
    print(f"Performing clustering on label '{chosen_label}' with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(filtered_latents)
    
    # Create a new figure for clustered visualization
    plt.figure(figsize=(10, 8))
    
    if include_other_labels:
        # Create a color map for labels
        color_map = {}
        for i, label in enumerate(unique_labels):
            color_map[label] = plt.cm.tab10(i % 10)
            
        # Visualize original t-SNE plot first for other labels
        for label in unique_labels:
            if label == chosen_label:
                # Don't plot the chosen label yet (will be colored by cluster)
                continue
            indices = np.where(labels == label)
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                        color=color_map[label], label=label,
                        alpha=0.3, edgecolors='w', s=12)
        
        # Plot clusters for the chosen label
        for cluster_id in range(n_clusters):
            cluster_points = filtered_indices[cluster_labels == cluster_id]
            plt.scatter(tsne_results[cluster_points, 0], tsne_results[cluster_points, 1],
                       color=cluster_colors[cluster_id % len(cluster_colors)], 
                       label=f"{chosen_label} - Cluster {cluster_id}",
                       alpha=0.6, edgecolors='w', s=30)
    else:
        # Plot just the clusters from the filtered t-SNE results
        for cluster_id in range(n_clusters):
            plt.scatter(tsne_results[cluster_labels == cluster_id, 0], 
                        tsne_results[cluster_labels == cluster_id, 1],
                        color=cluster_colors[cluster_id % len(cluster_colors)], 
                        label=f"{chosen_label} - Cluster {cluster_id}",
                        alpha=0.6, edgecolors='w', s=30)
    
    # Add plot details
    plt.title(f"t-SNE with Clustering for Label '{chosen_label}'")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print cluster statistics
    for cluster_id in range(n_clusters):
        count = np.sum(cluster_labels == cluster_id)
        percentage = count / len(cluster_labels) * 100
        print(f"Cluster {cluster_id}: {count} points ({percentage:.1f}%)")
    
    return tsne_results, cluster_labels, kmeans


def visualize_cluster_predictions(predictions,
                                   alphas=[0.7]*6,
                                   colors = [
                                            '#FF0000',  # Red
                                            '#00FFFF',  # Cyan
                                            '#0000FF',  # Blue
                                            '#FFFF00',  # Yellow
                                            '#FF00FF',  # Magenta
                                            '#FFA500',  # Orange
                                            '#00FF00',  # Lime Green
                                            '#800080',  # Purple
                                            ],
                                  title: str = "Phase Predictions"):


    fig = go.Figure()

    # Define a list of fixed colors
    
    
    
    # Create scatter plot for each unique prediction class
    for i, phase in enumerate(np.unique(predictions[:, 3])):
        mask = predictions[:, 3] == phase
        points = predictions[mask]
        
        if phase == 0:
            marker_size = 1
        else:
            marker_size = 1
        
        # Select color from the list, cycling through if more than 8 phases
        color = colors[i % len(colors)]
        # Select alpha from the list, cycling through if more than 8 phases
        alpha = alphas[i % len(alphas)]
        
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=marker_size, 
                color=color, 
                opacity=alpha
            ),
            name=f'Phase {int(phase)}'
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.85, y=1.85, z=0.65) 
            )
        ),
        width=800,
        height=600
    )

    return fig