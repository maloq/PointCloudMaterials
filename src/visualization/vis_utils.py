
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

#Create version of fuction visualize_predictions, but instead of phase label(which is classification class), use probability and visulaze in using colormap

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