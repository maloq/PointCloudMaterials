import numpy as np

def pc_normalize(pc, radius=None):
    if radius:
        pc = pc / radius
    else:
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
    return pc

# Simulate data
radius = 7.5
points = np.random.uniform(-radius, radius, (1000, 120, 3)).astype(np.float32)
# Apply sphere mask roughly (to match dataset logic check)
# But assuming uniform usage for approximate stats.

# Normalize
points_norm = pc_normalize(points, radius=radius) # range [-1, 1]

std = np.std(points_norm)
var = np.var(points_norm)
mean_abs = np.mean(np.abs(points_norm))

print(f"Current Range: [{points_norm.min():.2f}, {points_norm.max():.2f}]")
print(f"Current Std: {std:.4f}")
print(f"Current Var: {var:.4f}")
print(f"Mean Abs: {mean_abs:.4f}")

# Target: Unit Variance (Std=1)
scale_factor = 1.0 / std
print(f"Scale factor needed for Unit Variance: {scale_factor:.2f}")
print(f"Resulting Range: [{points_norm.min()*scale_factor:.2f}, {points_norm.max()*scale_factor:.2f}]")
