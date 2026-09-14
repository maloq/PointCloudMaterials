"""Legacy sampling imports for restored research scripts."""

from src.data.static_sources import read_off_file
from src.data.sampling import (
    _resolve_drop_func,
    _resolve_edge_drop_layers,
    calculate_center,
    calculate_stride,
    compute_dimensions,
    drop_points_farthest,
    drop_points_fps,
    farthest_point_sample,
    generate_samples,
    get_min_max_coords,
    get_random_samples,
    get_regular_samples,
)
