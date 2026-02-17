"""
Advanced Visualization Utilities for Synthetic Atomistic Datasets v2.0

Improvements over v1:
- Fixed camera angle for diagonal cuts (looks AT the cut surface)
- Radial Distribution Function (RDF) plot
- Coordination number distribution
- Bond angle distribution
- Phase composition statistics (pie chart + metrics)
- Grain size distribution histogram
- XY/XZ/YZ slice views
- Interface characterization
- Misorientation angle visualization
- Summary statistics panel
- Voronoi analysis for local environments
"""

from __future__ import annotations

import pathlib
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import Counter, defaultdict

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree, Voronoi, Delaunay
from scipy.ndimage import gaussian_filter1d

# Diagonal cut direction (normalized [1,1,1])
_DIAGONAL_DIRECTION = np.array([1.0, 1.0, 1.0], dtype=float) / np.sqrt(3.0)
_DEFAULT_GRAY_RGBA = tuple(mcolors.to_rgba("gray"))

# Color schemes
PHASE_CMAP = cm.get_cmap("Set2")
GRAIN_CMAP = cm.get_cmap("gist_ncar")
CRYSTAL_COLOR = "#4ECDC4"
LIQUID_COLOR = "#FF6B6B"
INTERFACE_COLOR = "#FFE66D"


def generate_visualizations(
    global_cfg: Any,
    grains: Sequence[Dict[str, Any]],
    atoms: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
    rng: np.random.Generator,
    output_dir: pathlib.Path,
) -> None:
    """
    Create comprehensive diagnostic visualizations for the generated dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract arrays
    atoms_array = np.asarray([atom["position"] for atom in atoms], dtype=float)
    phases = np.asarray([atom["phase_id"] for atom in atoms], dtype=str)
    grain_ids = np.array(
        [(-1 if atom.get("grain_id") is None else int(atom["grain_id"])) for atom in atoms],
        dtype=int,
    )
    
    # Build color maps
    color_map = _build_phase_color_map(phases)
    grain_color_map = _build_grain_color_map(grains)
    
    print("  Generating visualizations...")
    
    # 1. Global structure (full view)
    print("    • Global structure")
    _render_global_structure(
        global_cfg, grains, atoms, atoms_array, phases, grain_ids,
        metadata, grain_color_map, color_map,
        output_dir / "figure_global.png", rng
    )
    
    # 2. Diagonal cut view (FIXED CAMERA - looks AT the cut)
    print("    • Diagonal cut view")
    _render_global_structure_diagonal_cut(
        global_cfg, grains, atoms, atoms_array, phases, grain_ids,
        metadata, grain_color_map, color_map,
        output_dir / "figure_global_diagonal_cut.png", rng,
        # FIXED: Use positive direction to look AT the cut surface
        view_angles=_view_from_vector(_DIAGONAL_DIRECTION),
    )
    
    # 3. Statistical analysis panel (NEW)
    print("    • Statistical analysis")
    _render_statistics_panel(
        global_cfg, atoms_array, phases, grain_ids, grains, metadata,
        output_dir / "figure_statistics.png"
    )
    
    # 4. RDF and coordination analysis (NEW)
    print("    • RDF and coordination")
    _render_structural_analysis(
        global_cfg, atoms_array, phases,
        output_dir / "figure_structural_analysis.png", rng
    )
    
    # 5. Slice views (NEW)
    print("    • Slice views")
    _render_slice_views(
        global_cfg, atoms_array, phases, color_map,
        output_dir / "figure_slices.png"
    )
    
    # 6. Close-up view
    print("    • Close-up view")
    _render_closeup_view(
        global_cfg, atoms, atoms_array, phases, color_map,
        output_dir / "figure_closeup.png", rng
    )
    
    # 7. Local galleries
    print("    • Local galleries")
    _render_local_galleries(
        global_cfg, atoms, atoms_array, phases, metadata, output_dir, rng
    )
    
    # 8. Grain analysis (NEW)
    print("    • Grain analysis")
    _render_grain_analysis(
        global_cfg, grains, atoms_array, grain_ids, phases, grain_color_map,
        output_dir / "figure_grain_analysis.png"
    )
    
    # 9. Interface characterization (NEW)
    print("    • Interface characterization")
    _render_interface_analysis(
        global_cfg, atoms_array, phases, grain_ids, grains, color_map,
        output_dir / "figure_interfaces.png", rng
    )
    
    print("  Visualizations complete")


# =============================================================================
# STRUCTURAL ANALYSIS (RDF, Coordination, Bond Angles)
# =============================================================================

def _compute_rdf(
    positions: np.ndarray,
    box_size: float,
    r_max: Optional[float] = None,
    n_bins: int = 100,
    sample_size: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radial distribution function g(r)."""
    n = len(positions)
    if n < 2:
        return np.zeros(n_bins), np.zeros(n_bins)
    
    if r_max is None:
        r_max = box_size / 2
    
    rng = rng or np.random.default_rng()
    
    # Build spatial index
    tree = cKDTree(positions)
    
    # Sample atoms for RDF calculation
    actual_sample = min(sample_size, n)
    sample_indices = rng.choice(n, actual_sample, replace=False)
    
    # Histogram
    r_bins = np.linspace(0, r_max, n_bins + 1)
    dr = r_bins[1] - r_bins[0]
    hist = np.zeros(n_bins)
    
    for i in sample_indices:
        neighbors = tree.query_ball_point(positions[i], r_max)
        for j in neighbors:
            if j != i:
                dist = np.linalg.norm(positions[j] - positions[i])
                if 0 < dist < r_max:
                    bin_idx = int(dist / dr)
                    if 0 <= bin_idx < n_bins:
                        hist[bin_idx] += 1
    
    # Normalize to g(r)
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    shell_volumes = 4 * np.pi * r_centers**2 * dr
    rho = n / box_size**3
    expected = actual_sample * rho * shell_volumes
    
    g_r = np.divide(hist, expected, where=expected > 0, out=np.ones_like(hist))
    
    return r_centers, g_r


def _compute_coordination(
    positions: np.ndarray,
    cutoff: float,
    sample_size: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Compute coordination numbers for atoms."""
    n = len(positions)
    if n < 2:
        return np.array([0])
    
    rng = rng or np.random.default_rng()
    tree = cKDTree(positions)
    
    actual_sample = min(sample_size, n)
    sample_indices = rng.choice(n, actual_sample, replace=False)
    
    coordinations = []
    for i in sample_indices:
        neighbors = tree.query_ball_point(positions[i], cutoff)
        coord = len(neighbors) - 1  # Exclude self
        coordinations.append(coord)
    
    return np.array(coordinations)


def _compute_bond_angles(
    positions: np.ndarray,
    cutoff: float,
    sample_size: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Compute bond angle distribution."""
    n = len(positions)
    if n < 3:
        return np.array([])
    
    rng = rng or np.random.default_rng()
    tree = cKDTree(positions)
    
    actual_sample = min(sample_size, n)
    sample_indices = rng.choice(n, actual_sample, replace=False)
    
    angles = []
    for i in sample_indices:
        neighbors = tree.query_ball_point(positions[i], cutoff)
        neighbors = [j for j in neighbors if j != i]
        
        if len(neighbors) < 2:
            continue
        
        # Compute angles between neighbor pairs
        center = positions[i]
        for k, j1 in enumerate(neighbors[:6]):  # Limit to first 6 neighbors
            for j2 in neighbors[k+1:7]:
                v1 = positions[j1] - center
                v2 = positions[j2] - center
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)
    
    return np.array(angles)


def _render_structural_analysis(
    global_cfg: Any,
    positions: np.ndarray,
    phases: np.ndarray,
    output_path: pathlib.Path,
    rng: np.random.Generator,
) -> None:
    """Render RDF, coordination, and bond angle analysis."""
    if len(positions) < 10:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    avg_nn = global_cfg.avg_nn_dist
    L = global_cfg.L
    
    # Identify phase types
    unique_phases = np.unique(phases)
    crystal_phases = [p for p in unique_phases if 'crystal' in p.lower() or 'bcc' in p.lower() or 'fcc' in p.lower()]
    liquid_phases = [p for p in unique_phases if 'liquid' in p.lower() or 'amorphous' in p.lower()]
    
    # --- Panel 1: Overall RDF ---
    ax1 = axes[0, 0]
    r, gr = _compute_rdf(positions, L, r_max=4*avg_nn, rng=rng)
    ax1.plot(r / avg_nn, gr, 'b-', linewidth=2, label='All atoms')
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(1.0, color='red', linestyle=':', alpha=0.5, label='r = avg_nn_dist')
    ax1.set_xlabel('r / avg_nn_dist')
    ax1.set_ylabel('g(r)')
    ax1.set_title('Radial Distribution Function')
    ax1.legend()
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, max(3.5, np.max(gr) * 1.1))
    ax1.grid(True, alpha=0.3)
    
    # --- Panel 2: Phase-specific RDF ---
    ax2 = axes[0, 1]
    colors = {'crystal': CRYSTAL_COLOR, 'liquid': LIQUID_COLOR}
    
    for phase_type, phase_list in [('crystal', crystal_phases), ('liquid', liquid_phases)]:
        if not phase_list:
            continue
        mask = np.isin(phases, phase_list)
        if np.sum(mask) > 50:
            phase_positions = positions[mask]
            r, gr = _compute_rdf(phase_positions, L, r_max=4*avg_nn, rng=rng)
            ax2.plot(r / avg_nn, gr, linewidth=2, label=f'{phase_type.title()} phases',
                    color=colors.get(phase_type, 'gray'))
    
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('r / avg_nn_dist')
    ax2.set_ylabel('g(r)')
    ax2.set_title('RDF by Phase Type')
    ax2.legend()
    ax2.set_xlim(0, 4)
    ax2.grid(True, alpha=0.3)
    
    # --- Panel 3: Coordination histogram ---
    ax3 = axes[0, 2]
    cutoff = 1.4 * avg_nn  # First minimum in g(r)
    coords = _compute_coordination(positions, cutoff, rng=rng)
    
    bins = np.arange(0, 20) - 0.5
    ax3.hist(coords, bins=bins, color='steelblue', edgecolor='black', alpha=0.7, density=True)
    ax3.axvline(np.mean(coords), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(coords):.1f}')
    ax3.set_xlabel('Coordination Number')
    ax3.set_ylabel('Probability')
    ax3.set_title(f'Coordination Distribution (cutoff={cutoff:.2f}Å)')
    ax3.legend()
    ax3.set_xlim(0, 18)
    ax3.grid(True, alpha=0.3)
    
    # --- Panel 4: Bond angle distribution ---
    ax4 = axes[1, 0]
    angles = _compute_bond_angles(positions, cutoff, rng=rng)
    
    if len(angles) > 0:
        ax4.hist(angles, bins=50, range=(0, 180), color='coral', 
                edgecolor='black', alpha=0.7, density=True)
        # Mark characteristic angles
        ax4.axvline(60, color='blue', linestyle=':', alpha=0.7, label='60° (FCC/ICO)')
        ax4.axvline(90, color='green', linestyle=':', alpha=0.7, label='90° (BCC/FCC)')
        ax4.axvline(109.5, color='purple', linestyle=':', alpha=0.7, label='109.5° (tetrahedral)')
    ax4.set_xlabel('Bond Angle (degrees)')
    ax4.set_ylabel('Probability')
    ax4.set_title('Bond Angle Distribution')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.set_xlim(0, 180)
    ax4.grid(True, alpha=0.3)
    
    # --- Panel 5: Coordination by phase ---
    ax5 = axes[1, 1]
    phase_coords = {}
    for phase_type, phase_list in [('Crystal', crystal_phases), ('Liquid', liquid_phases)]:
        if not phase_list:
            continue
        mask = np.isin(phases, phase_list)
        if np.sum(mask) > 50:
            phase_positions = positions[mask]
            coords = _compute_coordination(phase_positions, cutoff, sample_size=2000, rng=rng)
            phase_coords[phase_type] = coords
    
    if phase_coords:
        labels = list(phase_coords.keys())
        data = [phase_coords[l] for l in labels]
        bp = ax5.boxplot(data, labels=labels, patch_artist=True)
        colors = [CRYSTAL_COLOR if 'Crystal' in l else LIQUID_COLOR for l in labels]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax5.set_ylabel('Coordination Number')
    ax5.set_title('Coordination by Phase Type')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # --- Panel 6: RDF comparison with ideal structures ---
    ax6 = axes[1, 2]
    r, gr = _compute_rdf(positions, L, r_max=4*avg_nn, n_bins=150, rng=rng)
    gr_smooth = gaussian_filter1d(gr, sigma=2)
    ax6.fill_between(r / avg_nn, 0, gr_smooth, alpha=0.3, color='blue')
    ax6.plot(r / avg_nn, gr_smooth, 'b-', linewidth=2, label='Measured')
    
    # Add ideal peak positions for reference
    # BCC: r/a = 0.866, 1.0, 1.414, 1.658, 1.732...
    # FCC: r/a = 0.707, 1.0, 1.225, 1.414, 1.581...
    bcc_peaks = [0.866, 1.0, 1.414, 1.658, 1.732]
    fcc_peaks = [0.707, 1.0, 1.225, 1.414, 1.581]
    
    for i, peak in enumerate(bcc_peaks[:3]):
        ax6.axvline(peak, color='red', linestyle='--', alpha=0.4, 
                   label='BCC peaks' if i == 0 else '')
    for i, peak in enumerate(fcc_peaks[:3]):
        ax6.axvline(peak, color='green', linestyle=':', alpha=0.4,
                   label='FCC peaks' if i == 0 else '')
    
    ax6.set_xlabel('r / avg_nn_dist')
    ax6.set_ylabel('g(r)')
    ax6.set_title('RDF with Crystal Reference Peaks')
    ax6.legend(fontsize=8)
    ax6.set_xlim(0, 3)
    ax6.grid(True, alpha=0.3)
    
    fig.suptitle('Structural Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# =============================================================================
# STATISTICS PANEL
# =============================================================================

def _render_statistics_panel(
    global_cfg: Any,
    positions: np.ndarray,
    phases: np.ndarray,
    grain_ids: np.ndarray,
    grains: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
    output_path: pathlib.Path,
) -> None:
    """Render summary statistics panel."""
    fig = plt.figure(figsize=(16, 10))
    
    # --- Phase pie chart ---
    ax1 = fig.add_subplot(2, 3, 1)
    phase_counts = Counter(phases)
    labels = list(phase_counts.keys())
    sizes = list(phase_counts.values())
    colors = [_phase_to_color(p) for p in labels]
    
    # Simplify labels
    short_labels = [p.replace('crystal_', '').replace('liquid_', 'liq_')[:15] for p in labels]
    
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=short_labels, autopct='%1.1f%%',
        colors=colors, explode=[0.02]*len(sizes),
        textprops={'fontsize': 8}
    )
    ax1.set_title('Phase Composition')
    
    # --- Phase bar chart ---
    ax2 = fig.add_subplot(2, 3, 2)
    x = np.arange(len(labels))
    bars = ax2.bar(x, sizes, color=colors, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Number of Atoms')
    ax2.set_title('Atoms per Phase')
    
    # Add count labels on bars
    for bar, count in zip(bars, sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*max(sizes),
                f'{count}', ha='center', va='bottom', fontsize=7)
    
    # --- Grain size distribution ---
    ax3 = fig.add_subplot(2, 3, 3)
    grain_sizes = Counter(grain_ids[grain_ids >= 0])
    if grain_sizes:
        sizes_array = np.array(list(grain_sizes.values()))
        ax3.hist(sizes_array, bins=min(30, len(grain_sizes)), 
                color='steelblue', edgecolor='black', alpha=0.7)
        ax3.axvline(np.mean(sizes_array), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(sizes_array):.0f}')
        ax3.axvline(np.median(sizes_array), color='orange', linestyle=':',
                   label=f'Median: {np.median(sizes_array):.0f}')
        ax3.legend()
    ax3.set_xlabel('Atoms per Grain')
    ax3.set_ylabel('Count')
    ax3.set_title('Grain Size Distribution')
    ax3.grid(True, alpha=0.3)
    
    # --- Text statistics panel ---
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axis('off')
    
    L = global_cfg.L
    n_atoms = len(positions)
    density = n_atoms / L**3
    
    stats_text = [
        f"{'='*40}",
        f"DATASET STATISTICS",
        f"{'='*40}",
        f"",
        f"Box size: {L:.1f} × {L:.1f} × {L:.1f} Å³",
        f"Volume: {L**3:.1f} Å³",
        f"",
        f"Total atoms: {n_atoms:,}",
        f"Target density: {global_cfg.rho_target:.4f} atoms/Å³",
        f"Actual density: {density:.4f} atoms/Å³",
        f"Avg NN distance: {global_cfg.avg_nn_dist:.3f} Å",
        f"",
        f"Number of grains: {len(grains)}",
        f"Number of phases: {len(phase_counts)}",
        f"",
    ]
    
    # Add perturbation info if available
    perturbs = metadata.get("perturbations", {})
    if perturbs:
        stats_text.append("PERTURBATIONS:")
        rot_bubbles = perturbs.get("rotation_bubbles", [])
        if rot_bubbles:
            stats_text.append(f"  Rotation bubbles: {len(rot_bubbles)}")
        dropouts = perturbs.get("dropouts", {})
        if dropouts:
            stats_text.append(f"  Vacancies: {dropouts.get('count', 0)}")
        density_bubbles = perturbs.get("density_bubbles", [])
        if density_bubbles:
            stats_text.append(f"  Density bubbles: {len(density_bubbles)}")
    
    ax4.text(0.05, 0.95, '\n'.join(stats_text), transform=ax4.transAxes,
            fontfamily='monospace', fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # --- Grain phase breakdown ---
    ax5 = fig.add_subplot(2, 3, 5)
    grain_phases = {}
    for grain in grains:
        phase = grain.get('base_phase_id', 'unknown')
        grain_phases[phase] = grain_phases.get(phase, 0) + 1
    
    if grain_phases:
        labels = list(grain_phases.keys())
        values = list(grain_phases.values())
        short_labels = [p.replace('crystal_', '').replace('liquid_', 'liq_')[:12] for p in labels]
        colors = [_phase_to_color(p) for p in labels]
        
        ax5.barh(short_labels, values, color=colors, edgecolor='black')
        ax5.set_xlabel('Number of Grains')
        ax5.set_title('Grains per Phase')
    
    # --- Density profile along Z ---
    ax6 = fig.add_subplot(2, 3, 6)
    n_bins = 50
    z_bins = np.linspace(0, L, n_bins + 1)
    z_counts, _ = np.histogram(positions[:, 2], bins=z_bins)
    z_centers = 0.5 * (z_bins[1:] + z_bins[:-1])
    bin_volume = L * L * (z_bins[1] - z_bins[0])
    z_density = z_counts / bin_volume
    
    ax6.plot(z_centers, z_density, 'b-', linewidth=2)
    ax6.fill_between(z_centers, 0, z_density, alpha=0.3)
    ax6.axhline(global_cfg.rho_target, color='red', linestyle='--', 
               label=f'Target: {global_cfg.rho_target:.4f}')
    ax6.set_xlabel('Z position (Å)')
    ax6.set_ylabel('Local density (atoms/Å³)')
    ax6.set_title('Density Profile along Z')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    fig.suptitle('Dataset Statistics', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# =============================================================================
# SLICE VIEWS
# =============================================================================

def _render_slice_views(
    global_cfg: Any,
    positions: np.ndarray,
    phases: np.ndarray,
    color_map: Dict[str, Any],
    output_path: pathlib.Path,
    slice_thickness: float = 3.0,
) -> None:
    """Render XY, XZ, YZ slice views through the center with improved clarity."""
    L = global_cfg.L
    center = L / 2
    avg_nn = global_cfg.avg_nn_dist
    
    # Thinner slices for clarity (about 1 atom layer)
    slice_thickness = min(slice_thickness, 1.5 * avg_nn)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    
    slices = [
        ('XY', 2, center, (0, 1), 'Z'),  # Z-slice
        ('XZ', 1, center, (0, 2), 'Y'),  # Y-slice
        ('YZ', 0, center, (1, 2), 'X'),  # X-slice
    ]
    
    # Create legend for phases
    unique_phases = np.unique(phases)
    phase_handles = []
    phase_labels = []
    
    for col, (name, axis, pos, plot_axes, slice_axis) in enumerate(slices):
        # Get atoms within slice
        mask = np.abs(positions[:, axis] - pos) < slice_thickness / 2
        slice_positions = positions[mask]
        slice_phases = phases[mask]
        
        # Top row: phase-colored with clear atom visualization
        ax_top = axes[0, col]
        ax_top.set_facecolor('#f8f8f8')
        
        if len(slice_positions) > 0:
            # Smart subsampling to keep structure visible
            max_points = 2500
            if len(slice_positions) > max_points:
                # Use stratified sampling to preserve phase distribution
                idx = np.random.choice(len(slice_positions), max_points, replace=False)
                slice_positions_sample = slice_positions[idx]
                slice_phases_sample = slice_phases[idx]
            else:
                slice_positions_sample = slice_positions
                slice_phases_sample = slice_phases
            
            # Plot each phase with distinct appearance
            for phase in np.unique(slice_phases_sample):
                phase_mask = slice_phases_sample == phase
                pts = slice_positions_sample[phase_mask]
                scatter = ax_top.scatter(
                    pts[:, plot_axes[0]], pts[:, plot_axes[1]],
                    c=[color_map.get(phase, _DEFAULT_GRAY_RGBA)],
                    s=20, alpha=0.85, edgecolors='black', linewidths=0.35,
                    label=phase.replace('crystal_', '').replace('liquid_', 'liq_')[:12]
                )
                # Collect handles for legend (only once)
                if col == 0 and phase not in phase_labels:
                    phase_handles.append(scatter)
                    phase_labels.append(phase.replace('crystal_', '').replace('liquid_', 'liq_')[:12])
        
        ax_top.set_xlim(0, L)
        ax_top.set_ylim(0, L)
        ax_top.set_aspect('equal')
        ax_top.set_title(f'{name} Plane ({slice_axis}={pos:.1f}Å)\n{len(slice_positions)} atoms', fontsize=10)
        ax_top.set_xlabel(['X (Å)', 'X (Å)', 'Y (Å)'][col])
        ax_top.set_ylabel(['Y (Å)', 'Z (Å)', 'Z (Å)'][col])
        ax_top.grid(True, alpha=0.25, linestyle='--')
        
        # Bottom row: density heatmap with better visualization
        ax_bot = axes[1, col]
        if len(slice_positions) > 0:
            # Use adaptive binning based on atom density
            n_bins = min(100, max(30, int(L / avg_nn)))
            heatmap, xedges, yedges = np.histogram2d(
                slice_positions[:, plot_axes[0]],
                slice_positions[:, plot_axes[1]],
                bins=n_bins, range=[[0, L], [0, L]]
            )
            
            # Normalize by bin area to get density
            bin_area = (L / n_bins) ** 2
            density_map = heatmap / bin_area
            
            im = ax_bot.imshow(
                density_map.T, origin='lower', extent=[0, L, 0, L],
                cmap='plasma', aspect='equal', interpolation='gaussian'
            )
            cbar = plt.colorbar(im, ax=ax_bot, shrink=0.8)
            cbar.set_label('ρ (atoms/Å²)', fontsize=9)
        
        ax_bot.set_title(f'{name} Density Map', fontsize=10)
        ax_bot.set_xlabel(['X (Å)', 'X (Å)', 'Y (Å)'][col])
        ax_bot.set_ylabel(['Y (Å)', 'Z (Å)', 'Z (Å)'][col])
    
    # Add legend for phases
    if phase_handles:
        fig.legend(phase_handles, phase_labels, loc='upper right', 
                  bbox_to_anchor=(0.99, 0.98), fontsize=8, framealpha=0.9,
                  title='Phases', title_fontsize=9)
    
    fig.suptitle(f'Cross-Section Views (slice thickness ≈ {slice_thickness:.1f}Å)', 
                fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# =============================================================================
# GRAIN ANALYSIS
# =============================================================================

def _render_grain_analysis(
    global_cfg: Any,
    grains: Sequence[Dict[str, Any]],
    positions: np.ndarray,
    grain_ids: np.ndarray,
    phases: np.ndarray,
    grain_color_map: Dict[int, Any],
    output_path: pathlib.Path,
) -> None:
    """Render grain-level analysis."""
    fig = plt.figure(figsize=(16, 10))
    
    L = global_cfg.L
    
    # --- 3D grain seeds with Voronoi regions ---
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    seed_positions = np.array([g['seed_position'] for g in grains])
    grain_ids_list = [g['grain_id'] for g in grains]
    colors = [grain_color_map.get(gid, _DEFAULT_GRAY_RGBA) for gid in grain_ids_list]
    
    ax1.scatter(seed_positions[:, 0], seed_positions[:, 1], seed_positions[:, 2],
               c=colors, s=100, edgecolors='black', linewidths=1, marker='o')
    _set_cube_axes(ax1, L)
    ax1.set_title('Grain Seeds')
    
    # --- Grain size vs position ---
    ax2 = fig.add_subplot(2, 3, 2)
    grain_sizes = {}
    for gid in np.unique(grain_ids[grain_ids >= 0]):
        grain_sizes[gid] = np.sum(grain_ids == gid)
    
    if grain_sizes and len(seed_positions) > 0:
        sizes = [grain_sizes.get(gid, 0) for gid in grain_ids_list]
        distances = np.linalg.norm(seed_positions - L/2, axis=1)
        ax2.scatter(distances, sizes, c=colors, s=60, edgecolors='black', alpha=0.7)
        ax2.set_xlabel('Distance from Center (Å)')
        ax2.set_ylabel('Grain Size (atoms)')
        ax2.set_title('Grain Size vs Position')
        ax2.grid(True, alpha=0.3)
    
    # --- Misorientation angle distribution ---
    ax3 = fig.add_subplot(2, 3, 3)
    misorientations = []
    for i, g1 in enumerate(grains):
        for g2 in grains[i+1:]:
            R1 = np.array(g1['base_rotation'])
            R2 = np.array(g2['base_rotation'])
            dR = R1.T @ R2
            # Misorientation angle from rotation matrix
            trace = np.clip(np.trace(dR), -1, 3)
            angle = np.degrees(np.arccos((trace - 1) / 2))
            misorientations.append(angle)
    
    if misorientations:
        ax3.hist(misorientations, bins=30, range=(0, 90), 
                color='purple', edgecolor='black', alpha=0.7)
        ax3.axvline(15, color='red', linestyle='--', label='Low-angle GB (15°)')
        ax3.set_xlabel('Misorientation Angle (degrees)')
        ax3.set_ylabel('Count')
        ax3.set_title('Grain Misorientation Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # --- Grain orientation spread ---
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    
    # Plot orientation triads for each grain
    for grain in grains[:20]:  # Limit to 20 for clarity
        seed = grain['seed_position']
        R = np.array(grain['base_rotation'])
        scale = 0.1 * L
        
        for j, color in enumerate(['red', 'green', 'blue']):
            direction = R[:, j] * scale
            ax4.quiver(seed[0], seed[1], seed[2],
                      direction[0], direction[1], direction[2],
                      color=color, alpha=0.6, linewidth=1)
    
    _set_cube_axes(ax4, L)
    ax4.set_title('Grain Orientations (RGB = XYZ)')
    
    # --- 3D Grain Boundaries Only ---
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    
    # Compute grain boundary atoms
    boundary_mask = np.zeros(len(positions), dtype=bool)
    if len(positions) > 0 and len(grains) > 1:
        pos_tree = cKDTree(positions)
        cutoff = 2.0  # Angstrom cutoff for neighbor search
        
        for i in range(len(positions)):
            if grain_ids[i] < 0:
                continue
            neighbors = pos_tree.query_ball_point(positions[i], cutoff)
            neighbor_grains = grain_ids[neighbors]
            # Mark as boundary if neighbors have different grains
            unique_neighbor_grains = set(neighbor_grains[neighbor_grains >= 0])
            if len(unique_neighbor_grains) > 1:
                boundary_mask[i] = True
    
    if np.any(boundary_mask):
        boundary_pos = positions[boundary_mask]
        boundary_gids = grain_ids[boundary_mask]
        
        # Subsample if too many
        sample_size = min(6000, len(boundary_pos))
        if sample_size < len(boundary_pos):
            idx = np.random.choice(len(boundary_pos), sample_size, replace=False)
            boundary_pos = boundary_pos[idx]
            boundary_gids = boundary_gids[idx]
        
        boundary_colors = [grain_color_map.get(int(g), _DEFAULT_GRAY_RGBA) for g in boundary_gids]
        ax5.scatter(boundary_pos[:, 0], boundary_pos[:, 1], boundary_pos[:, 2],
                   c=boundary_colors, s=10, alpha=0.8, edgecolors='black', linewidths=0.15)
    
    _set_cube_axes(ax5, L)
    ax5.set_title(f'Grain Boundary Atoms ({np.sum(boundary_mask):,})')
    
    # --- Phase distribution across grains ---
    ax6 = fig.add_subplot(2, 3, 6)
    grain_phase_counts = {}
    for grain in grains:
        gid = grain['grain_id']
        phase = grain.get('base_phase_id', 'unknown')
        mask = grain_ids == gid
        count = np.sum(mask)
        if phase not in grain_phase_counts:
            grain_phase_counts[phase] = []
        grain_phase_counts[phase].append(count)
    
    if grain_phase_counts:
        labels = list(grain_phase_counts.keys())
        data = [grain_phase_counts[l] for l in labels]
        short_labels = [l.replace('crystal_', '').replace('liquid_', 'liq_')[:12] for l in labels]
        
        bp = ax6.boxplot(data, labels=short_labels, patch_artist=True)
        for patch, label in zip(bp['boxes'], labels):
            patch.set_facecolor(_phase_to_color(label))
            patch.set_alpha(0.7)
        
        ax6.set_ylabel('Atoms per Grain')
        ax6.set_title('Grain Sizes by Phase')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Grain Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# =============================================================================
# INTERFACE ANALYSIS
# =============================================================================

def _render_interface_analysis(
    global_cfg: Any,
    positions: np.ndarray,
    phases: np.ndarray,
    grain_ids: np.ndarray,
    grains: Sequence[Dict[str, Any]],
    color_map: Dict[str, Any],
    output_path: pathlib.Path,
    rng: np.random.Generator,
) -> None:
    """Analyze and visualize interfaces between phases/grains."""
    if len(positions) < 100:
        return
    
    fig = plt.figure(figsize=(16, 10))
    
    L = global_cfg.L
    avg_nn = global_cfg.avg_nn_dist
    
    # Identify interface atoms (near phase boundaries)
    tree = cKDTree(positions)
    cutoff = 1.5 * avg_nn
    
    interface_mask = np.zeros(len(positions), dtype=bool)
    
    for i in range(len(positions)):
        neighbors = tree.query_ball_point(positions[i], cutoff)
        neighbor_phases = phases[neighbors]
        neighbor_grains = grain_ids[neighbors]
        
        # Interface if neighbors have different phases or grains
        if len(set(neighbor_phases)) > 1 or len(set(neighbor_grains[neighbor_grains >= 0])) > 1:
            interface_mask[i] = True
    
    interface_positions = positions[interface_mask]
    interface_phases = phases[interface_mask]
    
    # --- 3D interface visualization ---
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    if len(interface_positions) > 0:
        sample_size = min(5000, len(interface_positions))
        sample_idx = rng.choice(len(interface_positions), sample_size, replace=False)
        sample_pos = interface_positions[sample_idx]
        sample_phases = interface_phases[sample_idx]
        
        colors = [color_map.get(p, _DEFAULT_GRAY_RGBA) for p in sample_phases]
        ax1.scatter(sample_pos[:, 0], sample_pos[:, 1], sample_pos[:, 2],
                   c=colors, s=12, alpha=0.75, edgecolors='black', linewidths=0.2)
    _set_cube_axes(ax1, L)
    ax1.set_title(f'Interface Atoms ({np.sum(interface_mask):,})')
    
    # --- Interface fraction by phase ---
    ax2 = fig.add_subplot(2, 3, 2)
    phase_interface_frac = {}
    for phase in np.unique(phases):
        phase_mask = phases == phase
        phase_interface = phase_mask & interface_mask
        frac = np.sum(phase_interface) / max(1, np.sum(phase_mask))
        phase_interface_frac[phase] = frac
    
    if phase_interface_frac:
        labels = list(phase_interface_frac.keys())
        fracs = list(phase_interface_frac.values())
        short_labels = [l.replace('crystal_', '').replace('liquid_', 'liq_')[:12] for l in labels]
        colors = [_phase_to_color(l) for l in labels]
        
        bars = ax2.bar(short_labels, fracs, color=colors, edgecolor='black')
        ax2.set_ylabel('Interface Fraction')
        ax2.set_title('Interface Atoms per Phase')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')
    
    # --- RDF at interfaces vs bulk ---
    ax3 = fig.add_subplot(2, 3, 3)
    
    bulk_mask = ~interface_mask
    
    if np.sum(bulk_mask) > 100:
        r_bulk, gr_bulk = _compute_rdf(positions[bulk_mask], L, r_max=3*avg_nn, rng=rng)
        ax3.plot(r_bulk / avg_nn, gr_bulk, 'b-', linewidth=2, label='Bulk')
    
    if np.sum(interface_mask) > 100:
        r_int, gr_int = _compute_rdf(interface_positions, L, r_max=3*avg_nn, rng=rng)
        ax3.plot(r_int / avg_nn, gr_int, 'r-', linewidth=2, label='Interface')
    
    ax3.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('r / avg_nn_dist')
    ax3.set_ylabel('g(r)')
    ax3.set_title('RDF: Bulk vs Interface')
    ax3.legend()
    ax3.set_xlim(0, 3)
    ax3.grid(True, alpha=0.3)
    
    # --- Coordination at interfaces ---
    ax4 = fig.add_subplot(2, 3, 4)
    coord_cutoff = 1.4 * avg_nn
    
    bulk_coords = _compute_coordination(positions[bulk_mask], coord_cutoff, 
                                         sample_size=2000, rng=rng) if np.sum(bulk_mask) > 50 else np.array([])
    int_coords = _compute_coordination(interface_positions, coord_cutoff,
                                        sample_size=2000, rng=rng) if len(interface_positions) > 50 else np.array([])
    
    if len(bulk_coords) > 0 and len(int_coords) > 0:
        bins = np.arange(0, 18) - 0.5
        ax4.hist(bulk_coords, bins=bins, alpha=0.6, label=f'Bulk (μ={np.mean(bulk_coords):.1f})',
                color='blue', density=True)
        ax4.hist(int_coords, bins=bins, alpha=0.6, label=f'Interface (μ={np.mean(int_coords):.1f})',
                color='red', density=True)
        ax4.legend()
    ax4.set_xlabel('Coordination Number')
    ax4.set_ylabel('Probability')
    ax4.set_title('Coordination: Bulk vs Interface')
    ax4.grid(True, alpha=0.3)
    
    # --- Interface width distribution ---
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Calculate distance of interface atoms to nearest bulk atom
    if np.sum(bulk_mask) > 100 and np.sum(interface_mask) > 100:
        bulk_tree = cKDTree(positions[bulk_mask])
        distances, _ = bulk_tree.query(interface_positions, k=1)
        
        ax5.hist(distances, bins=50, color='purple', edgecolor='black', alpha=0.7)
        ax5.axvline(np.median(distances), color='red', linestyle='--',
                   label=f'Median: {np.median(distances):.2f}Å')
        ax5.set_xlabel('Distance to Bulk (Å)')
        ax5.set_ylabel('Count')
        ax5.set_title('Interface Atom Distance Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # --- XY slice showing interface ---
    ax6 = fig.add_subplot(2, 3, 6)
    z_center = L / 2
    z_thickness = 3.0
    
    slice_mask = np.abs(positions[:, 2] - z_center) < z_thickness / 2
    slice_pos = positions[slice_mask]
    slice_interface = interface_mask[slice_mask]
    
    if len(slice_pos) > 0:
        # Bulk atoms
        bulk_slice = slice_pos[~slice_interface]
        ax6.scatter(bulk_slice[:, 0], bulk_slice[:, 1], c='lightblue', s=8, alpha=0.6, 
                   edgecolors='steelblue', linewidths=0.2, label='Bulk')
        
        # Interface atoms
        int_slice = slice_pos[slice_interface]
        ax6.scatter(int_slice[:, 0], int_slice[:, 1], c='red', s=12, alpha=0.85,
                   edgecolors='darkred', linewidths=0.3, label='Interface')
    
    ax6.set_xlim(0, L)
    ax6.set_ylim(0, L)
    ax6.set_aspect('equal')
    ax6.set_xlabel('X (Å)')
    ax6.set_ylabel('Y (Å)')
    ax6.set_title('XY Slice: Interface Detection')
    ax6.legend(loc='upper right', fontsize=8)
    
    fig.suptitle('Interface Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# =============================================================================
# ORIGINAL FUNCTIONS (FIXED AND IMPROVED)
# =============================================================================

def _render_global_structure(
    global_cfg: Any,
    grains: Sequence[Dict[str, Any]],
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    phases: Sequence[str],
    grain_ids: np.ndarray,
    metadata: Dict[str, Any],
    grain_color_map: Dict[int, Any],
    color_map: Dict[str, Any],
    output_path: pathlib.Path,
    rng: np.random.Generator,
    view_angles: Optional[Tuple[float, float]] = None,
) -> None:
    """Render global structure overview."""
    _render_global_structure_core(
        global_cfg, grains, atoms, atom_positions, phases, grain_ids,
        metadata, grain_color_map, color_map, output_path, rng,
        original_indices=np.arange(len(atoms), dtype=int),
        view_angles=view_angles,
        is_diagonal_cut=False,
    )


def _render_global_structure_diagonal_cut(
    global_cfg: Any,
    grains: Sequence[Dict[str, Any]],
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    phases: Sequence[str],
    grain_ids: np.ndarray,
    metadata: Dict[str, Any],
    grain_color_map: Dict[int, Any],
    color_map: Dict[str, Any],
    output_path: pathlib.Path,
    rng: np.random.Generator,
    view_angles: Optional[Tuple[float, float]] = None,
) -> None:
    """Render diagonal cut view."""
    if len(atoms) == 0:
        return
    
    mask = _diagonal_cut_mask(atom_positions, global_cfg.L)
    if not np.any(mask):
        return
    
    kept_indices = np.flatnonzero(mask)
    filtered_atoms = [atoms[int(i)] for i in kept_indices]
    filtered_positions = atom_positions[mask]
    filtered_phases = [phases[int(i)] for i in kept_indices]
    filtered_grain_ids = grain_ids[mask]
    
    _render_global_structure_core(
        global_cfg, grains, filtered_atoms, filtered_positions,
        filtered_phases, filtered_grain_ids, metadata, grain_color_map,
        color_map, output_path, rng,
        original_indices=kept_indices,
        view_angles=view_angles,
        is_diagonal_cut=True,
    )


def _render_global_structure_core(
    global_cfg: Any,
    grains: Sequence[Dict[str, Any]],
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    phases: Sequence[str],
    grain_ids: np.ndarray,
    metadata: Dict[str, Any],
    grain_color_map: Dict[int, Any],
    color_map: Dict[str, Any],
    output_path: pathlib.Path,
    rng: np.random.Generator,
    original_indices: np.ndarray,
    view_angles: Optional[Tuple[float, float]],
    is_diagonal_cut: bool = False,
) -> None:
    """Core rendering for global structure."""
    total_atoms = len(atoms)
    if total_atoms == 0:
        return
    
    phase_array = np.asarray(phases, dtype=object)
    grain_ids = np.asarray(grain_ids, dtype=int)
    
    # Sample for display
    phase_sample_limit = 8000
    if total_atoms <= phase_sample_limit:
        phase_indices = np.arange(total_atoms, dtype=int)
    else:
        phase_indices = rng.choice(total_atoms, size=phase_sample_limit, replace=False)
    
    phase_positions = _ensure_point_array(atom_positions[phase_indices])
    phase_ids = phase_array[phase_indices]
    
    fig = plt.figure(figsize=(18, 12))
    
    # --- Panel 1: Phase overview ---
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    for phase in color_map:
        mask = phase_ids == phase
        if not np.any(mask):
            continue
        points = phase_positions[mask]
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                   color=color_map[phase], s=12, depthshade=True,
                   edgecolors='black', linewidths=0.2, alpha=0.85)
    ax1.set_title('Phases Overview')
    _set_cube_axes(ax1, global_cfg.L)
    
    # --- Panel 2: Grain coloring ---
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    sample_grain_ids = grain_ids[phase_indices]
    for gid in np.unique(sample_grain_ids):
        if gid < 0:
            continue
        mask = sample_grain_ids == gid
        points = phase_positions[mask]
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                   color=grain_color_map.get(gid, _DEFAULT_GRAY_RGBA),
                   s=12, depthshade=True, edgecolors='black', linewidths=0.2, alpha=0.85)
    ax2.set_title('Grain Coloring')
    _set_cube_axes(ax2, global_cfg.L)
    
    # --- Panel 3: Intermediate phases ---
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    intermediate_mask = np.array([str(p).startswith('intermediate_') for p in phase_array])
    if np.any(intermediate_mask):
        inter_positions = _ensure_point_array(atom_positions[intermediate_mask])
        sample_size = min(5000, len(inter_positions))
        if sample_size < len(inter_positions):
            idx = rng.choice(len(inter_positions), sample_size, replace=False)
            inter_positions = inter_positions[idx]
        ax3.scatter(inter_positions[:, 0], inter_positions[:, 1], inter_positions[:, 2],
                   color=INTERFACE_COLOR, s=14, depthshade=True, alpha=0.8,
                   edgecolors='black', linewidths=0.2)
    ax3.set_title('Intermediate Phases')
    _set_cube_axes(ax3, global_cfg.L)
    
    # --- Panel 4: Grain boundaries ---
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    boundary_mask, _ = _compute_boundary_indices(atom_positions, grain_ids, grains)
    if np.any(boundary_mask):
        boundary_positions = _ensure_point_array(atom_positions[boundary_mask])
        sample_size = min(5000, len(boundary_positions))
        if sample_size < len(boundary_positions):
            idx = rng.choice(len(boundary_positions), sample_size, replace=False)
            boundary_positions = boundary_positions[idx]
            boundary_grains = grain_ids[boundary_mask][idx]
        else:
            boundary_grains = grain_ids[boundary_mask]
        colors = [grain_color_map.get(int(g), _DEFAULT_GRAY_RGBA) for g in boundary_grains]
        ax4.scatter(boundary_positions[:, 0], boundary_positions[:, 1], boundary_positions[:, 2],
                   c=colors, s=14, depthshade=True, alpha=0.8,
                   edgecolors='black', linewidths=0.2)
    ax4.set_title('Grain Boundaries')
    _set_cube_axes(ax4, global_cfg.L)
    
    # --- Panel 5: Grain orientations ---
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    scale = global_cfg.avg_nn_dist * 2
    for grain in grains:
        seed = grain['seed_position']
        rotation = np.array(grain['base_rotation'])
        for j, color in enumerate(['red', 'green', 'blue']):
            direction = rotation[:, j] * scale
            ax5.quiver(seed[0], seed[1], seed[2],
                      direction[0], direction[1], direction[2],
                      color=color, alpha=0.6, linewidth=1.5)
    ax5.set_title('Grain Orientations')
    _set_cube_axes(ax5, global_cfg.L)
    
    # --- Panel 6: Perturbations ---
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    perturb = metadata.get('perturbations', {})
    
    # Rotation bubbles
    for bubble in perturb.get('rotation_bubbles', []):
        center = np.array(bubble['center'])
        ax6.scatter(*center, color='orange', s=80, marker='^', 
                   edgecolors='black', linewidths=1, label='Rotation')
    
    # Density bubbles
    for bubble in perturb.get('density_bubbles', []):
        center = np.array(bubble['center'])
        color = 'blue' if bubble.get('alpha', 0) > 0 else 'red'
        marker = 'o' if bubble.get('alpha', 0) > 0 else 's'
        ax6.scatter(*center, color=color, s=80, marker=marker,
                   edgecolors='black', linewidths=1)
    
    # Vacancies (sample)
    dropouts = perturb.get('dropouts', {}).get('events', [])
    if dropouts:
        vacancy_positions = [e.get('vacancy_position', [0,0,0]) for e in dropouts[:500]]
        vacancy_positions = np.array(vacancy_positions)
        ax6.scatter(vacancy_positions[:, 0], vacancy_positions[:, 1], vacancy_positions[:, 2],
                   color='black', s=20, marker='x', alpha=0.5)
    
    ax6.set_title('Perturbations')
    _set_cube_axes(ax6, global_cfg.L)
    
    # Apply view angles
    if view_angles is not None:
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.view_init(elev=view_angles[0], azim=view_angles[1])
    
    title = 'Global Structure Overview'
    if is_diagonal_cut:
        title += ' (Diagonal Cut)'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _render_closeup_view(
    global_cfg: Any,
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    phases: Sequence[str],
    color_map: Dict[str, Any],
    output_path: pathlib.Path,
    rng: np.random.Generator,
) -> None:
    """Render close-up view of corner region."""
    if len(atoms) == 0:
        return
    
    L = global_cfg.L
    corner_max = np.array([L/2, L/2, L/4])
    
    mask = np.all((atom_positions >= 0) & (atom_positions <= corner_max), axis=1)
    closeup_indices = np.flatnonzero(mask)
    
    if len(closeup_indices) == 0:
        return
    
    closeup_positions = atom_positions[closeup_indices]
    closeup_phases = np.asarray(phases)[closeup_indices]
    
    fig = plt.figure(figsize=(14, 6))
    
    # Left: 3D view
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    for phase in np.unique(closeup_phases):
        phase_mask = closeup_phases == phase
        points = closeup_positions[phase_mask]
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                   color=color_map.get(phase, _DEFAULT_GRAY_RGBA),
                   s=18, depthshade=True, alpha=0.85, edgecolors='black', linewidths=0.3)
    
    ax1.set_xlim(0, corner_max[0])
    ax1.set_ylim(0, corner_max[1])
    ax1.set_zlim(0, corner_max[2])
    ax1.set_title(f'Close-up (1/16 volume, {len(closeup_indices):,} atoms)')
    
    # Right: XY projection
    ax2 = fig.add_subplot(1, 2, 2)
    for phase in np.unique(closeup_phases):
        phase_mask = closeup_phases == phase
        points = closeup_positions[phase_mask]
        ax2.scatter(points[:, 0], points[:, 1],
                   color=color_map.get(phase, _DEFAULT_GRAY_RGBA),
                   s=12, alpha=0.75, edgecolors='black', linewidths=0.2)
    
    ax2.set_xlim(0, corner_max[0])
    ax2.set_ylim(0, corner_max[1])
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (Å)')
    ax2.set_ylabel('Y (Å)')
    ax2.set_title('XY Projection')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _render_local_galleries(
    global_cfg: Any,
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    phases: Sequence[str],
    metadata: Dict[str, Any],
    output_dir: pathlib.Path,
    rng: np.random.Generator,
) -> None:
    """Render local neighborhood galleries."""
    base_phases = sorted({p for p in phases if not str(p).startswith('intermediate_')})
    intermediate_phases = sorted({p for p in phases if str(p).startswith('intermediate_')})
    
    positions_tree = cKDTree(atom_positions)
    
    # Render with Delaunay triangulation edges
    _render_local_gallery(
        title='Base Phases - Local Environments (Delaunay)',
        filename=output_dir / 'figure_local_base.png',
        selected_phases=base_phases,
        atoms=atoms, atom_positions=atom_positions,
        positions_tree=positions_tree, global_cfg=global_cfg, rng=rng,
        edge_type='delaunay',
    )
    
    # Render with KNN edges (k=4)
    _render_local_gallery(
        title='Base Phases - Local Environments (KNN k=4)',
        filename=output_dir / 'figure_local_base_knn.png',
        selected_phases=base_phases,
        atoms=atoms, atom_positions=atom_positions,
        positions_tree=positions_tree, global_cfg=global_cfg, rng=rng,
        edge_type='knn',
        knn_k=3,
    )
    
    if intermediate_phases:
        _render_local_gallery(
            title='Intermediate Phases - Local Environments (Delaunay)',
            filename=output_dir / 'figure_local_intermediate.png',
            selected_phases=intermediate_phases,
            atoms=atoms, atom_positions=atom_positions,
            positions_tree=positions_tree, global_cfg=global_cfg, rng=rng,
            edge_type='delaunay',
        )



def _plot_object(
    ax: Any,
    atom_positions: np.ndarray,
) -> None:
    """Plot a discrete object centered at origin."""
    if len(atom_positions) == 0:
        return

    center = np.mean(atom_positions, axis=0)
    coords = atom_positions - center
    
    # Simple depth coloring
    distances = np.linalg.norm(coords, axis=1)
    max_dist = np.max(distances) if len(distances) > 0 else 1.0
    
    # Use different cmap for object view
    colors = cm.viridis(distances / (max_dist + 1e-6))
    
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
              c=colors, s=30, alpha=0.9, edgecolors='black', linewidths=0.3)
    
    # Delaunay edges for structure
    if len(coords) >= 4:
        try:
            tri = Delaunay(coords)
            edges = set()
            for simplex in tri.simplices:
                for i in range(4):
                    for j in range(i + 1, 4):
                        edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                        edges.add(edge)
            
            # Limit edges to avoid overcrowding
            edge_list = list(edges)
            if len(edge_list) > 1000:
                # Random sample
                idx = np.random.choice(len(edge_list), 1000, replace=False)
                edge_list = [edge_list[i] for i in idx]
            
            for i, j in edge_list:
                p1, p2 = coords[i], coords[j]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                       color='#555555', linewidth=0.5, alpha=0.3)
        except Exception as exc:
            warnings.warn(
                "Delaunay edge construction failed in _plot_object; "
                f"continuing without object edges. Error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    limit = max_dist * 1.2
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect([1, 1, 1])
    # Hide axes ticks for cleaner object view
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

def _render_local_gallery(
    title: str,
    filename: pathlib.Path,
    selected_phases: Sequence[str],
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    positions_tree: cKDTree,
    global_cfg: Any,
    rng: np.random.Generator,
    edge_type: str = 'delaunay',
    knn_k: int = 3,
) -> None:
    """Render gallery of local neighborhoods or whole objects."""
    if not selected_phases:
        return

    # Check for simple generator with object IDs
    use_objects = hasattr(global_cfg, 'objects_per_grain') and len(atoms) > 0 and 'object_id' in atoms[0]
    
    if use_objects:
        # Group by object_id
        # We need to map phase -> [object_id, ...]
        phase_to_objects = defaultdict(list)
        object_to_indices = defaultdict(list)
        
        # Build index map first
        for i, atom in enumerate(atoms):
             oid = atom.get('object_id', -1)
             if oid >= 0:
                 object_to_indices[oid].append(i)
        
        # Map objects to phases (using first atom of object)
        for oid, indices in object_to_indices.items():
            if indices:
                first_idx = indices[0]
                ph = atoms[first_idx]['phase_id']
                if ph in selected_phases:
                    phase_to_objects[ph].append(oid)
        
        phase_list = [p for p in selected_phases if phase_to_objects[p]]
    else:
        # Original logic: map phase -> atoms
        phase_to_atoms: Dict[str, List[Dict[str, Any]]] = {phase: [] for phase in selected_phases}
        for atom in atoms:
            phase = atom['phase_id']
            if phase in phase_to_atoms:
                phase_to_atoms[phase].append(atom)
        phase_list = [p for p in selected_phases if phase_to_atoms[p]]
    
    if not phase_list:
        return
    
    rows, cols = 4, len(phase_list)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), subplot_kw={'projection': '3d'})
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1:
        axes = axes.T
    
    for col, phase in enumerate(phase_list):
        if use_objects:
            obj_ids = phase_to_objects[phase]
            num_samples = min(rows, len(obj_ids))
            sample_oids = rng.choice(obj_ids, size=num_samples, replace=False)
            
            for row in range(rows):
                ax = axes[row, col] if cols > 1 else axes[row, 0]
                if row >= num_samples:
                    ax.axis('off')
                    continue
                
                oid = sample_oids[row]
                indices = object_to_indices[oid]
                pos = atom_positions[indices]
                
                _plot_object(ax, pos)
                
                short_phase = phase.replace('crystal_', '').replace('liquid_', 'liq_')[:15]
                ax.set_title(f'{short_phase}\nObj #{oid}', fontsize=9)

        else:
            phase_atoms = phase_to_atoms[phase]
            num_samples = min(rows, len(phase_atoms))
            sample_indices = rng.choice(len(phase_atoms), size=num_samples, replace=False)
            
            for row in range(rows):
                ax = axes[row, col] if cols > 1 else axes[row, 0]
                if row >= num_samples:
                    ax.axis('off')
                    continue
                
                center_atom = phase_atoms[int(sample_indices[row])]
                _plot_local_neighborhood(
                    ax=ax, center_atom=center_atom, atoms=atoms,
                    atom_positions=atom_positions, positions_tree=positions_tree,
                    global_cfg=global_cfg, target_count=48,
                    edge_type=edge_type, knn_k=knn_k,
                )
                short_phase = phase.replace('crystal_', '').replace('liquid_', 'liq_')[:15]
                ax.set_title(f'{short_phase}\n#{center_atom.get("final_index", "?")}', fontsize=9)
    
    if use_objects:
        title = "Base Phases - Whole Objects"

    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def _plot_local_neighborhood(
    ax: Any,
    center_atom: Dict[str, Any],
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    positions_tree: cKDTree,
    global_cfg: Any,
    target_count: int,
    edge_type: str = 'delaunay',
    knn_k: int = 3,
) -> None:
    """Plot local atomic neighborhood with bonds.
    
    Args:
        edge_type: 'delaunay' for Delaunay triangulation edges, 'knn' for k-nearest neighbor edges
        knn_k: Number of neighbors for KNN edges (only used if edge_type='knn')
    """
    # Detect if we are using the simple generator (has objects_per_grain)
    is_simple_generator = hasattr(global_cfg, 'objects_per_grain')
    
    if is_simple_generator:
        # Heuristic for simple generator to capture entire object
        # Calculate approximate object size
        total_objects = global_cfg.grain_count * global_cfg.objects_per_grain
        if total_objects > 0:
            vol_per_object = (global_cfg.L**3 * 0.15) / total_objects
            base_radius = (vol_per_object / (4/3 * np.pi))**(1/3)
            # Use a large factor to ensure we capture the whole object relative to the center atom
            # (which might be at the edge of the object)
            radius = base_radius * 5.0
            
            # Increase target count to see the shape
            target_count = min(int(global_cfg.points_per_object), 500)
        else:
             radius = 2.0 * global_cfg.avg_nn_dist
    else:
        radius = 2.0 * global_cfg.avg_nn_dist

    center_pos = np.array(center_atom['position'])
    
    idxs = positions_tree.query_ball_point(center_pos, r=radius)
    if len(idxs) < target_count:
        _, idxs = positions_tree.query(center_pos, k=min(target_count, len(atom_positions)))
        idxs = np.atleast_1d(idxs)
    
    coords = atom_positions[idxs] - center_pos
    
    # Rotate to align with crystal orientation if available
    phase = center_atom.get('phase_id', '')
    if not str(phase).startswith('amorphous') and 'orientation' in center_atom:
        rotation = np.array(center_atom['orientation'])
        if rotation.shape == (3, 3):
            coords = (rotation.T @ coords.T).T
    
    # Sort by distance and take nearest target_count
    dists_sq = np.sum(coords**2, axis=1)
    sort_idx = np.argsort(dists_sq)
    coords = coords[sort_idx[:target_count]]
    
    # Color by distance from center
    distances = np.linalg.norm(coords, axis=1)
    max_dist = np.max(distances) if len(distances) > 0 else radius
    colors = plt.cm.viridis(distances / (max_dist + 1e-6))
    
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
              c=colors, s=50, alpha=0.9, edgecolors='black', linewidths=0.4)
    
    # Draw edges based on edge_type
    if len(coords) >= 4:
        drawn = set()
        
        if edge_type == 'delaunay':
            # Use Delaunay triangulation for edges
            try:
                tri = Delaunay(coords)
                # Extract edges from simplices (tetrahedra in 3D)
                for simplex in tri.simplices:
                    for i in range(4):
                        for j in range(i + 1, 4):
                            edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                            if edge not in drawn:
                                drawn.add(edge)
                                p1, p2 = coords[edge[0]], coords[edge[1]]
                                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                       color='#555555', linewidth=0.6, alpha=0.5)
            except Exception as exc:
                warnings.warn(
                    f"Delaunay triangulation failed; falling back to KNN edges. Error: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                edge_type = 'knn'
        
        if edge_type == 'knn':
            # Use k-nearest neighbors for edges
            neighbor_tree = cKDTree(coords)
            k = min(knn_k + 1, len(coords))  # +1 because query includes self
            
            for i, point in enumerate(coords):
                _, indices = neighbor_tree.query(point, k=k)
                indices = np.atleast_1d(indices)
                for j in indices:
                    if j != i:
                        edge = (min(i, j), max(i, j))
                        if edge not in drawn:
                            drawn.add(edge)
                            p1, p2 = coords[i], coords[j]
                            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                   color='#884422', linewidth=0.7, alpha=0.6)
    
    # Dynamically set limits to fit the data
    if len(coords) > 0:
        limit = max(radius, np.max(np.abs(coords)) * 1.1)
    else:
        limit = radius
        
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect([1, 1, 1])


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _phase_to_color(phase: str) -> str:
    """Map phase name to color."""
    phase_lower = str(phase).lower()
    if 'liquid' in phase_lower or 'amorphous' in phase_lower:
        return LIQUID_COLOR
    elif 'bcc' in phase_lower:
        return '#4ECDC4'
    elif 'fcc' in phase_lower:
        return '#45B7D1'
    elif 'hcp' in phase_lower:
        return '#96CEB4'
    elif 'intermediate' in phase_lower:
        return INTERFACE_COLOR
    else:
        return '#95A5A6'


def _build_phase_color_map(phases: Sequence[str]) -> Dict[str, Any]:
    """Build color map for phases."""
    unique_phases = sorted(set(map(str, phases)))
    if not unique_phases:
        return {}
    
    color_map = {}
    for phase in unique_phases:
        color_map[phase] = mcolors.to_rgba(_phase_to_color(phase))
    
    return color_map


def _build_grain_color_map(grains: Sequence[Dict[str, Any]]) -> Dict[int, Any]:
    """Build color map for grains."""
    grain_ids = sorted({int(g['grain_id']) for g in grains if g.get('grain_id') is not None})
    if not grain_ids:
        return {}
    
    cmap = cm.get_cmap('gist_ncar')
    denom = max(1, len(grain_ids) - 1)
    return {gid: tuple(map(float, cmap(i / denom))) for i, gid in enumerate(grain_ids)}


def _compute_boundary_indices(
    atom_positions: np.ndarray,
    grain_ids: np.ndarray,
    grains: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify atoms at grain boundaries."""
    if atom_positions.size == 0 or grain_ids.size == 0:
        return np.zeros(len(atom_positions), dtype=bool), np.full(len(atom_positions), -1, dtype=int)
    
    grain_positions = np.array([g['seed_position'] for g in grains])
    num_grains = len(grain_positions)
    
    if num_grains < 2:
        return grain_ids < 0, np.full(len(atom_positions), -1, dtype=int)
    
    tree = cKDTree(grain_positions)
    _, neighbor_indices = tree.query(atom_positions, k=2)
    neighbor_indices = np.atleast_2d(neighbor_indices)
    
    second_nearest = neighbor_indices[:, -1]
    mask = (grain_ids < 0) | (grain_ids >= num_grains)
    valid = ~mask
    mask[valid] |= second_nearest[valid] != grain_ids[valid]
    
    return mask, second_nearest


def _view_from_vector(direction: np.ndarray) -> Tuple[float, float]:
    """Convert viewing direction vector to elevation/azimuth angles."""
    vec = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return 30.0, 45.0
    vec = vec / norm
    
    azim = float(np.degrees(np.arctan2(vec[1], vec[0])))
    elev = float(np.degrees(np.arcsin(np.clip(vec[2], -1, 1))))
    
    return elev, azim


def _ensure_point_array(points: Any) -> np.ndarray:
    """Ensure points are a proper Nx3 array."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1:
        if arr.size == 0:
            return arr.reshape(0, 3)
        return arr.reshape(-1, 3)
    return arr


def _set_cube_axes(ax: Any, box_size: float) -> None:
    """Set axis limits for cube visualization."""
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_zlim(0, box_size)
    ax.set_box_aspect([1, 1, 1])


def _diagonal_cut_mask(atom_positions: np.ndarray, box_size: float) -> np.ndarray:
    """Create mask for diagonal cut (keep half of volume)."""
    if atom_positions.size == 0:
        return np.zeros(len(atom_positions), dtype=bool)
    center = np.full(3, box_size / 2.0)
    relative = atom_positions - center
    projection = relative @ _DIAGONAL_DIRECTION
    return projection <= 0.0


__all__ = ['generate_visualizations']
