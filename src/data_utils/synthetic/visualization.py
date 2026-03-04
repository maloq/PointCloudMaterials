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
import textwrap
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
PHASE_CMAP = cm.get_cmap("tab10")
CRYSTAL_COLOR = "#4ECDC4"
LIQUID_COLOR = "#FF6B6B"
INTERFACE_COLOR = "#FFE66D"
_PAPER_FAMILY_DISPLAY_COLORS: Dict[str, str] = {
    "bcc": "#2F6DB3",
    "fcc": "#E3872D",
    "hcp": "#4E9C63",
    "amorphous": "#BE5A5A",
}


def _normalize_visualization_target(viz_target: Optional[str]) -> Optional[str]:
    if viz_target is None:
        return None
    target = str(viz_target).strip().lower()
    if target == "":
        raise ValueError("viz_target must be a non-empty string when provided.")
    supported_targets = {
        "closeup_paper",
        "global_diagonal_cut_paper",
        "local_base",
        "local_base_paper",
    }
    if target not in supported_targets:
        raise ValueError(
            "Unsupported visualization target. "
            f"Expected one of {sorted(supported_targets)}, got {viz_target!r}."
        )
    return target


def generate_visualizations(
    global_cfg: Any,
    grains: Sequence[Dict[str, Any]],
    atoms: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
    rng: np.random.Generator,
    output_dir: pathlib.Path,
    viz_target: Optional[str] = None,
    sampling_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Create comprehensive diagnostic visualizations for the generated dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_target_norm = _normalize_visualization_target(viz_target)
    
    # Extract arrays
    atoms_array = np.asarray([atom["position"] for atom in atoms], dtype=float)
    phases = np.asarray([atom["phase_id"] for atom in atoms], dtype=str)
    grain_ids = np.array(
        [(-1 if atom.get("grain_id") is None else int(atom["grain_id"])) for atom in atoms],
        dtype=int,
    )
    
    # Build color maps
    color_map = _build_phase_color_map(phases)
    if viz_target_norm == "local_base":
        print("  Generating visualizations (target=local_base)...")
        print("    • Local base gallery")
        _render_local_base_gallery(
            global_cfg,
            atoms,
            atoms_array,
            phases,
            output_dir,
            rng,
            color_map,
        )
        print("  Visualizations complete")
        return
    if viz_target_norm == "local_base_paper":
        print("  Generating visualizations (target=local_base_paper)...")
        print("    • Local base paper figure")
        _render_local_base_paper_gallery(
            global_cfg,
            atoms,
            atoms_array,
            phases,
            output_dir,
            rng,
            color_map,
        )
        print("  Visualizations complete")
        return
    if viz_target_norm == "closeup_paper":
        print("  Generating visualizations (target=closeup_paper)...")
        print("    • Paper close-up figures")
        _render_closeup_paper_views(
            global_cfg,
            atoms_array,
            output_dir,
        )
        print("  Visualizations complete")
        return
    if viz_target_norm == "global_diagonal_cut_paper":
        print("  Generating visualizations (target=global_diagonal_cut_paper)...")
        print("    • Paper diagonal-cut overview")
        _render_global_structure_diagonal_cut_paper(
            global_cfg,
            grains,
            atoms,
            atoms_array,
            phases,
            grain_ids,
            output_dir / "figure_global_diagonal_cut_paper.png",
            rng,
            sampling_cfg=sampling_cfg,
            view_angles=_view_from_vector(_DIAGONAL_DIRECTION),
        )
        print("  Visualizations complete")
        return

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
        global_cfg, atoms, atoms_array, phases, metadata, output_dir, rng, color_map
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


def _paper_family_color_for_phase(phase: str) -> str:
    phase_text = str(phase).strip().lower()
    if phase_text == "":
        raise ValueError("phase must be a non-empty string when resolving paper family colors.")
    if "bcc" in phase_text:
        return _PAPER_FAMILY_DISPLAY_COLORS["bcc"]
    if "fcc" in phase_text:
        return _PAPER_FAMILY_DISPLAY_COLORS["fcc"]
    if "hcp" in phase_text:
        return _PAPER_FAMILY_DISPLAY_COLORS["hcp"]
    if "amorphous" in phase_text or "glassy" in phase_text:
        return _PAPER_FAMILY_DISPLAY_COLORS["amorphous"]
    return "#7A7A7A"


def _sample_display_indices(
    n_points: int,
    sample_limit: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_total = int(n_points)
    if n_total < 0:
        raise ValueError(f"n_points must be non-negative, got {n_points}.")
    sample_limit_int = max(0, int(sample_limit))
    if n_total <= sample_limit_int:
        return np.arange(n_total, dtype=int)
    return np.sort(rng.choice(n_total, size=sample_limit_int, replace=False).astype(int, copy=False))


def _compute_group_centroids(
    positions: np.ndarray,
    group_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(positions, dtype=np.float32)
    gids = np.asarray(group_ids, dtype=int)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {pos.shape}.")
    if gids.shape != (pos.shape[0],):
        raise ValueError(
            f"group_ids must have shape ({pos.shape[0]},), got {gids.shape} for positions shape {pos.shape}."
        )
    valid_mask = gids >= 0
    if not np.any(valid_mask):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=int)

    centroids: List[np.ndarray] = []
    centroid_group_ids: List[int] = []
    for gid in np.unique(gids[valid_mask]):
        member_positions = pos[gids == int(gid)]
        if member_positions.shape[0] == 0:
            continue
        centroids.append(np.mean(member_positions, axis=0, dtype=np.float64).astype(np.float32))
        centroid_group_ids.append(int(gid))

    if not centroids:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=int)
    return np.stack(centroids, axis=0).astype(np.float32, copy=False), np.asarray(centroid_group_ids, dtype=int)


def _scatter_paper_global_panel(
    ax: Any,
    positions: np.ndarray,
    colors: Sequence[Any],
    *,
    box_size: float,
    title: str,
    point_size: float,
    linewidth: float,
    alpha: float,
    view_angles: Optional[Tuple[float, float]],
    point_edgecolors: Any = "black",
) -> None:
    pts = _ensure_point_array(positions).astype(np.float32, copy=False)
    color_values = list(colors)
    if pts.shape[0] != len(color_values):
        raise ValueError(
            f"positions length {pts.shape[0]} must match colors length {len(color_values)} for panel {title!r}."
        )

    ax.set_facecolor("white")
    if hasattr(ax, "set_proj_type"):
        ax.set_proj_type("ortho")
    if pts.shape[0] > 0:
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=color_values,
            s=float(point_size),
            depthshade=False,
            edgecolors=point_edgecolors,
            linewidths=float(linewidth),
            alpha=float(alpha),
        )
    _set_cube_axes(ax, box_size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if view_angles is not None:
        ax.view_init(elev=float(view_angles[0]), azim=float(view_angles[1]))


def _compute_regular_sample_center_atoms(
    atom_positions: np.ndarray,
    *,
    radius: float,
    overlap_fraction: float,
    drop_edge_samples: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    pts = _ensure_point_array(atom_positions).astype(np.float32, copy=False)
    if pts.shape[0] == 0:
        raise ValueError("Cannot compute regular sample centers from an empty atom array.")

    radius_use = float(radius)
    if radius_use <= 0.0:
        raise ValueError(f"Regular sample-center radius must be positive, got {radius}.")
    overlap_use = float(overlap_fraction)
    if not (0.0 <= overlap_use < 1.0):
        raise ValueError(
            f"Regular sample-center overlap_fraction must satisfy 0 <= overlap_fraction < 1, got {overlap_fraction}."
        )

    stride = radius_use * (1.0 - overlap_use)
    if stride <= 0.0:
        raise ValueError(
            f"Regular sample-center stride must be positive, got radius={radius_use}, overlap_fraction={overlap_use}."
        )

    min_coords = np.min(pts, axis=0)
    max_coords = np.max(pts, axis=0)
    min_center = min_coords + radius_use
    max_center = max_coords - radius_use
    if np.any(min_center >= max_center):
        raise RuntimeError(
            "Regular sample-center region is empty after edge padding. "
            f"min_center={min_center.tolist()}, max_center={max_center.tolist()}, radius={radius_use}."
        )

    dims = np.ceil((max_center - min_center) / stride).astype(int)
    if np.any(dims <= 0):
        raise RuntimeError(
            "Regular sample-center grid has non-positive dimensions. "
            f"dims={dims.tolist()}, min_center={min_center.tolist()}, max_center={max_center.tolist()}, stride={stride}."
        )

    if bool(drop_edge_samples):
        ranges = [(1, int(dim) - 1) if int(dim) >= 3 else (0, 0) for dim in dims.tolist()]
    else:
        ranges = [(0, int(dim)) for dim in dims.tolist()]
    if any(start >= stop for start, stop in ranges):
        raise RuntimeError(
            "Regular sample-center grid has zero valid centers after applying drop_edge_samples. "
            f"dims={dims.tolist()}, ranges={ranges}, drop_edge_samples={bool(drop_edge_samples)}."
        )

    grid_axes = [np.arange(start, stop, dtype=np.float64) for start, stop in ranges]
    mesh = np.meshgrid(*grid_axes, indexing="ij")
    grid_ijk = np.column_stack([axis.ravel() for axis in mesh]).astype(np.float64, copy=False)
    computed_centers = min_center[None, :] + grid_ijk * stride + radius_use

    tree = cKDTree(pts)
    _, nearest_indices = tree.query(computed_centers, k=1)
    nearest_indices = np.asarray(nearest_indices, dtype=int).reshape(-1)
    if nearest_indices.size == 0:
        raise RuntimeError(
            "Regular sample-center reconstruction produced zero snapped centers after KD-tree snapping."
        )
    snapped_centers = pts[nearest_indices]
    return snapped_centers.astype(np.float32, copy=False), nearest_indices


def _compute_local_structure_sample_centers(
    atom_positions: np.ndarray,
    sampling_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(sampling_cfg, dict):
        raise TypeError(
            f"sampling_cfg must be a dict when reconstructing local-structure centers, got {type(sampling_cfg)!r}."
        )
    sample_type = str(sampling_cfg.get("sample_type", "")).strip().lower()
    if sample_type == "":
        raise ValueError("sampling_cfg is missing required key 'sample_type'.")
    if sample_type != "regular":
        raise ValueError(
            "global_diagonal_cut_paper currently supports only regular synthetic local-structure sampling "
            "because random sample centers are not persisted in the generated dataset. "
            f"Got sample_type={sample_type!r}."
        )
    if "radius" not in sampling_cfg:
        raise ValueError("sampling_cfg is missing required key 'radius' for regular sample-center reconstruction.")
    if "overlap_fraction" not in sampling_cfg:
        raise ValueError(
            "sampling_cfg is missing required key 'overlap_fraction' for regular sample-center reconstruction."
        )
    return _compute_regular_sample_center_atoms(
        atom_positions,
        radius=float(sampling_cfg["radius"]),
        overlap_fraction=float(sampling_cfg["overlap_fraction"]),
        drop_edge_samples=bool(sampling_cfg.get("drop_edge_samples", True)),
    )


def _compute_neighbor_grain_boundary_mask(
    positions: np.ndarray,
    grain_ids: np.ndarray,
    *,
    cutoff: float,
) -> np.ndarray:
    pts = _ensure_point_array(positions).astype(np.float32, copy=False)
    gids = np.asarray(grain_ids, dtype=int)
    if gids.shape != (pts.shape[0],):
        raise ValueError(
            f"grain_ids must have shape ({pts.shape[0]},), got {gids.shape} for positions shape {pts.shape}."
        )
    cutoff_use = float(cutoff)
    if cutoff_use <= 0.0:
        raise ValueError(f"Boundary-neighbor cutoff must be positive, got {cutoff}.")

    boundary_mask = np.zeros((pts.shape[0],), dtype=bool)
    if pts.shape[0] < 2:
        return boundary_mask

    tree = cKDTree(pts)
    pair_indices = np.asarray(tree.query_pairs(cutoff_use, output_type="ndarray"), dtype=int)
    if pair_indices.size == 0:
        return boundary_mask
    pair_indices = pair_indices.reshape(-1, 2)

    g0 = gids[pair_indices[:, 0]]
    g1 = gids[pair_indices[:, 1]]
    diff_mask = (g0 >= 0) & (g1 >= 0) & (g0 != g1)
    if not np.any(diff_mask):
        return boundary_mask

    diff_pairs = pair_indices[diff_mask]
    boundary_mask[diff_pairs[:, 0]] = True
    boundary_mask[diff_pairs[:, 1]] = True
    return boundary_mask


def _render_global_structure_diagonal_cut_paper(
    global_cfg: Any,
    grains: Sequence[Dict[str, Any]],
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    phases: Sequence[str],
    grain_ids: np.ndarray,
    output_path: pathlib.Path,
    rng: np.random.Generator,
    sampling_cfg: Optional[Dict[str, Any]] = None,
    view_angles: Optional[Tuple[float, float]] = None,
) -> None:
    if len(atoms) == 0:
        raise ValueError("Cannot render paper diagonal-cut overview because atoms is empty.")
    if sampling_cfg is None:
        raise ValueError(
            "Cannot render paper diagonal-cut overview without sampling_cfg because the center panels "
            "must reproduce synthetic local-structure sample centers."
        )

    mask = _diagonal_cut_mask(atom_positions, global_cfg.L)
    if not np.any(mask):
        raise RuntimeError("Diagonal cut mask kept zero atoms; cannot render paper diagonal-cut overview.")

    kept_indices = np.flatnonzero(mask)
    filtered_positions = np.asarray(atom_positions[mask], dtype=np.float32)
    filtered_phases = np.asarray(phases, dtype=object)[mask]
    filtered_grain_ids = np.asarray(grain_ids, dtype=int)[mask]

    phase_raw_positions = filtered_positions
    phase_raw_colors = [_paper_family_color_for_phase(str(phase)) for phase in filtered_phases.tolist()]

    sampled_center_positions_all, sampled_center_atom_indices = _compute_local_structure_sample_centers(
        atom_positions,
        sampling_cfg,
    )
    center_mask = _diagonal_cut_mask(sampled_center_positions_all, global_cfg.L)
    if not np.any(center_mask):
        raise RuntimeError(
            "The reconstructed local-structure sample-center set has zero points inside the diagonal cut."
        )
    phase_center_positions = np.asarray(sampled_center_positions_all[center_mask], dtype=np.float32)
    phase_center_atom_indices = np.asarray(sampled_center_atom_indices[center_mask], dtype=int)
    full_to_filtered = np.full((atom_positions.shape[0],), -1, dtype=int)
    full_to_filtered[kept_indices] = np.arange(kept_indices.shape[0], dtype=int)
    phase_center_filtered_indices = np.asarray(full_to_filtered[phase_center_atom_indices], dtype=int)
    if np.any(phase_center_filtered_indices < 0):
        raise RuntimeError(
            "Some reconstructed local-structure centers fell inside the diagonal cut, but their snapped atom indices "
            "were not found in the diagonal-cut atom set. This indicates an internal indexing inconsistency."
        )
    phase_center_phases = np.asarray(phases, dtype=object)[phase_center_atom_indices]
    phase_center_grain_ids = np.asarray(grain_ids, dtype=int)[phase_center_atom_indices]
    phase_center_colors = [
        _paper_family_color_for_phase(str(phase))
        for phase in phase_center_phases.tolist()
    ]

    boundary_mask = _compute_neighbor_grain_boundary_mask(
        filtered_positions,
        filtered_grain_ids,
        cutoff=1.5 * float(global_cfg.avg_nn_dist),
    )
    boundary_positions = filtered_positions[boundary_mask]
    boundary_phases = filtered_phases[boundary_mask]

    boundary_raw_positions = boundary_positions
    boundary_raw_colors = [
        _paper_family_color_for_phase(str(phase))
        for phase in boundary_phases.tolist()
    ]

    sampled_center_boundary_mask = np.asarray(boundary_mask[phase_center_filtered_indices], dtype=bool)
    boundary_center_positions = np.asarray(phase_center_positions[sampled_center_boundary_mask], dtype=np.float32)
    boundary_center_phases = np.asarray(phase_center_phases, dtype=object)[sampled_center_boundary_mask]
    boundary_center_colors = [
        _paper_family_color_for_phase(str(phase))
        for phase in boundary_center_phases.tolist()
    ]

    fig = plt.figure(figsize=(15.0, 13.2), dpi=360, facecolor="white")
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")

    _scatter_paper_global_panel(
        ax1,
        phase_center_positions,
        phase_center_colors,
        box_size=float(global_cfg.L),
        title="Phase Overview · Sample Centers",
        point_size=5.8,
        linewidth=0.12,
        alpha=0.92,
        view_angles=view_angles,
    )
    _scatter_paper_global_panel(
        ax2,
        phase_raw_positions,
        phase_raw_colors,
        box_size=float(global_cfg.L),
        title="Phase Overview · Raw Points",
        point_size=1.2,
        linewidth=0.0,
        alpha=0.78,
        view_angles=view_angles,
        point_edgecolors="none",
    )
    _scatter_paper_global_panel(
        ax3,
        boundary_center_positions,
        boundary_center_colors,
        box_size=float(global_cfg.L),
        title="Grain Boundaries · Sample Centers",
        point_size=5.8,
        linewidth=0.12,
        alpha=0.92,
        view_angles=view_angles,
    )
    _scatter_paper_global_panel(
        ax4,
        boundary_raw_positions,
        boundary_raw_colors,
        box_size=float(global_cfg.L),
        title="Grain Boundaries · Raw Points",
        point_size=1.4,
        linewidth=0.0,
        alpha=0.80,
        view_angles=view_angles,
        point_edgecolors="none",
    )

    fig.suptitle("Global Structure Overview (Diagonal Cut)", fontsize=17, fontweight="bold", y=0.975)
    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.03, top=0.93, wspace=0.03, hspace=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


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
    
    corner_max = _compute_closeup_corner_max(global_cfg.L, volume_divisor=16)
    
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


def _compute_closeup_corner_max(
    box_size: float,
    *,
    volume_divisor: float,
    region_shape: Sequence[float] = (2.0, 2.0, 1.0),
) -> np.ndarray:
    if not np.isfinite(box_size) or float(box_size) <= 0.0:
        raise ValueError(f"box_size must be positive and finite, got {box_size}.")
    if not np.isfinite(volume_divisor) or float(volume_divisor) <= 0.0:
        raise ValueError(f"volume_divisor must be positive and finite, got {volume_divisor}.")

    shape = np.asarray(region_shape, dtype=np.float64)
    if shape.shape != (3,):
        raise ValueError(f"region_shape must have shape (3,), got {shape.shape}.")
    if np.any(~np.isfinite(shape)) or np.any(shape <= 0.0):
        raise ValueError(
            f"region_shape must contain positive finite values, got {shape.tolist()}."
        )

    scale = ((1.0 / float(volume_divisor)) / float(np.prod(shape))) ** (1.0 / 3.0)
    corner_max = float(box_size) * shape * scale
    if np.any(corner_max <= 0.0):
        raise RuntimeError(
            "Computed close-up extents must be positive, "
            f"got {corner_max.tolist()} for volume_divisor={volume_divisor}."
        )
    if np.any(corner_max > float(box_size) + 1e-8):
        raise RuntimeError(
            "Computed close-up extents exceed the simulation box, "
            f"got {corner_max.tolist()} for box_size={box_size}."
        )
    return corner_max.astype(np.float32, copy=False)


def _style_closeup_paper_3d_axes(ax3d: Any) -> None:
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])
    ax3d.set_xlabel("")
    ax3d.set_ylabel("")
    ax3d.set_zlabel("")
    ax3d.grid(False)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        if hasattr(axis, "pane"):
            axis.pane.fill = False
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 1.0))
        if hasattr(axis, "line"):
            axis.line.set_color((0.0, 0.0, 0.0, 1.0))


def _style_closeup_paper_2d_axes(ax2d: Any) -> None:
    ax2d.set_xticks([])
    ax2d.set_yticks([])
    ax2d.set_xlabel("")
    ax2d.set_ylabel("")
    ax2d.grid(False)
    for spine in ax2d.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


def _style_closeup_paper_axes(ax3d: Any, ax2d: Any) -> None:
    _style_closeup_paper_3d_axes(ax3d)
    _style_closeup_paper_2d_axes(ax2d)


def _save_closeup_paper_3d_svg(
    closeup_positions: np.ndarray,
    corner_max: np.ndarray,
    output_path: pathlib.Path,
    *,
    point_color: Any,
    point_edge_color: str,
    point_size_3d: float,
) -> None:
    svg_fig = plt.figure(figsize=(5.4, 5.4), dpi=260, facecolor="white")
    svg_ax = svg_fig.add_subplot(1, 1, 1, projection="3d")
    svg_ax.scatter(
        closeup_positions[:, 0],
        closeup_positions[:, 1],
        closeup_positions[:, 2],
        color=point_color,
        s=float(point_size_3d),
        depthshade=True,
        alpha=0.94,
        edgecolors=point_edge_color,
        linewidths=0.32,
    )
    svg_ax.set_xlim(0.0, float(corner_max[0]))
    svg_ax.set_ylim(0.0, float(corner_max[1]))
    svg_ax.set_zlim(0.0, float(corner_max[2]))
    if hasattr(svg_ax, "set_box_aspect"):
        svg_ax.set_box_aspect(
            (float(corner_max[0]), float(corner_max[1]), float(corner_max[2]))
        )
    if hasattr(svg_ax, "set_proj_type"):
        svg_ax.set_proj_type("ortho")
    svg_ax.view_init(elev=21.0, azim=36.0)
    _style_closeup_paper_3d_axes(svg_ax)
    svg_fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    svg_fig.savefig(output_path, bbox_inches="tight")
    plt.close(svg_fig)


def _render_closeup_paper_view(
    global_cfg: Any,
    atom_positions: np.ndarray,
    output_path: pathlib.Path,
    *,
    volume_divisor: int,
    point_color: Any = "#DADADA",
    point_edge_color: str = "#000000",
    point_size_3d: float = 34.0,
    point_size_xy: float = 26.0,
) -> None:
    if atom_positions.ndim != 2 or atom_positions.shape[1] != 3:
        raise ValueError(
            f"atom_positions must have shape (N, 3), got {atom_positions.shape}."
        )
    if atom_positions.shape[0] == 0:
        raise ValueError("Cannot render a paper close-up figure because atom_positions is empty.")

    corner_max = _compute_closeup_corner_max(
        global_cfg.L,
        volume_divisor=volume_divisor,
        region_shape=(1.0, 1.0, 1.0),
    )
    mask = np.all((atom_positions >= 0.0) & (atom_positions <= corner_max[None, :]), axis=1)
    closeup_indices = np.flatnonzero(mask)
    if closeup_indices.size == 0:
        raise RuntimeError(
            "Paper close-up selection produced no atoms for "
            f"volume_divisor=1/{volume_divisor} and extents {corner_max.tolist()}."
        )

    closeup_positions = atom_positions[closeup_indices]
    fig = plt.figure(figsize=(11.0, 5.2), dpi=260, facecolor="white")

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(
        closeup_positions[:, 0],
        closeup_positions[:, 1],
        closeup_positions[:, 2],
        color=point_color,
        s=float(point_size_3d),
        depthshade=True,
        alpha=0.94,
        edgecolors=point_edge_color,
        linewidths=0.32,
    )
    ax1.set_xlim(0.0, float(corner_max[0]))
    ax1.set_ylim(0.0, float(corner_max[1]))
    ax1.set_zlim(0.0, float(corner_max[2]))
    if hasattr(ax1, "set_box_aspect"):
        ax1.set_box_aspect(
            (float(corner_max[0]), float(corner_max[1]), float(corner_max[2]))
        )
    if hasattr(ax1, "set_proj_type"):
        ax1.set_proj_type("ortho")
    ax1.view_init(elev=21.0, azim=36.0)
    ax1.set_title(f"Corner close-up (1/{volume_divisor} volume, {closeup_indices.size:,} atoms)")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(
        closeup_positions[:, 0],
        closeup_positions[:, 1],
        color=point_color,
        s=float(point_size_xy),
        alpha=0.86,
        edgecolors=point_edge_color,
        linewidths=0.24,
    )
    ax2.set_xlim(0.0, float(corner_max[0]))
    ax2.set_ylim(0.0, float(corner_max[1]))
    ax2.set_aspect("equal")
    ax2.set_title("XY projection")

    _style_closeup_paper_axes(ax1, ax2)

    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.03, top=0.91, wspace=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    _save_closeup_paper_3d_svg(
        closeup_positions,
        corner_max,
        output_path.with_suffix(".svg"),
        point_color=point_color,
        point_edge_color=point_edge_color,
        point_size_3d=point_size_3d,
    )
    plt.close(fig)


def _render_closeup_paper_views(
    global_cfg: Any,
    atom_positions: np.ndarray,
    output_dir: pathlib.Path,
) -> None:
    _render_closeup_paper_view(
        global_cfg,
        atom_positions,
        output_dir / "figure_closeup_paper.png",
        volume_divisor=16,
    )
    _render_closeup_paper_view(
        global_cfg,
        atom_positions,
        output_dir / "figure_closeup_paper_1_32.png",
        volume_divisor=32,
    )


def _render_local_base_gallery(
    global_cfg: Any,
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    phases: Sequence[str],
    output_dir: pathlib.Path,
    rng: np.random.Generator,
    color_map: Dict[str, Any],
    *,
    positions_tree: Optional[cKDTree] = None,
) -> None:
    if len(atom_positions) == 0:
        return

    base_phases = sorted({p for p in phases if not str(p).startswith('intermediate_')})
    if not base_phases:
        return

    positions_tree_use = positions_tree if positions_tree is not None else cKDTree(atom_positions)
    use_objects = bool(
        len(atoms) > 0 and hasattr(global_cfg, 'objects_per_grain') and 'object_id' in atoms[0]
    )
    if use_objects:
        _render_local_gallery(
            title='Base Phases - Whole Objects',
            filename=output_dir / 'figure_local_base.png',
            selected_phases=base_phases,
            atoms=atoms,
            atom_positions=atom_positions,
            positions_tree=positions_tree_use,
            global_cfg=global_cfg,
            rng=rng,
            edge_type='delaunay',
        )
        return

    _render_local_representative_gallery(
        title='Base Phase Representatives (PCA Reciprocal)',
        filename=output_dir / 'figure_local_base.png',
        selected_phases=base_phases,
        atoms=atoms,
        atom_positions=atom_positions,
        positions_tree=positions_tree_use,
        global_cfg=global_cfg,
        rng=rng,
        color_map=color_map,
    )


def _load_reference_point_clouds(output_dir: pathlib.Path) -> Dict[str, np.ndarray]:
    reference_path = output_dir / "reference_point_clouds.npy"
    if not reference_path.exists():
        raise FileNotFoundError(
            "Paper local-base figure requires saved reference point clouds, "
            f"but the file is missing: {reference_path}."
        )
    loaded = np.load(reference_path, allow_pickle=True)
    if loaded.shape != ():
        raise ValueError(
            "reference_point_clouds.npy must store a dict-like object array, "
            f"got array shape {loaded.shape} from {reference_path}."
        )
    reference_point_clouds = loaded.item()
    if not isinstance(reference_point_clouds, dict):
        raise ValueError(
            "reference_point_clouds.npy must decode to a dict, "
            f"got {type(reference_point_clouds)!r} from {reference_path}."
        )
    return {
        str(key): np.asarray(value, dtype=np.float32)
        for key, value in reference_point_clouds.items()
    }


def _paper_family_specs(
    available_phases: Sequence[str],
    reference_point_clouds: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    available_set = {str(phase) for phase in available_phases}
    specs = [
        {
            "family": "BCC",
            "pure_phase": "bcc_iron",
            "pure_source": "reference",
            "perturbed_phase": "bcc_iron_perturbed",
            "display_color": "#2F6DB3",
        },
        {
            "family": "FCC",
            "pure_phase": "fcc_iron",
            "pure_source": "reference",
            "perturbed_phase": "fcc_iron_perturbed",
            "display_color": "#E3872D",
        },
        {
            "family": "HCP",
            "pure_phase": "hcp_iron",
            "pure_source": "reference",
            "perturbed_phase": "hcp_iron_perturbed",
            "display_color": "#4E9C63",
        },
        {
            "family": "Amorphous",
            "pure_phase": "amorphous_pure",
            "pure_source": "dataset",
            "perturbed_phase": None,
            "display_color": "#BE5A5A",
        },
    ]

    missing_items: List[str] = []
    resolved_specs: List[Dict[str, Any]] = []
    for spec in specs:
        pure_phase = spec["pure_phase"]
        perturbed_phase = spec["perturbed_phase"]
        pure_source = str(spec["pure_source"])
        if pure_source == "reference":
            if pure_phase not in reference_point_clouds:
                missing_items.append(f"reference:{pure_phase}")
                continue
        elif pure_source == "dataset":
            if pure_phase not in available_set:
                missing_items.append(f"dataset:{pure_phase}")
                continue
        else:
            raise ValueError(
                f"Unsupported pure_source {pure_source!r} for paper family spec {spec!r}."
            )
        if perturbed_phase is not None and perturbed_phase not in available_set:
            missing_items.append(f"dataset:{perturbed_phase}")
            continue
        resolved_specs.append(spec)

    if missing_items:
        raise ValueError(
            "Paper local-base figure requires all crystal family pairs to be available. "
            f"Missing entries: {missing_items}."
        )
    return resolved_specs


def _prepare_reference_local_points(
    reference_points: np.ndarray,
    *,
    target_count: int,
) -> np.ndarray:
    pts = np.asarray(reference_points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"reference_points must have shape (N, 3), got {pts.shape}.")
    if pts.shape[0] == 0:
        raise ValueError("reference_points must be non-empty.")

    center_idx = int(np.argmin(np.linalg.norm(pts, axis=1)))
    centered = pts - pts[center_idx]
    dists = np.linalg.norm(centered, axis=1)
    keep = np.argsort(dists)[: min(int(target_count), pts.shape[0])]
    local = centered[keep]
    if local.shape[0] == 0:
        raise RuntimeError("No local points remain after selecting a centered reference neighborhood.")
    return local.astype(np.float32, copy=False)


def _match_local_structure_scale(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    radius_percentile: float = 90.0,
) -> np.ndarray:
    source = np.asarray(source_points, dtype=np.float32)
    target = np.asarray(target_points, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"source_points must have shape (N, 3), got {source.shape}.")
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(f"target_points must have shape (N, 3), got {target.shape}.")
    if source.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError(
            "Cannot match local structure scale with empty point clouds: "
            f"source shape {source.shape}, target shape {target.shape}."
        )

    source_scale = float(np.percentile(np.linalg.norm(source, axis=1), float(radius_percentile)))
    target_scale = float(np.percentile(np.linalg.norm(target, axis=1), float(radius_percentile)))
    if not np.isfinite(source_scale) or source_scale <= 1e-8:
        raise ValueError(
            "Computed invalid source scale while matching local structure scale: "
            f"source_scale={source_scale}, radius_percentile={radius_percentile}."
        )
    if not np.isfinite(target_scale) or target_scale <= 1e-8:
        raise ValueError(
            "Computed invalid target scale while matching local structure scale: "
            f"target_scale={target_scale}, radius_percentile={radius_percentile}."
        )
    return (source * (target_scale / source_scale)).astype(np.float32, copy=False)


def _ensure_connected_edges(
    points: np.ndarray,
    edges: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}.")
    n_points = int(pts.shape[0])
    if n_points <= 1:
        return []

    parent = list(range(n_points))
    rank = [0] * n_points

    def _find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def _union(a: int, b: int) -> bool:
        root_a = _find(a)
        root_b = _find(b)
        if root_a == root_b:
            return False
        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1
        return True

    edge_set: set[Tuple[int, int]] = set()
    for raw_edge in edges:
        i = int(raw_edge[0])
        j = int(raw_edge[1])
        if i == j:
            continue
        edge = (min(i, j), max(i, j))
        edge_set.add(edge)
        _union(edge[0], edge[1])

    if len({_find(i) for i in range(n_points)}) == 1:
        return sorted(edge_set)

    dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    candidate_edges: List[Tuple[float, Tuple[int, int]]] = []
    for i in range(n_points):
        for j in range(i + 1, n_points):
            candidate_edges.append((float(dmat[i, j]), (i, j)))
    candidate_edges.sort(key=lambda item: item[0])

    for _, edge in candidate_edges:
        if _union(edge[0], edge[1]):
            edge_set.add(edge)
        if len({_find(i) for i in range(n_points)}) == 1:
            break

    if len({_find(i) for i in range(n_points)}) != 1:
        raise RuntimeError(
            "Failed to construct a connected edge graph for the paper local-base structure."
        )
    return sorted(edge_set)


def _paper_xyz_filename(family: str, column_kind: str) -> str:
    slug = str(family).strip().lower().replace(" ", "_")
    kind = str(column_kind).strip().lower().replace(" ", "_")
    return f"{kind}_{slug}.xyz"


def _save_structure_xyz(
    output_dir: pathlib.Path,
    file_name: str,
    points: np.ndarray,
    *,
    comment: str,
    element: str = "Fe",
) -> None:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}.")
    if pts.shape[0] == 0:
        raise ValueError(f"Cannot save empty XYZ structure to {output_dir / file_name}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / file_name
    with out_path.open("w") as f:
        f.write(f"{pts.shape[0]}\n")
        f.write(f"{comment}\n")
        for point in pts:
            f.write(
                f"{element} {float(point[0]):.8f} {float(point[1]):.8f} {float(point[2]):.8f}\n"
            )


def _prepare_local_structure_geometry(points: np.ndarray) -> Dict[str, Any]:
    local_oriented, _ = _orient_points_for_representative_view(points)
    edges, _ = _build_local_coordination_edges(
        local_oriented,
        min_shell_neighbors=2,
        max_shell_neighbors=5,
        shell_gap_ratio=1.22,
        edge_mode="coordination_shell_mutual",
    )
    return {
        "points": local_oriented,
        "edges": _ensure_connected_edges(local_oriented, edges),
    }


def _compute_local_structure_half_span(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}.")
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    span = float(np.max(maxs - mins))
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError(f"Computed invalid structure span {span} for points shape {pts.shape}.")
    return 0.5 * span


def _compute_radial_colormap_colors(
    points: np.ndarray,
    *,
    cmap_name: str = "viridis",
    radius_percentile: float = 88.0,
    gamma: float = 0.86,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)[:, :3]
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}.")
    if pts.shape[0] == 0:
        raise ValueError("Cannot compute colormap colors for an empty point cloud.")

    centroid = np.mean(pts, axis=0, dtype=np.float64)
    dists = np.linalg.norm(pts - centroid[None, :], axis=1)
    radius_scale = float(np.percentile(dists, float(radius_percentile)))
    if radius_scale <= 1e-12:
        t = np.zeros_like(dists, dtype=np.float32)
    else:
        t = np.clip(dists / radius_scale, 0.0, 1.0).astype(np.float32, copy=False)
    t = np.power(t, float(gamma)).astype(np.float32, copy=False)

    cmap = cm.get_cmap(str(cmap_name))
    colors = np.asarray(cmap(0.10 + 0.82 * (1.0 - t)), dtype=np.float32)
    if colors.shape[1] < 3:
        raise RuntimeError(
            f"Colormap {cmap_name!r} returned an invalid color array with shape {colors.shape}."
        )
    return np.clip(colors[:, :3], 0.0, 1.0).astype(np.float32, copy=False)


def _resolve_local_structure_colors(
    points: np.ndarray,
    *,
    color_mode: str,
    base_color: Any,
    cmap_name: str = "viridis",
) -> np.ndarray:
    mode = str(color_mode).strip().lower()
    if mode == "family":
        return _compute_center_to_edge_colors(points, base_color)
    if mode == "colormap":
        return _compute_radial_colormap_colors(points, cmap_name=cmap_name)
    raise ValueError(
        "Unsupported paper local-base color_mode. "
        f"Expected one of ['family', 'colormap'], got {color_mode!r}."
    )


def _draw_local_structure_panel(
    ax: Any,
    points: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    *,
    point_colors: np.ndarray,
    view_elev: float,
    view_azim: float,
    point_size: float,
    point_linewidth: float,
    edge_alpha: float,
    edge_linewidth: float,
    display_half_span: Optional[float] = None,
) -> None:
    local_oriented = np.asarray(points, dtype=np.float32)
    local_edges = [(int(edge[0]), int(edge[1])) for edge in edges]
    local_colors = np.asarray(point_colors, dtype=np.float32)
    if local_oriented.ndim != 2 or local_oriented.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {local_oriented.shape}.")
    if local_colors.shape != (local_oriented.shape[0], 3):
        raise ValueError(
            "point_colors must have shape (N, 3) matching points. "
            f"Got points shape {local_oriented.shape} and point_colors shape {local_colors.shape}."
        )

    ax.set_facecolor("white")
    if hasattr(ax, "set_proj_type"):
        ax.set_proj_type("ortho")
    ax.view_init(elev=float(view_elev), azim=float(view_azim))
    _draw_edges(
        ax,
        local_oriented,
        local_edges,
        point_colors=local_colors,
        edge_alpha=float(edge_alpha),
        edge_linewidth=float(edge_linewidth),
    )
    ax.scatter(
        local_oriented[:, 0],
        local_oriented[:, 1],
        local_oriented[:, 2],
        c=local_colors,
        s=float(point_size),
        alpha=0.97,
        edgecolors="#222222",
        linewidths=float(point_linewidth),
        depthshade=False,
    )
    _set_equal_axes_3d(ax, local_oriented, half_span=display_half_span)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if hasattr(axis, "pane"):
            axis.pane.fill = False
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 1.0))
        if hasattr(axis, "line"):
            axis.line.set_color((1.0, 1.0, 1.0, 1.0))


def _compute_paper_display_half_span(records: Sequence[Dict[str, Any]]) -> float:
    half_spans: List[float] = []
    for record in records:
        pure_geometry = record.get("pure_geometry")
        if pure_geometry is None:
            raise ValueError(f"Paper record is missing pure_geometry: keys={sorted(record.keys())}.")
        half_spans.append(_compute_local_structure_half_span(pure_geometry["points"]))
        perturbed_geometry = record.get("perturbed_geometry")
        if perturbed_geometry is not None:
            half_spans.append(_compute_local_structure_half_span(perturbed_geometry["points"]))
    if not half_spans:
        raise ValueError("Cannot compute paper display half-span because no panel geometries were prepared.")
    return float(max(half_spans) * 0.94)


def _layout_local_base_paper_axes(fig: Any, axes_arr: np.ndarray) -> Dict[str, float]:
    n_rows, n_cols = axes_arr.shape
    if n_cols != 2:
        raise ValueError(f"Expected exactly 2 columns for paper local-base layout, got shape {axes_arr.shape}.")

    left_x = 0.160
    axis_width = 0.385
    column_gap = -0.004
    right_x = left_x + axis_width + column_gap
    top = 0.922
    bottom = 0.040
    row_gap = -0.014
    axis_height = (top - bottom - row_gap * (n_rows - 1)) / n_rows
    if axis_height <= 0.0:
        raise RuntimeError(
            "Invalid paper local-base layout: computed non-positive axis_height "
            f"{axis_height} for n_rows={n_rows}."
        )

    for row_idx in range(n_rows):
        y0 = top - (row_idx + 1) * axis_height - row_gap * row_idx
        axes_arr[row_idx, 0].set_position([left_x, y0, axis_width, axis_height])
        axes_arr[row_idx, 1].set_position([right_x, y0, axis_width, axis_height])

    return {
        "left_header_x": left_x + 0.5 * axis_width,
        "right_header_x": right_x + 0.5 * axis_width,
        "row_label_x": left_x - 0.028,
        "header_y": 0.972,
    }


def _render_local_base_paper_variant(
    records: Sequence[Dict[str, Any]],
    output_path: pathlib.Path,
    *,
    display_half_span: float,
    color_mode: str,
    label_mode: str,
    cmap_name: str = "viridis",
    view_elev: float,
    view_azim: float,
) -> None:
    n_rows = len(records)
    n_cols = 2
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.35, 7.85),
        dpi=300,
        facecolor="white",
        subplot_kw={"projection": "3d"},
    )
    axes_arr = np.asarray(axes, dtype=object)
    if axes_arr.ndim == 1:
        axes_arr = axes_arr.reshape(n_rows, n_cols)

    layout = _layout_local_base_paper_axes(fig, axes_arr)

    for row_idx, record in enumerate(records):
        pure_geometry = record["pure_geometry"]
        pure_colors = _resolve_local_structure_colors(
            pure_geometry["points"],
            color_mode=color_mode,
            base_color=record["display_color"],
            cmap_name=cmap_name,
        )
        _draw_local_structure_panel(
            axes_arr[row_idx, 0],
            pure_geometry["points"],
            pure_geometry["edges"],
            point_colors=pure_colors,
            view_elev=view_elev,
            view_azim=view_azim,
            point_size=48.0,
            point_linewidth=0.26,
            edge_alpha=0.60,
            edge_linewidth=0.94,
            display_half_span=display_half_span,
        )

        perturbed_geometry = record.get("perturbed_geometry")
        if perturbed_geometry is None:
            axes_arr[row_idx, 1].set_axis_off()
        else:
            perturbed_colors = _resolve_local_structure_colors(
                perturbed_geometry["points"],
                color_mode=color_mode,
                base_color=record["display_color"],
                cmap_name=cmap_name,
            )
            _draw_local_structure_panel(
                axes_arr[row_idx, 1],
                perturbed_geometry["points"],
                perturbed_geometry["edges"],
                point_colors=perturbed_colors,
                view_elev=view_elev,
                view_azim=view_azim,
                point_size=48.0,
                point_linewidth=0.26,
                edge_alpha=0.60,
                edge_linewidth=0.94,
                display_half_span=display_half_span,
            )

    fig.text(
        layout["left_header_x"],
        layout["header_y"],
        "Pure phase",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="#202020",
    )
    fig.text(
        layout["right_header_x"],
        layout["header_y"],
        "Perturbed phase",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="#202020",
    )

    label_mode_norm = str(label_mode).strip().lower()
    if label_mode_norm not in {"family", "neutral"}:
        raise ValueError(
            "Unsupported label_mode for paper local-base variant. "
            f"Expected one of ['family', 'neutral'], got {label_mode!r}."
        )
    for row_idx, record in enumerate(records):
        bbox = axes_arr[row_idx, 0].get_position()
        y_center = 0.5 * (bbox.y0 + bbox.y1)
        label_color = (
            _darken_color(record["display_color"], factor=0.68)
            if label_mode_norm == "family"
            else "#303030"
        )
        fig.text(
            layout["row_label_x"],
            y_center,
            record["family"],
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=label_color,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _render_local_base_paper_gallery(
    global_cfg: Any,
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    phases: Sequence[str],
    output_dir: pathlib.Path,
    rng: np.random.Generator,
    color_map: Dict[str, Any],
    *,
    target_count: int = 80,
    candidate_limit: int = 256,
    view_elev: float = 22.0,
    view_azim: float = 38.0,
) -> None:
    if len(atom_positions) == 0:
        raise ValueError("Cannot render paper local-base figure because atom_positions is empty.")

    reference_point_clouds = _load_reference_point_clouds(output_dir)
    family_specs = _paper_family_specs(phases, reference_point_clouds)
    positions_tree = cKDTree(atom_positions)

    phase_to_atoms: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for atom in atoms:
        phase_to_atoms[str(atom.get("phase_id"))].append(atom)

    xyz_dir = output_dir / "figure_local_base_paper_xyz"
    records: List[Dict[str, Any]] = []
    for spec in family_specs:
        family = spec["family"]
        pure_phase = spec["pure_phase"]
        perturbed_phase = spec["perturbed_phase"]
        display_color = str(spec["display_color"])

        pure_source = str(spec["pure_source"])
        if pure_source == "reference":
            pure_points = _prepare_reference_local_points(
                reference_point_clouds[pure_phase],
                target_count=target_count,
            )
            pure_atoms_for_scale = phase_to_atoms.get(pure_phase, [])
            if not pure_atoms_for_scale:
                raise ValueError(
                    "Paper local-base figure requires unperturbed dataset samples to scale-match "
                    f"reference phase {pure_phase!r}, but none were found."
                )
            pure_scale_rep = _select_local_environment_closest_to_reference(
                pure_phase,
                pure_atoms_for_scale,
                atom_positions,
                positions_tree,
                global_cfg,
                rng,
                reference_points=pure_points,
                target_count=target_count,
                candidate_limit=min(len(pure_atoms_for_scale), max(candidate_limit, 1024)),
            )
            pure_points = _match_local_structure_scale(
                pure_points,
                np.asarray(pure_scale_rep["local_points"], dtype=np.float32),
            )
        elif pure_source == "dataset":
            pure_atoms = phase_to_atoms.get(pure_phase, [])
            if not pure_atoms:
                raise ValueError(
                    "Paper local-base figure could not find any pure samples for "
                    f"phase {pure_phase!r}."
                )
            pure_rep = _select_representative_local_environment(
                pure_phase,
                pure_atoms,
                atom_positions,
                positions_tree,
                global_cfg,
                rng,
                target_count=target_count,
                candidate_limit=candidate_limit,
            )
            pure_points = np.asarray(pure_rep["local_points"], dtype=np.float32)
        else:
            raise ValueError(
                f"Unsupported pure_source {pure_source!r} while rendering the paper local-base figure."
            )

        pure_geometry = _prepare_local_structure_geometry(pure_points)
        pure_oriented = np.asarray(pure_geometry["points"], dtype=np.float32)
        if pure_oriented.shape[0] != int(target_count):
            raise RuntimeError(
                "Paper local-base figure expected the pure structure to contain "
                f"{target_count} points, got {pure_oriented.shape[0]} for phase {pure_phase!r}."
            )

        record: Dict[str, Any] = {
            "family": family,
            "pure_phase": pure_phase,
            "perturbed_phase": perturbed_phase,
            "display_color": display_color,
            "pure_geometry": pure_geometry,
            "perturbed_geometry": None,
        }
        if perturbed_phase is None:
            records.append(record)
            continue

        perturbed_atoms = phase_to_atoms.get(perturbed_phase, [])
        if not perturbed_atoms:
            raise ValueError(
                "Paper local-base figure could not find any perturbed samples for "
                f"phase {perturbed_phase!r}."
            )
        representative = _select_local_environment_closest_to_reference(
            perturbed_phase,
            perturbed_atoms,
            atom_positions,
            positions_tree,
            global_cfg,
            rng,
            reference_points=pure_points,
            target_count=target_count,
            candidate_limit=len(perturbed_atoms),
        )
        perturbed_geometry = _prepare_local_structure_geometry(representative["local_points"])
        perturbed_oriented = np.asarray(perturbed_geometry["points"], dtype=np.float32)
        if perturbed_oriented.shape[0] != int(target_count):
            raise RuntimeError(
                "Paper local-base figure expected the perturbed structure to contain "
                f"{target_count} points, got {perturbed_oriented.shape[0]} for phase {perturbed_phase!r}."
            )
        record["perturbed_geometry"] = perturbed_geometry
        records.append(record)

    display_half_span = _compute_paper_display_half_span(records)

    for record in records:
        pure_geometry = record["pure_geometry"]
        pure_oriented = np.asarray(pure_geometry["points"], dtype=np.float32)
        _save_structure_xyz(
            xyz_dir,
            _paper_xyz_filename(record["family"], "pure"),
            pure_oriented,
            comment=(
                "Paper local-base figure | pure | "
                f"family={record['family']} | phase={record['pure_phase']}"
            ),
        )
        perturbed_geometry = record.get("perturbed_geometry")
        perturbed_phase_name = record.get("perturbed_phase")
        if perturbed_geometry is not None:
            perturbed_oriented = np.asarray(perturbed_geometry["points"], dtype=np.float32)
            _save_structure_xyz(
                xyz_dir,
                _paper_xyz_filename(record["family"], "perturbed"),
                perturbed_oriented,
                comment=(
                    "Paper local-base figure | perturbed | "
                    f"family={record['family']} | phase={perturbed_phase_name}"
                ),
            )

    _render_local_base_paper_variant(
        records,
        output_dir / "figure_local_base_paper.png",
        display_half_span=display_half_span,
        color_mode="family",
        label_mode="family",
        view_elev=view_elev,
        view_azim=view_azim,
    )
    _render_local_base_paper_variant(
        records,
        output_dir / "figure_local_base_paper_colormap.png",
        display_half_span=display_half_span,
        color_mode="colormap",
        label_mode="neutral",
        cmap_name="viridis",
        view_elev=view_elev,
        view_azim=view_azim,
    )


def _render_local_galleries(
    global_cfg: Any,
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    phases: Sequence[str],
    metadata: Dict[str, Any],
    output_dir: pathlib.Path,
    rng: np.random.Generator,
    color_map: Dict[str, Any],
) -> None:
    """Render local neighborhood galleries."""
    if len(atom_positions) == 0:
        return
    base_phases = sorted({p for p in phases if not str(p).startswith('intermediate_')})
    intermediate_phases = sorted({p for p in phases if str(p).startswith('intermediate_')})

    positions_tree = cKDTree(atom_positions)

    _render_local_base_gallery(
        global_cfg,
        atoms,
        atom_positions,
        phases,
        output_dir,
        rng,
        color_map,
        positions_tree=positions_tree,
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


def _darken_color(color: Any, factor: float = 0.62) -> str:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=np.float32)
    return mcolors.to_hex(np.clip(rgb * float(factor), 0.0, 1.0))


def _format_class_label(phase: str, *, width: int = 16) -> str:
    phase_text = str(phase).strip()
    if not phase_text:
        raise ValueError("phase must be a non-empty string.")
    return textwrap.fill(
        phase_text.replace("_", " "),
        width=max(8, int(width)),
        break_long_words=False,
        break_on_hyphens=False,
    )


def _extract_local_neighborhood_points(
    center_atom: Dict[str, Any],
    atom_positions: np.ndarray,
    positions_tree: cKDTree,
    global_cfg: Any,
    *,
    target_count: int,
    radius: Optional[float] = None,
) -> np.ndarray:
    if target_count < 2:
        raise ValueError(f"target_count must be >= 2, got {target_count}.")
    if atom_positions.ndim != 2 or atom_positions.shape[1] != 3:
        raise ValueError(
            f"atom_positions must have shape (N, 3), got {atom_positions.shape}."
        )

    center_pos = np.asarray(center_atom.get("position"), dtype=np.float32)
    if center_pos.shape != (3,):
        raise ValueError(
            "center_atom['position'] must resolve to shape (3,), "
            f"got {center_pos.shape} for atom keys={sorted(center_atom.keys())}."
        )

    radius_use = float(radius) if radius is not None else 2.0 * float(global_cfg.avg_nn_dist)
    if radius_use <= 0.0:
        raise ValueError(f"Neighborhood radius must be positive, got {radius_use}.")

    idxs = positions_tree.query_ball_point(center_pos, r=radius_use)
    target_k = min(int(target_count), int(len(atom_positions)))
    if len(idxs) < target_k:
        _, idxs = positions_tree.query(center_pos, k=target_k)
        idxs = np.atleast_1d(idxs)
    idxs_arr = np.asarray(idxs, dtype=int).reshape(-1)
    if idxs_arr.size == 0:
        raise RuntimeError(
            "Failed to resolve any local neighborhood points for center atom "
            f"final_index={center_atom.get('final_index', '?')} at position={center_pos.tolist()}."
        )

    coords = np.asarray(atom_positions[idxs_arr], dtype=np.float32) - center_pos[None, :]
    dists_sq = np.sum(coords * coords, axis=1)
    order = np.argsort(dists_sq)
    coords = coords[order[:target_k]]
    if coords.shape[0] == 0:
        raise RuntimeError(
            "No local coordinates remain after distance sorting for center atom "
            f"final_index={center_atom.get('final_index', '?')}."
        )
    return coords.astype(np.float32, copy=False)


def _compute_local_environment_descriptor(
    local_points: np.ndarray,
    *,
    descriptor_length: int = 24,
) -> np.ndarray:
    pts = np.asarray(local_points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"local_points must have shape (N, >=3), got {pts.shape}.")
    descriptor_length = max(4, int(descriptor_length))

    radial = np.sort(np.linalg.norm(pts[:, :3], axis=1))
    if radial.size == 0:
        raise ValueError("local_points must contain at least one point.")

    keep = radial[: min(descriptor_length, radial.size)].astype(np.float32, copy=False)
    positive = keep[keep > 1e-8]
    if positive.size == 0:
        scale = 1.0
    else:
        scale = float(np.median(positive[: min(6, positive.size)]))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise RuntimeError(
            "Failed to normalize local environment descriptor because the local radial scale "
            f"is invalid: scale={scale}, descriptor_length={descriptor_length}."
        )

    if keep.size < descriptor_length:
        keep = np.pad(keep, (0, descriptor_length - keep.size), mode="edge")
    return (keep / scale).astype(np.float32, copy=False)


def _select_representative_local_environment(
    phase: str,
    phase_atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    positions_tree: cKDTree,
    global_cfg: Any,
    rng: np.random.Generator,
    *,
    target_count: int = 48,
    candidate_limit: int = 256,
    descriptor_length: int = 24,
) -> Dict[str, Any]:
    if not phase_atoms:
        raise ValueError(f"Cannot select a representative for empty phase {phase!r}.")
    candidate_limit = max(1, int(candidate_limit))
    target_count = max(4, int(target_count))

    if len(phase_atoms) <= candidate_limit:
        candidate_offsets = np.arange(len(phase_atoms), dtype=int)
    else:
        candidate_offsets = np.sort(
            rng.choice(len(phase_atoms), size=candidate_limit, replace=False)
        )

    candidate_records: List[Dict[str, Any]] = []
    for offset in candidate_offsets.tolist():
        atom = phase_atoms[int(offset)]
        local_points = _extract_local_neighborhood_points(
            atom,
            atom_positions,
            positions_tree,
            global_cfg,
            target_count=target_count,
        )
        descriptor = _compute_local_environment_descriptor(
            local_points,
            descriptor_length=descriptor_length,
        )
        candidate_records.append(
            {
                "atom": atom,
                "local_points": local_points,
                "descriptor": descriptor,
            }
        )

    if not candidate_records:
        raise RuntimeError(
            f"Representative selection produced no candidate local environments for phase {phase!r}."
        )

    descriptor_matrix = np.stack(
        [record["descriptor"] for record in candidate_records],
        axis=0,
    ).astype(np.float32, copy=False)
    descriptor_center = np.mean(descriptor_matrix, axis=0, keepdims=True)
    descriptor_dist = np.linalg.norm(descriptor_matrix - descriptor_center, axis=1)
    best_idx = int(np.argmin(descriptor_dist))
    best = dict(candidate_records[best_idx])
    best["descriptor_distance_to_mean"] = float(descriptor_dist[best_idx])
    best["phase"] = str(phase)
    best["candidate_count_evaluated"] = int(len(candidate_records))
    return best


def _select_local_environment_closest_to_reference(
    phase: str,
    phase_atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    positions_tree: cKDTree,
    global_cfg: Any,
    rng: np.random.Generator,
    *,
    reference_points: np.ndarray,
    target_count: int = 80,
    candidate_limit: int = 256,
    descriptor_length: int = 24,
) -> Dict[str, Any]:
    if not phase_atoms:
        raise ValueError(f"Cannot select a reference-matched environment for empty phase {phase!r}.")

    reference_local = np.asarray(reference_points, dtype=np.float32)
    if reference_local.ndim != 2 or reference_local.shape[1] != 3:
        raise ValueError(
            f"reference_points must have shape (N, 3), got {reference_local.shape}."
        )
    reference_descriptor = _compute_local_environment_descriptor(
        reference_local,
        descriptor_length=descriptor_length,
    )

    candidate_limit = max(1, int(candidate_limit))
    if len(phase_atoms) <= candidate_limit:
        candidate_offsets = np.arange(len(phase_atoms), dtype=int)
    else:
        candidate_offsets = np.sort(
            rng.choice(len(phase_atoms), size=candidate_limit, replace=False)
        )

    best_record: Optional[Dict[str, Any]] = None
    best_distance = float("inf")
    for offset in candidate_offsets.tolist():
        atom = phase_atoms[int(offset)]
        local_points = _extract_local_neighborhood_points(
            atom,
            atom_positions,
            positions_tree,
            global_cfg,
            target_count=target_count,
        )
        descriptor = _compute_local_environment_descriptor(
            local_points,
            descriptor_length=descriptor_length,
        )
        distance = float(np.linalg.norm(descriptor - reference_descriptor))
        if distance < best_distance:
            best_distance = distance
            best_record = {
                "atom": atom,
                "local_points": local_points,
                "descriptor": descriptor,
                "reference_descriptor_distance": float(distance),
            }

    if best_record is None:
        raise RuntimeError(
            f"Reference-guided representative selection produced no candidates for phase {phase!r}."
        )
    best_record["phase"] = str(phase)
    best_record["candidate_count_evaluated"] = int(len(candidate_offsets))
    return best_record


def _compute_center_to_edge_colors(
    points: np.ndarray,
    base_color: Any,
    *,
    center_lighten: float = 0.76,
    edge_darken: float = 0.34,
    radius_percentile: float = 88.0,
    gamma: float = 0.86,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)[:, :3]
    base_rgb = np.asarray(mcolors.to_rgb(base_color), dtype=np.float32)

    if pts.shape[0] == 1:
        return base_rgb.reshape(1, 3)

    centroid = np.mean(pts, axis=0, dtype=np.float64)
    dists = np.linalg.norm(pts - centroid[None, :], axis=1)
    radius_scale = float(np.percentile(dists, float(radius_percentile)))
    if radius_scale <= 1e-12:
        t = np.zeros_like(dists, dtype=np.float32)
    else:
        t = np.clip(dists / radius_scale, 0.0, 1.0).astype(np.float32, copy=False)
    t = np.power(t, float(gamma)).astype(np.float32, copy=False)

    center_rgb = base_rgb + float(center_lighten) * (1.0 - base_rgb)
    edge_rgb = base_rgb * (1.0 - float(edge_darken))
    colors = center_rgb[None, :] * (1.0 - t[:, None]) + edge_rgb[None, :] * t[:, None]
    return np.clip(colors, 0.0, 1.0).astype(np.float32, copy=False)


def _resolve_local_coordination_shell(
    points: np.ndarray,
    *,
    min_shell_neighbors: int = 3,
    max_shell_neighbors: int = 6,
    shell_gap_ratio: float = 1.18,
) -> Dict[str, Any]:
    pts = np.asarray(points, dtype=np.float32)[:, :3]
    n_points = int(pts.shape[0])
    if n_points < 2:
        return {
            "directed_neighbors": [[] for _ in range(n_points)],
            "distance_matrix": np.full((n_points, n_points), np.inf, dtype=np.float32),
            "shell_neighbor_counts": np.zeros((n_points,), dtype=np.int32),
            "shell_cutoffs": np.zeros((n_points,), dtype=np.float32),
        }

    min_shell_neighbors = max(1, int(min_shell_neighbors))
    max_shell_neighbors = min(max(min_shell_neighbors, int(max_shell_neighbors)), n_points - 1)
    dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    sorted_idx = np.argsort(dmat, axis=1)
    sorted_dist = np.take_along_axis(dmat, sorted_idx, axis=1)

    candidate_count = min(n_points - 1, max_shell_neighbors + 1)
    local_shell_counts = np.zeros(n_points, dtype=np.int32)
    local_cutoffs = np.zeros(n_points, dtype=np.float32)
    directed_neighbors: List[List[int]] = [[] for _ in range(n_points)]

    for point_idx in range(n_points):
        candidate_dist = sorted_dist[point_idx, :candidate_count]
        finite_mask = np.isfinite(candidate_dist)
        candidate_dist = candidate_dist[finite_mask]
        candidate_neighbors = sorted_idx[point_idx, :candidate_dist.size]

        lower_bound = min(min_shell_neighbors, candidate_dist.size)
        upper_bound = min(max_shell_neighbors, candidate_dist.size)
        shell_count = upper_bound
        if candidate_dist.size >= 2:
            gap_start = max(0, lower_bound - 1)
            gap_stop = min(upper_bound, candidate_dist.size - 1)
            for gap_idx in range(gap_start, gap_stop):
                curr = float(candidate_dist[gap_idx])
                nxt = float(candidate_dist[gap_idx + 1])
                if curr <= 0.0:
                    continue
                if (nxt / curr) >= float(shell_gap_ratio):
                    shell_count = gap_idx + 1
                    break
        local_shell_counts[point_idx] = int(shell_count)

        if candidate_dist.size > shell_count:
            cutoff = 0.5 * (
                float(candidate_dist[shell_count - 1]) + float(candidate_dist[shell_count])
            )
        else:
            cutoff = float(candidate_dist[shell_count - 1]) * 1.06
        local_cutoffs[point_idx] = float(cutoff)

        neighbor_ids = [
            int(candidate_neighbors[idx])
            for idx, dist in enumerate(candidate_dist)
            if idx < int(shell_count) or float(dist) <= float(cutoff)
        ]
        if not neighbor_ids:
            neighbor_ids = [int(candidate_neighbors[0])]
        directed_neighbors[point_idx] = sorted(
            set(int(v) for v in neighbor_ids if int(v) != point_idx)
        )

    return {
        "directed_neighbors": directed_neighbors,
        "distance_matrix": dmat.astype(np.float32, copy=False),
        "shell_neighbor_counts": local_shell_counts,
        "shell_cutoffs": local_cutoffs,
    }


def _build_local_coordination_edges(
    points: np.ndarray,
    *,
    min_shell_neighbors: int = 3,
    max_shell_neighbors: int = 6,
    shell_gap_ratio: float = 1.18,
    edge_mode: str = "coordination_shell_mutual",
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    shell = _resolve_local_coordination_shell(
        points,
        min_shell_neighbors=min_shell_neighbors,
        max_shell_neighbors=max_shell_neighbors,
        shell_gap_ratio=shell_gap_ratio,
    )
    neighbor_lists = shell["directed_neighbors"]
    dmat = np.asarray(shell["distance_matrix"], dtype=np.float32)
    local_shell_counts = np.asarray(shell["shell_neighbor_counts"], dtype=np.int32)
    local_cutoffs = np.asarray(shell["shell_cutoffs"], dtype=np.float32)
    n_points = int(dmat.shape[0])
    if n_points < 2:
        return [], {
            "edge_mode": str(edge_mode),
            "num_edges": 0,
            "shell_neighbor_count_median": 0.0,
            "shell_cutoff_median": 0.0,
        }

    edge_mode_norm = str(edge_mode).strip().lower()
    alias_map = {
        "coordination_shell": "coordination_shell",
        "shell_union": "coordination_shell",
        "union": "coordination_shell",
        "coordination_shell_mutual": "coordination_shell_mutual",
        "shell_mutual": "coordination_shell_mutual",
        "mutual": "coordination_shell_mutual",
        "coordination_shell_degree_capped": "coordination_shell_degree_capped",
        "shell_degree_capped": "coordination_shell_degree_capped",
        "degree_capped": "coordination_shell_degree_capped",
    }
    if edge_mode_norm not in alias_map:
        raise ValueError(
            "Unsupported representative edge mode: "
            f"{edge_mode!r}. Expected one of "
            "['coordination_shell', 'coordination_shell_mutual', "
            "'coordination_shell_degree_capped']."
        )
    edge_mode_use = alias_map[edge_mode_norm]

    neighbor_sets = [set(neighbors) for neighbors in neighbor_lists]
    union_edges: set[Tuple[int, int]] = set()
    mutual_edges: set[Tuple[int, int]] = set()
    for point_idx, neighbors in enumerate(neighbor_lists):
        for neighbor_idx in neighbors:
            edge = (min(point_idx, neighbor_idx), max(point_idx, neighbor_idx))
            if edge[0] == edge[1]:
                continue
            union_edges.add(edge)
            if point_idx in neighbor_sets[neighbor_idx]:
                mutual_edges.add(edge)

    if edge_mode_use == "coordination_shell":
        edges = sorted(union_edges)
    elif edge_mode_use == "coordination_shell_mutual":
        edges = sorted(mutual_edges if mutual_edges else union_edges)
    else:
        candidate_edges = sorted(
            mutual_edges if mutual_edges else union_edges,
            key=lambda edge: float(dmat[edge[0], edge[1]]),
        )
        median_degree = int(
            np.clip(np.rint(np.median(local_shell_counts)), 3, max(3, max_shell_neighbors))
        )
        degrees = np.zeros((n_points,), dtype=np.int32)
        selected: List[Tuple[int, int]] = []
        for edge in candidate_edges:
            p0, p1 = edge
            if degrees[p0] >= median_degree or degrees[p1] >= median_degree:
                continue
            selected.append(edge)
            degrees[p0] += 1
            degrees[p1] += 1
        if not selected and candidate_edges:
            selected = [candidate_edges[0]]
        edges = selected

    edge_info = {
        "edge_mode": str(edge_mode_use),
        "num_edges": int(len(edges)),
        "shell_neighbor_count_median": float(np.median(local_shell_counts)),
        "shell_neighbor_count_min": int(np.min(local_shell_counts)),
        "shell_neighbor_count_max": int(np.max(local_shell_counts)),
        "shell_cutoff_median": float(np.median(local_cutoffs)),
        "candidate_union_edges": int(len(union_edges)),
        "candidate_mutual_edges": int(len(mutual_edges)),
    }
    if edge_mode_use == "coordination_shell_degree_capped":
        edge_info["degree_cap"] = int(
            np.clip(np.rint(np.median(local_shell_counts)), 3, max(3, max_shell_neighbors))
        )

    return edges, edge_info


def _draw_edges(
    ax: Any,
    points: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    *,
    point_colors: Optional[np.ndarray] = None,
    edge_color: str = "#5f5f5f",
    edge_alpha: float = 0.60,
    edge_linewidth: float = 0.94,
) -> None:
    pts = np.asarray(points, dtype=np.float32)[:, :3]
    point_rgb: Optional[np.ndarray] = None
    if point_colors is not None:
        point_rgb = np.clip(np.asarray(point_colors, dtype=np.float32)[:, :3], 0.0, 1.0)

    for edge in edges:
        p1, p2 = pts[int(edge[0])], pts[int(edge[1])]
        if point_rgb is not None:
            edge_rgb = 0.5 * (point_rgb[int(edge[0])] + point_rgb[int(edge[1])])
            edge_rgb = np.clip(edge_rgb * 0.78, 0.0, 1.0)
            color_use: Any = (float(edge_rgb[0]), float(edge_rgb[1]), float(edge_rgb[2]))
        else:
            color_use = edge_color
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=color_use,
            linewidth=float(edge_linewidth),
            alpha=float(edge_alpha),
        )


def _set_equal_axes_3d(ax: Any, coords: np.ndarray, *, half_span: Optional[float] = None) -> None:
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    center = 0.5 * (mins + maxs)
    if half_span is None:
        span = float(np.max(maxs - mins))
        if not np.isfinite(span) or span <= 0.0:
            span = 1.0
        half = 0.5 * span
    else:
        half = float(half_span)
        if not np.isfinite(half) or half <= 0.0:
            raise ValueError(f"half_span must be positive and finite, got {half_span}.")
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def _orient_points_for_representative_view(points: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    pts = np.asarray(points, dtype=np.float32)[:, :3]
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}.")
    if pts.shape[0] == 0:
        raise ValueError("points must contain at least one coordinate.")

    center_idx = int(np.argmin(np.linalg.norm(pts, axis=1)))
    centered = pts - pts[center_idx]
    cov = centered.T @ centered
    eigvals_raw, eigvecs_raw = np.linalg.eigh(cov)
    order = np.argsort(eigvals_raw)[::-1]
    eigvals = np.asarray(eigvals_raw[order], dtype=np.float64)
    basis = np.asarray(eigvecs_raw[:, order], dtype=np.float64)
    if basis.shape != (3, 3):
        raise RuntimeError(
            f"Internal error: expected PCA basis shape (3, 3), got {basis.shape}."
        )

    for axis_idx in range(3):
        proj = centered @ basis[:, axis_idx]
        anchor_idx = int(np.argmax(np.abs(proj)))
        if float(proj[anchor_idx]) < 0.0:
            basis[:, axis_idx] *= -1.0

    det_basis = float(np.linalg.det(basis))
    if not np.isfinite(det_basis):
        raise ValueError(f"Invalid PCA basis determinant: det={det_basis}.")
    if det_basis < 0.0:
        basis[:, 2] *= -1.0
        det_basis = float(np.linalg.det(basis))
    if det_basis <= 0.0:
        raise ValueError(f"Failed to build a right-handed PCA basis: det={det_basis}.")

    oriented = centered @ basis
    return oriented.astype(np.float32, copy=False), {
        "orientation_method": "pca",
        "center_index": int(center_idx),
        "pca_eigenvalues": [float(v) for v in eigvals.tolist()],
        "pca_basis_det": float(det_basis),
    }


def _render_local_representative_gallery(
    title: str,
    filename: pathlib.Path,
    selected_phases: Sequence[str],
    atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    positions_tree: cKDTree,
    global_cfg: Any,
    rng: np.random.Generator,
    color_map: Dict[str, Any],
    *,
    target_count: int = 48,
    candidate_limit: int = 256,
    view_elev: float = 22.0,
    view_azim: float = 38.0,
) -> None:
    if not selected_phases:
        return

    phase_to_atoms: Dict[str, List[Dict[str, Any]]] = {str(phase): [] for phase in selected_phases}
    for atom in atoms:
        phase = str(atom.get("phase_id"))
        if phase in phase_to_atoms:
            phase_to_atoms[phase].append(atom)
    phase_list = [phase for phase in selected_phases if phase_to_atoms[str(phase)]]
    if not phase_list:
        return

    n_panels = len(phase_list)
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / max(1, n_cols)))
    fig = plt.figure(figsize=(3.45 * n_cols, 3.5 * n_rows), dpi=220, facecolor="white")

    for pos, phase in enumerate(phase_list):
        representative = _select_representative_local_environment(
            str(phase),
            phase_to_atoms[str(phase)],
            atom_positions,
            positions_tree,
            global_cfg,
            rng,
            target_count=target_count,
            candidate_limit=candidate_limit,
        )
        local_oriented, _ = _orient_points_for_representative_view(representative["local_points"])
        base_color = color_map.get(str(phase), _DEFAULT_GRAY_RGBA)
        point_colors = _compute_center_to_edge_colors(local_oriented, base_color)
        edges, _ = _build_local_coordination_edges(
            local_oriented,
            min_shell_neighbors=2,
            max_shell_neighbors=5,
            shell_gap_ratio=1.22,
            edge_mode="coordination_shell_mutual",
        )

        ax = fig.add_subplot(n_rows, n_cols, pos + 1, projection="3d")
        ax.set_facecolor("white")
        if hasattr(ax, "set_proj_type"):
            ax.set_proj_type("ortho")
        ax.view_init(elev=float(view_elev), azim=float(view_azim))

        _draw_edges(
            ax,
            local_oriented,
            edges,
            point_colors=point_colors,
            edge_alpha=0.60,
            edge_linewidth=0.94,
        )
        ax.scatter(
            local_oriented[:, 0],
            local_oriented[:, 1],
            local_oriented[:, 2],
            c=point_colors,
            s=58,
            alpha=0.97,
            edgecolors="#222222",
            linewidths=0.36,
            depthshade=False,
        )
        _set_equal_axes_3d(ax, local_oriented)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            if hasattr(axis, "pane"):
                axis.pane.fill = False
                axis.pane.set_edgecolor((1.0, 1.0, 1.0, 1.0))
            if hasattr(axis, "line"):
                axis.line.set_color((1.0, 1.0, 1.0, 1.0))

        ax.set_title(
            _format_class_label(str(phase)),
            fontsize=10,
            color=_darken_color(base_color),
            pad=3,
            fontweight="bold",
        )

    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.965)
    fig.subplots_adjust(left=0.02, right=0.988, bottom=0.035, top=0.91, wspace=0.03, hspace=0.12)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


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
    radius: Optional[float] = None
    if hasattr(global_cfg, 'objects_per_grain'):
        total_objects = global_cfg.grain_count * global_cfg.objects_per_grain
        if total_objects > 0:
            vol_per_object = (global_cfg.L**3 * 0.15) / total_objects
            base_radius = (vol_per_object / (4/3 * np.pi))**(1/3)
            radius = base_radius * 5.0
            target_count = min(int(global_cfg.points_per_object), 500)

    coords = _extract_local_neighborhood_points(
        center_atom,
        atom_positions,
        positions_tree,
        global_cfg,
        target_count=target_count,
        radius=radius,
    )
    
    # Rotate to align with crystal orientation if available
    phase = center_atom.get('phase_id', '')
    if not str(phase).startswith('amorphous') and 'orientation' in center_atom:
        rotation = np.array(center_atom['orientation'])
        if rotation.shape == (3, 3):
            coords = (rotation.T @ coords.T).T
    
    # Color by distance from center
    distances = np.linalg.norm(coords, axis=1)
    max_dist = np.max(distances) if len(distances) > 0 else float(global_cfg.avg_nn_dist)
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
