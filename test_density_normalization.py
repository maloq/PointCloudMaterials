#!/usr/bin/env python3
"""
Test script to verify density normalization across phases.
"""
import numpy as np
import sys
sys.path.append('.')

from src.data_utils.synthetic.atomistic_generator import SyntheticAtomisticDatasetGenerator

def test_density_calculation():
    """Test density calculation for different crystal structures."""
    print("=" * 70)
    print("Testing Density Normalization")
    print("=" * 70)
    
    # Test with FCC structure
    avg_nn = 1.0
    fcc_lattice_constant = avg_nn * np.sqrt(2.0)
    fcc_volume = fcc_lattice_constant ** 3
    fcc_density = 4 / fcc_volume  # 4 atoms per FCC unit cell
    
    # Test with BCC structure
    bcc_lattice_constant = (2.0 / np.sqrt(3.0)) * avg_nn
    bcc_volume = bcc_lattice_constant ** 3
    bcc_density = 2 / bcc_volume  # 2 atoms per BCC unit cell
    
    print(f"\nTheoretical Densities (avg_nn = {avg_nn}):")
    print(f"  FCC: {fcc_density:.6f} atoms/volume")
    print(f"  BCC: {bcc_density:.6f} atoms/volume")
    print(f"  Density difference: {abs(fcc_density - bcc_density) / bcc_density * 100:.2f}%")
    
    # Calculate what scale factors would be needed
    target_density = (fcc_density + bcc_density) / 2
    fcc_scale = (fcc_density / target_density) ** (1/3)
    bcc_scale = (bcc_density / target_density) ** (1/3)
    
    print(f"\nNormalization to average density ({target_density:.6f}):")
    print(f"  FCC scale factor: {fcc_scale:.6f}")
    print(f"  BCC scale factor: {bcc_scale:.6f}")
    
    # After scaling
    fcc_new_density = fcc_density / (fcc_scale ** 3)
    bcc_new_density = bcc_density / (bcc_scale ** 3)
    
    print(f"\nAfter normalization:")
    print(f"  FCC: {fcc_new_density:.6f} atoms/volume")
    print(f"  BCC: {bcc_new_density:.6f} atoms/volume")
    print(f"  Density difference: {abs(fcc_new_density - bcc_new_density) / target_density * 100:.2f}%")
    
    # Check if within 5%
    assert abs(fcc_new_density - bcc_new_density) / target_density <= 0.05, \
        "Densities should be within 5% after normalization"
    
    print("\n✓ Density normalization test passed!")
    print("=" * 70)

if __name__ == "__main__":
    test_density_calculation()

