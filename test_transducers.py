#!/usr/bin/env python3
"""
Test script to debug transducer array creation
"""

import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.data import Vector
from kwave.utils.kwave_array import kWaveArray
from utils import normal_vector_to_euler_angles
import matplotlib.pyplot as plt


def test_transducer_array():
    """Test transducer array creation with different grid resolutions"""
    
    # Parameters
    element_width_mm = 1.6
    element_length_mm = 1.6
    width_pitch_mm = 0.8
    length_pitch_mm = 0.8
    width_n = 4
    length_n = 4
    
    # Test different grid spacings
    test_spacings_mm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    for dx_mm in test_spacings_mm:
        print(f"\n{'='*60}")
        print(f"Testing with dx = {dx_mm} mm")
        
        # Create grid (50mm cube)
        grid_size_mm = 50
        grid_voxels = int(grid_size_mm / dx_mm)
        if grid_voxels % 2 == 1:
            grid_voxels += 1
        
        dx = dx_mm * 1e-3  # Convert to meters
        kgrid = kWaveGrid(
            Vector([grid_voxels, grid_voxels, grid_voxels]),
            Vector([dx, dx, dx])
        )
        
        print(f"Grid: {grid_voxels}^3 voxels, {grid_size_mm}mm size")
        
        # Initialize kWaveArray
        karray = kWaveArray(
            axisymmetric=False,
            bli_tolerance=0.05,
            bli_type='sinc',
            single_precision=False,
            upsampling_rate=10
        )
        
        # Array positioned at center, pointing down
        array_center = np.array([0, 0, 0])
        normal = np.array([0, 0, 1])
        
        # Create orthonormal basis
        u = np.cross(normal, [1, 0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        v /= np.linalg.norm(v)
        
        # Calculate spacings
        width_spacing = element_width_mm + width_pitch_mm
        length_spacing = element_length_mm + length_pitch_mm
        
        # Add elements
        element_count = 0
        theta_degrees = normal_vector_to_euler_angles(normal)
        
        for i in range(width_n):
            for j in range(length_n):
                width_offset = (i - (width_n - 1) / 2.0) * width_spacing * 1e-3
                length_offset = (j - (length_n - 1) / 2.0) * length_spacing * 1e-3
                
                element_center = array_center + width_offset * u + length_offset * v
                
                karray.add_rect_element(
                    position=element_center.tolist(),
                    Lx=element_width_mm * 1e-3,
                    Ly=element_length_mm * 1e-3,
                    theta=theta_degrees
                )
                element_count += 1
        
        # Get source mask
        source_mask = karray.get_array_binary_mask(kgrid)
        
        # Analysis
        active_voxels = source_mask.sum()
        print(f"\nResults:")
        print(f"  Elements added: {element_count}")
        print(f"  Active voxels: {active_voxels}")
        print(f"  Voxels per element: {active_voxels/element_count:.1f}")
        print(f"  Grid points per pitch: {width_pitch_mm/dx_mm:.1f}")
        
        if active_voxels > 0:
            indices = np.where(source_mask)
            x_extent = (indices[0].max() - indices[0].min()) * dx_mm
            y_extent = (indices[1].max() - indices[1].min()) * dx_mm
            z_extent = (indices[2].max() - indices[2].min()) * dx_mm
            print(f"  Spatial extent: {x_extent:.1f} x {y_extent:.1f} x {z_extent:.1f} mm")
            
            # Expected extent
            expected_x = element_width_mm + (width_n - 1) * width_spacing
            expected_y = element_length_mm + (length_n - 1) * length_spacing
            print(f"  Expected extent: {expected_x:.1f} x {expected_y:.1f} mm")
            
            # Check if elements are resolved
            if x_extent < expected_x * 0.8 or y_extent < expected_y * 0.8:
                print("  ⚠️  Elements may be merging due to coarse resolution!")
            else:
                print("  ✓ Elements appear properly resolved")
        
        # Create XY slice visualization
        z_slice = grid_voxels // 2
        if source_mask[:, :, z_slice].any():
            plt.figure(figsize=(6, 6))
            plt.imshow(source_mask[:, :, z_slice].T, origin='lower', cmap='hot')
            plt.title(f'Transducer mask (XY plane) - dx={dx_mm}mm')
            plt.xlabel('X (voxels)')
            plt.ylabel('Y (voxels)')
            plt.colorbar(label='Source mask')
            plt.tight_layout()
            plt.savefig(f'transducer_test_dx_{dx_mm}mm.png')
            plt.close()
            print(f"  Saved visualization: transducer_test_dx_{dx_mm}mm.png")


if __name__ == "__main__":
    print("Testing transducer array creation with different grid resolutions...")
    test_transducer_array()
    print("\n✓ Test complete! Check the generated PNG files.") 