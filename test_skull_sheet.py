#!/usr/bin/env python3
"""
Test script for skull sheet functionality
Verifies that the skull sheet is correctly added to the simulation
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sim import add_skull_sheet_to_properties, SKULL_SHEET_THICKNESS_MM, SKULL_SHEET_DISTANCE_MM
from utils import create_water_grid

def test_skull_sheet():
    """Test the skull sheet addition to material properties"""
    
    print("Testing skull sheet functionality...")
    
    # Create a test water grid
    grid_dims = (50, 50, 100)
    spacing_mm = 0.5
    properties = create_water_grid(grid_dims_voxels=grid_dims, spacing_mm=spacing_mm)
    
    # Test parameters
    dx = spacing_mm / 1000  # Convert to meters
    array_center_kwave = np.array([0, 0, -0.015])  # 15mm from center (within grid)
    normal_vector = np.array([0, 0, 1])  # Pointing down
    
    # Original properties
    original_skull_count = properties['skull_mask'].sum()
    original_density_range = (properties['density'].min(), properties['density'].max())
    original_sound_speed_range = (properties['sound_speed'].min(), properties['sound_speed'].max())
    
    print(f"Original grid shape: {properties['density'].shape}")
    print(f"Original skull voxels: {original_skull_count}")
    print(f"Original density range: {original_density_range}")
    print(f"Original sound speed range: {original_sound_speed_range}")
    
    # Add skull sheet
    properties = add_skull_sheet_to_properties(properties, dx, array_center_kwave, normal_vector)
    
    # Check results
    new_skull_count = properties['skull_mask'].sum()
    new_density_range = (properties['density'].min(), properties['density'].max())
    new_sound_speed_range = (properties['sound_speed'].min(), properties['sound_speed'].max())
    
    print(f"\nAfter adding skull sheet:")
    print(f"New skull voxels: {new_skull_count}")
    print(f"New density range: {new_density_range}")
    print(f"New sound speed range: {new_sound_speed_range}")
    
    # Verify skull sheet was added
    assert new_skull_count > original_skull_count, "Skull sheet should increase skull voxel count"
    assert new_density_range[1] > original_density_range[1], "Skull sheet should increase max density"
    assert new_sound_speed_range[1] > original_sound_speed_range[1], "Skull sheet should increase max sound speed"
    
    print("\n✓ Skull sheet test passed!")
    print(f"Added {new_skull_count - original_skull_count} skull voxels")
    print(f"Skull sheet coverage: {100*(new_skull_count - original_skull_count)/np.prod(grid_dims):.2f}% of grid")
    
    return True

if __name__ == "__main__":
    test_skull_sheet() 