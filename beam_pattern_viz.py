#!/usr/bin/env python3
"""
Beam Pattern Visualization Tool
Creates beam pattern plots showing pressure field with FWHM measurements
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import json
import os
import argparse
import h5py
from pathlib import Path


def load_simulation_data(sim_dir):
    """Load simulation data from directory"""
    
    # Load config
    config_file = os.path.join(sim_dir, "config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Load pressure data
    pressure_file = os.path.join(sim_dir, "output.h5")
    if not os.path.exists(pressure_file):
        raise FileNotFoundError(f"Pressure file not found: {pressure_file}")
    
    with h5py.File(pressure_file, 'r') as f:
        if 'p' in f:
            pressure_data = f['p'][:]
            # Get actual grid dimensions from file
            Nx = int(f['Nx'][...].item())
            Ny = int(f['Ny'][...].item()) 
            Nz = int(f['Nz'][...].item())
            Nt = int(f['Nt'][...].item())
            actual_grid_shape = (Nx, Ny, Nz)
            print(f"HDF5 grid dimensions: {actual_grid_shape}, time steps: {Nt}")
        else:
            raise ValueError("No pressure data found in output file")
    
    # Load simulation data
    sim_data_file = os.path.join(sim_dir, "simulation_data.npz")
    if os.path.exists(sim_data_file):
        sim_data = np.load(sim_data_file)
        source_mask = sim_data['source_mask']
        focus_point = sim_data['focus_point']
    else:
        source_mask = None
        focus_point = None
    
    return pressure_data, config, source_mask, focus_point, actual_grid_shape


def find_peak_pressure_time_and_location(pressure_data, grid_shape, dx):
    """Find the time and location of peak pressure with corrected coordinate system"""
    
    # Reshape pressure data if needed
    if len(pressure_data.shape) == 3 and pressure_data.shape[0] == 1:
        # Shape is (1, time, flattened_space) - remove first dimension
        pressure_data = pressure_data[0]
    
    if len(pressure_data.shape) == 2:
        # Reshape from (time, flattened_space) to (time, x, y, z)
        # k-Wave uses column-major (Fortran) ordering, so we need to reshape with 'F' order
        p_data_4d = pressure_data.reshape((pressure_data.shape[0],) + grid_shape, order='F')
    else:
        p_data_4d = pressure_data
    
    print(f"Pressure data shape: {p_data_4d.shape}")
    print(f"Grid shape: {grid_shape}")
    
    # Find the time point with maximum pressure
    print(f"4D pressure data shape: {p_data_4d.shape}")
    if len(p_data_4d.shape) == 4:
        # Shape is (time, z, y, x)
        max_pressures_per_time = np.max(np.abs(p_data_4d), axis=(1,2,3))
        max_time_idx = np.argmax(max_pressures_per_time)
        max_pressure_overall = max_pressures_per_time[max_time_idx]
    else:
        # Shape is (z, y, x) - single time point
        max_pressure_overall = np.max(np.abs(p_data_4d))
        max_time_idx = 0
        # Add time dimension
        p_data_4d = p_data_4d[np.newaxis, ...]
    
    print(f"Maximum pressure: {max_pressure_overall:.2e} Pa at time index {max_time_idx}")
    
    # Find spatial location of maximum pressure
    pressure_at_max_time = np.abs(p_data_4d[max_time_idx])
    max_spatial_idx = np.unravel_index(np.argmax(pressure_at_max_time), pressure_at_max_time.shape)
    
    print(f"Max pressure voxel coordinates: {max_spatial_idx}")
    
    # Convert voxel coordinates to physical coordinates (mm)
    # k-Wave uses grid center as origin, so we need to offset by grid center
    grid_center_voxels = np.array(pressure_at_max_time.shape) / 2.0
    actual_focal_point_voxels = np.array(max_spatial_idx)
    
    # Calculate offset from grid center in voxels
    actual_focal_point_offset_voxels = actual_focal_point_voxels - grid_center_voxels
    
    # Convert to physical coordinates (mm)
    actual_focal_point_mm = actual_focal_point_offset_voxels * dx * 1000
    
    print(f"Grid center (voxels): {grid_center_voxels}")
    print(f"Actual focal point (voxels): {actual_focal_point_voxels}")
    print(f"Offset from center (voxels): {actual_focal_point_offset_voxels}")
    print(f"Actual focal point (mm): {actual_focal_point_mm}")
    
    return pressure_at_max_time, max_spatial_idx, actual_focal_point_mm, max_pressure_overall


def calculate_fwhm_with_bounds(profile, center_idx, dx_mm):
    """Calculate Full Width at Half Maximum with proper bounds"""
    if len(profile) == 0:
        return 0.0, 0, 0
    
    half_max_val = np.max(profile) / 2.0
    
    # Find indices where profile exceeds half maximum
    above_half_max = profile >= half_max_val
    if not np.any(above_half_max):
        return 0.0, 0, 0
    
    # Find the extent of the region above half maximum
    indices = np.where(above_half_max)[0]
    if len(indices) == 0:
        return 0.0, 0, 0
    
    # Get the first and last indices above half maximum
    start_idx = indices[0]
    end_idx = indices[-1]
    
    # Calculate FWHM width
    fwhm_width_voxels = end_idx - start_idx + 1
    fwhm_width_mm = fwhm_width_voxels * dx_mm
    
    return fwhm_width_mm, start_idx, end_idx


def create_beam_pattern_visualization(pressure_field, max_spatial_idx, dx_mm, 
                                    intended_focal_point_mm, actual_focal_point_mm,
                                    config, output_dir):
    """Create beam pattern visualization similar to the reference image"""
    
    # Get profiles through the focal point
    z_idx, y_idx, x_idx = max_spatial_idx
    
    # Create coordinate arrays in mm
    grid_shape = pressure_field.shape
    grid_center_voxels = np.array(grid_shape) / 2.0
    
    # Calculate coordinate arrays relative to grid center
    x_coords_mm = (np.arange(grid_shape[2]) - grid_center_voxels[2]) * dx_mm
    y_coords_mm = (np.arange(grid_shape[1]) - grid_center_voxels[1]) * dx_mm
    z_coords_mm = (np.arange(grid_shape[0]) - grid_center_voxels[0]) * dx_mm
    
    # Extract 2D slice through focal point (Y-X plane at focal Z)
    focal_slice = pressure_field[z_idx, :, :]
    
    # Calculate FWHM in each direction
    # For ultrasound: axial = along beam axis (Z), lateral = perpendicular to beam (X,Y)
    axial_profile = pressure_field[:, y_idx, x_idx]  # Along Z (axial - beam direction)
    lateral_x_profile = pressure_field[z_idx, y_idx, :]  # Along X (lateral)
    lateral_y_profile = pressure_field[z_idx, :, x_idx]  # Along Y (lateral)
    
    # Calculate FWHM with bounds
    axial_fwhm, axial_start, axial_end = calculate_fwhm_with_bounds(axial_profile, z_idx, dx_mm)
    lateral_x_fwhm, x_start, x_end = calculate_fwhm_with_bounds(lateral_x_profile, x_idx, dx_mm)
    lateral_y_fwhm, y_start, y_end = calculate_fwhm_with_bounds(lateral_y_profile, y_idx, dx_mm)
    
    print(f"\nFWHM Analysis:")
    print(f"  Axial (Z): {axial_fwhm:.2f} mm (voxels {axial_start}-{axial_end})")
    print(f"  Lateral X: {lateral_x_fwhm:.2f} mm (voxels {x_start}-{x_end})")
    print(f"  Lateral Y: {lateral_y_fwhm:.2f} mm (voxels {y_start}-{y_end})")
    
    # Create the main beam pattern plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create meshgrid for the 2D slice
    X, Y = np.meshgrid(x_coords_mm, y_coords_mm)
    
    # Plot the pressure field
    im = ax.contourf(X, Y, focal_slice, levels=50, cmap='hot')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Pressure [Pa]')
    
    # Calculate FWHM rectangle bounds in mm coordinates
    # Convert voxel indices to mm coordinates (for the 2D slice in X-Y plane)
    x_start_mm = (x_start - grid_center_voxels[2]) * dx_mm
    x_end_mm = (x_end - grid_center_voxels[2]) * dx_mm
    y_start_mm = (y_start - grid_center_voxels[1]) * dx_mm
    y_end_mm = (y_end - grid_center_voxels[1]) * dx_mm
    
    # Draw FWHM rectangle (lateral X and Y dimensions)
    rect_width = x_end_mm - x_start_mm
    rect_height = y_end_mm - y_start_mm
    rect_x = x_start_mm
    rect_y = y_start_mm
    
    # Add rectangle showing FWHM bounds
    rect = Rectangle((rect_x, rect_y), rect_width, rect_height, 
                    linewidth=2, edgecolor='cyan', facecolor='none', 
                    linestyle='--', alpha=0.8)
    ax.add_patch(rect)
    
    # Add dimension labels
    # Horizontal dimension (lateral X)
    ax.annotate('', xy=(x_start_mm, y_end_mm + 0.5), xytext=(x_end_mm, y_end_mm + 0.5),
                arrowprops=dict(arrowstyle='<->', color='cyan', lw=2))
    ax.text((x_start_mm + x_end_mm) / 2, y_end_mm + 0.8, 
            f'{lateral_x_fwhm:.2f} mm', ha='center', va='bottom', 
            color='cyan', fontweight='bold', fontsize=10)
    
    # Vertical dimension (lateral Y)
    ax.annotate('', xy=(x_end_mm + 0.5, y_start_mm), xytext=(x_end_mm + 0.5, y_end_mm),
                arrowprops=dict(arrowstyle='<->', color='cyan', lw=2))
    ax.text(x_end_mm + 0.8, (y_start_mm + y_end_mm) / 2, 
            f'{lateral_y_fwhm:.2f} mm', ha='left', va='center', 
            color='cyan', fontweight='bold', fontsize=10, rotation=90)
    
    # Mark focal points
    ax.plot(actual_focal_point_mm[2], actual_focal_point_mm[1], 'r*', 
            markersize=15, label='Actual Focus')
    ax.plot(intended_focal_point_mm[2], intended_focal_point_mm[1], 'b+', 
            markersize=15, markeredgewidth=3, label='Intended Focus')
    
    # Set labels and title
    ax.set_xlabel('x-position [mm]')
    ax.set_ylabel('y-position [mm]')
    ax.set_title(f'Beam Pattern At Second Harmonic\n'
                f'Frequency: {config["acoustic"]["frequency_hz"]/1e6:.1f} MHz, '
                f'Focal Length: {config["focus"]["focal_length_mm"]:.1f} mm')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Make axes equal
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'beam_pattern_fwhm.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create additional profile plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Axial profile (along Z - beam direction)
    axial_start_mm = (axial_start - grid_center_voxels[0]) * dx_mm
    axial_end_mm = (axial_end - grid_center_voxels[0]) * dx_mm
    axes[0, 0].plot(z_coords_mm, axial_profile, 'b-', linewidth=2)
    axes[0, 0].axhline(y=np.max(axial_profile)/2, color='r', linestyle='--', alpha=0.7)
    axes[0, 0].axvline(x=axial_start_mm, color='cyan', linestyle='--', alpha=0.7)
    axes[0, 0].axvline(x=axial_end_mm, color='cyan', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('z-position [mm]')
    axes[0, 0].set_ylabel('Pressure [Pa]')
    axes[0, 0].set_title(f'Axial Profile (FWHM: {axial_fwhm:.2f} mm)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Lateral X profile
    axes[0, 1].plot(x_coords_mm, lateral_x_profile, 'g-', linewidth=2)
    axes[0, 1].axhline(y=np.max(lateral_x_profile)/2, color='r', linestyle='--', alpha=0.7)
    axes[0, 1].axvline(x=x_start_mm, color='cyan', linestyle='--', alpha=0.7)
    axes[0, 1].axvline(x=x_end_mm, color='cyan', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('x-position [mm]')
    axes[0, 1].set_ylabel('Pressure [Pa]')
    axes[0, 1].set_title(f'Lateral X Profile (FWHM: {lateral_x_fwhm:.2f} mm)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Lateral Y profile
    axes[1, 0].plot(y_coords_mm, lateral_y_profile, 'm-', linewidth=2)
    axes[1, 0].axhline(y=np.max(lateral_y_profile)/2, color='r', linestyle='--', alpha=0.7)
    axes[1, 0].axvline(x=y_start_mm, color='cyan', linestyle='--', alpha=0.7)
    axes[1, 0].axvline(x=y_end_mm, color='cyan', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('y-position [mm]')
    axes[1, 0].set_ylabel('Pressure [Pa]')
    axes[1, 0].set_title(f'Lateral Y Profile (FWHM: {lateral_y_fwhm:.2f} mm)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 2D slice with contours
    axes[1, 1].contourf(X, Y, focal_slice, levels=20, cmap='hot')
    axes[1, 1].contour(X, Y, focal_slice, levels=[np.max(focal_slice)/2], colors='white', linewidths=2)
    axes[1, 1].plot(actual_focal_point_mm[2], actual_focal_point_mm[1], 'r*', markersize=12)
    axes[1, 1].plot(intended_focal_point_mm[2], intended_focal_point_mm[1], 'b+', markersize=12, markeredgewidth=2)
    axes[1, 1].set_xlabel('x-position [mm]')
    axes[1, 1].set_ylabel('y-position [mm]')
    axes[1, 1].set_title('2D Pressure Field with FWHM Contour')
    axes[1, 1].set_aspect('equal')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'beam_pattern_profiles.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'axial_fwhm_mm': axial_fwhm,
        'lateral_x_fwhm_mm': lateral_x_fwhm,
        'lateral_y_fwhm_mm': lateral_y_fwhm,
        'average_lateral_fwhm_mm': (lateral_x_fwhm + lateral_y_fwhm) / 2
    }


def correct_focal_analysis(sim_dir):
    """Correct the focal analysis with proper coordinate system"""
    
    # Load data
    pressure_data, config, source_mask, focus_point, actual_grid_shape = load_simulation_data(sim_dir)
    
    # Get grid parameters - use actual grid shape from HDF5 file
    grid_shape = actual_grid_shape
    dx = config['grid']['dx']  # meters
    dx_mm = dx * 1000  # mm
    
    print(f"Grid shape: {grid_shape}")
    print(f"Grid spacing: {dx_mm:.3f} mm")
    
    # Find peak pressure location
    pressure_field, max_spatial_idx, actual_focal_point_mm, max_pressure = \
        find_peak_pressure_time_and_location(pressure_data, grid_shape, dx)
    
    # Get intended focal point from config
    intended_focal_point_m = np.array(config['focus']['focus_point_m'])
    intended_focal_point_mm = intended_focal_point_m * 1000
    
    print(f"\nFocal Point Analysis:")
    print(f"  Intended focal point: {intended_focal_point_mm} mm")
    print(f"  Actual focal point: {actual_focal_point_mm} mm")
    
    # Calculate error
    focal_point_error_mm = np.linalg.norm(actual_focal_point_mm - intended_focal_point_mm)
    print(f"  Focal point error: {focal_point_error_mm:.2f} mm")
    
    # Create output directory for visualizations
    viz_dir = os.path.join(sim_dir, 'beam_pattern_viz')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create beam pattern visualization
    fwhm_results = create_beam_pattern_visualization(
        pressure_field, max_spatial_idx, dx_mm,
        intended_focal_point_mm, actual_focal_point_mm,
        config, viz_dir
    )
    
    # Calculate theoretical values
    wavelength_mm = (1500 / config['acoustic']['frequency_hz']) * 1000  # Assuming water (λ = c/f)
    aperture_width_mm = (config['array_config']['width_n'] * config['array_config']['element_width_mm'] + 
                        (config['array_config']['width_n'] - 1) * config['array_config']['width_pitch_mm'])
    aperture_length_mm = (config['array_config']['length_n'] * config['array_config']['element_length_mm'] + 
                         (config['array_config']['length_n'] - 1) * config['array_config']['length_pitch_mm'])
    aperture_diameter_mm = (aperture_width_mm + aperture_length_mm) / 2
    f_number = config['focus']['focal_length_mm'] / aperture_diameter_mm
    theoretical_lateral_fwhm = 1.02 * wavelength_mm * f_number
    
    # Save corrected focal analysis
    corrected_analysis = {
        "max_pressure_pa": float(max_pressure),
        "intended_focal_point_mm": intended_focal_point_mm.tolist(),
        "actual_focal_point_mm": actual_focal_point_mm.tolist(),
        "focal_point_error_mm": float(focal_point_error_mm),
        "focal_spot_fwhm_mm": {
            "axial": float(fwhm_results['axial_fwhm_mm']),
            "lateral_x": float(fwhm_results['lateral_x_fwhm_mm']),
            "lateral_y": float(fwhm_results['lateral_y_fwhm_mm']),
            "average_lateral": float(fwhm_results['average_lateral_fwhm_mm'])
        },
        "focal_volume_mm3": float((4/3) * np.pi * (fwhm_results['axial_fwhm_mm']/2) * 
                                 (fwhm_results['lateral_x_fwhm_mm']/2) * 
                                 (fwhm_results['lateral_y_fwhm_mm']/2)),
        "theoretical_lateral_fwhm_mm": float(theoretical_lateral_fwhm),
        "wavelength_mm": float(wavelength_mm),
        "f_number": float(f_number),
        "grid_info": {
            "grid_shape": list(grid_shape),
            "dx_mm": dx_mm,
            "max_spatial_idx": [int(x) for x in max_spatial_idx]
        }
    }
    
    # Save corrected analysis
    with open(os.path.join(viz_dir, "corrected_focal_analysis.json"), "w") as f:
        json.dump(corrected_analysis, f, indent=2)
    
    print(f"\n✓ Corrected focal analysis saved to: {viz_dir}")
    print(f"✓ Beam pattern visualizations saved to: {viz_dir}")
    
    return corrected_analysis


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create beam pattern visualization with FWHM measurements')
    parser.add_argument('sim_dir', help='Simulation directory path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.sim_dir):
        print(f"Error: Simulation directory not found: {args.sim_dir}")
        return
    
    try:
        corrected_analysis = correct_focal_analysis(args.sim_dir)
        
        print("\n" + "="*60)
        print("CORRECTED FOCAL ANALYSIS SUMMARY")
        print("="*60)
        print(f"Maximum pressure: {corrected_analysis['max_pressure_pa']:.0f} Pa")
        print(f"Intended focal point: {corrected_analysis['intended_focal_point_mm']} mm")
        print(f"Actual focal point: {corrected_analysis['actual_focal_point_mm']} mm")
        print(f"Focal point error: {corrected_analysis['focal_point_error_mm']:.2f} mm")
        print(f"FWHM - Axial (Z): {corrected_analysis['focal_spot_fwhm_mm']['axial']:.2f} mm")
        print(f"FWHM - Lateral X: {corrected_analysis['focal_spot_fwhm_mm']['lateral_x']:.2f} mm")
        print(f"FWHM - Lateral Y: {corrected_analysis['focal_spot_fwhm_mm']['lateral_y']:.2f} mm")
        print(f"FWHM - Average Lateral: {corrected_analysis['focal_spot_fwhm_mm']['average_lateral']:.2f} mm")
        print(f"Theoretical lateral FWHM: {corrected_analysis['theoretical_lateral_fwhm_mm']:.2f} mm")
        print(f"Focal volume: {corrected_analysis['focal_volume_mm3']:.1f} mm³")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 