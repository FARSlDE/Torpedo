#!/usr/bin/env python3
"""
Visualize pressure field from k-Wave simulation output
Adapted for synthetic aperture transducer array simulations
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import ndimage
import os
import json
from datetime import datetime
from pathlib import Path


def read_pressure_data(output_filename):
    """Read pressure data from HDF5 output file"""
    with h5py.File(output_filename, 'r') as f:
        print(f"Available datasets: {list(f.keys())}")
        
        # For synthetic aperture, we typically record time series
        if 'p' in f:
            p = f['p'][:]
            print(f"Pressure data shape: {p.shape}")
            return {'p': p}
        else:
            print("No pressure data found in output file")
            return None


def reconstruct_pressure_field(sensor_data, sensor_mask, grid_shape):
    """Reconstruct pressure field from sensor data"""
    # Handle different data shapes from k-Wave output
    if sensor_data.ndim == 3:
        # Shape is (1, num_time_points, num_sensor_points)
        sensor_data = sensor_data[0, :, :].T  # Transpose to (num_sensor_points, num_time_points)
    elif sensor_data.ndim == 2 and sensor_data.shape[0] > sensor_data.shape[1]:
        # Already in correct shape (num_sensor_points, num_time_points)
        pass
    else:
        # Might need to transpose
        sensor_data = sensor_data.T
    
    # We'll take RMS over time as a simple metric
    p_rms_sensor = np.sqrt(np.mean(np.square(sensor_data), axis=1))
    
    # Create full grid
    p_field = np.zeros(np.prod(grid_shape))
    
    # Map sensor data back to grid
    sensor_indices = np.where(sensor_mask.flatten(order='F'))[0]
    p_field[sensor_indices] = p_rms_sensor
    
    # Reshape to 3D
    p_field = p_field.reshape(grid_shape, order='F')
    
    return p_field


def create_combined_slice(pressure_field, bg_image, slice_idx, axis=0, 
                         alpha=0.5, cmap='hot', bg_cmap='gray', 
                         p_range=None, bg_range=None, 
                         array_center=None, element_positions=None):
    """Create a combined slice visualization with background and pressure overlay"""
    
    # Get slices
    if axis == 0:
        bg_slice = bg_image[slice_idx, :, :]
        p_slice = pressure_field[slice_idx, :, :]
        coord_idx = [1, 2]
        axis_name = 'X'
    elif axis == 1:
        bg_slice = bg_image[:, slice_idx, :]
        p_slice = pressure_field[:, slice_idx, :]
        coord_idx = [0, 2]
        axis_name = 'Y'
    else:  # axis == 2
        bg_slice = bg_image[:, :, slice_idx]
        p_slice = pressure_field[:, :, slice_idx]
        coord_idx = [0, 1]
        axis_name = 'Z'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot background
    if bg_range is None:
        vmin, vmax = np.percentile(bg_slice[bg_slice > 0], [1, 99]) if bg_slice.any() else (0, 1)
    else:
        vmin, vmax = bg_range
    ax.imshow(bg_slice.T, cmap=bg_cmap, aspect='equal', vmin=vmin, vmax=vmax, origin='lower')
    
    # Create overlay for pressure
    if p_range is None:
        vmin, vmax = 0, np.max(pressure_field) if pressure_field.max() > 0 else 1
    else:
        vmin, vmax = p_range
    
    # Create masked array for overlay
    masked_p = np.ma.array(p_slice.T, mask=p_slice.T <= vmin)
    
    # Plot pressure overlay
    overlay = ax.imshow(masked_p, cmap=cmap, alpha=alpha, 
                       norm=colors.Normalize(vmin=vmin, vmax=vmax),
                       origin='lower')
    
    # Add colorbar
    cbar = plt.colorbar(overlay, ax=ax)
    cbar.set_label('Pressure (Pa)')
    
    # Plot array center if provided
    if array_center is not None:
        if (axis == 0 and abs(slice_idx - array_center[0]) < 2) or \
           (axis == 1 and abs(slice_idx - array_center[1]) < 2) or \
           (axis == 2 and abs(slice_idx - array_center[2]) < 2):
            ax.plot(array_center[coord_idx[0]], array_center[coord_idx[1]], 
                   'g+', markersize=15, markeredgewidth=3, label='Array Center')
    
    # Plot element positions if provided
    if element_positions is not None and axis == 2:  # Only show on Z slices
        # Convert element positions from mm to grid indices
        for i, pos in enumerate(element_positions):
            if abs(pos[2] - slice_idx) < 2:  # If element is near this slice
                ax.plot(pos[coord_idx[0]], pos[coord_idx[1]], 
                       'wo', markersize=8, markeredgewidth=2)
    
    # Add labels and title
    axis_labels = ['X', 'Y', 'Z']
    ax.set_xlabel(f'{axis_labels[coord_idx[0]]} (grid points)')
    ax.set_ylabel(f'{axis_labels[coord_idx[1]]} (grid points)')
    ax.set_title(f'Pressure field at {axis_name}={slice_idx}')
    
    if array_center is not None:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def visualize_simulation_output(sim_dir):
    """Visualize simulation results from a single transmit element"""
    
    # Load simulation data
    sim_data = np.load(os.path.join(sim_dir, 'simulation_data.npz'))
    
    # Load configuration
    with open(os.path.join(sim_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # Extract data
    grid_hu = sim_data['grid_hu']
    skull_mask = sim_data['skull_mask']
    sensor_mask = sim_data['sensor_mask']
    source_mask = sim_data['source_mask']
    element_positions = sim_data['element_positions']
    array_center = sim_data['array_center_position']
    spacing_mm = sim_data['spacing_mm']
    use_sensor_mask = sim_data.get('use_sensor_mask', True)  # Default to True for backward compatibility
    
    # Convert positions to grid coordinates
    grid_origin = sim_data['grid_origin_voxel']
    element_positions_grid = (element_positions / spacing_mm) + grid_origin
    array_center_grid = array_center / spacing_mm
    
    # Read pressure data
    output_file = os.path.join(sim_dir, 'output.h5')
    if os.path.exists(output_file):
        pressure_data = read_pressure_data(output_file)
        
        if pressure_data and 'p' in pressure_data:
            # Reconstruct pressure field based on sensor mask mode
            if use_sensor_mask:
                print("Reconstructing pressure field from element mask sensors")
                p_field = reconstruct_pressure_field(pressure_data['p'], sensor_mask, grid_hu.shape)
                peak_timestep = None
            else:
                print("Visualizing full grid pressure data")
                p_field, peak_timestep = visualize_full_grid_pressure(pressure_data['p'], grid_hu.shape)
            
            # Create tissue map for background
            tissue_map = np.zeros_like(grid_hu, dtype=float)
            tissue_map[grid_hu > -100] = 0.3  # Tissue
            tissue_map[skull_mask] = 1.0  # Skull
            
            # Create output directory for visualizations
            viz_dir = os.path.join(sim_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # Find slice with maximum pressure
            max_idx = np.unravel_index(np.argmax(p_field), p_field.shape)
            
            # Create visualizations for each axis
            for axis, name in [(0, 'sagittal'), (1, 'coronal'), (2, 'axial')]:
                # Slice through maximum pressure
                slice_idx = max_idx[axis]
                fig = create_combined_slice(
                    p_field,
                    tissue_map,
                    slice_idx,
                    axis=axis,
                    alpha=0.7,
                    array_center=array_center_grid,
                    element_positions=element_positions_grid
                )
                title = f'Transmit Element - Max Pressure Slice'
                if not use_sensor_mask and peak_timestep is not None:
                    title += f' (t={peak_timestep})'
                fig.suptitle(title, fontsize=14)
                fig.savefig(os.path.join(viz_dir, f'pressure_{name}_max.png'), dpi=150)
                plt.close(fig)
                
                # Slice through array center
                slice_idx = int(array_center_grid[axis])
                fig = create_combined_slice(
                    p_field,
                    tissue_map,
                    slice_idx,
                    axis=axis,
                    alpha=0.7,
                    array_center=array_center_grid,
                    element_positions=element_positions_grid
                )
                title = f'Transmit Element - Array Center Slice'
                if not use_sensor_mask and peak_timestep is not None:
                    title += f' (t={peak_timestep})'
                fig.suptitle(title, fontsize=14)
                fig.savefig(os.path.join(viz_dir, f'pressure_{name}_center.png'), dpi=150)
                plt.close(fig)
            
            # Create a summary figure showing source and resulting field
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Show source mask
            source_z = np.where(source_mask)[2][0] if source_mask.any() else grid_hu.shape[2]//2
            ax = axes[0, 0]
            ax.imshow(tissue_map[:, :, source_z].T, cmap='gray', origin='lower')
            source_overlay = np.ma.masked_where(source_mask[:, :, source_z].T == 0, 
                                              source_mask[:, :, source_z].T)
            ax.imshow(source_overlay, cmap='Reds', alpha=0.8, origin='lower')
            ax.set_title('Transmit Element')
            ax.set_xlabel('X (grid points)')
            ax.set_ylabel('Y (grid points)')
            
            # Show sensor mask
            ax = axes[0, 1]
            ax.imshow(tissue_map[:, :, source_z].T, cmap='gray', origin='lower')
            if use_sensor_mask:
                sensor_overlay = np.ma.masked_where(sensor_mask[:, :, source_z].T == 0, 
                                                  sensor_mask[:, :, source_z].T)
                ax.imshow(sensor_overlay, cmap='Blues', alpha=0.8, origin='lower')
                ax.set_title('All Sensor Elements')
            else:
                ax.set_title('Sensor: Entire Grid')
            ax.set_xlabel('X (grid points)')
            ax.set_ylabel('Y (grid points)')
            
            # Show pressure field (XY slice)
            ax = axes[1, 0]
            p_slice_xy = p_field[:, :, max_idx[2]]
            im = ax.imshow(p_slice_xy.T, cmap='hot', origin='lower')
            ax.set_title(f'Pressure Field (Z={max_idx[2]})')
            ax.set_xlabel('X (grid points)')
            ax.set_ylabel('Y (grid points)')
            plt.colorbar(im, ax=ax, label='Pressure (Pa)')
            
            # Show pressure field (XZ slice)
            ax = axes[1, 1]
            p_slice_xz = p_field[:, max_idx[1], :]
            im = ax.imshow(p_slice_xz.T, cmap='hot', origin='lower')
            ax.set_title(f'Pressure Field (Y={max_idx[1]})')
            ax.set_xlabel('X (grid points)')
            ax.set_ylabel('Z (grid points)')
            plt.colorbar(im, ax=ax, label='Pressure (Pa)')
            
            mode_text = "Element Mask" if use_sensor_mask else f"Full Grid (t={peak_timestep})"
            plt.suptitle(f'Simulation Summary - Sensor Mode: {mode_text}')
            plt.tight_layout()
            fig.savefig(os.path.join(viz_dir, 'summary.png'), dpi=150)
            plt.close(fig)
            
            # Save metrics
            max_pressure = np.max(p_field)
            metrics = {
                'max_pressure_Pa': float(max_pressure),
                'max_pressure_location': [int(x) for x in max_idx],
                'rms_pressure_Pa': float(np.sqrt(np.mean(p_field**2))),
                'sensor_mode': 'element_mask' if use_sensor_mask else 'full_grid'
            }
            
            if not use_sensor_mask and peak_timestep is not None:
                metrics['peak_timestep'] = int(peak_timestep)
            
            with open(os.path.join(viz_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"Max pressure: {max_pressure:.2e} Pa")
            print(f"Max pressure location: {max_idx}")
            if not use_sensor_mask and peak_timestep is not None:
                print(f"Peak pressure timestep: {peak_timestep}")
            print(f"Visualizations saved to: {viz_dir}")
            
        else:
            print("No pressure data found in output file")
    else:
        print(f"Output file not found: {output_file}")


def visualize_full_grid_pressure(sensor_data, grid_shape):
    """Visualize pressure field when sensor is entire grid"""
    # Handle different data shapes
    if sensor_data.ndim == 3:
        # Shape is (1, num_time_points, num_grid_points)
        sensor_data = sensor_data[0, :, :]  # Remove first dimension
    
    # Check if we need to transpose
    # sensor_data should be (num_grid_points, num_time_points)
    if sensor_data.shape[0] < sensor_data.shape[1]:
        # Data is likely (num_time_points, num_grid_points), need to transpose
        sensor_data = sensor_data.T
    
    print(f"Full grid sensor data shape: {sensor_data.shape}")
    print(f"Grid shape: {grid_shape}, total points: {np.prod(grid_shape)}")
    
    # Find peak pressure timestep
    # Calculate RMS pressure at each timestep
    rms_per_timestep = np.sqrt(np.mean(np.square(sensor_data), axis=0))
    peak_timestep = np.argmax(rms_per_timestep)
    max_rms = rms_per_timestep[peak_timestep]
    print(f"Peak RMS pressure {max_rms:.2e} Pa at timestep {peak_timestep}")
    
    # Get pressure field at peak timestep
    p_peak = sensor_data[:, peak_timestep]
    
    # Reshape to 3D grid
    p_field = p_peak.reshape(grid_shape, order='F')
    
    return p_field, peak_timestep


def main():
    """Main function to visualize simulation results"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize k-Wave synthetic aperture simulation results')
    parser.add_argument('sim_dir', type=str, help='Simulation directory path')
    parser.add_argument('--all', action='store_true', 
                       help='Process all simulation directories matching pattern')
    
    args = parser.parse_args()
    
    if args.all:
        # Find all simulation directories
        sim_dirs = sorted(Path('.').glob('simulation_synth_aperture_*'))
        print(f"Found {len(sim_dirs)} simulation directories")
        
        for sim_dir in sim_dirs:
            print(f"\nProcessing: {sim_dir}")
            try:
                visualize_simulation_output(sim_dir)
            except Exception as e:
                print(f"Error processing {sim_dir}: {e}")
    else:
        # Process single directory
        visualize_simulation_output(args.sim_dir)


if __name__ == "__main__":
    main() 