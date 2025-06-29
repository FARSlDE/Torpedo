#!/usr/bin/env python3
"""
Utilities for k-Wave simulations
Contains data loading, conversion, and visualization functions
"""

import numpy as np
import json
from pathlib import Path
import os
import h5py
from scipy.ndimage import zoom
import pydicom
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class SimpleDICOMLoader:
    """Simple DICOM loader that returns numpy arrays directly"""
    
    def __init__(self):
        self.volume = None
        self.spacing = None
        self.origin = None
        self.orientation = None
        
    def load_directory(self, directory_path):
        """Load DICOM files from a directory and return numpy array
        
        Args:
            directory_path: Path to directory containing DICOM files
            
        Returns:
            dict with 'volume', 'spacing', 'origin', and 'orientation'
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise ValueError(f"Directory not found: {directory_path}")
            
        # Find all DICOM files
        dicom_files = []
        for file_path in directory_path.glob("*.dcm"):
            try:
                ds = pydicom.dcmread(str(file_path))
                if hasattr(ds, 'pixel_array'):
                    dicom_files.append((file_path, ds))
            except Exception as e:
                print(f"Warning: Could not read {file_path.name}: {e}")
                continue
                    
        if not dicom_files:
            raise ValueError("No valid DICOM files found")
            
        print(f"Found {len(dicom_files)} DICOM files")
            
        # Sort by instance number or slice location
        def get_sort_key(item):
            _, ds = item
            if hasattr(ds, 'InstanceNumber'):
                return float(ds.InstanceNumber)
            elif hasattr(ds, 'SliceLocation'):
                return float(ds.SliceLocation)
            else:
                return 0
                
        dicom_files.sort(key=get_sort_key)
        
        # Get metadata from first file
        first_ds = dicom_files[0][1]
        rows = int(first_ds.Rows)
        cols = int(first_ds.Columns)
        num_slices = len(dicom_files)
        
        # Get spacing
        pixel_spacing = [float(x) for x in first_ds.PixelSpacing]
        slice_thickness = float(first_ds.get('SliceThickness', 1.0))
        self.spacing = np.array([pixel_spacing[0], pixel_spacing[1], slice_thickness])
        
        # Get origin (if available)
        if hasattr(first_ds, 'ImagePositionPatient'):
            self.origin = np.array([float(x) for x in first_ds.ImagePositionPatient])
        else:
            self.origin = np.array([0.0, 0.0, 0.0])
        
        # Get orientation (if available)
        if hasattr(first_ds, 'ImageOrientationPatient'):
            iop = [float(x) for x in first_ds.ImageOrientationPatient]
            # ImageOrientationPatient gives us the first two columns of the rotation matrix
            row_cosines = np.array(iop[:3])
            col_cosines = np.array(iop[3:])
            # The third column is the cross product (slice direction)
            slice_cosines = np.cross(row_cosines, col_cosines)
            # Create the orientation matrix
            self.orientation = np.column_stack([row_cosines, col_cosines, slice_cosines])
        else:
            # Default to identity if not available
            self.orientation = np.eye(3)
        
        print(f"Loading volume: {rows}x{cols}x{num_slices}")
        print(f"Spacing: {self.spacing} mm")
        print(f"Origin: {self.origin} mm")
        print(f"Orientation matrix:\n{self.orientation}")
        
        # Create volume array
        self.volume = np.zeros((rows, cols, num_slices), dtype=np.float32)
        
        # Load all slices
        for i, (path, ds) in enumerate(dicom_files):
            # Get pixel array
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # Convert to Hounsfield Units
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slope = float(ds.RescaleSlope)
                intercept = float(ds.RescaleIntercept)
                pixel_array = pixel_array * slope + intercept
            
            self.volume[:, :, i] = pixel_array
            
            if i % 50 == 0:
                print(f"  Loaded {i+1}/{num_slices} slices...")
        
        print(f"Volume loaded successfully")
        print(f"HU range: [{self.volume.min():.0f}, {self.volume.max():.0f}]")
        
        return {
            'volume': self.volume,
            'spacing': self.spacing,
            'origin': self.origin,
            'orientation': self.orientation
        }


class SimpleMaterialConverter:
    """Convert HU to material properties - only skull and water"""
    
    def __init__(self):
        # Skull properties
        self.skull_hu_min = 300
        self.skull_hu_max = 2000
        self.skull_density_min = 1000  # kg/m³
        self.skull_density_max = 1900  # kg/m³
        self.skull_sound_speed_min = 1500  # m/s
        self.skull_sound_speed_max = 3100  # m/s
        self.skull_absorption_min = 4
        self.skull_absorption_max = 8.371
        
        # Water properties
        self.water_density = 1000  # kg/m³
        self.water_sound_speed = 1500  # m/s
        self.water_absorption = 0.001
        
    def create_skull_mask(self, ct_array):
        """Create skull mask from CT array"""
        return ct_array > self.skull_hu_min
    
    def convert_to_properties(self, ct_array):
        """Convert CT array to material properties"""
        # Create skull mask
        skull_mask = self.create_skull_mask(ct_array)
        
        # Initialize arrays with water properties
        shape = ct_array.shape
        density = np.full(shape, self.water_density, dtype=np.float32)
        sound_speed = np.full(shape, self.water_sound_speed, dtype=np.float32)
        absorption = np.full(shape, self.water_absorption, dtype=np.float32)
        
        # Apply skull properties where skull exists
        if skull_mask.any():
            # Get HU values for skull voxels
            skull_hu = ct_array[skull_mask]
            skull_hu_clipped = np.clip(skull_hu, self.skull_hu_min, self.skull_hu_max)
            
            # Linear mapping for density
            density[skull_mask] = self.skull_density_min + \
                                 (self.skull_density_max - self.skull_density_min) * \
                                 (skull_hu_clipped - self.skull_hu_min) / \
                                 (self.skull_hu_max - self.skull_hu_min)
            
            # Linear mapping for sound speed based on density
            skull_density = density[skull_mask]
            sound_speed[skull_mask] = self.skull_sound_speed_min + \
                                     (self.skull_sound_speed_max - self.skull_sound_speed_min) * \
                                     (skull_density - self.skull_density_min) / \
                                     (self.skull_density_max - self.skull_density_min)
            
            # Nonlinear mapping for absorption
            normalized_hu = (skull_hu_clipped - self.skull_hu_min) / \
                          (self.skull_hu_max - self.skull_hu_min)
            absorption[skull_mask] = self.skull_absorption_min + \
                                   (self.skull_absorption_max - self.skull_absorption_min) * \
                                   np.sqrt(1 - normalized_hu)
        
        return {
            'density': density,
            'sound_speed': sound_speed,
            'absorption': absorption,
            'skull_mask': skull_mask
        }


def convert_ras_mm_to_voxel(position_ras_mm, spacing_mm):
    """Convert RAS coordinates in mm to voxel indices
    
    This assumes the RAS coordinates are in the image coordinate system
    (not patient coordinate system) and just need scaling by voxel spacing.
    
    Args:
        position_ras_mm: (x,y,z) coordinates in RAS mm  
        spacing_mm: voxel spacing in mm
        
    Returns:
        (i,j,k) voxel indices
    """
    # Simple conversion: divide mm coordinates by voxel spacing
    position_voxel = position_ras_mm / spacing_mm
    return position_voxel


def convert_ras_normal_to_voxel(normal_ras):
    """Convert RAS normal vector to voxel space
    
    For RAS coordinate system aligned with image axes,
    the normal vector just needs normalization.
    
    Args:
        normal_ras: normal vector in RAS space
        
    Returns:
        normal vector in voxel space  
    """
    # Ensure unit vector
    normal = normal_ras / np.linalg.norm(normal_ras)
    return normal


def normal_vector_to_euler_angles(normal):
    """
    Convert a unit normal vector to Euler angles (in degrees).
    
    The normal vector represents the z-axis of the element's local coordinate system.
    We need to find the rotation angles that transform from global to local coordinates.
    
    Args:
        normal: Unit normal vector [nx, ny, nz]
        
    Returns:
        list: [rx, ry, rz] rotation angles in degrees for x-y'-z'' intrinsic rotations
    """
    # The element's z-axis should align with the normal vector
    # We need to construct a rotation matrix where the third column is the normal
    
    # Choose arbitrary x and y axes perpendicular to normal
    if abs(normal[2]) < 0.9:
        # Use world z-axis to generate perpendicular vector
        x_local = np.cross(normal, [0, 0, 1])
    else:
        # Use world x-axis if normal is nearly parallel to z
        x_local = np.cross(normal, [1, 0, 0])
    
    x_local = x_local / np.linalg.norm(x_local)
    y_local = np.cross(normal, x_local)
    y_local = y_local / np.linalg.norm(y_local)
    
    # Construct rotation matrix where columns are the local axes
    # R = [x_local, y_local, normal]
    R = np.column_stack([x_local, y_local, normal])
    
    # Extract Euler angles from rotation matrix
    # Using x-y'-z'' intrinsic rotation convention (as specified in k-Wave docs)
    # This is equivalent to ZYX extrinsic rotations
    
    # Check for gimbal lock
    if abs(R[0, 2]) >= 0.99:
        # Gimbal lock case
        if R[0, 2] > 0:
            ry = -np.pi/2
            rx = 0
            rz = np.arctan2(-R[1, 0], R[1, 1])
        else:
            ry = np.pi/2
            rx = 0
            rz = np.arctan2(R[1, 0], R[1, 1])
    else:
        # General case
        ry = -np.arcsin(R[0, 2])
        rx = np.arctan2(R[1, 2] / np.cos(ry), R[2, 2] / np.cos(ry))
        rz = np.arctan2(R[0, 1] / np.cos(ry), R[0, 0] / np.cos(ry))
    
    # Convert to degrees
    rx_deg = np.degrees(rx)
    ry_deg = np.degrees(ry)
    rz_deg = np.degrees(rz)
    
    return [rx_deg, ry_deg, rz_deg]


def resample_to_isotropic(ct_array, spacing_mm, target_spacing_mm=None):
    """
    Resample CT volume to isotropic spacing using scipy zoom.
    
    Args:
        ct_array: 3D CT volume
        spacing_mm: Current voxel spacing in mm [dx, dy, dz]
        target_spacing_mm: Target isotropic spacing in mm (if None, uses smallest spacing)
        
    Returns:
        tuple: (resampled_volume, new_spacing)
    """
    # If no target specified, use the smallest spacing (finest resolution)
    if target_spacing_mm is None:
        target_spacing_mm = float(np.min(spacing_mm))
    
    # Calculate zoom factors for each dimension
    zoom_factors = spacing_mm / target_spacing_mm
    
    print(f"\nResampling CT to isotropic spacing:")
    print(f"  Original spacing: {spacing_mm} mm")
    print(f"  Target spacing: {target_spacing_mm} mm")
    print(f"  Zoom factors: {zoom_factors}")
    print(f"  Original shape: {ct_array.shape}")
    
    # Resample using linear interpolation (order=1)
    # Use order=3 for higher quality but slower
    resampled = zoom(ct_array, zoom_factors, order=1, mode='constant', cval=ct_array.min())
    
    print(f"  Resampled shape: {resampled.shape}")
    print(f"  Size change: {ct_array.nbytes / 1e6:.1f} MB -> {resampled.nbytes / 1e6:.1f} MB")
    
    # New spacing is isotropic
    new_spacing = np.array([target_spacing_mm, target_spacing_mm, target_spacing_mm])
    
    return resampled, new_spacing


def extract_simulation_grid(ct_array, center_voxel, grid_size_mm, voxel_spacing_mm):
    """Extract isotropic simulation grid"""
    # Calculate grid size in voxels
    grid_voxels = int(np.round(grid_size_mm / voxel_spacing_mm))
    if grid_voxels % 2 == 1:
        grid_voxels += 1
        
    half_size = grid_voxels // 2
    
    # Define shifts from center voxel
    x_shift = 0
    y_shift = 0
    z_shift = 0
    shifts = np.array([x_shift, y_shift, z_shift])
    
    # Calculate bounds with shifts
    grid_origin = (center_voxel - half_size + shifts).astype(int)
    grid_end = grid_origin + grid_voxels
    
    # Create output grid filled with minimum HU (water)
    fill_value = ct_array.min()
    grid = np.full((grid_voxels, grid_voxels, grid_voxels), fill_value)
    
    # Calculate valid region to copy
    src_start = np.maximum(0, grid_origin)
    src_end = np.minimum(ct_array.shape, grid_end)
    
    dst_start = np.maximum(0, -grid_origin)
    dst_end = dst_start + (src_end - src_start)
    
    # Copy data
    grid[dst_start[0]:dst_end[0],
         dst_start[1]:dst_end[1],
         dst_start[2]:dst_end[2]] = \
        ct_array[src_start[0]:src_end[0],
                src_start[1]:src_end[1],
                src_start[2]:src_end[2]]
    
    return grid, grid_origin


def save_grid_cross_sections(grid, skull_mask, source_mask, output_dir, spacing):
    """Save cross-section images of the grid to verify coordinate transformations"""
    print("\nSaving grid cross-sections for visualization...")
    
    # Get grid dimensions
    nx, ny, nz = grid.shape
    
    # Create combined visualization arrays
    # 0: water/tissue, 1: skull, 2: source
    combined = np.zeros_like(grid)
    combined[skull_mask] = 1
    combined[source_mask] = 2
    
    # Define colormap
    colors = ['blue', 'white', 'red']  # water, skull, source
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Create figure with subplots for different cross-sections
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # XY plane (axial) - center and near transducers
    z_center = nz // 2
    z_transducer = np.where(source_mask.any(axis=(0, 1)))[0]
    if len(z_transducer) > 0:
        z_transducer = z_transducer[len(z_transducer)//2]
    else:
        z_transducer = z_center
    
    im0 = axes[0].imshow(combined[:, :, z_center].T, cmap=cmap, norm=norm, origin='lower')
    axes[0].set_title(f'XY plane (z={z_center})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    im1 = axes[1].imshow(combined[:, :, z_transducer].T, cmap=cmap, norm=norm, origin='lower')
    axes[1].set_title(f'XY plane at transducer (z={z_transducer})')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    
    # XZ plane (sagittal)
    y_center = ny // 2
    y_transducer = np.where(source_mask.any(axis=(0, 2)))[0]
    if len(y_transducer) > 0:
        y_transducer = y_transducer[len(y_transducer)//2]
    else:
        y_transducer = y_center
        
    im2 = axes[2].imshow(combined[:, y_center, :].T, cmap=cmap, norm=norm, origin='lower')
    axes[2].set_title(f'XZ plane (y={y_center})')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Z')
    
    im3 = axes[3].imshow(combined[:, y_transducer, :].T, cmap=cmap, norm=norm, origin='lower')
    axes[3].set_title(f'XZ plane at transducer (y={y_transducer})')
    axes[3].set_xlabel('X')
    axes[3].set_ylabel('Z')
    
    # YZ plane (coronal)
    x_center = nx // 2
    x_transducer = np.where(source_mask.any(axis=(1, 2)))[0]
    if len(x_transducer) > 0:
        x_transducer = x_transducer[len(x_transducer)//2]
    else:
        x_transducer = x_center
        
    im4 = axes[4].imshow(combined[x_center, :, :].T, cmap=cmap, norm=norm, origin='lower')
    axes[4].set_title(f'YZ plane (x={x_center})')
    axes[4].set_xlabel('Y')
    axes[4].set_ylabel('Z')
    
    im5 = axes[5].imshow(combined[x_transducer, :, :].T, cmap=cmap, norm=norm, origin='lower')
    axes[5].set_title(f'YZ plane at transducer (x={x_transducer})')
    axes[5].set_xlabel('Y')
    axes[5].set_ylabel('Z')
    
    # Add colorbar
    cbar = fig.colorbar(im0, ax=axes, ticks=[0, 1, 2], 
                       orientation='horizontal', pad=0.1, fraction=0.05)
    cbar.ax.set_xticklabels(['Water/Tissue', 'Skull', 'Source'])
    
    plt.suptitle('Grid Cross-Sections Showing Skull and Transducers')
    plt.tight_layout()
    
    # Save individual cross-sections
    for idx, (ax, name) in enumerate(zip(axes, 
                                        ['grid_xy_center', 'grid_xy_transducer',
                                         'grid_xz_center', 'grid_xz_transducer', 
                                         'grid_yz_center', 'grid_yz_transducer'])):
        extent = ax.get_images()[0].get_extent()
        fig_single = plt.figure(figsize=(6, 6))
        ax_single = fig_single.add_subplot(111)
        ax_single.imshow(ax.get_images()[0].get_array(), cmap=cmap, norm=norm, origin='lower')
        ax_single.set_title(ax.get_title())
        ax_single.set_xlabel(ax.get_xlabel())
        ax_single.set_ylabel(ax.get_ylabel())
        fig_single.savefig(os.path.join(output_dir, f'{name}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig_single)
    
    # Save combined figure
    fig.savefig(os.path.join(output_dir, 'grid_cross_sections.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved cross-section images to {output_dir}/")


def save_pressure_movie_data(sensor_data, source_mask, skull_mask, grid_shape, output_dir, config):
    """Save full pressure field data for movie visualization"""
    if sensor_data is None or 'p' not in sensor_data:
        print("Warning: No pressure data available for movie export")
        return
    
    p_data = sensor_data['p']
    print(f"Saving pressure movie data with shape: {p_data.shape}")
    
    # Reshape from (time_steps, total_voxels) to (time_steps, nx, ny, nz)
    time_steps, total_voxels = p_data.shape
    nx, ny, nz = grid_shape
    
    if total_voxels != nx * ny * nz:
        print(f"Warning: Voxel count mismatch. Expected {nx*ny*nz}, got {total_voxels}")
        return
    
    # Reshape using Fortran order to match k-Wave indexing
    pressure_4d = np.zeros((time_steps, nx, ny, nz), dtype=np.float32)
    for t in range(time_steps):
        pressure_4d[t] = p_data[t].reshape((nx, ny, nz), order='F')
    
    # Save as HDF5 for efficient storage and access
    movie_file = os.path.join(output_dir, "pressure_movie.h5")
    with h5py.File(movie_file, 'w') as f:
        # Save pressure data
        f.create_dataset('pressure', data=pressure_4d, compression='gzip', compression_opts=6)
        
        # Save source mask if available
        f.create_dataset('source_mask', data=source_mask)
        f.create_dataset('skull_mask', data=skull_mask)
        
        # Save metadata
        f.attrs['time_steps'] = time_steps
        f.attrs['grid_shape'] = grid_shape
        f.attrs['dt'] = config['timing']['dt']
        f.attrs['dx'] = config['grid']['dx']
        f.attrs['total_time_us'] = config['timing']['total_time_us']
        
        # Save time axis
        time_axis = np.arange(time_steps) * config['timing']['dt'] * 1e6  # Convert to microseconds
        f.create_dataset('time_us', data=time_axis)
        
        print(f"✓ Saved pressure movie data to {movie_file}")
        print(f"  Shape: {pressure_4d.shape}")
        print(f"  Size: {pressure_4d.nbytes / 1e9:.2f} GB")
        print(f"  Time range: 0 to {time_axis[-1]:.1f} µs")
    
    # Create a simple Python script for visualization
    create_movie_visualization_script(output_dir, movie_file, config)


def create_movie_visualization_script(output_dir, movie_file, config):
    """Create a Python script for visualizing the pressure movie data"""
    script_content = f'''#!/usr/bin/env python3
"""
Pressure wave movie visualization script
Auto-generated for visualizing k-Wave simulation results
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import os

def load_pressure_data(filename):
    """Load pressure movie data from HDF5 file"""
    with h5py.File(filename, 'r') as f:
        pressure = f['pressure'][:]
        time_us = f['time_us'][:]
        source_mask = f['source_mask'][:] if 'source_mask' in f else None
        metadata = dict(f.attrs)
    return pressure, time_us, source_mask, metadata

def create_movie(pressure_data, time_us, source_mask=None, output_prefix="pressure_movie", 
                plane='xy', slice_idx=None, fps=10, vmin=None, vmax=None):
    """Create movie animation of pressure wave propagation"""
    
    time_steps, nx, ny, nz = pressure_data.shape
    
    # Determine slice index if not provided
    if slice_idx is None:
        if plane == 'xy':
            z_center = nz // 2
            if source_mask is not None:
                z_transducer = np.where(source_mask.any(axis=(0, 1)))[0]
                if len(z_transducer) > 0:
                    slice_idx = z_transducer[len(z_transducer)//2]
                else:
                    slice_idx = z_center
            else:
                slice_idx = z_center
                
        elif plane == 'xz':
            y_center = ny // 2
            if source_mask is not None:
                y_transducer = np.where(source_mask.any(axis=(0, 2)))[0]
                if len(y_transducer) > 0:
                    slice_idx = y_transducer[len(y_transducer)//2]
                else:
                    slice_idx = y_center
            else:
                slice_idx = y_center
                
        elif plane == 'yz':
            x_center = nx // 2
            if source_mask is not None:
                x_transducer = np.where(source_mask.any(axis=(1, 2)))[0]
                if len(x_transducer) > 0:
                    slice_idx = x_transducer[len(x_transducer)//2]
                else:
                    slice_idx = x_center
            else:
                slice_idx = x_center
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data based on plane
    if plane == 'xy':
        data_slice = pressure_data[:, :, :, slice_idx]
        title_base = f'Pressure Wave - XY Plane (z={{slice_idx}})'
        xlabel, ylabel = 'X', 'Y'
    elif plane == 'xz':
        data_slice = pressure_data[:, :, slice_idx, :]
        title_base = f'Pressure Wave - XZ Plane (y={{slice_idx}})'
        xlabel, ylabel = 'X', 'Z'
    elif plane == 'yz':
        data_slice = pressure_data[:, slice_idx, :, :]
        title_base = f'Pressure Wave - YZ Plane (x={{slice_idx}})'
        xlabel, ylabel = 'Y', 'Z'
    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")
    
    # Set color scale if not provided
    if vmin is None:
        vmin = np.percentile(data_slice, 1)
    if vmax is None:
        vmax = np.percentile(data_slice, 99)
    
    # Create initial plot
    im = ax.imshow(data_slice[0].T, cmap='RdBu_r', vmin=vmin, vmax=vmax, 
                   origin='lower', animated=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = ax.set_title(title_base.format(slice_idx=slice_idx))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pressure (Pa)')
    
    # Animation function
    def animate(frame):
        im.set_array(data_slice[frame].T)
        time_str = f'Time: {{time_us[frame]:.1f}} µs'
        title.set_text(f'{{title_base.format(slice_idx=slice_idx)}} - {{time_str}}')
        return [im, title]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=time_steps, 
                                 interval=1000//fps, blit=True, repeat=True)
    
    # Save as GIF and MP4
    gif_file = f'{{output_prefix}}_{{plane}}_slice{{slice_idx}}.gif'
    mp4_file = f'{{output_prefix}}_{{plane}}_slice{{slice_idx}}.mp4'
    
    print(f"Saving animation as {{gif_file}}...")
    anim.save(gif_file, writer='pillow', fps=fps)
    
    try:
        print(f"Saving animation as {{mp4_file}}...")
        anim.save(mp4_file, writer='ffmpeg', fps=fps, bitrate=1800)
    except Exception as e:
        print(f"Could not save MP4 (ffmpeg not available?): {{e}}")
    
    plt.show()
    return anim

def main():
    """Main visualization function"""
    movie_file = "{os.path.basename(movie_file)}"
    
    if not os.path.exists(movie_file):
        print(f"Error: Movie file not found: {{movie_file}}")
        return
    
    print("Loading pressure movie data...")
    pressure, time_us, source_mask, metadata = load_pressure_data(movie_file)
    
    print(f"Loaded data:")
    print(f"  Shape: {{pressure.shape}}")
    print(f"  Time range: {{time_us[0]:.1f}} - {{time_us[-1]:.1f}} µs")
    print(f"  Grid spacing: {{metadata['dx']*1000:.3f}} mm")
    
    # Create movies for different planes
    print("\\nCreating XY plane movie (axial view)...")
    create_movie(pressure, time_us, source_mask, plane='xy', fps=10)
    
    print("\\nCreating XZ plane movie (sagittal view)...")  
    create_movie(pressure, time_us, source_mask, plane='xz', fps=10)
    
    print("\\nCreating YZ plane movie (coronal view)...")
    create_movie(pressure, time_us, source_mask, plane='yz', fps=10)
    
    print("\\n✓ Movie visualization complete!")
    print("Generated files:")
    print("  - pressure_movie_xy_slice*.gif/mp4")
    print("  - pressure_movie_xz_slice*.gif/mp4") 
    print("  - pressure_movie_yz_slice*.gif/mp4")

if __name__ == "__main__":
    main()
'''
    script_file = os.path.join(output_dir, "visualize_movie.py")
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_file, 0o755)  # Make executable
    print(f"✓ Created movie visualization script: {script_file}")
    print("  Run with: python visualize_movie.py")


def create_water_grid(grid_dims_voxels=None, spacing_mm=None, grid_size_mm=None):
    """Create a water-only grid for free-field simulation
    
    Args:
        grid_dims_voxels: Tuple of (nx, ny, nz) grid dimensions in voxels
        spacing_mm: Voxel spacing in mm (can be single value or tuple)
        grid_size_mm: Grid size in mm (deprecated, use grid_dims_voxels)
        
    Returns:
        dict with material properties
    """
    # Handle backward compatibility
    if grid_dims_voxels is None and grid_size_mm is not None:
        # Old behavior - cubic grid
        grid_voxels = int(np.round(grid_size_mm / spacing_mm))
        if grid_voxels % 2 == 1:
            grid_voxels += 1
        shape = (grid_voxels, grid_voxels, grid_voxels)
        actual_size_mm = np.array(shape) * spacing_mm
    else:
        # New behavior - custom dimensions
        shape = tuple(grid_dims_voxels)
        if isinstance(spacing_mm, (list, tuple, np.ndarray)):
            actual_size_mm = np.array(shape) * np.array(spacing_mm)
        else:
            actual_size_mm = np.array(shape) * spacing_mm
    
    # Water properties
    water_density = 1000  # kg/m³
    water_sound_speed = 1500  # m/s
    water_absorption = 0.001
    
    properties = {
        'density': np.full(shape, water_density, dtype=np.float32),
        'sound_speed': np.full(shape, water_sound_speed, dtype=np.float32),
        'absorption': np.full(shape, water_absorption, dtype=np.float32),
        'skull_mask': np.zeros(shape, dtype=bool),  # No skull in water
        'grid_shape': shape
    }
    
    print(f"Created water grid: {shape}")
    print(f"  Grid dimensions: {shape[0]} x {shape[1]} x {shape[2]} voxels")
    print(f"  Voxel spacing: {spacing_mm} mm")
    print(f"  Actual size: {actual_size_mm} mm")
    print(f"  Density: {water_density} kg/m³")
    print(f"  Sound speed: {water_sound_speed} m/s")
    print(f"  Absorption: {water_absorption}")
    
    return properties 