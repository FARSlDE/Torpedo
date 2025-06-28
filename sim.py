#!/usr/bin/env python3
"""
Simplified k-Wave simulation for ultrasound focusing
Uses kWaveArray for staircasing-free source definition
Records full pressure field for movie visualization
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
import os
import argparse
import glob

# k-Wave imports
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DG
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.checks import check_stability
from kwave.utils.signals import tone_burst
from kwave.data import Vector
from kwave.utils.kwave_array import kWaveArray

# Import utilities
from utils import (
    SimpleDICOMLoader, SimpleMaterialConverter,
    convert_ras_mm_to_voxel, normal_vector_to_euler_angles,
    resample_to_isotropic, extract_simulation_grid,
    save_grid_cross_sections, save_pressure_movie_data,
    create_water_grid
)

# ====================== SIMULATION PARAMETERS ======================
# Acoustic parameters
PRESSURE_PA = 1e6          # Source pressure in Pascal
CENTER_FREQ_HZ = 2.5e6     # Center frequency in Hz  
NUM_CYCLES = 3             # Number of cycles per burst
SOUND_SPEED_WATER = 1500   # m/s

# Grid parameters
PPW = 3                    # Points per wavelength
CFL = 0.3                  # CFL number
GRID_SIZE_MM = 50.0        # Grid size for skull simulations
GRID_SIZE_MM_WATER = 100.0 # Grid size for water simulations

# Transducer array parameters
ELEMENT_WIDTH_MM = 1.6     # Element width in mm
ELEMENT_LENGTH_MM = 1.6    # Element length in mm
WIDTH_PITCH_MM = 0.8       # Edge-to-edge spacing along width
LENGTH_PITCH_MM = 0.8      # Edge-to-edge spacing along length
ARRAY_WIDTH_N = 4          # Number of elements along width
ARRAY_LENGTH_N = 4         # Number of elements along length

# Focus parameters
DEFAULT_FOCAL_LENGTH_MM = 50.0  # Default focal length
DEFAULT_FWHM_LATERAL_MM = 2.0   # Lateral resolution
DEFAULT_FWHM_AXIAL_MM = 5.0     # Axial resolution

# kWaveArray parameters
BLI_TOLERANCE = 0.05       # Band-limited interpolant tolerance
BLI_TYPE = 'sinc'          # Interpolation type
UPSAMPLING_RATE = 10       # Higher = smoother representation


def create_transducer_array(karray, center_pos_m, normal_vector, 
                          element_width, element_length,
                          width_pitch, length_pitch,
                          width_n, length_n):
    """
    Create linear transducer array with rectangular elements
    
    Args:
        karray: kWaveArray object
        center_pos_m: Center position of array in meters
        normal_vector: Normal vector (pointing direction)
        element_width: Element width in meters
        element_length: Element length in meters
        width_pitch: Edge-to-edge spacing along width in meters
        length_pitch: Edge-to-edge spacing along length in meters
        width_n: Number of elements along width
        length_n: Number of elements along length
        
    Returns:
        element_positions: List of element center positions in meters
    """
    # Ensure normal vector is a unit vector
    normal = normal_vector / np.linalg.norm(normal_vector)
    
    # Create orthonormal basis for the array plane
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, [0, 0, 1])
    else:
        u = np.cross(normal, [1, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    
    element_positions = []
    
    # Calculate center-to-center spacing
    width_spacing = element_width + width_pitch
    length_spacing = element_length + length_pitch
    
    # Calculate total array dimensions for centering
    total_width = element_width + (width_n - 1) * width_spacing
    total_length = element_length + (length_n - 1) * length_spacing
    
    print(f"\nCreating {width_n}x{length_n} transducer array:")
    print(f"  Element size: {element_width*1e3:.1f}x{element_length*1e3:.1f} mm")
    print(f"  Edge-to-edge pitch: {width_pitch*1e3:.1f}x{length_pitch*1e3:.1f} mm")
    print(f"  Center-to-center: {width_spacing*1e3:.1f}x{length_spacing*1e3:.1f} mm")
    print(f"  Total aperture: {total_width*1e3:.1f}x{total_length*1e3:.1f} mm")
    
    # Add rectangular elements
    for i in range(width_n):
        for j in range(length_n):
            # Calculate element position relative to array center
            width_offset = (i - (width_n - 1) / 2.0) * width_spacing
            length_offset = (j - (length_n - 1) / 2.0) * length_spacing
            
            # Calculate absolute position in 3D space
            element_center = center_pos_m + width_offset * u + length_offset * v
            element_positions.append(element_center)
            
            # Convert normal to Euler angles
            theta_degrees = normal_vector_to_euler_angles(normal)
            
            # Add element to kWaveArray
            karray.add_rect_element(
                position=element_center.tolist(),
                Lx=element_width,
                Ly=element_length,
                theta=theta_degrees
            )
    
    return element_positions


def compute_element_delays(element_positions, focus_point, sound_speed=SOUND_SPEED_WATER):
    """
    Compute per-element transmit delays for focusing
    
    Args:
        element_positions: List of element positions in meters
        focus_point: Focus point position in meters
        sound_speed: Sound speed in m/s
        
    Returns:
        delays: Array of delays in seconds
    """
    num_elements = len(element_positions)
    distances = np.zeros(num_elements)
    
    # Calculate distance from each element to focus point
    for i, pos in enumerate(element_positions):
        distances[i] = np.linalg.norm(focus_point - pos)
    
    # Calculate delays (negative because we want earlier elements to fire later)
    max_distance = np.max(distances)
    delays = (max_distance - distances) / sound_speed
    
    print(f"\nComputed focusing delays:")
    print(f"  Focus point: {focus_point*1e3} mm")
    print(f"  Distance range: {distances.min()*1e3:.1f} - {distances.max()*1e3:.1f} mm")
    print(f"  Delay range: {delays.min()*1e6:.1f} - {delays.max()*1e6:.1f} µs")
    
    return delays


def create_source_signals(num_elements, delays, dt, total_time, 
                        freq=CENTER_FREQ_HZ, pressure=PRESSURE_PA, ncycles=NUM_CYCLES):
    """
    Create source signals with per-element delays
    
    Args:
        num_elements: Number of transducer elements
        delays: Per-element delays in seconds
        dt: Time step size
        total_time: Total simulation time
        freq: Transmit frequency
        pressure: Peak pressure
        ncycles: Number of cycles
        
    Returns:
        source_signals: Array of source signals (num_elements x time_steps)
    """
    Fs = 1/dt
    time_steps = int(total_time / dt)
    
    # Generate base burst
    burst = tone_burst(Fs, freq, ncycles, 'Rectangular', False)
    burst_length = burst.shape[1]
    
    # Create source signals array
    source_signals = np.zeros((num_elements, time_steps), dtype=np.float32)
    
    for i in range(num_elements):
        # Convert delay to samples
        delay_samples = int(delays[i] * Fs)
        
        # Place burst at appropriate time
        if delay_samples + burst_length <= time_steps:
            source_signals[i, delay_samples:delay_samples+burst_length] = burst.squeeze()
        else:
            # Truncate if necessary
            valid_length = time_steps - delay_samples
            if valid_length > 0:
                source_signals[i, delay_samples:] = burst.squeeze()[:valid_length]
        
        # Scale to desired pressure
        if source_signals[i].max() > 0:
            source_signals[i] *= pressure / source_signals[i].max()
    
    print(f"\nCreated source signals:")
    print(f"  Shape: {source_signals.shape}")
    print(f"  Burst duration: {burst_length/Fs*1e6:.1f} µs")
    
    return source_signals


def main():
    """Main simulation function"""
    parser = argparse.ArgumentParser(description='Simplified k-Wave Ultrasound Focusing Simulation')
    parser.add_argument('-view', action='store_true',
                       help='Export results for 3D Slicer viewing')
    parser.add_argument('-wn', type=int, default=ARRAY_WIDTH_N,
                       help=f'Array width elements (default: {ARRAY_WIDTH_N})')
    parser.add_argument('-ln', type=int, default=ARRAY_LENGTH_N,
                       help=f'Array length elements (default: {ARRAY_LENGTH_N})')
    parser.add_argument('-focal', type=float, default=DEFAULT_FOCAL_LENGTH_MM,
                       help=f'Focal length in mm (default: {DEFAULT_FOCAL_LENGTH_MM})')
    
    args = parser.parse_args()
    
    print("k-Wave Ultrasound Focusing Simulation")
    print("=" * 60)
    print(f"Array configuration: {args.wn}x{args.ln} elements")
    print(f"Focal length: {args.focal} mm")
    print("Recording full pressure field for movie visualization")
    
    # Try to load transducer position from JSON
    try:
        json_files = glob.glob('data/raw/labels_*.json')
        if not json_files:
            raise FileNotFoundError("No labels JSON files found")
            
        latest_json = max(json_files, key=os.path.getctime)
        with open(latest_json) as f:
            labels = json.load(f)
        
        # Use first transducer position
        transducer_idx = 0
        selected_transducer = labels['transducer_positions'][transducer_idx]
        
        position_ras_mm = np.array([
            selected_transducer['position']['x'],
            selected_transducer['position']['y'],
            selected_transducer['position']['z']
        ])
        normal_ras = np.array([
            selected_transducer['normal']['x'],
            selected_transducer['normal']['y'],
            selected_transducer['normal']['z']
        ])
        
        print(f"\nLoaded transducer position from {latest_json}")
        print(f"Position (RAS): {position_ras_mm} mm")
        print(f"Normal (RAS): {normal_ras}")
        
        # Load CT data
        ct_dir = Path("data/CTs/CT 0.625mm")
        if not ct_dir.exists():
            raise FileNotFoundError(f"CT directory not found: {ct_dir}")
            
        print(f"\nLoading CT data from: {ct_dir}")
        loader = SimpleDICOMLoader()
        ct_data = loader.load_directory(ct_dir)
        
        ct_array = ct_data['volume']
        spacing = ct_data['spacing']
        
        # Resample to isotropic if needed
        if not np.allclose(spacing[0], spacing):
            print("\nResampling to isotropic spacing...")
            ct_array, spacing = resample_to_isotropic(ct_array, spacing)
        
        # Convert position to voxel space
        position_voxel = convert_ras_mm_to_voxel(position_ras_mm, spacing)
        normal = normal_ras / np.linalg.norm(normal_ras)
        
        # Extract simulation grid
        grid, grid_origin_voxel = extract_simulation_grid(
            ct_array, position_voxel, GRID_SIZE_MM, spacing[0]
        )
        
        # Convert to material properties
        converter = SimpleMaterialConverter()
        properties = converter.convert_to_properties(grid)
        
        # Setup spacing
        dx = spacing[0] / 1000  # Convert mm to meters
        grid_size_mm = GRID_SIZE_MM
        
        print(f"\nUsing skull simulation mode")
        print(f"Grid size: {grid_size_mm} mm")
        
    except (FileNotFoundError, KeyError, IndexError) as e:
        print(f"\nFallback to free-field water simulation: {e}")
        
        # Create water-only grid
        grid_size_mm = GRID_SIZE_MM_WATER
        dx = 0.5e-3  # 0.5mm spacing
        spacing_mm = dx * 1000
        
        properties = create_water_grid(grid_size_mm, spacing_mm)
        grid = np.zeros(properties['grid_shape'])  # Dummy grid for cross-sections
        
        # Place transducer 10mm from top, centered in x-y, pointing down
        grid_center_m = np.array([0, 0, 0])  # k-Wave uses centered coordinates
        array_offset_m = (grid_size_mm/2 - 10) * 1e-3  # 10mm from top edge
        position_ras_mm = np.array([grid_size_mm/2, grid_size_mm/2, 10])
        normal_ras = np.array([0, 0, 1])  # Pointing down
        normal = normal_ras
        
        print(f"\nUsing water simulation mode")
        print(f"Grid size: {grid_size_mm} mm")
        print(f"Transducer at: {position_ras_mm} mm")
        print(f"Pointing: {normal}")
    
    # Setup k-Wave grid
    print("\nSetting up k-Wave simulation...")
    kgrid = kWaveGrid(
        Vector(properties['density'].shape),
        Vector([dx, dx, dx])
    )
    
    # Setup medium
    medium = kWaveMedium(
        sound_speed=properties['sound_speed'],
        density=properties['density'],
        alpha_coeff=properties['absorption'],
        alpha_power=1.1
    )
    
    # Calculate time stepping
    dt_stability_limit = check_stability(kgrid, medium)
    ppp = round(PPW / CFL)
    dt = 1 / (ppp * CENTER_FREQ_HZ)
    
    if dt > dt_stability_limit:
        print(f'Adjusting dt from {dt} to stability limit: {dt_stability_limit}')
        dt = dt_stability_limit
    
    # Calculate total simulation time
    # Need enough time for waves to propagate and reflect
    total_distance = grid_size_mm * 2e-3  # Convert to meters, x2 for round trip
    avg_sound_speed = np.mean(properties['sound_speed'])
    total_time = total_distance / avg_sound_speed * 1.5  # 1.5x safety factor
    
    time_steps = int(total_time / dt)
    kgrid.setTime(time_steps, dt)
    
    print(f"Time steps: {time_steps}")
    print(f"Time step size: {dt:.2e} s")
    print(f"Total simulation time: {total_time*1e6:.1f} µs")
    
    # Initialize kWaveArray
    karray = kWaveArray(
        axisymmetric=False,
        bli_tolerance=BLI_TOLERANCE,
        bli_type=BLI_TYPE,
        single_precision=False,
        upsampling_rate=UPSAMPLING_RATE
    )
    
    # Create transducer array
    # Array center in k-Wave coordinates (grid center is at origin)
    if 'grid_origin_voxel' in locals():
        # Skull mode: offset from skull surface
        array_center_kwave = np.zeros(3) + normal * 10e-3
    else:
        # Water mode: specific position
        array_center_kwave = np.array([0, 0, -array_offset_m])
    
    element_positions = create_transducer_array(
        karray, array_center_kwave, normal,
        ELEMENT_WIDTH_MM * 1e-3, ELEMENT_LENGTH_MM * 1e-3,
        WIDTH_PITCH_MM * 1e-3, LENGTH_PITCH_MM * 1e-3,
        args.wn, args.ln
    )
    
    # Calculate focus point
    focus_offset_m = args.focal * 1e-3  # Convert mm to meters
    focus_point = array_center_kwave + normal * focus_offset_m
    
    # Compute element delays for focusing
    delays = compute_element_delays(element_positions, focus_point, avg_sound_speed)
    
    # Create source signals with delays
    num_elements = args.wn * args.ln
    source_signals = create_source_signals(
        num_elements, delays, dt, total_time,
        CENTER_FREQ_HZ, PRESSURE_PA, NUM_CYCLES
    )
    
    # Get source mask and distributed signals
    print("\nGenerating source mask from kWaveArray...")
    source_mask = karray.get_array_binary_mask(kgrid)
    source = kSource()
    source.p_mask = source_mask
    source.p = karray.get_distributed_source_signal(kgrid, source_signals)
    
    print(f"Source mask shape: {source_mask.shape}")
    print(f"Active source voxels: {source_mask.sum()}")
    
    # Setup sensor - always record entire grid for movie
    sensor = kSensor()
    sensor.mask = np.ones(properties['density'].shape, dtype=bool)
    sensor.record = ['p']
    
    # Simulation options
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), 'data/simulations', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    sim_opts = SimulationOptions(
        pml_inside=True,
        pml_size=[10],
        save_to_disk=True,
        data_recast=True,
        input_filename=os.path.join(output_dir, "input.h5"),
        output_filename=os.path.join(output_dir, "output.h5")
    )
    
    exec_opts = SimulationExecutionOptions(
        is_gpu_simulation=True,
        delete_data=False,
        verbose_level=1
    )
    
    # Save configuration
    config = {
        "simulation_type": "focused_ultrasound",
        "array_config": {
            "width_n": args.wn,
            "length_n": args.ln,
            "element_width_mm": ELEMENT_WIDTH_MM,
            "element_length_mm": ELEMENT_LENGTH_MM,
            "width_pitch_mm": WIDTH_PITCH_MM,
            "length_pitch_mm": LENGTH_PITCH_MM
        },
        "focus": {
            "focal_length_mm": args.focal,
            "focus_point_m": focus_point.tolist(),
            "delays_us": (delays * 1e6).tolist()
        },
        "acoustic": {
            "frequency_hz": CENTER_FREQ_HZ,
            "pressure_pa": PRESSURE_PA,
            "num_cycles": NUM_CYCLES
        },
        "timing": {
            "dt": float(dt),
            "time_steps": int(time_steps),
            "total_time_us": float(total_time * 1e6)
        },
        "grid": {
            "dx_mm": float(dx * 1000),
            "grid_size_mm": float(grid_size_mm),
            "grid_shape": list(properties['density'].shape)
        },
        "karray_settings": {
            "bli_tolerance": BLI_TOLERANCE,
            "bli_type": BLI_TYPE,
            "upsampling_rate": UPSAMPLING_RATE
        }
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Run simulation
    print(f"\nRunning focused ultrasound simulation...")
    print(f"Output directory: {output_dir}")
    
    sensor_data = kspaceFirstOrder3DG(
        kgrid=kgrid,
        source=source,
        sensor=sensor,
        medium=medium,
        simulation_options=sim_opts,
        execution_options=exec_opts
    )
    
    # Save pressure movie data
    if sensor_data is not None:
        save_pressure_movie_data(
            sensor_data, source_mask, properties['skull_mask'],
            properties['density'].shape, output_dir, config
        )
    
    # Save simulation data
    np.savez(
        os.path.join(output_dir, "simulation_data.npz"),
        grid_hu=grid,
        density=properties['density'],
        sound_speed=properties['sound_speed'],
        absorption=properties['absorption'],
        skull_mask=properties['skull_mask'],
        source_mask=source_mask,
        element_positions=np.array(element_positions),
        focus_point=focus_point,
        delays=delays
    )
    
    # Save cross-sections
    save_grid_cross_sections(
        grid, properties['skull_mask'], source_mask,
        output_dir, dx * 1000
    )
    
    print("\n✓ Simulation complete!")
    print(f"Total elements: {num_elements}")
    print(f"Focus at: {args.focal} mm")
    print("✓ Full pressure field recorded for movie visualization")
    print("✓ kWaveArray used - no staircasing errors")
    
    # Export to NIfTI if requested
    if args.view:
        try:
            import sys
            if os.path.exists('tools/export_to_nifti.py'):
                sys.path.append('.')
                from tools.export_to_nifti import export_skull_and_transducers
                nifti_path = export_skull_and_transducers(output_dir)
                print(f"✓ NIfTI export successful: {nifti_path}")
        except Exception as e:
            print(f"Warning: NIfTI export failed: {e}")


if __name__ == "__main__":
    main() 