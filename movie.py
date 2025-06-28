#!/usr/bin/env python3
"""
Volumetric pressure field visualization using VTK
Interactive 3D rendering with time slider control

Visualization Strategy with Skull and Source Masks:
1. Source voxels (transducers) are always opaque gray (highest priority)
2. Skull voxels are always opaque white (second priority)
3. Water voxels are transparent unless there's a pressure wave
4. Pressure waves are visible everywhere with blue-white-red coloring

Implementation:
- Add different offsets to distinguish structures:
  - Source voxels: offset = 20 × pressure_range → gray color
  - Skull voxels: offset = 10 × pressure_range → white color
  - Water voxels: original values → colored by pressure
- Opacity transfer function:
  - Source/skull offsets → fully opaque (1.0)
  - Original pressure range → transparent unless significant pressure
- Color transfer function:
  - Source offset → uniform gray
  - Skull offset → uniform white
  - Original range → blue-white-red based on pressure

The visualization shows:
- Transducer elements as gray blocks
- Skull structure as uniform white
- Pressure waves as colored (blue=negative, red=positive)
- Water is invisible unless pressure waves present
"""

import numpy as np
import h5py
import vtk
from vtk.util import numpy_support
import os
import sys


class VolumetricPressureViewer:
    """Interactive volumetric viewer for pressure field data"""
    
    def __init__(self, filename, opacity_scale=1.0):
        """Initialize the viewer with pressure data file"""
        self.filename = filename
        self.opacity_scale = opacity_scale
        self.current_timestep = 0
        self.pressure_data = None
        self.time_us = None
        self.metadata = None
        self.volume_actor = None
        self.text_actor = None
        self.slider_widget = None
        self.renderer = None
        self.render_window = None
        self.interactor = None
        
        # Load data
        self.load_data()
        
        # Setup VTK pipeline
        self.setup_vtk_pipeline()
        
    def load_data(self):
        """Load pressure data from HDF5 file"""
        print(f"Loading pressure data from {self.filename}...")
        
        with h5py.File(self.filename, 'r') as f:
            # Load pressure data - shape: (time_steps, nx, ny, nz)
            self.pressure_data = f['pressure'][:]
            self.time_us = f['time_us'][:]
            
            # Load metadata
            self.metadata = dict(f.attrs)
            
            # Also load source mask if available
            if 'source_mask' in f:
                self.source_mask = f['source_mask'][:]
                print(f"  Loaded source mask with {self.source_mask.sum()} source voxels")
            else:
                self.source_mask = None
                print("  No source mask found in file")
                
            # Load skull mask for opacity control
            if 'skull_mask' in f:
                self.skull_mask = f['skull_mask'][:]
                print(f"  Loaded skull mask with {self.skull_mask.sum()} skull voxels")
            else:
                self.skull_mask = None
                print("  Warning: No skull mask found in file")
        
        self.time_steps, self.nx, self.ny, self.nz = self.pressure_data.shape
        
        print(f"Loaded data:")
        print(f"  Shape: {self.pressure_data.shape}")
        print(f"  Time range: {self.time_us[0]:.1f} - {self.time_us[-1]:.1f} µs")
        print(f"  Grid spacing: {self.metadata.get('dx', 0.001)*1000:.3f} mm")
        
        # Calculate pressure range for color mapping
        self.pmin = np.percentile(self.pressure_data, 0.1)
        self.pmax = np.percentile(self.pressure_data, 99.9)
        print(f"  Pressure range: [{self.pmin:.0f}, {self.pmax:.0f}] Pa")
        
        # Check if pressure range spans zero
        if self.pmin < 0 and self.pmax > 0:
            print(f"  Pressure spans zero - using symmetric color mapping")
        
    def create_volume_data(self, timestep):
        """Create VTK volume data for a specific timestep"""
        # Get pressure field for this timestep
        pressure_field = self.pressure_data[timestep].copy()
        
        # Create VTK image data
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(self.nx, self.ny, self.nz)
        
        # Set spacing (convert from meters to mm for better visualization)
        dx = self.metadata.get('dx', 0.001) * 1000  # Convert to mm
        image_data.SetSpacing(dx, dx, dx)
        
        if self.skull_mask is not None:
            # Store original pressure range for color mapping
            self.original_pressure_field = pressure_field.copy()
            
            # Create a modified pressure field for combined visualization
            # Strategy: Add different offsets for different structures
            pressure_range = abs(self.pmax - self.pmin)
            skull_offset = pressure_range * 10
            source_offset = pressure_range * 20  # Higher offset for sources
            
            # Add offset to source voxels first (highest priority)
            if self.source_mask is not None:
                pressure_field[self.source_mask] = source_offset  # Uniform value for gray
            
            # Add offset to skull voxels (but not where sources are)
            if self.source_mask is not None:
                skull_only = self.skull_mask & ~self.source_mask  # Skull but not source
            else:
                skull_only = self.skull_mask
            pressure_field[skull_only] = skull_offset  # Uniform value for white
            
            # Store the offsets for the color/opacity functions
            self.skull_offset = skull_offset
            self.source_offset = source_offset
        
        # Convert to VTK array
        flat_pressure = pressure_field.flatten(order='F')
        pressure_array = numpy_support.numpy_to_vtk(flat_pressure, deep=True, array_type=vtk.VTK_FLOAT)
        pressure_array.SetName("Pressure")
        
        # Add to image data
        image_data.GetPointData().SetScalars(pressure_array)
        
        return image_data
        
    def create_color_transfer_function(self):
        """Create color transfer function for pressure visualization"""
        color_func = vtk.vtkColorTransferFunction()
        
        # Blue-white-red diverging colormap
        # Blue for negative pressure, white for zero, red for positive
        
        if self.skull_mask is not None and hasattr(self, 'skull_offset'):
            # Water/pressure regions: blue-white-red based on pressure
            if self.pmin < 0 and self.pmax > 0:
                max_abs = max(abs(self.pmin), abs(self.pmax))
                color_func.AddRGBPoint(-max_abs, 1.0, 0.0, 0.0)  # Red
                color_func.AddRGBPoint(-max_abs/2, 1.0, 0.3, 0.3)  # Light red
                color_func.AddRGBPoint(0, 1.0, 1.0, 1.0)  # White
                color_func.AddRGBPoint(max_abs/2, 0.3, 0.3, 1.0)  # Light blue
                color_func.AddRGBPoint(max_abs, 0.0, 0.0, 1.0)  # Blue
            
            # Create LARGER gap to prevent interpolation artifacts
            gap_start = self.pmax * 10  # Increased from pmax + pressure_range
            gap_end = self.skull_offset * 0.8  # Start skull range earlier
            
            # End pressure range with red, then jump to white
            color_func.AddRGBPoint(gap_start, 1.0, 0.0, 0.0)  # Red
            color_func.AddRGBPoint(gap_end, 1.0, 1.0, 1.0)    # White
                
            # Skull regions: uniform white (no variation)
            color_func.AddRGBPoint(self.skull_offset - 100, 1.0, 1.0, 1.0)  # White
            color_func.AddRGBPoint(self.skull_offset, 1.0, 1.0, 1.0)        # White
            color_func.AddRGBPoint(self.skull_offset + 100, 1.0, 1.0, 1.0)  # White
            
            # Source regions: uniform gray (if sources exist)
            if hasattr(self, 'source_offset'):
                # Add break between skull and source
                source_gap_start = self.skull_offset + abs(self.pmax - self.pmin)
                source_gap_end = self.source_offset - abs(self.pmax - self.pmin)
                
                # End skull range with white
                color_func.AddRGBPoint(source_gap_start, 1.0, 1.0, 1.0)  # White
                # Start source range with gray
                gray_level = 0.5  # Medium gray
                color_func.AddRGBPoint(source_gap_end, gray_level, gray_level, gray_level)
                
                color_func.AddRGBPoint(self.source_offset - 100, gray_level, gray_level, gray_level)
                color_func.AddRGBPoint(self.source_offset, gray_level, gray_level, gray_level)
                color_func.AddRGBPoint(self.source_offset + 100, gray_level, gray_level, gray_level)
        else:
            # Standard color mapping
            if self.pmin < 0 and self.pmax > 0:
                max_abs = max(abs(self.pmin), abs(self.pmax))
                color_func.AddRGBPoint(-max_abs, 0.0, 0.0, 1.0)  # Blue
                color_func.AddRGBPoint(-max_abs/2, 0.3, 0.3, 1.0)  # Light blue
                color_func.AddRGBPoint(0, 1.0, 1.0, 1.0)  # White
                color_func.AddRGBPoint(max_abs/2, 1.0, 0.3, 0.3)  # Light red
                color_func.AddRGBPoint(max_abs, 1.0, 0.0, 0.0)  # Red
            else:
                color_func.AddRGBPoint(self.pmin, 0.0, 0.0, 1.0)  # Blue
                color_func.AddRGBPoint((self.pmin + self.pmax)/2, 1.0, 1.0, 1.0)  # White
                color_func.AddRGBPoint(self.pmax, 1.0, 0.0, 0.0)  # Red
            
        return color_func
        
    def create_opacity_transfer_function(self):
        """Create opacity transfer function for volume rendering"""
        opacity_func = vtk.vtkPiecewiseFunction()
        
        if self.skull_mask is not None and hasattr(self, 'skull_offset'):
            # Calculate thresholds for pressure wave visibility
            pressure_range = abs(self.pmax - self.pmin)
            low_threshold = pressure_range * 0.05   # 5% - where waves start to appear
            high_threshold = pressure_range * 0.2   # 20% - where waves are clearly visible
            
            print(f"\nOpacity function setup:")
            print(f"  Pressure range: {self.pmin:.0f} to {self.pmax:.0f}")
            print(f"  Low threshold: ±{low_threshold:.0f} Pa")
            print(f"  High threshold: ±{high_threshold:.0f} Pa")
            print(f"  Skull offset: {self.skull_offset:.0f}")
            if hasattr(self, 'source_offset'):
                print(f"  Source offset: {self.source_offset:.0f}")
                print(f"  Source mask voxels: {self.source_mask.sum() if self.source_mask is not None else 0}")
            
            print(f"\nColor mapping ranges:")
            print(f"  Pressure: {self.pmin:.0f} to {self.pmax:.0f} (blue-white-red)")
            gap_start = self.pmax + abs(self.pmax - self.pmin)
            gap_end = self.skull_offset - abs(self.pmax - self.pmin)
            print(f"  Gap: {gap_start:.0f} to {gap_end:.0f}")
            print(f"  Skull: {gap_end:.0f} to {self.skull_offset + abs(self.pmax - self.pmin):.0f} (white)")
            if hasattr(self, 'source_offset'):
                source_gap_start = self.skull_offset + abs(self.pmax - self.pmin)
                source_gap_end = self.source_offset - abs(self.pmax - self.pmin)
                print(f"  Source gap: {source_gap_start:.0f} to {source_gap_end:.0f}")
                print(f"  Source: {source_gap_end:.0f} to {self.source_offset + abs(self.pmax - self.pmin):.0f} (gray)")
            
            # Water/Non-skull regions: transparent except for pressure waves
            # Scale opacity values by user-provided scale factor
            scale = self.opacity_scale
            
            # Far negative pressures
            opacity_func.AddPoint(self.pmin, min(0.8 * scale, 1.0))  # Strong negative pressure visible
            opacity_func.AddPoint(self.pmin/2, min(0.6 * scale, 1.0))
            opacity_func.AddPoint(-high_threshold, min(0.5 * scale, 1.0))
            opacity_func.AddPoint(-low_threshold, min(0.2 * scale, 1.0))
            opacity_func.AddPoint(-low_threshold/2, 0.0)
            
            # Near zero - transparent
            opacity_func.AddPoint(0, 0.0)
            
            # Positive pressures
            opacity_func.AddPoint(low_threshold/2, 0.0)
            opacity_func.AddPoint(low_threshold, min(0.2 * scale, 1.0))
            opacity_func.AddPoint(high_threshold, min(0.5 * scale, 1.0))
            opacity_func.AddPoint(self.pmax/2, min(0.6 * scale, 1.0))
            opacity_func.AddPoint(self.pmax, min(0.8 * scale, 1.0))  # Strong positive pressure visible
            
            # Skull regions (offset values): always fully opaque
            # Make the entire skull range opaque
            opacity_func.AddPoint(self.skull_offset - abs(self.pmax - self.pmin), 1.0)
            opacity_func.AddPoint(self.skull_offset, 1.0)
            opacity_func.AddPoint(self.skull_offset + abs(self.pmax - self.pmin), 1.0)
            
            # Source regions (higher offset): also fully opaque
            if hasattr(self, 'source_offset'):
                opacity_func.AddPoint(self.source_offset - abs(self.pmax - self.pmin), 1.0)
                opacity_func.AddPoint(self.source_offset, 1.0)
                opacity_func.AddPoint(self.source_offset + abs(self.pmax - self.pmin), 1.0)
        else:
            # Fallback: standard pressure-based opacity
            threshold = abs(self.pmax - self.pmin) * 0.1
            
            opacity_func.AddPoint(self.pmin, 0.3)
            opacity_func.AddPoint(-threshold, 0.2)
            opacity_func.AddPoint(-threshold/2, 0.1)
            opacity_func.AddPoint(0, 0.0)
            opacity_func.AddPoint(threshold/2, 0.1)
            opacity_func.AddPoint(threshold, 0.2)
            opacity_func.AddPoint(self.pmax, 0.3)
        
        return opacity_func
        
    def setup_vtk_pipeline(self):
        """Setup the complete VTK rendering pipeline"""
        # Create renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.1)  # Dark gray background
        
        # Create render window
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetSize(1200, 800)
        self.render_window.SetWindowName("Volumetric Pressure Field Viewer")
        self.render_window.AddRenderer(self.renderer)
        
        # Create interactor
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        
        # Create initial volume
        self.create_volume_actor()
        
        # Add volume to renderer
        self.renderer.AddVolume(self.volume_actor)
        
        # Setup camera
        self.setup_camera()
        
        # Add text display
        self.create_text_display()
        
        # Add time slider
        self.create_time_slider()
        
        # Add axes widget
        self.add_axes_widget()
        
        # Add keyboard shortcuts
        self.interactor.AddObserver("KeyPressEvent", self.key_press_event)
        
    def create_volume_actor(self):
        """Create the volume actor with mapper and properties"""
        # Get initial volume data
        image_data = self.create_volume_data(self.current_timestep)
        
        # Create volume mapper
        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputData(image_data)
        volume_mapper.SetBlendModeToComposite()
        
        # Create volume property
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetInterpolationTypeToLinear()
        #volume_property.SetInterpolationTypeToNearest()  # Changed from Linear to Nearest

        volume_property.ShadeOff()  # Turn off shading for clearer visualization
        
        # Set color and opacity transfer functions
        color_func = self.create_color_transfer_function()
        opacity_func = self.create_opacity_transfer_function()
        
        volume_property.SetColor(color_func)
        volume_property.SetScalarOpacity(opacity_func)
        
        # Create volume actor
        self.volume_actor = vtk.vtkVolume()
        self.volume_actor.SetMapper(volume_mapper)
        self.volume_actor.SetProperty(volume_property)
        
    def setup_camera(self):
        """Setup initial camera position"""
        # Calculate center of volume
        bounds = self.volume_actor.GetBounds()
        center = [(bounds[1] + bounds[0])/2,
                  (bounds[3] + bounds[2])/2,
                  (bounds[5] + bounds[4])/2]
        
        # Position camera to view from an angle
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(center[0] + (bounds[1] - bounds[0]) * 1.5,
                          center[1] + (bounds[3] - bounds[2]) * 1.0,
                          center[2] + (bounds[5] - bounds[4]) * 1.5)
        camera.SetFocalPoint(center)
        camera.SetViewUp(0, 0, 1)
        
        self.renderer.ResetCamera()
        
    def create_text_display(self):
        """Create text display for time and instructions"""
        self.text_actor = vtk.vtkTextActor()
        self.text_actor.SetInput(f"Time: {self.time_us[self.current_timestep]:.1f} µs")
        
        text_property = self.text_actor.GetTextProperty()
        text_property.SetFontSize(18)
        text_property.SetColor(1.0, 1.0, 1.0)
        text_property.SetFontFamilyToArial()
        text_property.BoldOn()
        
        self.text_actor.SetPosition(10, 10)
        self.renderer.AddActor2D(self.text_actor)
        
        # Add instructions
        instructions = vtk.vtkTextActor()
        instructions.SetInput("Mouse: Rotate | Shift+Mouse: Pan | Ctrl+Mouse: Zoom | Space: Play/Pause | R: Reset View")
        
        inst_property = instructions.GetTextProperty()
        inst_property.SetFontSize(12)
        inst_property.SetColor(0.8, 0.8, 0.8)
        inst_property.SetFontFamilyToArial()
        
        instructions.SetPosition(10, 770)
        self.renderer.AddActor2D(instructions)
        
    def create_time_slider(self):
        """Create time slider widget"""
        # Create slider representation
        slider_rep = vtk.vtkSliderRepresentation2D()
        
        # Set slider properties
        slider_rep.SetMinimumValue(0)
        slider_rep.SetMaximumValue(self.time_steps - 1)
        slider_rep.SetValue(self.current_timestep)
        slider_rep.SetTitleText("Time")
        slider_rep.GetTitleProperty().SetColor(1.0, 1.0, 1.0)
        slider_rep.GetLabelProperty().SetColor(1.0, 1.0, 1.0)
        
        # Set slider position (right side of window)
        slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        slider_rep.GetPoint1Coordinate().SetValue(0.85, 0.1)
        slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        slider_rep.GetPoint2Coordinate().SetValue(0.85, 0.9)
        
        # Customize slider appearance
        slider_rep.SetSliderLength(0.02)
        slider_rep.SetSliderWidth(0.03)
        slider_rep.SetEndCapLength(0.01)
        slider_rep.SetEndCapWidth(0.03)
        slider_rep.SetTubeWidth(0.005)
        slider_rep.GetSliderProperty().SetColor(0.5, 0.5, 0.5)
        slider_rep.GetCapProperty().SetColor(0.5, 0.5, 0.5)
        slider_rep.GetSelectedProperty().SetColor(0.8, 0.8, 0.8)
        
        # Create slider widget
        self.slider_widget = vtk.vtkSliderWidget()
        self.slider_widget.SetInteractor(self.interactor)
        self.slider_widget.SetRepresentation(slider_rep)
        self.slider_widget.SetAnimationModeToAnimate()
        self.slider_widget.EnabledOn()
        
        # Add observer for slider interaction
        self.slider_widget.AddObserver(vtk.vtkCommand.InteractionEvent, self.slider_callback)
        
    def add_axes_widget(self):
        """Add 3D axes widget to show orientation"""
        axes = vtk.vtkAxesActor()
        
        # Create orientation marker widget
        widget = vtk.vtkOrientationMarkerWidget()
        widget.SetOrientationMarker(axes)
        widget.SetInteractor(self.interactor)
        widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        widget.SetEnabled(1)
        widget.InteractiveOff()
        
    def slider_callback(self, obj, event):
        """Handle slider value changes"""
        value = int(obj.GetRepresentation().GetValue())
        if value != self.current_timestep:
            self.update_timestep(value)
            
    def update_timestep(self, timestep):
        """Update the displayed timestep"""
        self.current_timestep = timestep
        
        # Update volume data
        image_data = self.create_volume_data(timestep)
        self.volume_actor.GetMapper().SetInputData(image_data)
        
        # Update time display
        self.text_actor.SetInput(f"Time: {self.time_us[timestep]:.1f} µs")
        
        # Render
        self.render_window.Render()
        
    def key_press_event(self, obj, event):
        """Handle keyboard events"""
        key = self.interactor.GetKeySym()
        
        if key == "space":
            # Toggle play/pause animation
            self.toggle_animation()
        elif key == "Left":
            # Previous timestep
            if self.current_timestep > 0:
                self.slider_widget.GetRepresentation().SetValue(self.current_timestep - 1)
                self.update_timestep(self.current_timestep - 1)
        elif key == "Right":
            # Next timestep
            if self.current_timestep < self.time_steps - 1:
                self.slider_widget.GetRepresentation().SetValue(self.current_timestep + 1)
                self.update_timestep(self.current_timestep + 1)
        elif key == "Home":
            # Go to first timestep
            self.slider_widget.GetRepresentation().SetValue(0)
            self.update_timestep(0)
        elif key == "End":
            # Go to last timestep
            self.slider_widget.GetRepresentation().SetValue(self.time_steps - 1)
            self.update_timestep(self.time_steps - 1)
        elif key == "r" or key == "R":
            # Reset camera
            self.renderer.ResetCamera()
            self.render_window.Render()
            
    def toggle_animation(self):
        """Toggle animation playback"""
        if not hasattr(self, 'animating'):
            self.animating = False
            
        self.animating = not self.animating
        
        if self.animating:
            # Start animation
            self.interactor.CreateRepeatingTimer(50)  # 20 FPS
            self.interactor.AddObserver("TimerEvent", self.timer_callback)
        else:
            # Stop animation
            self.interactor.DestroyTimer()
            
    def timer_callback(self, obj, event):
        """Timer callback for animation"""
        if self.animating:
            # Advance to next timestep
            next_step = (self.current_timestep + 1) % self.time_steps
            self.slider_widget.GetRepresentation().SetValue(next_step)
            self.update_timestep(next_step)
            
    def start(self):
        """Start the interactive viewer"""
        print("\nStarting volumetric viewer...")
        print("Controls:")
        print("  Mouse: Rotate view")
        print("  Shift+Mouse: Pan view")
        print("  Ctrl+Mouse: Zoom")
        print("  Space: Play/Pause animation")
        print("  Left/Right arrows: Previous/Next timestep")
        print("  R: Reset camera view")
        print("  Q: Quit")
        print("\nVisualization key:")
        print("  Gray blocks: Transducer elements")
        print("  White solid: Skull structure")
        print("  Blue/Red waves: Pressure (negative/positive)")
        print("\nTip: If pressure waves are hard to see, try:")
        print("  python volumetric_movie.py --opacity-scale 2.0")
        
        # Initialize and start
        self.render_window.Render()
        self.interactor.Initialize()
        self.interactor.Start()


def main():
    """Main function"""
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Volumetric pressure field visualization')
    parser.add_argument('movie_file', nargs='?', default=None,
                       help='Path to pressure_movie.h5 file')
    parser.add_argument('--opacity-scale', type=float, default=1.0,
                       help='Scale factor for pressure wave opacity (default: 1.0)')
    
    args = parser.parse_args()
    
    # Determine movie file path
    if args.movie_file:
        movie_file = args.movie_file
    else:
        # Default to looking in current directory
        movie_file = "pressure_movie.h5"
        
        # If not in current directory, use the specified directory
        if not os.path.exists(movie_file):
            # Use the directory with skull mask data
            default_dir = "sar_simulation_20250620_183708"
            movie_file = os.path.join(default_dir, "pressure_movie.h5")
    
    # Check if file exists
    if not os.path.exists(movie_file):
        print(f"Error: Pressure movie file not found: {movie_file}")
        print("\nUsage: python volumetric_movie.py [pressure_movie.h5]")
        print("\nMake sure to run the SAR simulation with -movie flag first:")
        print("  python sar.py -movie")
        return
    
    print(f"Loading pressure movie from: {movie_file}")
    if args.opacity_scale != 1.0:
        print(f"Using opacity scale factor: {args.opacity_scale}")
    
    # Create and start viewer
    viewer = VolumetricPressureViewer(movie_file, opacity_scale=args.opacity_scale)
    viewer.start()


if __name__ == "__main__":
    main() 