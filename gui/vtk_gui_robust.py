#!/usr/bin/env python3
"""
Robust VTK GUI Application with better error handling
Supports both pressure_movie.h5 and output.h5 files
"""

import os
import sys
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime

# Trame imports
from trame.app import get_server
from trame.ui.vuetify2 import SinglePageLayout
from trame.widgets import vtk as vtk_widgets, vuetify2 as vuetify, html

# VTK imports
import vtk
from vtk import (
    vtkImageData, vtkSmartVolumeMapper, vtkVolume, vtkVolumeProperty,
    vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor,
    vtkColorTransferFunction, vtkPiecewiseFunction, vtkFloatArray,
    vtkImageMarchingCubes, vtkPolyDataNormals, vtkPolyDataMapper, vtkActor,
    vtkOutlineFilter, vtkAxesActor
)


class RobustVTKGUI:
    """Robust VTK GUI with better error handling"""
    
    def __init__(self):
        # Create server
        self.server = get_server(client_type='vue2')
        self.state = self.server.state
        self.ctrl = self.server.controller
        
        # Initialize state
        self.state.simulation_dirs = []
        self.state.selected_dir = None
        self.state.visualization_mode = 'transducer'
        self.state.current_timestep = 0
        self.state.max_timestep = 0
        self.state.playing = False
        self.state.status_message = "Select a simulation directory"
        self.state.error_message = ""
        
        # VTK setup
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.15)
        
        self.render_window = vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetOffScreenRendering(1)
        
        self.interactor = vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        
        # Data holders
        self.pressure_data = None
        self.current_actors = []
        self.volume_actor = None
        
        # Scan directories
        self.scan_simulation_directories()
        
        # Setup UI
        self.setup_ui()
        
        # Register triggers
        self.ctrl.update_timestep = self.update_timestep
        
        # Load initial visualization if available
        if self.state.simulation_dirs:
            self.load_current_visualization()
        
    def scan_simulation_directories(self):
        """Scan for simulation directories with available data"""
        sim_path = Path('data/simulations')
        if not sim_path.exists():
            sim_path.mkdir(parents=True, exist_ok=True)
            
        dirs = []
        for item in sorted(sim_path.iterdir(), reverse=True):
            if item.is_dir():
                # Check available files
                has_transducer = (item / 'transducers.npy').exists()
                has_pressure_movie = (item / 'pressure_movie.h5').exists()
                has_output = (item / 'output.h5').exists()
                
                # Only add if it has at least one valid file
                if has_transducer or has_pressure_movie or has_output:
                    # Try to determine which pressure file is valid
                    pressure_file = None
                    if has_pressure_movie:
                        try:
                            with h5py.File(item / 'pressure_movie.h5', 'r') as f:
                                if 'pressure' in f:
                                    pressure_file = 'pressure_movie.h5'
                        except:
                            pass
                    
                    if not pressure_file and has_output:
                        try:
                            with h5py.File(item / 'output.h5', 'r') as f:
                                if 'p' in f:
                                    pressure_file = 'output.h5'
                        except:
                            pass
                    
                    dirs.append({
                        'text': item.name,
                        'value': str(item),
                        'has_transducer': has_transducer,
                        'has_pressure': pressure_file is not None,
                        'pressure_file': pressure_file
                    })
        
        self.state.simulation_dirs = dirs
        if dirs:
            self.state.selected_dir = dirs[0]['value']
            
    def clear_visualization(self):
        """Clear all actors from renderer"""
        for actor in self.current_actors:
            self.renderer.RemoveActor(actor)
        self.current_actors = []
        
        if self.volume_actor:
            self.renderer.RemoveVolume(self.volume_actor)
            self.volume_actor = None
        
        # Keep axes
        axes = vtkAxesActor()
        axes.SetTotalLength(10, 10, 10)
        self.renderer.AddActor(axes)
        self.current_actors.append(axes)
        
    def load_transducer_data(self, sim_dir):
        """Load transducer visualization"""
        try:
            self.state.error_message = ""
            npy_file = os.path.join(sim_dir, 'transducers.npy')
            
            if not os.path.exists(npy_file):
                self.state.error_message = "No transducers.npy found"
                return
                
            # Clear previous
            self.clear_visualization()
            
            # Load data
            data = np.load(npy_file)
            if data.dtype == bool:
                data = data.astype(np.uint8)
                
            self.state.status_message = f"Transducers: {data.shape}, {data.sum()} active voxels"
            
            # Create VTK data
            spacing = [0.5, 0.5, 0.5]
            vtk_data = vtkImageData()
            vtk_data.SetDimensions(data.shape)
            vtk_data.SetSpacing(spacing)
            vtk_data.SetOrigin(0, 0, 0)
            
            # Convert to VTK
            flat_data = data.flatten(order='F')
            vtk_array = vtk.vtkUnsignedCharArray()
            vtk_array.SetNumberOfTuples(flat_data.size)
            for i in range(flat_data.size):
                vtk_array.SetValue(i, int(flat_data[i]))
            vtk_data.GetPointData().SetScalars(vtk_array)
            
            # Create surface
            if data.any():
                mc = vtkImageMarchingCubes()
                mc.SetInputData(vtk_data)
                mc.SetValue(0, 0.5)
                mc.Update()
                
                normals = vtkPolyDataNormals()
                normals.SetInputConnection(mc.GetOutputPort())
                normals.ComputePointNormalsOn()
                normals.Update()
                
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(normals.GetOutputPort())
                
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(1.0, 0.2, 0.2)
                actor.GetProperty().SetOpacity(1.0)
                actor.GetProperty().SetSpecular(0.5)
                
                self.renderer.AddActor(actor)
                self.current_actors.append(actor)
            
            # Add outline
            outline = vtkOutlineFilter()
            outline.SetInputData(vtk_data)
            outline_mapper = vtkPolyDataMapper()
            outline_mapper.SetInputConnection(outline.GetOutputPort())
            outline_actor = vtkActor()
            outline_actor.SetMapper(outline_mapper)
            outline_actor.GetProperty().SetColor(1, 1, 1)
            self.renderer.AddActor(outline_actor)
            self.current_actors.append(outline_actor)
            
            self.renderer.ResetCamera()
            self.ctrl.view_update()
            
        except Exception as e:
            self.state.error_message = f"Error: {str(e)}"
            
    def load_pressure_data(self, sim_dir):
        """Load pressure data from available files"""
        try:
            self.state.error_message = ""
            self.clear_visualization()
            
            # Find which pressure file to use
            pressure_file = None
            dir_info = next((d for d in self.state.simulation_dirs if d['value'] == sim_dir), None)
            
            if dir_info and dir_info.get('pressure_file'):
                pressure_file = os.path.join(sim_dir, dir_info['pressure_file'])
            else:
                # Try both files
                for fname in ['pressure_movie.h5', 'output.h5']:
                    test_file = os.path.join(sim_dir, fname)
                    if os.path.exists(test_file):
                        try:
                            with h5py.File(test_file, 'r') as f:
                                if 'pressure' in f or 'p' in f:
                                    pressure_file = test_file
                                    break
                        except:
                            continue
            
            if not pressure_file:
                self.state.error_message = "No valid pressure data found"
                return
                
            # Load the data
            with h5py.File(pressure_file, 'r') as f:
                # Handle different data formats
                if 'pressure' in f:
                    self.pressure_data = f['pressure'][:]
                    pressure_key = 'pressure'
                elif 'p' in f:
                    # Handle k-Wave output format
                    p_data = f['p'][:]
                    # p might be (1, time_steps, total_grid_points)
                    if p_data.ndim == 3 and p_data.shape[0] == 1:
                        p_data = p_data[0]  # Remove first dimension
                    
                    # Need grid dimensions to reshape
                    if all(k in f for k in ['Nx', 'Ny', 'Nz']):
                        nx = int(f['Nx'][...].item())
                        ny = int(f['Ny'][...].item())
                        nz = int(f['Nz'][...].item())
                        
                        # Reshape from (time_steps, total_points) to (time_steps, nx, ny, nz)
                        if p_data.ndim == 2:
                            num_steps = p_data.shape[0]
                            self.pressure_data = np.zeros((num_steps, nx, ny, nz))
                            for t in range(num_steps):
                                self.pressure_data[t] = p_data[t].reshape((nx, ny, nz), order='F')
                        else:
                            self.state.error_message = "Unexpected pressure data format"
                            return
                    else:
                        self.state.error_message = "Missing grid dimensions in output file"
                        return
                    pressure_key = 'p'
                else:
                    self.state.error_message = "No pressure data found in file"
                    return
                
                # Get time data if available
                self.time_us = f['time_us'][:] if 'time_us' in f else None
                
                # Get metadata
                self.metadata = {}
                if 'dx' in f:
                    self.metadata['dx'] = float(f['dx'][...].item())
                elif 'dx' in f.attrs:
                    self.metadata['dx'] = f.attrs['dx']
                else:
                    self.metadata['dx'] = 0.001  # Default 1mm
            
            # Update state
            self.time_steps, self.nx, self.ny, self.nz = self.pressure_data.shape
            self.state.max_timestep = self.time_steps - 1
            self.state.current_timestep = 0
            
            # Calculate pressure range
            self.pmin = np.percentile(self.pressure_data, 0.1)
            self.pmax = np.percentile(self.pressure_data, 99.9)
            
            self.state.status_message = f"Pressure data: {self.pressure_data.shape}, range [{self.pmin:.0f}, {self.pmax:.0f}] Pa"
            
            # Create visualization
            self.create_pressure_visualization()
            
        except Exception as e:
            self.state.error_message = f"Error loading pressure: {str(e)}"
            import traceback
            traceback.print_exc()
            
    def create_pressure_visualization(self):
        """Create volume rendering for pressure"""
        if self.pressure_data is None:
            return
            
        # Create volume data
        image_data = self.create_volume_data(self.state.current_timestep)
        
        # Create mapper
        mapper = vtkSmartVolumeMapper()
        mapper.SetInputData(image_data)
        mapper.SetBlendModeToComposite()
        
        # Create property
        prop = vtkVolumeProperty()
        prop.SetInterpolationTypeToLinear()
        prop.ShadeOff()
        
        # Transfer functions
        color_func = vtkColorTransferFunction()
        if self.pmin < 0 and self.pmax > 0:
            max_abs = max(abs(self.pmin), abs(self.pmax))
            color_func.AddRGBPoint(-max_abs, 0.0, 0.0, 1.0)
            color_func.AddRGBPoint(0, 1.0, 1.0, 1.0)
            color_func.AddRGBPoint(max_abs, 1.0, 0.0, 0.0)
        else:
            color_func.AddRGBPoint(self.pmin, 0.0, 0.0, 1.0)
            color_func.AddRGBPoint((self.pmin + self.pmax)/2, 1.0, 1.0, 1.0)
            color_func.AddRGBPoint(self.pmax, 1.0, 0.0, 0.0)
            
        opacity_func = vtkPiecewiseFunction()
        threshold = abs(self.pmax - self.pmin) * 0.1
        opacity_func.AddPoint(self.pmin, 0.3)
        opacity_func.AddPoint(-threshold, 0.1)
        opacity_func.AddPoint(0, 0.0)
        opacity_func.AddPoint(threshold, 0.1)
        opacity_func.AddPoint(self.pmax, 0.3)
        
        prop.SetColor(color_func)
        prop.SetScalarOpacity(opacity_func)
        
        # Create volume
        self.volume_actor = vtkVolume()
        self.volume_actor.SetMapper(mapper)
        self.volume_actor.SetProperty(prop)
        
        self.renderer.AddVolume(self.volume_actor)
        self.renderer.ResetCamera()
        self.ctrl.view_update()
        
    def create_volume_data(self, timestep):
        """Create VTK volume data for timestep"""
        pressure_field = self.pressure_data[timestep]
        
        image_data = vtkImageData()
        image_data.SetDimensions(self.nx, self.ny, self.nz)
        
        dx = self.metadata.get('dx', 0.001) * 1000
        image_data.SetSpacing(dx, dx, dx)
        
        flat_pressure = pressure_field.flatten(order='F')
        pressure_array = vtkFloatArray()
        pressure_array.SetNumberOfTuples(flat_pressure.size)
        for i in range(flat_pressure.size):
            pressure_array.SetValue(i, float(flat_pressure[i]))
        pressure_array.SetName("Pressure")
        
        image_data.GetPointData().SetScalars(pressure_array)
        return image_data
        
    def on_directory_change(self, directory):
        """Handle directory selection"""
        self.state.selected_dir = directory
        self.load_current_visualization()
        
    def on_mode_change(self, mode):
        """Handle mode change"""
        self.state.visualization_mode = mode
        self.load_current_visualization()
        
    def load_current_visualization(self):
        """Load visualization based on current selection"""
        if not self.state.selected_dir:
            return
            
        if self.state.visualization_mode == 'transducer':
            self.load_transducer_data(self.state.selected_dir)
        else:
            self.load_pressure_data(self.state.selected_dir)
            
    def update_timestep(self, timestep):
        """Update pressure timestep"""
        if self.pressure_data is None or self.volume_actor is None:
            return
            
        self.state.current_timestep = int(timestep)
        image_data = self.create_volume_data(self.state.current_timestep)
        self.volume_actor.GetMapper().SetInputData(image_data)
        
        if self.time_us is not None:
            self.state.status_message = f"Time: {self.time_us[self.state.current_timestep]:.1f} µs"
        else:
            self.state.status_message = f"Frame: {self.state.current_timestep}/{self.state.max_timestep}"
        
        self.ctrl.view_update()
        
    def toggle_play(self):
        """Toggle animation"""
        self.state.playing = not self.state.playing
        
        if self.state.playing and self.pressure_data is not None:
            # Use JavaScript timer for animation
            self.server.js_call(
                """
                if (window.animationTimer) {
                    clearInterval(window.animationTimer);
                }
                window.animationTimer = setInterval(() => {
                    const nextStep = (state.current_timestep + 1) % (state.max_timestep + 1);
                    state.current_timestep = nextStep;
                    trigger('update_timestep', nextStep);
                }, 50); // 20 FPS
                """
            )
        else:
            # Stop animation
            self.server.js_call(
                """
                if (window.animationTimer) {
                    clearInterval(window.animationTimer);
                    window.animationTimer = null;
                }
                """
            )
            
    def reset_camera(self):
        """Reset camera view"""
        self.renderer.ResetCamera()
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(45)
        camera.Elevation(30)
        camera.Zoom(1.2)
        self.ctrl.view_update()
        
    def clear_all(self):
        """Clear all visualizations (eject button)"""
        # Stop any playing animation
        if self.state.playing:
            self.state.playing = False
            # Stop JavaScript animation timer
            self.server.js_call(
                """
                if (window.animationTimer) {
                    clearInterval(window.animationTimer);
                    window.animationTimer = null;
                }
                """
            )
        
        # Clear visualization
        self.clear_visualization()
        
        # Reset state
        self.state.current_timestep = 0
        self.state.max_timestep = 0
        self.state.status_message = "Visualization cleared. Select data to load."
        self.state.error_message = ""
        self.pressure_data = None
        
        # Update view
        self.ctrl.view_update()
        
    def setup_ui(self):
        """Setup user interface"""
        with SinglePageLayout(self.server) as layout:
            layout.title = "VTK Visualization"
            
            with layout.toolbar:
                # Directory selector
                vuetify.VSelect(
                    v_model=("selected_dir",),
                    items=("simulation_dirs",),
                    label="Simulation",
                    dense=True,
                    hide_details=True,
                    style="max-width: 300px; margin-right: 20px",
                    change=(self.on_directory_change, "[$event]")
                )
                
                # Mode buttons
                with vuetify.VBtnToggle(
                    v_model=("visualization_mode",),
                    mandatory=True,
                    dense=True,
                    style="margin-right: 20px",
                    change=(self.on_mode_change, "[$event]")
                ):
                    vuetify.VBtn("Transducer", value="transducer")
                    vuetify.VBtn("Pressure", value="pressure")
                
                # Big play button for pressure mode
                with vuetify.VBtn(
                    v_show="visualization_mode === 'pressure'",
                    icon=True,
                    x_large=True,
                    color="primary",
                    style="margin-right: 20px",
                    click=self.toggle_play
                ):
                        vuetify.VIcon(
                            "mdi-play",
                            v_if="!playing",
                            x_large=True
                    )
                        vuetify.VIcon(
                            "mdi-pause",
                            v_if="playing",
                            x_large=True
                )
                
                # Time slider for pressure mode
                with html.Div(v_show="visualization_mode === 'pressure'", style="display: flex; align-items: center; margin-right: 20px"):
                    vuetify.VSlider(
                        v_model=("current_timestep",),
                        min=0,
                        max=("max_timestep",),
                        step=1,
                        hide_details=True,
                        dense=True,
                        style="width: 200px; margin-right: 10px",
                        change=(self.update_timestep, "[$event]")
                    )
                    
                    vuetify.VChip(
                        "{{ current_timestep }}/{{ max_timestep }}",
                        small=True,
                        label=True,
                        outlined=True
                    )
                
                vuetify.VSpacer()
                
                # Eject button with tooltip
                with vuetify.VTooltip(bottom=True):
                    with vuetify.Template(v_slot_activator="{ on, attrs }"):
                        with vuetify.VBtn(
                            icon=True,
                            color="error",
                            style="margin-right: 10px",
                            click=self.clear_all,
                            v_bind="attrs",
                            v_on="on"
                        ):
                            vuetify.VIcon("mdi-eject")
                    html.Span("Clear all visualizations")
                
                # Reset camera button
                vuetify.VBtn("Reset Camera", click=self.reset_camera, small=True, outlined=True)
                
            with layout.content:
                with vuetify.VContainer(fluid=True, classes="pa-0 fill-height"):
                    # Error display
                    vuetify.VAlert(
                        "{{ error_message }}",
                        v_if="error_message",
                        type="error",
                        dense=True,
                        text=True
                    )
                    
                    # Status display
                    vuetify.VAlert(
                        "{{ status_message }}",
                        type="info",
                        dense=True,
                        text=True
                    )
                    
                    # VTK view
                    view = vtk_widgets.VtkLocalView(self.render_window)
                    self.ctrl.view_update = view.update
                    
    def start(self, port=8888, host='0.0.0.0'):
        """Start server"""
        print(f"\nVTK Visualization Server")
        print(f"URL: http://{host}:{port}")
        print("\nFeatures:")
        print("- Auto-detects valid data files")
        print("- Handles both pressure_movie.h5 and output.h5")
        print("- Error recovery and status messages")
        print("\nPress Ctrl+C to stop\n")
        
        self.server.start(
            port=port,
            host=host,
            open_browser=False,
            backend='aiohttp',
            exec_mode='main'
        )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8888)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()
    
    gui = RobustVTKGUI()
    gui.start(port=args.port, host=args.host)


if __name__ == "__main__":
    main() 