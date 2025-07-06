#!/usr/bin/env python3
"""
VTK GUI Application with Directory Selection
Allows users to browse simulation directories and view transducers or pressure movies
"""

import os
import sys
import numpy as np
import h5py
import json
from pathlib import Path
from datetime import datetime

# Trame imports
from trame.app import get_server, asynchronous
from trame.ui.vuetify2 import SinglePageLayout
from trame.widgets import vtk as vtk_widgets, vuetify2 as vuetify, html
import asyncio

# VTK imports
import vtk
from vtk import (
    vtkImageData, vtkSmartVolumeMapper, vtkVolume, vtkVolumeProperty,
    vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor,
    vtkColorTransferFunction, vtkPiecewiseFunction, vtkFloatArray,
    vtkImageMarchingCubes, vtkPolyDataNormals, vtkPolyDataMapper, vtkActor,
    vtkOutlineFilter, vtkAxesActor, vtkOrientationMarkerWidget, vtkPlaneSource,
    vtkHexahedron, vtkUnstructuredGrid, vtkDataSetSurfaceFilter
)


class VTKVisualizationGUI:
    """Main GUI application for VTK visualizations"""
    
    def __init__(self):
        # Create server
        self.server = get_server(client_type='vue2')
        self.state = self.server.state
        self.ctrl = self.server.controller
        
        # Initialize state variables
        self.state.simulation_dirs = []
        self.state.selected_dir = None
        self.state.visualization_mode = 'transducer'  # 'transducer' or 'pressure'
        self.state.current_timestep = 0
        self.state.max_timestep = 0
        self.state.playing = False
        self.state.loading = False
        self.state.loading_message = "Loading..."
        self.state.status_message = "Select a simulation directory to begin"
        
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
        self.transducer_data = None
        self.pressure_data = None
        self.current_actor = None
        self.volume_actor = None
        
        # Scan for simulation directories
        self.scan_simulation_directories()
        
        # Register controller methods
        self.ctrl.update_timestep = self.update_timestep
        
        # Whenever current_timestep changes, repaint
        @self.state.change("current_timestep")
        def _frame_changed(**kwargs):
            if "current_timestep" in kwargs:
                self.update_timestep(int(kwargs["current_timestep"]))
        
        # Setup UI
        self.setup_ui()
        
    def scan_simulation_directories(self):
        """Scan data/simulations for available directories"""
        sim_path = Path('data/simulations')
        if not sim_path.exists():
            sim_path.mkdir(parents=True, exist_ok=True)
            
        dirs = []
        for item in sorted(sim_path.iterdir(), reverse=True):
            if item.is_dir():
                # Check if it has the required files
                has_transducer = (item / 'transducers.npy').exists()
                has_pressure = (item / 'pressure_movie.h5').exists()
                
                if has_transducer or has_pressure:
                    dirs.append({
                        'text': item.name,
                        'value': str(item),
                        'has_transducer': has_transducer,
                        'has_pressure': has_pressure
                    })
        
        self.state.simulation_dirs = dirs
        if dirs:
            self.state.selected_dir = dirs[0]['value']
            
    def load_transducer_data(self, sim_dir):
        """Load and visualize transducer data"""
        try:
            npy_file = os.path.join(sim_dir, 'transducers.npy')
            
            if not os.path.exists(npy_file):
                self.state.status_message = f"No transducers.npy found in {sim_dir}"
                return
                
            # Show loading state
            self.state.loading = True
            self.state.loading_message = "Loading transducer data..."
            self.state.status_message = "Loading transducer data..."
            self.ctrl.view_update()
                
            # Clear previous actors
            self.clear_visualization()
            
            # Load element coordinates
            element_coords = np.load(npy_file, allow_pickle=True)
            num_elements = len(element_coords)
            
            self.state.status_message = f"Loading {num_elements} transducer elements..."
            
            # Get spacing from config if available
            spacing = 0.5  # Default spacing in mm
            cfg_file = Path(sim_dir) / "config.json"
            if cfg_file.exists():
                with open(cfg_file) as f:
                    config = json.load(f)
                    spacing = config["grid"]["dx"] * 1000   # m → mm
            
            # Create individual rectangle for each element
            for i, (corner1, corner2) in enumerate(element_coords):
                # Corners are in meters from k-Wave, convert to mm for VTK
                # k-Wave uses grid center as origin, so we need to offset
                
                # Get grid offset if available from config
                grid_offset = np.array([0, 0, 0])
                if cfg_file.exists():
                    # k-Wave grid is centered at origin, VTK uses corner as origin
                    grid_size_m = np.array(config["grid"]["grid_size_mm"]) / 1000
                    grid_offset = grid_size_m / 2
                
                # Convert to VTK coordinates (mm) with offset
                c1 = (corner1 + grid_offset) * 1000
                c2 = (corner2 + grid_offset) * 1000
                
                # For thin transducer elements, create a thin box representation
                # The two corners define opposite corners of a rectangle
                # We'll create a very thin box to represent the planar element
                thickness = 0.1  # Very thin (0.1 mm)
                
                # Calculate center
                center = (c1 + c2) / 2.0
                
                # For a rectangle defined by opposite corners, we need to find all 4 corners
                # The element_coords contain left_corner and right_corner which are diagonally opposite
                # We need to construct the other two corners
                
                # Find the extent in each dimension
                x_vals = [c1[0], c2[0]]
                y_vals = [c1[1], c2[1]]
                z_vals = [c1[2], c2[2]]
                
                # Create all 8 vertices of a thin box
                vertices = vtk.vtkPoints()
                vertices.SetNumberOfPoints(8)
                
                # Bottom face (4 vertices)
                vertices.SetPoint(0, min(x_vals), min(y_vals), min(z_vals))
                vertices.SetPoint(1, max(x_vals), min(y_vals), min(z_vals))
                vertices.SetPoint(2, max(x_vals), max(y_vals), min(z_vals))
                vertices.SetPoint(3, min(x_vals), max(y_vals), min(z_vals))
                
                # Top face (4 vertices) - offset by small thickness
                # Determine which dimension has the smallest extent
                x_extent = abs(max(x_vals) - min(x_vals))
                y_extent = abs(max(y_vals) - min(y_vals))
                z_extent = abs(max(z_vals) - min(z_vals))
                
                # Add thickness in the thinnest dimension
                if x_extent <= y_extent and x_extent <= z_extent:
                    # X is thinnest - add thickness in X
                    offset = thickness if x_extent < thickness else 0
                    vertices.SetPoint(4, min(x_vals) - offset/2, min(y_vals), min(z_vals))
                    vertices.SetPoint(5, max(x_vals) + offset/2, min(y_vals), min(z_vals))
                    vertices.SetPoint(6, max(x_vals) + offset/2, max(y_vals), min(z_vals))
                    vertices.SetPoint(7, min(x_vals) - offset/2, max(y_vals), min(z_vals))
                elif y_extent <= x_extent and y_extent <= z_extent:
                    # Y is thinnest - add thickness in Y
                    offset = thickness if y_extent < thickness else 0
                    vertices.SetPoint(4, min(x_vals), min(y_vals) - offset/2, min(z_vals))
                    vertices.SetPoint(5, max(x_vals), min(y_vals) - offset/2, min(z_vals))
                    vertices.SetPoint(6, max(x_vals), max(y_vals) + offset/2, min(z_vals))
                    vertices.SetPoint(7, min(x_vals), max(y_vals) + offset/2, min(z_vals))
                else:
                    # Z is thinnest - add thickness in Z
                    offset = thickness if z_extent < thickness else 0
                    vertices.SetPoint(4, min(x_vals), min(y_vals), max(z_vals) + offset)
                    vertices.SetPoint(5, max(x_vals), min(y_vals), max(z_vals) + offset)
                    vertices.SetPoint(6, max(x_vals), max(y_vals), max(z_vals) + offset)
                    vertices.SetPoint(7, min(x_vals), max(y_vals), max(z_vals) + offset)
                
                # Create the box using vtkHexahedron
                hexahedron = vtk.vtkHexahedron()
                for j in range(8):
                    hexahedron.GetPointIds().SetId(j, j)
                
                # Create unstructured grid
                grid = vtk.vtkUnstructuredGrid()
                grid.SetPoints(vertices)
                grid.InsertNextCell(hexahedron.GetCellType(), hexahedron.GetPointIds())
                
                # Extract surface for rendering
                surface_filter = vtk.vtkDataSetSurfaceFilter()
                surface_filter.SetInputData(grid)
                surface_filter.Update()
                
                # Create mapper and actor
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(surface_filter.GetOutputPort())
                
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(1.0, 0.2, 0.2)  # Red color
                actor.GetProperty().SetOpacity(0.9)
                actor.GetProperty().SetSpecular(0.3)
                actor.GetProperty().SetSpecularPower(20)
                actor.GetProperty().SetEdgeVisibility(True)
                actor.GetProperty().SetEdgeColor(0.5, 0.1, 0.1)
                actor.GetProperty().SetLineWidth(2)
                
                self.renderer.AddActor(actor)
            
            # Add coordinate axes
            axes = vtkAxesActor()
            axes.SetTotalLength(20, 20, 20)
            self.renderer.AddActor(axes)
            
            # Add grid outline based on config
            if cfg_file.exists():
                grid_shape = config["grid"]["grid_shape"]
                grid_size_mm = config["grid"]["grid_size_mm"]
                
                # Create outline
                outline_source = vtk.vtkOutlineSource()
                outline_source.SetBounds(
                    0, grid_size_mm[0],
                    0, grid_size_mm[1], 
                    0, grid_size_mm[2]
                )
                
                outline_mapper = vtkPolyDataMapper()
                outline_mapper.SetInputConnection(outline_source.GetOutputPort())
                
                outline_actor = vtkActor()
                outline_actor.SetMapper(outline_mapper)
                outline_actor.GetProperty().SetColor(1, 1, 1)
                self.renderer.AddActor(outline_actor)
            
            # Reset camera
            self.renderer.ResetCamera()
            
            # Update status
            self.state.status_message = f"Loaded {num_elements} transducer elements"
            
            # Hide loading state
            self.state.loading = False
            self.ctrl.view_update()
            
        except Exception as e:
            self.state.loading = False
            self.state.status_message = f"Error loading transducer data: {str(e)}"
            
    def load_pressure_data(self, sim_dir):
        """Load and prepare pressure movie data"""
        try:
            h5_file = os.path.join(sim_dir, 'pressure_movie.h5')
            
            if not os.path.exists(h5_file):
                self.state.status_message = f"No pressure_movie.h5 found in {sim_dir}"
                return
                
            # Show loading state
            self.state.loading = True
            self.state.loading_message = "Loading pressure data (this may take a moment)..."
            self.state.status_message = "Loading pressure data..."
            self.ctrl.view_update()
                
            # Clear previous visualization
            self.clear_visualization()
            
            # Load data
            with h5py.File(h5_file, 'r') as f:
                self.pressure_data = f['pressure'][:]
                self.time_us = f['time_us'][:] if 'time_us' in f else None
                self.metadata = dict(f.attrs)
                
                # Load masks if available
                self.source_mask = f['source_mask'][:] if 'source_mask' in f else None
                self.skull_mask = f['skull_mask'][:] if 'skull_mask' in f else None
            
            # Update state
            self.time_steps, self.nx, self.ny, self.nz = self.pressure_data.shape
            self.state.max_timestep = self.time_steps - 1
            self.state.current_timestep = 0
            
            # Calculate pressure range
            self.pmin = np.percentile(self.pressure_data, 0.1)
            self.pmax = np.percentile(self.pressure_data, 99.9)
            
            self.state.status_message = f"Loaded pressure data: {self.pressure_data.shape}, range [{self.pmin:.0f}, {self.pmax:.0f}] Pa"
            
            # Create initial visualization
            self.create_pressure_visualization()
            
            # Hide loading state
            self.state.loading = False
            
        except Exception as e:
            self.state.loading = False
            self.state.status_message = f"Error loading pressure data: {str(e)}"
            
    def create_pressure_visualization(self):
        """Create volume rendering for pressure data"""
        if self.pressure_data is None:
            return
            
        # Get current timestep data
        image_data = self.create_volume_data(self.state.current_timestep)
        
        # Create volume mapper
        volume_mapper = vtkSmartVolumeMapper()
        volume_mapper.SetInputData(image_data)
        volume_mapper.SetBlendModeToComposite()
        
        # Create volume property
        volume_property = vtkVolumeProperty()
        volume_property.SetInterpolationTypeToLinear()
        volume_property.ShadeOff()
        
        # Set transfer functions
        color_func = self.create_color_transfer_function()
        opacity_func = self.create_opacity_transfer_function()
        
        volume_property.SetColor(color_func)
        volume_property.SetScalarOpacity(opacity_func)
        
        # Create volume actor
        if self.volume_actor:
            self.renderer.RemoveVolume(self.volume_actor)
            
        self.volume_actor = vtkVolume()
        self.volume_actor.SetMapper(volume_mapper)
        self.volume_actor.SetProperty(volume_property)
        
        self.renderer.AddVolume(self.volume_actor)
        self.renderer.ResetCamera()
        self.ctrl.view_update()
        
    def create_volume_data(self, timestep):
        """Create VTK volume data for a specific timestep"""
        pressure_field = self.pressure_data[timestep].copy()
        
        # Create VTK image data
        image_data = vtkImageData()
        image_data.SetDimensions(self.nx, self.ny, self.nz)
        
        dx = self.metadata.get('dx', 0.001) * 1000  # Convert to mm
        image_data.SetSpacing(dx, dx, dx)
        
        # Convert to VTK array
        flat_pressure = pressure_field.flatten(order='F')
        pressure_array = vtkFloatArray()
        pressure_array.SetNumberOfTuples(flat_pressure.size)
        for i in range(flat_pressure.size):
            pressure_array.SetValue(i, float(flat_pressure[i]))
        pressure_array.SetName("Pressure")
        
        image_data.GetPointData().SetScalars(pressure_array)
        return image_data
        
    def create_color_transfer_function(self):
        """Create color transfer function for pressure"""
        color_func = vtkColorTransferFunction()
        
        if self.pmin < 0 and self.pmax > 0:
            max_abs = max(abs(self.pmin), abs(self.pmax))
            color_func.AddRGBPoint(-max_abs, 0.0, 0.0, 1.0)  # Blue (negative)
            color_func.AddRGBPoint(-max_abs/2, 0.3, 0.3, 1.0)
            color_func.AddRGBPoint(0, 1.0, 1.0, 1.0)  # White (zero)
            color_func.AddRGBPoint(max_abs/2, 1.0, 0.3, 0.3)
            color_func.AddRGBPoint(max_abs, 1.0, 0.0, 0.0)  # Red (positive)
        else:
            color_func.AddRGBPoint(self.pmin, 0.0, 0.0, 1.0)
            color_func.AddRGBPoint((self.pmin + self.pmax)/2, 1.0, 1.0, 1.0)
            color_func.AddRGBPoint(self.pmax, 1.0, 0.0, 0.0)
            
        return color_func
        
    def create_opacity_transfer_function(self):
        """Create opacity transfer function"""
        opacity_func = vtkPiecewiseFunction()
        
        threshold = abs(self.pmax - self.pmin) * 0.1
        opacity_func.AddPoint(self.pmin, 0.3)
        opacity_func.AddPoint(-threshold, 0.1)
        opacity_func.AddPoint(0, 0.0)
        opacity_func.AddPoint(threshold, 0.1)
        opacity_func.AddPoint(self.pmax, 0.3)
        
        return opacity_func
        
    def clear_visualization(self):
        """Clear all actors from the renderer"""
        self.renderer.RemoveAllViewProps()
        
        # Re-add axes
        axes = vtkAxesActor()
        axes.SetTotalLength(10, 10, 10)
        self.renderer.AddActor(axes)
        
    def on_directory_change(self, directory):
        """Handle directory selection change"""
        self.state.selected_dir = directory
        self.load_current_visualization()
        
    def on_mode_change(self, mode):
        """Handle visualization mode change"""
        # Stop any playing animation when switching modes
        if self.state.playing:
            self.state.playing = False
        
        self.state.visualization_mode = mode
        self.load_current_visualization()
        
    def load_current_visualization(self):
        """Load visualization based on current mode and directory"""
        if not self.state.selected_dir:
            return
            
        if self.state.visualization_mode == 'transducer':
            self.load_transducer_data(self.state.selected_dir)
        else:
            self.load_pressure_data(self.state.selected_dir)
            
    def update_timestep(self, timestep):
        """Update pressure visualization timestep"""
        if self.pressure_data is None or self.volume_actor is None:
            return
            
        self.state.current_timestep = int(timestep)
        
        # Update volume data
        image_data = self.create_volume_data(self.state.current_timestep)
        self.volume_actor.GetMapper().SetInputData(image_data)
        
        # Update time display
        if self.time_us is not None:
            self.state.status_message = f"Time: {self.time_us[self.state.current_timestep]:.1f} µs | Frame: {self.state.current_timestep}/{self.state.max_timestep}"
        else:
            self.state.status_message = f"Frame: {self.state.current_timestep}/{self.state.max_timestep}"
        
        self.ctrl.view_update()
        
    def toggle_play(self):
        """Toggle animation playback"""
        self.state.playing = not self.state.playing
        
        # Start the loop once. It will exit by itself when playing=False
        if self.state.playing:
            asynchronous.create_task(self._animate())
            
    async def _animate(self):
        """Advance frames at ~15 FPS while self.state.playing is True"""
        fps = 15
        dt = 1.0 / fps
        
        while self.state.playing and self.pressure_data is not None:
            with self.state:
                # Advance the timestep *inside the state transaction*
                self.state.current_timestep = (
                    self.state.current_timestep + 1
                ) % (self.state.max_timestep + 1)
                
            await asyncio.sleep(dt)  # frame pacing
            
    def reset_camera(self):
        """Reset camera to default view"""
        self.renderer.ResetCamera()
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(45)
        camera.Elevation(30)
        camera.Zoom(1.2)
        self.ctrl.view_update()
        
    def setup_ui(self):
        """Setup the user interface"""
        with SinglePageLayout(self.server) as layout:
            layout.title = "VTK Visualization GUI"
            
            # Toolbar
            with layout.toolbar:
                # Directory selector
                vuetify.VSelect(
                    v_model=("selected_dir",),
                    items=("simulation_dirs",),
                    label="Simulation Directory",
                    dense=True,
                    hide_details=True,
                    style="max-width: 300px; margin-right: 20px",
                    change=(self.on_directory_change, "[$event]")
                )
                
                # Mode selector
                with vuetify.VBtnToggle(
                    v_model=("visualization_mode",),
                    mandatory=True,
                    dense=True,
                    style="margin-right: 20px",
                    change=(self.on_mode_change, "[$event]")
                ):
                    vuetify.VBtn("Transducer", value="transducer")
                    vuetify.VBtn("Pressure", value="pressure")
                
                # Pressure controls (shown only in pressure mode)
                with vuetify.VRow(
                    v_show="visualization_mode === 'pressure'",
                    align="center",
                    no_gutters=True,
                    style="margin-left: 20px"
                ):
                    with vuetify.VBtn(
                        icon=True,
                        click=self.toggle_play,
                        small=True
                    ):
                        vuetify.VIcon("mdi-play", v_if="!playing")
                        vuetify.VIcon("mdi-pause", v_if="playing")
                    
                    vuetify.VSlider(
                        v_model=("current_timestep",),
                        min=0,
                        max=("max_timestep",),
                        step=1,
                        hide_details=True,
                        dense=True,
                        style="max-width: 200px; margin: 0 10px",
                        change=(self.update_timestep, "[$event]")
                    )
                    
                    vuetify.VChip(
                        "{{ current_timestep }}/{{ max_timestep }}",
                        small=True,
                        label=True,
                        outlined=True
                    )
                
                vuetify.VSpacer()
                
                # Camera controls
                vuetify.VBtn(
                    "Reset Camera",
                    click=self.reset_camera,
                    small=True,
                    outlined=True
                )
                
            # Main content
            with layout.content:
                with vuetify.VContainer(fluid=True, classes="pa-0 fill-height"):
                    # Status bar
                    vuetify.VAlert(
                        "{{ status_message }}",
                        type="info",
                        dense=True,
                        text=True,
                        style="margin-bottom: 0"
                    )
                    
                    # Loading overlay
                    with vuetify.VOverlay(v_model=("loading",), opacity=0.8):
                        with vuetify.VContainer(fluid=True, classes="d-flex justify-center align-center fill-height"):
                            with vuetify.VCol(cols="auto", classes="text-center"):
                                vuetify.VProgressCircular(
                                    indeterminate=True,
                                    size=64,
                                    width=8,
                                    color="primary"
                                )
                                vuetify.VCard(
                                    "{{ loading_message }}",
                                    flat=True,
                                    color="transparent",
                                    classes="mt-4 text-h6 white--text"
                    )
                    
                    # VTK view
                    view = vtk_widgets.VtkLocalView(self.render_window)
                    self.ctrl.view_update = view.update
                    
    def start(self, port=3333, host='0.0.0.0'):
        """Start the web server"""
        print(f"\nStarting VTK Visualization GUI")
        print(f"Server: http://{host}:{port}")
        print("\nFeatures:")
        print("- Select simulation directory from dropdown")
        print("- Switch between Transducer and Pressure views")
        print("- Play/pause pressure animations")
        print("- Interactive 3D controls (mouse/touch)")
        print("\nPress Ctrl+C to stop the server\n")
        
        self.server.start(
            port=port,
            host=host,
            open_browser=False,
            show_connection_info=False,
            backend='aiohttp',
            exec_mode='main'
        )


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VTK Visualization GUI')
    parser.add_argument('--port', type=int, default=3333, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    
    args = parser.parse_args()
    
    # Create and start GUI
    gui = VTKVisualizationGUI()
    gui.start(port=args.port, host=args.host)


if __name__ == "__main__":
    main()