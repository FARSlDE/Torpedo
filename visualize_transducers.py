#!/usr/bin/env python3
"""
VTK visualizer for transducers.npy
Visualizes the transducer mask array saved from k-Wave simulation
"""

import vtk
import numpy as np
import sys
import os
import argparse


def create_transducer_viewer(npy_file, spacing_mm=0.5):
    """Create VTK viewer for transducer mask array
    
    Args:
        npy_file: Path to transducers.npy file
        spacing_mm: Voxel spacing in mm (default 0.5)
    """
    
    # Load numpy array
    print(f"Loading: {npy_file}")
    data = np.load(npy_file)
    
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Data range: {data.min()} to {data.max()}")
    print(f"Active voxels: {data.sum()} ({100*data.sum()/data.size:.2f}%)")
    
    # Convert boolean to uint8
    if data.dtype == bool:
        data = data.astype(np.uint8)
    
    # Get spacing
    spacing = [spacing_mm, spacing_mm, spacing_mm]
    print(f"Voxel spacing: {spacing} mm")
    
    # Create VTK image data
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(data.shape)
    vtk_data.SetSpacing(spacing)
    vtk_data.SetOrigin(0, 0, 0)
    
    # Convert numpy array to VTK array
    flat_data = data.flatten(order='F')
    vtk_array = vtk.vtkUnsignedCharArray()
    vtk_array.SetNumberOfTuples(flat_data.size)
    for i in range(flat_data.size):
        vtk_array.SetValue(i, int(flat_data[i]))
    vtk_data.GetPointData().SetScalars(vtk_array)
    
    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.2, 0.2, 0.3)
    
    # Add lighting
    light = vtk.vtkLight()
    light.SetPosition(100, 100, 100)
    light.SetFocalPoint(0, 0, 0)
    light.SetColor(1.0, 1.0, 1.0)
    light.SetIntensity(1.0)
    renderer.AddLight(light)
    
    # Create transducer surface using marching cubes
    if data.any():
        print("Creating transducer surface...")
        trans_mc = vtk.vtkImageMarchingCubes()
        trans_mc.SetInputData(vtk_data)
        trans_mc.SetValue(0, 0.5)  # Threshold at 0.5
        trans_mc.Update()
        
        # Compute normals
        trans_normals = vtk.vtkPolyDataNormals()
        trans_normals.SetInputConnection(trans_mc.GetOutputPort())
        trans_normals.ComputePointNormalsOn()
        trans_normals.ComputeCellNormalsOff()
        trans_normals.Update()
        
        # Create mapper and actor
        trans_mapper = vtk.vtkPolyDataMapper()
        trans_mapper.SetInputConnection(trans_normals.GetOutputPort())
        
        trans_actor = vtk.vtkActor()
        trans_actor.SetMapper(trans_mapper)
        trans_actor.GetProperty().SetColor(1.0, 0.2, 0.2)  # Red
        trans_actor.GetProperty().SetOpacity(1.0)
        trans_actor.GetProperty().SetSpecular(0.5)
        trans_actor.GetProperty().SetSpecularPower(30)
        trans_actor.GetProperty().SetAmbient(0.3)
        trans_actor.GetProperty().SetDiffuse(0.7)
        
        renderer.AddActor(trans_actor)
        print(f"  Transducer surface points: {trans_normals.GetOutput().GetNumberOfPoints()}")
    else:
        print("No transducer data found!")
    
    # Alternative visualization: show individual voxels as cubes
    if data.sum() < 5000:  # Only for small number of voxels
        print(f"\nAdding individual voxel cubes...")
        non_zero = np.where(data > 0)
        
        for i in range(len(non_zero[0])):
            x, y, z = non_zero[0][i], non_zero[1][i], non_zero[2][i]
            
            cube = vtk.vtkCubeSource()
            cube.SetXLength(spacing[0] * 0.8)  # Slightly smaller for visibility
            cube.SetYLength(spacing[1] * 0.8)
            cube.SetZLength(spacing[2] * 0.8)
            cube.SetCenter(x * spacing[0], y * spacing[1], z * spacing[2])
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(cube.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.8, 0.0)  # Yellow
            actor.GetProperty().SetOpacity(0.8)
            
            renderer.AddActor(actor)
    
    # Add grid outline
    outline = vtk.vtkOutlineFilter()
    outline.SetInputData(vtk_data)
    outline_mapper = vtk.vtkPolyDataMapper()
    outline_mapper.SetInputConnection(outline.GetOutputPort())
    outline_actor = vtk.vtkActor()
    outline_actor.SetMapper(outline_mapper)
    outline_actor.GetProperty().SetColor(1, 1, 1)
    renderer.AddActor(outline_actor)
    
    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1000, 800)
    render_window.SetWindowName(f"Transducer Mask - {os.path.basename(npy_file)}")
    render_window.AddRenderer(renderer)
    
    # Create interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    # Add axes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(20, 20, 20)
    
    axes_widget = vtk.vtkOrientationMarkerWidget()
    axes_widget.SetOrientationMarker(axes)
    axes_widget.SetInteractor(interactor)
    axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
    axes_widget.SetEnabled(1)
    axes_widget.InteractiveOff()
    
    # Set camera
    camera = renderer.GetActiveCamera()
    renderer.ResetCamera()
    camera.Azimuth(45)
    camera.Elevation(30)
    camera.Zoom(1.2)
    renderer.ResetCameraClippingRange()
    
    # Add text
    text = vtk.vtkTextActor()
    text.SetInput(f"Transducer mask: {data.sum()} active voxels\n"
                  f"Grid: {data.shape[0]}×{data.shape[1]}×{data.shape[2]} @ {spacing_mm}mm\n"
                  f"Mouse: Rotate | Shift+Mouse: Pan | Ctrl+Mouse: Zoom | Q: Quit")
    text.GetTextProperty().SetFontSize(14)
    text.GetTextProperty().SetColor(1, 1, 1)
    text.SetPosition(10, 10)
    renderer.AddActor2D(text)
    
    # Analyze spatial distribution
    if data.any():
        indices = np.where(data)
        print(f"\nSpatial distribution:")
        print(f"  X range: {indices[0].min()} - {indices[0].max()} (voxels)")
        print(f"  Y range: {indices[1].min()} - {indices[1].max()} (voxels)")
        print(f"  Z range: {indices[2].min()} - {indices[2].max()} (voxels)")
        print(f"  X extent: {(indices[0].max()-indices[0].min())*spacing_mm:.1f} mm")
        print(f"  Y extent: {(indices[1].max()-indices[1].min())*spacing_mm:.1f} mm")
        print(f"  Z extent: {(indices[2].max()-indices[2].min())*spacing_mm:.1f} mm")
    
    return render_window, interactor


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize transducer mask from k-Wave simulation')
    parser.add_argument('npy_file', nargs='?', default='./transducers.npy',
                       help='Path to transducers.npy file (default: ./transducers.npy)')
    parser.add_argument('-s', '--spacing', type=float, default=0.5,
                       help='Voxel spacing in mm (default: 0.5)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.npy_file):
        # Try to find it in recent simulation directories
        sim_dirs = sorted([d for d in os.listdir('data/simulations') 
                          if os.path.isdir(os.path.join('data/simulations', d))],
                         reverse=True)
        
        for sim_dir in sim_dirs[:5]:  # Check last 5 simulations
            test_path = os.path.join('data/simulations', sim_dir, 'transducers.npy')
            if os.path.exists(test_path):
                args.npy_file = test_path
                print(f"Found transducers.npy in: {sim_dir}")
                break
        else:
            print(f"Error: {args.npy_file} not found")
            print("Run simulation first to generate transducers.npy")
            return
    
    try:
        # Create viewer
        render_window, interactor = create_transducer_viewer(args.npy_file, args.spacing)
        
        # Start interaction
        print("\nStarting 3D viewer...")
        print("Controls:")
        print("  Mouse: Rotate view")
        print("  Shift+Mouse: Pan view")
        print("  Ctrl+Mouse: Zoom")
        print("  Q: Quit")
        
        render_window.Render()
        interactor.Initialize()
        interactor.Start()
        
    except Exception as e:
        print(f"Error creating viewer: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 