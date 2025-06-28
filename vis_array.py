#!/usr/bin/env python3
"""
Simple VTK visualizer for skull_and_transducers.nii.gz
"""

import vtk
import numpy as np
import nibabel as nib
import sys
import os

def create_nifti_viewer(nifti_file):
    """Create VTK viewer for NIfTI file with skull and transducers"""
    
    # Load NIfTI file
    print(f"Loading: {nifti_file}")
    nii_img = nib.load(nifti_file)
    data = nii_img.get_fdata().astype(np.uint8)
    affine = nii_img.affine
    
    print(f"Data shape: {data.shape}")
    print(f"Data range: {data.min()} to {data.max()}")
    print(f"Unique labels: {np.unique(data)}")
    print(f"Background voxels (label 0): {(data == 0).sum()}")
    print(f"Skull voxels (label 1): {(data == 1).sum()}")
    print(f"Transducer voxels (label 2): {(data == 2).sum()}")
    
    # Get voxel spacing from affine
    spacing = np.abs(np.diag(affine)[:3])
    print(f"Voxel spacing: {spacing} mm")
    
    # Create VTK image data
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(data.shape)
    vtk_data.SetSpacing(spacing)
    vtk_data.SetOrigin(0, 0, 0)
    
    # Convert numpy array to VTK array (proper way)
    flat_data = data.flatten(order='F')
    vtk_array = vtk.vtkUnsignedCharArray()
    vtk_array.SetNumberOfTuples(flat_data.size)
    for i in range(flat_data.size):
        vtk_array.SetValue(i, int(flat_data[i]))
    vtk_data.GetPointData().SetScalars(vtk_array)
    
    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.2, 0.2, 0.3)  # Light gray background
    
    # Add lighting
    light = vtk.vtkLight()
    light.SetPosition(100, 100, 100)
    light.SetFocalPoint(0, 0, 0)
    light.SetColor(1.0, 1.0, 1.0)
    light.SetIntensity(1.0)
    renderer.AddLight(light)
    
    # Create skull surface (label = 1) using marching cubes
    if (data == 1).any():
        print("Creating skull surface...")
        skull_mc = vtk.vtkImageMarchingCubes()
        skull_mc.SetInputData(vtk_data)
        skull_mc.SetValue(0, 0.9)  # Threshold between 0 and 1
        skull_mc.Update()
        
        # Smooth the surface
        skull_smooth = vtk.vtkSmoothPolyDataFilter()
        skull_smooth.SetInputConnection(skull_mc.GetOutputPort())
        skull_smooth.SetNumberOfIterations(10)
        skull_smooth.Update()
        
        # Compute normals
        skull_normals = vtk.vtkPolyDataNormals()
        skull_normals.SetInputConnection(skull_smooth.GetOutputPort())
        skull_normals.ComputePointNormalsOn()
        skull_normals.ComputeCellNormalsOff()
        skull_normals.Update()
        
        # Create skull mapper and actor
        skull_mapper = vtk.vtkPolyDataMapper()
        skull_mapper.SetInputConnection(skull_normals.GetOutputPort())
        
        skull_actor = vtk.vtkActor()
        skull_actor.SetMapper(skull_mapper)
        skull_actor.GetProperty().SetColor(0.9, 0.9, 0.8)  # Bone white
        skull_actor.GetProperty().SetOpacity(0.7)
        skull_actor.GetProperty().SetSpecular(0.3)
        skull_actor.GetProperty().SetSpecularPower(20)
        skull_actor.GetProperty().SetAmbient(0.2)
        skull_actor.GetProperty().SetDiffuse(0.8)
        
        renderer.AddActor(skull_actor)
        print(f"  Skull surface points: {skull_normals.GetOutput().GetNumberOfPoints()}")
    else:
        print("No skull data found!")
    
    # Create transducer surface (label = 2)
    if (data == 2).any():
        print("Creating transducer surface...")
        trans_mc = vtk.vtkImageMarchingCubes()
        trans_mc.SetInputData(vtk_data)
        trans_mc.SetValue(0, 1.9)  # Threshold between 1 and 2
        trans_mc.Update()
        
        # Compute normals
        trans_normals = vtk.vtkPolyDataNormals()
        trans_normals.SetInputConnection(trans_mc.GetOutputPort())
        trans_normals.ComputePointNormalsOn()
        trans_normals.ComputeCellNormalsOff()
        trans_normals.Update()
        
        # Create transducer mapper and actor
        trans_mapper = vtk.vtkPolyDataMapper()
        trans_mapper.SetInputConnection(trans_normals.GetOutputPort())
        
        trans_actor = vtk.vtkActor()
        trans_actor.SetMapper(trans_mapper)
        trans_actor.GetProperty().SetColor(1.0, 0.2, 0.2)  # Bright red
        trans_actor.GetProperty().SetOpacity(1.0)
        trans_actor.GetProperty().SetSpecular(0.5)
        trans_actor.GetProperty().SetSpecularPower(30)
        trans_actor.GetProperty().SetAmbient(0.3)
        trans_actor.GetProperty().SetDiffuse(0.7)
        
        renderer.AddActor(trans_actor)
        print(f"  Transducer surface points: {trans_normals.GetOutput().GetNumberOfPoints()}")
    else:
        print("No transducer data found!")
    
    # Alternative: Show as cubes if surfaces are empty
    if not (data == 1).any() and not (data == 2).any():
        print("No label data found - showing all non-zero voxels...")
        
        # Show all non-zero as cubes
        cube_source = vtk.vtkCubeSource()
        cube_source.SetXLength(spacing[0])
        cube_source.SetYLength(spacing[1]) 
        cube_source.SetZLength(spacing[2])
        
        # Find all non-zero voxels
        non_zero = np.where(data > 0)
        print(f"Found {len(non_zero[0])} non-zero voxels")
        
        for i in range(min(1000, len(non_zero[0]))):  # Limit to 1000 cubes for performance
            x, y, z = non_zero[0][i], non_zero[1][i], non_zero[2][i]
            value = data[x, y, z]
            
            cube = vtk.vtkCubeSource()
            cube.SetXLength(spacing[0])
            cube.SetYLength(spacing[1])
            cube.SetZLength(spacing[2])
            cube.SetCenter(x * spacing[0], y * spacing[1], z * spacing[2])
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(cube.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            if value == 1:  # Skull
                actor.GetProperty().SetColor(0.9, 0.9, 0.8)
            elif value == 2:  # Transducers
                actor.GetProperty().SetColor(1.0, 0.2, 0.2)
            else:
                actor.GetProperty().SetColor(0.5, 0.5, 0.5)
            
            renderer.AddActor(actor)
    
    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1000, 800)
    render_window.SetWindowName(f"Skull and Transducers - {os.path.basename(nifti_file)}")
    render_window.AddRenderer(renderer)
    
    # Create interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    
    # Add axes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(20, 20, 20)  # 20mm axes
    
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
    
    # Add text with more info
    text = vtk.vtkTextActor()
    skull_count = (data == 1).sum()
    trans_count = (data == 2).sum()
    text.SetInput(f"Bone-colored: Skull ({skull_count} voxels) | Red: Transducers ({trans_count} voxels)\n"
                  f"Mouse: Rotate | Shift+Mouse: Pan | Ctrl+Mouse: Zoom | Q: Quit")
    text.GetTextProperty().SetFontSize(14)
    text.GetTextProperty().SetColor(1, 1, 1)
    text.SetPosition(10, 10)
    renderer.AddActor2D(text)
    
    return render_window, interactor

def main():
    """Main function"""
    
    nifti_file = './skull_and_transducers.nii.gz'
    
    try:
        # Create viewer
        render_window, interactor = create_nifti_viewer(nifti_file)
        
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
