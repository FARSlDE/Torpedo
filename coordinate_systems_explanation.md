# Coordinate Systems and Scales in Pressure Visualizations

## 📏 **Scale and Units Summary**

| Aspect | Units | Coordinate System | Origin |
|--------|-------|------------------|---------|
| **Axes Labels** | **Millimeters (mm)** | **Physical coordinates** | **Grid center** |
| **Extent Values** | **Millimeters (mm)** | **Centered at origin** | **Grid center = (0,0,0)** |
| **Internal Data** | Voxels | k-Wave grid indices | Corner = (0,0,0) |
| **Markers** | **Millimeters (mm)** | **Physical coordinates** | **Grid center** |

## 🎯 **Key Answer: The visualizations show MILLIMETERS in a CENTERED coordinate system**

## 📊 **Detailed Breakdown**

### **1. Axis Labels and Extent**
From the visualization code:
```python
# XY view example
extent = [-data.shape[2]/2*self.dx_mm, data.shape[2]/2*self.dx_mm,
          -data.shape[1]/2*self.dx_mm, data.shape[1]/2*self.dx_mm]
xlabel, ylabel = 'X [mm]', 'Y [mm]'
```

**What this means:**
- **Units**: Millimeters (mm) 
- **Origin**: Center of the grid
- **Range**: From -half_grid_size to +half_grid_size
- **Example**: For 50x50x100 grid with 0.125mm spacing:
  - X-axis: -3.125mm to +3.125mm (50 × 0.125 ÷ 2)
  - Y-axis: -3.125mm to +3.125mm 
  - Z-axis: -6.25mm to +6.25mm (100 × 0.125 ÷ 2)

### **2. k-Wave vs Visualization Coordinates**

#### **k-Wave Internal System:**
- **Origin**: Grid corner at (0, 0, 0) voxels
- **Units**: Voxel indices
- **Range**: 0 to grid_size-1 in each dimension
- **Storage**: Fortran order (Z, Y, X)

#### **Visualization System:**
- **Origin**: Grid center at (0, 0, 0) mm
- **Units**: Physical millimeters
- **Range**: -half_size to +half_size mm
- **Conversion**: `mm = (voxel - grid_center) × voxel_spacing`

### **3. Coordinate Conversion Examples**

For a 50×50×100 grid with 0.125mm spacing:

```python
# Grid properties
grid_shape = (50, 50, 100)  # (Z, Y, X) in k-Wave
dx_mm = 0.125  # millimeters per voxel
grid_center_voxels = (25, 25, 50)  # Center in voxel coordinates

# Convert voxel to physical coordinates
def voxel_to_mm(voxel_coords):
    offset_voxels = voxel_coords - grid_center_voxels
    return offset_voxels * dx_mm

# Examples:
voxel_to_mm([0, 0, 0])     # → [-3.125, -3.125, -6.25] mm (corner)
voxel_to_mm([25, 25, 50])  # → [0, 0, 0] mm (center)
voxel_to_mm([49, 49, 99])  # → [+3.0, +3.0, +6.125] mm (opposite corner)
```

### **4. Axis Orientations**

The coordinate system follows standard conventions:

```python
if self.current_view == 'xy':
    # Looking down Z-axis
    xlabel, ylabel = 'X [mm]', 'Y [mm]'
    # X = left-right, Y = front-back
    
elif self.current_view == 'xz':
    # Looking along Y-axis  
    xlabel, ylabel = 'X [mm]', 'Z [mm]'
    # X = left-right, Z = up-down
    
else:  # yz view
    # Looking along X-axis
    xlabel, ylabel = 'Y [mm]', 'Z [mm]'
    # Y = front-back, Z = up-down
```

### **5. Marker Positioning**

Reference markers (transducer, focal points) use the same coordinate system:

```python
# Convert marker from k-Wave to visualization coordinates
def _get_marker_position(self, voxel_coords):
    data = self.pressure_3d
    
    if self.current_view == 'xy':
        x_mm = (voxel_coords[2] - data.shape[2]/2) * self.dx_mm
        y_mm = (voxel_coords[1] - data.shape[1]/2) * self.dx_mm
        return (x_mm, y_mm)
```

**Result**: Markers appear at their correct physical positions in millimeters.

## 🔄 **Data Flow Summary**

1. **k-Wave Simulation**: 
   - Runs in voxel coordinates (0 to N-1)
   - Uses grid center as origin for physics calculations

2. **Data Storage**: 
   - Pressure data stored as 4D array: (time, Z, Y, X)
   - Voxel indices with Fortran ordering

3. **Visualization Conversion**:
   - Converts voxel indices to physical millimeters
   - Centers coordinate system at grid center
   - Displays with proper axis labels and extents

## 📐 **Practical Implications**

### **For Your 50×50×100 Test Grid:**
- **Total physical size**: 6.25 × 6.25 × 12.5 mm
- **Voxel spacing**: 0.125 mm
- **Display range**: 
  - X: -3.125 to +3.125 mm
  - Y: -3.125 to +3.125 mm  
  - Z: -6.25 to +6.25 mm

### **Reading the Plots:**
- **Zero point (0,0)**: Physical center of the simulation grid
- **Positive X**: Right side of the plot
- **Positive Y**: Top of the plot (in XY view)
- **Positive Z**: Deeper into the medium (in XZ view)

### **Marker Interpretation:**
- **Cyan Square (Transducer)**: Shows physical location in mm
- **Lime Triangle (Intended Focus)**: Target location in mm
- **Red Circle (Actual Focus)**: Measured location in mm
- **Distance measurements**: All in millimeters from grid center

## ⚠️ **Important Notes**

1. **NOT in voxels**: The axis labels show millimeters, not voxel indices
2. **NOT absolute coordinates**: Origin is at grid center, not world coordinates  
3. **Consistent scaling**: All measurements (FWHM, distances) are in millimeters
4. **k-Wave compatible**: Matches k-Wave's centered coordinate convention

This coordinate system makes it easy to:
- Measure distances in real physical units (mm)
- Compare with experimental setups
- Understand the scale of focal spots and beam patterns
- Relate to medical imaging coordinates 