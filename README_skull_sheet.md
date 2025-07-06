# Skull Sheet Feature

This feature adds a 2mm thick skull sheet parallel to the transducer array in your k-Wave ultrasound simulations.

## Usage

To enable the skull sheet, add the `-skull_sheet` flag when running `sim.py`:

```bash
python sim.py -skull_sheet
```

You can combine it with other options:

```bash
python sim.py -skull_sheet -focal 30 -wn 8 -ln 8 -gpu
```

## Parameters

The skull sheet parameters are defined at the top of `sim.py`:

- `SKULL_SHEET_THICKNESS_MM = 2.0` - Thickness of the skull sheet (2mm)
- `SKULL_SHEET_DISTANCE_MM = 1.0` - Distance from transducer array (1mm)
- `SKULL_SHEET_SOUND_SPEED = 2300` - Sound speed in skull tissue (2300 m/s)
- `SKULL_SHEET_DENSITY = 1500` - Density of skull tissue (1500 kg/m³)
- `SKULL_SHEET_ABSORPTION = 6.0` - Absorption coefficient

## Features

- **Parallel to array**: The skull sheet is automatically positioned parallel to the transducer array, regardless of array orientation
- **Realistic properties**: Uses acoustic properties based on actual skull tissue measurements
- **Both simulation modes**: Works in both skull simulation mode (with CT data) and water simulation mode
- **Configurable**: Easy to modify parameters at the top of the file

## Technical Details

The skull sheet is implemented as a planar region with uniform acoustic properties:

1. **Positioning**: Located at a specified distance from the transducer array center
2. **Orientation**: Perpendicular to the array's normal vector (parallel to the array face)
3. **Properties**: Homogeneous skull tissue properties throughout the sheet
4. **Integration**: Seamlessly integrated with existing material property conversion

## Testing

Run the test script to verify the skull sheet functionality:

```bash
python test_skull_sheet.py
```

## Output

When the skull sheet is enabled, you'll see additional output:

```
Skull sheet: Enabled
...
Adding skull sheet to simulation:
  Thickness: 2.0 mm (4 voxels)
  Distance from array: 1.0 mm (2 voxels)
  Array center (voxels): [25. 25. 35.]
  Sheet center (voxels): [25. 25. 37.]
  Normal vector: [0. 0. 1.]
  Skull properties: 2300 m/s, 1500 kg/m³, 6.0 absorption
  Skull sheet voxels: 2500
  Sheet coverage: 1.00% of grid
...
✓ Skull sheet included: 2.0mm thick at 1.0mm distance
```

The skull sheet configuration is also saved in the `config.json` file for each simulation.

## Visualization

The skull sheet will appear in:
- Cross-section visualizations (saved as PNG files)
- VTK visualization (if using the GUI)
- Material property maps
- Movie visualizations

## Applications

This feature is useful for:
- Transcranial ultrasound simulations
- Studying skull aberration effects
- Comparing skull vs. water-only simulations
- Acoustic focusing optimization through skull tissue 