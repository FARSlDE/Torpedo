# Torpedo

A simplified k-Wave simulation for focused ultrasound using linear transducer arrays. This code uses kWaveArray to avoid staircasing errors and records the full pressure field for movie visualization.

## Features

- **Linear transducer arrays** with configurable WN×LN element grids
- **Focus-based delays** for beam steering and focusing
- **Staircasing-free** source definition using kWaveArray
- **Full pressure field recording** for movie visualization
- **Automatic fallback** to water simulation if no CT data available
- **3D visualization** support via NIfTI export

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic simulation with default parameters:
```bash
python sim.py
```

### Custom array configuration:
```bash
# 8x8 array with 60mm focal length
python sim.py -wn 8 -ln 8 -focal 60
```

### Options:
- `-wn N`: Number of elements along width (default: 4)
- `-ln N`: Number of elements along length (default: 4)
- `-focal MM`: Focal length in mm (default: 50)
- `-view`: Export results for 3D Slicer viewing

## Simulation Parameters

Key parameters are defined as constants at the top of `sim.py`:

- `PRESSURE_PA`: Source pressure (1 MPa)
- `CENTER_FREQ_HZ`: Center frequency (2.5 MHz)
- `ELEMENT_WIDTH_MM`: Element width (1.6 mm)
- `ELEMENT_LENGTH_MM`: Element length (1.6 mm)
- `WIDTH_PITCH_MM`: Edge-to-edge spacing along width (0.8 mm)
- `LENGTH_PITCH_MM`: Edge-to-edge spacing along length (0.8 mm)

## Simulation Modes

1. **Skull Mode**: If `labels_*.json` files exist with transducer positions and CT data is available
2. **Water Mode**: Automatic fallback for free-field simulations in water (100×100×100 mm grid)

## Output

Results are saved to `data/simulations/YYYYMMDD_HHMMSS/`:

- `pressure_movie.h5`: Full 4D pressure field data
- `config.json`: Simulation configuration
- `simulation_data.npz`: Material properties and arrays
- `grid_*.png`: Cross-section visualizations
- `visualize_movie.py`: Auto-generated script for creating animations

## Visualization

After simulation, run the generated visualization script:
```bash
cd data/simulations/YYYYMMDD_HHMMSS/
python visualize_movie.py
```

This creates animated GIFs and MP4s showing pressure wave propagation in different planes.

## Code Structure

- `sim.py`: Main simulation script
- `utils.py`: Utilities for data loading, conversion, and visualization
- `requirements.txt`: Python dependencies

