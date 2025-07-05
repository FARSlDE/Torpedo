#!/bin/bash

# Update package list first
apt-get update -y

# Install system dependencies
apt-get install -y vim
apt-get install -y libhdf5-serial-dev
apt-get install -y libxrender1
apt-get install -y libopenblas-dev  # Required for MinkowskiEngine
apt-get install -y ninja-build  # Add ninja for faster builds
apt-get install -y xvfb
apt-get install -y libgl1-mesa-glx libglu1-mesa


# Fix NumPy compatibility issue first
echo "Fixing NumPy compatibility..."
pip uninstall -y numpy
pip install "numpy<2.0"
pip install tqdm
pip install trame>=2.5.0 trame-vuetify>=2.3.0 trame-vtk>=2.5.0
pip install vtk

pip install scipy nibabel matplotlib k-wave-python pydicom pillow pylibjpeg