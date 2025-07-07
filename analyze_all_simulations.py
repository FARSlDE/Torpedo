#!/usr/bin/env python3
"""
Script to run corrected focal analysis on all existing simulation directories
"""

import os
import glob
from pathlib import Path
import subprocess
import sys

def find_simulation_directories(base_dir="data/simulations"):
    """Find all simulation directories"""
    sim_dirs = []
    
    # Look for directories with timestamp pattern
    pattern = os.path.join(base_dir, "20*")
    for dir_path in glob.glob(pattern):
        if os.path.isdir(dir_path):
            # Check if it has the required files
            config_file = os.path.join(dir_path, "config.json")
            output_file = os.path.join(dir_path, "output.h5")
            
            if os.path.exists(config_file) and os.path.exists(output_file):
                sim_dirs.append(dir_path)
    
    return sorted(sim_dirs)

def run_beam_pattern_analysis(sim_dir):
    """Run beam pattern analysis on a simulation directory"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {sim_dir}")
    print('='*60)
    
    try:
        # Run the beam pattern visualization
        result = subprocess.run([
            sys.executable, "beam_pattern_viz.py", sim_dir
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✓ Successfully analyzed {sim_dir}")
            return True
        else:
            print(f"✗ Error analyzing {sim_dir}")
            print(f"  Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout analyzing {sim_dir}")
        return False
    except Exception as e:
        print(f"✗ Exception analyzing {sim_dir}: {e}")
        return False

def main():
    """Main function"""
    print("Corrected Focal Analysis for All Simulations")
    print("=" * 60)
    
    # Find all simulation directories
    sim_dirs = find_simulation_directories()
    
    if not sim_dirs:
        print("No simulation directories found!")
        return
    
    print(f"Found {len(sim_dirs)} simulation directories:")
    for sim_dir in sim_dirs:
        print(f"  - {sim_dir}")
    
    # Process each directory
    successful = 0
    failed = 0
    
    for sim_dir in sim_dirs:
        if run_beam_pattern_analysis(sim_dir):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Total directories: {len(sim_dirs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        print(f"\n✓ Corrected focal analysis completed for {successful} simulations")
        print("✓ Check the 'beam_pattern_viz' subdirectories for results")

if __name__ == "__main__":
    main() 