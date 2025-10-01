#!/usr/bin/env python3
"""
Mini Pipeline Runner
====================

Simple runner script for the BlueCloud Mini Pipeline.
This script orchestrates the complete analysis workflow.

Author: BlueCloud Hackathon 2025
"""

import sys
import os
from pipeline.main_pipeline import MainPipeline


def main():
    print("🌊🦈 BlueCloud Mini Pipeline Runner")
    print("=" * 40)

    # Configuration
    base_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(base_dir)

    config = {

        'plankton_netcdf_dir': os.path.join(base_dir, 'data/planktonoutputs'),
        'use_sample_plankton': True,  # Use sample data for testing
        'output_dir': os.path.join(base_dir, 'data'),
        'skate_csv_path': os.path.join(base_dir, 'data/Skates_Track.csv'),
        'model_type': 'random_forest',
        'random_state': 42,
        # Elasmobranch data paths
        'capture_csv_path': os.path.join(project_root, 'Capture_2025.1.0', 'Capture_Quantity.csv'),
        'species_metadata_path': os.path.join(project_root, 'Capture_2025.1.0', 'CL_FI_SPECIES_GROUPS.csv'),
        'water_area_path': os.path.join(project_root, 'Capture_2025.1.0', 'CL_FI_WATERAREA_GROUPS.csv')
    }

    # Check if skate data exists
    if not os.path.exists(config['skate_csv_path']):
        print(f"❌ Error: Skate data not found at {config['skate_csv_path']}")
        print("Please ensure Skates_Track.csv is in the deliverable4 directory.")
        sys.exit(1)

    print("✅ Skate data found!")

    # Check if plankton data directory exists
    if not os.path.exists(config['plankton_netcdf_dir']):
        print(
            f"⚠️  Warning: Plankton data directory not found at {config['plankton_netcdf_dir']}")
        print("   The pipeline will use sample plankton data for testing.")
        print("   To use real data, download NetCDF files from BlueCloud VLab and place them in the planktonoutputs/ directory.")
    else:
        # Check if directory has NetCDF files
        netcdf_files = [f for f in os.listdir(
            config['plankton_netcdf_dir']) if f.endswith('.nc')]
        if netcdf_files:
            print(
                f"✅ Found {len(netcdf_files)} NetCDF files in planktonoutputs/")
        else:
            print("⚠️  Warning: No NetCDF files found in planktonoutputs/ directory")
            print("   The pipeline will use sample plankton data for testing.")

    print(f"📁 Output directory: {config['output_dir']}")

    # Initialize and run pipeline
    try:
        pipeline = MainPipeline(config)
        results = pipeline.run_full_pipeline()

        print("\n🎉 Pipeline completed successfully!")
        print("Check the 'data' directory for all outputs.")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
