#!/usr/bin/env python3
"""
Plankton Data Processor Module
=============================

This module handles the processing of plankton NetCDF data.
It's designed to work with NetCDF files containing plankton abundance data
and environmental parameters.

Author: BlueCloud Hackathon 2025
"""

import pandas as pd
import numpy as np
import xarray as xr
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PlanktonProcessor:
    def __init__(self, netcdf_dir=None, netcdf_files=None):
        """
        Initialize the Plankton Processor

        Args:
            netcdf_dir (str): Directory containing NetCDF files
            netcdf_files (list): List of specific NetCDF file paths
        """
        self.netcdf_dir = netcdf_dir
        self.netcdf_files = netcdf_files
        self.data = None
        self.processed_data = None
        self.stats = {}
        self.datasets = {}

    def discover_netcdf_files(self):
        """Discover NetCDF files in the directory"""
        print("🔍 Discovering NetCDF files...")

        if self.netcdf_files:
            files = self.netcdf_files
        elif self.netcdf_dir:
            # Look for NetCDF files in directory and subdirectories
            patterns = ['*.nc', '*.nc4', '*.netcdf', '*interp.nc']
            files = []

            # Search in main directory
            for pattern in patterns:
                files.extend(glob.glob(os.path.join(self.netcdf_dir, pattern)))

            # Search in subdirectories (results, temporal, etc.)
            for root, dirs, filenames in os.walk(self.netcdf_dir):
                for pattern in patterns:
                    files.extend(glob.glob(os.path.join(root, pattern)))

            # Remove duplicates
            files = list(set(files))
        else:
            raise ValueError(
                "Either netcdf_dir or netcdf_files must be provided")

        if not files:
            print("⚠️ No NetCDF files found")
            print("   Expected locations:")
            print(f"   - {self.netcdf_dir}/*.nc")
            print(f"   - {self.netcdf_dir}/results/*interp.nc")
            print(f"   - {self.netcdf_dir}/temporal/*/*.nc")
            return []

        print(f"✅ Found {len(files)} NetCDF files:")
        for file in files:
            print(f"  - {os.path.basename(file)}")

        return files

    def load_netcdf_data(self, file_path):
        """Load a single NetCDF file"""
        try:
            print(f"📂 Loading: {os.path.basename(file_path)}")
            ds = xr.open_dataset(file_path)

            # Print basic info about the dataset
            print(f"  Dimensions: {dict(ds.dims)}")
            print(f"  Variables: {list(ds.data_vars.keys())}")
            print(f"  Coordinates: {list(ds.coords.keys())}")

            return ds

        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None

    def extract_plankton_species(self, dataset):
        """Extract plankton species data from dataset"""
        print("🦐 Extracting plankton species data...")

        # Common plankton species variable names
        plankton_keywords = [
            'acartia', 'calanus', 'metridia', 'oithona', 'temora',
            'copepod', 'plankton', 'zooplankton', 'phytoplankton',
            'abundance', 'biomass', 'concentration'
        ]

        plankton_vars = []
        for var_name in dataset.data_vars:
            var_lower = var_name.lower()
            if any(keyword in var_lower for keyword in plankton_keywords):
                plankton_vars.append(var_name)

        print(f"  Found plankton variables: {plankton_vars}")
        return plankton_vars

    def extract_environmental_data(self, dataset):
        """Extract environmental data from dataset"""
        print("🌡️ Extracting environmental data...")

        # Common environmental variable names
        env_keywords = [
            'temperature', 'temp', 'salinity', 'salt', 'nitrate', 'nitrite',
            'phosphate', 'silicate', 'oxygen', 'chlorophyll', 'chl',
            'depth', 'pressure', 'density'
        ]

        env_vars = []
        for var_name in dataset.data_vars:
            var_lower = var_name.lower()
            if any(keyword in var_lower for keyword in env_keywords):
                env_vars.append(var_name)

        print(f"  Found environmental variables: {env_vars}")
        return env_vars

    def convert_to_dataframe(self, dataset, variables=None):
        """Convert NetCDF dataset to pandas DataFrame"""
        print("📊 Converting to DataFrame...")

        if variables is None:
            variables = list(dataset.data_vars.keys())

        # Convert to DataFrame
        df = dataset[variables].to_dataframe().reset_index()

        # Clean up the DataFrame
        df = df.dropna()

        print(f"  DataFrame shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")

        return df

    def process_single_file(self, file_path):
        """Process a single NetCDF file"""
        dataset = self.load_netcdf_data(file_path)
        if dataset is None:
            return None

        # Extract different types of data
        plankton_vars = self.extract_plankton_species(dataset)
        env_vars = self.extract_environmental_data(dataset)

        # Convert to DataFrame
        all_vars = plankton_vars + env_vars
        if all_vars:
            df = self.convert_to_dataframe(dataset, all_vars)
            df['file_source'] = os.path.basename(file_path)
            return df
        else:
            print(f"⚠️ No relevant variables found in {file_path}")
            return None

    def process_all_files(self):
        """Process all NetCDF files"""
        print("🌊 Processing all NetCDF files...")

        files = self.discover_netcdf_files()
        if not files:
            print("❌ No files to process")
            return None

        all_dataframes = []

        for file_path in files:
            df = self.process_single_file(file_path)
            if df is not None:
                all_dataframes.append(df)

        if not all_dataframes:
            print("❌ No data processed successfully")
            return None

        # Combine all DataFrames
        self.data = pd.concat(all_dataframes, ignore_index=True, sort=False)

        print(f"✅ Combined data: {len(self.data):,} records")
        return self.data

    def add_temporal_features(self):
        """Add temporal features to plankton data"""
        print("📅 Adding temporal features...")

        # Look for time-related columns
        time_columns = [
            col for col in self.data.columns if 'time' in col.lower()]

        if time_columns:
            time_col = time_columns[0]  # Use first time column found

            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(self.data[time_col]):
                self.data[time_col] = pd.to_datetime(
                    self.data[time_col], errors='coerce')

            # Add temporal features
            self.data['year'] = self.data[time_col].dt.year
            self.data['month'] = self.data[time_col].dt.month
            self.data['day_of_year'] = self.data[time_col].dt.dayofyear
            self.data['season'] = self.data['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
            })

            print(f"✅ Added temporal features using column: {time_col}")
        else:
            print("⚠️ No time column found - skipping temporal features")

    def add_spatial_features(self):
        """Add spatial features to plankton data"""
        print("🗺️ Adding spatial features...")

        # Look for coordinate columns
        lat_columns = [
            col for col in self.data.columns if 'lat' in col.lower()]
        lon_columns = [
            col for col in self.data.columns if 'lon' in col.lower()]

        if lat_columns and lon_columns:
            lat_col = lat_columns[0]
            lon_col = lon_columns[0]

            # Add spatial bins for aggregation
            self.data['lat_bin'] = np.round(self.data[lat_col], 1)
            self.data['lon_bin'] = np.round(self.data[lon_col], 1)

            # Calculate spatial statistics
            self.stats['spatial_bounds'] = {
                'lat_min': self.data[lat_col].min(),
                'lat_max': self.data[lat_col].max(),
                'lon_min': self.data[lon_col].min(),
                'lon_max': self.data[lon_col].max()
            }

            print(
                f"✅ Added spatial features using columns: {lat_col}, {lon_col}")
        else:
            print("⚠️ No coordinate columns found - skipping spatial features")

    def calculate_summary_statistics(self):
        """Calculate summary statistics"""
        print("📈 Calculating summary statistics...")

        # Basic stats
        self.stats['total_records'] = len(self.data)
        self.stats['unique_files'] = self.data['file_source'].nunique(
        ) if 'file_source' in self.data.columns else 1

        # Plankton abundance stats
        plankton_cols = [col for col in self.data.columns
                         if any(keyword in col.lower() for keyword in
                                ['acartia', 'calanus', 'metridia', 'oithona', 'temora', 'abundance'])]

        if plankton_cols:
            plankton_stats = {}
            for col in plankton_cols:
                if self.data[col].dtype in ['float64', 'int64']:
                    plankton_stats[col] = {
                        'mean': self.data[col].mean(),
                        'std': self.data[col].std(),
                        'min': self.data[col].min(),
                        'max': self.data[col].max(),
                        'count': self.data[col].count()
                    }

            self.stats['plankton_summary'] = plankton_stats
            print(
                f"✅ Calculated statistics for {len(plankton_cols)} plankton variables")

        # Environmental stats
        env_cols = [col for col in self.data.columns
                    if any(keyword in col.lower() for keyword in
                           ['temperature', 'salinity', 'nitrate', 'phosphate', 'silicate'])]

        if env_cols:
            env_stats = {}
            for col in env_cols:
                if self.data[col].dtype in ['float64', 'int64']:
                    env_stats[col] = {
                        'mean': self.data[col].mean(),
                        'std': self.data[col].std(),
                        'min': self.data[col].min(),
                        'max': self.data[col].max()
                    }

            self.stats['environmental_summary'] = env_stats
            print(
                f"✅ Calculated statistics for {len(env_cols)} environmental variables")

    def create_sample_data(self):
        """Create sample data for testing when no NetCDFs are available"""
        print("🧪 Creating sample plankton data for testing...")

        # Create sample data that mimics typical plankton NetCDF structure
        np.random.seed(42)

        # Create spatial grid
        lats = np.linspace(50, 55, 20)  # North Sea region
        lons = np.linspace(1, 5, 20)

        # Create temporal range
        dates = pd.date_range('2021-08-01', '2021-09-30', freq='D')

        # Create sample data
        data = []
        for lat in lats:
            for lon in lons:
                for date in dates:
                    # Add some spatial and temporal patterns
                    lat_factor = np.sin(lat * np.pi / 180)
                    lon_factor = np.cos(lon * np.pi / 180)
                    time_factor = np.sin(date.dayofyear * 2 * np.pi / 365)

                    data.append({
                        'latitude': lat,
                        'longitude': lon,
                        'time': date,
                        'acartia_abundance': np.random.exponential(100) * lat_factor * time_factor,
                        'calanus_abundance': np.random.exponential(200) * lon_factor * time_factor,
                        'metridia_abundance': np.random.exponential(50) * (lat_factor + lon_factor),
                        'temperature': 15 + 5 * np.sin(date.dayofyear * 2 * np.pi / 365) + np.random.normal(0, 1),
                        'salinity': 35 + np.random.normal(0, 0.5),
                        'nitrate': np.random.exponential(5),
                        'file_source': 'sample_data.nc'
                    })

        self.data = pd.DataFrame(data)
        print(f"✅ Created sample data: {len(self.data):,} records")

        return self.data

    def process(self, use_sample_data=False):
        """Run the complete processing pipeline"""
        print("🦐🌊 Starting Plankton Data Processing")
        print("=" * 40)

        try:
            if use_sample_data:
                # Use sample data for testing
                self.create_sample_data()
            else:
                # Process real NetCDF files
                self.process_all_files()

            if self.data is None:
                print("❌ No data to process")
                return None

            # Add features
            self.add_temporal_features()
            self.add_spatial_features()

            # Calculate statistics
            self.calculate_summary_statistics()

            # Store processed data
            self.processed_data = self.data.copy()

            print("\n🎯 Plankton Processing Summary:")
            print(f"  📊 Total records: {len(self.processed_data):,}")
            print(f"  📁 Source files: {self.stats.get('unique_files', 1)}")

            if 'spatial_bounds' in self.stats:
                bounds = self.stats['spatial_bounds']
                print(
                    f"  🗺️ Spatial bounds: Lat {bounds['lat_min']:.1f}-{bounds['lat_max']:.1f}°N, Lon {bounds['lon_min']:.1f}-{bounds['lon_max']:.1f}°E")

            if 'plankton_summary' in self.stats:
                print(
                    f"  🦐 Plankton variables: {len(self.stats['plankton_summary'])}")

            if 'environmental_summary' in self.stats:
                print(
                    f"  🌡️ Environmental variables: {len(self.stats['environmental_summary'])}")

            return self.processed_data

        except Exception as e:
            print(f"❌ Error in processing: {e}")
            raise

    def get_summary(self):
        """Get processing summary"""
        return {
            'data': self.processed_data,
            'stats': self.stats,
            'shape': self.processed_data.shape if self.processed_data is not None else None
        }

    def export_processed_data(self, output_path):
        """Export processed data to CSV"""
        if self.processed_data is not None:
            self.processed_data.to_csv(output_path, index=False)
            print(f"💾 Exported processed data to: {output_path}")
        else:
            print("❌ No processed data to export")


def main():
    """Test the plankton processor"""
    import os

    # Test with sample data
    processor = PlanktonProcessor()
    processed_data = processor.process(use_sample_data=True)

    if processed_data is not None:
        # Export results
        output_path = "/home/samwork/Documents/coding/bluecloud-hackathon-2025/deliverable4/data/plankton_processed.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processor.export_processed_data(output_path)

        print("✅ Plankton processor test completed successfully!")
    else:
        print("❌ Plankton processor test failed")


if __name__ == "__main__":
    main()
