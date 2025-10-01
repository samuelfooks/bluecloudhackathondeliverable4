#!/usr/bin/env python3
"""
Elasmobranch Processor Module
=============================

This module processes FAO capture data to extract elasmobranch species
and creates coarse raster grids by mapping FAO fishing area codes to spatial regions.

Author: BlueCloud Hackathon 2025
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


class ElasmobranchProcessor:
    def __init__(self, capture_csv_path, species_metadata_path, water_area_path):
        """
        Initialize the Elasmobranch Processor

        Args:
            capture_csv_path (str): Path to Capture_Quantity.csv
            species_metadata_path (str): Path to CL_FI_SPECIES_GROUPS.csv
            water_area_path (str): Path to CL_FI_WATERAREA_GROUPS.csv
        """
        self.capture_csv_path = capture_csv_path
        self.species_metadata_path = species_metadata_path
        self.water_area_path = water_area_path

        self.capture_data = None
        self.species_metadata = None
        self.water_area_metadata = None
        self.elasmobranch_species = None
        self.elasmobranch_data = None
        self.raster_data = None

        # FAO Fishing Area to spatial bounding boxes mapping
        self.fao_area_bounds = {
            # Atlantic Ocean
            '18': {'name': 'Arctic Sea', 'bounds': (60, 85, -180, 180)},
            '21': {'name': 'Atlantic, Northwest', 'bounds': (40, 70, -80, -40)},
            '27': {'name': 'Atlantic, Northeast', 'bounds': (40, 70, -20, 40)},
            '31': {'name': 'Atlantic, Western Central', 'bounds': (0, 40, -80, -20)},
            '34': {'name': 'Atlantic, Eastern Central', 'bounds': (0, 40, -20, 20)},
            '37': {'name': 'Mediterranean and Black Sea', 'bounds': (30, 45, -5, 40)},
            '41': {'name': 'Atlantic, Southwest', 'bounds': (-40, 0, -80, -20)},
            '47': {'name': 'Atlantic, Southeast', 'bounds': (-40, 0, -20, 20)},
            '48': {'name': 'Atlantic, Antarctic', 'bounds': (-80, -40, -80, 20)},

            # Indian Ocean
            '51': {'name': 'Indian Ocean, Western', 'bounds': (-40, 30, 20, 80)},
            '57': {'name': 'Indian Ocean, Eastern', 'bounds': (-40, 30, 80, 140)},
            '58': {'name': 'Indian Ocean, Antarctic', 'bounds': (-80, -40, 20, 140)},

            # Pacific Ocean
            '61': {'name': 'Pacific, Northwest', 'bounds': (20, 60, 120, 180)},
            '67': {'name': 'Pacific, Northeast', 'bounds': (20, 60, -180, -120)},
            '71': {'name': 'Pacific, Western Central', 'bounds': (-20, 20, 120, 180)},
            '77': {'name': 'Pacific, Eastern Central', 'bounds': (-20, 20, -180, -120)},
            '81': {'name': 'Pacific, Southwest', 'bounds': (-60, -20, 120, 180)},
            '87': {'name': 'Pacific, Southeast', 'bounds': (-60, -20, -180, -120)},
            '88': {'name': 'Pacific, Antarctic', 'bounds': (-80, -60, 120, -120)},

            # Antarctic
            '98': {'name': 'Antarctic areas NEI', 'bounds': (-80, -40, -180, 180)},
            '99': {'name': 'Marine areas outside Antarctic', 'bounds': (-40, 80, -180, 180)},
        }

    def load_data(self):
        """Load all required datasets"""
        print("📊 Loading FAO datasets...")

        # Load capture quantity data
        print("  Loading capture quantity data...")
        self.capture_data = pd.read_csv(self.capture_csv_path)
        print(f"    Loaded {len(self.capture_data):,} capture records")

        # Load species metadata
        print("  Loading species metadata...")
        self.species_metadata = pd.read_csv(self.species_metadata_path)
        print(f"    Loaded {len(self.species_metadata):,} species records")

        # Load water area metadata
        print("  Loading water area metadata...")
        self.water_area_metadata = pd.read_csv(self.water_area_path)
        print(
            f"    Loaded {len(self.water_area_metadata):,} water area records")

        print("✅ Data loading complete!")

    def identify_elasmobranch_species(self):
        """Identify elasmobranch species from metadata"""
        print("🦈 Identifying elasmobranch species...")

        # Define elasmobranch keywords for identification
        elasmobranch_keywords = [
            'shark', 'ray', 'skate', 'chimaera', 'elasmobranch',
            'mako', 'thresher', 'basking', 'tiger', 'sixgill',
            'squale', 'requin', 'raie', 'chimère',  # French
            'tiburón', 'raya', 'quimera',  # Spanish
            'акула', 'скат', 'химера',  # Russian
            '鲨鱼', '鳐鱼', '银鲛'  # Chinese
        ]

        # Create a mask for elasmobranch species
        elasmobranch_mask = pd.Series([False] * len(self.species_metadata))

        for keyword in elasmobranch_keywords:
            # Check in English names
            mask_en = self.species_metadata['Name_En'].str.contains(
                keyword, case=False, na=False)
            # Check in French names
            mask_fr = self.species_metadata['Name_Fr'].str.contains(
                keyword, case=False, na=False)
            # Check in Spanish names
            mask_es = self.species_metadata['Name_Es'].str.contains(
                keyword, case=False, na=False)
            # Check in scientific names
            mask_sci = self.species_metadata['Scientific_Name'].str.contains(
                keyword, case=False, na=False)
            # Check in ISSCAAP group
            mask_group = self.species_metadata['ISSCAAP_Group_En'].str.contains(
                keyword, case=False, na=False)

            elasmobranch_mask |= (mask_en | mask_fr |
                                  mask_es | mask_sci | mask_group)

        # Filter elasmobranch species
        self.elasmobranch_species = self.species_metadata[elasmobranch_mask].copy(
        )

        print(f"  Found {len(self.elasmobranch_species)} elasmobranch species")
        print(
            f"  Species codes: {self.elasmobranch_species['3A_Code'].tolist()[:10]}...")

        return self.elasmobranch_species

    def join_capture_with_metadata(self):
        """Join capture data with species and area metadata"""
        print("🔗 Joining capture data with metadata...")

        # Join with species metadata
        merged_data = self.capture_data.merge(
            self.elasmobranch_species[[
                '3A_Code', 'Name_En', 'Scientific_Name', 'ISSCAAP_Group_En']],
            left_on='SPECIES.ALPHA_3_CODE',
            right_on='3A_Code',
            how='inner'
        )

        # Join with water area metadata
        merged_data = merged_data.merge(
            self.water_area_metadata[['Code', 'Name_En',
                                      'Ocean_Group_En', 'InlandMarine_Group_En']],
            left_on='AREA.CODE',
            right_on='Code',
            how='left'
        )

        # Clean up column names after join
        merged_data = merged_data.rename(columns={
            'Name_En_x': 'Name_En',  # Species name
            'Name_En_y': 'Area_Name_En'  # Area name
        })

        # Drop the duplicate Code column
        merged_data = merged_data.drop(columns=['Code'])

        # Filter for marine areas only (exclude inland waters)
        marine_mask = merged_data['InlandMarine_Group_En'].notna()
        self.elasmobranch_data = merged_data[marine_mask].copy()

        print(
            f"  Joined data: {len(self.elasmobranch_data):,} elasmobranch capture records")
        print(
            f"  Time range: {self.elasmobranch_data['PERIOD'].min()} - {self.elasmobranch_data['PERIOD'].max()}")
        print(
            f"  Total catch: {self.elasmobranch_data['VALUE'].sum():,.0f} tonnes")

        return self.elasmobranch_data

    def create_coarse_raster(self, resolution=2.0, target_area=None):
        """
        Create coarse raster grid for elasmobranch distribution

        Args:
            resolution (float): Grid resolution in degrees
            target_area (str): Specific FAO area code to focus on (e.g., '27' for North Sea)
        """
        print(f"🗺️ Creating coarse raster grid (resolution: {resolution}°)...")

        # Filter data for target area if specified
        if target_area:
            filtered_data = self.elasmobranch_data[self.elasmobranch_data['AREA.CODE'] == target_area].copy(
            )
            print(
                f"  Focusing on FAO area {target_area}: {len(filtered_data):,} records")
        else:
            filtered_data = self.elasmobranch_data.copy()

        # Aggregate data by area, species, and period
        area_summary = filtered_data.groupby(['AREA.CODE', 'PERIOD']).agg({
            'VALUE': 'sum',
            'Name_En': 'first',
            'Scientific_Name': 'first'
        }).reset_index()

        # Create global raster grid
        min_lon, max_lon = -180, 180
        min_lat, max_lat = -90, 90

        lons = np.arange(min_lon, max_lon + resolution, resolution)
        lats = np.arange(min_lat, max_lat + resolution, resolution)

        # Initialize grid
        grid_data = np.zeros((len(lats), len(lons)))
        grid_info = []

        # Map catch data to grid cells
        for _, row in area_summary.iterrows():
            area_code = str(row['AREA.CODE'])

            if area_code in self.fao_area_bounds:
                bounds = self.fao_area_bounds[area_code]['bounds']
                lat_min, lat_max, lon_min, lon_max = bounds

                # Find grid cells within this FAO area
                lat_indices = np.where(
                    (lats >= lat_min) & (lats <= lat_max))[0]
                lon_indices = np.where(
                    (lons >= lon_min) & (lons <= lon_max))[0]

                # Distribute catch across all cells in the area
                if len(lat_indices) > 0 and len(lon_indices) > 0:
                    catch_per_cell = row['VALUE'] / \
                        (len(lat_indices) * len(lon_indices))

                    for lat_idx in lat_indices:
                        for lon_idx in lon_indices:
                            grid_data[lat_idx, lon_idx] += catch_per_cell

                            # Store cell information
                            grid_info.append({
                                'lat': lats[lat_idx],
                                'lon': lons[lon_idx],
                                'catch': catch_per_cell,
                                'area_code': area_code,
                                'area_name': self.fao_area_bounds[area_code]['name'],
                                'period': row['PERIOD']
                            })

        # Store raster data
        self.raster_data = {
            'grid': grid_data,
            'lons': lons,
            'lats': lats,
            'resolution': resolution,
            'grid_info': pd.DataFrame(grid_info),
            'area_summary': area_summary
        }

        print(f"  Grid size: {len(lats)} x {len(lons)}")
        print(f"  Non-zero cells: {np.count_nonzero(grid_data)}")
        print(f"  Total catch distributed: {grid_data.sum():,.0f} tonnes")

        return self.raster_data

    def filter_for_north_sea(self):
        """Filter data for North Sea region (Atlantic, Northeast)"""
        print("🌊 Filtering for North Sea region...")

        # North Sea is in FAO area 27 (Atlantic, Northeast)
        north_sea_data = self.elasmobranch_data[self.elasmobranch_data['AREA.CODE'] == 27].copy(
        )

        print(f"  North Sea elasmobranch records: {len(north_sea_data):,}")
        print(
            f"  Time range: {north_sea_data['PERIOD'].min()} - {north_sea_data['PERIOD'].max()}")
        print(f"  Total catch: {north_sea_data['VALUE'].sum():,.0f} tonnes")

        # Show top species
        top_species = north_sea_data.groupby(['3A_Code', 'Name_En']).agg({
            'VALUE': 'sum'
        }).reset_index().sort_values('VALUE', ascending=False).head(10)

        print("  Top elasmobranch species in North Sea:")
        for _, row in top_species.iterrows():
            print(f"    {row['Name_En']}: {row['VALUE']:,.0f} tonnes")

        return north_sea_data

    def export_raster_data(self, output_dir):
        """Export raster data to CSV and other formats"""
        print("💾 Exporting raster data...")

        os.makedirs(output_dir, exist_ok=True)

        if self.raster_data is None:
            print("❌ No raster data to export")
            return None

        # Export grid as CSV
        grid_df = pd.DataFrame(self.raster_data['grid'])
        grid_df.columns = [
            f"lon_{lon:.1f}" for lon in self.raster_data['lons']]
        grid_df.index = [f"lat_{lat:.1f}" for lat in self.raster_data['lats']]

        csv_path = os.path.join(output_dir, 'elasmobranch_raster_grid.csv')
        grid_df.to_csv(csv_path)

        # Export grid info
        info_path = os.path.join(output_dir, 'elasmobranch_grid_info.csv')
        self.raster_data['grid_info'].to_csv(info_path, index=False)

        # Export summary data
        summary_path = os.path.join(output_dir, 'elasmobranch_summary.csv')
        self.elasmobranch_data.to_csv(summary_path, index=False)

        print(f"  ✅ Exported raster grid: {csv_path}")
        print(f"  ✅ Exported grid info: {info_path}")
        print(f"  ✅ Exported summary: {summary_path}")

        return {
            'raster_csv': csv_path,
            'grid_info': info_path,
            'summary': summary_path
        }

    def process(self, target_area='27', resolution=1.0):
        """Run the complete elasmobranch processing pipeline"""
        print("🦈🌊 Starting Elasmobranch Processing")
        print("=" * 45)

        try:
            # Load data
            self.load_data()

            # Identify elasmobranch species
            self.identify_elasmobranch_species()

            # Join data
            self.join_capture_with_metadata()

            # Filter for North Sea (or target area)
            north_sea_data = self.filter_for_north_sea()

            # Create coarse raster
            self.create_coarse_raster(
                resolution=resolution, target_area=target_area)

            print("\n🎯 Elasmobranch Processing Summary:")
            print(
                f"  📊 Total elasmobranch records: {len(self.elasmobranch_data):,}")
            print(f"  🦈 Species identified: {len(self.elasmobranch_species)}")
            print(
                f"  🌍 Total catch: {self.elasmobranch_data['VALUE'].sum():,.0f} tonnes")
            print(
                f"  📅 Time range: {self.elasmobranch_data['PERIOD'].min()} - {self.elasmobranch_data['PERIOD'].max()}")

            if self.raster_data:
                print(f"  🗺️ Raster grid: {self.raster_data['grid'].shape}")
                print(
                    f"  📍 Non-zero cells: {np.count_nonzero(self.raster_data['grid'])}")

            return {
                'elasmobranch_data': self.elasmobranch_data,
                'elasmobranch_species': self.elasmobranch_species,
                'north_sea_data': north_sea_data,
                'raster_data': self.raster_data
            }

        except Exception as e:
            print(f"❌ Error in processing: {e}")
            raise

    def get_summary(self):
        """Get processing summary"""
        # Calculate stats if data is available
        stats = {}
        if self.elasmobranch_data is not None and not self.elasmobranch_data.empty:
            stats['unique_species'] = len(
                self.elasmobranch_species) if self.elasmobranch_species else 0
            stats['unique_areas'] = self.elasmobranch_data['AREA'].nunique(
            ) if 'AREA' in self.elasmobranch_data.columns else 0
            stats['total_catch'] = self.elasmobranch_data['VALUE'].sum(
            ) if 'VALUE' in self.elasmobranch_data.columns else 0
            stats['date_range'] = (self.elasmobranch_data['PERIOD'].min(), self.elasmobranch_data['PERIOD'].max(
            )) if 'PERIOD' in self.elasmobranch_data.columns else (None, None)
        else:
            # Default values when no data is available
            stats['unique_species'] = 0
            stats['unique_areas'] = 0
            stats['total_catch'] = 0
            stats['date_range'] = (None, None)

        return {
            'data': self.elasmobranch_data,
            'stats': stats,
            'elasmobranch_species': self.elasmobranch_species,
            'raster_data': self.raster_data,
            'fao_area_bounds': self.fao_area_bounds
        }


def main():
    """Test the elasmobranch processor"""
    import os

    # Set up paths
    base_dir = "/home/samwork/Documents/coding/bluecloud-hackathon-2025"
    capture_csv = os.path.join(
        base_dir, "Capture_2025.1.0", "Capture_Quantity.csv")
    species_metadata = os.path.join(
        base_dir, "Capture_2025.1.0", "CL_FI_SPECIES_GROUPS.csv")
    water_area_metadata = os.path.join(
        base_dir, "Capture_2025.1.0", "CL_FI_WATERAREA_GROUPS.csv")
    output_dir = os.path.join(base_dir, "deliverable4", "data")

    # Check if files exist
    for file_path in [capture_csv, species_metadata, water_area_metadata]:
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found!")
            return

    # Initialize and run processor
    processor = ElasmobranchProcessor(
        capture_csv_path=capture_csv,
        species_metadata_path=species_metadata,
        water_area_path=water_area_metadata
    )

    results = processor.process(
        target_area='27', resolution=1.0)  # Focus on North Sea

    # Export results
    processor.export_raster_data(output_dir)

    print("\n✅ Elasmobranch processor test completed successfully!")


if __name__ == "__main__":
    main()
