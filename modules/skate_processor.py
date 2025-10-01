#!/usr/bin/env python3
"""
Skate Data Processor Module
===========================

This module handles the processing and analysis of skate tracking data.
It loads CSV data, calculates movement metrics, and prepares data for analysis.

Author: BlueCloud Hackathon 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SkateProcessor:
    def __init__(self, csv_path):
        """
        Initialize the Skate Processor
        
        Args:
            csv_path (str): Path to the skate tracking CSV file
        """
        self.csv_path = csv_path
        self.data = None
        self.processed_data = None
        self.stats = {}
    
    def load_data(self):
        """Load skate tracking data from CSV"""
        print("🦈 Loading skate tracking data...")
        
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"✅ Loaded {len(self.data):,} skate tracking records")
            
            # Basic validation
            required_columns = ['id', 'Date', 'Latitude', 'Longitude']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert date column
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            
            # Calculate basic stats
            self.stats['total_records'] = len(self.data)
            self.stats['unique_skates'] = self.data['id'].nunique()
            self.stats['date_range'] = (self.data['Date'].min(), self.data['Date'].max())
            self.stats['spatial_bounds'] = {
                'lat_min': self.data['Latitude'].min(),
                'lat_max': self.data['Latitude'].max(),
                'lon_min': self.data['Longitude'].min(),
                'lon_max': self.data['Longitude'].max()
            }
            
            print(f"📊 Data spans from {self.stats['date_range'][0].strftime('%Y-%m-%d')} to {self.stats['date_range'][1].strftime('%Y-%m-%d')}")
            print(f"🦈 Tracking {self.stats['unique_skates']} individual skates")
            
            return self.data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def add_temporal_features(self):
        """Add temporal features to the data"""
        print("📅 Adding temporal features...")
        
        self.data['year'] = self.data['Date'].dt.year
        self.data['month'] = self.data['Date'].dt.month
        self.data['day_of_year'] = self.data['Date'].dt.dayofyear
        self.data['week'] = self.data['Date'].dt.isocalendar().week
        self.data['day_of_week'] = self.data['Date'].dt.dayofweek
        self.data['hour'] = self.data['Date'].dt.hour
        
        # Add seasonal features
        self.data['season'] = self.data['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        
        print("✅ Temporal features added")
    
    def calculate_movement_metrics(self):
        """Calculate movement metrics for each skate"""
        print("🏃‍♂️ Calculating movement metrics...")
        
        # Sort by skate ID and date
        self.data = self.data.sort_values(['id', 'Date'])
        
        # Calculate differences
        self.data['lat_diff'] = self.data.groupby('id')['Latitude'].diff()
        self.data['lon_diff'] = self.data.groupby('id')['Longitude'].diff()
        
        # Calculate distance (simplified - not accounting for Earth's curvature)
        self.data['distance'] = np.sqrt(
            self.data['lat_diff']**2 + self.data['lon_diff']**2
        )
        
        # Calculate speed (distance per day)
        self.data['time_diff'] = self.data.groupby('id')['Date'].diff().dt.total_seconds() / 86400  # days
        self.data['speed'] = self.data['distance'] / self.data['time_diff'].replace(0, np.nan)
        
        # Calculate direction (bearing)
        self.data['direction'] = np.arctan2(self.data['lon_diff'], self.data['lat_diff']) * 180 / np.pi
        
        # Add cumulative distance
        self.data['cumulative_distance'] = self.data.groupby('id')['distance'].cumsum()
        
        print("✅ Movement metrics calculated")
    
    def add_spatial_features(self):
        """Add spatial analysis features"""
        print("🗺️ Adding spatial features...")
        
        # Calculate center of activity for each skate
        skate_centers = self.data.groupby('id').agg({
            'Latitude': 'mean',
            'Longitude': 'mean'
        }).reset_index()
        skate_centers.columns = ['id', 'center_lat', 'center_lon']
        
        # Merge back to main data
        self.data = self.data.merge(skate_centers, on='id')
        
        # Calculate distance from center
        self.data['dist_from_center'] = np.sqrt(
            (self.data['Latitude'] - self.data['center_lat'])**2 + 
            (self.data['Longitude'] - self.data['center_lon'])**2
        )
        
        # Add spatial bins (for aggregation)
        self.data['lat_bin'] = np.round(self.data['Latitude'], 1)
        self.data['lon_bin'] = np.round(self.data['Longitude'], 1)
        
        print("✅ Spatial features added")
    
    def filter_and_clean(self):
        """Filter and clean the data"""
        print("🧹 Filtering and cleaning data...")
        
        initial_count = len(self.data)
        
        # Remove records with missing coordinates
        self.data = self.data.dropna(subset=['Latitude', 'Longitude'])
        
        # Remove extreme outliers (beyond reasonable bounds)
        self.data = self.data[
            (self.data['Latitude'] >= 40) & (self.data['Latitude'] <= 70) &
            (self.data['Longitude'] >= -20) & (self.data['Longitude'] <= 20)
        ]
        
        # Remove extreme movement speeds (> 10 degrees per day)
        self.data = self.data[self.data['speed'] <= 10]
        
        # Remove records with zero or negative time differences
        self.data = self.data[self.data['time_diff'] > 0]
        
        final_count = len(self.data)
        removed_count = initial_count - final_count
        
        print(f"✅ Cleaned data: removed {removed_count:,} records ({removed_count/initial_count*100:.1f}%)")
        print(f"📊 Final dataset: {final_count:,} records")
    
    def calculate_summary_statistics(self):
        """Calculate summary statistics"""
        print("📈 Calculating summary statistics...")
        
        # Overall statistics
        self.stats['total_distance'] = self.data['distance'].sum()
        self.stats['avg_speed'] = self.data['speed'].mean()
        self.stats['max_speed'] = self.data['speed'].max()
        self.stats['avg_distance_per_day'] = self.data['distance'].mean()
        
        # Per-skate statistics
        skate_stats = self.data.groupby('id').agg({
            'distance': ['sum', 'mean', 'max'],
            'speed': ['mean', 'max'],
            'Date': ['min', 'max', 'count'],
            'dist_from_center': 'max'
        }).round(4)
        
        # Flatten column names
        skate_stats.columns = ['_'.join(col).strip() for col in skate_stats.columns]
        skate_stats = skate_stats.reset_index()
        
        self.stats['skate_summary'] = skate_stats
        
        print("✅ Summary statistics calculated")
    
    def process(self):
        """Run the complete processing pipeline"""
        print("🦈🌊 Starting Skate Data Processing")
        print("=" * 40)
        
        try:
            # Load data
            self.load_data()
            
            # Add features
            self.add_temporal_features()
            self.calculate_movement_metrics()
            self.add_spatial_features()
            
            # Clean data
            self.filter_and_clean()
            
            # Calculate statistics
            self.calculate_summary_statistics()
            
            # Store processed data
            self.processed_data = self.data.copy()
            
            print("\n🎯 Skate Processing Summary:")
            print(f"  📊 Total records: {len(self.processed_data):,}")
            print(f"  🦈 Individual skates: {self.stats['unique_skates']}")
            print(f"  📅 Date range: {self.stats['date_range'][0].strftime('%Y-%m-%d')} to {self.stats['date_range'][1].strftime('%Y-%m-%d')}")
            print(f"  🏃‍♂️ Total distance: {self.stats['total_distance']:.2f} degrees")
            print(f"  ⚡ Average speed: {self.stats['avg_speed']:.4f} degrees/day")
            
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
    """Test the skate processor"""
    import os
    
    # Test with sample data
    csv_path = "/home/samwork/Documents/coding/bluecloud-hackathon-2025/deliverable4/Skates_Track.csv"
    
    if os.path.exists(csv_path):
        processor = SkateProcessor(csv_path)
        processed_data = processor.process()
        
        # Export results
        output_path = "/home/samwork/Documents/coding/bluecloud-hackathon-2025/deliverable4/data/skate_processed.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processor.export_processed_data(output_path)
        
        print("✅ Skate processor test completed successfully!")
    else:
        print(f"❌ Test file not found: {csv_path}")


if __name__ == "__main__":
    main()

