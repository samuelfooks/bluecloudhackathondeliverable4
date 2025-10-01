#!/usr/bin/env python3
"""
Visualizer Module
================

This module creates comprehensive visualizations including interactive maps,
charts, and dashboards for the skate-plankton ecosystem analysis.

Author: BlueCloud Hackathon 2025
"""

from .png_processor import PNGProcessor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium import plugins
import os
import warnings
warnings.filterwarnings('ignore')

# Import PNG processor


class Visualizer:
    def __init__(self, output_dir='outputs'):
        """
        Initialize the Visualizer

        Args:
            output_dir (str): Directory to save visualization outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Initialize PNG processor
        self.png_processor = PNGProcessor()
        self.png_data = None

    def create_skate_movement_map(self, skate_data, output_file='skate_movement_map.html'):
        """Create interactive map showing skate movement patterns"""
        print("🗺️ Creating skate movement map...")

        # Calculate map center
        center_lat = skate_data['Latitude'].mean()
        center_lon = skate_data['Longitude'].mean()

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='OpenStreetMap'
        )

        # Sample data for performance (every 3rd point)
        sample_data = skate_data.iloc[::3].copy()

        # Create color map for different skates
        unique_skates = sample_data['id'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_skates)))

        # Add skate tracks
        # Limit to first 10 skates
        for i, skate_id in enumerate(unique_skates[:10]):
            skate_points = sample_data[sample_data['id'] == skate_id]

            if len(skate_points) > 1:
                # Create line
                folium.PolyLine(
                    locations=skate_points[['Latitude', 'Longitude']].values,
                    color=f'#{int(colors[i][0]*255):02x}{int(colors[i][1]*255):02x}{int(colors[i][2]*255):02x}',
                    weight=3,
                    opacity=0.7,
                    popup=f'Skate {skate_id}'
                ).add_to(m)

                # Add start marker
                folium.CircleMarker(
                    location=[skate_points.iloc[0]['Latitude'],
                              skate_points.iloc[0]['Longitude']],
                    radius=5,
                    color='green',
                    fill=True,
                    popup=f'Start - Skate {skate_id}'
                ).add_to(m)

                # Add end marker
                folium.CircleMarker(
                    location=[skate_points.iloc[-1]['Latitude'],
                              skate_points.iloc[-1]['Longitude']],
                    radius=5,
                    color='red',
                    fill=True,
                    popup=f'End - Skate {skate_id}'
                ).add_to(m)

        # Add heatmap
        heat_data = [[row['Latitude'], row['Longitude']]
                     for idx, row in sample_data.iterrows()]
        plugins.HeatMap(heat_data, name='Skate Density').add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Save map
        map_path = os.path.join(self.output_dir, output_file)
        m.save(map_path)

        print(f"✅ Skate movement map saved to: {map_path}")
        return map_path

    def create_plankton_distribution_map(self, plankton_data, output_file='plankton_distribution_map.html'):
        """Create interactive map showing plankton distribution"""
        print("🦐 Creating plankton distribution map...")

        if plankton_data is None or 'latitude' not in plankton_data.columns:
            print("⚠️ No plankton data available for mapping")
            return None

        # Calculate map center
        center_lat = plankton_data['latitude'].mean()
        center_lon = plankton_data['longitude'].mean()

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='OpenStreetMap'
        )

        # Find plankton abundance columns
        abundance_cols = [
            col for col in plankton_data.columns if 'abundance' in col.lower()]

        if abundance_cols:
            # Sample data for performance
            sample_data = plankton_data.sample(min(1000, len(plankton_data)))

            # Create markers for each plankton species
            # Limit to first 5 species
            for i, col in enumerate(abundance_cols[:5]):
                # Filter out zero values
                species_data = sample_data[sample_data[col] > 0]

                if len(species_data) > 0:
                    # Create color scale
                    max_val = species_data[col].max()
                    colors = plt.cm.viridis(species_data[col] / max_val)

                    # Add markers
                    for i, (idx, row) in enumerate(species_data.iterrows()):
                        folium.CircleMarker(
                            location=[row['latitude'], row['longitude']],
                            radius=5,
                            color=f'#{int(colors[i][0]*255):02x}{int(colors[i][1]*255):02x}{int(colors[i][2]*255):02x}',
                            fill=True,
                            opacity=0.7,
                            popup=f'{col}: {row[col]:.2f}'
                        ).add_to(m)

        # Save map
        map_path = os.path.join(self.output_dir, output_file)
        m.save(map_path)

        print(f"✅ Plankton distribution map saved to: {map_path}")
        return map_path

    def create_time_series_plots(self, skate_data, plankton_data=None, output_file='time_series_analysis.png'):
        """Create time series analysis plots"""
        print("📈 Creating time series plots...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Skate movement over time
        daily_movement = skate_data.groupby('Date').agg({
            'distance': 'mean',
            'speed': 'mean',
            'Latitude': 'mean',
            'Longitude': 'mean'
        }).reset_index()

        axes[0, 0].plot(daily_movement['Date'], daily_movement['distance'],
                        'b-', linewidth=2, label='Distance')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Average Distance')
        axes[0, 0].set_title('Skate Movement Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Skate speed over time
        axes[0, 1].plot(daily_movement['Date'],
                        daily_movement['speed'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Average Speed')
        axes[0, 1].set_title('Skate Speed Over Time')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Spatial distribution over time
        axes[1, 0].scatter(daily_movement['Longitude'], daily_movement['Latitude'],
                           c=daily_movement['Date'].dt.dayofyear, cmap='viridis', s=50)
        axes[1, 0].set_xlabel('Longitude')
        axes[1, 0].set_ylabel('Latitude')
        axes[1, 0].set_title('Skate Spatial Distribution Over Time')

        # 4. Plankton abundance over time (if available)
        if plankton_data is not None and 'time' in plankton_data.columns:
            plankton_daily = plankton_data.groupby('time').agg({
                col: 'mean' for col in plankton_data.columns if 'abundance' in col.lower()
            }).reset_index()

            if len(plankton_daily.columns) > 1:
                abundance_cols = [
                    col for col in plankton_daily.columns if 'abundance' in col.lower()]
                for col in abundance_cols[:3]:  # Plot first 3 species
                    axes[1, 1].plot(plankton_daily['time'], plankton_daily[col],
                                    label=col.replace('_abundance', ''), linewidth=2)
                axes[1, 1].set_xlabel('Date')
                axes[1, 1].set_ylabel('Abundance')
                axes[1, 1].set_title('Plankton Abundance Over Time')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        else:
            # Show skate count by month
            monthly_counts = skate_data.groupby('month').size()
            axes[1, 1].bar(monthly_counts.index, monthly_counts.values)
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Number of Records')
            axes[1, 1].set_title('Skate Records by Month')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, output_file)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Time series plots saved to: {plot_path}")
        return plot_path

    def create_correlation_heatmap(self, skate_data, plankton_data=None, output_file='correlation_heatmap.png'):
        """Create correlation heatmap between variables"""
        print("🔥 Creating correlation heatmap...")

        # Prepare data for correlation analysis
        correlation_data = skate_data[[
            'Latitude', 'Longitude', 'month', 'day_of_year', 'distance', 'speed']].copy()

        # Add temporal features
        correlation_data['sin_month'] = np.sin(
            2 * np.pi * correlation_data['month'] / 12)
        correlation_data['cos_month'] = np.cos(
            2 * np.pi * correlation_data['month'] / 12)

        # Add plankton data if available
        if plankton_data is not None:
            abundance_cols = [
                col for col in plankton_data.columns if 'abundance' in col.lower()]
            if abundance_cols:
                # Sample plankton data to match skate data size
                plankton_sample = plankton_data.sample(
                    min(len(skate_data), len(plankton_data)))
                for col in abundance_cols[:5]:  # Limit to first 5 species
                    correlation_data[col] = plankton_sample[col].values[:len(
                        correlation_data)]

        # Calculate correlation matrix
        corr_matrix = correlation_data.corr()

        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})

        plt.title('Correlation Matrix: Skate Movement and Plankton Data')
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, output_file)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Correlation heatmap saved to: {plot_path}")
        return plot_path

    def create_dashboard(self, skate_data, plankton_data=None, model=None, output_file='dashboard.html'):
        """Create comprehensive interactive dashboard"""
        print("📊 Creating interactive dashboard...")

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Skate Movement Patterns', 'Plankton Distribution',
                            'Movement Over Time', 'Model Performance'),
            specs=[[{"type": "scattergeo"}, {"type": "scattergeo"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )

        # 1. Skate movement patterns (simplified)
        sample_skates = skate_data['id'].unique()[:5]
        for skate_id in sample_skates:
            skate_data_subset = skate_data[skate_data['id'] == skate_id]
            fig.add_trace(
                go.Scattergeo(
                    lon=skate_data_subset['Longitude'],
                    lat=skate_data_subset['Latitude'],
                    mode='lines+markers',
                    name=f'Skate {skate_id}',
                    line=dict(width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )

        # 2. Plankton distribution (if available)
        if plankton_data is not None and 'latitude' in plankton_data.columns:
            abundance_cols = [
                col for col in plankton_data.columns if 'abundance' in col.lower()]
            if abundance_cols:
                # Sample data for performance
                sample_data = plankton_data.sample(
                    min(500, len(plankton_data)))
                col = abundance_cols[0]  # Use first abundance column

                fig.add_trace(
                    go.Scattergeo(
                        lon=sample_data['longitude'],
                        lat=sample_data['latitude'],
                        mode='markers',
                        name=f'{col}',
                        marker=dict(
                            size=sample_data[col]/sample_data[col].max()*10,
                            color=sample_data[col],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title=f"{col}")
                        )
                    ),
                    row=1, col=2
                )

        # 3. Movement over time
        daily_movement = skate_data.groupby('Date').agg({
            'distance': 'mean',
            'speed': 'mean'
        }).reset_index()

        fig.add_trace(
            go.Scatter(
                x=daily_movement['Date'],
                y=daily_movement['distance'],
                mode='lines+markers',
                name='Average Distance',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )

        # 4. Model performance (if available)
        if model is not None and hasattr(model, 'metrics'):
            # Create dummy data for model performance visualization
            metrics_data = pd.DataFrame({
                'Metric': ['R² Score', 'RMSE', 'MAE'],
                'Value': [model.metrics['r2'], model.metrics['rmse'], model.metrics['mae']]
            })

            fig.add_trace(
                go.Bar(
                    x=metrics_data['Metric'],
                    y=metrics_data['Value'],
                    name='Model Metrics',
                    marker_color=['green', 'orange', 'red']
                ),
                row=2, col=2
            )
        else:
            # Show skate count by month
            monthly_counts = skate_data.groupby('month').size()
            fig.add_trace(
                go.Bar(
                    x=monthly_counts.index,
                    y=monthly_counts.values,
                    name='Records by Month',
                    marker_color='lightblue'
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="Skate-Plankton Ecosystem Analysis Dashboard",
            showlegend=True
        )

        # Update geo subplots
        fig.update_geos(
            showcoastlines=True,
            coastlinecolor="Black",
            showland=True,
            landcolor="lightgray",
            showocean=True,
            oceancolor="lightblue"
        )

        # Save dashboard
        dashboard_path = os.path.join(self.output_dir, output_file)
        fig.write_html(dashboard_path)

        print(f"✅ Interactive dashboard saved to: {dashboard_path}")
        return dashboard_path

    def create_summary_report(self, skate_data, plankton_data=None, model=None, output_file='summary_report.png'):
        """Create summary report visualization"""
        print("📋 Creating summary report...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Data overview
        # Prepare plankton data info
        if plankton_data is not None:
            plankton_records = f"{len(plankton_data):,}"
            plankton_vars = len(
                [col for col in plankton_data.columns if 'abundance' in col.lower()])
        else:
            plankton_records = 'N/A'
            plankton_vars = 'N/A'

        overview_text = f"""
        Data Overview:
        
        Skate Data:
        • Total Records: {len(skate_data):,}
        • Individual Skates: {skate_data['id'].nunique()}
        • Date Range: {skate_data['Date'].min().strftime('%Y-%m-%d')} to {skate_data['Date'].max().strftime('%Y-%m-%d')}
        • Spatial Range: Lat {skate_data['Latitude'].min():.2f}-{skate_data['Latitude'].max():.2f}°N
        • Spatial Range: Lon {skate_data['Longitude'].min():.2f}-{skate_data['Longitude'].max():.2f}°E
        
        Plankton Data:
        • Total Records: {plankton_records}
        • Variables: {plankton_vars}
        """

        axes[0, 0].text(0.05, 0.95, overview_text, transform=axes[0, 0].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].axis('off')

        # 2. Movement statistics
        movement_stats = skate_data.groupby('id').agg({
            'distance': ['sum', 'mean', 'max'],
            'speed': ['mean', 'max'],
            'Date': 'count'
        }).round(4)

        # Flatten column names
        movement_stats.columns = [
            '_'.join(col).strip() for col in movement_stats.columns]

        axes[0, 1].hist(movement_stats['distance_sum'],
                        bins=20, alpha=0.7, color='blue')
        axes[0, 1].set_xlabel('Total Distance per Skate')
        axes[0, 1].set_ylabel('Number of Skates')
        axes[0, 1].set_title('Distribution of Total Distance')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Temporal patterns
        monthly_stats = skate_data.groupby('month').agg({
            'distance': 'mean',
            'speed': 'mean',
            'id': 'nunique'
        })

        ax3 = axes[1, 0]
        ax3_twin = ax3.twinx()

        bars = ax3.bar(monthly_stats.index,
                       monthly_stats['distance'], alpha=0.7, color='green')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Average Distance', color='green')
        ax3.tick_params(axis='y', labelcolor='green')

        line = ax3_twin.plot(monthly_stats.index,
                             monthly_stats['id'], 'ro-', linewidth=2)
        ax3_twin.set_ylabel('Number of Skates', color='red')
        ax3_twin.tick_params(axis='y', labelcolor='red')

        axes[1, 0].set_title('Monthly Patterns')

        # 4. Model performance (if available)
        if model is not None and hasattr(model, 'metrics'):
            metrics_text = f"""
            Model Performance:
            
            Model Type: {type(model.model).__name__}
            R² Score: {model.metrics['r2']:.4f}
            RMSE: {model.metrics['rmse']:.4f}
            MAE: {model.metrics['mae']:.4f}
            
            Features Used: {len(model.feature_names)}
            Training Samples: {len(model.training_data['X'])}
            Test Samples: {len(model.test_data['X'])}
            """

            axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        else:
            axes[1, 1].text(0.5, 0.5, 'No Model Available', transform=axes[1, 1].transAxes,
                            fontsize=16, ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

        plt.suptitle(
            'Skate-Plankton Ecosystem Analysis Summary Report', fontsize=16, y=0.98)
        plt.tight_layout()

        report_path = os.path.join(self.output_dir, output_file)
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Summary report saved to: {report_path}")
        return report_path

    def load_png_data(self):
        """Load and process PNG data"""
        print("🖼️ Loading PNG data...")
        self.png_data = self.png_processor.process_all_images()
        return self.png_data

    def create_enhanced_skate_map_with_png(self, skate_data, output_file='enhanced_skate_map.html'):
        """Create enhanced skate map with PNG overlays"""
        print("🗺️ Creating enhanced skate map with PNG overlays...")

        # Load PNG data if not already loaded
        if self.png_data is None:
            self.load_png_data()

        # Calculate map center
        center_lat = skate_data['Latitude'].mean()
        center_lon = skate_data['Longitude'].mean()

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='OpenStreetMap'
        )

        # Add PNG overlays as image overlays
        if self.png_data:
            overlay_data = self.png_processor.create_map_overlay_data()

            # Add plankton species overlays
            for layer in overlay_data['plankton_layers']:
                # Create bounds for North Sea region (approximate)
                bounds = [[50, 1], [55, 5]]  # North Sea bounds

                # Add image overlay
                folium.raster_layers.ImageOverlay(
                    image=layer['base64_data'],
                    bounds=bounds,
                    opacity=0.6,
                    name=f"🦐 {layer['display_name']}",
                    show=False
                ).add_to(m)

            # Add environmental overlays
            for layer in overlay_data['environmental_layers']:
                bounds = [[50, 1], [55, 5]]  # North Sea bounds

                folium.raster_layers.ImageOverlay(
                    image=layer['base64_data'],
                    bounds=bounds,
                    opacity=0.6,
                    name=f"🌡️ {layer['display_name']}",
                    show=False
                ).add_to(m)

        # Add skate tracks
        sample_data = skate_data.iloc[::3].copy()
        unique_skates = sample_data['id'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_skates)))

        for i, skate_id in enumerate(unique_skates[:10]):
            skate_points = sample_data[sample_data['id'] == skate_id]

            if len(skate_points) > 1:
                folium.PolyLine(
                    locations=skate_points[['Latitude', 'Longitude']].values,
                    color=f'#{int(colors[i][0]*255):02x}{int(colors[i][1]*255):02x}{int(colors[i][2]*255):02x}',
                    weight=3,
                    opacity=0.8,
                    popup=f'Skate {skate_id}'
                ).add_to(m)

        # Add heatmap
        heat_data = [[row['Latitude'], row['Longitude']]
                     for idx, row in sample_data.iterrows()]
        plugins.HeatMap(heat_data, name='Skate Density', show=True).add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Save map
        map_path = os.path.join(self.output_dir, output_file)
        m.save(map_path)

        print(f"✅ Enhanced skate map saved to: {map_path}")
        return map_path

    def create_png_dashboard(self, skate_data, plankton_data=None, output_file='png_dashboard.html'):
        """Create dashboard with PNG figures and skate data"""
        print("📊 Creating PNG dashboard...")

        # Load PNG data if not already loaded
        if self.png_data is None:
            self.load_png_data()

        if not self.png_data:
            print("❌ No PNG data available")
            return None

        # Create HTML dashboard
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BlueCloud Skate-Plankton Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: #2c3e50;
        }
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .panel {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .panel h2 {
            color: #34495e;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }
        .image-container {
            text-align: center;
            margin: 15px 0;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .image-title {
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0 5px 0;
        }
        .stats {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 4px;
            margin: 15px 0;
        }
        .stats h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        .stats p {
            margin: 5px 0;
            color: #7f8c8d;
        }
        .full-width {
            grid-column: 1 / -1;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .image-card {
            border: 1px solid #ddd;
            border-radius: 4px;
            overflow: hidden;
            transition: transform 0.3s ease;
        }
        .image-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .image-card img {
            width: 100%;
            height: auto;
        }
        .image-card .title {
            padding: 10px;
            background: #f8f9fa;
            font-weight: bold;
            color: #2c3e50;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌊 BlueCloud Skate-Plankton Ecosystem Dashboard</h1>
        <p>Interactive visualization of skate movements and plankton distributions</p>
    </div>
    
    <div class="dashboard">
"""

        # Add skate data statistics
        html_content += f"""
        <div class="panel">
            <h2>🦈 Skate Tracking Data</h2>
            <div class="stats">
                <h3>Summary Statistics</h3>
                <p><strong>Total Records:</strong> {len(skate_data):,}</p>
                <p><strong>Unique Skates:</strong> {skate_data['id'].nunique()}</p>
                <p><strong>Date Range:</strong> {skate_data['Date'].min()} to {skate_data['Date'].max()}</p>
                <p><strong>Latitude Range:</strong> {skate_data['Latitude'].min():.3f}° to {skate_data['Latitude'].max():.3f}°</p>
                <p><strong>Longitude Range:</strong> {skate_data['Longitude'].min():.3f}° to {skate_data['Longitude'].max():.3f}°</p>
            </div>
        </div>
"""

        # Add plankton data statistics
        if plankton_data is not None:
            html_content += f"""
        <div class="panel">
            <h2>🦐 Plankton Data</h2>
            <div class="stats">
                <h3>Summary Statistics</h3>
                <p><strong>Total Records:</strong> {len(plankton_data):,}</p>
                <p><strong>Species:</strong> {len([col for col in plankton_data.columns if 'abundance' in col.lower()])}</p>
                <p><strong>Environmental Variables:</strong> {len([col for col in plankton_data.columns if any(env in col.lower() for env in ['temperature', 'salinity', 'nitrate', 'phosphate', 'silicate'])])}</p>
            </div>
        </div>
"""

        # Add plankton species images
        plankton_images = self.png_processor.get_plankton_images()
        if plankton_images:
            html_content += """
        <div class="panel full-width">
            <h2>🦐 Plankton Species Distributions</h2>
            <div class="image-grid">
"""
            for name, img_data in plankton_images.items():
                html_content += f"""
                <div class="image-card">
                    <img src="{img_data['base64_data']}" alt="{name}">
                    <div class="title">{name.replace('_', ' ').title()}</div>
                </div>
"""
            html_content += """
            </div>
        </div>
"""

        # Add environmental images
        env_images = self.png_processor.get_environmental_images()
        if env_images:
            html_content += """
        <div class="panel full-width">
            <h2>🌡️ Environmental Parameters</h2>
            <div class="image-grid">
"""
            for name, img_data in env_images.items():
                html_content += f"""
                <div class="image-card">
                    <img src="{img_data['base64_data']}" alt="{name}">
                    <div class="title">{name.replace('_', ' ').title()}</div>
                </div>
"""
            html_content += """
            </div>
        </div>
"""

        html_content += """
    </div>
</body>
</html>
"""

        # Save dashboard
        dashboard_path = os.path.join(self.output_dir, output_file)
        with open(dashboard_path, 'w') as f:
            f.write(html_content)

        print(f"✅ PNG dashboard saved to: {dashboard_path}")
        return dashboard_path

    def create_integrated_final_map(self, skate_data, plankton_data=None, output_file='final_integrated_map.html'):
        """Create the final integrated map with all data and PNG overlays"""
        print("🗺️ Creating final integrated map...")

        # Load PNG data if not already loaded
        if self.png_data is None:
            self.load_png_data()

        # Calculate map center
        center_lat = skate_data['Latitude'].mean()
        center_lon = skate_data['Longitude'].mean()

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='OpenStreetMap'
        )

        # Add PNG overlays
        if self.png_data:
            overlay_data = self.png_processor.create_map_overlay_data()

            # Add plankton species overlays
            for i, layer in enumerate(overlay_data['plankton_layers']):
                bounds = [[50, 1], [55, 5]]  # North Sea bounds

                folium.raster_layers.ImageOverlay(
                    image=layer['base64_data'],
                    bounds=bounds,
                    opacity=0.5,
                    name=f"🦐 {layer['display_name']}",
                    show=(i == 0)  # Show first layer by default
                ).add_to(m)

            # Add environmental overlays
            for layer in overlay_data['environmental_layers']:
                bounds = [[50, 1], [55, 5]]  # North Sea bounds

                folium.raster_layers.ImageOverlay(
                    image=layer['base64_data'],
                    bounds=bounds,
                    opacity=0.5,
                    name=f"🌡️ {layer['display_name']}",
                    show=False
                ).add_to(m)

        # Add skate tracks with enhanced styling
        # Every 2nd point for better performance
        sample_data = skate_data.iloc[::2].copy()
        unique_skates = sample_data['id'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_skates)))

        for i, skate_id in enumerate(unique_skates[:15]):  # Show more skates
            skate_points = sample_data[sample_data['id'] == skate_id]

            if len(skate_points) > 1:
                # Create polyline with gradient colors
                folium.PolyLine(
                    locations=skate_points[['Latitude', 'Longitude']].values,
                    color=f'#{int(colors[i][0]*255):02x}{int(colors[i][1]*255):02x}{int(colors[i][2]*255):02x}',
                    weight=4,
                    opacity=0.8,
                    popup=f'Skate {skate_id} - {len(skate_points)} points'
                ).add_to(m)

                # Add start marker
                folium.CircleMarker(
                    location=[skate_points.iloc[0]['Latitude'],
                              skate_points.iloc[0]['Longitude']],
                    radius=6,
                    color='green',
                    fill=True,
                    fillOpacity=0.8,
                    popup=f'Start - Skate {skate_id}'
                ).add_to(m)

                # Add end marker
                folium.CircleMarker(
                    location=[skate_points.iloc[-1]['Latitude'],
                              skate_points.iloc[-1]['Longitude']],
                    radius=6,
                    color='red',
                    fill=True,
                    fillOpacity=0.8,
                    popup=f'End - Skate {skate_id}'
                ).add_to(m)

        # Add heatmap
        heat_data = [[row['Latitude'], row['Longitude']]
                     for idx, row in sample_data.iterrows()]
        plugins.HeatMap(heat_data, name='Skate Density Heatmap',
                        show=True).add_to(m)

        # Add layer control with better organization
        folium.LayerControl(
            position='topright',
            collapsed=False
        ).add_to(m)

        # Add custom CSS for better styling
        m.get_root().html.add_child(folium.Element("""
        <style>
            .leaflet-control-layers {
                background: white !important;
                border-radius: 5px !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
            }
            .leaflet-control-layers label {
                font-weight: bold !important;
            }
        </style>
        """))

        # Save map
        map_path = os.path.join(self.output_dir, output_file)
        m.save(map_path)

        print(f"✅ Final integrated map saved to: {map_path}")
        return map_path


def main():
    """Test the visualizer"""
    from skate_processor import SkateProcessor
    from plankton_processor import PlanktonProcessor

    print("🧪 Testing Visualizer...")

    # Load sample data
    skate_processor = SkateProcessor(
        "/home/samwork/Documents/coding/bluecloud-hackathon-2025/deliverable4/Skates_Track.csv")
    skate_data = skate_processor.process()

    plankton_processor = PlanktonProcessor()
    plankton_data = plankton_processor.process(use_sample_data=True)

    # Create visualizations
    viz = Visualizer(
        output_dir="/home/samwork/Documents/coding/bluecloud-hackathon-2025/deliverable4/data")

    # Create all visualizations
    viz.create_skate_movement_map(skate_data)
    viz.create_plankton_distribution_map(plankton_data)
    viz.create_time_series_plots(skate_data, plankton_data)
    viz.create_correlation_heatmap(skate_data, plankton_data)
    viz.create_dashboard(skate_data, plankton_data)
    viz.create_summary_report(skate_data, plankton_data)

    print("✅ Visualizer test completed successfully!")


if __name__ == "__main__":
    main()
