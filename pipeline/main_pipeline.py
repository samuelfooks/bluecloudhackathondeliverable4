#!/usr/bin/env python3
"""
Main Pipeline Orchestrator
=========================

This module orchestrates the complete mini pipeline, coordinating all modules
to process skate data, plankton data, train models, and create visualizations.

Author: BlueCloud Hackathon 2025
"""

from modules import (
    SkateProcessor,
    PlanktonProcessor,
    ElasmobranchProcessor,
    TinyModel,
    Visualizer
)
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for package imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Import modules from the modules package


class MainPipeline:
    def __init__(self, config=None):
        """
        Initialize the Main Pipeline

        Args:
            config (dict): Configuration dictionary with paths and parameters
        """
        self.config = config or self.get_default_config()
        self.output_dir = self.config['output_dir']

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize components
        self.skate_processor = None
        self.plankton_processor = None
        self.elasmobranch_processor = None
        self.model = None
        self.visualizer = None

        # Data containers
        self.skate_data = None
        self.plankton_data = None
        self.elasmobranch_data = None
        self.results = {}

    def get_default_config(self):
        """Get default configuration"""
        # Use relative paths from the current file location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level from pipeline/ to deliverable4/
        base_dir = os.path.dirname(current_dir)
        # Go up one more level to project root
        project_root = os.path.dirname(base_dir)

        return {
            'skate_csv_path': os.path.join(base_dir, 'Skates_Track.csv'),
            'plankton_netcdf_dir': os.path.join(base_dir, 'planktonoutputs'),
            'plankton_netcdf_files': None,
            'use_sample_plankton': True,  # Use sample data for testing
            'capture_csv_path': os.path.join(project_root, 'Capture_2025.1.0', 'Capture_Quantity.csv'),
            'species_metadata_path': os.path.join(project_root, 'Capture_2025.1.0', 'CL_FI_SPECIES_GROUPS.csv'),
            'water_area_path': os.path.join(project_root, 'Capture_2025.1.0', 'CL_FI_WATERAREA_GROUPS.csv'),
            'output_dir': os.path.join(base_dir, 'data'),
            'model_type': 'random_forest',
            'test_size': 0.2,
            'random_state': 42
        }

    def validate_inputs(self):
        """Validate input files and configuration"""
        print("🔍 Validating inputs...")

        # Check skate data
        if not os.path.exists(self.config['skate_csv_path']):
            raise FileNotFoundError(
                f"Skate CSV not found: {self.config['skate_csv_path']}")

        # Check plankton data
        if self.config['plankton_netcdf_dir'] and not os.path.exists(self.config['plankton_netcdf_dir']):
            print(
                f"⚠️ Plankton NetCDF directory not found: {self.config['plankton_netcdf_dir']}")
            print("   Will use sample plankton data for testing")
            self.config['use_sample_plankton'] = True

        print("✅ Input validation complete")

    def process_skate_data(self):
        """Process skate tracking data"""
        print("\n🦈 Processing Skate Data")
        print("=" * 30)

        self.skate_processor = SkateProcessor(self.config['skate_csv_path'])
        self.skate_data = self.skate_processor.process()

        # Export processed data
        skate_output = os.path.join(self.output_dir, 'skate_processed.csv')
        self.skate_processor.export_processed_data(skate_output)

        self.results['skate_data'] = self.skate_data
        self.results['skate_stats'] = self.skate_processor.get_summary()

        return self.skate_data

    def process_plankton_data(self):
        """Process plankton data"""
        print("\n🦐 Processing Plankton Data")
        print("=" * 30)

        self.plankton_processor = PlanktonProcessor(
            netcdf_dir=self.config['plankton_netcdf_dir'],
            netcdf_files=self.config.get('plankton_netcdf_files', None)
        )

        self.plankton_data = self.plankton_processor.process(
            use_sample_data=self.config['use_sample_plankton']
        )

        # Export processed data
        if self.plankton_data is not None:
            plankton_output = os.path.join(
                self.output_dir, 'plankton_processed.csv')
            self.plankton_processor.export_processed_data(plankton_output)

        self.results['plankton_data'] = self.plankton_data
        self.results['plankton_stats'] = self.plankton_processor.get_summary()

        return self.plankton_data

    def process_elasmobranch_data(self):
        """Process elasmobranch capture data"""
        print("\n🦈 Processing Elasmobranch Data")
        print("=" * 35)

        # Check if capture data files exist
        capture_files = [
            self.config['capture_csv_path'],
            self.config['species_metadata_path'],
            self.config['water_area_path']
        ]

        missing_files = [f for f in capture_files if not os.path.exists(f)]
        if missing_files:
            print("⚠️ Missing capture data files:")
            for f in missing_files:
                print(f"  - {f}")
            print("   Skipping elasmobranch processing")
            return None

        self.elasmobranch_processor = ElasmobranchProcessor(
            capture_csv_path=self.config['capture_csv_path'],
            species_metadata_path=self.config['species_metadata_path'],
            water_area_path=self.config['water_area_path']
        )

        # Process elasmobranch data (focus on North Sea - area 27)
        results = self.elasmobranch_processor.process(
            target_area='27', resolution=1.0)

        self.elasmobranch_data = results['elasmobranch_data']

        # Export processed data
        elasmobranch_output = os.path.join(
            self.output_dir, 'elasmobranch_processed.csv')
        self.elasmobranch_processor.export_raster_data(self.output_dir)

        self.results['elasmobranch_data'] = self.elasmobranch_data
        self.results['elasmobranch_stats'] = self.elasmobranch_processor.get_summary()

        return self.elasmobranch_data

    def train_model(self):
        """Train the tiny model"""
        print("\n🤖 Training Tiny Model")
        print("=" * 25)

        if self.skate_data is None:
            raise ValueError("Skate data not processed yet")

        self.model = TinyModel(
            model_type=self.config['model_type'],
            random_state=self.config['random_state']
        )

        self.model.fit(self.skate_data, self.plankton_data)

        # Create performance plots
        if hasattr(self.model, 'test_data') and self.model.test_data is not None:
            self.model.create_performance_plots(
                self.model.test_data['y'],
                self.model.predictions,
                output_dir=self.output_dir
            )

        self.results['model'] = self.model
        self.results['model_summary'] = self.model.get_summary()

        return self.model

    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n📊 Creating Visualizations")
        print("=" * 30)

        self.visualizer = Visualizer(output_dir=self.output_dir)

        # Create all visualizations
        viz_results = {}

        # 1. Skate movement map
        skate_map = self.visualizer.create_skate_movement_map(self.skate_data)
        viz_results['skate_map'] = skate_map

        # 2. Plankton distribution map
        if self.plankton_data is not None:
            plankton_map = self.visualizer.create_plankton_distribution_map(
                self.plankton_data)
            viz_results['plankton_map'] = plankton_map

        # 3. Time series plots
        time_series = self.visualizer.create_time_series_plots(
            self.skate_data, self.plankton_data)
        viz_results['time_series'] = time_series

        # 4. Correlation heatmap
        correlation = self.visualizer.create_correlation_heatmap(
            self.skate_data, self.plankton_data)
        viz_results['correlation'] = correlation

        # 5. Interactive dashboard
        dashboard = self.visualizer.create_dashboard(
            self.skate_data, self.plankton_data, self.model)
        viz_results['dashboard'] = dashboard

        # 6. Summary report
        summary_report = self.visualizer.create_summary_report(
            self.skate_data, self.plankton_data, self.model)
        viz_results['summary_report'] = summary_report

        # 7. NEW: Enhanced skate map with PNG overlays
        enhanced_map = self.visualizer.create_enhanced_skate_map_with_png(
            self.skate_data)
        viz_results['enhanced_map'] = enhanced_map

        # 8. NEW: PNG dashboard
        png_dashboard = self.visualizer.create_png_dashboard(
            self.skate_data, self.plankton_data)
        viz_results['png_dashboard'] = png_dashboard

        # 9. NEW: Final integrated map (THE MAIN OUTPUT)
        final_map = self.visualizer.create_integrated_final_map(
            self.skate_data, self.plankton_data)
        viz_results['final_integrated_map'] = final_map

        self.results['visualizations'] = viz_results

        return viz_results

    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n📋 Generating Analysis Report")
        print("=" * 35)

        report_path = os.path.join(self.output_dir, 'analysis_report.txt')

        with open(report_path, 'w') as f:
            f.write("SKATE-PLANKTON ECOSYSTEM ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Skate data summary
            if 'skate_stats' in self.results:
                stats = self.results['skate_stats']['stats']
                f.write("SKATE DATA SUMMARY:\n")
                f.write(f"  Total Records: {stats['total_records']:,}\n")
                f.write(f"  Individual Skates: {stats['unique_skates']}\n")
                f.write(
                    f"  Date Range: {stats['date_range'][0].strftime('%Y-%m-%d')} to {stats['date_range'][1].strftime('%Y-%m-%d')}\n")
                f.write(
                    f"  Total Distance: {stats['total_distance']:.2f} degrees\n")
                f.write(
                    f"  Average Speed: {stats['avg_speed']:.4f} degrees/day\n\n")

            # Plankton data summary
            if 'plankton_stats' in self.results:
                stats = self.results['plankton_stats']['stats']
                f.write("PLANKTON DATA SUMMARY:\n")
                f.write(f"  Total Records: {stats['total_records']:,}\n")
                f.write(f"  Source Files: {stats.get('unique_files', 1)}\n")
                if 'plankton_summary' in stats:
                    f.write(
                        f"  Plankton Variables: {len(stats['plankton_summary'])}\n")
                if 'environmental_summary' in stats:
                    f.write(
                        f"  Environmental Variables: {len(stats['environmental_summary'])}\n")
                f.write("\n")

            # Model performance
            if 'model_summary' in self.results:
                model_summary = self.results['model_summary']
                if 'metrics' in model_summary:
                    metrics = model_summary['metrics']
                    f.write("MODEL PERFORMANCE:\n")
                    f.write(
                        f"  Model Type: {type(model_summary['model']).__name__}\n")
                    f.write(f"  R² Score: {metrics['r2']:.4f}\n")
                    f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
                    f.write(f"  MAE: {metrics['mae']:.4f}\n")
                    f.write(
                        f"  Features Used: {len(model_summary['feature_names'])}\n\n")

            # Output files
            f.write("OUTPUT FILES:\n")
            f.write("  Data Files:\n")
            f.write("    - skate_processed.csv\n")
            f.write("    - plankton_processed.csv\n")
            f.write("  Visualizations:\n")
            f.write("    - skate_movement_map.html\n")
            f.write("    - plankton_distribution_map.html\n")
            f.write("    - time_series_analysis.png\n")
            f.write("    - correlation_heatmap.png\n")
            f.write("    - dashboard.html\n")
            f.write("    - summary_report.png\n")
            f.write("    - model_performance.png\n")

        print(f"✅ Analysis report saved to: {report_path}")
        return report_path

    def run_full_pipeline(self):
        """Run the complete pipeline"""
        print("🌊🦈 Starting BlueCloud Mini Pipeline")
        print("=" * 45)

        start_time = datetime.now()

        try:
            # Validate inputs
            self.validate_inputs()

            # Process data
            self.process_skate_data()
            self.process_plankton_data()
            self.process_elasmobranch_data()

            # Train model
            self.train_model()

            # Create visualizations
            self.create_visualizations()

            # Generate report
            self.generate_report()

            # Final summary
            end_time = datetime.now()
            duration = end_time - start_time

            print("\n🎯 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 40)
            print(f"⏱️ Total runtime: {duration.total_seconds():.1f} seconds")
            print(f"📁 Output directory: {self.output_dir}")
            print(f"📊 Skate records processed: {len(self.skate_data):,}")
            plankton_count = len(
                self.plankton_data) if self.plankton_data is not None else 'N/A'
            elasmobranch_count = len(
                self.elasmobranch_data) if self.elasmobranch_data is not None else 'N/A'

            print(f"🦐 Plankton records processed: {plankton_count:,}" if isinstance(
                plankton_count, int) else f"🦐 Plankton records processed: {plankton_count}")
            print(f"🦈 Elasmobranch records processed: {elasmobranch_count:,}" if isinstance(
                elasmobranch_count, int) else f"🦈 Elasmobranch records processed: {elasmobranch_count}")

            if self.model and hasattr(self.model, 'metrics'):
                print(f"🤖 Model R² score: {self.model.metrics['r2']:.4f}")

            print("\n📋 Generated Files:")
            print("  🗺️ Interactive maps and dashboards")
            print("  📈 Time series and correlation plots")
            print("  📊 Model performance visualizations")
            print("  📄 Comprehensive analysis report")

            return self.results

        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            raise


def main():
    """Main function to run the pipeline"""
    # You can customize the configuration here
    config = {
        'skate_csv_path': "/home/samwork/Documents/coding/bluecloud-hackathon-2025/deliverable4/Skates_Track.csv",
        'plankton_netcdf_dir': None,  # Set this when NetCDFs are available
        'use_sample_plankton': True,  # Use sample data for testing
        'output_dir': "/home/samwork/Documents/coding/bluecloud-hackathon-2025/deliverable4/data",
        'model_type': 'random_forest',
        'random_state': 42
    }

    # Initialize and run pipeline
    pipeline = MainPipeline(config)
    results = pipeline.run_full_pipeline()

    print("\n✅ Mini pipeline completed successfully!")
    return results


if __name__ == "__main__":
    main()
