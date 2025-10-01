# BlueCloud Mini Pipeline: Skate-Plankton Ecosystem Analysis

A modular mini-pipeline that integrates skate tracking data, plankton distribution, and machine learning to analyze marine ecosystem dynamics. This is a simplified version of the larger BlueCloud hackathon project, designed for rapid prototyping and demonstration.

**✅ Status**: The mini pipeline is working and ready for use with proper NetCDF plankton data from the BlueCloud VLab.

## 🌊 Overview

This mini-pipeline demonstrates the integration of:
1. **Skate Movement Data** - Tracking individual skate movements in the North Sea
2. **Plankton Distribution** - NetCDF-based plankton abundance data from BlueCloud VLab
3. **Tiny ML Model** - Simple correlation analysis between predator behavior and prey distribution
4. **Interactive Visualizations** - Maps, charts, and dashboards

## 📊 Data Flow

```
BlueCloud VLab → NetCDF Files → planktonoutputs/ → PlanktonProcessor → Analysis
     ↓
DTU Skate Data → data/Skates_Track.csv → SkateProcessor → Analysis
     ↓
Both datasets → TinyModel → Correlations → Visualizer → Outputs
```

## 🏗️ Architecture

The pipeline is organized into modular components:

```text
deliverable4/
├── modules/
│   ├── skate_processor.py      # Skate data processing and analysis
│   ├── plankton_processor.py   # Plankton data processing (NetCDF ready)
│   ├── tiny_model.py          # Simple ML model for correlation
│   └── visualizer.py          # Visualization and mapping
├── pipeline/
│   └── main_pipeline.py       # Main orchestrator
├── data/
│   ├── Skates_Track.csv       # Input skate tracking data
│   └── outputs/              # Generated outputs
├── planktonoutputs/           # NetCDF files from BlueCloud VLab
│   ├── *.nc                  # Plankton abundance NetCDF files
│   └── *.interp.nc           # Interpolated plankton data
├── run_mini_pipeline.py      # Simple runner script
├── pyproject.toml            # Modern Python project configuration
├── requirements.txt          # Traditional pip requirements
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Install uv Package Manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies with uv
```bash
uv run pip install -r requirements.txt
```

### 3. Prepare Plankton Data from BlueCloud VLab

**Required**: Download NetCDF files from the BlueCloud-Plankton VLab and store them in the `planktonoutputs/` folder:

1. **Access BlueCloud VLab**: [BlueCloud VLab](https://blue-cloud.d4science.org/group/zoo-phytoplankton_eov/jupyterhub)
2. **Run DIVAndNN Analysis**: Process Continuous Plankton Recorder (CPR) data
3. **Download NetCDF Files**: Save the generated `*.nc` and `*.interp.nc` files
4. **Store in Project**: Place all NetCDF files in `deliverable4/planktonoutputs/`

**Benefits of VLab Processing**:
- Pre-configured environment with all Julia dependencies
- Optimized for DIVAndNN interpolation analysis
- High-performance computing resources
- Ready-to-use NetCDF outputs

### 4. Run the Complete Pipeline
```bash
uv run python run_mini_pipeline.py
```

**Note**: The `planktonoutputs/` directory has been created for you. Place your BlueCloud VLab NetCDF files there before running the pipeline.

### 5. Run Individual Modules
```bash
uv run python -c "
from modules.skate_processor import SkateProcessor
from modules.plankton_processor import PlanktonProcessor
from modules.tiny_model import TinyModel
from modules.visualizer import Visualizer

# Process skate data
skate_processor = SkateProcessor('data/Skates_Track.csv')
skate_data = skate_processor.process()

# Process plankton data from BlueCloud VLab outputs
plankton_processor = PlanktonProcessor('planktonoutputs/')
plankton_data = plankton_processor.process()

# Train tiny model
model = TinyModel()
model.fit(skate_data, plankton_data)

# Create visualizations
viz = Visualizer()
viz.create_dashboard(skate_data, plankton_data, model)
"
```

## 📊 Data Sources & Requirements

### Skate Data (`Skates_Track.csv`)
- **Source**: DTU Aqua acoustic tracking data
- **Format**: CSV with columns: `id`, `Date`, `Latitude`, `Longitude`
- **Content**: 4,926 tracking records from 50+ individual skates
- **Time Range**: August-September 2021
- **Location**: North Sea (50-55°N, 1-5°E)
- **File Location**: `deliverable4/data/Skates_Track.csv`

### Plankton Data (NetCDFs)
- **Source**: BlueCloud-Plankton VLab → DIVAndNN analysis → NetCDF outputs
- **Processing**: Continuous Plankton Recorder (CPR) data → DIVAndNN interpolation
- **Format**: NetCDF files with plankton abundance data
- **Variables**: Species abundance (Acartia, Calanus finmarchicus, etc.), environmental parameters
- **Spatial Coverage**: North Sea region
- **Temporal Resolution**: Monthly/quarterly
- **File Location**: `deliverable4/planktonoutputs/` (download from VLab)
- **File Pattern**: `*.nc` and `*.interp.nc` files from VLab analysis
- **Required Processing**: Use [BlueCloud VLab](https://blue-cloud.d4science.org/group/zoo-phytoplankton_eov/jupyterhub)

### Elasmobranch Data (FAO Capture Data)
- **Source**: FAO Fisheries and Aquaculture database
- **Format**: CSV files with catch data
- **Variables**: Species codes, catch quantities, fishing areas, time periods
- **Spatial Coverage**: Global (mapped to FAO fishing areas)
- **Temporal Resolution**: Annual (1950-2023)
- **File Location**: `Capture_2025.1.0/` directory

## 🔧 Module Details

### 1. Skate Processor (`modules/skate_processor.py`)
**Purpose**: Process and analyze skate tracking data

**Key Features**:
- Load and validate skate tracking CSV
- Calculate movement metrics (distance, speed, direction)
- Add temporal features (month, day of year, season)
- Spatial analysis and filtering
- Data quality checks

**Output**:
- Processed skate DataFrame with enhanced features
- Movement statistics and summaries

### 2. Plankton Processor (`modules/plankton_processor.py`)
**Purpose**: Process plankton NetCDF data from BlueCloud VLab

**Data Source**: Reads from `planktonoutputs/` directory
- Automatically discovers `*.nc` and `*.interp.nc` files
- Falls back to sample data if no NetCDF files found

**Key Features**:
- Load NetCDF files using xarray
- Extract plankton species abundance
- Spatial and temporal interpolation
- Data aggregation and resampling
- Coordinate system handling

**Output**:
- Processed plankton DataFrame
- Spatial grids for visualization

### 3. Tiny Model (`modules/tiny_model.py`)
**Purpose**: Simple machine learning model for correlation analysis

**Key Features**:
- Random Forest regression
- Feature engineering (spatial, temporal, environmental)
- Model training and validation
- Performance metrics calculation
- Feature importance analysis

**Model Features**:
- Geographic coordinates (lat/lon)
- Temporal features (month, day of year)
- Movement metrics (distance, speed)
- Environmental variables (when available)

### 4. Visualizer (`modules/visualizer.py`)
**Purpose**: Create comprehensive visualizations and maps

**Key Features**:
- Interactive maps (Folium/Plotly)
- Time series plots
- Correlation heatmaps
- Distribution plots
- Dashboard creation

**Output Files**:
- `skate_movement_map.html` - Interactive skate tracks
- `plankton_distribution_map.html` - Plankton abundance maps
- `correlation_dashboard.html` - Combined analysis dashboard
- `model_performance.png` - Model metrics visualization

## 📈 Expected Outputs

### 1. Interactive Maps
- **Skate Movement Map**: Individual tracks with temporal coloring
- **Plankton Distribution Map**: Abundance patterns across the study area
- **Combined Dashboard**: Side-by-side comparison

### 2. Analysis Results
- **Movement Patterns**: Daily/monthly skate behavior
- **Spatial Correlations**: Skate-plankton relationship maps
- **Model Performance**: R² score, RMSE, feature importance

### 3. Data Products
- **Processed Datasets**: Clean, analysis-ready data
- **Model Artifacts**: Trained model and predictions
- **Summary Statistics**: Key metrics and insights

## 🎯 Use Cases

1. **Marine Conservation**: Understand skate habitat preferences
2. **Ecosystem Monitoring**: Track predator-prey relationships
3. **Climate Research**: Analyze seasonal patterns
4. **Fisheries Management**: Inform sustainable practices

## 🔧 Customization

### Adding New Data Sources
```python
# Extend the processor classes
class CustomProcessor(BaseProcessor):
    def process(self, data_path):
        # Add your custom processing logic
        pass
```

### Modifying the Model
```python
# Adjust model parameters
model = RandomForestRegressor(
    n_estimators=200,      # More trees
    max_depth=15,         # Deeper trees
    min_samples_split=5,  # More samples per split
    random_state=42
)
```

### Custom Visualizations
```python
# Create custom plots
viz = Visualizer()
viz.create_custom_plot(data, plot_type='heatmap')
```

## 📚 Dependencies & Package Management

### Package Management with uv
This project uses `uv` for fast Python package management. The project includes:
- `pyproject.toml` - Modern Python project configuration
- `requirements.txt` - Traditional pip requirements (for compatibility)

### Core Libraries
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **xarray**: NetCDF data handling
- **scikit-learn**: Machine learning

### Visualization
- **matplotlib/seaborn**: Static plots
- **plotly**: Interactive charts
- **folium**: Interactive maps

### Geospatial
- **rasterio**: Raster data processing
- **geopandas**: Vector data handling

### Installation Commands
```bash
# Using uv (recommended)
uv run pip install -r requirements.txt

# Using traditional pip
pip install -r requirements.txt
```

## 🚧 Development Status

- ✅ **Skate Processor**: Complete and tested
- ✅ **Visualizer**: Complete with multiple output formats
- ✅ **Tiny Model**: Basic implementation ready
- ✅ **Plankton Processor**: Ready for NetCDF integration
- ✅ **Main Pipeline**: Working mini pipeline implementation
- ✅ **Package Management**: pyproject.toml and requirements.txt configured

## 🤝 Contributing

This is part of the BlueCloud Hackathon 2025 project. To contribute:

1. Fork the repository
2. Create a feature branch
3. Add your module or enhancement
4. Test with sample data
5. Submit a pull request

## 📄 License

This project is part of the BlueCloud Hackathon 2025 deliverables.

---

**BlueCloud Hackathon 2025** | Mini Pipeline for Marine Ecosystem Analysis

*Ready for your NetCDF plankton data! 🦈🌊*