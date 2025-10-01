# Plankton Outputs Directory

This directory should contain NetCDF files generated from the BlueCloud-Plankton VLab analysis.

## Required Files

Download the following files from the [BlueCloud VLab](https://blue-cloud.d4science.org/group/zoo-phytoplankton_eov/jupyterhub) and place them in this directory:

### NetCDF Files
- `*.nc` - Raw plankton abundance data
- `*.interp.nc` - Interpolated plankton data from DIVAndNN analysis

### Expected File Types
- **Species Abundance**: Files containing plankton species abundance data
- **Environmental Data**: Files with temperature, salinity, nutrients, etc.
- **Interpolated Data**: Spatially and temporally interpolated data

## Data Source
- **Origin**: BlueCloud-Plankton VLab DIVAndNN analysis
- **Input Data**: Continuous Plankton Recorder (CPR) data
- **Processing**: DIVAndNN interpolation for spatial and temporal gaps
- **Coverage**: North Sea region
- **Resolution**: Monthly/quarterly temporal resolution

## Usage
The mini pipeline will automatically detect and process all NetCDF files in this directory when running the plankton analysis.

## File Naming Convention
Files typically follow patterns like:
- `species_name-1.nc` (e.g., `Acartia-1.nc`, `Calanus_finmarchicus-1.nc`)
- `environmental_variable-1.nc` (e.g., `temperature-1.nc`, `salinity-1.nc`)
- `*interp.nc` for interpolated data

## Note
If this directory is empty, the plankton processor will skip plankton analysis and focus on skate data only.
