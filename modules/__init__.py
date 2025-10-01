"""
BlueCloud Mini Pipeline Modules
==============================

This package contains all the processing modules for the BlueCloud mini pipeline:
- skate_processor: Processes skate tracking data
- plankton_processor: Processes plankton data from NetCDF files
- elasmobranch_processor: Processes elasmobranch capture data
- tiny_model: Machine learning model for predictions
- visualizer: Creates visualizations and reports

Author: BlueCloud Hackathon 2025
"""

__version__ = "1.0.0"
__author__ = "BlueCloud Hackathon 2025"

# Import all main classes for easy access
from .skate_processor import SkateProcessor
from .plankton_processor import PlanktonProcessor
from .elasmobranch_processor import ElasmobranchProcessor
from .tiny_model import TinyModel
from .visualizer import Visualizer

__all__ = [
    'SkateProcessor',
    'PlanktonProcessor',
    'ElasmobranchProcessor',
    'TinyModel',
    'Visualizer'
]
