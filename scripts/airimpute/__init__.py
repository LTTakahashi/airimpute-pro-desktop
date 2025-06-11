"""
AirImpute - Scientific Air Quality Data Imputation Library
=========================================================

This package provides the core imputation algorithms for AirImpute Pro Desktop.
"""

__version__ = "1.0.0"
__author__ = "AirImpute Research Team"

from .core import ImputationEngine
from .methods import get_available_methods
from .validation import validate_results

__all__ = [
    "ImputationEngine",
    "get_available_methods",
    "validate_results",
]