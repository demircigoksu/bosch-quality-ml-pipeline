"""
Bosch Quality ML Pipeline - Source package

This package contains the core modules for the Bosch Quality Classification pipeline:
- config: Configuration parameters and paths
- train: Model training utilities
- inference: Model inference and prediction functions
"""

from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    MODEL_PATH,
    RANDOM_SEED,
    XGBOOST_PARAMS,
)

__version__ = "1.0.0"
__author__ = "Bosch Quality ML Team"

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR", 
    "MODELS_DIR",
    "MODEL_PATH",
    "RANDOM_SEED",
    "XGBOOST_PARAMS",
]
