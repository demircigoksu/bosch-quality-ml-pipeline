"""
Configuration module for Bosch Quality Classification pipeline.

This module contains all configuration parameters for data processing,
model training, and deployment.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"

# Data configuration - Bosch dataset files
TRAIN_NUMERIC_PATH = DATA_DIR / "train_numeric.csv"
TEST_NUMERIC_PATH = DATA_DIR / "test_numeric.csv"
TRAIN_CATEGORICAL_PATH = DATA_DIR / "train_categorical.csv"
TEST_CATEGORICAL_PATH = DATA_DIR / "test_categorical.csv"
TRAIN_DATE_PATH = DATA_DIR / "train_date.csv"
TEST_DATE_PATH = DATA_DIR / "test_date.csv"

# Sampling configuration (for faster development)
SAMPLE_SIZE = 100_000  # Number of rows to load for training
USE_SAMPLING = True    # Set to False to use full dataset

# Model configuration
MODEL_NAME = "bosch_quality_classifier"
MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.pkl"

# Training parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Model hyperparameters
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'scale_pos_weight': 200,  # Handle class imbalance (~1:200 ratio)
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'use_label_encoder': False
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'class_weight': 'balanced',  # Handle class imbalance
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

# Feature engineering
MAX_MISSING_THRESHOLD = 0.9  # Drop features with >90% missing values
FILL_VALUE = -999  # Value to fill remaining NaN (marker for "not measured")
CORRELATION_THRESHOLD = 0.95  # Drop highly correlated features

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Bosch Quality Prediction API"
API_VERSION = "1.0.0"

# Streamlit configuration
STREAMLIT_PORT = 8501
