"""
Bosch kalite tahmin projesi ayarları.
"""

import os
from pathlib import Path

# Proje dizinleri
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"

# Orijinal veri dosyaları
TRAIN_NUMERIC_PATH = DATA_DIR / "train_numeric.csv"
TEST_NUMERIC_PATH = DATA_DIR / "test_numeric.csv"
TRAIN_CATEGORICAL_PATH = DATA_DIR / "train_categorical.csv"
TEST_CATEGORICAL_PATH = DATA_DIR / "test_categorical.csv"
TRAIN_DATE_PATH = DATA_DIR / "train_date.csv"
TEST_DATE_PATH = DATA_DIR / "test_date.csv"

# Temizlenmiş veri
TRAIN_CLEAN_PATH = DATA_DIR / "train_numeric_clean.csv"
TEST_CLEAN_PATH = DATA_DIR / "test_numeric_clean_alt.csv"

# Örnekleme ayarları
SAMPLE_SIZE = 100_000
USE_SAMPLING = False

# Model dosyaları
MODEL_NAME = "final_model"
MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.pkl"
FEATURES_PATH = MODELS_DIR / "feature_names.pkl"
CONFIG_PATH = MODELS_DIR / "model_config.pkl"

# Eğitim parametreleri
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# XGBoost hiperparametreleri
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'scale_pos_weight': 200,  # sınıf dengesizliği için (~1:200)
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'use_label_encoder': False
}

# Random Forest hiperparametreleri
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'class_weight': 'balanced',
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

# Feature engineering eşikleri
MAX_MISSING_THRESHOLD = 0.9   # %90+ eksik veri varsa sütunu at
FILL_VALUE = -999             # eksik değer dolgu işareti
CORRELATION_THRESHOLD = 0.95  # yüksek korelasyonlu sütunları at

# API ayarları
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Bosch Quality Prediction API"
API_VERSION = "1.0.0"

# Streamlit portu
STREAMLIT_PORT = 8501
