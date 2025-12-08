"""
Training module for Bosch Quality Classification.

This module handles data loading, preprocessing, model training,
and model persistence.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef
import xgboost as xgb
import pickle
from pathlib import Path

from config import (
    TRAIN_DATA_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    XGBOOST_PARAMS,
    MAX_MISSING_THRESHOLD
)


def load_data(data_path=TRAIN_DATA_PATH):
    """
    Load training data from CSV file.
    
    Args:
        data_path: Path to the training data CSV file
        
    Returns:
        pandas.DataFrame: Loaded training data
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_data(df, target_col='Response'):
    """
    Preprocess the data by handling missing values and feature selection.
    
    Args:
        df: Input dataframe
        target_col: Name of the target column
        
    Returns:
        X: Feature matrix
        y: Target vector
    """
    print("Preprocessing data...")
    
    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Remove features with too many missing values
    missing_pct = X.isnull().sum() / len(X)
    cols_to_keep = missing_pct[missing_pct < MAX_MISSING_THRESHOLD].index
    X = X[cols_to_keep]
    print(f"Features after removing high-missing columns: {X.shape[1]}")
    
    # Fill remaining missing values with median
    X = X.fillna(X.median())
    
    print(f"Final feature shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y


def train_model(X_train, y_train, model_type='xgboost'):
    """
    Train a classification model.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to train ('xgboost' or 'random_forest')
        
    Returns:
        Trained model
    """
    print(f"Training {model_type} model...")
    
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    print("Model training completed.")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    print("Evaluating model...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    metrics = {
        'roc_auc': roc_auc,
        'mcc': mcc
    }
    
    return metrics


def save_model(model, model_path=MODEL_PATH):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        model_path: Path to save the model
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")


def main():
    """
    Main training pipeline.
    """
    print("=" * 50)
    print("Bosch Quality Classification - Training Pipeline")
    print("=" * 50)
    
    # Load data
    df = load_data()
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    model = train_model(X_train, y_train, model_type='xgboost')
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model)
    
    print("\n" + "=" * 50)
    print("Training pipeline completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
