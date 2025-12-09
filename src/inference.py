"""
Inference module for Bosch Quality Classification.

This module handles loading trained models and making predictions
on new data.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Union, List, Dict

# Model paths
MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "final_model.pkl"
FEATURES_PATH = MODELS_DIR / "feature_names.pkl"
CONFIG_PATH = MODELS_DIR / "model_config.pkl"


def load_model(model_path=None):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model object
    """
    if model_path is None:
        model_path = MODEL_PATH
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def load_feature_names():
    """Load feature names used in training."""
    if FEATURES_PATH.exists():
        return joblib.load(FEATURES_PATH)
    return None


def load_config():
    """Load model configuration."""
    if CONFIG_PATH.exists():
        return joblib.load(CONFIG_PATH)
    return {'threshold': 0.35}


def apply_feature_engineering(df):
    """Apply feature engineering to input data."""
    X = df.copy()
    
    # Get original columns (excluding engineered ones)
    original_cols = [c for c in df.columns if not c.startswith('row_')]
    
    # Row statistics
    X['row_mean'] = df[original_cols].mean(axis=1)
    X['row_std'] = df[original_cols].std(axis=1)
    X['row_min'] = df[original_cols].min(axis=1)
    X['row_max'] = df[original_cols].max(axis=1)
    X['row_range'] = X['row_max'] - X['row_min']
    X['row_nonzero'] = (df[original_cols] != 0).sum(axis=1)
    
    return X


def preprocess_input(data: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
    """
    Preprocess input data for prediction.
    
    Args:
        data: Input data as DataFrame, dict, or list of dicts
        
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    # Convert to DataFrame if necessary
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Remove Id and Response if present
    for col in ['Id', 'Response']:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Apply feature engineering
    df = apply_feature_engineering(df)
    
    # Fill missing values
    df = df.fillna(0)
    
    return df


def predict(model, data: Union[pd.DataFrame, Dict, List[Dict]], return_proba: bool = False):
    """
    Make predictions on input data.
    
    Args:
        model: Trained model
        data: Input data for prediction
        return_proba: If True, return prediction probabilities
        
    Returns:
        Predictions (class labels or probabilities)
    """
    # Preprocess input
    X = preprocess_input(data)
    
    # Make predictions
    if return_proba:
        predictions = model.predict_proba(X)
        return predictions
    else:
        predictions = model.predict(X)
        return predictions


def predict_failure_probability(model, data: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
    """
    Predict the probability of manufacturing failure.
    
    Args:
        model: Trained model
        data: Input data for prediction
        
    Returns:
        Array of failure probabilities
    """
    probabilities = predict(model, data, return_proba=True)
    # Return probability of positive class (failure)
    failure_proba = probabilities[:, 1]
    return failure_proba


def batch_predict(model, data_path: Union[str, Path], output_path: Union[str, Path] = None):
    """
    Make predictions on a batch of data from a CSV file.
    
    Args:
        model: Trained model
        data_path: Path to input CSV file
        output_path: Optional path to save predictions
        
    Returns:
        DataFrame with predictions
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Making predictions on {len(df)} samples...")
    predictions = predict(model, df)
    failure_probabilities = predict_failure_probability(model, df)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'prediction': predictions,
        'failure_probability': failure_probabilities
    })
    
    # Save if output path is provided
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    
    return results


def main():
    """
    Example usage of inference module.
    """
    print("=" * 50)
    print("Bosch Quality Classification - Inference")
    print("=" * 50)
    
    # Load model
    model = load_model()
    
    # Example: Single prediction
    example_data = {
        'feature_1': 0.5,
        'feature_2': 1.2,
        'feature_3': -0.3
    }
    
    prediction = predict(model, example_data)
    probability = predict_failure_probability(model, example_data)
    
    print(f"\nExample prediction:")
    print(f"Input: {example_data}")
    print(f"Prediction: {prediction[0]}")
    print(f"Failure probability: {probability[0]:.4f}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
