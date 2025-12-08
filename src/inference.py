"""
Inference module for Bosch Quality Classification.

This module handles loading trained models and making predictions
on new data.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Union, List, Dict

from config import MODEL_PATH


def load_model(model_path=MODEL_PATH):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model object
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from {model_path}")
    return model


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
    
    # Fill missing values with 0 (or median from training)
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
