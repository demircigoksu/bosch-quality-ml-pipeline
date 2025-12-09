"""
FastAPI application for Bosch Quality Prediction.

This module provides a REST API for making manufacturing failure predictions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from inference import load_model, predict_failure_probability
from config import API_TITLE, API_VERSION, MODEL_PATH

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="API for predicting manufacturing failures in Bosch quality dataset"
)

# Load model at startup
model = None


@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts."""
    global model
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Model will need to be trained first using src/train.py")


class FeatureData(BaseModel):
    """Schema for input features."""
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature names and their values",
        example={"feature_1": 0.5, "feature_2": 1.2, "feature_3": -0.3}
    )


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    prediction: int = Field(
        ...,
        description="Binary prediction: 0 (no failure) or 1 (failure)"
    )
    probability: float = Field(
        ...,
        description="Probability of manufacturing failure (0-1)",
        ge=0.0,
        le=1.0
    )
    status: str = Field(
        default="success",
        description="Status of the prediction"
    )


class BatchFeatureData(BaseModel):
    """Schema for batch prediction input."""
    samples: List[Dict[str, float]] = Field(
        ...,
        description="List of sample feature dictionaries",
        example=[
            {"feature_1": 0.5, "feature_2": 1.2},
            {"feature_1": -0.3, "feature_2": 0.8}
        ]
    )


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions for each sample"
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Bosch Quality Prediction API",
        "version": API_VERSION,
        "endpoints": {
            "/health": "Health check",
            "/predict": "Single prediction",
            "/predict/batch": "Batch predictions",
            "/docs": "Interactive API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(data: FeatureData):
    """
    Make a single prediction.
    
    Args:
        data: Feature data for prediction
        
    Returns:
        Prediction response with failure probability and binary prediction
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using src/train.py"
        )
    
    try:
        # Make prediction
        df = pd.DataFrame([data.features])
        failure_prob = predict_failure_probability(model, df)[0]
        prediction = 1 if failure_prob > 0.5 else 0
        
        return PredictionResponse(
            prediction=prediction,
            probability=float(failure_prob)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(data: BatchFeatureData):
    """
    Make batch predictions.
    
    Args:
        data: Batch feature data for predictions
        
    Returns:
        Batch prediction response with all predictions
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using src/train.py"
        )
    
    try:
        # Make predictions
        df = pd.DataFrame(data.samples)
        failure_probs = predict_failure_probability(model, df)
        predictions = [1 if prob > 0.5 else 0 for prob in failure_probs]
        
        # Create response
        responses = [
            PredictionResponse(
                prediction=pred,
                probability=float(prob)
            )
            for prob, pred in zip(failure_probs, predictions)
        ]
        
        return BatchPredictionResponse(predictions=responses)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    
    print(f"Starting Bosch Quality Prediction API...")
    print(f"API documentation available at http://{API_HOST}:{API_PORT}/docs")
    
    uvicorn.run(app, host=API_HOST, port=API_PORT)
