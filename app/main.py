"""
Bosch Kalite Tahmin REST API.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from inference import load_model, predict_failure_probability
from config import API_TITLE, API_VERSION, MODEL_PATH

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Bosch üretim hattı kalite tahmini API'si"
)

model = None


@app.on_event("startup")
async def startup_event():
    """Uygulama başlarken modeli yükle."""
    global model
    try:
        model = load_model(MODEL_PATH)
        print("Model yüklendi")
    except Exception as e:
        print(f"Model yüklenemedi: {e}")


class FeatureData(BaseModel):
    """Girdi feature'ları."""
    features: Dict[str, float] = Field(
        ...,
        description="Feature değerleri",
        example={"feature_1": 0.5, "feature_2": 1.2}
    )


class PredictionResponse(BaseModel):
    """Tahmin sonucu."""
    prediction: int = Field(..., description="0: sağlam, 1: hatalı")
    probability: float = Field(..., description="Hata olasılığı", ge=0.0, le=1.0)
    status: str = Field(default="success")


class BatchFeatureData(BaseModel):
    """Toplu tahmin girdisi."""
    samples: List[Dict[str, float]] = Field(
        ...,
        description="Örnek listesi",
        example=[{"feature_1": 0.5}, {"feature_1": 1.2}]
    )


class BatchPredictionResponse(BaseModel):
    """Toplu tahmin sonucu."""
    predictions: List[PredictionResponse] = Field(...)


@app.get("/")
async def root():
    """API bilgisi."""
    return {
        "message": "Bosch Kalite Tahmin API",
        "version": API_VERSION,
        "endpoints": {
            "/health": "Sağlık kontrolü",
            "/predict": "Tekil tahmin",
            "/predict/batch": "Toplu tahmin",
            "/docs": "API dokümantasyonu"
        }
    }


@app.get("/health")
async def health_check():
    """Sağlık kontrolü."""
    model_status = "loaded" if model is not None else "not loaded"
    return {"status": "healthy", "model_status": model_status}


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(data: FeatureData):
    """Tekil tahmin yap."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model yüklü değil")
    
    try:
        df = pd.DataFrame([data.features])
        failure_prob = predict_failure_probability(model, df)[0]
        prediction = 1 if failure_prob > 0.5 else 0
        
        return PredictionResponse(prediction=prediction, probability=float(failure_prob))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(data: BatchFeatureData):
    """Toplu tahmin yap."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model yüklü değil")
    
    try:
        df = pd.DataFrame(data.samples)
        failure_probs = predict_failure_probability(model, df)
        predictions = [1 if prob > 0.5 else 0 for prob in failure_probs]
        
        responses = [
            PredictionResponse(prediction=pred, probability=float(prob))
            for prob, pred in zip(failure_probs, predictions)
        ]
        
        return BatchPredictionResponse(predictions=responses)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    
    print(f"API başlatılıyor...")
    print(f"Dokümantasyon: http://{API_HOST}:{API_PORT}/docs")
    
    uvicorn.run(app, host=API_HOST, port=API_PORT)
