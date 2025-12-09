"""
Tahmin (inference) modülü.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Union, List, Dict

# Model dosya yolları
MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "final_model.pkl"
FEATURES_PATH = MODELS_DIR / "feature_names.pkl"
CONFIG_PATH = MODELS_DIR / "model_config.pkl"


def load_model(model_path=None):
    """Eğitilmiş modeli yükle."""
    if model_path is None:
        model_path = MODEL_PATH
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")
    
    model = joblib.load(model_path)
    print(f"Model yüklendi: {model_path}")
    return model


def load_feature_names():
    """Feature isimlerini yükle."""
    if FEATURES_PATH.exists():
        return joblib.load(FEATURES_PATH)
    return None


def load_config():
    """Model ayarlarını yükle."""
    if CONFIG_PATH.exists():
        return joblib.load(CONFIG_PATH)
    return {'threshold': 0.35}


def apply_feature_engineering(df):
    """Veri üzerinde feature engineering uygula."""
    X = df.copy()
    
    original_cols = [c for c in df.columns if not c.startswith('row_')]
    
    # satır bazlı istatistikler
    X['row_mean'] = df[original_cols].mean(axis=1)
    X['row_std'] = df[original_cols].std(axis=1)
    X['row_min'] = df[original_cols].min(axis=1)
    X['row_max'] = df[original_cols].max(axis=1)
    X['row_range'] = X['row_max'] - X['row_min']
    X['row_nonzero'] = (df[original_cols] != 0).sum(axis=1)
    
    return X


def preprocess_input(data: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
    """Girdi verisini tahmin için hazırla."""
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Id ve Response varsa çıkar
    for col in ['Id', 'Response']:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    df = apply_feature_engineering(df)
    df = df.fillna(0)
    
    return df


def predict(model, data: Union[pd.DataFrame, Dict, List[Dict]], return_proba: bool = False):
    """Tahmin yap."""
    X = preprocess_input(data)
    
    if return_proba:
        return model.predict_proba(X)
    else:
        return model.predict(X)


def predict_failure_probability(model, data: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
    """Hata olasılığını hesapla."""
    probabilities = predict(model, data, return_proba=True)
    return probabilities[:, 1]


def batch_predict(model, data_path: Union[str, Path], output_path: Union[str, Path] = None):
    """CSV dosyasından toplu tahmin yap."""
    print(f"Veri yükleniyor: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"{len(df)} örnek için tahmin yapılıyor...")
    predictions = predict(model, df)
    failure_probabilities = predict_failure_probability(model, df)
    
    results = pd.DataFrame({
        'prediction': predictions,
        'failure_probability': failure_probabilities
    })
    
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"Sonuçlar kaydedildi: {output_path}")
    
    return results


def main():
    """Örnek kullanım."""
    print("=" * 50)
    print("Bosch Kalite - Tahmin Modülü")
    print("=" * 50)
    
    model = load_model()
    
    example_data = {
        'feature_1': 0.5,
        'feature_2': 1.2,
        'feature_3': -0.3
    }
    
    prediction = predict(model, example_data)
    probability = predict_failure_probability(model, example_data)
    
    print(f"\nÖrnek tahmin:")
    print(f"Girdi: {example_data}")
    print(f"Tahmin: {prediction[0]}")
    print(f"Hata olasılığı: {probability[0]:.4f}")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
