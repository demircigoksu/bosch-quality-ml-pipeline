"""
Model eğitim modülü.
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
    RANDOM_FOREST_PARAMS,
    MAX_MISSING_THRESHOLD
)


def load_data(data_path=TRAIN_DATA_PATH):
    """Eğitim verisini yükle."""
    print(f"Veri yükleniyor: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    return df


def preprocess_data(df, target_col='Response'):
    """Veriyi ön işle."""
    print("Veri işleniyor...")
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # çok eksik veri olan sütunları çıkar
    missing_pct = X.isnull().sum() / len(X)
    cols_to_keep = missing_pct[missing_pct < MAX_MISSING_THRESHOLD].index
    X = X[cols_to_keep]
    print(f"Kalan sütun sayısı: {X.shape[1]}")
    
    # kalan eksikleri medyan ile doldur
    X = X.fillna(X.median())
    
    print(f"Son feature boyutu: {X.shape}")
    print(f"Hedef dağılımı:\n{y.value_counts()}")
    
    return X, y


def train_model(X_train, y_train, model_type='xgboost'):
    """Model eğit."""
    print(f"{model_type} eğitiliyor...")
    
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    else:
        raise ValueError(f"Bilinmeyen model: {model_type}")
    
    model.fit(X_train, y_train)
    print("Eğitim tamamlandı.")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Model performansını değerlendir."""
    print("Değerlendirme...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))
    
    return {'roc_auc': roc_auc, 'mcc': mcc}


def save_model(model, model_path=MODEL_PATH):
    """Modeli diske kaydet."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model kaydedildi: {model_path}")


def main():
    """Ana eğitim akışı."""
    print("=" * 50)
    print("Bosch Kalite Sınıflandırma - Eğitim")
    print("=" * 50)
    
    df = load_data()
    X, y = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} örnek")
    print(f"Test: {X_test.shape[0]} örnek")
    
    model = train_model(X_train, y_train, model_type='xgboost')
    evaluate_model(model, X_test, y_test)
    save_model(model)
    
    print("\n" + "=" * 50)
    print("Tamamlandı!")
    print("=" * 50)


if __name__ == "__main__":
    main()
