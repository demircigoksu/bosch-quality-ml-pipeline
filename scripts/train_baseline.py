"""
Baseline Model Eğitim Scripti.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, f1_score, precision_score, recall_score
)
from xgboost import XGBClassifier
import joblib
import os

print("=" * 60)
print("BASELINE MODEL EĞİTİMİ")
print("=" * 60)

# 1. Veri Yükleme
DATA_PATH = 'data/train_numeric_clean.csv'
print(f"\n1. Veri yükleniyor: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"   Boyut: {df.shape[0]:,} satır x {df.shape[1]:,} sütun")

# 2. Feature ve Target ayırma
X = df.drop(['Id', 'Response'], axis=1)
y = df['Response']

print(f"\n2. Hedef değişken dağılımı:")
print(f"   Sağlam (0): {(y==0).sum():,}")
print(f"   Hatalı (1): {(y==1).sum():,}")
print(f"   Dengesizlik oranı: 1:{int((y==0).sum()/(y==1).sum())}")

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"\n3. Train/Test Split:")
print(f"   Train: {X_train.shape[0]:,} satır ({y_train.sum():,} hatalı)")
print(f"   Test:  {X_test.shape[0]:,} satır ({y_test.sum():,} hatalı)")

# 4. Model Eğitimi
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n4. Model Eğitimi:")
print(f"   scale_pos_weight: {scale_pos_weight:.1f}")

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1,
    verbosity=1
)

print("   Eğitim başlıyor...")
model.fit(X_train, y_train)
print("   ✅ Eğitim tamamlandı!")

# 5. Değerlendirme
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"\n5. MODEL SONUÇLARI:")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=['Sağlam (0)', 'Hatalı (1)']))

auc_score = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\n📊 Önemli Metrikler:")
print(f"   AUC-ROC:   {auc_score:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n📋 Confusion Matrix:")
print(f"   TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
print(f"   FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

# 6. Model Kaydetme
os.makedirs('models', exist_ok=True)
MODEL_PATH = 'models/baseline_model.pkl'
joblib.dump(model, MODEL_PATH)
print(f"\n6. Model kaydedildi: {MODEL_PATH}")

# Feature names da kaydet
FEATURES_PATH = 'models/feature_names.pkl'
joblib.dump(X.columns.tolist(), FEATURES_PATH)
print(f"   Feature names kaydedildi: {FEATURES_PATH}")

print("\n" + "=" * 60)
print("İŞLEM TAMAMLANDI")
print("=" * 60)
