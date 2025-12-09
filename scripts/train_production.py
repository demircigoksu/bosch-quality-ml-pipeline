"""
Production-Ready Model Training Script
Recall odaklı model - hataları kaçırmamak öncelikli
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, f1_score, precision_score, recall_score,
    precision_recall_curve
)
from xgboost import XGBClassifier
import joblib
import os

print("=" * 60)
print("PRODUCTION MODEL EĞİTİMİ (Recall Odaklı)")
print("=" * 60)

# 1. Veri Yükleme
DATA_PATH = 'data/train_numeric_clean.csv'
print(f"\n1. Veri yükleniyor: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"   Boyut: {df.shape[0]:,} satır x {df.shape[1]:,} sütun")

# 2. Feature ve Target ayırma
X = df.drop(['Id', 'Response'], axis=1)
y = df['Response']

# 3. Feature Engineering
print(f"\n2. Feature Engineering...")
X['row_mean'] = X.mean(axis=1)
X['row_std'] = X.std(axis=1)
X['row_min'] = X.min(axis=1)
X['row_max'] = X.max(axis=1)
X['row_range'] = X['row_max'] - X['row_min']
X['row_nonzero'] = (X != 0).sum(axis=1)

print(f"   Feature sayısı: {X.shape[1]}")

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"\n3. Train/Test Split:")
print(f"   Train: {X_train.shape[0]:,} satır ({y_train.sum():,} hatalı)")
print(f"   Test:  {X_test.shape[0]:,} satır ({y_test.sum():,} hatalı)")

# 5. Model Eğitimi
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n4. Model Eğitimi:")
print(f"   scale_pos_weight: {scale_pos_weight:.1f}")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.05,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1,
    verbosity=1
)

print("   Eğitim başlıyor...")
model.fit(X_train, y_train)
print("   ✅ Eğitim tamamlandı!")

# 6. Farklı threshold değerlerini test et
print(f"\n5. Threshold Analizi:")
print("=" * 50)
y_pred_proba = model.predict_proba(X_test)[:, 1]

thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5]
best_threshold = 0.5
best_f1 = 0

print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 50)

for thresh in thresholds_to_test:
    y_pred_temp = (y_pred_proba >= thresh).astype(int)
    prec = precision_score(y_test, y_pred_temp, zero_division=0)
    rec = recall_score(y_test, y_pred_temp, zero_division=0)
    f1_temp = f1_score(y_test, y_pred_temp, zero_division=0)
    
    print(f"{thresh:<12.2f} {prec:<12.4f} {rec:<12.4f} {f1_temp:<12.4f}")
    
    if f1_temp > best_f1:
        best_f1 = f1_temp
        best_threshold = thresh

# Dengeli threshold seç (Recall ~%50, Precision makul)
recall_threshold = 0.35  # Dengeli threshold
print(f"\n   Seçilen Threshold: {recall_threshold}")

# Final tahmin
y_pred = (y_pred_proba >= recall_threshold).astype(int)

# 7. Değerlendirme
print(f"\n6. PRODUCTION MODEL SONUÇLARI:")
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

# 8. İş Değeri Analizi
print(f"\n7. İŞ DEĞERİ ANALİZİ:")
print("=" * 50)
tn, fp, fn, tp = cm.ravel()
total_failures = fn + tp
caught_failures = tp
missed_failures = fn

print(f"   Toplam Hatalı Parça: {total_failures}")
print(f"   ✅ Yakalanan Hatalar: {caught_failures} ({100*caught_failures/total_failures:.1f}%)")
print(f"   ❌ Kaçırılan Hatalar: {missed_failures} ({100*missed_failures/total_failures:.1f}%)")
print(f"\n   ⚠️ False Positive (Gereksiz İnceleme): {fp:,}")
print(f"   ❌ False Negative (KRİTİK): {fn}")

# 9. Model ve Config Kaydetme
os.makedirs('models', exist_ok=True)

# Model kaydet
MODEL_PATH = 'models/final_model.pkl'
joblib.dump(model, MODEL_PATH)
print(f"\n8. Model kaydedildi: {MODEL_PATH}")

# Feature names kaydet
FEATURES_PATH = 'models/feature_names.pkl'
joblib.dump(X.columns.tolist(), FEATURES_PATH)
print(f"   Features: {FEATURES_PATH}")

# Config kaydet
config = {
    'threshold': recall_threshold,
    'feature_count': X.shape[1],
    'model_type': 'XGBClassifier',
    'auc_roc': auc_score,
    'f1_score': f1,
    'recall': recall,
    'precision': precision
}
CONFIG_PATH = 'models/model_config.pkl'
joblib.dump(config, CONFIG_PATH)
print(f"   Config: {CONFIG_PATH}")

print("\n" + "=" * 60)
print("İŞLEM TAMAMLANDI")
print("=" * 60)
