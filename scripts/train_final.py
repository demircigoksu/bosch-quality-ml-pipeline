"""
Final Model Eğitim Scripti (Feature Engineering dahil).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, f1_score, precision_score, recall_score,
    precision_recall_curve
)
from xgboost import XGBClassifier
import joblib
import os

print("=" * 60)
print("FINAL MODEL EĞİTİMİ")
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

print("   - Satır bazlı istatistikler ekleniyor...")
X['row_mean'] = X.mean(axis=1)
X['row_std'] = X.std(axis=1)
X['row_min'] = X.min(axis=1)
X['row_max'] = X.max(axis=1)
X['row_range'] = X['row_max'] - X['row_min']

print("   - İstasyon bazlı özellikler ekleniyor...")
stations = set()
for col in X.columns:
    if col.startswith('L') and '_S' in col:
        parts = col.split('_')
        if len(parts) >= 2:
            stations.add(f"{parts[0]}_{parts[1]}")

for station in list(stations)[:10]:
    station_cols = [c for c in X.columns if c.startswith(station + '_')]
    if station_cols:
        X[f'{station}_mean'] = X[station_cols].mean(axis=1)

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
    max_depth=8,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1,
    verbosity=1
)

print("   Eğitim başlıyor...")
model.fit(X_train, y_train)
print("   ✅ Eğitim tamamlandı!")

# 6. Threshold Optimizasyonu
print(f"\n5. Threshold Optimizasyonu:")
y_pred_proba = model.predict_proba(X_test)[:, 1]

precision_arr, recall_arr, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-10)
optimal_idx = np.argmax(f1_scores[:-1])
optimal_threshold = thresholds[optimal_idx]

print(f"   Optimal Threshold: {optimal_threshold:.4f}")
print(f"   (Varsayılan 0.5 yerine)")

# Optimal threshold ile tahmin
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

# 7. Değerlendirme
print(f"\n6. FINAL MODEL SONUÇLARI:")
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
print(f"   Yakalanan Hatalar: {caught_failures} ({100*caught_failures/total_failures:.1f}%)")
print(f"   Kaçırılan Hatalar: {missed_failures} ({100*missed_failures/total_failures:.1f}%)")
print(f"\n   ⚠️ False Positive (Gereksiz İnceleme): {fp:,}")
print(f"   ❌ False Negative (KRİTİK - Müşteriye Ulaşan Hata): {fn}")

# 9. Feature Importance
print(f"\n8. EN ÖNEMLİ 15 ÖZELLİK:")
print("=" * 50)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(15).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# 10. Model Kaydetme
os.makedirs('models', exist_ok=True)
MODEL_PATH = 'models/final_model.pkl'
joblib.dump(model, MODEL_PATH)
print(f"\n9. Model kaydedildi: {MODEL_PATH}")

# Feature names ve threshold kaydet
FEATURES_PATH = 'models/feature_names_final.pkl'
THRESHOLD_PATH = 'models/optimal_threshold.pkl'
joblib.dump(X.columns.tolist(), FEATURES_PATH)
joblib.dump(optimal_threshold, THRESHOLD_PATH)
print(f"   Feature names: {FEATURES_PATH}")
print(f"   Optimal threshold: {THRESHOLD_PATH}")

# Karşılaştırma
print(f"\n10. BASELINE vs FINAL KARŞILAŞTIRMASI:")
print("=" * 50)
print(f"   Baseline F1-Score:  0.0189")
print(f"   Final F1-Score:     {f1:.4f}")
print(f"   İyileşme:           {(f1/0.0189 - 1)*100:.1f}%")

print("\n" + "=" * 60)
print("İŞLEM TAMAMLANDI")
print("=" * 60)
