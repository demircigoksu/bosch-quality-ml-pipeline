# 📊 Bosch Kalite Tahmin Modeli - Detaylı Teknik Rapor

**Proje:** Bosch Production Line Performance  
**Tarih:** 9 Aralık 2025  
**Versiyon:** 1.0  

---

## 1. Yönetici Özeti

Bu rapor, Bosch üretim hattı kalite tahmin projesinin teknik detaylarını, model performansını ve iş önerilerini içermektedir.

### Temel Bulgular
- **Veri Seti:** 1.2M satır, 970 sütun (100K örneklem kullanıldı)
- **Sınıf Dengesizliği:** 1:175 (çok aşırı dengesiz)
- **Eksik Veri:** Ortalama %81
- **Final Model:** XGBoost + SMOTE + Threshold Optimization
- **AUC-ROC:** 0.6684 | **F1-Score:** 0.0894

---

## 2. Veri Seti Analizi

### 2.1 Veri Kaynağı
- **Platform:** Kaggle Competition
- **Dosya:** train_numeric.csv (1.99 GB)
- **Örnekleme:** İlk 100,000 satır (RAM kısıtı nedeniyle)

### 2.2 Hedef Değişken Dağılımı
```
Response = 0 (Sağlam): 99,432 (%99.43)
Response = 1 (Hatalı):    568 (%0.57)
Dengesizlik Oranı: 1:175
```

### 2.3 Eksik Veri Analizi
| Kategori | Oran |
|----------|------|
| Ortalama eksik | %81 |
| %90+ eksik sütunlar | 610 sütun (kaldırıldı) |
| Kalan sütunlar | 358 |

### 2.4 Üretim Hattı Yapısı
```
L0: 12 istasyon (S0-S11)
L1: 8 istasyon (S12-S19)
L2: 4 istasyon (S20-S23)
L3: 27 istasyon (S24-S51) - En büyük hat
```

---

## 3. Feature Engineering

### 3.1 Oluşturulan Özellikler (24 yeni feature)

| Kategori | Özellik | Açıklama |
|----------|---------|----------|
| **Satır İstatistikleri** | row_mean | Satır ortalaması |
| | row_std | Satır standart sapması |
| | row_min/max | Min/max değerler |
| | row_non_null | Dolu hücre sayısı |
| **İstasyon Bazlı** | station_X_mean | Her istasyonun ortalaması |
| | station_X_std | Her istasyonun std sapması |
| **Eksik Veri Pattern** | missing_ratio | Eksik veri oranı |

### 3.2 Veri Ön İşleme
1. **%90+ eksik sütunları kaldır** → 610 sütun silindi
2. **Kalan eksik verileri -999 ile doldur** (XGBoost missing handle eder)
3. **SMOTE ile oversampling** → 1:175 → 1:3 oranına

---

## 4. Model Geliştirme Süreci

### 4.1 Baseline Model
```python
XGBClassifier(
    scale_pos_weight=175,  # Sınıf ağırlığı
    max_depth=6,
    n_estimators=100
)
```
**Sonuç:** AUC: 0.6655, F1: 0.0711

### 4.2 Optimize Edilmiş Model
```python
XGBClassifier(
    scale_pos_weight=175,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=300,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50
)
```
**+ SMOTE + Threshold Optimization (0.55)**

**Sonuç:** AUC: 0.6684, F1: 0.0894 (+26% iyileşme)

### 4.3 Hiperparametre Arama
- **Yöntem:** GridSearchCV
- **CV:** Stratified 3-Fold
- **Metrik:** AUC-ROC

---

## 5. Model Performansı

### 5.1 Metrik Karşılaştırması

| Metrik | Baseline | Final | Değişim |
|--------|----------|-------|---------|
| AUC-ROC | 0.6655 | 0.6684 | +0.4% |
| F1-Score | 0.0711 | 0.0894 | **+25.7%** |
| Precision | 0.0411 | 0.1231 | +199.5% |
| Recall | 0.2632 | 0.0702 | -73.3% |

### 5.2 Confusion Matrix (Threshold=0.55)

```
                 Tahmin: Sağlam    Tahmin: Hatalı
Gerçek: Sağlam      19,829            57
Gerçek: Hatalı         106             8
```

### 5.3 Metrik Yorumları

**Precision (12.31%):**
- Model "hatalı" dediğinde %12.31 doğru
- 8 doğru hatalı tespit / 65 toplam hatalı tahmini

**Recall (7.02%):**
- Gerçek hataların %7.02'sini yakalıyor
- 8 yakalanan / 114 gerçek hatalı

**Düşük Skorların Nedeni:**
1. Aşırı dengesiz veri (1:175)
2. %81 eksik veri
3. Sınırlı örneklem (100K)

---

## 6. Maliyet Analizi

### 6.1 Birim Maliyetler (Varsayımsal)
| Hata Tipi | Maliyet | Açıklama |
|-----------|---------|----------|
| False Positive | $10 | Gereksiz inceleme işçiliği |
| False Negative | $500 | İade + garanti + lojistik + prestij |

### 6.2 Test Seti Maliyet Hesabı

```
True Negative (TN):  19,829 parça → $0 (sorun yok)
True Positive (TP):       8 parça → $0 (başarılı tespit)
False Positive (FP):     57 parça → $570 (gereksiz inceleme)
False Negative (FN):    106 parça → $53,000 (kaçan hatalar)

TOPLAM MALİYET: $53,570
```

### 6.3 ROI Analizi

| Senaryo | Maliyet | Tasarruf |
|---------|---------|----------|
| AI Olmadan | $57,000 (tüm hatalar müşteriye) | - |
| AI ile | $53,570 | $3,430 (%6) |

**Not:** Threshold düşürülerek recall artırılabilir, ancak FP maliyeti artar.

### 6.4 Threshold Senaryoları

| Threshold | Recall | FP | FN | Toplam Maliyet |
|-----------|--------|-----|-----|----------------|
| 0.55 | 7% | 57 | 106 | $53,570 |
| 0.40 | 15% | 150 | 97 | $50,000 |
| 0.30 | 25% | 300 | 85 | $45,500 |
| 0.20 | 40% | 600 | 68 | $40,000 |

---

## 7. Feature Importance

### 7.1 En Önemli 20 Özellik

| Sıra | Feature | Importance | İstasyon |
|------|---------|------------|----------|
| 1 | L3_S32_F3850 | 0.045 | L3-S32 |
| 2 | L3_S30_F3754 | 0.038 | L3-S30 |
| 3 | L3_S33_F3855 | 0.032 | L3-S33 |
| 4 | row_mean | 0.028 | (Türetilmiş) |
| 5 | L0_S1_F24 | 0.025 | L0-S1 |
| ... | ... | ... | ... |

### 7.2 İstasyon Bazlı Analiz

```
L3 Hattı: %60 önem (Kritik!)
L0 Hattı: %20 önem
L1 Hattı: %12 önem
L2 Hattı: %8 önem
```

### 7.3 Aksiyon Önerileri

1. **L3-S30, S32, S33 istasyonları:** Öncelikli bakım
2. **L0-S1 istasyonu:** İkincil öncelik
3. **Türetilmiş özellikler:** row_mean yüksek önem → genel sensör ortalaması kritik

---

## 8. Teknik Altyapı

### 8.1 Teknoloji Stack'i
| Bileşen | Teknoloji |
|---------|-----------|
| ML Framework | XGBoost, scikit-learn |
| Oversampling | imbalanced-learn (SMOTE) |
| API | FastAPI |
| UI | Streamlit |
| Deployment | Docker, docker-compose |
| Versiyon Kontrolü | Git, GitHub |

### 8.2 API Endpoints
| Endpoint | Method | Açıklama |
|----------|--------|----------|
| /health | GET | Sağlık kontrolü |
| /predict | POST | Tek tahmin |
| /predict/batch | POST | Toplu tahmin |
| /docs | GET | Swagger UI |

### 8.3 API Response Formatı
```json
{
  "prediction": 1,
  "probability": 0.85
}
```

---

## 9. Kısıtlar ve İyileştirme Önerileri

### 9.1 Mevcut Kısıtlar
1. Sadece numerik veriler kullanıldı (categorical, date hariç)
2. 100K örneklem (1.2M'in %8'i)
3. SHAP analizi için ek kütüphane gerekli

### 9.2 İyileştirme Önerileri
| Öneri | Beklenen Etki | Zorluk |
|-------|---------------|--------|
| Tüm veri kullanımı | +5-10% AUC | Yüksek (RAM) |
| Kategorik veri ekleme | +3-5% AUC | Orta |
| Zaman verisi ekleme | +2-4% AUC | Orta |
| Ensemble (LightGBM+XGB) | +1-3% AUC | Düşük |
| Derin Öğrenme | ? | Çok Yüksek |

---

## 10. Sonuç

### 10.1 Başarılar
✅ End-to-end ML pipeline tamamlandı
✅ Baseline'a göre %26 F1 iyileşmesi
✅ Production-ready deployment (Docker)
✅ Kullanıcı dostu UI (Streamlit)
✅ REST API (FastAPI)

### 10.2 Proje Teslim Durumu
| Gereksinim | Durum |
|------------|-------|
| EDA Notebook | ✅ |
| Baseline Model | ✅ |
| Feature Engineering | ✅ |
| Hiperparametre Opt. | ✅ |
| Final Pipeline | ✅ |
| Streamlit UI | ✅ |
| FastAPI | ✅ |
| Docker Deployment | ✅ |
| GitHub Repo | ✅ |
| README.md | ✅ |
| Sunum Slaytları | ✅ |

### 10.3 Sonraki Adımlar
1. Pilot test (tek üretim hattı)
2. Gerçek zamanlı veri entegrasyonu
3. Model izleme ve yeniden eğitim pipeline'ı

---

**Rapor Sonu**

*Zero2End Machine Learning Bootcamp - Final Projesi*
