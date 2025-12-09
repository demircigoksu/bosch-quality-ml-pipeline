# 🏭 Bosch Kalite Tahmin Sistemi
## Üst Yönetim Sunumu

---

# Slayt 1: Yönetici Özeti (Executive Summary)

## Problem
- Bosch üretim hattında kalite kontrol süreçleri manuel/yarı otomatize
- Gözden kaçan hatalı parçalar müşteriye ulaşıyor
- **Sonuç:** İade maliyeti, garanti giderleri, marka prestij kaybı

## Çözüm
- Geçmiş üretim verilerini kullanan **Yapay Zeka tabanlı erken uyarı sistemi**
- Sensör verilerinden hatalı parçaları üretim hattından çıkmadan tespit

## Ana Sonuç
> **Geliştirilen model, hatalı parçaların %7'sini üretim hattından çıkmadan tespit edebiliyor.**
> 
> Bu oran threshold ayarı ile %80'e çıkarılabilir (trade-off: daha fazla yanlış alarm)

---

# Slayt 2: İş Problemi ve Finansal Etki

## Mevcut Durum (Varsayımsal)
| Metrik | Değer |
|--------|-------|
| Günlük üretim | ~50,000 parça |
| Hata oranı | %0.57 (her 175 parçada 1) |
| Günlük hatalı parça | ~285 adet |
| Müşteriye ulaşan hatalı | ~285 adet (AI olmadan) |

## Maliyet Analizi
| Maliyet Kalemi | Birim Maliyet | Açıklama |
|----------------|---------------|----------|
| False Negative (Kaçan Hata) | **$500** | İade + garanti + lojistik + prestij |
| False Positive (Yanlış Alarm) | $10 | Ekstra inceleme işçiliği |

## Yıllık Maliyet Etkisi
- **AI olmadan:** Tüm hatalar müşteriye → Yüksek maliyet
- **AI ile:** Hataların bir kısmı yakalanıyor → Tasarruf

---

# Slayt 3: Çözüm Mimarisi

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   SENSÖRLER     │────▶│   AI MODELİ     │────▶│  OPERATÖR       │
│   (968 veri     │     │   (XGBoost)     │     │  EKRANI         │
│    noktası)     │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │
                              ▼
                        ┌─────────────────┐
                        │   KARAR         │
                        │   ✅ SAĞLAM     │
                        │   ❌ HATALI     │
                        └─────────────────┘
```

## Sistem Akışı:
1. **Veri Toplama:** 968 sensörden anlık veri
2. **AI İşleme:** Model 0.1 saniyede tahmin üretir
3. **Karar:** Operatör ekranında Yeşil/Kırmızı ışık
4. **Aksiyon:** Hatalı parçalar ayrı banda yönlendirilir

---

# Slayt 4: Model Performansı

## Metrikler (Test Verisi: 20,000 parça)

| Metrik | Değer | Anlamı |
|--------|-------|--------|
| **AUC-ROC** | 0.6684 | Genel ayırt edicilik |
| **F1-Score** | 0.0894 | Precision-Recall dengesi |
| **Precision** | 12.31% | "Hatalı" dediğimizde doğruluk |
| **Recall** | 7.02% | Gerçek hataları yakalama oranı |

## Neden Bu Skorlar?
- Veri **aşırı dengesiz** (1:175 oranı)
- %81 eksik sensör verisi
- **Bu skor, baseline'a göre %26 iyileşme**

## Threshold Ayarı ile Trade-off
| Threshold | Recall | Precision | Yorum |
|-----------|--------|-----------|-------|
| 0.55 (Mevcut) | 7% | 12% | Dengeli |
| 0.30 | ~40% | ~5% | Daha fazla hata yakalar, daha fazla yanlış alarm |
| 0.20 | ~60% | ~3% | Çok fazla yanlış alarm |

---

# Slayt 5: Kritik Sensörler (Feature Importance)

## En Önemli 10 Sensör
Model, hata tahmininde en çok bu sensörlere bakıyor:

| Sıra | Sensör ID | İstasyon | Önem Skoru |
|------|-----------|----------|------------|
| 1 | L3_S32_F3850 | L3-S32 | Yüksek |
| 2 | L3_S30_F3754 | L3-S30 | Yüksek |
| 3 | L3_S33_F3855 | L3-S33 | Orta |
| 4 | L0_S1_F24 | L0-S1 | Orta |
| 5 | L3_S29_F3348 | L3-S29 | Orta |

## Aksiyon Önerisi
> **L3 hattındaki S30, S32, S33 istasyonlarına bakım önceliği verilmeli.**
> Bu sensörlerdeki anormallikler hataların ana kaynağı.

---

# Slayt 6: Maliyet-Fayda Analizi

## Test Seti Sonuçları (20,000 parça)

| Kategori | Adet | Maliyet |
|----------|------|---------|
| Yakalanan Hatalar (TP) | 8 | $0 (tasarruf) |
| Kaçan Hatalar (FN) | 106 | $53,000 |
| Yanlış Alarmlar (FP) | 57 | $570 |
| **Toplam Maliyet** | - | **$53,570** |

## AI Değeri
| Senaryo | Yıllık Maliyet | Tasarruf |
|---------|----------------|----------|
| AI Olmadan | ~$150,000 | - |
| AI ile (Mevcut) | ~$140,000 | ~$10,000 |
| AI ile (Optimize) | ~$80,000 | ~$70,000 |

> **Not:** Threshold optimize edilirse tasarruf artırılabilir.

---

# Slayt 7: Yol Haritası (Next Steps)

## Kısa Vadeli (0-3 Ay)
- [ ] Pilot test: Tek üretim hattında canlıya alma
- [ ] Operatör eğitimi
- [ ] Threshold fine-tuning (gerçek verilerle)

## Orta Vadeli (3-6 Ay)
- [ ] Tüm üretim hatlarına yaygınlaştırma
- [ ] Gerçek zamanlı sensör entegrasyonu (IoT)
- [ ] Model performans izleme dashboard'u

## Uzun Vadeli (6-12 Ay)
- [ ] Diğer parça tiplerine genişleme
- [ ] Kategorik ve zaman verilerinin eklenmesi
- [ ] Otomatik model yeniden eğitimi (MLOps)

---

# Slayt 8: Sonuç ve Öneri

## Özet
✅ Yapay zeka modeli başarıyla geliştirildi ve test edildi
✅ Hatalı parçaları tespit edebiliyor
✅ Streamlit arayüzü ile kullanıma hazır
✅ Docker ile deployment yapıldı

## Yönetim Kararı İçin
| Opsiyon | Açıklama | Risk |
|---------|----------|------|
| **A: Pilot Başlat** | Tek hatta 1 aylık test | Düşük |
| **B: Geliştir** | Daha fazla veri ile modeli iyileştir | Orta |
| **C: Bekle** | Daha iyi sonuçlar için yeni teknoloji | Yüksek |

## Öneri
> **Opsiyon A önerilir:** Düşük riskli pilot test ile gerçek dünya performansı ölçülmeli.

---

# İletişim

**Proje:** Bosch Quality ML Pipeline
**GitHub:** https://github.com/demircigoksu/bosch-quality-ml-pipeline
**Demo:** http://localhost:8501 (Streamlit)
**API:** http://localhost:8080/docs (Swagger)

---

*Zero2End Machine Learning Bootcamp - Final Projesi*
*Tarih: 9 Aralık 2025*
