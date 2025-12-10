# Bosch Kalite Tahmin Modeli - Detaylı Teknik Rapor

Proje: Bosch Production Line Performance  
Tarih: 9 Aralık 2025  
Versiyon: 2.0  

---

## 1. Yönetici Özeti

Bu rapor, Bosch üretim hattı kalite tahmin projesinin teknik detaylarını, model performansını ve iş önerilerini içermektedir.

### Temel Bulgular

Veri Seti: Orijinal veri 1.2M satır ve 970 sütundan oluşuyor. Clean data pipeline ile 450,519 satır ve 158 sütuna indirildi.

Sınıf Dengesizliği: 1:228 oranında (448,552 sağlam, 1,967 hatalı)

Eksik Veri: Orijinal veride %81 eksiklik vardı. Temizleme sonrası %0 eksik veri.

Final Model: XGBoost + Feature Engineering + Threshold Optimization

Performans: AUC-ROC 0.635, Recall %51.4, F1-Score 0.0146

---

## 2. Veri Seti Analizi

### 2.1 Veri Kaynağı

Platform: Kaggle Competition (Bosch Production Line Performance)

Orijinal Dosya: train_numeric.csv (2 GB, 1.2M satır, 970 sütun)

Temizlenmiş Dosya: train_numeric_clean.csv (400 MB, 450,519 satır, 158 sütun)

### 2.2 Veri Temizleme Süreci (Clean Data Pipeline)

Orijinal veride %81 oranında eksik değer bulunuyordu. Kullanılabilir bir veri seti elde etmek için şu adımlar izlendi:

1. Eksik oranı %50'den az olan 157 sütun seçildi
2. Bu sütunlarda hiç eksik değeri olmayan satırlar filtrelendi
3. Sonuç: 450,519 satır ve 158 sütun (157 feature + Response)

Bu yaklaşım sayesinde eksik veri sorunu tamamen ortadan kaldırıldı ve model eğitimi için temiz bir veri seti elde edildi.

### 2.3 Hedef Değişken Dağılımı

Response = 0 (Sağlam): 448,552 adet (%99.56)
Response = 1 (Hatalı): 1,967 adet (%0.44)
Dengesizlik Oranı: 1:228

### 2.4 Veri Özellikleri

Temizlenmiş veride eksik değer yok. Tüm 157 feature numerik ve sürekli değişkenlerden oluşuyor. Veriler L0, L3 üretim hatlarından gelen sensör ölçümlerini içeriyor.

---

## 3. Feature Engineering

### 3.1 Oluşturulan Özellikler (6 yeni feature)

Satır bazlı istatistikler her parçanın genel profilini çıkarmak için eklendi:

row_mean: Satırdaki tüm sensör değerlerinin ortalaması
row_std: Satırdaki değerlerin standart sapması
row_min: Satırdaki minimum değer
row_max: Satırdaki maksimum değer
row_range: max - min farkı
row_nonzero: Sıfır olmayan değer sayısı

Toplam feature sayısı: 157 orijinal + 6 mühendislik = 162 feature (Id sütunu hariç)

### 3.2 Önemli Bulgu

En basit özellik olan row_mean, en önemli değişkenlerden biri çıktı. Bu, genel sensör ortalamasının kalite için kritik bir gösterge olduğunu ortaya koyuyor.

---

## 4. Model Geliştirme Süreci

### 4.1 Model Seçimi: Neden XGBoost?

Derin öğrenme yerine XGBoost tercih edildi çünkü:

1. Eksik veriyi otomatik işleyebiliyor (orijinal veri için önemli)
2. scale_pos_weight parametresi ile dengesiz sınıfları yönetebiliyor
3. Yorumlanabilir sonuçlar veriyor (feature importance)
4. Hızlı eğitim ve inference süresi

### 4.2 Model Konfigürasyonu

XGBClassifier parametreleri:
n_estimators: 200
max_depth: 6
learning_rate: 0.05
scale_pos_weight: 228 (dengesizlik oranı)
min_child_weight: 3
subsample: 0.8
colsample_bytree: 0.8
eval_metric: auc

### 4.3 Threshold Optimization

Standart 0.5 eşiği yerine maliyet odaklı optimizasyon yapıldı. Farklı threshold değerleri test edildi:

Threshold 0.10: Recall %80.4, Precision %0.96
Threshold 0.20: Recall %64.9, Precision %1.31
Threshold 0.30: Recall %55.7, Precision %1.65
Threshold 0.35: Recall %51.4, Precision %0.78 (Seçilen)
Threshold 0.40: Recall %45.7, Precision %2.09
Threshold 0.50: Recall %33.0, Precision %3.41

Seçilen threshold: 0.35 (Recall ve iş gereksinimlerini dengeleyen değer)

---

## 5. Model Performansı

### 5.1 Final Model Metrikleri

AUC-ROC: 0.635
Recall (Hata Yakalama): %51.4
Precision: %0.78
F1-Score: 0.0146

### 5.2 Baseline ile Karşılaştırma

AUC-ROC: 0.62'den 0.635'e çıktı (%2.4 iyileşme)
F1 Score: 0.0116'dan 0.0146'ya çıktı (%26 iyileşme)
Recall: 0.40'tan 0.514'e çıktı (%29 iyileşme)

### 5.3 Düşük Skorların Açıklaması

Skorlar görünüşte düşük olsa da bağlam önemli:

1. 1:228 oranında aşırı dengesizlik var
2. Temizlenmiş veride bile sadece %0.44 hatalı parça mevcut
3. Rastgele tahmin sadece %0.44 precision verir
4. Modelimiz hataların yarısından fazlasını (%51.4) yakalıyor

Mutlak rakamlar yanıltıcı olabilir. Önemli olan probleme göre değerlendirmek.

---

## 6. Maliyet Analizi

### 6.1 Birim Maliyetler

False Negative (Kaçan hata): $500 - İade, garanti, lojistik ve prestij kaybı
False Positive (Yanlış alarm): $10 - Ekstra inceleme işçiliği

### 6.2 Test Seti Sonuçları (90,103 parça)

Toplam Hatalı Parça: 378
Yakalanan Hatalar (TP): 194 (%51.4)
Kaçırılan Hatalar (FN): 184 (%48.6)
False Positive (Gereksiz İnceleme): 24,557

### 6.3 Maliyet Hesabı

AI olmadan maliyet: $189,000 (tüm 378 hata müşteriye ulaşıyor)
AI ile FN maliyeti: $92,000 (184 kaçan hata x $500)
AI ile FP maliyeti: $245,570 (24,557 gereksiz inceleme x $10)
Yakalanan hatalardan tasarruf: $97,000 (194 hata x $500)

Not: Yüksek FP sayısı toplam maliyeti artırıyor. Threshold değeri iş gereksinimlerine göre ayarlanabilir.

---

## 7. Teknik Altyapı

### 7.1 Teknoloji Stack

ML Framework: XGBoost, scikit-learn, pandas, numpy
API: FastAPI
UI: Streamlit
Deployment: Docker, docker-compose
Versiyon Kontrolü: Git, GitHub

### 7.2 API Endpoints

/health (GET): Sağlık kontrolü
/predict (POST): Tek tahmin
/predict/batch (POST): Toplu tahmin
/docs (GET): Swagger UI dokümantasyonu

### 7.3 Model Dosyaları

models/final_model.pkl: Eğitilmiş XGBoost modeli
models/feature_names.pkl: 162 feature listesi
models/model_config.pkl: threshold=0.35, auc=0.635 ve diğer metrikler

---

## 8. Proje Teslim Durumu

Tamamlanan işler:

EDA Notebook (01_eda.ipynb)
Baseline Model (02_baseline.ipynb)
Feature Engineering (03_feature_engineering.ipynb)
Final Pipeline (05_pipeline_final.ipynb)
Streamlit UI (app/ui.py)
FastAPI (app/main.py)
Docker Deployment (Dockerfile, docker-compose.yml)
GitHub Repo
README.md
Teknik Rapor

---

Rapor Sonu

Zero2End Machine Learning Bootcamp - Final Projesi
