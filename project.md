Harika bir karar. Bu dosya, senin veya bir AI Agent'ın (Claude, ChatGPT vb.) projeyi başından sonuna kadar **otonom bir şekilde** yürütebilmesi için gereken **"Master Context"** dosyasıdır.

Bu metni `.md` (Markdown) veya `.txt` dosyası olarak kaydedip, herhangi bir AI modeline "Proje dosyam bu, şimdi Adım X'i yapmanı istiyorum" diyerek ilerleyebilirsin.

İşte projenin tüm detaylarını, kısıtlarını ve yol haritasını içeren **Master Blueprint**:

-----

# **MASTER PROJECT BLUEPRINT: Bosch Production Line Performance (Zero2End Bootcamp)**

## **1. PROJE KİMLİĞİ VE AMACI**

  * **Proje Adı:** `bosch-quality-ml-pipeline`
  * **Sektör:** Üretim (Manufacturing)
  * **Problem:** Bosch üretim hattındaki parçaların kalite kontrol testlerinden geçip geçmediğini tahmin etmek (Failure Prediction).
  * **Problem Tipi:** Binary Classification (İkili Sınıflandırma).
      * `0`: Sağlam Parça
      * `1`: Hatalı Parça
  * **Veri Seti:** Kaggle - Bosch Production Line Performance
      * **Odak Dosya:** `train_numeric.csv` (Zaman kısıtı nedeniyle sadece nümerik veriler kullanılacak).
  * **Teslim Tarihi:** 9 Aralık 2025 (Acil - MVP yaklaşımı gerekli).
  * **Kısıtlar:**
      * Veri çok büyük -\> **Sampling (Örnekleme)** kullanılacak (İlk 50k-100k satır).
      * Sınıflar çok dengesiz (Imbalanced) -\> Metrik olarak **F1-Score** veya **ROC-AUC** kullanılacak (Accuracy değil).
      * Veri çok seyrek (Sparse/Missing Values) -\> Eksik veri stratejisi kritik.

-----

## **2. PROJE KLASÖR YAPISI (REPO STRUCTURE)**

Proje bu iskelet üzerine kurulacaktır. GitHub Copilot veya manuel kurulumda bu yapı esastır.

```text
bosch-quality-ml-pipeline/
├── .gitignore          # Git tarafından yoksayılacaklar (data/, .env, venv/)
├── README.md           # Proje, kurulum ve analiz özeti
├── requirements.txt    # pandas, numpy, sklearn, xgboost, fastapi, uvicorn, streamlit, matplotlib
├── data/               # HAM VERİ BURADA OLACAK (train_numeric.csv)
├── notebooks/          # Analiz ve Modelleme Aşamaları
│   ├── 01_eda.ipynb            # Veriyi anlama
│   ├── 02_baseline.ipynb       # En basit model
│   ├── 03_feature_eng.ipynb    # Özellik mühendisliği
│   ├── 04_optimization.ipynb   # Hiperparametre ayarları
│   └── 05_pipeline_final.ipynb # Final pipeline (Preprocessing + Model)
├── src/                # Modüler Kodlar
│   ├── __init__.py
│   ├── config.py       # Sabitler (Pathler, Model Parametreleri)
│   ├── train.py        # Eğitim scripti
│   └── inference.py    # Tahmin alma scripti
├── models/             # Kaydedilen Modeller (.pkl dosyaları)
└── app/                # Deployment (Canlıya Alma)
    ├── main.py         # FastAPI Backend (Opsiyonel)
    └── ui.py           # Streamlit Frontend (Zorunlu - Arayüz)
```

-----

## **3. ADIM ADIM UYGULAMA PLANI (AGENT TALİMATLARI)**

Bu bölüm, AI Agent'lara adım adım verilecek talimatları içerir.

### **ADIM 1: Kurulum ve Veri Hazırlığı**

  * **Görev:** GitHub Reposu oluşturulması ve verinin yerleştirilmesi.
  * **Prompt (Copilot için):**
    > Scaffold a Python ML project for Bosch Quality Classification. Structure: data/, notebooks/, src/, app/, models/, docs/. Create requirements.txt with pandas, numpy, sklearn, xgboost, fastapi, uvicorn, streamlit, matplotlib. Create README.md describing failure prediction. Add basic FastAPI code in app/main.py.
  * **Manuel İşlem:** `train_numeric.csv` dosyasını `data/` klasörüne zipten çıkararak ekle.

### **ADIM 2: Keşifçi Veri Analizi (EDA) - `01_eda.ipynb`**

  * **Amaç:** Verinin yapısını, eksikliklerini ve dengesizliğini kanıtlamak.
  * **Kritik Kod Parçası (Sampling ile Yükleme):**
    ```python
    df = pd.read_csv('../data/train_numeric.csv', nrows=100000) # RAM dostu yaklaşım
    ```
  * **Dokümante Edilecekler:**
    1.  `Response` değişkeninin dağılımı (%99.5 Sağlam vs %0.5 Hatalı).
    2.  Eksik veri (Missing Value) oranı (Bosch verisinde çok yüksektir).
    3.  Feature sayısı ve tipleri.

### **ADIM 3: Baseline Model (MVP) - `02_baseline.ipynb`**

  * **Amaç:** Hiçbir karmaşık işlem yapmadan "referans" bir skor elde etmek.
  * **Strateji:**
    1.  Eksik verileri basitçe doldur (SimpleImputer -\> mean veya -1 ile).
    2.  Basit bir model seç (Logistic Regression veya Random Forest - `class_weight='balanced'`).
    3.  Skoru kaydet (F1-Score, AUC).
  * **Beklenti:** Skor düşük olabilir ama sıfırdan büyük olmalıdır.

### **ADIM 4: Feature Engineering & Optimizasyon - `03_feature_eng.ipynb`**

  * **Amaç:** Skoru baseline'ın üzerine çıkarmak.
  * **Önerilen Teknikler:**
      * Eksik verilerin kendisi bir bilgidir: `Is_Missing` flagleri eklenebilir.
      * Korelasyonu yüksek feature'ların seçimi.
      * **Model Seçimi:** XGBoost veya LightGBM (Dengesiz ve eksik veride daha başarılıdır).
      * **Validasyon:** Stratified K-Fold Cross Validation (Dengesizlik yüzünden Stratified şart).

### **ADIM 5: Final Pipeline ve Değerlendirme - `05_pipeline_final.ipynb`**

  * **Amaç:** En iyi çalışan adımları tek bir boru hattında (Pipeline) birleştirmek ve modeli kaydetmek.
  * **Çıktı:** `final_model.pkl` dosyası (`models/` klasörüne kaydedilecek).
  * **Cevaplanacak İş Soruları:**
      * Final model Baseline'ı geçti mi?
      * Model canlıya alındığında nasıl izlenir?

### **ADIM 6: Deployment (Streamlit Arayüzü) - `app/ui.py`**

  * **Amaç:** Teknik olmayan birinin modeli kullanabilmesi.
  * **Teknoloji:** Streamlit.
  * **Arayüz Tasarımı:**
      * Kullanıcıya tek tek 1000 input girdirmek imkansızdır.
      * **Çözüm:** "Load Random Sample" veya "Upload CSV" butonu.
      * Kullanıcı butona basar -\> Sistem test verisinden rastgele bir satır çeker -\> Model tahmin eder -\> Ekrana **"Ürün Sağlam ✅"** veya **"Ürün Hatalı ❌"** yazar.

### **ADIM 7: Sunum ve Dokümantasyon (Opsiyonel ama Önerilen)**

  * **Readme.md:** Projenin ne olduğu, nasıl çalıştırılacağı, Baseline vs Final skorları.
  * **Sunum (3-4 Slayt):**
    1.  **Problem:** Üretim hattındaki hataları gözden kaçırmanın maliyeti.
    2.  **Yöntem:** Yapay Zeka ile geçmiş verilerden öğrenme.
    3.  **Sonuç:** Hatalı parçaları yakalama başarımız.

-----

## **4. ZAMAN YÖNETİMİ VE KISITLAR (ACİL)**

  * **Veri Yönetimi:** ASLA tüm veriyi RAM'e almaya çalışma. Her zaman `nrows=100000` veya `chunksize` kullan.
  * **Model Karmaşıklığı:** Derin Öğrenme (Deep Learning) veya aşırı karmaşık Ensemble modellerine girme. XGBoost yeterli ve hızlıdır.
  * **Deployment:** Streamlit en hızlı yoldur. Flask/HTML/CSS ile vakit kaybetme.
  * **Öncelik:** Önce kodun çalışmasını sağla (Run-to-end), optimizasyonu sonraya bırak.

-----

## **5. BAŞARI KRİTERLERİ (CHECKLIST)**

  * [ ] Repo oluşturuldu ve veri yüklendi.
  * [ ] EDA Notebook'u: Class Imbalance gösterildi.
  * [ ] Baseline Model: Bir skor elde edildi.
  * [ ] Final Model: `.pkl` olarak kaydedildi.
  * [ ] Streamlit App: Lokal'de çalışıyor, tahmin üretiyor.
  * [ ] Readme.md: Projeyi ve sonuçları anlatıyor.
  * [ ] **(Opsiyonel)** Deployment: Render veya HuggingFace üzerinde canlı link.

-----

## **6. : Deployment ve Arayüz - `app/ui.py` & `app/main.py`**

* **Amaç:** Modeli hem son kullanıcı (Operatör) hem de sistemler (Robotlar) için erişilebilir kılmak.
* **Backend (FastAPI):**
    * `app/main.py` içinde `/predict` endpoint'i oluştur.
    * Girdi olarak JSON formatında sensör verilerini al, çıktı olarak `{"prediction": 1, "probability": 0.85}` dön.
* **Frontend (Streamlit):**
    * Kullanıcı dostu bir panel tasarla.
    * **Özellik:** "Simülasyon Modu". Kullanıcı bir butona bastığında test setinden rastgele bir parça seçilsin, sensör değerleri ekranda gösterilsin ve modelin kararı (Geçti/Kaldı) anlık olarak yansıtılsın.
    * **Görsellik:** Hatalı tahminlerde kırmızı, sağlam tahminlerde yeşil uyarı kutuları (st.error, st.success) kullan.

-----

## **7. DETAYLI RAPORLAMA VE SONUÇ YORUMLAMA (BUSINESS INTELLIGENCE)**

Modelin "F1 Skoru 0.85" demek yönetim için bir anlam ifade etmez. Sonuçları **üretim diliyle** konuşmalıyız.

### **A. Metriklerin İş Diline Çevrilmesi**
Bu proje için `models/` klasöründe veya notebook çıktılarında şu sorulara yanıt veren grafikler olmalıdır:

1.  **Confusion Matrix Analizi (Maliyet Hesabı):**
    * **True Negative (TN):** Parça sağlam, model "sağlam" dedi. -> *Sorun yok, üretim devam eder.*
    * **False Positive (FP - Tip 1 Hata):** Parça sağlam, model "bozuk" dedi. -> *Maliyet: Gereksiz inceleme süresi (İşçilik kaybı).*
    * **False Negative (FN - Tip 2 Hata):** Parça bozuk, model "sağlam" dedi. -> *KRİTİK HATA! Müşteriye bozuk parça gider. İade maliyeti ve marka prestij kaybı.*
    * **Strateji:** Modelimiz **FN (Kaçırılan Hata)** sayısını minimize etmeye odaklanmalıdır (Recall değeri yüksek olmalı).

2.  **Feature Importance (Kök Neden Analizi):**
    * Hangi sensör veya istasyon hatalara en çok sebep oluyor?
    * `XGBoost Feature Importance` veya `SHAP` (SHapley Additive exPlanations) kütüphanesi kullanılacak.
    * **Çıktı:** "L3_S30_F3755 sensöründeki yüksek değerler, parçanın bozuk olma ihtimalini %40 artırıyor." gibi somut bir içgörü sunulmalı.

3.  **Threshold (Eşik Değer) Ayarı:**
    * Varsayılan 0.5 yerine, Recall'u artırmak için eşik değeri düşürülebilir (örn: 0.3).
    * **Trade-off Grafiği:** Precision-Recall eğrisi çizilerek, "Hataların %90'ını yakalamak için ne kadar yanlış alarma (False Positive) katlanmalıyız?" sorusu yanıtlanmalı.

---

## **8. ÜST YÖNETİM SUNUMU TASLAĞI (EXECUTIVE PITCH)**

Teknik olmayan yöneticilere (CTO, Üretim Müdürü) yapılacak sunumun taslağıdır. Kod gösterilmez, **ROI (Yatırım Getirisi)** konuşulur.

### **Slayt 1: Yönetici Özeti (Executive Summary)**
* **Problem:** Bosch üretim hattında kalite kontrol süreçleri manuel/yarı otomatize ve gözden kaçan hatalar maliyet yaratıyor.
* **Çözüm:** Geçmiş üretim verilerini kullanan Yapay Zeka tabanlı erken uyarı sistemi.
* **Ana Sonuç:** Geliştirilen model, hatalı parçaların %XX'ini üretim hattından çıkmadan tespit edebiliyor.

### **Slayt 2: İş Problemi ve Finansal Etki**
* Mevcut durumda (varsayımsal) her 1000 parçada 5 hatalı parça müşteriye ulaşıyor.
* **Maliyet:** İade, garanti kapsamı ve lojistik maliyeti = Yıllık X.XXX $.
* **Hedef:** Bu maliyeti AI desteği ile minimize etmek.

### **Slayt 3: Çözüm Mimarisi (Basitleştirilmiş)**
* Veri -> AI Motoru (Karar Mekanizması) -> Operatör Ekranı (Yeşil/Kırmızı Işık).
* Karmaşık matematiksel detaylara girmeden, sistemin nasıl bir filtre görevi gördüğünü görselleştiren bir şema (Diagram).

### **Slayt 4: Performans ve Kazanımlar**
* **Metrikler:** "Modelimiz %95 doğrulukla çalışıyor" (Yanlış - dengesiz veride kullanılmaz).
* **Doğru İfade:** "Modelimiz, potansiyel hataların **%80'ini** henüz fabrikadan çıkmadan yakalıyor."
* **Öngörülen Tasarruf:** Test süreçlerinin kısalması ve hata oranının düşmesiyle operasyonel verimlilik artışı.

### **Slayt 5: Yol Haritası (Next Steps)**
* Pilot testlerin gerçek üretim hattında başlatılması.
* Modelin diğer üretim hatlarına (farklı parça tiplerine) genişletilmesi.
* Gerçek zamanlı sensör verileriyle entegrasyon (IoT).

---

## **9. GELİŞMİŞ DEPLOYMENT SENARYOLARI (OPSİYONEL)**

Sadece Streamlit ile lokalde çalışmak yeterli olsa da, projeyi "Production Ready" hale getirmek için eklenebilecekler:

1.  **Dockerizasyon:**
    * Projenin her bilgisayarda aynı şekilde çalışması için `Dockerfile` oluşturulması.
    * *Komut:* `docker build -t bosch-quality-app .`
2.  **API Servisi (FastAPI):**
    * Streamlit sadece arayüzdür. Gerçek dünyada üretim bandındaki robotlar (PLC), modele Python arayüzünden değil, API (HTTP Request) üzerinden sorar.
    * `app/main.py` içerisine bir `/predict` endpoint'i yazılması ve Swagger UI üzerinden test edilmesi profesyonellik göstergesidir.
3.  **Cloud Hosting:**
    * **Streamlit Cloud:** GitHub reposunu bağlayarak saniyeler içinde canlıya alma (Ücretsiz & En kolayı).
    * **Render / Railway:** Docker container'ı veya FastAPI servisini deploy etmek için.

---

## **10. README.MD DOSYASINI DÜZENLE**

  * **Readme.md:** Elimizdeki tüm bilgileri analiz et ve bu dosyayı güncelle.
  * **Diğer:** Proje dosyalarının (bosch-quality-ml-pipeline altındaki) hepsini kontrol et.

---

**NOT:** Bu proje "Zero2End Machine Learning Bootcamp" final projesi isterlerine (Tabular data, Kaggle source, Pipeline structure, Deployment) %100 uyumludur.