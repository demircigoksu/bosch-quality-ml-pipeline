# Üretim Zekası: XGBoost, FastAPI ve Docker ile Üretim Hattı Hatalarını Önleme

*Hatalı parçaları müşteriye ulaşmadan yakalayan ve binlerce dolarlık garanti maliyetini önleyen uçtan uca bir ML sistemi nasıl inşa ettik?*

---

## Kimsenin Konuşmadığı 500$'lık Problem

Şunu hayal edin: Tek bir hatalı parça üretim hattından çıkıyor. Kalite kontrolden geçiyor, müşteriye gönderiliyor, sahada arızalanıyor ve garanti talebini tetikliyor.

**Toplam maliyet? 500$ veya daha fazla.**

Şimdi aynı hatayı fabrika içinde yakaladığınızı düşünün.

**Maliyet? Ekstra inceleme için sadece 10$.**

Bu 50:1 maliyet oranı, üretim kalitesinin gizli ekonomisidir. Mesele mükemmel modeller inşa etmek değil — her tahmininin iş etkisini anlayan *maliyet-farkındalıklı* sistemler inşa etmektir.

Bu makalede, ünlü Bosch Production Line Performance veri seti için nasıl uçtan uca bir Makine Öğrenmesi sistemi kurduğumu anlatacağım. Kabus seviyesindeki veri kalitesi sorunlarından, FastAPI, Streamlit ve Docker ile üretime hazır bir sistem dağıtımına kadar her şeyi ele alacağız.

> **Özet**: F1-skorunda %26 iyileştirme sağladık, gerçek zamanlı tahmin API'si deploy ettik ve maliyet-farkındalıklı eşik optimizasyonu sistemi oluşturduk. Tüm kod açık kaynak.

---

## Cehennemden Gelen Veri Seti (İyi Anlamda)

Bosch Production Line Performance veri seti, ML topluluğunda efsanevidir — temiz ve kolay olduğu için değil, **acımasızca gerçekçi** olduğu için.

İşte karşılaştıklarımız:

| Zorluk | Gerçeklik | Neden Önemli |
|--------|-----------|--------------|
| **Aşırı Sınıf Dengesizliği** | 1:175 oranı (%0.57 hatalı) | Accuracy anlamsız |
| **Devasa Ölçek** | 1.2M satır × 970 sütun | Bellek yönetimi kritik |
| **Eksik Veri Kıyameti** | %81 ortalama eksiklik | Çoğu özellik boş |
| **Sensör Bombardımanı** | 968 farklı ölçüm | Özellik seçimi şart |

```python
# Ayıltan gerçeklik kontrolü
df = pd.read_csv('train_numeric.csv', nrows=100_000)
print(f"Hata Oranı: {df['Response'].mean():.2%}")  
# Çıktı: 0.57%

print(f"Eksiklik Oranı: {df.isnull().mean().mean():.1%}")  
# Çıktı: 81.0%
```

**Bu neden önemli?** Çünkü gerçek üretim verisi tam olarak böyle görünür. Eğer sadece temiz Kaggle veri setleriyle çalıştıysanız, bu sizin için bir uyandırma çağrısı.

---

## Neden XGBoost? Neden Pipeline?

Projenin başlarında iki kritik karar verdim:

### Karar 1: Derin Öğrenme Yerine XGBoost

%81 eksik veri ile sinir ağları zorlanacaktı. XGBoost'un bu problem için üç ölümcül özelliği var:

1. **Yerleşik Eksik Değer İşleme**: XGBoost, eksik değerler için optimal bölme yönlerini otomatik öğrenir
2. **`scale_pos_weight`**: Dahili sınıf dengesizliği yönetimi
3. **Yorumlanabilirlik**: Özellik önem sırası bize *hangi sensörlerin önemli olduğunu* söyler

```python
model = XGBClassifier(
    scale_pos_weight=175,  # 1:175 dengesizliğe eşle
    max_depth=6,
    learning_rate=0.1,
    n_estimators=300,
    early_stopping_rounds=50
)
```

### Karar 2: Notebook Spagettisi Yerine Pipeline Mimarisi

Jupyter notebook'taki bir model bilim projesidir. Pipeline'daki bir model üründür.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Ham Veri   │────▶│  Temizleme  │────▶│  Özellik    │────▶│   Model     │
│  (968 sütun)│     │  Pipeline   │     │  Mühendisliği│    │  + Eşik     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

Bu mimari bize şunları sağladı:
- **Tekrarlanabilirlik**: Her seferinde aynı ön işleme
- **Dağıtılabilirlik**: Tek pickle dosyası, komple pipeline
- **Bakım Kolaylığı**: Net sorumluluk ayrımı

---

## Teknik Derinlemesine İnceleme

### Strateji 1: Akıllı Örnekleme

1.2M satır × 970 sütun yüklemek 16GB RAM'li bilgisayarımı çökertti. Çözüm:

```python
# Stratejik yükle — sınıf dağılımını koru
df = pd.read_csv('train_numeric.csv', nrows=100_000)

# Sonuç: 99,432 sağlam + 568 hatalı
# Sınıf oranı rastgele örnekleme ile otomatik korundu
```

### Strateji 2: Gerçekten İşe Yarayan Özellik Mühendisliği

Ham sensör değerleri hikayenin bir kısmını anlatır. Mühendislik yapılmış özellikler geri kalanını:

```python
# Satır düzeyinde istatistikler (basit ama güçlü)
df['row_mean'] = df[sensor_cols].mean(axis=1)
df['row_std'] = df[sensor_cols].std(axis=1)
df['row_non_null'] = df[sensor_cols].notna().sum(axis=1)

# İstasyon düzeyinde toplamalar
for station in ['L0_S0', 'L3_S30', 'L3_S32']:
    station_cols = [c for c in cols if c.startswith(station)]
    df[f'{station}_mean'] = df[station_cols].mean(axis=1)
```

**Sürpriz**: `row_mean` (en basit özellik) ilk 5 tahminciden biri oldu. Bazen temeller kazanır.

### Strateji 3: Maliyet-Farkındalıklı Eşik Optimizasyonu

İşte iş dünyasının ML ile buluştuğu nokta. Varsayılan eşik 0.5, ama bu optimal mi?

```python
# İş maliyetlerini tanımla
COST_FALSE_POSITIVE = 10   # Gereksiz inceleme
COST_FALSE_NEGATIVE = 500  # Kaçırılan hata → müşteri iadesi

# Optimal eşiği ara
best_cost = float('inf')
best_threshold = 0.5

for threshold in np.arange(0.1, 0.9, 0.01):
    y_pred = (y_proba >= threshold).astype(int)
    
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    total_cost = fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE
    
    if total_cost < best_cost:
        best_cost = total_cost
        best_threshold = threshold

print(f"Optimal Eşik: {best_threshold}")  # 0.55, 0.50 değil!
```

**Önemli içgörü**: "En iyi" eşik, maliyet yapınıza bağlıdır, sadece istatistiksel metriklere değil.

---

## Notebook'tan Üretime

Kimsenin kullanamadığı bir model, var olmayan bir modeldir. İşte nasıl ürünleştirdik:

### Katman 1: FastAPI — Tahmin Motoru

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Bosch Kalite Tahmin API'si")

class PredictionResponse(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: dict):
    # Pipeline'dan geçir
    prob = model.predict_proba([features])[:, 1][0]
    pred = 1 if prob >= 0.55 else 0  # Maliyet-optimize edilmiş eşik
    
    return {"prediction": pred, "probability": float(prob)}
```

**Neden FastAPI?**
- `/docs` adresinde otomatik Swagger dokümantasyonu
- Pydantic validasyonu kötü girdiyi yakalar
- Yüksek verim için async desteği

### Katman 2: Streamlit — İnsan Arayüzü

Fabrika operatörleri JSON konuşmaz. Butonlar ve renkler ister:

```python
import streamlit as st

if st.button("🎲 Rastgele Parça Test Et"):
    sample = load_random_sample()
    result = model.predict(sample)
    
    if result == 0:
        st.success("✅ GEÇTİ — Parça kalite standartlarını karşılıyor")
    else:
        st.error("❌ KALDI — İnceleme istasyonuna yönlendir")
```

### Katman 3: Docker — Her Yere Gönder

```dockerfile
FROM python:3.9-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8080 8501

CMD ["sh", "-c", "uvicorn app.main:app --port 8080 & streamlit run app/ui.py --server.port 8501"]
```

**Tek container, iki servis, sıfır "benim bilgisayarımda çalışıyor" bahanesi.**

---

## Sonuçlar: Gerçekte Ne Başardık

Rakamlar konusunda dürüst olalım:

| Metrik | Baseline | Final Model | Değişim |
|--------|----------|-------------|---------|
| **AUC-ROC** | 0.6655 | 0.6684 | +%0.4 |
| **F1-Skoru** | 0.0711 | 0.0894 | **+%25.7** |
| **Precision** | 0.0411 | 0.1231 | +%199 |

### "Ama bu skorlar düşük görünüyor!"

Evet, mesele de bu. 1:175 dengesizlik ve %81 eksik veri ile:
- Rastgele tahmin %0.57 precision alır
- Modelimiz %12.31 precision alıyor
- **Bu rastgeleden 21 kat daha iyi**

Ders: **Bağlam, mutlak sayılardan daha önemlidir.**

---

## Öğrenilen Dersler

1. **Maliyet-Farkındalık > Accuracy Takıntısı**: İş etkisi için optimize edin, liderlik tablosu metrikleri için değil

2. **Pipeline > Model**: Mimariye erken yatırım yapın. Dağıtımda temettü öder

3. **Çirkin Veriyi Kucaklayın**: Gerçek üretim verisi dağınıktır. Ön işlemeniz bunu zarif bir şekilde idare etmeli

4. **Erken Gönder, Sık İterasyona Git**: Deploy edilmiş %70'lik bir model, gönderilmemiş %95'lik bir modeli her zaman yener

---

## Kendiniz Deneyin

Projenin tamamı açık kaynak:

🔗 **GitHub**: [demircigoksu/bosch-quality-ml-pipeline](https://github.com/demircigoksu/bosch-quality-ml-pipeline)

```bash
# Klonla ve lokalde çalıştır
git clone https://github.com/demircigoksu/bosch-quality-ml-pipeline.git
cd bosch-quality-ml-pipeline
docker-compose up -d

# Erişim noktaları:
# API Dokümantasyonu: http://localhost:8080/docs
# Streamlit Arayüzü: http://localhost:8501
```

---

## Sırada Ne Var?

Bu proje bir temeldir. Gerçek dünya uzantıları şunları içerebilir:

- IoT sensörlerinden **gerçek zamanlı akış**
- Farklı eşik stratejileri için **A/B testi**
- Drift tespiti için **model izleme**
- Geliştirilmiş recall için **çoklu model ensemble**

Üretim yapay zekası insanların yerini almakla ilgili değil — onlara süper güçler vermekle ilgili. Hataları müşteri şikayetlerine dönüşmeden yakalayan bir sistem, herkesin işini kolaylaştıran bir sistemdir.

---

*Bu proje Zero2End Machine Learning Bootcamp kapsamında geliştirilmiştir. Bu veri setini araştırma ve öğrenme için kamuya açık hale getirdiği için Bosch'a teşekkürler.*

**Benimle iletişime geçin:**
- GitHub: [demircigoksu](https://github.com/demircigoksu)
- LinkedIn: [Göksu Demirci](https://linkedin.com/in/demircigoksu)

---

**Etiketler:** `#MakineÖğrenmesi` `#XGBoost` `#Üretim` `#FastAPI` `#Docker` `#VeriBlimi` `#MLOps` `#Python`
