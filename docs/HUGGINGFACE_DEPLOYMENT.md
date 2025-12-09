# 🚀 Hugging Face Spaces Deployment Rehberi

## Adım 1: Hugging Face Hesabı ve Token
1. https://huggingface.co adresine git ve hesap oluştur/giriş yap
2. https://huggingface.co/settings/tokens adresinden yeni token oluştur
   - Token type: **Write** (repo oluşturmak için)
   - Token'ı kopyala

## Adım 2: Space Oluştur
1. https://huggingface.co/new-space adresine git
2. Ayarlar:
   - **Space name:** `bosch-quality-prediction`
   - **License:** MIT
   - **SDK:** Streamlit
   - **Space hardware:** CPU basic (free)
3. "Create Space" butonuna tıkla

## Adım 3: Dosyaları Yükle (Web Arayüzü ile)

### 3.1 Önce küçük dosyaları yükle:
Space sayfasında "Files" sekmesine git ve şu dosyaları yükle:
- `app/ui.py`
- `src/config.py`
- `src/inference.py`
- `src/__init__.py`
- `requirements.txt`
- `models/final_model.pkl`
- `models/feature_names.pkl`
- `models/model_config.pkl`

### 3.2 Büyük veri dosyaları için:
- `data/train_numeric_clean.csv` (400MB)
- `data/test_numeric_clean_alt.csv` (400MB)

Bunları yüklemek için "Add file" > "Upload files" kullan
Hugging Face otomatik olarak Git LFS kullanır.

## Adım 4: README.md Güncelle
Space'te README.md dosyasını şu içerikle güncelle:

```yaml
---
title: Bosch Quality Prediction
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app/ui.py
pinned: false
license: mit
---
```

## Adım 5: Klasör Yapısı
Space'te şu yapı olmalı:
```
bosch-quality-prediction/
├── README.md (yaml header ile)
├── requirements.txt
├── app/
│   └── ui.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   └── inference.py
├── models/
│   ├── final_model.pkl
│   ├── feature_names.pkl
│   └── model_config.pkl
└── data/
    ├── train_numeric_clean.csv
    └── test_numeric_clean_alt.csv
```

## Alternatif: Git CLI ile Yükleme (WSL)

```bash
# WSL'de çalıştır
cd /home/goksu/code/bosch-quality-ml-pipeline

# Hugging Face remote ekle
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/bosch-quality-prediction

# LFS için dosyaları track et
git lfs track "*.csv"
git lfs track "*.pkl"

# Commit ve push
git add .
git commit -m "Initial HF Space deployment"
git push hf main
```

## Sorun Giderme

### "Out of memory" hatası
- Free tier 16GB RAM sağlar, bu yeterli olmalı
- Veriyi chunk'lar halinde okumayı dene

### "App not starting" hatası
- requirements.txt'i kontrol et
- app_file yolunun doğru olduğundan emin ol

### Büyük dosya yükleme sorunu
- Hugging Face Git LFS'i otomatik kullanır
- Web arayüzünden yüklerken 5GB'a kadar dosya desteklenir
