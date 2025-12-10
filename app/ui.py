"""
Bosch Kalite Tahmin Arayuzu (Streamlit).
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import joblib
from pathlib import Path
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent.parent / "src"))

# Paths
MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"


@st.cache_resource
def load_model():
    """Model yukle."""
    model = joblib.load(MODELS_DIR / "final_model.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    config = joblib.load(MODELS_DIR / "model_config.pkl")
    return model, feature_names, config


@st.cache_data
def load_test_data():
    """Test verisi yukle."""
    for fname in ["test_numeric_clean_alt.csv", "train_numeric_clean.csv"]:
        fpath = DATA_DIR / fname
        if fpath.exists():
            return pd.read_csv(fpath, nrows=5000)
    return None


def main():
    st.set_page_config(page_title="Bosch Kalite Tahmin", page_icon="🔧", layout="wide")
    
    st.title("🔧 Bosch Uretim Hatti Kalite Tahmini")
    st.markdown("*XGBoost tabanli urun kalite tahmin sistemi*")
    
    # Veri indirme uyarisi
    st.warning("""
    ⚠️ **Onemli:** Dosya boyutu kisitlamasi nedeniyle test verileri bu uygulamaya dahil edilememistir.  
    
    **CSV dosyalarini indirmek icin:** [Google Drive](https://drive.google.com/drive/u/1/folders/1-Qobnb-MZkYQ3-Gi2JaQKZ4H185CuYex)  
    
    Indirdikten sonra **"📁 CSV Yukle"** sekmesinden dosyanizi yukleyebilirsiniz.
    """)
    
    try:
        model, feature_columns, config = load_model()
        default_threshold = config.get('threshold', 0.35)
    except Exception as e:
        st.error(f"Model yuklenemedi: {e}")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        threshold = st.slider("Karar Esigi (Threshold)", 0.1, 0.9, default_threshold, 0.05,
                              help="Dusuk = daha fazla hatali tespit, Yuksek = daha az false positive")
        
        st.divider()
        st.header("📊 Model Bilgisi")
        st.metric("Ozellik Sayisi", len(feature_columns))
        st.metric("Varsayilan Esik", f"{default_threshold:.0%}")
        st.metric("AUC-ROC", f"{config.get('auc_roc', 0.635):.3f}")
        
        st.divider()
        st.markdown("""
        **Nasil Calisir?**
        1. Sensor verileri girilir
        2. Model hata olasiligi hesaplar  
        3. Esik degerine gore karar verilir
        """)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎲 Rastgele Ornek", "✏️ Manuel Test", "📤 CSV Yukle"])
    
    with tab1:
        show_random_prediction(model, feature_columns, threshold)
    
    with tab2:
        show_manual_prediction(model, feature_columns, threshold)
    
    with tab3:
        show_file_upload(model, feature_columns, threshold)


def add_features(df, cols):
    """Ozellik muhendisligi."""
    X = df.copy()
    X['row_mean'] = df[cols].mean(axis=1)
    X['row_std'] = df[cols].std(axis=1)
    X['row_min'] = df[cols].min(axis=1)
    X['row_max'] = df[cols].max(axis=1)
    X['row_range'] = X['row_max'] - X['row_min']
    X['row_nonzero'] = (df[cols] != 0).sum(axis=1)
    X['missing_ratio'] = df[cols].isna().mean(axis=1)
    return X


def show_gauge(proba, threshold):
    """Risk gostergesi."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={'suffix': '%'},
        title={'text': "Hata Riski"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred" if proba >= threshold else "green"},
            'steps': [
                {'range': [0, 35], 'color': "#90EE90"},
                {'range': [35, 70], 'color': "#FFD700"},
                {'range': [70, 100], 'color': "#FF6B6B"}
            ],
            'threshold': {'line': {'color': "black", 'width': 3}, 'value': threshold * 100}
        }
    ))
    fig.update_layout(height=280, margin=dict(t=80, b=20))
    return fig


def show_random_prediction(model, feature_columns, threshold):
    """Rastgele ornek testi."""
    st.header("🎲 Rastgele Ornek")
    
    test_data = load_test_data()
    
    if test_data is None:
        st.warning("⚠️ Test verisi bulunamadi. Sentetik veri uretiliyor...")
        use_synthetic = True
    else:
        use_synthetic = False
        st.success(f"✅ {len(test_data)} satir test verisi yuklendi")
    
    if st.button("🎲 Yeni Ornek Sec", type="primary"):
        if use_synthetic:
            # Sentetik veri uret
            sample_data = {col: np.random.uniform(-0.5, 0.5) for col in feature_columns if col.startswith('L')}
            sample_data['row_mean'] = np.random.uniform(0.1, 0.4)
            sample_data['row_std'] = np.random.uniform(0.05, 0.2)
            sample_data['row_min'] = 0.0
            sample_data['row_max'] = np.random.uniform(0.3, 0.8)
            sample_data['row_range'] = sample_data['row_max'] - sample_data['row_min']
            sample_data['row_nonzero'] = np.random.randint(80, 140)
            sample_data['missing_ratio'] = np.random.uniform(0.7, 0.9)
            
            for col in feature_columns:
                if col not in sample_data:
                    sample_data[col] = 0.0
            
            X_pred = pd.DataFrame([sample_data])[feature_columns]
            actual = None
        else:
            idx = np.random.randint(0, len(test_data))
            sample = test_data.iloc[[idx]]
            
            orig_cols = [c for c in sample.columns if c not in ['Id', 'Response']]
            X = add_features(sample, orig_cols)
            
            for col in feature_columns:
                if col not in X.columns:
                    X[col] = 0
            
            X_pred = X[feature_columns]
            actual = int(sample['Response'].values[0]) if 'Response' in sample.columns else None
        
        proba = model.predict_proba(X_pred)[0, 1]
        pred = 1 if proba >= threshold else 0
        
        st.divider()
        
        # Sonuclar
        c1, c2, c3 = st.columns(3)
        c1.metric("Hata Olasiligi", f"{proba:.1%}")
        
        if pred == 1:
            c2.error("❌ Tahmin: HATALI")
        else:
            c2.success("✅ Tahmin: SAGLAM")
        
        if actual is not None:
            if actual == 1:
                c3.warning("⚠️ Gercek: HATALI")
            else:
                c3.info("ℹ️ Gercek: SAGLAM")
        
        # Gauge
        st.plotly_chart(show_gauge(proba, threshold), use_container_width=True)


def show_manual_prediction(model, feature_columns, threshold):
    """Manuel test - detayli."""
    st.header("✏️ Manuel Test")
    
    # Ozellik aciklamalari
    with st.expander("📖 Ozellik Aciklamalari", expanded=True):
        st.markdown("""
        | Ozellik | Aciklama | Normal Aralik | Riskli Aralik |
        |---------|----------|---------------|---------------|
        | **Sensor Ortalamasi** | Tum sensorlerin ortalama degeri | 0.10 - 0.25 | > 0.40 |
        | **Degiskenlik (Std)** | Sensorler arasi tutarsizlik | 0.05 - 0.15 | > 0.30 |
        | **Min Deger** | En dusuk sensor okumasi | 0.00 | - |
        | **Max Deger** | En yuksek sensor okumasi | 0.30 - 0.50 | > 0.80 |
        | **Aktif Sensor** | Sifir olmayan sensor sayisi | > 100 | < 70 |
        | **Eksik Veri Orani** | Bos sensor orani | 0.70 - 0.80 | > 0.90 |
        
        💡 **Ipucu:** Riskli aralikta degerler, uretimdeki potansiyel sorunlara isaret eder.
        """)
    
    st.divider()
    
    # Hazir senaryolar
    scenario = st.selectbox("🎯 Senaryo Sec", [
        "Ozel Degerler Gir",
        "✅ Normal Parca (Dusuk Risk)",
        "⚠️ Supheli Parca (Orta Risk)", 
        "❌ Riskli Parca (Yuksek Risk)"
    ])
    
    scenarios = {
        "✅ Normal Parca (Dusuk Risk)": (0.15, 0.08, 0.0, 0.40, 125, 0.75),
        "⚠️ Supheli Parca (Orta Risk)": (0.32, 0.22, 0.0, 0.85, 90, 0.82),
        "❌ Riskli Parca (Yuksek Risk)": (0.55, 0.40, 0.0, 1.0, 55, 0.93)
    }
    
    st.divider()
    
    if scenario == "Ozel Degerler Gir":
        c1, c2 = st.columns(2)
        with c1:
            row_mean = st.slider("📊 Sensor Ortalamasi", 0.0, 1.0, 0.20, 0.01,
                                help="Yuksek deger = anomali riski artar")
            row_std = st.slider("📈 Degiskenlik (Std)", 0.0, 0.5, 0.10, 0.01,
                               help="Yuksek = tutarsiz uretim")
            row_min = st.slider("⬇️ Min Deger", 0.0, 0.5, 0.0, 0.01)
        with c2:
            row_max = st.slider("⬆️ Max Deger", 0.0, 1.0, 0.50, 0.01)
            row_nonzero = st.slider("🔢 Aktif Sensor Sayisi", 20, 150, 100,
                                   help="Dusuk = veri eksikligi")
            missing_ratio = st.slider("❓ Eksik Veri Orani", 0.5, 1.0, 0.80, 0.01,
                                     help="Yuksek = daha az veri")
    else:
        row_mean, row_std, row_min, row_max, row_nonzero, missing_ratio = scenarios[scenario]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Sensor Ort.", f"{row_mean:.2f}")
        c2.metric("Degiskenlik", f"{row_std:.2f}")
        c3.metric("Aktif Sensor", row_nonzero)
    
    st.divider()
    
    if st.button("🔍 Analiz Et", type="primary", use_container_width=True):
        # Feature vector olustur
        features = {col: 0.0 for col in feature_columns}
        features.update({
            'row_mean': row_mean,
            'row_std': row_std,
            'row_min': row_min,
            'row_max': row_max,
            'row_range': row_max - row_min,
            'row_nonzero': float(row_nonzero),
            'missing_ratio': missing_ratio
        })
        
        X = pd.DataFrame([features])[feature_columns]
        proba = model.predict_proba(X)[0, 1]
        pred = 1 if proba >= threshold else 0
        
        st.divider()
        
        # Sonuc kartlari
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hata Olasiligi", f"{proba:.1%}")
        c2.metric("Risk Skoru", f"{min(proba * 150, 100):.0f}/100")
        c3.metric("Karar Esigi", f"{threshold:.0%}")
        c4.metric("Guven", f"{abs(proba - threshold) * 100:.0f}%")
        
        # Karar
        if pred == 1:
            st.error("❌ **SONUC: HATALI** - Bu parca inceleme istasyonuna yonlendirilmeli!")
        else:
            st.success("✅ **SONUC: SAGLAM** - Bu parca kalite standartlarini karsilamaktadir.")
        
        # Gauge grafik
        st.plotly_chart(show_gauge(proba, threshold), use_container_width=True)
        
        # Faktor analizi
        st.subheader("📋 Detayli Faktor Analizi")
        
        factors = [
            ("Sensor Ortalamasi", row_mean, "< 0.25", row_mean <= 0.25, 
             "Normal" if row_mean <= 0.25 else "Yuksek - dikkat!"),
            ("Degiskenlik", row_std, "< 0.15", row_std <= 0.15,
             "Tutarli" if row_std <= 0.15 else "Tutarsiz uretim!"),
            ("Deger Araligi", row_max - row_min, "< 0.50", (row_max - row_min) <= 0.50,
             "Normal" if (row_max - row_min) <= 0.50 else "Genis aralik!"),
            ("Aktif Sensor", row_nonzero, "> 100", row_nonzero >= 100,
             "Yeterli veri" if row_nonzero >= 100 else "Veri eksik!"),
            ("Veri Kalitesi", 1 - missing_ratio, "> 0.20", (1 - missing_ratio) >= 0.20,
             "Kabul edilebilir" if (1 - missing_ratio) >= 0.20 else "Cok fazla eksik!")
        ]
        
        for name, val, ideal, ok, comment in factors:
            icon = "✅" if ok else "⚠️"
            st.markdown(f"{icon} **{name}:** `{val:.2f}` (ideal: {ideal}) - _{comment}_")
        
        # Maliyet analizi
        st.subheader("💰 Maliyet Etkisi")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            **Hatali parca kacirsa:**
            - Garanti maliyeti: ~$500
            - Marka itibar kaybi
            """)
        with c2:
            st.markdown("""
            **Gereksiz inceleme yapilsa:**
            - Ekstra iscilik: ~$10
            - Uretim gecikmesi: Minimal
            """)


def show_file_upload(model, feature_columns, threshold):
    """CSV yukleme."""
    st.header("📤 CSV Yukle")
    
    uploaded = st.file_uploader("CSV dosyasi secin", type=['csv'])
    
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"✅ Yuklendi: {len(df)} satir, {len(df.columns)} sutun")
            
            if st.button("Toplu Tahmin Yap", type="primary"):
                orig_cols = [c for c in df.columns if c not in ['Id', 'Response']]
                X = add_features(df, orig_cols)
                
                for col in feature_columns:
                    if col not in X.columns:
                        X[col] = 0
                
                probas = model.predict_proba(X[feature_columns])[:, 1]
                preds = (probas >= threshold).astype(int)
                
                df['Olasilik'] = probas
                df['Tahmin'] = preds
                df['Sonuc'] = df['Tahmin'].map({0: 'SAGLAM', 1: 'HATALI'})
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Toplam", len(df))
                c2.metric("Hatali", int(preds.sum()))
                c3.metric("Saglam", int(len(df) - preds.sum()))
                
                st.dataframe(df[['Olasilik', 'Sonuc']].head(20))
                
                csv = df.to_csv(index=False)
                st.download_button("📥 Sonuclari Indir", csv, "tahminler.csv", "text/csv")
                
        except Exception as e:
            st.error(f"Hata: {e}")


if __name__ == "__main__":
    main()
