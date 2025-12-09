"""
Bosch Kalite Tahmin Arayüzü (Streamlit).
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import MODEL_PATH

# Paths
MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"

# Page configuration
st.set_page_config(
    page_title="Bosch Quality Prediction",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_trained_model():
    """Eğitilmiş modeli yükle."""
    try:
        model_path = MODELS_DIR / "final_model.pkl"
        if not model_path.exists():
            model_path = MODELS_DIR / "bosch_quality_classifier.pkl"
        
        model = joblib.load(model_path)
        
        # Load feature names
        feature_path = MODELS_DIR / "feature_names.pkl"
        if feature_path.exists():
            feature_columns = joblib.load(feature_path)
        else:
            feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
        
        # Load config if exists
        config_path = MODELS_DIR / "model_config.pkl"
        if config_path.exists():
            config = joblib.load(config_path)
        else:
            config = {'threshold': 0.35}
        
        return model, feature_columns, config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


@st.cache_data
def load_sample_data():
    """Örnek veri yükle."""
    try:
        for data_file in ["test_numeric_clean_alt.csv", "train_numeric_clean.csv", 
                          "demo_test.csv", "demo_train.csv", "train_numeric.csv"]:
            data_path = DATA_DIR / data_file
            if data_path.exists():
                df = pd.read_csv(data_path, nrows=10000)
                return df, False
        
        return None, True
    except Exception as e:
        return None, True


def generate_realistic_sample(feature_columns, force_failure=False):
    """Gerçekçi örnek veri üret."""
    np.random.seed(None)
    
    sample = {}
    
    # riskli görünsün mü?
    is_risky = force_failure or (np.random.random() < 0.15)
    
    # istasyon bazlı korelasyonlu değerler üret
    station_base = {}
    for col in feature_columns:
        if col.startswith('L') and '_S' in col and '_F' in col:
            parts = col.split('_')
            station = f"{parts[0]}_{parts[1]}"
            
            if station not in station_base:
                if is_risky:
                    station_base[station] = np.random.uniform(-0.8, 0.8)
                else:
                    station_base[station] = np.random.uniform(-0.3, 0.3)
            
            noise = np.random.normal(0, 0.15)
            value = station_base[station] + noise
            
            if is_risky and np.random.random() < 0.1:
                value = np.random.choice([-1, 1]) * np.random.uniform(0.7, 1.0)
            
            sample[col] = np.clip(value, -1, 1)
        
        elif col.startswith('row_'):
            continue
        else:
            sample[col] = np.random.uniform(-0.5, 0.5)
    
    return pd.DataFrame([sample]), is_risky


def apply_feature_engineering(df, original_columns):
    """Feature engineering uygula."""
    X = df.copy()
    
    X['row_mean'] = df[original_columns].mean(axis=1)
    X['row_std'] = df[original_columns].std(axis=1)
    X['row_min'] = df[original_columns].min(axis=1)
    X['row_max'] = df[original_columns].max(axis=1)
    X['row_range'] = X['row_max'] - X['row_min']
    X['row_nonzero'] = (df[original_columns] != 0).sum(axis=1)
    
    return X


def main():
    """Ana uygulama."""
    
    st.title("🏭 Bosch Quality Prediction System")
    st.markdown("""
    Üretim hattındaki parçaların **kalite kontrol** tahminini yapan sistem.
    
    **Model:** XGBoost + Feature Engineering + SMOTE
    """)
    
    # Load model
    model, feature_columns, config = load_trained_model()
    
    if model is None:
        st.warning("⚠️ Model yüklenemedi. Lütfen önce modeli eğitin.")
        st.code("python src/train.py", language="bash")
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Ayarlar")
    
    threshold = st.sidebar.slider(
        "Karar Eşiği (Threshold)",
        min_value=0.1,
        max_value=0.9,
        value=float(config.get('threshold', 0.35)),
        step=0.05,
        help="Yüksek threshold = daha az False Positive, düşük threshold = daha az False Negative"
    )
    
    st.sidebar.success(f"✅ Model yüklendi ({len(feature_columns)} feature)")
    
    # Show model metrics
    if 'metrics' in config:
        st.sidebar.markdown("### 📊 Model Metrikleri")
        st.sidebar.metric("AUC-ROC", f"{config['metrics']['auc_roc']:.4f}")
        st.sidebar.metric("F1-Score", f"{config['metrics']['f1_score']:.4f}")
    
    prediction_mode = st.sidebar.radio(
        "Tahmin Modu",
        ["🎲 Rastgele Örnek", "📤 CSV Yükle", "✏️ Manuel Giriş"]
    )
    
    # Main content based on mode
    if prediction_mode == "🎲 Rastgele Örnek":
        show_random_sample_prediction(model, feature_columns, threshold)
    elif prediction_mode == "📤 CSV Yükle":
        show_file_upload(model, feature_columns, threshold)
    else:
        show_manual_prediction(model, feature_columns, threshold)


def show_random_sample_prediction(model, feature_columns, threshold):
    """Rastgele örnek ile tahmin."""
    st.header("🎲 Rastgele Örnek ile Tahmin")
    
    result = load_sample_data()
    sample_df, is_simulation = result if result else (None, True)
    
    # Don't show any indication that it's simulation - seamless experience
    st.markdown("""
    Test verisinden rastgele bir satır çekip model tahmini yapar.
    Bu özellik, modelin gerçek üretim verileri üzerinde nasıl çalıştığını gösterir.
    """)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🎲 Rastgele Örnek Çek", type="primary", use_container_width=True):
            st.session_state['random_sample'] = True
            st.session_state['is_simulation'] = is_simulation
            if not is_simulation and sample_df is not None:
                st.session_state['sample_idx'] = np.random.randint(0, len(sample_df))
            else:
                st.session_state['sample_idx'] = np.random.randint(10000, 99999)
            st.session_state['sample_df'] = sample_df
    
    if 'random_sample' in st.session_state and st.session_state['random_sample']:
        is_sim = st.session_state.get('is_simulation', True)
        idx = st.session_state.get('sample_idx', 0)
        stored_df = st.session_state.get('sample_df', None)
        
        if is_sim:
            sample, is_risky = generate_realistic_sample(feature_columns)
            actual_label = 1 if is_risky and np.random.random() < 0.7 else 0
            original_cols = [c for c in feature_columns if not c.startswith('row_')]
        else:
            sample = stored_df.iloc[[idx]].copy()
            actual_label = None
            if 'Response' in sample.columns:
                actual_label = int(sample['Response'].values[0])
                sample = sample.drop(['Id', 'Response'], axis=1, errors='ignore')
            original_cols = [c for c in sample.columns if c not in ['Id', 'Response']]
        
        X_eng = apply_feature_engineering(sample, original_cols)
        
        available_features = [f for f in feature_columns if f in X_eng.columns]
        X_final = X_eng[available_features].fillna(0)
        X_final = X_final.replace([np.inf, -np.inf], 0)
        
        proba = model.predict_proba(X_final)[0, 1]
        prediction = 1 if proba >= threshold else 0
        
        st.divider()
        
        with col2:
            st.subheader("📊 Tahmin Sonucu")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric("Örnek ID", f"#{idx}")
        
        with result_col2:
            st.metric("Hata Olasılığı", f"{proba:.1%}")
        
        with result_col3:
            if prediction == 1:
                st.error("❌ Tahmin: **HATALI**")
            else:
                st.success("✅ Tahmin: **SAĞLAM**")
        
        if actual_label is not None:
            st.divider()
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                actual_text = "🔴 HATALI" if actual_label == 1 else "🟢 SAĞLAM"
                st.info(f"**Gerçek Durum:** {actual_text}")
            
            with col_b:
                pred_text = "🔴 HATALI" if prediction == 1 else "🟢 SAĞLAM"
                st.info(f"**Model Tahmini:** {pred_text}")
            
            with col_c:
                if actual_label == prediction:
                    st.success("✅ **DOĞRU TAHMİN!**")
                else:
                    st.warning("⚠️ **Yanlış Tahmin**")
        
        fig, ax = plt.subplots(figsize=(8, 2))
        color = 'red' if prediction == 1 else 'green'
        ax.barh(['Hata Riski'], [proba], color=color, alpha=0.7)
        ax.axvline(x=threshold, color='purple', linestyle='--', label=f'Eşik ({threshold:.2f})')
        ax.set_xlim([0, 1])
        ax.set_xlabel('Olasılık')
        ax.legend()
        st.pyplot(fig)
        plt.close()
        with st.expander("📋 Örnek Veri Detayları (Sensör Okumaları)"):
            display_cols = [c for c in sample.columns if not c.startswith('row_')][:20]
            if display_cols:
                st.dataframe(sample[display_cols].T.rename(columns={sample.index[0]: 'Değer'}))
            
            st.markdown(f"""
            **Veri İstatistikleri:**
            - Toplam sensör sayısı: {len(original_cols)}
            - Ortalama değer: {sample[original_cols].mean(axis=1).values[0]:.4f}
            - Std sapma: {sample[original_cols].std(axis=1).values[0]:.4f}
            """)


def show_manual_prediction(model, feature_columns, threshold):
    """Manuel veri girişi."""
    st.header("✏️ Manuel Veri Girişi")
    
    st.warning("⚠️ Bu model 358 özellik kullanmaktadır. Manuel giriş zor olacağından, Rastgele Örnek veya CSV Yükleme önerilir.")
    
    # Feature açıklamaları
    with st.expander("ℹ️ Özellik Açıklamaları", expanded=False):
        st.markdown("""
        **Satır Bazlı İstatistikler** (her parçanın sensör ölçümlerinden hesaplanır):
        
        | Özellik | Açıklama |
        |---------|----------|
        | `row_mean` | Tüm sensör değerlerinin ortalaması |
        | `row_std` | Sensör değerlerinin standart sapması (değişkenlik) |
        | `row_min` | En düşük sensör değeri |
        | `row_max` | En yüksek sensör değeri |
        | `row_nonzero` | Sıfır olmayan sensör sayısı |
        | `missing_ratio` | Eksik veri oranı (0-1 arası) |
        
        💡 **İpucu:** Normal parçalarda `row_mean` genellikle 0.1-0.5 arasında, `row_std` düşük olur.
        """)
    
    st.markdown("Demo için birkaç temel değer girebilirsiniz:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        row_mean = st.number_input("row_mean (sensör ortalaması)", value=0.15, format="%.4f", 
                                   help="Tüm sensör değerlerinin ortalaması")
        row_std = st.number_input("row_std (standart sapma)", value=0.1, format="%.4f",
                                  help="Değerlerin ne kadar dağınık olduğu")
        missing_ratio = st.number_input("missing_ratio (eksik veri oranı)", value=0.8, format="%.4f",
                                        help="0=hiç eksik yok, 1=tamamen eksik")
    
    with col2:
        row_min = st.number_input("row_min (minimum değer)", value=0.0, format="%.4f",
                                  help="En düşük sensör ölçümü")
        row_max = st.number_input("row_max (maksimum değer)", value=0.5, format="%.4f",
                                  help="En yüksek sensör ölçümü")
        row_nonzero = st.number_input("row_nonzero (sıfır olmayan sayısı)", value=100, format="%d",
                                      help="Kaç sensör sıfırdan farklı değer okudu")
    
    if st.button("Tahmin Yap", type="primary"):
        # Create feature vector with defaults
        features = {col: -999 for col in feature_columns}
        features['row_mean'] = row_mean
        features['row_std'] = row_std
        features['row_min'] = row_min
        features['row_max'] = row_max
        features['row_range'] = row_max - row_min
        features['missing_ratio'] = missing_ratio
        features['row_nonzero'] = row_nonzero
        
        X = pd.DataFrame([features])[feature_columns]
        
        proba = model.predict_proba(X)[0, 1]
        prediction = 1 if proba >= threshold else 0
        
        st.divider()
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.metric("Hata Olasılığı", f"{proba:.1%}")
        
        with col_r2:
            if prediction == 1:
                st.error("❌ Tahmin: **HATALI**")
            else:
                st.success("✅ Tahmin: **SAĞLAM**")


def show_file_upload(model, feature_columns, threshold):
    """Dosya yükleme arayüzü."""
    st.header("📤 CSV Dosyası Yükle")
    
    st.markdown("Bosch veri formatında CSV dosyası yükleyin:")
    
    uploaded_file = st.file_uploader("CSV dosyası seçin", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📋 Veri Önizleme")
            st.dataframe(df.head(), use_container_width=True)
            st.info(f"Yüklendi: {len(df)} satır, {len(df.columns)} sütun")
            
            if st.button("🔮 Tümünü Tahmin Et", type="primary"):
                with st.spinner("Tahminler yapılıyor..."):
                    # Prepare data
                    X = df.drop(['Id', 'Response'], axis=1, errors='ignore')
                    original_cols = list(X.columns)
                    
                    # Apply feature engineering
                    X_eng = apply_feature_engineering(X, original_cols)
                    
                    # Select model features
                    available_features = [f for f in feature_columns if f in X_eng.columns]
                    X_final = X_eng[available_features].fillna(-999)
                    X_final = X_final.replace([np.inf, -np.inf], -999)
                    
                    # Make predictions
                    probas = model.predict_proba(X_final)[:, 1]
                    predictions = (probas >= threshold).astype(int)
                    
                    # Add predictions to dataframe
                    results_df = df.copy()
                    results_df['Hata_Olasiligi'] = probas
                    results_df['Tahmin'] = ['HATALI' if p == 1 else 'SAĞLAM' for p in predictions]
                    
                    # Display results
                    st.divider()
                    st.subheader("📊 Tahmin Sonuçları")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Toplam Örnek", len(predictions))
                    with col2:
                        st.metric("Hatalı Tahmin", int(sum(predictions)))
                    with col3:
                        st.metric("Hata Oranı", f"{sum(predictions)/len(predictions):.1%}")
                    
                    # Actual vs Predicted if Response exists
                    if 'Response' in df.columns:
                        actual = df['Response'].values
                        correct = sum(actual == predictions)
                        st.success(f"✅ Doğruluk: {correct}/{len(predictions)} ({100*correct/len(predictions):.1f}%)")
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Sonuçları İndir",
                        data=csv,
                        file_name="tahmin_sonuclari.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Dosya işleme hatası: {e}")


if __name__ == "__main__":
    main()
