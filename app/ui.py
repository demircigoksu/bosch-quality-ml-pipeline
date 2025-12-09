"""
Streamlit UI for Bosch Quality Prediction.

This module provides an interactive web interface for making
manufacturing failure predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# Add src directory to path
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
    """Load the trained model and config (cached)."""
    try:
        model = joblib.load(MODELS_DIR / "bosch_quality_classifier.pkl")
        feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
        
        # Load config if exists
        config_path = MODELS_DIR / "model_config.pkl"
        if config_path.exists():
            config = joblib.load(config_path)
        else:
            config = {'best_threshold': 0.5}
        
        return model, feature_columns, config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


@st.cache_data
def load_sample_data():
    """Load sample data for random prediction."""
    try:
        data_path = DATA_DIR / "train_numeric.csv"
        if data_path.exists():
            # Load only first 10000 rows for sampling
            df = pd.read_csv(data_path, nrows=10000)
            return df
        return None
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None


def apply_feature_engineering(df, original_columns):
    """Apply feature engineering to input data."""
    X = df.copy()
    
    # Row statistics
    X['row_mean'] = df[original_columns].mean(axis=1)
    X['row_std'] = df[original_columns].std(axis=1)
    X['row_min'] = df[original_columns].min(axis=1)
    X['row_max'] = df[original_columns].max(axis=1)
    X['row_range'] = X['row_max'] - X['row_min']
    
    # Missing patterns
    X['missing_count'] = df[original_columns].isnull().sum(axis=1)
    X['missing_ratio'] = X['missing_count'] / len(original_columns)
    X['non_zero_count'] = (df[original_columns] != 0).sum(axis=1)
    
    # Station-based features
    stations = {}
    for col in original_columns:
        parts = col.split('_')
        if len(parts) >= 2:
            station = parts[0]
            if station not in stations:
                stations[station] = []
            stations[station].append(col)
    
    for station, cols in stations.items():
        if len(cols) > 1:
            station_data = df[cols]
            X[f'{station}_mean'] = station_data.mean(axis=1)
            X[f'{station}_std'] = station_data.std(axis=1)
            X[f'{station}_missing'] = station_data.isnull().sum(axis=1)
            X[f'{station}_nonzero'] = (station_data != 0).sum(axis=1)
    
    return X


def main():
    """Main Streamlit application."""
    
    # Title and description
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
        value=float(config.get('best_threshold', 0.55)),
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
    """Show random sample prediction - PROJECT REQUIREMENT."""
    st.header("🎲 Rastgele Örnek ile Tahmin")
    
    st.markdown("""
    Test verisinden rastgele bir satır çekip model tahmini yapar.
    Bu özellik, modelin gerçek veriler üzerinde nasıl çalıştığını gösterir.
    """)
    
    # Load sample data
    sample_df = load_sample_data()
    
    if sample_df is None:
        st.warning("⚠️ Örnek veri bulunamadı. Lütfen `data/train_numeric.csv` dosyasını ekleyin.")
        return
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🎲 Rastgele Örnek Çek", type="primary", use_container_width=True):
            st.session_state['random_sample'] = True
            st.session_state['sample_idx'] = np.random.randint(0, len(sample_df))
    
    if 'random_sample' in st.session_state and st.session_state['random_sample']:
        idx = st.session_state['sample_idx']
        sample = sample_df.iloc[[idx]].copy()
        
        # Get actual label if exists
        actual_label = None
        if 'Response' in sample.columns:
            actual_label = int(sample['Response'].values[0])
            sample = sample.drop(['Id', 'Response'], axis=1, errors='ignore')
        
        # Get original columns (excluding Id and Response)
        original_cols = [c for c in sample.columns if c not in ['Id', 'Response']]
        
        # Apply feature engineering
        X_eng = apply_feature_engineering(sample, original_cols)
        
        # Select only model features
        available_features = [f for f in feature_columns if f in X_eng.columns]
        X_final = X_eng[available_features].fillna(-999)
        X_final = X_final.replace([np.inf, -np.inf], -999)
        
        # Make prediction
        proba = model.predict_proba(X_final)[0, 1]
        prediction = 1 if proba >= threshold else 0
        
        # Display results
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
        
        # Show actual vs predicted if available
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
        
        # Visualization
        fig, ax = plt.subplots(figsize=(8, 2))
        color = 'red' if prediction == 1 else 'green'
        ax.barh(['Hata Riski'], [proba], color=color, alpha=0.7)
        ax.axvline(x=threshold, color='purple', linestyle='--', label=f'Eşik ({threshold:.2f})')
        ax.set_xlim([0, 1])
        ax.set_xlabel('Olasılık')
        ax.legend()
        st.pyplot(fig)
        
        # Show some feature values
        with st.expander("📋 Örnek Veri Detayları"):
            # Show non-null features
            non_null = sample.iloc[0].dropna()
            if len(non_null) > 0:
                st.dataframe(non_null.head(20).to_frame().T)


def show_manual_prediction(model, feature_columns, threshold):
    """Show manual prediction interface."""
    st.header("✏️ Manuel Veri Girişi")
    
    st.warning("⚠️ Bu model 358 özellik kullanmaktadır. Manuel giriş zor olacağından, Rastgele Örnek veya CSV Yükleme önerilir.")
    
    # Simple demo with a few features
    st.markdown("Demo için birkaç temel değer girebilirsiniz:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        row_mean = st.number_input("row_mean", value=0.0, format="%.4f")
        row_std = st.number_input("row_std", value=0.0, format="%.4f")
        missing_ratio = st.number_input("missing_ratio", value=0.8, format="%.4f")
    
    with col2:
        row_min = st.number_input("row_min", value=0.0, format="%.4f")
        row_max = st.number_input("row_max", value=0.0, format="%.4f")
        non_zero_count = st.number_input("non_zero_count", value=100, format="%d")
    
    if st.button("Tahmin Yap", type="primary"):
        # Create feature vector with defaults
        features = {col: -999 for col in feature_columns}
        features['row_mean'] = row_mean
        features['row_std'] = row_std
        features['row_min'] = row_min
        features['row_max'] = row_max
        features['row_range'] = row_max - row_min
        features['missing_ratio'] = missing_ratio
        features['non_zero_count'] = non_zero_count
        
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
    """Show file upload interface."""
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
