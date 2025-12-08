"""
Streamlit UI for Bosch Quality Prediction.

This module provides an interactive web interface for making
manufacturing failure predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from inference import load_model, predict_failure_probability
from config import MODEL_PATH


# Page configuration
st.set_page_config(
    page_title="Bosch Quality Prediction",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_trained_model():
    """Load the trained model (cached)."""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please train the model first using `python src/train.py`")
        return None


def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🏭 Bosch Quality Prediction System")
    st.markdown("""
    This application predicts manufacturing failures based on production line data.
    Upload your data or enter features manually to get predictions.
    """)
    
    # Sidebar
    st.sidebar.header("Configuration")
    prediction_mode = st.sidebar.radio(
        "Select Prediction Mode",
        ["Single Prediction", "Batch Prediction", "File Upload"]
    )
    
    # Load model
    model = load_trained_model()
    
    if model is None:
        st.warning("⚠️ Model not loaded. Please train the model first.")
        st.code("python src/train.py", language="bash")
        return
    
    st.sidebar.success("✅ Model loaded successfully")
    
    # Main content based on mode
    if prediction_mode == "Single Prediction":
        show_single_prediction(model)
    elif prediction_mode == "Batch Prediction":
        show_batch_prediction(model)
    else:
        show_file_upload(model)


def show_single_prediction(model):
    """Show single prediction interface."""
    st.header("Single Sample Prediction")
    
    st.markdown("Enter feature values for prediction:")
    
    # Create input fields
    col1, col2, col3 = st.columns(3)
    
    features = {}
    with col1:
        features['feature_1'] = st.number_input("Feature 1", value=0.0, format="%.4f")
        features['feature_2'] = st.number_input("Feature 2", value=0.0, format="%.4f")
        features['feature_3'] = st.number_input("Feature 3", value=0.0, format="%.4f")
    
    with col2:
        features['feature_4'] = st.number_input("Feature 4", value=0.0, format="%.4f")
        features['feature_5'] = st.number_input("Feature 5", value=0.0, format="%.4f")
        features['feature_6'] = st.number_input("Feature 6", value=0.0, format="%.4f")
    
    with col3:
        features['feature_7'] = st.number_input("Feature 7", value=0.0, format="%.4f")
        features['feature_8'] = st.number_input("Feature 8", value=0.0, format="%.4f")
        features['feature_9'] = st.number_input("Feature 9", value=0.0, format="%.4f")
    
    # Predict button
    if st.button("Predict", type="primary"):
        with st.spinner("Making prediction..."):
            try:
                df = pd.DataFrame([features])
                failure_prob = predict_failure_probability(model, df)[0]
                prediction = 1 if failure_prob > 0.5 else 0
                
                # Display results
                st.divider()
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Failure Probability",
                        value=f"{failure_prob:.2%}"
                    )
                
                with col2:
                    status = "🔴 FAILURE" if prediction == 1 else "🟢 NO FAILURE"
                    st.metric(
                        label="Prediction",
                        value=status
                    )
                
                # Visualization
                fig, ax = plt.subplots(figsize=(6, 2))
                ax.barh(['Failure Risk'], [failure_prob], color='red' if prediction == 1 else 'green')
                ax.set_xlim([0, 1])
                ax.set_xlabel('Probability')
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")


def show_batch_prediction(model):
    """Show batch prediction interface."""
    st.header("Batch Prediction")
    
    st.markdown("Enter multiple samples (one per line, comma-separated):")
    
    sample_data = """0.5, 1.2, -0.3, 0.8, 1.1, -0.5, 0.2, 0.9, 1.5
-0.3, 0.8, 1.2, -0.7, 0.4, 1.0, -0.2, 0.6, 0.3"""
    
    text_input = st.text_area(
        "Paste your data here:",
        value=sample_data,
        height=150
    )
    
    if st.button("Predict Batch", type="primary"):
        try:
            # Parse input
            lines = text_input.strip().split('\n')
            data_list = []
            for line in lines:
                values = [float(x.strip()) for x in line.split(',')]
                data_list.append(values)
            
            # Create DataFrame
            df = pd.DataFrame(data_list)
            
            # Make predictions
            failure_probs = predict_failure_probability(model, df)
            predictions = [1 if prob > 0.5 else 0 for prob in failure_probs]
            
            # Display results
            st.divider()
            st.subheader("Batch Results")
            
            results_df = pd.DataFrame({
                'Sample': range(1, len(predictions) + 1),
                'Failure Probability': [f"{p:.2%}" for p in failure_probs],
                'Prediction': ['FAILURE' if p == 1 else 'NO FAILURE' for p in predictions]
            })
            
            st.dataframe(results_df, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", len(predictions))
            with col2:
                st.metric("Predicted Failures", sum(predictions))
            with col3:
                st.metric("Failure Rate", f"{sum(predictions)/len(predictions):.1%}")
            
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")


def show_file_upload(model):
    """Show file upload interface."""
    st.header("File Upload Prediction")
    
    st.markdown("Upload a CSV file with your features:")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            st.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
            
            if st.button("Predict All", type="primary"):
                with st.spinner("Making predictions..."):
                    # Make predictions
                    failure_probs = predict_failure_probability(model, df)
                    predictions = [1 if prob > 0.5 else 0 for prob in failure_probs]
                    
                    # Add predictions to dataframe
                    results_df = df.copy()
                    results_df['failure_probability'] = failure_probs
                    results_df['prediction'] = predictions
                    
                    # Display results
                    st.divider()
                    st.subheader("Prediction Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Samples", len(predictions))
                    with col2:
                        st.metric("Predicted Failures", sum(predictions))
                    with col3:
                        st.metric("Failure Rate", f"{sum(predictions)/len(predictions):.1%}")
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")


if __name__ == "__main__":
    main()
