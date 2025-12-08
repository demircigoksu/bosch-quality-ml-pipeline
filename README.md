# Bosch Quality ML Pipeline

End-to-end machine learning project for detecting manufacturing failures in the Bosch production dataset. This project implements a complete ML pipeline from exploratory data analysis to deployment with REST API and interactive UI.

## 🎯 Project Overview

This project tackles the **Bosch Production Line Performance** problem - a binary classification task to predict manufacturing failures based on production line measurements. Early detection of failures can help:
- Reduce waste and improve product quality
- Optimize production processes
- Minimize costs associated with defects
- Improve overall manufacturing efficiency

## 📊 Dataset

The Bosch dataset contains anonymized production line measurements with:
- **Target variable**: Binary indicator of product failure (0 = no failure, 1 = failure)
- **Features**: Thousands of anonymized sensor measurements from the production line
- **Challenge**: Highly imbalanced dataset with sparse features and missing values

## 🏗️ Project Structure

```
bosch-quality-ml-pipeline/
├── data/                       # Data directory (train/test CSV files)
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_baseline.ipynb      # Baseline model development
│   └── 03_pipeline.ipynb      # Full ML pipeline development
├── src/                        # Source code modules
│   ├── config.py              # Configuration parameters
│   ├── train.py               # Model training pipeline
│   └── inference.py           # Inference and prediction logic
├── app/                        # Application deployment
│   ├── main.py                # FastAPI REST API
│   └── ui.py                  # Streamlit interactive UI
├── models/                     # Saved model artifacts
├── docs/                       # Documentation
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/demircigoksu/bosch-quality-ml-pipeline.git
cd bosch-quality-ml-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Bosch dataset and place it in the `data/` directory:
   - `train.csv` - Training data
   - `test.csv` - Test data

## 📓 Usage

### 1. Exploratory Data Analysis

Start with the notebooks to understand the data:
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Model Training

Train the model using the training pipeline:
```bash
python src/train.py
```

This will:
- Load and preprocess the training data
- Train an XGBoost classifier
- Evaluate performance on test set
- Save the trained model to `models/`

### 3. Making Predictions

Use the inference module for predictions:
```bash
python src/inference.py
```

### 4. REST API Deployment

Launch the FastAPI server:
```bash
python app/main.py
```

The API will be available at `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

**API Endpoints:**
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions

### 5. Interactive UI

Launch the Streamlit application:
```bash
streamlit run app/ui.py
```

The UI provides:
- Single sample prediction
- Batch prediction
- CSV file upload for bulk predictions
- Interactive visualizations

## 🔧 Configuration

All configuration parameters are in `src/config.py`:
- Model hyperparameters
- Data paths
- Feature engineering settings
- API configuration

## 📦 Dependencies

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms and utilities
- **xgboost** - Gradient boosting framework
- **fastapi** - Modern web framework for APIs
- **uvicorn** - ASGI server for FastAPI
- **streamlit** - Interactive web applications
- **matplotlib** - Data visualization

## 🎯 Model Performance

The model uses XGBoost for classification with:
- Matthews Correlation Coefficient (MCC) for evaluation
- ROC-AUC score for imbalanced data
- Handles missing values and sparse features
- Feature selection based on missing value threshold

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👥 Authors

- Göksu Demirci

## 🙏 Acknowledgments

- Bosch for providing the dataset
- Kaggle competition community
