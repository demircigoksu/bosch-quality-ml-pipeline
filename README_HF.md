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

# 🏭 Bosch Production Line Quality Prediction

Manufacturing failure prediction system using XGBoost.

## Model Performance
- **AUC-ROC:** 0.635
- **Recall:** 51.4% (catches half of defective parts)
- **Dataset:** 450K clean samples from Bosch Production Line Performance dataset

## Features
- Real-time quality prediction
- Interactive threshold adjustment
- Random sample testing from production data
