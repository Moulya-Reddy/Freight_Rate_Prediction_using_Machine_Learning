# Freight Rate Prediction using Machine Learning

> Machine Learning Engineer Assessment Submission

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-green)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

This project presents an end-to-end machine learning pipeline for predicting freight transportation rates from historical shipment records.

The solution was developed as part of a Machine Learning Engineer assessment and emphasizes:

- Robust data preprocessing
- Leakage-safe feature engineering
- Time-aware model validation
- Comparative evaluation of multiple machine learning models
- Hyperparameter optimization
- Reproducible prediction generation

Rather than optimizing solely for leaderboard performance, the project focuses on building a production-style pipeline that reflects how freight pricing models would be developed and evaluated in real-world logistics systems.

---

# Problem Statement

Given historical shipment data, the objective is to predict the freight rate (`posted_rate`) for unseen shipment requests.

The assessment required generating:

- Predictions for **12,000 validation shipments**
- A **31-day freight rate forecast** for a fixed shipment lane during December 2025
- A reproducible machine learning pipeline
- Documentation describing the methodology

---

# Dataset

The development dataset contains approximately **48,000 historical shipment records** collected between January and October 2025.

Each shipment contains information including:

- Pickup location
- Delivery location
- Distance
- Shipment weight
- Equipment type
- Market index
- Quote signal
- Shipment date
- Freight rate (target)

Validation data contains unseen shipments for which freight rates must be predicted.

---

# Project Structure

```
freight-rate-solution/
│
├── data/
│   ├── train_test.csv
│   ├── validation.csv
│   ├── december_chart_inputs.csv
│
├── models/
│   └── model.joblib
│
├── scorer_results/
│   └── candidate_december.png
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── features.py
│   ├── data_prep.py
│   ├── models.py
│   └── cv.py
│
├── validation_predictions.csv
├── training_report.json
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

---

# Data Exploration

Before model development, the dataset was analyzed for:

- Missing values
- Duplicate records
- Invalid observations
- Target distribution
- Feature correlations
- Temporal coverage

### Data Quality Issues

| Issue | Resolution |
|--------|------------|
| Negative shipment weights | Converted using absolute value |
| Missing shipment weights | Filled using equipment-specific training median |
| Missing market index values | Filled using training-set median |
| Unseen cities in validation | Handled through CatBoost together with historical lane statistics and coordinate features |

---

# Feature Engineering

Feature engineering was designed to maximize predictive performance while preventing information leakage.

## Calendar Features

- Month
- Day of Week
- Week Number
- Day of Year

---

## Shipment Features

- Distance
- Weight
- Equipment Type
- Market Index
- Quote Signal

---

## Geographic Features

- Pickup Latitude
- Pickup Longitude
- Delivery Latitude
- Delivery Longitude

---

## Historical Features

Leakage-safe historical statistics were generated using expanding-window calculations.

These include:

- Lane historical rate per mile
- Lane shipment count
- Equipment historical rate per mile

Historical statistics were computed using only information available prior to each shipment.

For prediction datasets, historical statistics were generated using the complete training history, ensuring no future information was introduced.

---

# Validation Strategy

A random train-test split would leak future pricing information because freight prices evolve over time.

Instead, a **time-based validation strategy** was adopted.

Training period:

```
January 2025
↓

September 5 2025
```

Validation period:

```
September 6 2025
↓

October 31 2025
```

Hyperparameter tuning was performed using expanding-window cross-validation.

This approach better reflects real-world deployment where future shipments are predicted using historical data.

---

# Models Evaluated

Five regression models were compared.

| Model | Purpose |
|---------|------------|
| Linear Regression | Baseline |
| HistGradientBoosting | Gradient boosting baseline |
| LightGBM | Gradient boosting |
| XGBoost | Gradient boosting |
| CatBoost | Final production model |

---

# Model Performance

| Model | MAE | MAPE |
|---------|------:|------:|
| **CatBoost** | **110.30** | **4.86%** |
| HistGradientBoosting | 122.66 | 5.32% |
| LightGBM | 154.22 | 6.85% |
| XGBoost | 156.58 | 6.99% |
| Linear Regression | 401.92 | 17.98% |

CatBoost consistently achieved the lowest validation error and was therefore selected as the final production model.

---

# Hyperparameter Optimization

Optuna was used to tune the strongest gradient boosting models.

Optimized parameters included:

- Learning Rate
- Tree Depth
- Number of Trees
- L2 Regularization
- Number of Leaves
- Column Sampling
- Row Sampling

Although Optuna improved cross-validation performance, the untuned CatBoost model achieved slightly better performance on the final holdout dataset and was therefore selected for deployment.

---

# Prediction Pipeline

The prediction workflow is:

```
Raw Data
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Model Training
      │
      ▼
Model Selection
      │
      ▼
Prediction
      │
      ▼
Validation
```

---

# Results

The solution successfully generated:

- validation_predictions.csv
- December prediction file
- candidate_december.png
- Saved production model
- Complete training report

All generated files successfully pass the provided validation script.

---

# Installation

Clone the repository.

```bash
git clone https://github.com/Moulya-Reddy/Freight_Rate_Prediction_using_Machine_Learning.git

cd Freight_Rate_Prediction_using_Machine_Learning
```

Install development dependencies.

```bash
python -m pip install -r requirements-dev.txt
```

---

# Training

```bash
python -m src.train
```

This will:

- preprocess data
- compare all models
- perform hyperparameter tuning
- save the best model
- generate training_report.json

---

# Prediction

```bash
python -m src.predict
```

Generates

- validation_predictions.csv
- December predictions

---

# Validation

Install the official assessment requirements.

```bash
python -m pip install -r requirements.txt
```

Run

```bash
python score.py \
--predictions validation_predictions.csv \
--december-predictions data/december_chart_inputs.csv
```

Expected output:

```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart:
candidate_december.png
```

---

# Future Improvements

Potential enhancements include:

- Real-time market signal forecasting
- Recency-weighted lane history
- Lane-equipment interaction features
- Larger Optuna search budget
- Ensemble learning
- Online model retraining

---

# Author

**Moulya Reddy**

Machine Learning Engineer Assessment

August 2026

---

Thank you for reviewing this submission.
