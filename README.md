# Air Quality Index (AQI) Prediction using Machine Learning

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![ML Framework](https://img.shields.io/badge/Framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning project designed to predict air quality levels based on atmospheric variables. This repository provides a complete pipeline from data preprocessing and exploratory data analysis (EDA) to model training and performance evaluation.


## 📊 Overview

This project focuses on predicting air quality levels by processing a dataset of 23,500+ records. It transitions from raw data ingestion and preprocessing to advanced time-series forecasting and interactive data visualization.

### Key Features
*   **Data Preprocessing:** Automated date conversion and handling of sensor readings.
*   **Feature Engineering:** Implementation of `MinMaxScaler` for normalized model training.
*   **Time-Series Modeling:** Daily seasonality forecasting using the Prophet algorithm.
*   **Performance Metrics:** Evaluation using MAE, RMSE, and R-squared scores.
*   **Interactive Dashboards:** Dynamic visualizations using Plotly for "Actual vs. Predicted" analysis.

## 🛠 Tech Stack

*   **Language:** Python
*   **Data Analysis:** Pandas, NumPy
*   **Machine Learning:** Scikit-Learn, Prophet
*   **Visualization:** Matplotlib, Seaborn, Plotly

## 🚀 Getting Started

### Prerequisites
Ensure you have Python installed, then install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn prophet plotly
```

### Dataset Structure
The model expects a CSV file named `air_pollution_data.csv` with the following columns:
*   `city`, `date`, `aqi` (Target)
*   `co`, `no`, `no2`, `o3`, `so2`, `pm2_5`, `pm10`, `nh3`

## 📈 Implementation Workflow

### 1. Data Cleaning & Scaling
The project converts raw date strings into Python datetime objects and scales the target `aqi` variable to a range between 0 and 1 to improve model convergence.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
prophet_df['y'] = scaler.fit_transform(prophet_df[['y']])
```

### 2. Model Training
The Prophet model is configured to recognize daily seasonality patterns in the air quality data.

```python
from prophet import Prophet

model = Prophet(daily_seasonality=True)
model.fit(train_df)
```

### 3. Forecasting
Predictions are generated for a 30-day future horizon, providing upper and lower uncertainty bounds.

## 🧪 Evaluation Results

The model was evaluated on a 20% hold-out test set with the following results:

| Metric | Value |
| :--- | :--- |
| **Mean Absolute Error (MAE)** | 0.3205 |
| **Root Mean Square Error (RMSE)** | 0.3978 |
| **R-squared (R2) Score** | -0.2182 |

> **Note:** The R2 score suggests that while the model captures general trends, the high volatility in daily air quality data presents significant forecasting challenges.

## 🖼 Visualizations

The project includes two types of visualizations:
1.  **Static Comparison:** Matplotlib plots showing the overlap of actual vs. predicted AQI.
2.  **Interactive Dashboard:** A Plotly-based interface allowing users to hover over specific dates to compare forecasted values against ground truth.
