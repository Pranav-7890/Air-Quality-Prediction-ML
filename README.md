# 🌫️ Air Quality Forecasting System – Ahmedabad

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Prophet](https://img.shields.io/badge/Model-Prophet-orange.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)

An end-to-end time-series forecasting project designed to predict the **Air Quality Index (AQI)** for Ahmedabad, India. This system leverages historical data to capture seasonal patterns and provide actionable 7-day future insights.

---

## 📌 Project Overview
Urban air quality is highly volatile due to industrial activity and seasonal changes. This project focuses on predicting the AQI for Ahmedabad using time-series forecasting techniques. By analyzing historical air pollution data, the system captures seasonal patterns and generates a 7-day future AQI forecast, helping assess upcoming air quality trends.

## 🚀 Key Features
* **Time-Series Analysis:** Implemented using the **Prophet** library to model trend, seasonality, and irregular variations in AQI data.
* **Interactive Dashboard:** Built with **Plotly** for dynamic exploration of actual vs. predicted AQI values.
* **Clean Data Pipeline:** * Mean imputation for missing values.
    * Robust date-time conversion and alignment.
* **Feature Engineering:** * MinMaxScaler normalization.
    * Chronological train-test split (80/20) to maintain temporal consistency.

---

## 🛠️ Tech Stack
* **Programming Language:** Python
* **Libraries & Tools:**
    * **Data Science:** Pandas, NumPy, Scikit-learn
    * **Forecasting:** Prophet (Meta)
    * **Visualization:** Matplotlib, Seaborn, Plotly
* **Development Environment:** Google Colab

---

## 📂 Project Structure
```text
Air-Quality-Forecasting-System/
│
├── data/
│   ├── raw_data.csv              # Original AQI dataset
│   └── processed_data.csv        # Cleaned and preprocessed data
│
├── notebooks/
│   └── aqi_forecasting.ipynb     # Main Colab notebook
│
├── visuals/
│   ├── static_plot.png           # Matplotlib AQI comparison
│   └── interactive_dashboard.html# Plotly interactive dashboard
│
├── README.md                     # Project documentation
└── requirements.txt              # Required Python libraries
